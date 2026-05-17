#!/usr/bin/env python
"""Re-label the pilot rows that previously failed B/C grading using the
May-16 hardened prompt, then re-judge them, and report the delta.

Goal: verify that the hardening
  (a) removes the "goal committed" / "instruction has been read" anthropomorphic
      phrasing that produced 7 of 10 C-fails in the pilot, and
  (b) removes ``image_region`` bullets from ``image_patch`` rows, and
  (c) does not regress B-grounding on the rows that were already passing.

Inputs (on disk):
  - data/labels/libero_goal_pilot/labels.jsonl     (243 pilot label rows)
  - data/eval/libero_pilot_judge.jsonl             (gpt-5.1 multimodal judge results)
  - data/labels/libero_goal_pilot/frames_cache/    (486 cached camera frames)

Outputs:
  - data/eval/sanity_check_hardened/labels.jsonl   (re-labeled rows, hardened prompt)
  - data/eval/sanity_check_hardened/judge.jsonl    (judge results on the re-labeled rows)
  - prints a side-by-side delta table

Usage::

    OPENAI_API_KEY=... PYTHONPATH=src \
      python scripts/labeling/sanity_check_hardened_prompt.py
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

from nla.labeling.grader import GradeInput, grade_many_async
from nla.labeling.openai_client import label_many_async
from nla.labeling.prompts import PositionLabelInput

logger = logging.getLogger(__name__)

LABELS_PATH = Path("data/labels/libero_goal_pilot/labels.jsonl")
JUDGE_PATH = Path("data/eval/libero_pilot_judge.jsonl")
FRAMES_CACHE = Path("data/labels/libero_goal_pilot/frames_cache")
OUT_DIR = Path("data/eval/sanity_check_hardened")
LABELS_OUT = OUT_DIR / "labels.jsonl"
JUDGE_OUT = OUT_DIR / "judge.jsonl"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-passing-controls", type=int, default=4,
                   help="Random passing rows added to the failure set as controls.")
    p.add_argument("--label-concurrency", type=int, default=8)
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--label-model", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-keys", nargs="+", default=["image", "wrist_image"])
    p.add_argument("--reset", action="store_true",
                   help="Delete prior sanity outputs before running.")
    return p


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _select_examples(label_rows: list[dict], judge_rows: list[dict],
                     n_controls: int, seed: int) -> list[dict]:
    by_id = {r["example_id"]: r for r in label_rows if r.get("description")}
    fails = sorted({
        g["example_id"]
        for g in judge_rows
        if (g.get("grounding") or {}).get("verdict") != "specific"
        or (g.get("appropriateness") or {}).get("verdict") != "appropriate"
    })
    fail_ids = [eid for eid in fails if eid in by_id]
    pass_ids = sorted(set(by_id) - set(fail_ids))
    rng = random.Random(seed)
    rng.shuffle(pass_ids)

    picks: list[dict] = []
    for eid in fail_ids:
        picks.append(by_id[eid])

    by_pt: dict[str, list[str]] = collections.defaultdict(list)
    for eid in pass_ids:
        by_pt[by_id[eid]["meta"]["position_type"]].append(eid)
    quota = {"last_text": max(1, n_controls // 2),
             "image_patch": max(1, n_controls // 4),
             "anchor": max(1, n_controls // 4)}
    for pt, k in quota.items():
        for eid in by_pt.get(pt, [])[:k]:
            picks.append(by_id[eid])

    seen: set[str] = set()
    deduped: list[dict] = []
    for r in picks:
        if r["example_id"] in seen:
            continue
        seen.add(r["example_id"])
        deduped.append(r)
    return deduped


def _row_to_input(row: dict, video_keys: list[str]) -> PositionLabelInput | None:
    meta = row["meta"]
    sid = meta.get("source_example_id") or row["example_id"].split("@")[0]
    image_paths: list[str] = []
    for vk in video_keys:
        p = FRAMES_CACHE / f"{sid}__{vk}.jpg"
        if p.exists():
            image_paths.append(str(p))
    if not image_paths:
        logger.warning("no cached frames for %s -> skip", row["example_id"])
        return None

    instruction = meta.get("instruction", "")
    seq_len = int(meta.get("seq_len", 143))
    placeholder = (
        "<image: 128 patches><image: 128 patches> "
        + instruction.strip()
        + " <action>"
    )

    ipm = meta.get("image_patch_meta")
    if ipm is not None and not isinstance(ipm, (list, tuple)):
        ipm = None

    return PositionLabelInput(
        example_id=row["example_id"],
        instruction=instruction,
        decoded_text_context=placeholder,
        position_index=int(meta.get("position_index", 0)),
        position_type=meta.get("position_type", "last_text"),
        sequence_length=seq_len,
        image_paths=image_paths,
        image_patch_meta=tuple(ipm) if ipm is not None else None,
        episode_index=meta.get("episode_index"),
        step_index=meta.get("step_index"),
        extra={"source_example_id": sid},
    )


_FORBIDDEN_RE = re.compile(
    r"(committing to|committed to|"
    r"instruction has been read|goal committed|has been read and committed to)",
    re.IGNORECASE,
)
_IMAGE_REGION_BULLET_RE = re.compile(r"^\s*-\s*image_region:", re.IGNORECASE | re.MULTILINE)


def _scan_phrasings(rows: list[dict]) -> dict[str, int]:
    n_forbidden = 0
    n_image_region_on_patch = 0
    for r in rows:
        desc = r.get("description") or ""
        if _FORBIDDEN_RE.search(desc):
            n_forbidden += 1
        if r.get("meta", {}).get("position_type") == "image_patch":
            if _IMAGE_REGION_BULLET_RE.search(desc):
                n_image_region_on_patch += 1
    return {
        "forbidden_phrasing_rows": n_forbidden,
        "image_region_on_image_patch_rows": n_image_region_on_patch,
    }


async def _amain(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    if args.label_model:
        os.environ["OPENAI_LABELING_MODEL"] = args.label_model
    if args.judge_model:
        os.environ["OPENAI_GRADER_MODEL"] = args.judge_model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        for p in (LABELS_OUT, JUDGE_OUT):
            if p.exists():
                p.unlink()

    label_rows = _load_jsonl(LABELS_PATH)
    judge_rows = _load_jsonl(JUDGE_PATH)
    if not label_rows:
        print(f"ERROR: no rows in {LABELS_PATH}", file=sys.stderr)
        return 2

    selected = _select_examples(label_rows, judge_rows,
                                args.n_passing_controls, args.seed)
    print(f"Selected {len(selected)} pilot rows for sanity check "
          f"({sum(1 for r in selected if r['example_id'] in {g['example_id'] for g in judge_rows if (g.get('grounding') or {}).get('verdict') != 'specific' or (g.get('appropriateness') or {}).get('verdict') != 'appropriate'})}"
          f" prior failures + control passers).")

    inputs: list[PositionLabelInput] = []
    skipped: list[str] = []
    for r in selected:
        inp = _row_to_input(r, args.video_keys)
        if inp is None:
            skipped.append(r["example_id"])
            continue
        inputs.append(inp)
    if skipped:
        print(f"  skipped {len(skipped)} rows (missing cached frames): {skipped}")

    print(f"Re-labeling {len(inputs)} rows with the hardened prompt at "
          f"concurrency={args.label_concurrency} ...")
    n_new = await label_many_async(
        inputs, LABELS_OUT,
        concurrency=args.label_concurrency,
        resume=True,
    )
    print(f"  -> {n_new} new label rows in {LABELS_OUT}")

    new_label_rows = _load_jsonl(LABELS_OUT)
    by_id_new = {r["example_id"]: r for r in new_label_rows if r.get("description")}
    print(f"  total usable re-labeled rows in file: {len(by_id_new)}")

    grade_inputs: list[GradeInput] = []
    for inp in inputs:
        nr = by_id_new.get(inp.example_id)
        if nr is None or nr.get("error") or not nr.get("description"):
            print(f"  WARN: missing re-label for {inp.example_id}; skipping judge")
            continue
        grade_inputs.append(GradeInput(
            example_id=inp.example_id,
            variant_id="hardened",
            description=nr["description"],
            instruction=inp.instruction,
            position_type=inp.position_type,
            image_paths=inp.image_paths,
            seq_len=inp.sequence_length,
            position_index=inp.position_index,
        ))

    print(f"Judging {len(grade_inputs)} re-labeled rows at "
          f"concurrency={args.judge_concurrency} ...")
    n_new_grades = await grade_many_async(
        grade_inputs, JUDGE_OUT,
        concurrency=args.judge_concurrency,
        resume=True,
    )
    print(f"  -> {n_new_grades} new judge rows in {JUDGE_OUT}")

    new_grades = _load_jsonl(JUDGE_OUT)
    new_grades_by_id = {g["example_id"]: g for g in new_grades}

    old_grades_by_id = {g["example_id"]: g for g in judge_rows}
    old_labels_by_id = {r["example_id"]: r for r in label_rows}

    forbidden_old = _scan_phrasings(list(old_labels_by_id.values()))
    forbidden_new = _scan_phrasings(list(by_id_new.values()))

    print()
    print("=" * 78)
    print("Phrasing audit (whole pilot vs. hardened re-label set)")
    print("=" * 78)
    print(f"{'metric':50s}  {'old (pilot)':>12s}  {'new (hardened)':>14s}")
    pilot_n = sum(1 for r in old_labels_by_id.values() if r.get("description"))
    new_n = len(by_id_new)
    print(f"  total label rows audited                          "
          f"  {pilot_n:>12d}  {new_n:>14d}")
    print(f"  rows with anthropomorphic phrasing                "
          f"  {forbidden_old['forbidden_phrasing_rows']:>12d}  "
          f"{forbidden_new['forbidden_phrasing_rows']:>14d}")
    print(f"  image_patch rows with image_region: bullet        "
          f"  {forbidden_old['image_region_on_image_patch_rows']:>12d}  "
          f"{forbidden_new['image_region_on_image_patch_rows']:>14d}")

    print()
    print("=" * 78)
    print("Per-row B/C verdict delta (selected sanity-check rows)")
    print("=" * 78)
    print(f"  {'example_id':54s}  {'pos_type':12s}  {'B old':>5s}  {'B new':>5s}  "
          f"{'C old':>5s}  {'C new':>5s}")
    n_b_old_pass = n_c_old_pass = 0
    n_b_new_pass = n_c_new_pass = 0
    n_b_recovered = n_c_recovered = 0
    n_b_regressed = n_c_regressed = 0
    for inp in inputs:
        eid = inp.example_id
        og = old_grades_by_id.get(eid, {})
        ng = new_grades_by_id.get(eid, {})
        b_old = (og.get("grounding") or {}).get("verdict") == "specific"
        c_old = (og.get("appropriateness") or {}).get("verdict") == "appropriate"
        b_new = (ng.get("grounding") or {}).get("verdict") == "specific"
        c_new = (ng.get("appropriateness") or {}).get("verdict") == "appropriate"
        n_b_old_pass += int(b_old)
        n_c_old_pass += int(c_old)
        n_b_new_pass += int(b_new)
        n_c_new_pass += int(c_new)
        if not b_old and b_new:
            n_b_recovered += 1
        if not c_old and c_new:
            n_c_recovered += 1
        if b_old and not b_new:
            n_b_regressed += 1
        if c_old and not c_new:
            n_c_regressed += 1
        b_old_s = "P" if b_old else "F"
        c_old_s = "P" if c_old else "F"
        b_new_s = "P" if b_new else "F"
        c_new_s = "P" if c_new else "F"
        print(f"  {eid:54s}  {inp.position_type:12s}  "
              f"{b_old_s:>5s}  {b_new_s:>5s}  {c_old_s:>5s}  {c_new_s:>5s}")

    print()
    n = len(inputs)
    print("=" * 78)
    print(f"Aggregate on the {n}-row sanity slice")
    print("=" * 78)
    if n:
        print(f"  B-pass: old {n_b_old_pass}/{n} ({n_b_old_pass/n*100:.0f}%) "
              f"-> new {n_b_new_pass}/{n} ({n_b_new_pass/n*100:.0f}%)  "
              f"[+{n_b_recovered} recovered, -{n_b_regressed} regressed]")
        print(f"  C-pass: old {n_c_old_pass}/{n} ({n_c_old_pass/n*100:.0f}%) "
              f"-> new {n_c_new_pass}/{n} ({n_c_new_pass/n*100:.0f}%)  "
              f"[+{n_c_recovered} recovered, -{n_c_regressed} regressed]")

    failed_ids = [eid for eid in (g["example_id"] for g in judge_rows)
                  if (old_grades_by_id[eid].get("grounding") or {}).get("verdict") != "specific"
                  or (old_grades_by_id[eid].get("appropriateness") or {}).get("verdict") != "appropriate"]
    failed_ids = [eid for eid in failed_ids if eid in new_grades_by_id]
    if failed_ids:
        c_recover_on_fails = sum(
            1 for eid in failed_ids
            if (old_grades_by_id[eid].get("appropriateness") or {}).get("verdict") != "appropriate"
            and (new_grades_by_id[eid].get("appropriateness") or {}).get("verdict") == "appropriate"
        )
        c_fails_old = sum(
            1 for eid in failed_ids
            if (old_grades_by_id[eid].get("appropriateness") or {}).get("verdict") != "appropriate"
        )
        if c_fails_old:
            print(f"  C-pass on prior C-failures: {c_recover_on_fails}/{c_fails_old} recovered "
                  f"({c_recover_on_fails/c_fails_old*100:.0f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())

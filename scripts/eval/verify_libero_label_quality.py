#!/usr/bin/env python
"""Phase 5 quality gate: stratified-sample 500 gold captions across the 4
LIBERO suites and judge them against cached frames.

Greenlight criterion (printed at the end):
    grounding=specific           >= 95%
    appropriateness=appropriate  >= 95%

Sampling:
    - Pool all (suite, source_id, position_index, position_type, description)
      from every labels.jsonl under --labels-root that has a corresponding
      frames_cache entry.
    - Stratified-sample to target --sample-size, with one bucket per
      (position_type, suite) pair. Within a bucket sampling is uniform with a
      fixed seed for reproducibility.

Outputs:
    --out-jsonl judges with variant_id="libero_gold". Idempotent / resumable.

Example::

    OPENAI_API_KEY=... PYTHONPATH=src python scripts/eval/verify_libero_label_quality.py \\
        --labels-root data/labels/libero_4suite_stride2 \\
        --out-jsonl   data/eval/libero_quality_judge.jsonl \\
        --sample-size 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


CAMERA_SUFFIXES = ("__observation_images_image.jpg", "__observation_images_wrist_image.jpg")
# Frame cache filenames for LIBERO use the modality keys with dots replaced by
# underscores; tolerate both schemes by also trying the LeRobot raw key style.
_FALLBACK_CAMERA_SUFFIXES = ("__image.jpg", "__wrist_image.jpg",
                             "__observation.images.image.jpg",
                             "__observation.images.wrist_image.jpg")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels-root",
                   default="data/labels/libero_4suite_stride2",
                   help="Parent dir containing libero_<suite>/labels.jsonl + frames_cache/.")
    p.add_argument("--out-jsonl",
                   default="data/eval/libero_quality_judge.jsonl",
                   help="Output JSONL (grader rows).")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--max-calls", type=int, default=600,
                   help="Hard safety cap (default 600 to leave 100-call buffer).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-model", default=None,
                   help="Override OPENAI_GRADER_MODEL.")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--variant-id", default="libero_gold")
    p.add_argument("--green-threshold", type=float, default=0.95,
                   help="Required pass rate for both B and C verdicts.")
    return p


def _find_frames(source_id: str, cache: Path) -> list[str]:
    paths: list[str] = []
    for suffix in CAMERA_SUFFIXES + _FALLBACK_CAMERA_SUFFIXES:
        cand = cache / f"{source_id}{suffix}"
        if cand.exists():
            paths.append(str(cand))
    return paths


def _iter_suite_rows(suite_dir: Path):
    labels_path = suite_dir / "labels.jsonl"
    frames_cache = suite_dir / "frames_cache"
    if not labels_path.exists():
        return
    if not frames_cache.exists():
        return
    suite = suite_dir.name
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            desc = (obj.get("description") or "").strip()
            if not desc:
                continue
            meta = obj.get("meta") or {}
            src = meta.get("source_example_id")
            pos_idx = meta.get("position_index")
            pos_type = meta.get("position_type")
            instr = meta.get("instruction") or ""
            if src is None or pos_idx is None or pos_type is None:
                continue
            frames = _find_frames(str(src), frames_cache)
            if not frames:
                continue
            yield {
                "suite": suite,
                "source_id": str(src),
                "position_index": int(pos_idx),
                "position_type": str(pos_type),
                "description": desc,
                "instruction": str(instr),
                "image_paths": frames,
            }


def _stratified_sample(rows: list[dict], target_n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["suite"], r["position_type"])].append(r)
    if not by_key:
        return []
    n_buckets = len(by_key)
    per_bucket = max(1, target_n // n_buckets)
    sampled: list[dict] = []
    for key in sorted(by_key):
        bucket = by_key[key]
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_bucket])
    if len(sampled) > target_n:
        rng.shuffle(sampled)
        sampled = sampled[:target_n]
    elif len(sampled) < target_n:
        leftover: list[dict] = []
        for key in sorted(by_key):
            bucket = by_key[key]
            leftover.extend(bucket[per_bucket:])
        rng.shuffle(leftover)
        sampled.extend(leftover[: target_n - len(sampled)])
    print(f"  bucket breakdown:")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in sampled:
        counts[(r["suite"], r["position_type"])] += 1
    for key in sorted(counts):
        print(f"    {key[0]:<8} / {key[1]:<14} : {counts[key]}")
    return sampled


def _summarize(rows: list[dict]) -> dict:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        suite = (r.get("meta") or {}).get("suite") or "?"
        ptype = r.get("position_type") or "_unk"
        by_key[(suite, ptype)].append(r)
    summary: dict = {}
    overall = {"n": 0, "b_pass": 0, "c_pass": 0}
    for (suite, ptype), bucket in sorted(by_key.items()):
        n = len(bucket)
        b_pass = sum(1 for r in bucket
                     if (r.get("grounding") or {}).get("verdict") == "specific")
        c_pass = sum(1 for r in bucket
                     if (r.get("appropriateness") or {}).get("verdict") == "appropriate")
        summary[f"{suite}/{ptype}"] = {
            "n": n,
            "grounding_specific_pct": (b_pass / n) if n else None,
            "appropriateness_appropriate_pct": (c_pass / n) if n else None,
        }
        overall["n"] += n
        overall["b_pass"] += b_pass
        overall["c_pass"] += c_pass
    summary["__overall__"] = overall
    return summary


async def _amain(args) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    if args.judge_model:
        os.environ["OPENAI_GRADER_MODEL"] = args.judge_model

    from nla.labeling.grader import GradeInput, grade_many_async

    labels_root = Path(args.labels_root)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {labels_root}/ for suite directories ...")
    suite_dirs = sorted(p for p in labels_root.glob("libero_*") if p.is_dir())
    if not suite_dirs:
        print(f"ERROR: no libero_* subdirs under {labels_root}.", file=sys.stderr)
        return 2

    all_rows: list[dict] = []
    for sd in suite_dirs:
        n_before = len(all_rows)
        all_rows.extend(_iter_suite_rows(sd))
        print(f"  {sd.name}: {len(all_rows) - n_before} grade-eligible rows")
    if not all_rows:
        print("ERROR: no grade-eligible rows found across suites.", file=sys.stderr)
        return 2
    print(f"  total grade-eligible: {len(all_rows)}")

    sample = _stratified_sample(all_rows, args.sample_size, args.seed)
    if len(sample) > args.max_calls:
        print(f"ERROR: sample size {len(sample)} > --max-calls {args.max_calls}",
              file=sys.stderr)
        return 3
    print(f"  -> sampled {len(sample)} rows")

    inputs: list[GradeInput] = []
    suite_by_eid: dict[str, str] = {}
    ptype_by_eid: dict[str, str] = {}
    for r in sample:
        eid = f"{r['suite']}::{r['source_id']}@p{r['position_index']}_{r['position_type']}"
        suite_by_eid[eid] = r["suite"]
        ptype_by_eid[eid] = r["position_type"]
        inputs.append(GradeInput(
            example_id=eid,
            variant_id=args.variant_id,
            description=r["description"],
            instruction=r["instruction"],
            position_type=r["position_type"],
            image_paths=r["image_paths"],
            seq_len=None,
            position_index=r["position_index"],
        ))

    print(f"Grading {len(inputs)} rows -> {out_path} (concurrency={args.concurrency})")
    n_new = await grade_many_async(
        inputs,
        output_jsonl=out_path,
        concurrency=args.concurrency,
        resume=True,
    )
    print(f"  -> {n_new} new grades written")

    rows: list[dict] = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            eid = obj.get("example_id") or ""
            obj.setdefault("meta", {})
            if "::" in eid:
                obj["meta"]["suite"] = eid.split("::", 1)[0]
            elif eid in suite_by_eid:
                obj["meta"]["suite"] = suite_by_eid[eid]
            if not obj.get("position_type") and eid in ptype_by_eid:
                obj["position_type"] = ptype_by_eid[eid]
            rows.append(obj)

    summary = _summarize(rows)
    overall = summary.pop("__overall__")
    print("\n" + "=" * 78)
    print("Per-bucket verdicts (grounding=specific%   appropriateness=appropriate%)")
    print("=" * 78)
    for k in sorted(summary):
        s = summary[k]
        b = s["grounding_specific_pct"]
        c = s["appropriateness_appropriate_pct"]
        bp = "  N/A" if b is None else f"{b * 100:5.1f}%"
        cp = "  N/A" if c is None else f"{c * 100:5.1f}%"
        print(f"  {k:<32}  n={s['n']:>3}  B={bp}   C={cp}")

    n_total = overall["n"]
    if n_total == 0:
        print("\nNo rows graded; cannot evaluate green threshold.")
        return 2
    b_overall = overall["b_pass"] / n_total
    c_overall = overall["c_pass"] / n_total
    print()
    print(f"OVERALL  n={n_total}  B={b_overall * 100:5.2f}%   C={c_overall * 100:5.2f}%")
    threshold = args.green_threshold
    green = b_overall >= threshold and c_overall >= threshold
    print(f"GREEN LIGHT? {'YES' if green else 'NO'}  "
          f"(threshold: both >= {threshold * 100:.0f}%)")
    return 0 if green else 4


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())

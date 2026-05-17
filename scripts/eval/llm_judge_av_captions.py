#!/usr/bin/env python
"""LLM-as-judge eval of AV-generated captions against the cached camera frames.

Pipeline per sample:

    activation h
        --> AV.generate(h)  -->  caption_pred  (the thing we're evaluating)
        --> gold caption (from labels.jsonl)   (reference)
        --> camera frames for the source step  (ground truth)

We then send the **same image(s)** the labeler saw to ``gpt-5.1`` (via the
existing ``nla.labeling.grader.GPT51Grader`` infra) along with the candidate
caption, and ask it to grade on three axes:

    Axis B -- GROUNDING:                specific vs generic
    Axis C -- APPROPRIATENESS:          appropriate vs inappropriate
    Axis D -- TEMPLATE_DISTINGUISHABLE: specific vs template (anti-collapse)

Axis D is the V3-specific anti-template-collapse axis. It is a stricter
form of axis B: instead of "could this label describe a different scene?"
it asks "would this exact caption verbatim also be a reasonable label for
many *different but similar* manipulation scenes?". Pass = the caption
commits to scene-fingerprinting details that wouldn't generalise; fail =
the caption is reusable boilerplate that happens to mention the workspace.
V2 had high axis-B pass but low axis-D pass because of template collapse.

We grade *both* the gold label (sanity floor) and the AV-generated caption,
so any drop from gold→pred is the AV's contribution.

Usage (LIBERO)::

    OPENAI_API_KEY=... PYTHONPATH=src python scripts/eval/llm_judge_av_captions.py \
        --ckpt-dir         data/sft/libero_goal_pilot_v3 \
        --activations-root data/activations/libero_goal_pilot \
        --labels-jsonl     data/labels/libero_goal_pilot/labels.jsonl \
        --frames-cache     data/labels/libero_goal_pilot/frames_cache \
        --video-keys       image wrist_image \
        --per-position     12 \
        --out-jsonl        data/sft/libero_goal_pilot_v3/llm_judge.jsonl

``--video-keys`` is required: it lists the camera-key tokens used to build
per-row image filenames ``{frames_cache}/{source_id}__{video_key}.jpg``.
Pass the *exact* tokens your frame cache uses (e.g. ``image wrist_image``
for LIBERO). The script silently drops any row whose tokens resolve to
zero on-disk frames and logs a warning so an empty cache cannot
masquerade as a real grade.

Output:
    - JSONL row per (variant, sample) with full grade JSON.
    - Console summary aggregated by ``position_type`` and ``variant``.
    - Console side-by-side dump of N samples with the judge's verbatim reasons.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--activations-root", required=True)
    p.add_argument("--labels-jsonl", required=True)
    p.add_argument("--frames-cache", required=True,
                   help="Directory of cached camera frames as {source_id}__{video_key}.jpg")
    p.add_argument("--video-keys", nargs="+", required=True,
                   help="Camera-key tokens that compose per-row image filenames "
                        "as {frames_cache}/{source_id}__{video_key}.jpg "
                        "(e.g. 'image wrist_image' for LIBERO). The tokens must "
                        "match what your labeling pipeline / extract_label_frames.py "
                        "wrote into --frames-cache; rows that resolve to zero "
                        "on-disk frames are dropped from grading.")
    p.add_argument("--per-position", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--held-out-fraction", type=float, default=0.05)
    p.add_argument("--split-by", default="episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--judge-model", default=None,
                   help="Override OPENAI_GRADER_MODEL (default: gpt-5.1).")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--print-n", type=int, default=10,
                   help="How many side-by-side samples to print to console.")
    return p


def _image_paths_for(
    source_id: str,
    frames_cache: Path,
    video_keys: list[str],
) -> list[str]:
    """Resolve cached frame files for ``source_id`` against ``video_keys``.

    For each ``video_key`` we look up ``{frames_cache}/{source_id}__{video_key}.jpg``
    and append it iff the file exists. Missing keys are silently skipped so a
    partially-cached row still gets judged on the keys we do have. The caller
    is responsible for dropping rows whose returned list is empty.
    """
    paths: list[str] = []
    for key in video_keys:
        candidate = frames_cache / f"{source_id}__{key}.jpg"
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def _instruction_lookup_from_labels(labels_path: Path) -> dict[str, str]:
    """Map source_example_id -> instruction (first occurrence)."""
    out: dict[str, str] = {}
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("meta") or {}
            src = meta.get("source_example_id")
            ins = meta.get("instruction")
            if src and ins and src not in out:
                out[src] = ins
    return out


def _summarize(rows: list[dict]) -> dict:
    """Aggregate grade rows by (variant_id, position_type).

    Reports per-bucket pass rates on all three axes. Axis D's denominator
    only counts rows whose row actually has axis D populated, so a partial
    backfill of legacy B/C-only grades doesn't artificially deflate it.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["variant_id"], r.get("position_type") or "_unk")
        by_key.setdefault(key, []).append(r)

    summary: dict = {}
    for (variant, ptype), bucket in sorted(by_key.items()):
        n = len(bucket)
        b_pass = sum(
            1 for r in bucket
            if (r.get("grounding") or {}).get("verdict") == "specific"
        )
        c_pass = sum(
            1 for r in bucket
            if (r.get("appropriateness") or {}).get("verdict") == "appropriate"
        )
        d_rows = [r for r in bucket if r.get("template_distinguishable")]
        n_d = len(d_rows)
        d_pass = sum(
            1 for r in d_rows
            if (r.get("template_distinguishable") or {}).get("verdict") == "specific"
        )
        summary[f"{variant}/{ptype}"] = {
            "n": n,
            "grounding_specific_pct": b_pass / n if n else None,
            "appropriateness_appropriate_pct": c_pass / n if n else None,
            "n_template": n_d,
            "template_distinguishable_specific_pct": (
                d_pass / n_d if n_d else None
            ),
        }
    return summary


async def _amain(args) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. `export OPENAI_API_KEY=sk-...` first.",
              file=sys.stderr)
        return 2

    if args.judge_model:
        os.environ["OPENAI_GRADER_MODEL"] = args.judge_model

    from nla.training.checkpoint import load_av_from_sft
    from nla.training.dataset import LabeledPositionDataset, collate_labeled_positions
    from nla.labeling.grader import GradeInput, grade_many_async

    ckpt_dir = Path(args.ckpt_dir)
    frames_cache = Path(args.frames_cache)
    labels_path = Path(args.labels_jsonl)

    print(f"Loading AV from {ckpt_dir}/av/ ...")
    av = load_av_from_sft(ckpt_dir / "av", device=args.device, freeze=True)

    print(f"Loading val labels (split_by={args.split_by}, held_out={args.held_out_fraction}) ...")
    val_ds = LabeledPositionDataset(
        args.activations_root, args.labels_jsonl,
        seed=args.seed,
        held_out_fraction=args.held_out_fraction,
        held_out=True,
        split_by=args.split_by,
    )
    print(f"  -> {len(val_ds)} val rows")

    by_pos: dict[str, list[int]] = {}
    for i, entry in enumerate(val_ds.labels):
        by_pos.setdefault(entry.position_type, []).append(i)
    print(f"  -> per-position counts: { {k: len(v) for k, v in by_pos.items()} }")

    instr_map = _instruction_lookup_from_labels(labels_path)

    rng = torch.Generator().manual_seed(args.seed)
    all_inputs: list[GradeInput] = []
    sample_index: list[dict] = []   # parallel metadata for console printing

    for ptype, indices in by_pos.items():
        perm = torch.randperm(len(indices), generator=rng).tolist()
        chosen = [indices[k] for k in perm[: args.per_position]]
        # Filter to ones we have images for
        viable = []
        for i in chosen:
            entry = val_ds.labels[i]
            ipaths = _image_paths_for(entry.source_example_id, frames_cache, args.video_keys)
            if ipaths:
                viable.append((i, ipaths))
        if len(viable) < len(chosen):
            print(f"  warn: {len(chosen) - len(viable)} samples in {ptype} had no cached frames; dropping them")
        if not viable:
            continue

        batch = collate_labeled_positions([val_ds[i] for i, _ in viable])
        acts = batch["activations"].to(args.device)

        do_sample = float(args.temperature) > 0.0
        with torch.no_grad():
            out = av.generate(
                activations=acts,
                position_types=batch["position_type"],
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=float(args.temperature) if do_sample else 1.0,
            )
        gen_texts = out["text"]

        for b, (i, ipaths) in enumerate(viable):
            entry = val_ds.labels[i]
            instr = instr_map.get(entry.source_example_id, "")
            example_id_short = f"{entry.source_example_id}@p{entry.position_index}_{entry.position_type}"

            gold_input = GradeInput(
                example_id=example_id_short,
                variant_id="gold",
                description=entry.description,
                instruction=instr,
                position_type=entry.position_type,
                image_paths=ipaths,
                seq_len=None,
                position_index=entry.position_index,
            )
            pred_input = GradeInput(
                example_id=example_id_short,
                variant_id="av_pred",
                description=gen_texts[b].strip(),
                instruction=instr,
                position_type=entry.position_type,
                image_paths=ipaths,
                seq_len=None,
                position_index=entry.position_index,
            )
            all_inputs.extend([gold_input, pred_input])
            sample_index.append({
                "example_id": example_id_short,
                "position_type": entry.position_type,
                "instruction": instr,
                "gold": entry.description,
                "generated": gen_texts[b].strip(),
                "images": ipaths,
            })

    out_path = Path(args.out_jsonl)
    print(f"\nGrading {len(all_inputs)} (gold + AV) pairs with concurrency={args.concurrency} ...")
    n_new = await grade_many_async(
        all_inputs,
        output_jsonl=out_path,
        concurrency=args.concurrency,
        resume=True,
    )
    print(f"  -> {n_new} new grades written to {out_path}")

    # Load + decorate with position_type for aggregation
    rows = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            # Find the matching sample to attach position_type
            for s in sample_index:
                if row["example_id"] == s["example_id"]:
                    row["position_type"] = s["position_type"]
                    break
            rows.append(row)

    summary = _summarize(rows)
    print("\n" + "=" * 78)
    print(
        "Aggregate verdicts  "
        "(B=grounding-specific%   C=appropriate%   D=template-distinguishable-specific%)"
    )
    print("=" * 78)
    for k in sorted(summary):
        s = summary[k]
        b = s["grounding_specific_pct"]
        c = s["appropriateness_appropriate_pct"]
        d = s["template_distinguishable_specific_pct"]
        b_s = f"{b*100:5.1f}%" if b is not None else "  n/a"
        c_s = f"{c*100:5.1f}%" if c is not None else "  n/a"
        d_s = f"{d*100:5.1f}%" if d is not None else "  n/a"
        print(
            f"  {k:<32}  n={s['n']:>3}  B={b_s}   C={c_s}   "
            f"D={d_s} (n={s['n_template']})"
        )

    print("\n" + "=" * 78)
    print(f"Side-by-side judgements (first {args.print_n} samples)")
    print("=" * 78)
    grade_by_key = {(r["variant_id"], r["example_id"]): r for r in rows}
    for s in sample_index[: args.print_n]:
        gold = grade_by_key.get(("gold", s["example_id"]), {})
        pred = grade_by_key.get(("av_pred", s["example_id"]), {})
        print()
        print(f"--- {s['example_id']}   position={s['position_type']} ---")
        print(f"instruction: {s['instruction']}")
        gold_g = (gold.get("grounding") or {})
        gold_a = (gold.get("appropriateness") or {})
        gold_d = (gold.get("template_distinguishable") or {})
        pred_g = (pred.get("grounding") or {})
        pred_a = (pred.get("appropriateness") or {})
        pred_d = (pred.get("template_distinguishable") or {})
        print(
            f"[GOLD]   B={gold_g.get('verdict','?'):<10}  "
            f"C={gold_a.get('verdict','?'):<14}  "
            f"D={gold_d.get('verdict','?')}"
        )
        print(f"         B-reason: {gold_g.get('reason','')}")
        print(f"         C-reason: {gold_a.get('reason','')}")
        print(f"         D-reason: {gold_d.get('reason','')}")
        print(
            f"[AV]     B={pred_g.get('verdict','?'):<10}  "
            f"C={pred_a.get('verdict','?'):<14}  "
            f"D={pred_d.get('verdict','?')}"
        )
        print(f"         B-reason: {pred_g.get('reason','')}")
        print(f"         C-reason: {pred_a.get('reason','')}")
        print(f"         D-reason: {pred_d.get('reason','')}")
        # Show first 2 bullets of gold + generated for orientation
        def _first2(text):
            return "\n           ".join(text.strip().splitlines()[:2])
        print(f"GOLD>    {_first2(s['gold'])}")
        print(f"GEN >    {_first2(s['generated'])}")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Diagnostic: how intent-conditional are the AV's captions?

For each held-out activation, generate two captions from the same
activation but with two different ``target_intent`` strings (the row's
matched intent and the CF pair's mismatched_source intent). Then measure
how different the two captions are -- if AV is genuinely intent-aware,
the captions should differ substantively (different "target", "spatial",
"gripper" bullets). If AV only differs on the "- task:" line, the codec
carries very little intent-specific signal and CF eval steer_lift will
be small regardless of how good val/cosine looks.

Two metrics:
  - char_overlap_ratio: fraction of characters identical between the two
    captions (1.0 = same caption, 0.0 = totally different). High value
    => AV is NOT actually intent-conditional.
  - bullet_difference_count: how many of the 5-6 bullets differ between
    the two captions. Ideally 3+ bullets differ.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/av_caption_intent_diff.py \\
        --sft-dir data/sft/v9_combined_12k \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
        --n-samples 10 \\
        --out-json data/eval/v9_combined_12k_paired_captions.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch


def _bullet_lines(text: str) -> list[str]:
    return [
        line.strip() for line in (text or "").split("\n")
        if line.strip().startswith("-")
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--activations-root", required=True)
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    sft_dir = Path(args.sft_dir)
    print(f"[caption_diff] loading AV from {sft_dir / 'av'} ...", flush=True)
    t0 = time.time()
    from nla.extraction.storage import ActivationShardReader
    from nla.training.checkpoint import load_av_from_sft

    reader = ActivationShardReader(args.activations_root)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)
    print(f"[caption_diff] AV loaded ({time.time()-t0:.1f}s)", flush=True)

    rng = random.Random(args.seed)
    rows = []
    with open(args.pairs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("source_example_id") or not r.get("target_intent"):
                continue
            si = (r.get("source_intent") or "").strip()
            ti = (r.get("target_intent") or "").strip()
            if not si or si == ti:
                continue
            rows.append(r)
    rng.shuffle(rows)
    rows = rows[: args.n_samples]
    print(f"[caption_diff] sampled {len(rows)} CF pairs", flush=True)

    results = []
    for i, row in enumerate(rows):
        sid = row["source_example_id"]
        matched_intent = row["target_intent"]
        mismatched_intent = row["source_intent"]
        item = reader.get(sid)
        features = item["features"]
        # Use the row's position_index if labeled, else sample.
        if row.get("position_index") is not None and row.get("position_type"):
            pos = int(row["position_index"])
            ptype = str(row["position_type"])
        else:
            from nla.training.dataset import TokenPositionSampler
            sampler = TokenPositionSampler(seed=args.seed + i)
            ptype, pos = sampler.sample(item["attention_mask"], item["image_mask"])
        h = features[pos].contiguous().to(torch.float32)

        with torch.no_grad():
            out_matched = av.generate(
                h.unsqueeze(0).to(args.device), [ptype],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0, top_p=1.0, do_sample=False,
                target_intent_texts=[matched_intent],
            )
            out_mismatched = av.generate(
                h.unsqueeze(0).to(args.device), [ptype],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0, top_p=1.0, do_sample=False,
                target_intent_texts=[mismatched_intent],
            )
        cap_m = out_matched["text"][0].strip()
        cap_mm = out_mismatched["text"][0].strip()

        # Metrics
        overlap = SequenceMatcher(None, cap_m, cap_mm).ratio()
        b_m = _bullet_lines(cap_m)
        b_mm = _bullet_lines(cap_mm)
        b_diff = sum(
            1 for x, y in zip(b_m, b_mm) if x != y
        ) + abs(len(b_m) - len(b_mm))

        results.append({
            "source_example_id": sid,
            "matched_intent": matched_intent,
            "mismatched_intent": mismatched_intent,
            "caption_matched": cap_m,
            "caption_mismatched": cap_mm,
            "char_overlap_ratio": overlap,
            "n_bullets_matched": len(b_m),
            "n_bullets_mismatched": len(b_mm),
            "bullet_difference_count": b_diff,
        })
        print(
            f"  [{i+1:>2}/{len(rows)}] {sid}\n"
            f"     matched:    {matched_intent[:80]!r}\n"
            f"     mismatched: {mismatched_intent[:80]!r}\n"
            f"     overlap={overlap:.3f}  bullets_diff={b_diff}",
            flush=True,
        )

    # Aggregate
    if results:
        mean_overlap = sum(r["char_overlap_ratio"] for r in results) / len(results)
        mean_b_diff = sum(r["bullet_difference_count"] for r in results) / len(results)
        print("\n=== AGGREGATE ===")
        print(f"  n_samples              = {len(results)}")
        print(f"  mean char_overlap      = {mean_overlap:.4f}")
        print(f"  mean bullet_diff_count = {mean_b_diff:.2f}")
        # Interpretation hint
        if mean_overlap > 0.9:
            print("  → AV is NOT intent-conditional. Captions barely change with intent.")
            print("    Codec steering signal will be small; expect steer_lift ~ v8 level.")
        elif mean_overlap > 0.7:
            print("  → AV is weakly intent-conditional. Only the task: line shifts.")
            print("    Codec will steer slightly; expect modest steer_lift improvement.")
        elif mean_overlap > 0.5:
            print("  → AV is moderately intent-conditional. Multiple bullets shift.")
            print("    Expect publishable steer_lift improvement over v8.")
        else:
            print("  → AV is strongly intent-conditional. Captions substantively differ.")
            print("    High chance of clear publishable steer_lift > +0.10.")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "config": vars(args),
        "n_samples": len(results),
        "mean_char_overlap": mean_overlap if results else None,
        "mean_bullet_diff_count": mean_b_diff if results else None,
        "samples": results,
    }, indent=2))
    print(f"\nWrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

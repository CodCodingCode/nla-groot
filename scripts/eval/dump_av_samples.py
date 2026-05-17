#!/usr/bin/env python
"""Eyeball eval: dump gold vs AV-generated captions on held-out activations.

For each ``position_type`` in ``POSITION_MIX``, draw N val examples, run
``AV.generate`` (greedy and sampled), and print gold caption + generated
caption + per-row FVE for both teacher-forced and closed-loop.

This is the human-readable companion to the numerical metrics — it tells you
*why* a given FVE is what it is.

Usage::

    PYTHONPATH=src python scripts/eval/dump_av_samples.py \
        --ckpt-dir         data/sft/libero_goal_pilot_v3 \
        --activations-root data/activations/libero_goal_pilot \
        --labels-jsonl     data/labels/libero_goal_pilot/labels.jsonl \
        --per-position     6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True, help="SFT run dir with av/ and ar/ subdirs.")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--labels-jsonl", required=True)
    p.add_argument("--per-position", type=int, default=6)
    p.add_argument("--temperatures", type=float, nargs="+", default=(0.0, 0.7))
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--held-out-fraction", type=float, default=0.05,
                   help="Must match the value used at training time.")
    p.add_argument("--split-by", default="episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-jsonl", default=None,
                   help="Optional: also write each sample as a JSONL row.")
    return p


def _per_row_fve(target: torch.Tensor, pred: torch.Tensor) -> tuple[float, float]:
    """Return (cosine, MSE) for a single row pair."""
    t = target.float()
    p = pred.float()
    cos = torch.nn.functional.cosine_similarity(t.unsqueeze(0), p.unsqueeze(0)).item()
    mse = ((t - p) ** 2).mean().item()
    return cos, mse


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from nla.training.checkpoint import load_av_from_sft, load_ar_from_sft
    from nla.training.dataset import LabeledPositionDataset, collate_labeled_positions

    print(f"Loading AV from {args.ckpt_dir}/av/ ...")
    av = load_av_from_sft(Path(args.ckpt_dir) / "av", device=args.device, freeze=True)
    print(f"Loading AR from {args.ckpt_dir}/ar/ ...")
    ar = load_ar_from_sft(Path(args.ckpt_dir) / "ar", device=args.device, freeze=True)
    alpha = ar.cfg.alpha
    print(f"α (from AR config) = {alpha:.4f}")

    print(f"Loading held-out labels (split_by={args.split_by}, held_out_fraction={args.held_out_fraction}) ...")
    val_ds = LabeledPositionDataset(
        args.activations_root, args.labels_jsonl,
        seed=args.seed,
        held_out_fraction=args.held_out_fraction,
        held_out=True,
        split_by=args.split_by,
    )
    print(f"  -> {len(val_ds)} val rows total")

    # Group val indices by position_type so we can pick N per stratum.
    by_pos: dict[str, list[int]] = {}
    for i, entry in enumerate(val_ds.labels):
        by_pos.setdefault(entry.position_type, []).append(i)
    print(f"  -> per-position counts: { {k: len(v) for k, v in by_pos.items()} }")

    rng = torch.Generator().manual_seed(args.seed)
    out_rows: list[dict] = []

    for ptype, indices in by_pos.items():
        if not indices:
            continue
        # Sample without replacement up to per-position items.
        perm = torch.randperm(len(indices), generator=rng).tolist()
        chosen = [indices[k] for k in perm[: args.per_position]]
        batch = collate_labeled_positions([val_ds[i] for i in chosen])
        acts = batch["activations"].to(args.device)
        gold = batch["description"]
        pos_types = batch["position_type"]

        # Teacher-forced AR (gold caption -> ĥ).
        with torch.no_grad():
            tf_pred = ar(gold, device=args.device).float() * alpha

        # Closed-loop AV.generate at each requested temperature.
        gen_by_temp: dict[float, list[str]] = {}
        cl_pred_by_temp: dict[float, torch.Tensor] = {}
        for temp in args.temperatures:
            do_sample = float(temp) > 0.0
            with torch.no_grad():
                out = av.generate(
                    activations=acts,
                    position_types=pos_types,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=float(temp) if do_sample else 1.0,
                )
            gen_by_temp[temp] = out["text"]
            with torch.no_grad():
                cl_pred_by_temp[temp] = ar(out["text"], device=args.device).float() * alpha

        print()
        print("=" * 78)
        print(f"position_type = {ptype}   (showing {len(chosen)} samples)")
        print("=" * 78)
        for b in range(acts.shape[0]):
            tgt_row = acts[b].float()
            tf_cos, tf_mse = _per_row_fve(tgt_row, tf_pred[b])
            print()
            print(f"--- sample {b+1}/{len(chosen)}  position={ptype} ---")
            print(f"[GOLD]")
            for line in gold[b].splitlines():
                print(f"  {line}")
            print(f"  -> teacher_forced  cos={tf_cos:+.3f}  mse={tf_mse:.2f}")
            for temp in args.temperatures:
                tag = "greedy" if temp == 0.0 else f"t={temp:g}"
                cl_cos, cl_mse = _per_row_fve(tgt_row, cl_pred_by_temp[temp][b])
                print(f"[AV {tag}]")
                gen_text = gen_by_temp[temp][b].strip()
                for line in gen_text.splitlines():
                    print(f"  {line}")
                print(f"  -> closed_loop({tag})  cos={cl_cos:+.3f}  mse={cl_mse:.2f}")
                out_rows.append({
                    "position_type": ptype,
                    "temperature": temp,
                    "gold": gold[b],
                    "generated": gen_text,
                    "tf_cosine": tf_cos,
                    "tf_mse": tf_mse,
                    "cl_cosine": cl_cos,
                    "cl_mse": cl_mse,
                })

    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_jsonl).open("w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nWrote {len(out_rows)} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Diagnostic 1 — teacher-forced vs closed-loop reconstruction split.

Isolates AR from AV on the *reconstruction* axis, on the same 100 samples as
the cached n=100 CF eval. No sim, GPU-light (~1-2 min).

Two reconstructions of the real image-patch grid ``h_grid`` [128, H]:
  - teacher_forced :  ĥ = AR(gold GPT caption)        -> AR's ceiling
  - closed_loop    :  ĥ = AR(AV(h_pos, target_intent)) -> end-to-end

Decision rule:
  - teacher_forced already weak  -> AR is the bottleneck (fails on a perfect caption)
  - teacher_forced good, closed_loop collapses -> AV is the bottleneck
  The gap (tf - cl) = AV's damage; the tf floor = AR's ceiling.

Reported per arm: fve, mse, cosine (raw), cosine_residual (mean-centered).
Raw cosine is inflated by the shared mean direction; cosine_residual and fve
measure the scene-distinguishing part that actually matters.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/eval/diag/m1_recon_split.py \
      --sft-dir data/sft/v9_combined_12k --device cuda \
      --out-json data/eval/diag/m1_recon_split_v9_combined_12k.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft
from nla.training.fve import fve_per_token
from nla.training.sim_reward import encode_texts_with_ar
from sample_set import ROOT, load_samples  # script-dir is on sys.path when run as a file


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """[B,K,H] or [B,H] -> [N, H] float32 on CPU."""
    t = t.detach().float().cpu()
    return t.reshape(-1, t.shape[-1]) if t.dim() == 3 else t


def _metrics(target: torch.Tensor, pred: torch.Tensor) -> dict:
    """Raw fve/mse/cosine + mean-centered residual cosine. target,pred [N,H]."""
    m = fve_per_token(target, pred)  # uses per-dim mean in its FVE denominator
    mu = target.mean(dim=0, keepdim=True)
    tc, pc = target - mu, pred - mu
    cos_res = torch.nn.functional.cosine_similarity(tc, pc, dim=-1).mean().item()
    return {"fve": m["fve"], "mse": m["mse"], "cosine": m["cosine"],
            "cosine_residual": cos_res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--labels-jsonl",
                    default="data/labels/libero_4suite_v4_combined/labels.jsonl",
                    help="Gold-caption source. NOTE: only v4 labels cover the CF "
                         "eval positions; v6 labeled different steps (0/100 join). "
                         "The v9 AR trained on v6, so teacher-forced here is OOD and "
                         "exaggerated -- read the real AR ceiling from training "
                         "val fve vs closed_greedy/fve instead. closed_loop is valid.")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    sft_dir = Path(args.sft_dir)
    if not sft_dir.is_absolute():
        sft_dir = ROOT / sft_dir

    samples = load_samples(limit=args.limit, load_tensors=True, labels=args.labels_jsonl)
    samples = [s for s in samples if s.gold_caption]  # need gold for teacher-forced
    print(f"[m1] {len(samples)} samples with gold captions from {args.labels_jsonl}")

    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)

    tgt_rows, tf_rows, cl_rows = [], [], []
    captions = []
    t0 = time.time()
    for i, s in enumerate(samples):
        h_pos = torch.from_numpy(s.h_pos).unsqueeze(0).to(args.device)   # [1,H]
        target = torch.from_numpy(s.h_grid)                              # [128,H]

        # closed-loop caption from AV (mirror compare_cf_steer_checkpoints.py)
        with torch.no_grad():
            out = av.generate(h_pos, [s.position_type],
                              max_new_tokens=args.max_new_tokens,
                              temperature=0.0, do_sample=False,
                              target_intent_texts=[s.target_intent])
        cap = out["text"][0]
        captions.append(cap)

        h_tf = _flatten(encode_texts_with_ar(ar, [s.gold_caption], device=args.device))
        h_cl = _flatten(encode_texts_with_ar(ar, [cap], device=args.device))

        tgt_rows.append(target)
        tf_rows.append(h_tf)
        cl_rows.append(h_cl)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {time.time()-t0:.0f}s", flush=True)

    target = torch.cat(tgt_rows, dim=0)
    tf = torch.cat(tf_rows, dim=0)
    cl = torch.cat(cl_rows, dim=0)
    # guard against any shape mismatch between AR output grid and real grid
    assert tf.shape == target.shape == cl.shape, (tf.shape, target.shape, cl.shape)

    teacher_forced = _metrics(target, tf)
    closed_loop = _metrics(target, cl)
    gap = {k: teacher_forced[k] - closed_loop[k] for k in teacher_forced}

    result = {
        "sft_dir": str(sft_dir),
        "n_samples": len(samples),
        "teacher_forced": teacher_forced,   # AR ceiling (gold caption)
        "closed_loop": closed_loop,         # end-to-end (AV caption)
        "av_damage_gap": gap,               # tf - cl ; large => AV is the weak link
    }

    out = Path(args.out_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\n=== Diagnostic 1: reconstruction split ===")
    for name, d in (("teacher_forced (AR ceiling)", teacher_forced),
                    ("closed_loop   (AV->AR)     ", closed_loop)):
        print(f"  {name}: fve={d['fve']:+.3f}  cos={d['cosine']:.3f}  "
              f"cos_resid={d['cosine_residual']:+.3f}  mse={d['mse']:.3f}")
    print(f"  AV damage (tf-cl): fve {gap['fve']:+.3f}  cos_resid {gap['cosine_residual']:+.3f}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

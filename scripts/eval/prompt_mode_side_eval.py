#!/usr/bin/env python
"""Compare codec reconstruction quality under TRAINING vs EVAL prompt modes.

Background
----------
SFT training trains AV exclusively with ``target_intent=None`` (multi-slot
descriptive prompt, K=128 image_patch tokens, 4-5 bullet target). At
``compare_cf_steer_checkpoints.py`` eval time, AV is called with
``target_intent_texts=[intent]`` which forces ``num_slots=1`` and switches
to the intent-conditioned template. AV has never seen this prompt during
SFT -- it's running OOD on the input side.

This script measures the codec's reconstruction quality under both prompt
modes side-by-side using the SAME activations as input. If training-mode
codec significantly beats eval-mode codec on cosine / MSE / FVE-style
metrics, that's evidence the OOD prompt is degrading the codec.

Outputs
-------
- Per-row: cosine, MSE in both modes, plus the two captions side-by-side
- Aggregate: mean cosine / MSE per mode
- Random-control: replace each ĥ with a magnitude-matched gaussian and
  measure the same metrics. If a mode's cosine is barely above random,
  the codec is content-free in that mode.

Usage
-----
    PYTHONPATH=src .venv/bin/python scripts/eval/prompt_mode_side_eval.py \\
        --sft-dir data/sft/v8_full_6400 \\
        --activations-root data/activations/libero_4suite_v4_combined \\
        --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
        --n-samples 16 \\
        --out-json data/eval/v8_full_6400_prompt_mode_side.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft-dir", required=True)
    p.add_argument("--activations-root", required=True)
    p.add_argument("--pairs-path", required=True,
                   help="CF pairs JSONL -- we use the rows just to get "
                        "(source_id, source_intent) pairs; we don't run sim.")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", required=True)
    p.add_argument("--position-type", default="image_patch",
                   help="Token role used by AV's prompt builder. Must match "
                        "how the source activation was extracted. v8 was "
                        "trained on image_patch.")
    args = p.parse_args()

    sft_dir = Path(args.sft_dir)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[prompt_mode_side_eval] loading models from {sft_dir}", flush=True)
    t0 = time.time()
    from nla.extraction.storage import ActivationShardReader
    from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft

    reader = ActivationShardReader(args.activations_root)
    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)
    print(f"  models loaded ({time.time()-t0:.1f}s)", flush=True)

    # Sample N pairs deterministically (skipping non-CF rows).
    rng = random.Random(args.seed)
    rows: list[dict] = []
    with open(args.pairs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("source_example_id") and r.get("source_intent")):
                continue
            rows.append(r)
    rng.shuffle(rows)
    rows = rows[: args.n_samples]
    print(f"[prompt_mode_side_eval] sampled {len(rows)} rows", flush=True)

    results: list[dict] = []
    for i, row in enumerate(rows):
        sid = row["source_example_id"]
        intent = row["source_intent"]
        item = reader.get(sid)
        features = item["features"]
        attn = item["attention_mask"]
        img_m = item["image_mask"]
        # Pick a deterministic image_patch position for this row (we'll
        # extract the full multi-slot activation set from the same row, so
        # this position just identifies the "anchor" patch we score against).
        from nla.training.dataset import TokenPositionSampler
        sampler = TokenPositionSampler(seed=args.seed + i)
        ptype, pos = sampler.sample(attn, img_m)
        if ptype != args.position_type:
            # Force to image_patch for a fair test; otherwise position-mix
            # could swamp the prompt-mode signal.
            from nla.extraction.sampler import _image_patch_index
            local_rng = np.random.default_rng(args.seed + i)
            pos_alt = _image_patch_index(
                torch.as_tensor(attn), torch.as_tensor(img_m), local_rng,
            )
            if pos_alt is None:
                continue
            ptype, pos = args.position_type, int(pos_alt)
        h_unscaled = features[pos].contiguous().to(torch.float32)

        # === MODE A: TRAINING-MODE PROMPT (multi-slot, no intent) ===
        # AV.generate with no target_intent -> multi-slot K=128 prompt
        # (the prompt SFT actually trained on).
        # For multi-slot mode, AV needs the FULL set of patch activations
        # for this row, not just the one at `pos`. The dataset's standard
        # extraction stores them in features at the image_patch positions.
        # We rebuild the multi-slot input the same way the SFT dataset does.
        with torch.no_grad():
            try:
                out_train = av.generate(
                    h_unscaled.unsqueeze(0).to(args.device),
                    [ptype],
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    target_intent_texts=None,  # TRAIN MODE
                )
                caption_train = out_train["text"][0]
            except Exception as e:
                caption_train = f"<error: {e}>"

        # === MODE B: EVAL-MODE PROMPT (single-slot + intent) ===
        with torch.no_grad():
            try:
                out_eval = av.generate(
                    h_unscaled.unsqueeze(0).to(args.device),
                    [ptype],
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    target_intent_texts=[intent],  # EVAL MODE
                )
                caption_eval = out_eval["text"][0]
            except Exception as e:
                caption_eval = f"<error: {e}>"

        # AR on both captions -> ĥ per mode
        with torch.no_grad():
            try:
                ar_train_scaled = ar([caption_train], device=args.device)
                ar_eval_scaled = ar([caption_eval], device=args.device)
                # AR returns alpha-scaled (or per the head, the alpha-corrected
                # output). To get backbone-space vectors:
                alpha = float(ar.cfg.alpha)
                hat_train = (ar_train_scaled.float() * alpha)[0]
                hat_eval = (ar_eval_scaled.float() * alpha)[0]
            except Exception as e:
                results.append({
                    "source_example_id": sid,
                    "intent": intent,
                    "error": f"ar forward failed: {e}",
                })
                continue

        h_cpu = h_unscaled.cpu()
        # If AR head is spatial (K>1), we get [K, H]. Compare against the
        # SAME source activation (broadcast / mean-pool).
        def _metrics(hat: torch.Tensor, gt: torch.Tensor) -> dict:
            if hat.dim() == 2:
                # [K, H] -> compare gt to each row, take mean (per-position
                # reconstruction error), plus also compare gt to the mean-pooled
                # hat for the "broadcast" view.
                gt_b = gt.unsqueeze(0).expand_as(hat)
                cos_per = torch.nn.functional.cosine_similarity(
                    hat.float(), gt_b.float(), dim=-1,
                ).mean().item()
                mse_per = (hat.float() - gt_b.float()).pow(2).mean().item()
                hat_mean = hat.mean(dim=0)
                cos_mean = torch.nn.functional.cosine_similarity(
                    hat_mean.unsqueeze(0).float(),
                    gt.unsqueeze(0).float(),
                    dim=-1,
                ).item()
                mse_mean = (hat_mean.float() - gt.float()).pow(2).mean().item()
                norm_hat = hat_mean.norm().item()
            else:
                cos_per = torch.nn.functional.cosine_similarity(
                    hat.unsqueeze(0).float(), gt.unsqueeze(0).float(), dim=-1,
                ).item()
                mse_per = (hat.float() - gt.float()).pow(2).mean().item()
                cos_mean = cos_per
                mse_mean = mse_per
                norm_hat = hat.norm().item()
            return {
                "cosine_per_position": cos_per,
                "mse_per_position": mse_per,
                "cosine_mean_pooled": cos_mean,
                "mse_mean_pooled": mse_mean,
                "norm_hat": norm_hat,
                "norm_gt": gt.norm().item(),
            }

        m_train = _metrics(hat_train.cpu(), h_cpu)
        m_eval = _metrics(hat_eval.cpu(), h_cpu)

        # Random control: gaussian matched in shape + norm.
        rand_v = torch.randn_like(hat_train.cpu())
        # Rescale each random row to match the norm of the corresponding
        # hat_train row (for spatial), or just hat_train.norm() (for scalar).
        if rand_v.dim() == 2:
            scale = hat_train.cpu().norm(dim=-1, keepdim=True) / rand_v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            scale = hat_train.cpu().norm() / rand_v.norm().clamp_min(1e-6)
        rand_v = rand_v * scale
        m_rand = _metrics(rand_v, h_cpu)

        results.append({
            "source_example_id": sid,
            "intent": intent,
            "position_type": ptype,
            "position_index": int(pos),
            "caption_train_mode": caption_train,
            "caption_eval_mode": caption_eval,
            "metrics_train_mode": m_train,
            "metrics_eval_mode": m_eval,
            "metrics_random_control": m_rand,
        })
        # Concise per-row stdout
        print(
            f"[{i+1:>2}/{len(rows)}] {sid}\n"
            f"     intent: {intent[:80]!r}\n"
            f"     train  cosine={m_train['cosine_per_position']:+.3f} "
            f"mse={m_train['mse_per_position']:.1f} norm={m_train['norm_hat']:.1f}\n"
            f"     eval   cosine={m_eval['cosine_per_position']:+.3f} "
            f"mse={m_eval['mse_per_position']:.1f} norm={m_eval['norm_hat']:.1f}\n"
            f"     rand   cosine={m_rand['cosine_per_position']:+.3f} "
            f"mse={m_rand['mse_per_position']:.1f} norm={m_rand['norm_hat']:.1f}",
            flush=True,
        )

    # Aggregate
    valid = [r for r in results if "error" not in r]
    def _mean(key1: str, key2: str) -> float:
        vals = [r[key1][key2] for r in valid]
        return sum(vals) / max(1, len(vals))

    aggregate = {}
    for mode in ("metrics_train_mode", "metrics_eval_mode", "metrics_random_control"):
        aggregate[mode] = {
            "cosine_per_position": _mean(mode, "cosine_per_position"),
            "mse_per_position": _mean(mode, "mse_per_position"),
            "cosine_mean_pooled": _mean(mode, "cosine_mean_pooled"),
            "mse_mean_pooled": _mean(mode, "mse_mean_pooled"),
            "norm_hat": _mean(mode, "norm_hat"),
        }

    aggregate["gt_norm_mean"] = sum(r["metrics_train_mode"]["norm_gt"] for r in valid) / max(1, len(valid))

    print("\n=== AGGREGATE ===")
    print(f"  n_valid = {len(valid)}/{len(results)}  "
          f"(gt norm mean = {aggregate['gt_norm_mean']:.1f})")
    for mode_short, mode_key in [
        ("TRAIN MODE (multi-slot, no intent)", "metrics_train_mode"),
        ("EVAL MODE  (single-slot + intent) ", "metrics_eval_mode"),
        ("RANDOM CTRL (magnitude-matched)   ", "metrics_random_control"),
    ]:
        m = aggregate[mode_key]
        print(f"  {mode_short}: cos_per={m['cosine_per_position']:+.4f}  "
              f"mse_per={m['mse_per_position']:>7.1f}  "
              f"cos_mean={m['cosine_mean_pooled']:+.4f}  "
              f"norm={m['norm_hat']:.1f}")

    # Deltas:
    d_train_vs_rand = (
        aggregate["metrics_train_mode"]["cosine_per_position"]
        - aggregate["metrics_random_control"]["cosine_per_position"]
    )
    d_eval_vs_rand = (
        aggregate["metrics_eval_mode"]["cosine_per_position"]
        - aggregate["metrics_random_control"]["cosine_per_position"]
    )
    d_train_vs_eval = (
        aggregate["metrics_train_mode"]["cosine_per_position"]
        - aggregate["metrics_eval_mode"]["cosine_per_position"]
    )
    print(f"\n  Δ cosine_per_position:")
    print(f"    TRAIN  - random        = {d_train_vs_rand:+.4f}   (codec working in train mode?)")
    print(f"    EVAL   - random        = {d_eval_vs_rand:+.4f}   (codec working in eval mode?)")
    print(f"    TRAIN  - EVAL          = {d_train_vs_eval:+.4f}   (OOD penalty)")

    out_path.write_text(json.dumps({
        "config": vars(args),
        "n_samples": len(results),
        "n_valid": len(valid),
        "aggregate": aggregate,
        "samples": results,
    }, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

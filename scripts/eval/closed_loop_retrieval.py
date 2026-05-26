#!/usr/bin/env python
"""Closed-loop retrieval-margin eval.

Per the V3 eval refactor: the scalar closed-loop cosine on its own cannot
distinguish "real ground-truth grounding" from V2-style template collapse.
A model that produces the same average ``ĥ`` for every input gets decent
absolute cosine (it lands near the dataset mean) but has zero discriminative
power. The fix is to measure the **margin** between matched and cross pairs:

    matched_i  =  cos(h_i,   AR(AV(h_i)))
    cross_ij   =  cos(h_i,   AR(AV(h_j)))    for j != i
    margin     =  mean_i matched_i  -  mean_{i!=j} cross_ij
    retrieval@K = fraction of i where rank(matched) <= K in row i

If ``margin`` is near zero (or worse, negative), the AR↔AV pair is doing
mean prediction or template collapse — exactly the failure mode V2 had.
A healthy V3 should produce a clearly positive margin and a retrieval@1
well above the random baseline of ``1/N``.

Usage (LIBERO 4-suite V3)::

    PYTHONPATH=src python scripts/eval/closed_loop_retrieval.py \\
        --ckpt-dir         data/sft/libero_4suite_v3 \\
        --activations-root data/activations/libero_4suite_combined \\
        --labels-jsonl     data/labels/libero_4suite_combined/labels.jsonl \\
        --n-samples        256 \\
        --temperature      0.0 \\
        --out-json         data/sft/libero_4suite_v3/retrieval_margin.json

Output ``retrieval_margin.json``::

    {
      "n":                  256,
      "matched_cos_mean":   0.74,
      "cross_cos_mean":     0.41,
      "margin":             0.33,
      "retrieval_at_1":     0.42,
      "retrieval_at_5":     0.71,
      "by_position": { "image_patch": {...}, "last_text": {...}, "anchor": {...} },
      "config":  { ... reproducibility ... }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True,
                   help="SFT run dir with ar/ and av/ subdirs.")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--labels-jsonl", required=True)
    p.add_argument("--n-samples", type=int, default=256,
                   help="Total val items to evaluate, balanced across position types.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="AV decode temperature. 0.0 = greedy.")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-batch size for AV.generate / AR.forward. NOT N.")
    p.add_argument("--held-out-fraction", type=float, default=0.05,
                   help="Must match the value used at training time.")
    p.add_argument("--split-by", default="episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", required=True,
                   help="Output JSON path for the retrieval-margin summary.")
    p.add_argument("--out-jsonl", default=None,
                   help="Optional per-sample JSONL (gold, generated, matched_cos, top1_idx, ...).")
    p.add_argument("--spatial-diagnostics", action="store_true",
                   help="Stage-3 plan: when the AR head emits (B, N, H) "
                        "(head_type='spatial'), also write per-spatial-position "
                        "retrieval margin to the summary JSON under "
                        "'by_spatial_position'. No-op when AR is scalar. Use "
                        "this to verify the spatial decoder differentiates "
                        "across the grid; uniform margins per position would "
                        "indicate spatial collapse.")
    return p


def _summarize_rows(rows: list[dict], cross_offdiag_mean: float) -> dict:
    """Aggregate per-row matched cosine + ranks into a summary dict."""
    if not rows:
        return {"n": 0}
    matched = [r["matched_cos"] for r in rows]
    ranks = [r["matched_rank"] for r in rows]  # 1-based: 1 = best
    n = len(rows)
    return {
        "n": n,
        "matched_cos_mean": float(sum(matched) / n),
        "cross_cos_mean": float(cross_offdiag_mean),
        "margin": float(sum(matched) / n - cross_offdiag_mean),
        "retrieval_at_1": float(sum(1 for r in ranks if r <= 1) / n),
        "retrieval_at_5": float(sum(1 for r in ranks if r <= 5) / n),
        "retrieval_at_10": float(sum(1 for r in ranks if r <= 10) / n),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from nla.training.checkpoint import load_av_from_sft, load_ar_from_sft
    from nla.training.dataset import LabeledPositionDataset, collate_labeled_positions

    ckpt_dir = Path(args.ckpt_dir)
    print(f"Loading AV from {ckpt_dir}/av/ ...", flush=True)
    av = load_av_from_sft(ckpt_dir / "av", device=args.device, freeze=True)
    print(f"Loading AR from {ckpt_dir}/ar/ ...", flush=True)
    ar = load_ar_from_sft(ckpt_dir / "ar", device=args.device, freeze=True)
    alpha = float(ar.cfg.alpha)
    print(f"  α = {alpha:.4f}", flush=True)

    print(f"Loading held-out val set ...", flush=True)
    val_ds = LabeledPositionDataset(
        args.activations_root, args.labels_jsonl,
        seed=args.seed,
        held_out_fraction=args.held_out_fraction,
        held_out=True,
        split_by=args.split_by,
    )
    print(f"  -> {len(val_ds)} val rows", flush=True)

    # Stratify by position_type so the sample is balanced, then sample
    # ``--n-samples / n_strata`` rows per stratum (deterministic via seed).
    by_pos: dict[str, list[int]] = {}
    for i, entry in enumerate(val_ds.labels):
        by_pos.setdefault(entry.position_type, []).append(i)
    strata = sorted(by_pos.keys())
    per_stratum = max(1, args.n_samples // max(1, len(strata)))
    print(f"  -> per-position counts: {dict((k, len(v)) for k, v in by_pos.items())}", flush=True)
    print(f"  -> sampling {per_stratum} per position type", flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    chosen_indices: list[int] = []
    for ptype in strata:
        indices = by_pos[ptype]
        perm = torch.randperm(len(indices), generator=rng).tolist()
        for k in perm[:per_stratum]:
            chosen_indices.append(indices[k])

    if not chosen_indices:
        print("ERROR: no val rows after stratified sampling.", file=sys.stderr)
        return 2

    print(f"Materializing {len(chosen_indices)} val items + running AV / AR ...", flush=True)
    t0 = time.time()

    all_acts_list: list[torch.Tensor] = []
    all_gold: list[str] = []
    all_generated: list[str] = []
    all_position_types: list[str] = []

    BS = max(1, int(args.batch_size))
    for start in range(0, len(chosen_indices), BS):
        batch_idx = chosen_indices[start : start + BS]
        batch = collate_labeled_positions([val_ds[i] for i in batch_idx])
        acts = batch["activations"].to(args.device)
        gold = batch["description"]
        pos_types = batch["position_type"]

        do_sample = float(args.temperature) > 0.0
        with torch.no_grad():
            out = av.generate(
                activations=acts,
                position_types=pos_types,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=float(args.temperature) if do_sample else 1.0,
            )
        generated = [t.strip() for t in out["text"]]

        all_acts_list.append(acts.detach().to("cpu").float())
        all_gold.extend(gold)
        all_generated.extend(generated)
        all_position_types.extend(pos_types)

        elapsed = time.time() - t0
        done = min(len(chosen_indices), start + BS)
        print(f"  [{done}/{len(chosen_indices)}]  elapsed {elapsed:5.1f}s", flush=True)

    H_raw = torch.cat(all_acts_list, dim=0)  # (N, D) raw activations
    N, D = H_raw.shape
    assert N == len(chosen_indices)
    assert N == len(all_generated)
    print(f"  generated {N} AV captions in {time.time()-t0:5.1f}s. Now AR-encoding ...", flush=True)

    # AR-encode every generated caption (in batches) to produce ĥ_i.
    ar_out_scaled: list[torch.Tensor] = []
    ar_out_spatial: list[torch.Tensor] = []   # Stage-3: (B, M, D) when spatial head
    ar_is_spatial = False
    for start in range(0, N, BS):
        chunk = all_generated[start : start + BS]
        with torch.no_grad():
            pred_scaled = ar(chunk, device=args.device).float()
        if pred_scaled.dim() == 3:
            # Stage-3 spatial head: (B, M, D). Mean-pool over M for the
            # legacy 2D pipeline so all downstream metrics keep working,
            # but also retain the raw 3D tensor for per-spatial-position
            # diagnostics.
            ar_is_spatial = True
            ar_out_spatial.append(pred_scaled.detach().to("cpu"))
            pred_scaled = pred_scaled.mean(dim=1)
        ar_out_scaled.append(pred_scaled.detach().to("cpu"))
    H_hat_scaled = torch.cat(ar_out_scaled, dim=0)            # (N, D), scaled
    H_hat_raw = H_hat_scaled * alpha                          # back to unscaled space
    assert H_hat_raw.shape == (N, D)
    H_hat_spatial_raw: torch.Tensor | None = None
    if ar_is_spatial:
        H_hat_spatial_scaled = torch.cat(ar_out_spatial, dim=0)   # (N, M, D)
        H_hat_spatial_raw = H_hat_spatial_scaled * alpha
        print(
            f"  AR head is spatial; retained per-position tensor of shape "
            f"{tuple(H_hat_spatial_raw.shape)}",
            flush=True,
        )

    # Pairwise cosine. Cosine is scale-invariant so it doesn't matter whether
    # both sides are scaled or unscaled, but be consistent.
    H_n = torch.nn.functional.normalize(H_raw, dim=1)
    Hhat_n = torch.nn.functional.normalize(H_hat_raw, dim=1)
    sims = H_n @ Hhat_n.T                                     # (N, N) cosine matrix
    diag = sims.diag()                                        # matched cosine
    # Off-diagonal mean = (total - diag) / (N^2 - N).
    cross_offdiag_mean = ((sims.sum() - diag.sum()) / max(1, N * (N - 1))).item()

    # Per-row matched rank (1-based): how many ĥ_j have higher sim than ĥ_i?
    matched_rank = (sims > diag.unsqueeze(1)).sum(dim=1) + 1   # (N,) ints in [1..N]

    rows: list[dict] = []
    for i in range(N):
        rows.append({
            "idx": i,
            "position_type": all_position_types[i],
            "gold": all_gold[i],
            "generated": all_generated[i],
            "matched_cos": float(diag[i].item()),
            "matched_rank": int(matched_rank[i].item()),
        })

    overall = _summarize_rows(rows, cross_offdiag_mean)
    # Per-position breakdown — compute matched mean per position type, plus
    # rank-based retrieval. Cross mean is recomputed per stratum using only
    # off-diagonal entries among the same stratum's rows so it's apples-to-
    # apples ("could AR pick the right caption from N candidates *of this
    # position type*?").
    by_position: dict[str, dict] = {}
    for ptype in strata:
        ptype_mask = torch.tensor(
            [1.0 if p == ptype else 0.0 for p in all_position_types]
        ).bool()
        if not ptype_mask.any():
            continue
        idx = ptype_mask.nonzero(as_tuple=False).flatten().tolist()
        sub = sims[torch.tensor(idx)][:, torch.tensor(idx)]
        sub_diag = sub.diag()
        m = len(idx)
        sub_off = ((sub.sum() - sub_diag.sum()) / max(1, m * (m - 1))).item()
        sub_rank = (sub > sub_diag.unsqueeze(1)).sum(dim=1) + 1
        by_position[ptype] = {
            "n": m,
            "matched_cos_mean": float(sub_diag.mean().item()),
            "cross_cos_mean": float(sub_off),
            "margin": float(sub_diag.mean().item() - sub_off),
            "retrieval_at_1": float(((sub_rank <= 1).sum() / m).item()),
            "retrieval_at_5": float(((sub_rank <= min(5, m)).sum() / m).item()),
        }

    by_spatial_position: dict[str, dict] | None = None
    if args.spatial_diagnostics and H_hat_spatial_raw is not None:
        # For each spatial slot k, compute pairwise retrieval just over the
        # k-th predicted vector across rows. A healthy spatial decoder
        # produces position-specific margins that vary (left-of-frame vs.
        # right-of-frame ĥ disagree across rows); a collapsed decoder
        # produces uniform margins ≈ the mean-pooled value.
        n_spatial = int(H_hat_spatial_raw.shape[1])
        by_spatial_position = {}
        for k in range(n_spatial):
            slot_hat = H_hat_spatial_raw[:, k, :]                       # (N, D)
            slot_hat_n = torch.nn.functional.normalize(slot_hat, dim=1)
            slot_sims = H_n @ slot_hat_n.T                              # (N, N)
            slot_diag = slot_sims.diag()
            slot_off = (
                (slot_sims.sum() - slot_diag.sum()) / max(1, N * (N - 1))
            ).item()
            slot_rank = (slot_sims > slot_diag.unsqueeze(1)).sum(dim=1) + 1
            by_spatial_position[str(k)] = {
                "matched_cos_mean": float(slot_diag.mean().item()),
                "cross_cos_mean": float(slot_off),
                "margin": float(slot_diag.mean().item() - slot_off),
                "retrieval_at_1": float(((slot_rank <= 1).sum() / N).item()),
            }
        # Spatial collapse summary: std across positions of the per-slot
        # matched_cos_mean. A value near zero means the spatial head
        # outputs identical content per slot (collapse); higher std means
        # the head actually carries position-specific structure.
        slot_means = torch.tensor(
            [s["matched_cos_mean"] for s in by_spatial_position.values()]
        )
        slot_margins = torch.tensor(
            [s["margin"] for s in by_spatial_position.values()]
        )
        by_spatial_position["_collapse_diagnostic"] = {
            "n_positions": n_spatial,
            "matched_cos_mean_std_across_positions": float(slot_means.std().item()),
            "margin_std_across_positions": float(slot_margins.std().item()),
            "matched_cos_mean_min": float(slot_means.min().item()),
            "matched_cos_mean_max": float(slot_means.max().item()),
        }

    summary = {
        "n": N,
        **overall,
        "by_position": by_position,
        **({"by_spatial_position": by_spatial_position}
           if by_spatial_position is not None else {}),
        "config": {
            "ckpt_dir": str(ckpt_dir),
            "activations_root": args.activations_root,
            "labels_jsonl": args.labels_jsonl,
            "n_samples": args.n_samples,
            "per_stratum": per_stratum,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": BS,
            "held_out_fraction": args.held_out_fraction,
            "split_by": args.split_by,
            "seed": args.seed,
            "alpha": alpha,
            "elapsed_s": time.time() - t0,
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 78)
    print("RETRIEVAL MARGIN SUMMARY")
    print("=" * 78)
    print(f"  n                : {N}")
    print(f"  matched_cos_mean : {summary['matched_cos_mean']:+.4f}")
    print(f"  cross_cos_mean   : {summary['cross_cos_mean']:+.4f}")
    print(f"  margin           : {summary['margin']:+.4f}")
    print(f"  retrieval@1      : {summary['retrieval_at_1']:.4f}")
    print(f"  retrieval@5      : {summary['retrieval_at_5']:.4f}")
    print(f"  retrieval@10     : {summary['retrieval_at_10']:.4f}")
    for ptype, s in by_position.items():
        print(f"  [{ptype:14s}] n={s['n']:3d} matched={s['matched_cos_mean']:+.3f} "
              f"cross={s['cross_cos_mean']:+.3f} margin={s['margin']:+.3f} "
              f"r@1={s['retrieval_at_1']:.2f}")
    if by_spatial_position is not None:
        diag = by_spatial_position.get("_collapse_diagnostic") or {}
        print(
            f"  spatial: n_pos={diag.get('n_positions')} "
            f"matched_std={diag.get('matched_cos_mean_std_across_positions', 0):.4f} "
            f"margin_std={diag.get('margin_std_across_positions', 0):.4f} "
            f"(near-zero = spatial collapse)"
        )
    print(f"  -> {out_path}")

    if args.out_jsonl:
        jsonl_path = Path(args.out_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  -> per-sample rows: {jsonl_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

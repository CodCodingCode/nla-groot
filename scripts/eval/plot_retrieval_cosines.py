#!/usr/bin/env python
"""Plot matched vs cross cosine for closed-loop retrieval (h vs AR(AV(h)))."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="data/sft/libero_4suite_v5_base_qwen")
    p.add_argument("--activations-root", default="data/activations/libero_4suite_v4_combined")
    p.add_argument("--labels-jsonl", default="data/labels/libero_4suite_v5_combined/labels.jsonl")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir or ckpt / "post_sft_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft
    from nla.training.dataset import LabeledPositionDataset, collate_labeled_positions

    av = load_av_from_sft(ckpt / "av", device=args.device, freeze=True)
    ar = load_ar_from_sft(ckpt / "ar", device=args.device, freeze=True)
    alpha = float(ar.cfg.alpha)

    val_ds = LabeledPositionDataset(
        args.activations_root, args.labels_jsonl,
        seed=0, held_out_fraction=0.05, held_out=True, split_by="episode",
    )
    by_pos: dict[str, list[int]] = {}
    for i, entry in enumerate(val_ds.labels):
        by_pos.setdefault(entry.position_type, []).append(i)
    strata = sorted(by_pos.keys())
    per_stratum = max(1, args.n_samples // max(1, len(strata)))
    rng = torch.Generator().manual_seed(0)
    chosen: list[int] = []
    for ptype in strata:
        indices = by_pos[ptype]
        perm = torch.randperm(len(indices), generator=rng).tolist()
        chosen.extend(indices[perm[k]] for k in perm[:per_stratum])

    BS = 8
    acts_list, generated, pos_types = [], [], []
    for start in range(0, len(chosen), BS):
        batch = collate_labeled_positions([val_ds[i] for i in chosen[start : start + BS]])
        acts = batch["activations"].to(args.device)
        with torch.no_grad():
            out = av.generate(
                activations=acts, position_types=batch["position_type"],
                max_new_tokens=160, do_sample=False,
            )
        acts_list.append(acts.detach().cpu().float())
        generated.extend(t.strip() for t in out["text"])
        pos_types.extend(batch["position_type"])

    H_raw = torch.cat(acts_list, dim=0)
    N = H_raw.shape[0]
    h_hat_list = []
    for start in range(0, N, BS):
        with torch.no_grad():
            pred = ar(generated[start : start + BS], device=args.device).float()
        h_hat_list.append(pred.detach().cpu())
    H_hat = torch.cat(h_hat_list, dim=0) * alpha

    H_n = torch.nn.functional.normalize(H_raw, dim=1)
    Hhat_n = torch.nn.functional.normalize(H_hat, dim=1)
    sims = (H_n @ Hhat_n.T).numpy()

    matched = np.diag(sims)
    max_cross = np.array([np.max(np.concatenate([sims[i, :i], sims[i, i + 1 :]])) for i in range(N)])
    mean_cross = np.array([
        (sims[i].sum() - sims[i, i]) / max(1, N - 1) for i in range(N)
    ])
    pos_arr = np.array(pos_types)

    stats = {"n": N, "matched_mean": float(matched.mean()), "max_cross_mean": float(max_cross.mean()),
             "margin_max": float((matched - max_cross).mean())}
    for ptype in strata:
        m = pos_arr == ptype
        stats[ptype] = {
            "matched_mean": float(matched[m].mean()),
            "max_cross_mean": float(max_cross[m].mean()),
            "margin": float((matched[m] - max_cross[m]).mean()),
        }
    (out_dir / "retrieval_cosine_stats.json").write_text(json.dumps(stats, indent=2))

    colors = {"image_patch": "#e45756", "last_text": "#4c78a8", "anchor": "#72b7b2"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Closed-loop retrieval: cos(h, ĥ) — real h vs AR(AV(h)) reconstructions", fontsize=12)

    # (0,0) histogram matched vs max wrong
    ax = axes[0, 0]
    bins = np.linspace(0.2, 1.0, 35)
    ax.hist(matched, bins=bins, alpha=0.55, label=f"matched (mean={matched.mean():.3f})", color="#54a24b")
    ax.hist(max_cross, bins=bins, alpha=0.45, label=f"best wrong (mean={max_cross.mean():.3f})", color="#b279a2")
    ax.axvline(matched.mean(), color="#54a24b", ls="--", lw=1.5)
    ax.axvline(max_cross.mean(), color="#b279a2", ls="--", lw=1.5)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title("Matched vs hardest negative (all positions)")
    ax.legend(fontsize=8)

    # (0,1) by position — matched histograms
    ax = axes[0, 1]
    for ptype in strata:
        m = pos_arr == ptype
        ax.hist(matched[m], bins=bins, alpha=0.5, label=f"{ptype} matched μ={matched[m].mean():.3f}",
                color=colors.get(ptype, "#888"))
    ax.set_xlabel("matched cosine cos(hᵢ, ĥᵢ)")
    ax.set_title("Matched cosine by position type")
    ax.legend(fontsize=8)

    # (1,0) scatter matched vs max_cross
    ax = axes[1, 0]
    for ptype in strata:
        m = pos_arr == ptype
        ax.scatter(matched[m], max_cross[m], alpha=0.45, s=22, label=ptype, color=colors.get(ptype, "#888"))
    lim = (0.25, 1.0)
    ax.plot(lim, lim, "k--", lw=1, alpha=0.4)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("matched cos(hᵢ, ĥᵢ)")
    ax.set_ylabel("best wrong cos(hᵢ, ĥⱼ≠ᵢ)")
    ax.set_title("Per-sample: correct vs best impostor")
    ax.legend(fontsize=8)
    above = (matched > max_cross).mean()
    ax.text(0.03, 0.97, f"correct wins: {above:.1%}", transform=ax.transAxes, va="top", fontsize=9)

    # (1,1) CDF
    ax = axes[1, 1]
    for ptype in strata:
        m = pos_arr == ptype
        xs = np.sort(matched[m])
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=f"{ptype} matched", color=colors.get(ptype, "#888"))
    ax.set_xlabel("matched cosine")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of matched reconstruction quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png = out_dir / "retrieval_cosine_plot.png"
    pdf = out_dir / "retrieval_cosine_plot.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    print(f"Wrote {png}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Diagnostic 3 — intent-separation / collapse (the cosine viz, done right).

On the same 100 samples, asks at which stage the per-task distinctions die:
caption text  ->  reconstructed vector ĥ  vs  real activation h.

All cosines are computed in MEAN-CENTERED space (subtract the per-dim mean),
because raw activations share one dominant direction that inflates every
cosine toward 1 and hides the structure (this is the same effect that makes
FVE negative while cosine looks high).

For each level we report within-task vs between-task mean cosine and the
separation = within - between (higher => task identity is preserved):
  - real h (mean-pooled grid)  : does the policy itself distinguish tasks?
  - codec ĥ (AV->AR, pooled)   : does the codec preserve those distinctions?
  - caption text (token Jaccard): are AV's captions task-specific?

Decision rule:
  - real-h separates, ĥ collapses          -> AR squashes distinct intents (improve AR)
  - captions collapse (low text separation)-> AV not intent-specific (improve AV)
  - real-h itself collapses                -> upstream; steering may be hopeless

Outputs: a JSON summary + (if matplotlib present) centered-cosine heatmaps and
2D PCA scatters for ĥ and real-h, colored by task.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/eval/diag/m3_intent_separation.py \
      --sft-dir data/sft/v9_combined_12k --device cuda \
      --out-dir data/eval/diag/m3_v9_combined_12k
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nla.training.checkpoint import load_ar_from_sft, load_av_from_sft
from nla.training.sim_reward import encode_texts_with_ar
from sample_set import ROOT, load_samples  # script-dir is on sys.path when run as a file


def _centered_cosine_matrix(X: np.ndarray) -> np.ndarray:
    """X [N,H] -> [N,N] cosine on mean-centered, L2-normalized rows."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = np.linalg.norm(Xc, axis=1, keepdims=True)
    Xn = Xc / np.clip(n, 1e-8, None)
    return Xn @ Xn.T


def _within_between(cos: np.ndarray, labels: list[str]) -> dict:
    lab = np.asarray(labels)
    N = len(lab)
    iu, ju = np.triu_indices(N, k=1)
    same = lab[iu] == lab[ju]
    vals = cos[iu, ju]
    within = float(vals[same].mean()) if same.any() else float("nan")
    between = float(vals[~same].mean()) if (~same).any() else float("nan")
    return {"within": within, "between": between, "separation": within - between,
            "n_within_pairs": int(same.sum()), "n_between_pairs": int((~same).sum())}


def _token_jaccard_separation(texts: list[str], labels: list[str]) -> dict:
    toks = [set(t.lower().split()) for t in texts]
    N = len(toks)
    cos = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            a, b = toks[i], toks[j]
            u = len(a | b)
            jac = (len(a & b) / u) if u else 0.0
            cos[i, j] = cos[j, i] = jac
    return _within_between(cos, labels)


def _maybe_plots(out: Path, name: str, X: np.ndarray, labels: list[str], cos: np.ndarray):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[m3] matplotlib unavailable ({e}); skipping plots")
        return
    order = np.argsort(labels)
    # heatmap (ordered by task)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cos[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{name}: centered cosine (ordered by task)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out / f"{name}_cosine_heatmap.png", dpi=120); plt.close(fig)
    # 2D PCA
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T
    uniq = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in uniq:
        m = np.asarray(labels) == t
        ax.scatter(pcs[m, 0], pcs[m, 1], s=22, label=t[:24])
    ax.set_title(f"{name}: PCA(2) of mean-centered vectors")
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout(); fig.savefig(out / f"{name}_pca2.png", dpi=120); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sft_dir = Path(args.sft_dir)
    if not sft_dir.is_absolute():
        sft_dir = ROOT / sft_dir
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    samples = load_samples(limit=args.limit, load_tensors=True)
    print(f"[m3] {len(samples)} samples")

    ar = load_ar_from_sft(sft_dir / "ar", device=args.device, freeze=True)
    av = load_av_from_sft(sft_dir / "av", device=args.device, freeze=True)

    labels = [s.target_task for s in samples]
    src_labels = [s.source_task or "?" for s in samples]
    real_h = np.stack([s.h_grid.mean(axis=0) for s in samples])  # [N,H] pooled real
    hhat, captions = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        h_pos = torch.from_numpy(s.h_pos).unsqueeze(0).to(args.device)
        with torch.no_grad():
            cap = av.generate(h_pos, [s.position_type],
                              max_new_tokens=args.max_new_tokens,
                              temperature=0.0, do_sample=False,
                              target_intent_texts=[s.target_intent])["text"][0]
        captions.append(cap)
        v = encode_texts_with_ar(ar, [cap], device=args.device).detach().float().cpu().numpy()
        v = v.reshape(-1, v.shape[-1]).mean(axis=0)  # pool grid -> [H]
        hhat.append(v)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {time.time()-t0:.0f}s", flush=True)
    hhat = np.stack(hhat)

    cos_real = _centered_cosine_matrix(real_h)
    cos_hhat = _centered_cosine_matrix(hhat)
    summary = {
        "sft_dir": str(sft_dir),
        "n_samples": len(samples),
        "n_tasks": len(set(labels)),
        # real h is the SOURCE scene's activation -> group by source_task (its real
        # label). Grouping it by target_task is apples-to-oranges (the target is a
        # counterfactual the source activation has no reason to encode).
        "real_h_separation_by_source": _within_between(cos_real, src_labels),
        "real_h_separation_by_target": _within_between(cos_real, labels),
        # codec hhat encodes the TARGET intent -> group by target_task.
        "codec_hhat_separation": _within_between(cos_hhat, labels),
        "caption_text_separation": _token_jaccard_separation(captions, labels),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _maybe_plots(out, "real_h", real_h, labels, cos_real)
    _maybe_plots(out, "codec_hhat", hhat, labels, cos_hhat)

    print("\n=== Diagnostic 3: intent separation (within - between, centered cosine) ===")
    for k in ("caption_text_separation", "codec_hhat_separation",
              "real_h_separation_by_source", "real_h_separation_by_target"):
        d = summary[k]
        print(f"  {k:30s} within={d['within']:+.3f} between={d['between']:+.3f} "
              f"sep={d['separation']:+.3f}")
    rs = summary["real_h_separation_by_source"]["separation"]
    hs = summary["codec_hhat_separation"]["separation"]
    cs = summary["caption_text_separation"]["separation"]
    print("\ninterpretation hint:")
    if cs < 0.05:
        print("  -> captions barely task-specific: suspect AV")
    if hs > 0.1:
        print("  -> codec ĥ separates by target intent: AR is NOT collapsing intents")
    if rs > 0.05 and hs < rs * 0.5:
        print("  -> real h (by source) separates but ĥ collapses: suspect AR")
    if rs < 0.05:
        print("  -> real h (by source) barely separates: pooled activation lacks "
              "task structure (or upstream/α)")
    print(f"\nWrote {out}/summary.json (+ heatmaps/pca if matplotlib present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

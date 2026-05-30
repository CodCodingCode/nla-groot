"""Render the charts embedded in docs/sft_plan/v9_implementation.md.

All quantitative charts are drawn from the real v8_full_6400 metrics.jsonl —
the run whose "high cosine, deeply negative FVE" pathology motivates v9.
Outputs PNGs to docs/sft_plan/assets/v9/.

    PYTHONPATH=src .venv/bin/python scripts/eval/plot_v9_implementation_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / "data/sft/v8_full_6400/metrics.jsonl"
OUT = ROOT / "docs/sft_plan/assets/v9"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 203.977  # codec scale (p75 activation norm) — see v7_overview §1
INK = "#1b2330"
GRID = "#d6dbe3"
BLUE = "#2563eb"
RED = "#e11d48"
GREEN = "#059669"
AMBER = "#d97706"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def load_val(path: Path):
    rows = [json.loads(l) for l in path.open() if l.strip()]
    val = [r for r in rows if "cosine" in r and "fve" in r]
    val.sort(key=lambda r: r["step"])
    return val


def g(val, key):
    return np.array([r[key] for r in val], dtype=float)


def chart_pathology(val):
    """Dual-axis: cosine flat ~0.52 while FVE climbs but stays < 0."""
    steps = g(val, "step")
    fve = g(val, "closed_greedy/fve")
    cos = g(val, "closed_greedy/cosine")

    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.axhspan(-650, 0, color=RED, alpha=0.045, zorder=0)
    l1, = ax1.plot(steps, fve, color=RED, lw=2.4, marker="o", ms=4,
                   label="closed-loop FVE (want > 0)")
    ax1.axhline(0, color=RED, ls="--", lw=1.2, alpha=0.7)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("closed-loop FVE", color=RED)
    ax1.tick_params(axis="y", labelcolor=RED)
    ax1.set_ylim(-650, 60)

    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(steps, cos, color=BLUE, lw=2.4, marker="s", ms=4,
                   label="closed-loop cosine")
    ax2.set_ylabel("closed-loop cosine", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.set_ylim(0.30, 0.75)

    ax2.annotate("direction learned early,\nthen plateaus (~0.55)",
                 xy=(steps[-4], cos[-4]), xytext=(3050, 0.435), color=BLUE, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax1.annotate("magnitude never\ncalibrates — FVE\nstuck at −74",
                 xy=(steps[-1], fve[-1]), xytext=(4550, -320), color=RED, fontsize=9.5,
                 ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax1.set_title("v8 pathology: high cosine, deeply negative FVE\n"
                  "(the gap v9's decomposed loss targets)", fontweight="bold")
    ax1.legend(handles=[l1, l2], loc="upper left", framealpha=0.95)
    fig.tight_layout()
    p = OUT / "v8_pathology_cosine_vs_fve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_mse(val):
    """Reconstruction MSE decay vs the v9 target band (log scale)."""
    steps = g(val, "step")
    vmse = g(val, "mse")
    cmse = g(val, "closed_greedy/mse")
    target = 0.005 * ALPHA**2  # v9 success bound: val MSE / alpha^2 < 0.005

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(steps, vmse, color=AMBER, lw=2.2, marker="o", ms=4, label="val MSE (teacher-forced)")
    ax.plot(steps, cmse, color=RED, lw=2.2, marker="s", ms=4, label="closed-loop MSE")
    ax.axhline(target, color=GREEN, ls="--", lw=1.6,
               label=f"v9 target  ≈ {target:.0f}  (MSE/α² < 0.005)")
    ax.fill_between(steps, target, 0, color=GREEN, alpha=0.06)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("reconstruction MSE (α-scaled units, log)")
    ax.annotate("v8 ends at ~369 — ~1.8× above\ntarget, and flattening",
                xy=(steps[-1], cmse[-1]), xytext=(2600, 250), color=RED, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_title("v8 reconstruction MSE plateaus above the v9 bar", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    p = OUT / "v8_mse_vs_target.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_position(val):
    """Per-position closed-loop FVE: image_patch is the hard axis."""
    steps = g(val, "step")
    ip = g(val, "closed_greedy/fve/position=image_patch")
    lt = g(val, "closed_greedy/fve/position=last_text")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(0, color=INK, ls="--", lw=1.0, alpha=0.6)
    ax.plot(steps, lt, color=BLUE, lw=2.2, marker="o", ms=4, label="last_text (intent token)")
    ax.plot(steps, ip, color=RED, lw=2.2, marker="s", ms=4, label="image_patch (visual evidence)")
    ax.fill_between(steps, ip, lt, color=AMBER, alpha=0.10)
    ax.set_xlabel("training step")
    ax.set_ylabel("closed-loop FVE by position")
    ax.annotate("image_patch lags — the\nposition that carries the\nvisual grounding the\npolicy actually consumes",
                xy=(steps[-4], ip[-4]), xytext=(2500, -300), color=RED, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_title("Per-position FVE: image_patch is the bottleneck", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    p = OUT / "v8_fve_by_position.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_decomposition():
    """Conceptual: how MSE splits into magnitude + direction, and which v9 frees."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    # Illustrate two unit-ish vectors with a magnitude mismatch.
    t = np.array([1.0, 0.0])
    ang = np.deg2rad(58)  # cosine ~0.53, matching v8
    p = 0.62 * np.array([np.cos(ang), np.sin(ang)])  # shorter + off-angle
    for v, c, lab, lw in [(t, GREEN, "target  h", 3), (p, RED, "recon  ĥ", 3)]:
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))
        ax.text(v[0] * 1.06, v[1] * 1.06, lab, color=c, fontsize=11, fontweight="bold")
    # magnitude gap along the target direction
    ax.plot([p[0], (p[0])], [p[1], p[1]], alpha=0)
    proj = (p @ t) * t
    ax.annotate("", xy=t, xytext=proj,
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.6, ls="--"))
    ax.text(0.78, -0.07, "magnitude error\n(∥ĥ∥ − ∥h∥)²", color=AMBER, fontsize=9.5)
    ax.annotate("direction error\n(1 − cos)", xy=(0.32, 0.30), color=BLUE, fontsize=9.5)
    ax.add_patch(plt.matplotlib.patches.Arc((0, 0), 0.7, 0.7, angle=0, theta1=0,
                                            theta2=58, color=BLUE, lw=1.6))
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.25, 0.95)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("v9 decomposed loss: split MSE into the two errors,\n"
                 "weight magnitude independently", fontweight="bold")
    ax.text(0.02, 0.86,
            r"$\|\hat h-h\|^2 = (\|\hat h\|-\|h\|)^2 + 2\|\hat h\|\|h\|\,(1-\cos)$",
            fontsize=11)
    fig.tight_layout()
    pth = OUT / "loss_decomposition.png"
    fig.savefig(pth, bbox_inches="tight")
    plt.close(fig)
    return pth


def main():
    val = load_val(V8)
    print(f"loaded {len(val)} v8 val rows")
    for fn in (chart_pathology, chart_mse, chart_position):
        print("wrote", fn(val).relative_to(ROOT))
    print("wrote", chart_decomposition().relative_to(ROOT))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Programmatically build a W&B workspace with explicit chart panels for
every training metric we care about, so the run page never shows
"There's no data for the selected runs" when the default panel filter
is mismatched against our metric names.

Defaults assume our standard project (nla-groot) and the SFT metric
schema (train/loss, train/ce, train/ar_mse, train/ar_nce,
train/action_consistency_loss, train/gpu_memory_gb,
train/gpu_memory_reserved_gb, val/*, val/closed_greedy/*). Adapt the
list of LinePlot(y=...) entries if a new training script logs different
keys.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/build_wandb_workspace.py \\
        --project nla-groot \\
        --entity nathanyan2008p-personal \\
        --name "v9 Headline"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k, v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entity", default="nathanyan2008p-personal")
    ap.add_argument("--project", default="nla-groot")
    ap.add_argument("--name", default="SFT Headline")
    args = ap.parse_args()

    _load_dotenv()
    try:
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        print("FATAL: pip install wandb-workspaces", file=sys.stderr)
        return 2

    sections = [
        ws.Section(
            name="🎯 Headline (codec quality)",
            panels=[
                wr.LinePlot(
                    title="closed_greedy/cosine (higher = better)",
                    y=["val/closed_greedy/cosine"], smoothing_factor=0.0,
                ),
                wr.LinePlot(
                    title="closed_greedy/mse (lower = better)",
                    y=["val/closed_greedy/mse"], smoothing_factor=0.0,
                ),
                wr.LinePlot(
                    title="closed_greedy/fve (higher = better)",
                    y=["val/closed_greedy/fve"], smoothing_factor=0.0,
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="📉 Train losses",
            panels=[
                wr.LinePlot(title="train/loss",   y=["train/loss"],   smoothing_factor=0.3),
                wr.LinePlot(title="train/ce",     y=["train/ce"],     smoothing_factor=0.3),
                wr.LinePlot(title="train/ar_mse", y=["train/ar_mse"], smoothing_factor=0.3),
                wr.LinePlot(title="train/ar_nce", y=["train/ar_nce"], smoothing_factor=0.3),
                wr.LinePlot(
                    title="train/action_consistency_loss",
                    y=["train/action_consistency_loss"], smoothing_factor=0.3,
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="🧪 Val metrics (every 400 steps)",
            panels=[
                wr.LinePlot(title="val/cosine", y=["val/cosine"]),
                wr.LinePlot(title="val/mse",    y=["val/mse"]),
                wr.LinePlot(title="val/ce",     y=["val/ce"]),
                wr.LinePlot(title="val/fve",    y=["val/fve"]),
            ],
            is_open=True,
        ),
        ws.Section(
            name="💾 GPU memory (OOM watch)",
            panels=[
                wr.LinePlot(title="train/gpu_memory_gb (allocated)",
                            y=["train/gpu_memory_gb"]),
                wr.LinePlot(title="train/gpu_memory_reserved_gb",
                            y=["train/gpu_memory_reserved_gb"]),
            ],
            is_open=False,
        ),
    ]

    workspace = ws.Workspace(
        name=args.name,
        entity=args.entity,
        project=args.project,
        sections=sections,
    )
    saved = workspace.save()
    print(f"OK created workspace: {saved.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build a W&B workspace with explicit chart panels for the GRPO run schema.

Mirrors scripts/eval/build_wandb_workspace.py (SFT) but with the GRPO
metric keys defined in src/nla/training/grpo.py._WANDB_TRAIN_KEYS.

Every LinePlot pins ``x="_step"`` -- W&B's auto-incremented counter that
``wandb.log(payload, step=N)`` writes to. The SFT script already learned
this lesson the hard way (a stale step_metric kwarg silently rendered
"no data" on every panel even with data logged); doing it explicitly
here is the belt-and-braces fix.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/build_wandb_workspace_grpo.py \\
        --project nla-groot \\
        --entity nathanyan2008p-personal \\
        --name "GRPO Headline"
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
    ap.add_argument("--name", default="GRPO Headline")
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
            name="🎯 Headline reward (is GRPO learning to steer?)",
            panels=[
                wr.LinePlot(
                    title="train/sim_reward_mean ↑ (mean r_sim per rollout)",
                    y=["train/sim_reward_mean"], x="_step",
                    smoothing_factor=0.5,
                ),
                wr.LinePlot(
                    title="train/sim_reward_best ↑ (best-of-K per group)",
                    y=["train/sim_reward_best"], x="_step",
                    smoothing_factor=0.5,
                ),
                wr.LinePlot(
                    title="train/reward_mean ↑ (blended reward, recon+sim)",
                    y=["train/reward_mean"], x="_step",
                    smoothing_factor=0.5,
                ),
                wr.LinePlot(
                    title="train/sim_predicate_pos_frac ↑ (fraction of rollouts with predicate hit)",
                    y=["train/sim_predicate_pos_frac"], x="_step",
                    smoothing_factor=0.5,
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="📉 Train losses",
            panels=[
                wr.LinePlot(
                    title="train/loss ↓ (total)",
                    y=["train/loss"], x="_step", smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/pg_loss (policy gradient -- can be negative)",
                    y=["train/pg_loss"], x="_step", smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/kl_loss ↓ (KL anchor; should stay small)",
                    y=["train/kl_loss"], x="_step", smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/ar_mse ↓ (reconstruction error if AR cotrain)",
                    y=["train/ar_mse"], x="_step", smoothing_factor=0.3,
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="🧪 Codec val metrics (closed-loop, every eval cadence)",
            panels=[
                wr.LinePlot(
                    title="val/closed_greedy/cosine ↑",
                    y=["val/closed_greedy/cosine"], x="_step",
                ),
                wr.LinePlot(
                    title="val/closed_greedy/mse ↓",
                    y=["val/closed_greedy/mse"], x="_step",
                ),
                wr.LinePlot(
                    title="val/closed_greedy/fve ↑",
                    y=["val/closed_greedy/fve"], x="_step",
                ),
                wr.LinePlot(
                    title="val/cosine ↑ (teacher-forced)",
                    y=["val/cosine"], x="_step",
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="🩺 Reward diagnostics + rollout health",
            panels=[
                wr.LinePlot(
                    title="train/sim_active_frac (fraction of rows that produced a sim reward)",
                    y=["train/sim_active_frac"], x="_step",
                    smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/sim_success_any_frac ↑ (any-step success rate)",
                    y=["train/sim_success_any_frac"], x="_step",
                    smoothing_factor=0.5,
                ),
                wr.LinePlot(
                    title="train/advantage_abs_mean (group-relative advantage scale)",
                    y=["train/advantage_abs_mean"], x="_step",
                    smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/kl_token_mean (per-token KL vs SFT ref)",
                    y=["train/kl_token_mean"], x="_step",
                    smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/gen_len_mean (mean rollout token length)",
                    y=["train/gen_len_mean"], x="_step",
                    smoothing_factor=0.3,
                ),
                wr.LinePlot(
                    title="train/dynamic_sampling_drop_frac (rows dropped for zero-advantage group)",
                    y=["train/dynamic_sampling_drop_frac"], x="_step",
                    smoothing_factor=0.3,
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="💾 GPU memory (OOM watch)",
            panels=[
                wr.LinePlot(
                    title="train/gpu_memory_gb (allocated)",
                    y=["train/gpu_memory_gb"], x="_step",
                ),
                wr.LinePlot(
                    title="train/gpu_memory_reserved_gb",
                    y=["train/gpu_memory_reserved_gb"], x="_step",
                ),
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

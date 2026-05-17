# Reproducibility bundle for the nla-groot workshop paper

This directory is the appendix-grade reproducibility package referenced
from `paper/main.tex`, App. A.

## Contents

| File | Purpose |
|------|---------|
| `requirements.txt`        | Frozen Python 3.10 environment used for the reported runs (`pip freeze` of the project `.venv`). Includes editable install of `Isaac-GR00T` pinned to a specific upstream commit. |
| `canonical_commands.sh`   | The exact entry points (extraction, labeling, SFT, GRPO, post-SFT eval bundle, LIBERO steerability eval, scorecard) used to produce the numbers in the paper. |

## External assets (not in this folder)

| Asset | How to obtain |
|-------|---------------|
| `nvidia/GR00T-N1.7-3B`                       | Hugging Face (open). |
| `nvidia/GR00T-N1.7-LIBERO/libero_{suite}`    | Hugging Face. We use the `libero_goal` variant for closed-loop sim. |
| `Qwen/Qwen3-4B-Instruct-2507`                | Hugging Face (base LM for AV / AR). |
| LIBERO sim assets + `libero_uv` venv         | `third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv` (separate venv, MuJoCo + osmesa CPU rendering). |
| OpenAI multimodal model (for labeling/judge) | `OPENAI_API_KEY` env var; model name configurable via `OPENAI_LABELING_MODEL` (default `gpt-5-mini`). |

## Hardware

Reported runs used a single NVIDIA H100 (80 GB) for SFT and a separate
CPU-only LIBERO sim process; rollouts are roughly 1 minute / episode
under `MUJOCO_GL=osmesa`.

## Seeds

- SFT: `--seed 0` (also persisted in the per-run `config.json`).
- Steerability rollouts: `seeds: [0, 1, 2]` per the eval YAML.
- Steerability AB / sim arms: pinned in
  `scripts/eval/steerability_v1_vs_v3.yaml`.

#!/usr/bin/env bash
# Canonical commands used to produce the numbers reported in the paper.
# These are documentation; do not source this file blindly.
#
# Environment assumptions:
#   - Repo at $REPO=/home/<user>/nla-groot
#   - $REPO/.venv : main Python 3.10 venv (see paper/repro/requirements.txt)
#   - $REPO/third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv :
#     LIBERO simulator venv (separate; uses MUJOCO_GL=osmesa).
#   - $REPO/checkpoints/GR00T-N1.7-LIBERO/libero_goal : the official
#     LIBERO Goal posttrained checkpoint (downloaded once from HF).
#   - OPENAI_API_KEY exported for labeling / judge scripts.

set -euo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel)}
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH=src

# -------------------------------------------------------------------
# 1) Extract GR00T backbone activations for a LIBERO suite.
# -------------------------------------------------------------------
python scripts/extraction/run_extract.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
  --embodiment-tag LIBERO_PANDA \
  --dataset-id libero_goal \
  --out-root data/activations/libero_goal_pilot \
  --steps-per-traj 32 --step-stride 4

# -------------------------------------------------------------------
# 2) Label activations with the multimodal teacher.
# -------------------------------------------------------------------
python scripts/labeling/run_label.py \
  --activations-root data/activations/libero_goal_pilot \
  --out-jsonl        data/labels/libero_goal_pilot/labels.jsonl \
  --frames-cache     data/labels/libero_goal_pilot/frames_cache

# -------------------------------------------------------------------
# 3) Joint SFT on the combined LIBERO 4-suite corpus (V3 recipe).
#    Reproduces data/sft/libero_4suite_v3/v3_scorecard.json.
# -------------------------------------------------------------------
python scripts/training/run_sft.py \
  --activations-root data/activations/libero_4suite_combined \
  --labels-jsonl     data/labels/libero_4suite_combined/labels.jsonl \
  --output-dir       data/sft/libero_4suite_v3 \
  --stats-json       data/activations/libero_4suite_combined/stats.json \
  --batch-size 4 --grad-accum-steps 1 \
  --learning-rate 1e-4 --warmup-steps 500 --total-steps 15000 \
  --ar-contrastive-weight 0.5 --nce-temperature 0.1 \
  --ar-nce-hard-negatives-per-anchor 4 \
  --ar-av-mix-max 0.4 --ar-av-mix-warmup-frac 0.3 \
  --balance-position-mix --split-by episode --held-out-fraction 0.05 \
  --eval-closed-loop --closed-loop-temperatures 0.0 0.7 \
  --eval-every 500 --save-every 2500 --log-every 5 \
  --seed 0

# -------------------------------------------------------------------
# 4) Optional GRPO phase (frozen AR as reward).
# -------------------------------------------------------------------
python scripts/training/run_grpo.py \
  --activations-root data/activations/libero_4suite_combined \
  --sft-dir          data/sft/libero_4suite_v3 \
  --output-dir       data/grpo/libero_4suite_v3_grpo \
  --seed 0

# -------------------------------------------------------------------
# 5) Post-SFT eval bundle (retrieval, AV samples, LLM judge, scorecard).
# -------------------------------------------------------------------
bash scripts/eval/run_post_sft_evals.sh \
  --sft-dir data/sft/libero_4suite_v3

# -------------------------------------------------------------------
# 6) LIBERO Goal closed-loop steerability eval (v1 vs v3, 3 seeds).
#    Drives scripts/eval/steerability_eval.py with the public YAML.
# -------------------------------------------------------------------
python scripts/eval/steerability_eval.py \
  --config scripts/eval/steerability_v1_vs_v3.yaml

# -------------------------------------------------------------------
# 7) Stitch comparison video and patch the scorecard.
# -------------------------------------------------------------------
bash scripts/eval/make_v1_vs_v3_grid.sh
python scripts/eval/build_v3_scorecard.py \
  --sft-dir data/sft/libero_4suite_v3 \
  --sim-ab-json data/sft/libero_4suite_v3/sim_ab.json

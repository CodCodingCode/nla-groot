#!/usr/bin/env bash
# V5 sim-GRPO pilot: 5h wall-clock (or override WALL_HOURS), 18 workers, K=4.
#
# Usage:
#   WALL_HOURS=5 bash scripts/training/launch_v5_sim_grpo_pilot.sh
#   tail -f data/grpo/libero_4suite_v5_sim_grpo_pilot/grpo.log
#
# Env:
#   WALL_HOURS       default 5
#   MAX_STEPS        cap passed to run_grpo (default 2000; wall usually hits first)
#   STEER_PORT       default 5556
#   SFT_DIR, ACT_ROOT, GRPO_OUT

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
ACT_ROOT="${ACT_ROOT:-data/activations/libero_4suite_v4_combined}"
GRPO_OUT="${GRPO_OUT:-data/grpo/libero_4suite_v5_sim_grpo_pilot}"
STEER_PORT="${STEER_PORT:-5556}"
WALL_HOURS="${WALL_HOURS:-5}"
MAX_STEPS="${MAX_STEPS:-2000}"
WALL_SECS=$(( WALL_HOURS * 3600 ))
PYTHON="${PYTHON_BIN:-.venv/bin/python}"
LIBERO_PY="${LIBERO_PY:-third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
MODEL_PATH="${GROOT_MODEL_PATH:-checkpoints/GR00T-N1.7-LIBERO/libero_goal}"
SIM_CACHE="${SIM_CACHE:-data/grpo/libero_4suite_v5_sim_grpo_pilot/sim_reward_cache.jsonl}"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

echo "[v5-sim-pilot] stopping prior V5 GRPO jobs (steer server untouched)"
pkill -f 'run_grpo.py.*libero_4suite_v5' 2>/dev/null || true
sleep 3

STEER_LOG_DIR="${SFT_DIR}/steer_server_logs"
STEER_PID_FILE="${STEER_LOG_DIR}/server.pid"
if command -v nc >/dev/null 2>&1 && nc -z localhost "${STEER_PORT}" 2>/dev/null \
   && [[ -f "${STEER_PID_FILE}" ]] && kill -0 "$(cat "${STEER_PID_FILE}")" 2>/dev/null; then
  echo "[v5-sim-pilot] steer server OK pid=$(cat "${STEER_PID_FILE}") port=${STEER_PORT}"
else
  echo "[v5-sim-pilot] launching steer server..."
  bash scripts/eval/launch_steer_server.sh \
    --sft-dir "$SFT_DIR" --port "$STEER_PORT" --log-dir "${STEER_LOG_DIR}" \
    --ready-timeout 300 -- \
    --model-path "$MODEL_PATH" --embodiment-tag LIBERO_PANDA \
    --steer-text-file scripts/eval/default_steer_boot.txt
fi

mkdir -p "$GRPO_OUT" logs
echo "[v5-sim-pilot] starting sim GRPO: wall=${WALL_HOURS}h max_steps=${MAX_STEPS} -> ${GRPO_OUT}"

nohup timeout --signal=TERM "${WALL_SECS}" \
  "$PYTHON" scripts/training/run_grpo.py \
    --sft-dir "$SFT_DIR" \
    --activations-root "$ACT_ROOT" \
    --output-dir "$GRPO_OUT" \
    --batch-size 2 \
    --rollouts-per-activation 4 \
    --total-steps "$MAX_STEPS" \
    --eval-every 50 \
    --save-every 100 \
    --log-every 10 \
    --sim-reward-weight 0.5 \
    --sim-counterfactual-pairs-path data/grpo/libero_goal_counterfactual_pairs.jsonl \
    --sim-counterfactual-pairs-path-extra data/grpo/libero_spatial_counterfactual_pairs.jsonl \
    --sim-counterfactual-pairs-path-extra data/grpo/libero_object_counterfactual_pairs.jsonl \
    --sim-counterfactual-pairs-path-extra data/grpo/libero_10_counterfactual_pairs.jsonl \
    --cf-eligible-ids-path data/grpo/libero_4suite_cf_eligible.json \
    --sim-policy-host localhost \
    --sim-policy-port "$STEER_PORT" \
    --sim-n-workers 18 \
    --sim-max-steps 100 \
    --sim-rollout-python "$LIBERO_PY" \
    --sim-cache-path "$SIM_CACHE" \
    --dynamic-sampling \
    --use-ppo-clip \
    --disable-kl-anchor \
    --rollout-temperature-high 1.6 \
    --beta 0.0 \
    --seed 0 \
  >> "${GRPO_OUT}/grpo.log" 2>&1 &

echo $! > logs/v5_sim_grpo_pilot.pid
echo "[v5-sim-pilot] GRPO pid=$(cat logs/v5_sim_grpo_pilot.pid) wall=${WALL_HOURS}h"
echo "[v5-sim-pilot] monitor: tail -f ${GRPO_OUT}/grpo.log ${GRPO_OUT}/metrics.jsonl"

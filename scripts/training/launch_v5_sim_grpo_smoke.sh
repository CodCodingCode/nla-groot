#!/usr/bin/env bash
# V5 sim-GRPO smoke: steer server + short GRPO with 18 LIBERO workers.
#
# Usage (from repo root):
#   bash scripts/training/launch_v5_sim_grpo_smoke.sh
#
# Logs:
#   data/grpo/libero_4suite_v5_sim_grpo_smoke/grpo.log
#   data/sft/libero_4suite_v5_base_qwen/steer_server_logs/

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
ACT_ROOT="${ACT_ROOT:-data/activations/libero_4suite_v4_combined}"
GRPO_OUT="${GRPO_OUT:-data/grpo/libero_4suite_v5_sim_grpo_smoke}"
STEER_PORT="${STEER_PORT:-5556}"
PYTHON="${PYTHON_BIN:-.venv/bin/python}"
LIBERO_PY="${LIBERO_PY:-third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
MODEL_PATH="${GROOT_MODEL_PATH:-checkpoints/GR00T-N1.7-LIBERO/libero_goal}"

export PYTHONPATH=src
if [[ -f .env ]]; then set -a; source .env; set +a; fi

echo "[v5-sim-smoke] stopping prior GRPO / steer on port ${STEER_PORT}"
pkill -f 'run_grpo.py.*libero_4suite_v5' 2>/dev/null || true
pkill -f "run_gr00t_server_nla_steer.py.*--port ${STEER_PORT}" 2>/dev/null || true
sleep 3

STEER_LOG_DIR="${SFT_DIR}/steer_server_logs"
STEER_PID_FILE="${STEER_LOG_DIR}/server.pid"
_port_busy=0
if command -v nc >/dev/null 2>&1 && nc -z localhost "${STEER_PORT}" 2>/dev/null; then
  _port_busy=1
fi
if [[ "${_port_busy}" -eq 1 ]] && [[ -f "${STEER_PID_FILE}" ]] && kill -0 "$(cat "${STEER_PID_FILE}")" 2>/dev/null; then
  echo "[v5-sim-smoke] steer server already up on port ${STEER_PORT} (pid=$(cat "${STEER_PID_FILE}"))"
else
  echo "[v5-sim-smoke] launching steer server (SFT_DIR=${SFT_DIR}, port=${STEER_PORT})"
  bash scripts/eval/launch_steer_server.sh \
    --sft-dir "$SFT_DIR" \
    --port "$STEER_PORT" \
    --log-dir "${STEER_LOG_DIR}" \
    --ready-timeout 300 \
    -- \
    --model-path "$MODEL_PATH" \
    --embodiment-tag LIBERO_PANDA \
    --steer-text-file scripts/eval/default_steer_boot.txt
fi

mkdir -p "$GRPO_OUT" logs
echo "[v5-sim-smoke] starting GRPO smoke -> ${GRPO_OUT}"
nohup "$PYTHON" scripts/training/run_grpo.py \
  --sft-dir "$SFT_DIR" \
  --activations-root "$ACT_ROOT" \
  --output-dir "$GRPO_OUT" \
  --batch-size 2 \
  --rollouts-per-activation 2 \
  --total-steps 5 \
  --eval-every 100 \
  --save-every 100 \
  --log-every 1 \
  --sim-reward-weight 0.5 \
  --sim-counterfactual-pairs-path data/grpo/libero_goal_counterfactual_pairs.jsonl \
  --sim-counterfactual-pairs-path-extra data/grpo/libero_spatial_counterfactual_pairs.jsonl \
  --sim-counterfactual-pairs-path-extra data/grpo/libero_object_counterfactual_pairs.jsonl \
  --sim-counterfactual-pairs-path-extra data/grpo/libero_10_counterfactual_pairs.jsonl \
  --cf-eligible-ids-path data/grpo/libero_4suite_cf_eligible.json \
  --sim-policy-host localhost \
  --sim-policy-port "$STEER_PORT" \
  --sim-n-workers 18 \
  --sim-max-steps 50 \
  --sim-rollout-python "$LIBERO_PY" \
  --dynamic-sampling \
  --use-ppo-clip \
  --disable-kl-anchor \
  --rollout-temperature-high 1.6 \
  --beta 0.0 \
  --seed 0 \
  >> "${GRPO_OUT}/grpo.log" 2>&1 &
echo $! > logs/v5_sim_grpo_smoke.pid
echo "[v5-sim-smoke] GRPO pid=$(cat logs/v5_sim_grpo_smoke.pid)"
echo "[v5-sim-smoke] tail -f ${GRPO_OUT}/grpo.log"

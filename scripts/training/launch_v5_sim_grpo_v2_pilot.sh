#!/usr/bin/env bash
# V2 sim-GRPO pilot (eval-v2 aligned): pure-ish sim reward + contrastive
# language-swap controls, KL anchor on, goal-only CF pool.
#
# Defaults below reward the *gap* between matched and mismatched_source
# intent arms (the train-side analog of the publishable
# ``semantic_gap_predicate`` metric) instead of just the matched
# predicate. This is the V2 "fix the auto-pass eval" recipe: sim_blend
# dropped to 0.4, judge to 0.25, contrastive sim weights enabled,
# sim_eval_protocol=language_swap so train sees the same channel as the
# holdout compare's eval-v2 default.
#
# Usage:
#   WALL_HOURS=16 bash scripts/training/launch_v5_sim_grpo_v2_pilot.sh
#   tail -f data/grpo/libero_4suite_v5_sim_grpo_v2_pilot/grpo.log
#
# Env:
#   WALL_HOURS          default 16
#   MAX_STEPS           default 2000 (wall usually stops first)
#   K_ROLLOUTS          default 4 (use 8 if VRAM stable with KL on)
#   DISABLE_KL          set to 1 for fallback-b (OOM)
#   STEER_PORT          default 5556
#   SIM_REWARD_WEIGHT   default 0.4 (was 1.0; recon picks up the slack)
#   SIM_JUDGE_WEIGHT    default 0.25 (was 0)
#   SIM_CONTRAST_W      default 0.5 (eval-v2 contrastive sim reward)
#   SIM_NULL_W          default 0.25 (eval-v2 null-control reward)
#   SIM_W_PREDICATE     default 1.0 (densify shaping; was 2.0)
#   SIM_EVAL_PROTOCOL   default language_swap (matches holdout compare)
#   SFT_DIR, ACT_ROOT, GRPO_OUT

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
ACT_ROOT="${ACT_ROOT:-data/activations/libero_4suite_v4_combined}"
GRPO_OUT="${GRPO_OUT:-data/grpo/libero_4suite_v5_sim_grpo_v2_pilot}"
STEER_PORT="${STEER_PORT:-5556}"
WALL_HOURS="${WALL_HOURS:-16}"
MAX_STEPS="${MAX_STEPS:-2000}"
K_ROLLOUTS="${K_ROLLOUTS:-4}"
SIM_REWARD_WEIGHT="${SIM_REWARD_WEIGHT:-0.4}"
SIM_JUDGE_WEIGHT="${SIM_JUDGE_WEIGHT:-0.25}"
SIM_CONTRAST_W="${SIM_CONTRAST_W:-0.5}"
SIM_NULL_W="${SIM_NULL_W:-0.25}"
SIM_W_PREDICATE="${SIM_W_PREDICATE:-1.0}"
SIM_EVAL_PROTOCOL="${SIM_EVAL_PROTOCOL:-language_swap}"
WALL_SECS=$(( WALL_HOURS * 3600 ))
PYTHON="${PYTHON_BIN:-.venv/bin/python}"
LIBERO_PY="${LIBERO_PY:-third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
MODEL_PATH="${GROOT_MODEL_PATH:-checkpoints/GR00T-N1.7-LIBERO/libero_goal}"
SIM_CACHE="${SIM_CACHE:-${GRPO_OUT}/sim_reward_cache.jsonl}"
CF_MANIFEST="${CF_MANIFEST:-data/grpo/libero_4suite_cf_eligible_goal_only.json}"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

if [[ ! -f "$CF_MANIFEST" ]]; then
  echo "[v2-sim-pilot] missing $CF_MANIFEST — run goal-only manifest build first"
  exit 1
fi

echo "[v2-sim-pilot] stopping prior V5 GRPO jobs (steer server untouched)"
pkill -f 'run_grpo.py.*libero_4suite_v5' 2>/dev/null || true
sleep 3

STEER_LOG_DIR="${SFT_DIR}/steer_server_logs"
STEER_PID_FILE="${STEER_LOG_DIR}/server.pid"
if command -v nc >/dev/null 2>&1 && nc -z localhost "${STEER_PORT}" 2>/dev/null \
   && [[ -f "${STEER_PID_FILE}" ]] && kill -0 "$(cat "${STEER_PID_FILE}")" 2>/dev/null; then
  echo "[v2-sim-pilot] steer server OK pid=$(cat "${STEER_PID_FILE}") port=${STEER_PORT}"
else
  echo "[v2-sim-pilot] launching steer server..."
  bash scripts/eval/launch_steer_server.sh \
    --sft-dir "$SFT_DIR" --port "$STEER_PORT" --log-dir "${STEER_LOG_DIR}" \
    --ready-timeout 300 -- \
    --model-path "$MODEL_PATH" --embodiment-tag LIBERO_PANDA \
    --steer-text-file scripts/eval/default_steer_boot.txt
fi

KL_ARGS=()
if [[ "${DISABLE_KL:-0}" == "1" ]]; then
  echo "[v2-sim-pilot] WARNING: KL anchor disabled (fallback-b)"
  KL_ARGS+=(--disable-kl-anchor --beta 0.0)
else
  KL_ARGS+=(--beta 0.03)
fi

mkdir -p "$GRPO_OUT" logs
echo "[v2-sim-pilot] V2 eval-v2 arm: sim=${SIM_REWARD_WEIGHT} judge=${SIM_JUDGE_WEIGHT}" \
     "contrast_w=${SIM_CONTRAST_W} null_w=${SIM_NULL_W} w_pred=${SIM_W_PREDICATE}" \
     "protocol=${SIM_EVAL_PROTOCOL}" \
     "KL=$([[ ${DISABLE_KL:-0} == 1 ]] && echo off || echo on)" \
     "K=${K_ROLLOUTS} wall=${WALL_HOURS}h -> ${GRPO_OUT}"

nohup timeout --signal=TERM "${WALL_SECS}" \
  "$PYTHON" scripts/training/run_grpo.py \
    --sft-dir "$SFT_DIR" \
    --activations-root "$ACT_ROOT" \
    --output-dir "$GRPO_OUT" \
    --batch-size 2 \
    --rollouts-per-activation "$K_ROLLOUTS" \
    --total-steps "$MAX_STEPS" \
    --learning-rate 1e-5 \
    --warmup-steps 50 \
    --eval-every 50 \
    --save-every 100 \
    --save-step-snapshots \
    --log-every 10 \
    --sim-reward-weight "$SIM_REWARD_WEIGHT" \
    --judge-reward-weight "$SIM_JUDGE_WEIGHT" \
    --sim-eval-protocol "$SIM_EVAL_PROTOCOL" \
    --sim-contrastive-weight "$SIM_CONTRAST_W" \
    --sim-null-control-weight "$SIM_NULL_W" \
    --sim-w-predicate "$SIM_W_PREDICATE" \
    "${KL_ARGS[@]}" \
    --sim-counterfactual-pairs-path data/grpo/libero_goal_counterfactual_pairs.jsonl \
    --sim-counterfactual-pairs-path-extra data/grpo/libero_spatial_counterfactual_pairs.jsonl \
    --sim-counterfactual-pairs-path-extra data/grpo/libero_object_counterfactual_pairs.jsonl \
    --sim-counterfactual-pairs-path-extra data/grpo/libero_10_counterfactual_pairs.jsonl \
    --cf-eligible-ids-path "$CF_MANIFEST" \
    --sim-policy-host localhost \
    --sim-policy-port "$STEER_PORT" \
    --sim-n-workers 18 \
    --sim-batch-size 4 \
    --sim-max-steps 100 \
    --sim-rollout-python "$LIBERO_PY" \
    --sim-cache-path "$SIM_CACHE" \
    --dynamic-sampling \
    --use-ppo-clip \
    --rollout-temperature-high 1.6 \
    --seed 0 \
  >> "${GRPO_OUT}/grpo.log" 2>&1 &

echo $! > logs/v5_sim_grpo_v2_pilot.pid
echo "[v2-sim-pilot] GRPO pid=$(cat logs/v5_sim_grpo_v2_pilot.pid) wall=${WALL_HOURS}h"
echo "[v2-sim-pilot] monitor: tail -f ${GRPO_OUT}/grpo.log ${GRPO_OUT}/metrics.jsonl"
echo "[v2-sim-pilot] held-out steer eval (run after each save_every=100 checkpoint):"
echo "  GRPO_AV_DIR=${GRPO_OUT}/av STEER_PORT=${STEER_PORT} \\"
echo "    bash scripts/eval/run_grpo_steer_holdout.sh"
echo "[v2-sim-pilot] per-step snapshots (for checkpoint sweeps):"
echo "  GRPO_AV_DIR=${GRPO_OUT}/av_steps/step_000100 STEER_PORT=${STEER_PORT} \\"
echo "    bash scripts/eval/run_grpo_steer_holdout.sh"

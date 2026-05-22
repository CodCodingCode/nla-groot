#!/usr/bin/env bash
# Record MP4s for CF steer comparison (SFT vs GRPO AV).
#
# 1) Winner only (SFT predicate hit from n4 eval):
# 2) Full 8-sample x 2 conditions (16 videos) — long run.
#
# Usage:
#   bash scripts/eval/launch_cf_steer_film.sh winner   # ~5 min
#   bash scripts/eval/launch_cf_steer_film.sh all8    # ~1–2 h
#
# Requires steer server on 5556.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=src

SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
GRPO_AV="${GRPO_AV:-data/grpo/libero_4suite_v5_sim_grpo_pilot/av}"
PAIRS="${PAIRS:-data/grpo/libero_goal_counterfactual_pairs.jsonl}"
ACT="${ACT:-data/activations/libero_4suite_v4_combined}"
PORT="${STEER_PORT:-5556}"
PYTHON="${PYTHON_BIN:-.venv/bin/python}"
PREV_JSON="${PREV_JSON:-data/eval/cf_steer_sft_vs_grpo_pilot_n4.json}"

MODE="${1:-all8}"

if ! nc -z localhost "$PORT" 2>/dev/null; then
  echo "ERROR: steer server not on port $PORT. Launch first:"
  echo "  bash scripts/eval/launch_steer_server.sh --sft-dir $SFT_DIR --port $PORT ..."
  exit 1
fi

case "$MODE" in
  winner)
    OUT="data/eval/cf_steer_videos/winner_bowl_on_cabinet"
    mkdir -p "$OUT"
    echo "[film] SFT winner: put_the_bowl_on_top_of_the_cabinet (from n4 eval)"
    "$PYTHON" scripts/eval/compare_cf_steer_checkpoints.py \
      --sft-dir "$SFT_DIR" \
      --grpo-av-dir "$GRPO_AV" \
      --pairs-path "$PAIRS" \
      --activations-root "$ACT" \
      --reuse-pairs-json "$PREV_JSON" \
      --only-source-id "goal__traj000323_step000044" \
      --conditions "sft_av" \
      --video-dir "$OUT" \
      --policy-port "$PORT" \
      --out-json "$OUT/summary.json"
    echo "[film] MP4: $OUT/00_goal__traj000323_step000044__sft_av__put_the_bowl_on_top_of_the_cabinet/rollout.mp4"
    ;;
  all8)
    OUT="data/eval/cf_steer_videos/pilot_n8"
    mkdir -p "$OUT"
    LOG="logs/cf_steer_film_n8.log"
    echo "[film] 8 samples x sft_av + grpo_av -> $OUT (background)"
    nohup "$PYTHON" scripts/eval/compare_cf_steer_checkpoints.py \
      --sft-dir "$SFT_DIR" \
      --grpo-av-dir "$GRPO_AV" \
      --pairs-path "$PAIRS" \
      --activations-root "$ACT" \
      --n-samples 8 \
      --conditions "sft_av,grpo_av" \
      --video-dir "$OUT" \
      --policy-port "$PORT" \
      --out-json "$OUT/summary.json" \
      > "$LOG" 2>&1 &
    echo $! > logs/cf_steer_film_n8.pid
    echo "[film] pid=$(cat logs/cf_steer_film_n8.pid) log=$LOG"
    ;;
  *)
    echo "Usage: $0 {winner|all8}"
    exit 1
    ;;
esac

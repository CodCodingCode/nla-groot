#!/usr/bin/env bash
# Run the publishable GRPO sim-steer eval on a held-out CF slice.
#
# One-shot wrapper:
#   1. Builds the held-out CF eval manifest (idempotent; skipped if present)
#   2. Runs compare_cf_steer_checkpoints with matched/mismatched intent arms
#      and semantic/null/wrong_placement causal arms
#   3. Builds the grpo_steer_scorecard from the compare output
#
# Operators should run this *after* a GRPO checkpoint save (every --save-every
# steps or at end of run). The steer server on $STEER_PORT must already be up
# (use scripts/eval/launch_steer_server.sh).
#
# Usage:
#   bash scripts/eval/run_grpo_steer_holdout.sh
#
# Env overrides:
#   SFT_DIR, GRPO_AV_DIR, ACT_ROOT, STEER_PORT, N_SAMPLES, OUT_DIR,
#   NARRATIVE (publishable | audit), PAIRS_PRIMARY, PAIRS_EXTRA (space-sep),
#   SIM_BATCH_SIZE (default 4: rollouts per batched subprocess; set to 1 to
#                   replay the legacy one-rollout-per-process behavior),
#   SIM_N_WORKERS (default auto: 1 when batched, else min(4, total_jobs)),
#   REUSE_SFT_FROM (path to prior compare JSON or SFT cache; skips SFT sim),
#   WRITE_SFT_CACHE (path to write slim SFT-only cache for reuse).
#
# Two-tier presets (set EVAL_TIER to override the defaults above):
#   EVAL_TIER=screen      → N_SAMPLES=32, matched/semantic arms only
#                           (~30-45 min after batching refactor; go/no-go).
#   EVAL_TIER=publishable → full defaults (N_SAMPLES=64, all arms +
#                           no_steer, ~4h).
#
# Eval protocol (default: language_swap, eval-v2 honest semantic_gap):
#   EVAL_PROTOCOL=language_swap → override the policy obs language per
#                                 intent arm so matched/mismatched_source
#                                 actually differ in the policy's language
#                                 channel (recommended default).
#   EVAL_PROTOCOL=legacy        → keep the env's BDDL task_description
#                                 (reproduces pre-2026-05 numbers, but
#                                 semantic_gap is structurally zero).
# See docs/evals/sim_steer_rollout.md and scripts/eval/eval_protocol.md.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH=src

SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
GRPO_AV_DIR="${GRPO_AV_DIR:-data/grpo/libero_4suite_v5_sim_grpo_v2_pilot/av}"
ACT_ROOT="${ACT_ROOT:-data/activations/libero_4suite_v4_combined}"
STEER_PORT="${STEER_PORT:-5556}"
OUT_DIR="${OUT_DIR:-data/eval/grpo_steer_holdout}"
NARRATIVE="${NARRATIVE:-publishable}"
PAIRS_PRIMARY="${PAIRS_PRIMARY:-data/grpo/libero_goal_counterfactual_pairs.jsonl}"
PAIRS_EXTRA=(${PAIRS_EXTRA:-data/grpo/libero_spatial_counterfactual_pairs.jsonl data/grpo/libero_object_counterfactual_pairs.jsonl data/grpo/libero_10_counterfactual_pairs.jsonl})
PYTHON="${PYTHON_BIN:-.venv/bin/python}"
LIBERO_PY="${LIBERO_PY:-third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"
HELD_OUT_FRACTION="${HELD_OUT_FRACTION:-0.05}"
SEED="${SEED:-0}"
SLICE="${SLICE:-all}"
SIM_BATCH_SIZE="${SIM_BATCH_SIZE:-4}"
SIM_N_WORKERS="${SIM_N_WORKERS:-}"  # empty = auto in compare
REUSE_SFT_FROM="${REUSE_SFT_FROM:-}"
WRITE_SFT_CACHE="${WRITE_SFT_CACHE:-}"
EVAL_TIER="${EVAL_TIER:-publishable}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-language_swap}"

# Two-tier presets. Explicit N_SAMPLES / INTENT_ARMS / CAUSAL_ARMS env vars
# still override (set after the preset so the user wins).
case "$EVAL_TIER" in
  screen)
    : "${N_SAMPLES:=32}"
    : "${INTENT_ARMS:=matched}"
    : "${CAUSAL_ARMS:=semantic,no_steer}"
    ;;
  publishable)
    : "${N_SAMPLES:=64}"
    : "${INTENT_ARMS:=matched,mismatched_source}"
    : "${CAUSAL_ARMS:=semantic,no_steer,matched_null,wrong_placement}"
    ;;
  *)
    echo "[holdout-eval] FATAL: unknown EVAL_TIER=${EVAL_TIER} (expected screen|publishable)" >&2
    exit 2
    ;;
esac
echo "[holdout-eval] EVAL_TIER=${EVAL_TIER} EVAL_PROTOCOL=${EVAL_PROTOCOL} N_SAMPLES=${N_SAMPLES} INTENT=${INTENT_ARMS} CAUSAL=${CAUSAL_ARMS}"

mkdir -p "$OUT_DIR"

# 1. Build held-out manifest if missing.
MANIFEST_PREFIX="${OUT_DIR}/libero_4suite_cf_eval_${SLICE}"
EVAL_PAIRS="${MANIFEST_PREFIX}_pairs.jsonl"
EVAL_MANIFEST="${MANIFEST_PREFIX}_eval_manifest.json"
TRAIN_MANIFEST="${MANIFEST_PREFIX}_train_manifest.json"
if [[ ! -f "$EVAL_PAIRS" || ! -f "$TRAIN_MANIFEST" ]]; then
  echo "[holdout-eval] building held-out manifest (slice=${SLICE})..."
  EXTRA_ARGS=()
  for p in "${PAIRS_EXTRA[@]}"; do
    EXTRA_ARGS+=(--pairs-extra "$p")
  done
  "$PYTHON" scripts/training/build_grpo_cf_eval_manifest.py \
    --pairs "$PAIRS_PRIMARY" "${EXTRA_ARGS[@]}" \
    --activations-root "$ACT_ROOT" \
    --seed "$SEED" --held-out-fraction "$HELD_OUT_FRACTION" --split-by episode \
    --slice "$SLICE" \
    --out "$MANIFEST_PREFIX"
else
  echo "[holdout-eval] reusing existing $EVAL_PAIRS"
fi

# 2. Run compare on held-out slice with full arm matrix.
COMPARE_JSON="${OUT_DIR}/cf_steer_compare.json"
echo "[holdout-eval] compare SFT vs GRPO, n=${N_SAMPLES}, arms=${INTENT_ARMS} x ${CAUSAL_ARMS}, sim-batch=${SIM_BATCH_SIZE}"
COMPARE_EXTRA=()
if [[ -n "$SIM_N_WORKERS" ]]; then
  COMPARE_EXTRA+=(--sim-n-workers "$SIM_N_WORKERS")
fi
if [[ -n "$REUSE_SFT_FROM" ]]; then
  COMPARE_EXTRA+=(--reuse-sft-from "$REUSE_SFT_FROM")
  echo "[holdout-eval] reusing SFT arm results from $REUSE_SFT_FROM"
fi
if [[ -n "$WRITE_SFT_CACHE" ]]; then
  COMPARE_EXTRA+=(--write-sft-cache "$WRITE_SFT_CACHE")
fi
"$PYTHON" scripts/eval/compare_cf_steer_checkpoints.py \
  --sft-dir "$SFT_DIR" \
  --grpo-av-dir "$GRPO_AV_DIR" \
  --pairs-path "$EVAL_PAIRS" \
  --activations-root "$ACT_ROOT" \
  --exclude-ids-path "$TRAIN_MANIFEST" \
  --require-held-out \
  --deterministic-order \
  --forbid-sim-cache \
  --n-samples "$N_SAMPLES" \
  --seed "$SEED" \
  --conditions sft_av,grpo_av \
  --intent-arms "$INTENT_ARMS" \
  --causal-arms "$CAUSAL_ARMS" \
  --eval-protocol "$EVAL_PROTOCOL" \
  --policy-port "$STEER_PORT" \
  --sim-rollout-python "$LIBERO_PY" \
  --sim-batch-size "$SIM_BATCH_SIZE" \
  "${COMPARE_EXTRA[@]}" \
  --out-json "$COMPARE_JSON"

# 3. Build scorecard.
SCORECARD_JSON="${OUT_DIR}/grpo_steer_scorecard.json"
GRPO_RUN_DIR="$(dirname "$GRPO_AV_DIR")"
GRPO_METRICS_ARG=()
if [[ -f "${GRPO_RUN_DIR}/metrics.jsonl" ]]; then
  GRPO_METRICS_ARG+=(--grpo-metrics "${GRPO_RUN_DIR}/metrics.jsonl")
fi
"$PYTHON" scripts/eval/build_grpo_steer_scorecard.py \
  --compare-json "$COMPARE_JSON" \
  --narrative "$NARRATIVE" \
  "${GRPO_METRICS_ARG[@]}" \
  --out-json "$SCORECARD_JSON"

echo "[holdout-eval] done"
echo "  compare:   $COMPARE_JSON"
echo "  scorecard: $SCORECARD_JSON"

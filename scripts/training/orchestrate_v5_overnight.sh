#!/usr/bin/env bash
#
# orchestrate_v5_overnight.sh
#
# Unattended end-to-end V5 pipeline (user-approved May 2026):
#   1. V5 step labeling (4 LIBERO suites)
#   2. Validate + expand to position rows
#   3. Merge combined labels
#   4. Re-mine hard negatives
#   5. Fresh SFT on base Qwen3-4B (NOT resumed from V4 checkpoint)
#   6. Post-SFT metric check
#   7. GRPO recon pilot on frozen AR
#
# Usage:
#   nohup bash scripts/training/orchestrate_v5_overnight.sh \
#     > logs/v5_overnight.boot 2>&1 &
#   echo $! > logs/v5_overnight.pid
#
#   tail -f logs/v5_overnight.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ORCH_VERSION="v5.overnight.1"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/v5_overnight.log"
PID_FILE="${LOG_DIR}/v5_overnight.pid"

mkdir -p "$LOG_DIR"
echo $$ > "$PID_FILE"

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH=src

# Data paths
ACT_PER_SUITE="${ACT_PER_SUITE:-data/activations/libero_4suite_stride2}"
ACT_COMBINED="${ACT_COMBINED:-data/activations/libero_4suite_v4_combined}"
LBL_ROOT="${LBL_ROOT:-data/labels/libero_4suite_v5}"
LBL_COMBINED="${LBL_COMBINED:-data/labels/libero_4suite_v5_combined}"
SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
GRPO_DIR="${GRPO_DIR:-data/grpo/libero_4suite_v5_base_qwen_grpo}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
LABEL_MODEL="${OPENAI_LABELING_MODEL:-gpt-5.1-mini}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-64}"
SFT_STEPS="${SFT_STEPS:-3000}"
GRPO_STEPS="${GRPO_STEPS:-500}"

SUITES=(spatial goal object 10)

stage_start() { STAGE_T0=$(date +%s); STAGE_NAME="$1"; log ">>> STAGE: $1"; }
stage_ok() {
  local elapsed=$(( $(date +%s) - STAGE_T0 ))
  log "[orchestrate] stage=${STAGE_NAME} status=ok elapsed_s=${elapsed}"
}
stage_fail() {
  local elapsed=$(( $(date +%s) - STAGE_T0 ))
  log "[orchestrate] stage=${STAGE_NAME} status=fail elapsed_s=${elapsed} msg=$*"
  exit 1
}

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

log "==== V5 overnight orchestrator ${ORCH_VERSION} ===="
log "repo=$REPO_ROOT"
log "log=$LOG_FILE"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
stage_start "preflight"

[[ -x "$PYTHON" ]] || stage_fail "missing $PYTHON"
[[ -n "${OPENAI_API_KEY:-}" ]] || stage_fail "OPENAI_API_KEY not set"

for s in "${SUITES[@]}"; do
  [[ -f "${ACT_PER_SUITE}/libero_${s}/manifest.json" ]] || \
    stage_fail "missing activations ${ACT_PER_SUITE}/libero_${s}"
done
[[ -f "${ACT_COMBINED}/stats_pooled.json" ]] || \
  stage_fail "missing ${ACT_COMBINED}/stats_pooled.json"

stage_ok

# ---------------------------------------------------------------------------
# Free GPU: stop other training jobs in this repo (user granted full permission)
# ---------------------------------------------------------------------------
stage_start "gpu_cleanup"
if pgrep -f "scripts/training/run_sft.py" >/dev/null 2>&1; then
  log "Stopping existing run_sft.py processes to free GPU for V5 SFT"
  pkill -f "scripts/training/run_sft.py" || true
  sleep 15
fi
stage_ok

# ---------------------------------------------------------------------------
# V5 step labeling (one nested JSON call per timestep per suite)
# ---------------------------------------------------------------------------
for SUITE in "${SUITES[@]}"; do
  stage_start "label_${SUITE}"
  ACT_DIR="${ACT_PER_SUITE}/libero_${SUITE}"
  LBL_DIR="${LBL_ROOT}/libero_${SUITE}"
  DS_DIR="third_party/Isaac-GR00T/examples/LIBERO/libero_${SUITE}_no_noops_1.0.0_lerobot"
  mkdir -p "$LBL_DIR"

  if [[ -f "${LBL_DIR}/labels_steps.jsonl" ]] && [[ -s "${LBL_DIR}/labels_steps.jsonl" ]]; then
    n=$(wc -l < "${LBL_DIR}/labels_steps.jsonl")
    log "  ${SUITE}: labels_steps.jsonl exists (${n} rows), skipping label API"
  else
    log "  ${SUITE}: V5 step labeling -> ${LBL_DIR}"
    "$PYTHON" scripts/labeling/run_label.py \
      --activations-root "$ACT_DIR" \
      --dataset-root "$DS_DIR" \
      --labels-dir "$LBL_DIR" \
      --prompt-mode v5 \
      --suite "$SUITE" \
      --concurrency "$LABEL_CONCURRENCY" \
      --model "$LABEL_MODEL" \
      2>&1 | tee -a "${LBL_DIR}/run_label.log"
  fi
  stage_ok

  stage_start "validate_${SUITE}"
  "$PYTHON" scripts/labeling/validate_v5_labels.py \
    --in "${LBL_DIR}/labels_steps.jsonl" \
    --show-errors 5 --jaccard \
    2>&1 | tee "${LBL_DIR}/validate_v5.log" || \
    log "WARN: ${SUITE} validator reported invalid rows (expand uses --skip-invalid)"
  stage_ok

  stage_start "expand_${SUITE}"
  "$PYTHON" scripts/labeling/expand_step_labels.py \
    --labels-steps "${LBL_DIR}/labels_steps.jsonl" \
    --activations-root "$ACT_DIR" \
    --out "${LBL_DIR}/labels.jsonl" \
    --suite "$SUITE" \
    --skip-invalid \
    2>&1 | tee "${LBL_DIR}/expand.log"
  stage_ok
done

# ---------------------------------------------------------------------------
# Merge + hard negatives
# ---------------------------------------------------------------------------
stage_start "merge_combined"
mkdir -p "$LBL_COMBINED"
"$PYTHON" scripts/labeling/build_v5_combined_labels.py \
  --per-suite-root "$LBL_ROOT" \
  --out "$LBL_COMBINED" \
  2>&1 | tee "${LBL_COMBINED}/merge.log"
stage_ok

stage_start "mine_hard_negatives"
"$PYTHON" scripts/training/mine_hard_negatives.py \
  --activations-root "$ACT_COMBINED" \
  --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
  --out "${ACT_COMBINED}/hard_negatives_v5.jsonl" \
  --per-position-type \
  --last-text-strategy random_same_ptype \
  --jaccard-cap 0.55 \
  --top-k 8 \
  2>&1 | tee "${LBL_COMBINED}/mine_hard_neg.log"
stage_ok

mkdir -p "${LOG_DIR}/v5_guard"
date -Iseconds > "${LOG_DIR}/v5_guard/labels_ready.flag"

# ---------------------------------------------------------------------------
# SFT — fresh LoRA on base Qwen3-4B (V5 labels + V5 prompts; NOT V4 checkpoint)
# Delegated to watch_v5_sft_guard.sh when sft_started.flag exists (~14h wall budget).
# ---------------------------------------------------------------------------
if [[ -f "${LOG_DIR}/v5_guard/sft_started.flag" ]]; then
  log "SKIP sft_v5_base_qwen — sft_guard owns training (see logs/v5_guard/sft_guard.log)"
  stage_start "sft_v5_base_qwen"
  stage_ok
else
stage_start "sft_v5_base_qwen"
mkdir -p "$SFT_DIR"
"$PYTHON" scripts/training/run_sft.py \
  --activations-root "$ACT_COMBINED" \
  --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
  --stats-json "${ACT_COMBINED}/stats_pooled.json" \
  --output-dir "$SFT_DIR" \
  --base-model "$BASE_MODEL" \
  --av-prompt-version context_v5 \
  --ar-prompt-version context_v5 \
  --image-patch-pooling strided_image_multi \
  --image-patch-pooling-strided-k 8 \
  --av-num-image-slots 8 \
  --balance-position-mix \
  --position-mix-json '{"last_text": 0.33, "image_patch": 0.34, "anchor": 0.33}' \
  --ar-contrastive-weight 0.5 \
  --ar-nce-hard-negative-source topk_cosine \
  --ar-nce-hard-negative-index-path "${ACT_COMBINED}/hard_negatives_v5.jsonl" \
  --ar-av-mix-max 0.4 \
  --ar-av-mix-warmup-frac 0.3 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --warmup-steps 200 \
  --total-steps "$SFT_STEPS" \
  --eval-every 250 \
  --save-every 500 \
  --eval-closed-loop \
  --closed-loop-temps 0.0 0.7 \
  --closed-loop-max-batches 64 \
  2>&1 | tee "${SFT_DIR}/sft.log"
stage_ok
fi

if [[ -f "${LOG_DIR}/v5_guard/sft_success.flag" ]]; then
  log "SKIP sft_metrics_check + grpo — post_guard handles tail"
  stage_start "sft_metrics_check"
  stage_ok
  stage_start "grpo_recon"
  stage_ok
else
stage_start "sft_metrics_check"
if [[ -f "${SFT_DIR}/metrics.jsonl" ]]; then
  "$PYTHON" scripts/ci/check_sft_metrics.py \
    "${SFT_DIR}/metrics.jsonl" \
    --batch-size 4 \
    --config "${SFT_DIR}/config.json" \
    --require-closed-loop \
    --max-tf-closed-fve-gap 0.08 \
    2>&1 | tee "${SFT_DIR}/metrics_check.log" || \
    log "WARN: metrics check failed (continuing to GRPO)"
fi
stage_ok

# ---------------------------------------------------------------------------
# GRPO recon (after SFT; AR frozen for reward)
# ---------------------------------------------------------------------------
stage_start "grpo_recon"
mkdir -p "$GRPO_DIR"
"$PYTHON" scripts/training/run_grpo.py \
  --sft-dir "$SFT_DIR" \
  --activations-root "$ACT_COMBINED" \
  --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
  --output-dir "$GRPO_DIR" \
  --beta 0.02 \
  --total-steps "$GRPO_STEPS" \
  --rollouts-per-activation 8 \
  --eval-every 50 \
  --save-every 100 \
  2>&1 | tee "${GRPO_DIR}/grpo.log"
stage_ok
fi

log "==== V5 overnight orchestrator COMPLETE ===="
log "  labels: ${LBL_COMBINED}/labels.jsonl"
log "  sft:    ${SFT_DIR}"
log "  grpo:   ${GRPO_DIR}"

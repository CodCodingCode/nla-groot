#!/usr/bin/env bash
# Guard 3/3: after SFT success — metrics check, GRPO with retries, final report.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
GUARD_LOG="${REPO_ROOT}/logs/v5_guard/post_guard.log"
PID_FILE="${REPO_ROOT}/logs/v5_guard/post_guard.pid"
SFT_OK="${REPO_ROOT}/logs/v5_guard/sft_success.flag"
DONE_FLAG="${REPO_ROOT}/logs/v5_guard/pipeline_complete.flag"
POLL_SECS="${POLL_SECS:-180}"
GRPO_WALL_HOURS="${GRPO_WALL_HOURS:-4}"
GRPO_WALL_SECS=$(( GRPO_WALL_HOURS * 3600 ))

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH=src
ACT_COMBINED="${ACT_COMBINED:-data/activations/libero_4suite_v4_combined}"
LBL_COMBINED="${LBL_COMBINED:-data/labels/libero_4suite_v5_combined}"
SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
GRPO_DIR="${GRPO_DIR:-data/grpo/libero_4suite_v5_base_qwen_grpo}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [post_guard] $*" | tee -a "$GUARD_LOG"; }

if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi

resolve_sft_dir() {
  if [[ -f "${REPO_ROOT}/logs/v5_guard/sft_dir.txt" ]]; then
    cat "${REPO_ROOT}/logs/v5_guard/sft_dir.txt"
    return
  fi
  for d in "${SFT_DIR}" "${SFT_DIR}_retry2" "${SFT_DIR}_retry3"; do
    [[ -f "${d}/av/adapter_config.json" ]] && { echo "$d"; return; }
  done
  echo "$SFT_DIR"
}

log "post_guard boot — waiting for SFT success"
while [[ ! -f "$SFT_OK" ]]; do sleep "$POLL_SECS"; done

SFT_DIR=$(resolve_sft_dir)
log "using SFT_DIR=$SFT_DIR"

# Metrics check (non-fatal)
if [[ -f "${SFT_DIR}/metrics.jsonl" ]]; then
  "$PYTHON" scripts/ci/check_sft_metrics.py \
    "${SFT_DIR}/metrics.jsonl" \
    --batch-size 4 --config "${SFT_DIR}/config.json" \
    --require-closed-loop --max-tf-closed-fve-gap 0.10 \
    >> "${SFT_DIR}/metrics_check.log" 2>&1 || \
    log "WARN metrics check failed (see metrics_check.log)"
fi

# AV samples for morning inspection
"$PYTHON" scripts/eval/dump_av_samples.py \
  --ckpt-dir "$SFT_DIR" \
  --activations-root "$ACT_COMBINED" \
  --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
  --per-position 6 --seed 0 --temperatures 0.0 0.7 \
  --out-jsonl "${SFT_DIR}/samples_guard.jsonl" \
  >> "${GRPO_DIR}/dump_samples.log" 2>&1 || true

attempt=1
while [[ "$attempt" -le 3 ]]; do
  mkdir -p "$GRPO_DIR"
  log "GRPO attempt=$attempt"
  pkill -f "scripts/training/run_grpo.py.*libero_4suite_v5" 2>/dev/null || true
  sleep 10
  set +e
  timeout --signal=TERM "${GRPO_WALL_SECS}" \
    "$PYTHON" scripts/training/run_grpo.py \
      --sft-dir "$SFT_DIR" \
      --activations-root "$ACT_COMBINED" \
      --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
      --output-dir "${GRPO_DIR}_a${attempt}" \
      --beta 0.02 --total-steps 500 \
      --rollouts-per-activation 8 \
      --eval-every 50 --save-every 100 \
      >> "${GRPO_DIR}_a${attempt}/grpo.log" 2>&1
  rc=$?
  set -e
  if [[ -f "${GRPO_DIR}_a${attempt}/metrics.jsonl" ]] || [[ $rc -eq 124 ]]; then
    ln -sfn "${GRPO_DIR}_a${attempt}" "$GRPO_DIR" 2>/dev/null || true
    date -Iseconds > "$DONE_FLAG"
    log "pipeline_complete.flag written (GRPO attempt=$attempt rc=$rc)"
    exit 0
  fi
  attempt=$(( attempt + 1 ))
done

log "GRPO failed — SFT still valid; see sft_success.flag"
date -Iseconds > "$DONE_FLAG"
exit 0

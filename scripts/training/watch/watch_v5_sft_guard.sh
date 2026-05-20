#!/usr/bin/env bash
# Guard 2/3: wait for labels, run ~14h wall-clock SFT on base Qwen3-4B with retries.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
GUARD_LOG="${REPO_ROOT}/logs/v5_guard/sft_guard.log"
PID_FILE="${REPO_ROOT}/logs/v5_guard/sft_guard.pid"
READY_FLAG="${REPO_ROOT}/logs/v5_guard/labels_ready.flag"
SFT_FLAG="${REPO_ROOT}/logs/v5_guard/sft_started.flag"
SFT_OK="${REPO_ROOT}/logs/v5_guard/sft_success.flag"
LOCK_FILE="${REPO_ROOT}/logs/v5_guard/sft.lock"
mkdir -p "${REPO_ROOT}/logs/v5_guard"
echo $$ > "$PID_FILE"

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH=src
SFT_WALL_HOURS="${SFT_WALL_HOURS:-14}"
SFT_WALL_SECS=$(( SFT_WALL_HOURS * 3600 ))
MAX_SFT_ATTEMPTS="${MAX_SFT_ATTEMPTS:-3}"
POLL_SECS="${POLL_SECS:-120}"

ACT_COMBINED="${ACT_COMBINED:-data/activations/libero_4suite_v4_combined}"
LBL_COMBINED="${LBL_COMBINED:-data/labels/libero_4suite_v5_combined}"
SFT_DIR="${SFT_DIR:-data/sft/libero_4suite_v5_base_qwen}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PROBE_STEPS="${PROBE_STEPS:-80}"

# Log to file + stderr only (never stdout — probe_total_steps is captured via $()).
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [sft_guard] $*"
  echo "$msg" >> "$GUARD_LOG"
  echo "$msg" >&2
}

if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi

wait_labels() {
  log "waiting for $READY_FLAG"
  while [[ ! -f "$READY_FLAG" ]]; do sleep "$POLL_SECS"; done
  while [[ ! -f "${LBL_COMBINED}/labels.jsonl" ]]; do sleep 30; done
  while [[ ! -f "${ACT_COMBINED}/hard_negatives_v5.jsonl" ]]; do sleep 30; done
  log "labels + hard_negatives ready"
}

probe_total_steps() {
  local probe_dir="${SFT_DIR}_probe"
  mkdir -p "$probe_dir"
  log "probing ${PROBE_STEPS} steps for steps/sec estimate"
  local t0=$(date +%s)
  if ! "$PYTHON" scripts/training/run_sft.py \
      --activations-root "$ACT_COMBINED" \
      --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
      --stats-json "${ACT_COMBINED}/stats_pooled.json" \
      --output-dir "$probe_dir" \
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
      --batch-size 4 --learning-rate 1e-4 --warmup-steps 20 \
      --total-steps "$PROBE_STEPS" \
      --eval-every 999999 --save-every 999999 --log-every 20 \
      >> "${probe_dir}/probe.log" 2>&1; then
    log "probe failed — fallback total_steps=8000"
    echo 8000
    return
  fi
  local t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))
  [[ "$elapsed" -lt 1 ]] && elapsed=1
  local steps=$(( SFT_WALL_SECS * 85 / 100 * PROBE_STEPS / elapsed ))
  [[ "$steps" -lt 2000 ]] && steps=2000
  [[ "$steps" -gt 50000 ]] && steps=50000
  log "probe elapsed=${elapsed}s -> total_steps=${steps} for ${SFT_WALL_HOURS}h budget"
  echo "$steps" > "${REPO_ROOT}/logs/v5_guard/probe_total_steps.txt"
  echo "$steps"
}

validate_steps() {
  local n="$1"
  if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -ge 500 ]]; then
    echo "$n"
  else
    log "WARN invalid total_steps='$n' — using fallback 8000"
    echo 8000
  fi
}

run_sft_attempt() {
  local attempt="$1"
  local total_steps="$2"
  local out_dir="${SFT_DIR}"
  if [[ "$attempt" -gt 1 ]]; then
    out_dir="${SFT_DIR}_retry${attempt}"
  fi
  mkdir -p "$out_dir"

  log "SFT attempt=${attempt} out=${out_dir} steps=${total_steps} wall=${SFT_WALL_HOURS}h"
  pkill -f "scripts/training/run_sft.py" 2>/dev/null || true
  sleep 20

  set +e
  timeout --signal=TERM "${SFT_WALL_SECS}" \
    "$PYTHON" scripts/training/run_sft.py \
      --activations-root "$ACT_COMBINED" \
      --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
      --stats-json "${ACT_COMBINED}/stats_pooled.json" \
      --output-dir "$out_dir" \
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
      --ar-av-mix-max 0.4 --ar-av-mix-warmup-frac 0.3 \
      --batch-size 4 --learning-rate 1e-4 --warmup-steps 200 \
      --total-steps "$total_steps" \
      --eval-every 250 --save-every 500 --log-every 10 \
      --eval-closed-loop --closed-loop-temps 0.0 0.7 \
      --closed-loop-max-batches 64 \
      >> "${out_dir}/sft.log" 2>&1
  local rc=$?
  set -e
  log "SFT attempt=${attempt} exit=$rc (124=timeout is OK)"

  if [[ -f "${out_dir}/av/adapter_config.json" && -f "${out_dir}/ar/adapter_config.json" \
      && -f "${out_dir}/metrics.jsonl" ]]; then
    local last_step
    last_step=$("$PYTHON" -c "
import json
p='${out_dir}/metrics.jsonl'
s=0
for line in open(p):
    try: s=max(s, json.loads(line).get('step',0))
    except: pass
print(s)
")
    [[ "$last_step" -ge 500 ]] && return 0
  fi
  return 1
}

log "sft_guard boot"
wait_labels

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "another process holds sft.lock — exit"
  exit 0
fi

date -Iseconds > "$SFT_FLAG"
ln -sfn "$SFT_DIR" "${REPO_ROOT}/data/sft/libero_4suite_v5_active" 2>/dev/null || true

# Reuse cached probe if guard crashed after probe (e.g. stdout bug) and SKIP_PROBE=1.
if [[ "${SKIP_PROBE:-0}" == "1" && -f "${REPO_ROOT}/logs/v5_guard/probe_total_steps.txt" ]]; then
  total_steps=$(validate_steps "$(tr -d ' \n' < "${REPO_ROOT}/logs/v5_guard/probe_total_steps.txt")")
  log "SKIP_PROBE=1 using cached total_steps=${total_steps}"
else
  total_steps=$(validate_steps "$(probe_total_steps)")
fi
attempt=1
while [[ "$attempt" -le "$MAX_SFT_ATTEMPTS" ]]; do
  if run_sft_attempt "$attempt" "$total_steps"; then
    date -Iseconds > "$SFT_OK"
    echo "$SFT_DIR" > "${REPO_ROOT}/logs/v5_guard/sft_dir.txt"
    log "SFT SUCCESS flag written"
    exit 0
  fi
  attempt=$(( attempt + 1 ))
  total_steps=$(( total_steps * 9 / 10 ))
  log "retry with reduced steps=$total_steps"
  sleep 60
done

log "SFT FAILED after $MAX_SFT_ATTEMPTS attempts"
exit 1

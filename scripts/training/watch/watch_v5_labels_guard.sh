#!/usr/bin/env bash
# Guard 1/3: ensure V5 labeling → validate → expand → merge → hard-negs completes.
# Retries failed/stalled suites; writes logs/v5_guard/labels_ready.flag when done.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
GUARD_LOG="${REPO_ROOT}/logs/v5_guard/labels_guard.log"
PID_FILE="${REPO_ROOT}/logs/v5_guard/labels_guard.pid"
READY_FLAG="${REPO_ROOT}/logs/v5_guard/labels_ready.flag"
mkdir -p "${REPO_ROOT}/logs/v5_guard"
echo $$ > "$PID_FILE"

PYTHON="${PYTHON_BIN:-.venv/bin/python}"
export PYTHONPATH=src
POLL_SECS="${POLL_SECS:-180}"
MAX_RETRIES="${MAX_RETRIES:-5}"

ACT_PER_SUITE="${ACT_PER_SUITE:-data/activations/libero_4suite_stride2}"
ACT_COMBINED="${ACT_COMBINED:-data/activations/libero_4suite_v4_combined}"
LBL_ROOT="${LBL_ROOT:-data/labels/libero_4suite_v5}"
LBL_COMBINED="${LBL_COMBINED:-data/labels/libero_4suite_v5_combined}"
LABEL_MODEL="${OPENAI_LABELING_MODEL:-gpt-5.1-mini}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-64}"
SUITES=(spatial goal object 10)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [labels_guard] $*" | tee -a "$GUARD_LOG"; }

if [[ -f "${REPO_ROOT}/.env" ]]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi

expected_steps() {
  local suite="$1"
  local idx="${ACT_PER_SUITE}/libero_${suite}/index.jsonl"
  [[ -f "$idx" ]] || { echo 0; return; }
  wc -l < "$idx" | tr -d ' '
}

labeled_steps() {
  local f="${LBL_ROOT}/libero_${1}/labels_steps.jsonl"
  [[ -f "$f" ]] || { echo 0; return; }
  wc -l < "$f" | tr -d ' '
}

run_suite_pipeline() {
  local suite="$1"
  local attempt="$2"
  local ACT_DIR="${ACT_PER_SUITE}/libero_${suite}"
  local LBL_DIR="${LBL_ROOT}/libero_${suite}"
  local DS_DIR="third_party/Isaac-GR00T/examples/LIBERO/libero_${suite}_no_noops_1.0.0_lerobot"
  mkdir -p "$LBL_DIR"

  log "suite=${suite} attempt=${attempt} START pipeline"
  if ! "$PYTHON" scripts/labeling/run_label.py \
      --activations-root "$ACT_DIR" \
      --dataset-root "$DS_DIR" \
      --labels-dir "$LBL_DIR" \
      --prompt-mode v5 \
      --suite "$suite" \
      --concurrency "$LABEL_CONCURRENCY" \
      --model "$LABEL_MODEL" \
      >> "${LBL_DIR}/run_label.log" 2>&1; then
    log "suite=${suite} run_label FAILED"
    return 1
  fi

  "$PYTHON" scripts/labeling/validate_v5_labels.py \
    --in "${LBL_DIR}/labels_steps.jsonl" --show-errors 3 --jaccard \
    >> "${LBL_DIR}/validate_v5.log" 2>&1 || true

  if ! "$PYTHON" scripts/labeling/expand_step_labels.py \
      --labels-steps "${LBL_DIR}/labels_steps.jsonl" \
      --activations-root "$ACT_DIR" \
      --out "${LBL_DIR}/labels.jsonl" \
      --suite "$suite" \
      --skip-invalid \
      >> "${LBL_DIR}/expand.log" 2>&1; then
    log "suite=${suite} expand FAILED"
    return 1
  fi

  local n_exp
  n_exp=$(wc -l < "${LBL_DIR}/labels.jsonl" | tr -d ' ')
  [[ "$n_exp" -gt 100 ]] || { log "suite=${suite} too few expanded rows: $n_exp"; return 1; }
  log "suite=${suite} OK expanded_rows=$n_exp"
  return 0
}

merge_and_mine() {
  mkdir -p "$LBL_COMBINED"
  "$PYTHON" scripts/labeling/build_v5_combined_labels.py \
    --per-suite-root "$LBL_ROOT" --out "$LBL_COMBINED" \
    >> "${LBL_COMBINED}/merge.log" 2>&1 || return 1
  local n
  n=$(wc -l < "${LBL_COMBINED}/labels.jsonl" | tr -d ' ')
  [[ "$n" -gt 10000 ]] || { log "merge too few rows: $n"; return 1; }

  "$PYTHON" scripts/training/mine_hard_negatives.py \
    --activations-root "$ACT_COMBINED" \
    --labels-jsonl "${LBL_COMBINED}/labels.jsonl" \
    --out "${ACT_COMBINED}/hard_negatives_v5.jsonl" \
    --per-position-type \
    --last-text-strategy random_same_ptype \
    --jaccard-cap 0.55 --top-k 8 \
    >> "${LBL_COMBINED}/mine_hard_neg.log" 2>&1 || return 1
  [[ -f "${ACT_COMBINED}/hard_negatives_v5.jsonl" ]] || return 1
  log "merge+mine OK combined_rows=$n"
  return 0
}

declare -A RETRIES
orchestrator_running() {
  [[ -f "${REPO_ROOT}/logs/v5_overnight.pid" ]] && \
    kill -0 "$(cat "${REPO_ROOT}/logs/v5_overnight.pid")" 2>/dev/null
}

any_label_job() {
  pgrep -f "scripts/labeling/run_label.py" >/dev/null 2>&1
}

log "labels_guard boot pid=$$"

while true; do
  if [[ -f "$READY_FLAG" ]]; then
    sleep "$POLL_SECS"
    continue
  fi

  all_ok=1
  for suite in "${SUITES[@]}"; do
    exp=$(expected_steps "$suite")
    got=$(labeled_steps "$suite")
    has_exp=0
    [[ -f "${LBL_ROOT}/libero_${suite}/labels.jsonl" ]] && \
      has_exp=$(wc -l < "${LBL_ROOT}/libero_${suite}/labels.jsonl" | tr -d ' ')

    need=0
    if [[ "$exp" -gt 0 && "$got" -lt $(( exp * 95 / 100 )) ]]; then need=1; fi
    if [[ "$has_exp" -lt $(( exp * 3 * 95 / 100 )) ]]; then need=1; fi

    if [[ "$need" -eq 1 ]]; then
      all_ok=0
      r=${RETRIES[$suite]:-0}
      if [[ "$r" -lt "$MAX_RETRIES" ]]; then
        if pgrep -f "run_label.py.*libero_${suite}" >/dev/null 2>&1; then
          log "suite=${suite} labeling in progress ($got/$exp) — wait"
        elif orchestrator_running && any_label_job; then
          log "suite=${suite} incomplete ($got/$exp) — orchestrator busy elsewhere, defer"
        elif orchestrator_running && [[ "$suite" != "spatial" ]]; then
          log "suite=${suite} incomplete — defer until orchestrator finishes prior suites"
        else
          RETRIES[$suite]=$(( r + 1 ))
          run_suite_pipeline "$suite" "${RETRIES[$suite]}" || true
        fi
      else
        log "ERROR suite=${suite} exceeded retries ($got/$exp)"
      fi
    else
      log "suite=${suite} OK steps=$got/$exp expanded=$has_exp"
    fi
  done

  if [[ "$all_ok" -eq 1 ]]; then
    if merge_and_mine; then
      date -Iseconds > "$READY_FLAG"
      log "labels_ready.flag written — Guard 1 done"
    fi
  fi

  sleep "$POLL_SECS"
done

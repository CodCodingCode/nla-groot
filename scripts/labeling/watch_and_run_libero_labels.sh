#!/usr/bin/env bash
#
# Per-suite labeling watcher.
#
# Walks the 4 LIBERO suites in order. For each one, waits for its activation
# manifest to appear, then runs the suite's labeling job. This lets labeling
# of an earlier suite (OpenAI API) overlap with extraction of a later suite
# (GPU), which is the natural parallelism between the two stages.
#
# Per-suite labeling is resumable via run_label.py's state file, so killing
# this watcher mid-run won't lose committed labels — relaunching picks up
# where the suite left off.
#
# Usage::
#
#   mkdir -p logs
#   nohup bash scripts/labeling/watch_and_run_libero_labels.sh \
#       > logs/libero_labels_watch.boot 2>&1 &
#   echo "labels watcher pid=$!"
#
# Inspect::
#
#   tail -f logs/libero_labels_watch.log
#   ls    data/labels/libero_4suite_stride2/libero_*/labels.jsonl
#
# Abort::
#
#   kill $(cat /tmp/libero_labels_watch.pid)
#
# Once all 4 manifests at data/labels/libero_4suite_stride2/libero_<suite>/
# exist, the separate SFT watcher (watch_and_run_libero_sft.sh) takes over.

set -uo pipefail
cd "$(dirname "$0")/../.."

SUITES=(goal spatial object 10)
ACT_ROOT="data/activations/libero_4suite_stride2"
LBL_ROOT="data/labels/libero_4suite_stride2"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/libero_labels_watch.log"

POLL_SECS="${POLL_SECS:-60}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"
POS_PER_EX="${POS_PER_EX:-2}"
CONCURRENCY="${CONCURRENCY:-128}"

PY=".venv/bin/python"

mkdir -p "$LOG_DIR" "$LBL_ROOT"

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

trap 'log "Labels watcher caught signal; exiting (pid=$$)."; exit 130' INT TERM

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    log "FATAL: OPENAI_API_KEY is not set (and .env did not provide one). Aborting."
    exit 2
fi
if [ ! -x "$PY" ]; then
    log "FATAL: $PY does not exist or is not executable."
    exit 2
fi

log "==== libero-4suite LABELS watcher booting ===="
log "  cwd:           $(pwd)"
log "  suites:        ${SUITES[*]}"
log "  poll every:    ${POLL_SECS}s"
log "  positions/ex:  ${POS_PER_EX}"
log "  concurrency:   ${CONCURRENCY}"
log "  activations:   ${ACT_ROOT}"
log "  labels out:    ${LBL_ROOT}"
log ""

# Each suite is processed in order. For each suite:
#   wait for its activation manifest → run run_label.py with --guarantee-strata.
for SUITE in "${SUITES[@]}"; do
    ACT_DIR="${ACT_ROOT}/libero_${SUITE}"
    LBL_DIR="${LBL_ROOT}/libero_${SUITE}"
    DS_DIR="third_party/Isaac-GR00T/examples/LIBERO/libero_${SUITE}_no_noops_1.0.0_lerobot"
    LOG_PER_SUITE="${LBL_ROOT}/label_${SUITE}.log"

    log "Phase: libero_${SUITE}"

    if [ -f "${LBL_DIR}/manifest.json" ] && [ -s "${LBL_DIR}/labels.jsonl" ]; then
        n=$(wc -l < "${LBL_DIR}/labels.jsonl")
        log "  libero_${SUITE}: already done (${n} rows). Skipping."
        continue
    fi

    log "  libero_${SUITE}: waiting for activations manifest at ${ACT_DIR}/manifest.json"
    deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
    while [ ! -f "${ACT_DIR}/manifest.json" ]; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            log "FATAL: ${MAX_WAIT_HOURS}h elapsed waiting for libero_${SUITE} manifest. Aborting."
            exit 3
        fi
        sleep "$POLL_SECS"
    done
    # Belt-and-suspenders: the manifest is written at the end of extraction.
    # Sleep one extra poll cycle to be sure the last shard's safetensors are
    # fully flushed before we start reading.
    sleep "$POLL_SECS"

    n_act=$(jq -r '.num_examples // 0' "${ACT_DIR}/manifest.json" 2>/dev/null || echo 0)
    log "  libero_${SUITE}: activations ready (${n_act} examples). Starting labeling."

    mkdir -p "${LBL_DIR}"
    set +e
    PYTHONPATH=src $PY scripts/labeling/run_label.py \
        --activations-root      "${ACT_DIR}" \
        --dataset-root          "${DS_DIR}" \
        --labels-dir            "${LBL_DIR}" \
        --positions-per-example "${POS_PER_EX}" \
        --guarantee-strata \
        --concurrency           "${CONCURRENCY}" 2>&1 | tee -a "${LOG_PER_SUITE}" | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e

    if [ $rc -ne 0 ]; then
        log "WARN: libero_${SUITE} labeling exited ${rc}. Continuing to next suite anyway."
        log "      (Re-run later or relaunch the watcher to retry; run_label.py resumes from state.)"
        continue
    fi

    if [ -s "${LBL_DIR}/labels.jsonl" ]; then
        n_lbl=$(wc -l < "${LBL_DIR}/labels.jsonl")
        log "  libero_${SUITE}: DONE (${n_lbl} label rows)."
    else
        log "  libero_${SUITE}: labels.jsonl missing or empty after labeling exit 0. Investigate ${LOG_PER_SUITE}."
    fi
done

log ""
log "==== ALL 4 SUITES LABELED ===="
for SUITE in "${SUITES[@]}"; do
    f="${LBL_ROOT}/libero_${SUITE}/labels.jsonl"
    if [ -s "$f" ]; then
        log "  libero_${SUITE}: $(wc -l < "$f") rows"
    else
        log "  libero_${SUITE}: MISSING"
    fi
done
log "SFT watcher (watch_and_run_libero_sft.sh) should now trip on its label-manifest check."

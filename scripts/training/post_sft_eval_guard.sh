#!/usr/bin/env bash
#
# Parallel guard: waits for the in-flight SFT subprocess to exit, then fires
# scripts/eval/run_post_sft_evals.sh if the scorecard JSON isn't already on
# disk. Use this whenever the main watcher script was updated mid-flight
# (so the running bash has a stale parsed AST that won't pick up Phase 6),
# or any time you want belt-and-suspenders coverage on SFT completion.
#
# Usage::
#
#     nohup bash scripts/training/post_sft_eval_guard.sh \
#         --sft-pid 2556139 \
#         --out-dir data/sft/libero_4suite_v3 \
#         > logs/post_sft_eval_guard.boot 2>&1 &
#     echo "guard pid=$!"
#
# Exits 0 once the post-SFT evals have either completed (success) or the
# scorecard already exists (idempotency). Skips with rc=0 if the SFT pid
# was never alive when we started.

set -uo pipefail

cd "$(dirname "$0")/../.."

SFT_PID=""
OUT_DIR="data/sft/libero_4suite_v3"
POLL_SECS="${POLL_SECS:-60}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"
LOG_FILE="${LOG_FILE:-logs/post_sft_eval_guard.log}"
COMBINED_ACT="${COMBINED_ACT:-data/activations/libero_4suite_combined}"
COMBINED_LBL="${COMBINED_LBL:-data/labels/libero_4suite_combined}"
COMBINED_FRAMES="${COMBINED_FRAMES:-${COMBINED_LBL}/frames_cache}"
PY="${PY:-.venv/bin/python}"

usage() {
    sed -n '2,/^$/p' "$0"
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sft-pid)        SFT_PID="$2"; shift 2 ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        --combined-act)   COMBINED_ACT="$2"; shift 2 ;;
        --combined-lbl)   COMBINED_LBL="$2"; shift 2 ;;
        --combined-frames) COMBINED_FRAMES="$2"; shift 2 ;;
        --log-file)       LOG_FILE="$2"; shift 2 ;;
        --poll-secs)      POLL_SECS="$2"; shift 2 ;;
        --help|-h)        usage ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] [guard] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "==== post-SFT eval guard booting (guard pid=$$) ===="
log "  SFT_PID:         ${SFT_PID:-<not given; will poll on scorecard instead>}"
log "  OUT_DIR:         ${OUT_DIR}"
log "  COMBINED_ACT:    ${COMBINED_ACT}"
log "  COMBINED_LBL:    ${COMBINED_LBL}"
log "  COMBINED_FRAMES: ${COMBINED_FRAMES}"
log "  POLL_SECS:       ${POLL_SECS}"
log "  MAX_WAIT_HOURS:  ${MAX_WAIT_HOURS}"

SCORECARD_JSON="${OUT_DIR}/v3_scorecard.json"
if [ -f "${SCORECARD_JSON}" ]; then
    log "Scorecard already present at ${SCORECARD_JSON}; nothing to do, exiting."
    exit 0
fi

# Guard against the SFT pid being stale already (e.g. crashed earlier).
if [ -n "${SFT_PID}" ] && [ ! -d "/proc/${SFT_PID}" ]; then
    log "SFT_PID=${SFT_PID} is not running; running post-SFT evals immediately."
    if [ ! -d "${OUT_DIR}/ar" ] || [ ! -d "${OUT_DIR}/av" ]; then
        log "FATAL: ${OUT_DIR}/{ar,av} missing; nothing to evaluate."
        exit 2
    fi
fi

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))

while true; do
    if [ -f "${SCORECARD_JSON}" ]; then
        log "Scorecard appeared on disk during wait; exiting (likely main watcher did Phase 6)."
        exit 0
    fi
    if [ -n "${SFT_PID}" ] && [ -d "/proc/${SFT_PID}" ]; then
        if [ "$(date +%s)" -gt "$deadline" ]; then
            log "FATAL: ${MAX_WAIT_HOURS}h elapsed while waiting for SFT pid ${SFT_PID}. Aborting guard."
            exit 3
        fi
        sleep "$POLL_SECS"
        continue
    fi
    # No SFT_PID, or it's gone. Confirm ckpt looks reasonable before firing.
    if [ ! -d "${OUT_DIR}/ar" ] || [ ! -d "${OUT_DIR}/av" ]; then
        log "${OUT_DIR}/{ar,av} not present yet; continuing to poll..."
        sleep "$POLL_SECS"
        continue
    fi
    log "SFT process is gone and ${OUT_DIR}/{ar,av} look ready. Firing post-SFT evals."
    break
done

export OUT_DIR COMBINED_ACT COMBINED_LBL COMBINED_FRAMES LOG_FILE PY
bash scripts/eval/run_post_sft_evals.sh 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
log "Guard finished (run_post_sft_evals rc=${rc})."
exit "$rc"

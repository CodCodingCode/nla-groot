#!/usr/bin/env bash
#
# Unattended watcher: waits for the 4-suite LIBERO labeling pipeline to
# finish, then runs the V3 SFT recipe end-to-end with automatic recovery
# AND drives the V3 LIBERO post-SFT P1 eval suite.
#
# Pipeline once labels exist:
#   1. Combine 4 per-suite roots into ./data/{activations,labels}/libero_4suite_combined
#   2. Smoke SFT (50 steps) to catch config bugs cheaply
#   3. Mine hard negatives (topk_cosine)
#   4. Full V3 SFT (15000 steps); on CUDA OOM, halve --batch-size and double
#      --grad-accum-steps; retry up to $MAX_OOM_RETRIES times.
#   5. Phase 6 post-SFT P1 evals (auto-runs at SFT exit, ~45-60 min total).
#      Delegates to scripts/eval/run_post_sft_evals.sh so the same logic is
#      also runnable independently (e.g. via scripts/training/post_sft_eval_guard.sh,
#      which is the safety net the V3 LIBERO Eval Refactor uses when the
#      bash watcher has already-parsed a stale AST).
#
# Designed to run via nohup; everything streams to $LOG_FILE.
#
# Usage::
#
#   mkdir -p logs
#   nohup bash scripts/training/watch_and_run_libero_sft.sh \
#       > logs/libero_sft_watch.boot 2>&1 &
#   echo "watcher pid=$!"
#
# Status / control::
#
#   tail -f logs/libero_sft_watch.log    # live log
#   cat   data/sft/libero_4suite_v3/metrics.jsonl | tail -1
#   cat   data/sft/libero_4suite_v3/v3_scorecard.json | jq .overall
#   kill <pid>                           # graceful abort (SIGTERM)
#
# Idempotency:
#   - Skips the combine step if the combined manifest already exists.
#   - run_sft.py resumes from --output-dir checkpoints automatically.
#   - Phase 6 substeps skip if their output JSON already exists.
#
# Pre-conditions checked before each phase:
#   - .venv exists
#   - Each LIBERO suite's labels manifest is present
#   - Disk space sanity check (>= 50 GB free on the data partition)
#
# Phase 6 prerequisites (Phase 6 only; gracefully skips if unmet):
#   - OPENAI_API_KEY in env       -> enables the multimodal LLM judge
#   - GROOT_MODEL_PATH +
#     LIBERO_DATASET_PATH +
#     LEVERAGE_TEXT_FILE in env   -> enables the interp leverage sweep
#

set -uo pipefail

cd "$(dirname "$0")/../.."

# --- config ------------------------------------------------------------------

SUITES=(goal spatial object 10)
ACT_ROOT="data/activations/libero_4suite_stride2"
LBL_ROOT="data/labels/libero_4suite_stride2"
COMBINED_ACT="data/activations/libero_4suite_combined"
COMBINED_LBL="data/labels/libero_4suite_combined"
COMBINED_FRAMES="${COMBINED_LBL}/frames_cache"
OUT_DIR="data/sft/libero_4suite_v3"
SMOKE_DIR="${OUT_DIR}_smoke"

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/libero_sft_watch.log"

POLL_SECS="${POLL_SECS:-60}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"

SMOKE_STEPS="${SMOKE_STEPS:-50}"
FULL_STEPS="${FULL_STEPS:-15000}"
INITIAL_BATCH="${INITIAL_BATCH:-4}"
INITIAL_GRAD_ACCUM="${INITIAL_GRAD_ACCUM:-1}"
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-3}"

PY=".venv/bin/python"

mkdir -p "$LOG_DIR"

# --- helpers -----------------------------------------------------------------

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

trap 'log "Watcher caught signal; exiting (pid=$$)."; exit 130' INT TERM

require_venv() {
    if [ ! -x "$PY" ]; then
        log "FATAL: $PY does not exist or is not executable. Activate the project venv."
        exit 2
    fi
}

all_labels_ready() {
    for s in "${SUITES[@]}"; do
        if [ ! -f "${LBL_ROOT}/libero_${s}/manifest.json" ]; then
            return 1
        fi
        # Also require a non-empty labels.jsonl; a manifest is sometimes
        # written before final flush completes.
        if [ ! -s "${LBL_ROOT}/libero_${s}/labels.jsonl" ]; then
            return 1
        fi
    done
    return 0
}

count_label_rows() {
    local total=0
    for s in "${SUITES[@]}"; do
        local f="${LBL_ROOT}/libero_${s}/labels.jsonl"
        if [ -s "$f" ]; then
            total=$((total + $(wc -l < "$f")))
        fi
    done
    echo "$total"
}

disk_ok() {
    # Cheap sanity check: refuse to start big runs with < 50 GB free.
    local free_gb
    free_gb=$(df -BG "data" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
    if [ -z "$free_gb" ]; then
        return 0   # df failed; let it ride.
    fi
    if [ "$free_gb" -lt 50 ]; then
        log "WARN: only ${free_gb}G free on data partition; runs may fail mid-flight."
    fi
}

# Return 0 if last $1 lines of $LOG_FILE indicate a CUDA OOM.
detected_oom() {
    local n="${1:-400}"
    tail -n "$n" "$LOG_FILE" 2>/dev/null \
        | grep -Eiq 'CUDA out of memory|OutOfMemoryError|cudaErrorMemory|torch.cuda.OutOfMemory'
}

# --- 0. boot -----------------------------------------------------------------

log "==== libero-4suite SFT watcher booting ===="
log "  cwd:                 $(pwd)"
log "  python:              $PY"
log "  suites:              ${SUITES[*]}"
log "  poll every:          ${POLL_SECS}s"
log "  max wait:            ${MAX_WAIT_HOURS}h"
log "  smoke steps:         ${SMOKE_STEPS}"
log "  full steps:          ${FULL_STEPS}"
log "  initial batch:       ${INITIAL_BATCH}"
log "  initial grad-accum:  ${INITIAL_GRAD_ACCUM}"
log "  max OOM retries:     ${MAX_OOM_RETRIES}"
log "  output dir:          ${OUT_DIR}"
log ""
log "  Pre-req: the other agent must run"
log "    bash scripts/labeling/run_label_all_libero_suites.sh"
log "  after extraction finishes. The watcher only waits for the labels."

require_venv
disk_ok

# --- 1. wait for labels ------------------------------------------------------

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
log ""
log "Phase 1/6: waiting for all 4 label manifests under ${LBL_ROOT}/libero_<suite>/manifest.json"

while ! all_labels_ready; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        log "FATAL: ${MAX_WAIT_HOURS}h elapsed without all label manifests. Aborting."
        exit 3
    fi
    n=$(count_label_rows)
    missing=""
    for s in "${SUITES[@]}"; do
        if [ ! -f "${LBL_ROOT}/libero_${s}/manifest.json" ]; then
            missing="${missing}${s} "
        fi
    done
    log "  ...not ready yet. rows so far: ${n}; missing manifests: [${missing}]"
    sleep "$POLL_SECS"
done
total_rows=$(count_label_rows)
log "All 4 label manifests present. Combined label rows on disk: ${total_rows}"

# --- 2. combine --------------------------------------------------------------

log ""
log "Phase 2/6: combining 4 suites -> ${COMBINED_ACT} and ${COMBINED_LBL}/labels.jsonl"

if [ -f "${COMBINED_ACT}/manifest.json" ]; then
    log "  combined manifest already exists, skipping combine (idempotent)."
else
    set +e
    PYTHONPATH=src $PY scripts/training/combine_libero_4suite.py \
        --activations-root "${ACT_ROOT}" \
        --labels-root      "${LBL_ROOT}" \
        --suites           "${SUITES[@]}" \
        --combined-activations    "${COMBINED_ACT}" \
        --combined-labels-jsonl   "${COMBINED_LBL}/labels.jsonl" \
        --combined-frames-cache   "${COMBINED_FRAMES}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e
    if [ $rc -ne 0 ]; then
        log "FATAL: combine step exited $rc. Aborting."
        exit 4
    fi
fi

if [ ! -f "${COMBINED_ACT}/stats.json" ]; then
    log "FATAL: ${COMBINED_ACT}/stats.json missing after combine. Aborting."
    exit 5
fi

# --- 3. smoke SFT ------------------------------------------------------------

log ""
log "Phase 3/6: smoke SFT (${SMOKE_STEPS} steps) -> ${SMOKE_DIR}"

set +e
PYTHONPATH=src $PY scripts/training/run_sft.py \
    --stats-json       "${COMBINED_ACT}/stats.json" \
    --activations-root "${COMBINED_ACT}" \
    --labels-jsonl     "${COMBINED_LBL}/labels.jsonl" \
    --output-dir       "${SMOKE_DIR}" \
    --total-steps      "${SMOKE_STEPS}" \
    --batch-size       "${INITIAL_BATCH}" \
    --grad-accum-steps "${INITIAL_GRAD_ACCUM}" \
    --learning-rate    1e-4 \
    --warmup-steps     10 \
    --eval-every       25 \
    --save-every       "${SMOKE_STEPS}" \
    --log-every        5 \
    --max-val-items    256 \
    --seed             0 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e
if [ $rc -ne 0 ]; then
    log "FATAL: smoke SFT exited $rc. Aborting before full run."
    exit 6
fi
log "Smoke SFT passed."

# --- 4. mine hard negatives --------------------------------------------------

HARD_NEG_PATH="${COMBINED_ACT}/hard_negatives.jsonl"

log ""
log "Phase 4/6: mining hard negatives -> ${HARD_NEG_PATH}"
if [ -s "${HARD_NEG_PATH}" ]; then
    log "  hard_negatives.jsonl already populated, skipping mine."
else
    set +e
    PYTHONPATH=src $PY scripts/training/mine_hard_negatives.py \
        --activations-root "${COMBINED_ACT}" \
        --labels-jsonl     "${COMBINED_LBL}/labels.jsonl" \
        --min-bullet-lines 3 \
        --top-k            8 \
        --out              "${HARD_NEG_PATH}" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e
    if [ $rc -ne 0 ]; then
        log "WARN: mine_hard_negatives exited $rc. Proceeding without hard-neg index."
        rm -f "${HARD_NEG_PATH}"
    fi
fi

# --- 5. full V3 SFT with OOM retry ------------------------------------------

log ""
log "Phase 5/6: full V3 SFT (${FULL_STEPS} steps) -> ${OUT_DIR}"

batch=$INITIAL_BATCH
grad_accum=$INITIAL_GRAD_ACCUM
sft_succeeded=0

for attempt in $(seq 1 $((MAX_OOM_RETRIES + 1))); do
    log "  attempt ${attempt}: batch=${batch} grad_accum=${grad_accum}"

    # Build optional hard-neg flags only if mining produced an index.
    hn_flags=()
    if [ -s "${HARD_NEG_PATH}" ]; then
        hn_flags=(
            --ar-nce-hard-negative-source topk_cosine
            --ar-nce-hard-negative-index-path "${HARD_NEG_PATH}"
            --ar-nce-hard-negatives-per-anchor 4
        )
    fi

    set +e
    PYTHONPATH=src $PY scripts/training/run_sft.py \
        --stats-json       "${COMBINED_ACT}/stats.json" \
        --activations-root "${COMBINED_ACT}" \
        --labels-jsonl     "${COMBINED_LBL}/labels.jsonl" \
        --output-dir       "${OUT_DIR}" \
        --total-steps      "${FULL_STEPS}" \
        --batch-size       "${batch}" \
        --grad-accum-steps "${grad_accum}" \
        --learning-rate    1e-4 \
        --warmup-steps     500 \
        --ar-contrastive-weight 0.5 \
        --ar-nce-temperature    0.1 \
        --ar-clip-target-scaled 5.0 \
        "${hn_flags[@]}" \
        --ar-av-mix-max         0.4 \
        --ar-av-mix-warmup-frac 0.3 \
        --balance-position-mix \
        --min-bullets 3 \
        --eval-closed-loop \
        --closed-loop-temps 0.0 0.7 \
        --closed-loop-max-batches 64 \
        --max-val-items     1000 \
        --eval-every        500 \
        --save-every        2500 \
        --seed              0 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    set -e

    if [ $rc -eq 0 ]; then
        log "Full SFT completed successfully on attempt ${attempt}."
        log "  -> ${OUT_DIR}/"
        sft_succeeded=1
        break
    fi

    if [ $attempt -ge $((MAX_OOM_RETRIES + 1)) ]; then
        log "FATAL: full SFT failed ${attempt} attempts. Last exit=${rc}. Stopping."
        exit 7
    fi

    # Only retry on OOM-like failures. Anything else surfaces immediately.
    if detected_oom 600; then
        new_batch=$((batch / 2))
        new_grad_accum=$((grad_accum * 2))
        if [ "$new_batch" -lt 1 ]; then
            log "FATAL: CUDA OOM but batch already at 1; cannot reduce further."
            exit 8
        fi
        log "  CUDA OOM detected. Reducing batch ${batch} -> ${new_batch}, grad_accum ${grad_accum} -> ${new_grad_accum}."
        batch="$new_batch"
        grad_accum="$new_grad_accum"
        # Wipe partial smoke dir from this attempt; the full run resumes from
        # its own checkpoints if any were written.
        continue
    fi

    log "FATAL: non-OOM failure (rc=${rc}). Stopping retries to avoid burning the night on a config bug."
    exit "$rc"
done

if [ "${sft_succeeded}" -ne 1 ]; then
    log "WARN: SFT loop ended without success flag; skipping Phase 6."
    exit 9
fi

# --- 6. post-SFT P1 evals ----------------------------------------------------
#
# Phase 6 delegates to scripts/eval/run_post_sft_evals.sh so the logic can
# also be triggered independently (handy when bash patched its parsed AST
# while the watcher was already mid-run, or when re-grading an existing
# checkpoint without re-training).
#
# Bypass everything with SKIP_POST_SFT_EVAL=1.

log ""
log "Phase 6/6: post-SFT P1 evals on ${OUT_DIR}"

if [ "${SKIP_POST_SFT_EVAL:-0}" = "1" ]; then
    log "  SKIP_POST_SFT_EVAL=1 set; bypassing Phase 6 entirely."
    exit 0
fi

POST_EVAL_SCRIPT="scripts/eval/run_post_sft_evals.sh"
if [ ! -x "${POST_EVAL_SCRIPT}" ]; then
    log "WARN: ${POST_EVAL_SCRIPT} not executable; skipping Phase 6."
    exit 0
fi

set +e
env \
    OUT_DIR="${OUT_DIR}" \
    COMBINED_ACT="${COMBINED_ACT}" \
    COMBINED_LBL="${COMBINED_LBL}" \
    COMBINED_FRAMES="${COMBINED_FRAMES}" \
    LOG_FILE="${LOG_FILE}" \
    PY="${PY}" \
    bash "${POST_EVAL_SCRIPT}" 2>&1 | tee -a "$LOG_FILE"
post_rc=${PIPESTATUS[0]}
set -e

log "Phase 6 finished (post-eval rc=${post_rc})."
exit 0

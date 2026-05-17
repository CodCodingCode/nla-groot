#!/usr/bin/env bash
#
# Standalone Phase-6 post-SFT eval runner.
#
# Invoked by:
#   - scripts/training/watch_and_run_libero_sft.sh (Phase 6 block, when bash
#     successfully picked up the wired version)
#   - scripts/training/post_sft_eval_guard.sh (parallel guard that fires this
#     when the main watcher exited before it could pick up Phase 6)
#   - manually, when re-grading an existing SFT checkpoint
#
# Required env vars (the watcher exports all of these for you):
#   OUT_DIR              SFT run directory containing av/ and ar/.
#   COMBINED_ACT         data/activations/libero_4suite_combined root.
#   COMBINED_LBL         data/labels/libero_4suite_combined root.
#   COMBINED_FRAMES      ${COMBINED_LBL}/frames_cache.
#   LOG_FILE             where to tee progress.
#   PY                   path to the venv python (default: .venv/bin/python).
#
# Optional knobs (sensible defaults baked in):
#   SKIP_POST_SFT_EVAL   set =1 to bail out before running anything.
#   EVAL_RETRIEVAL_N     total val items for the retrieval margin (default 256).
#   EVAL_JUDGE_PER_POSITION  per-position N for the LLM judge (default 12).
#   EVAL_JUDGE_CONCURRENCY   OpenAI concurrency (default 8).
#   GROOT_MODEL_PATH         GR00T policy checkpoint; gates the leverage sweep.
#   LIBERO_DATASET_PATH      LIBERO LeRobot dataset root; gates the leverage sweep.
#   LIBERO_EMBODIMENT_TAG    default LIBERO_PANDA.
#   LEVERAGE_TEXT_FILE       bullet-list text file to inject; gates the leverage sweep.
#
# Each substep is best-effort: a missing prereq or non-zero exit is logged
# and skipped so the next substep can still run, and the scorecard is always
# written at the end so the final-status pointer is well-defined.

set -uo pipefail

# Resolve to repo root so relative paths line up regardless of cwd.
cd "$(dirname "$0")/../.."

OUT_DIR="${OUT_DIR:?OUT_DIR must be set (e.g. data/sft/libero_4suite_v3)}"
COMBINED_ACT="${COMBINED_ACT:-data/activations/libero_4suite_combined}"
COMBINED_LBL="${COMBINED_LBL:-data/labels/libero_4suite_combined}"
COMBINED_FRAMES="${COMBINED_FRAMES:-${COMBINED_LBL}/frames_cache}"
LOG_FILE="${LOG_FILE:-logs/libero_sft_watch.log}"
PY="${PY:-.venv/bin/python}"

EVAL_RETRIEVAL_N="${EVAL_RETRIEVAL_N:-256}"
EVAL_JUDGE_PER_POSITION="${EVAL_JUDGE_PER_POSITION:-12}"
EVAL_JUDGE_CONCURRENCY="${EVAL_JUDGE_CONCURRENCY:-8}"

GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-}"
LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-}"
LIBERO_EMBODIMENT_TAG="${LIBERO_EMBODIMENT_TAG:-LIBERO_PANDA}"
LEVERAGE_TEXT_FILE="${LEVERAGE_TEXT_FILE:-}"

mkdir -p "$(dirname "$LOG_FILE")"

eval_log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] [post-sft-eval] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

run_eval_step() {
    # Wraps one eval substep: prints a header, runs the command, logs the
    # outcome but never aborts the runner on non-zero.
    local label="$1"; shift
    eval_log "==== ${label} ===="
    eval_log "     \$ $*"
    set +e
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
        eval_log "${label} OK"
    else
        eval_log "${label} FAILED rc=${rc} (continuing to next eval step)"
    fi
    return 0
}

eval_log "==== run_post_sft_evals starting on ${OUT_DIR} ===="
eval_log "  COMBINED_ACT:    ${COMBINED_ACT}"
eval_log "  COMBINED_LBL:    ${COMBINED_LBL}"
eval_log "  COMBINED_FRAMES: ${COMBINED_FRAMES}"
eval_log "  PY:              ${PY}"

if [ "${SKIP_POST_SFT_EVAL:-0}" = "1" ]; then
    eval_log "SKIP_POST_SFT_EVAL=1 set; bailing out."
    exit 0
fi

if [ ! -d "${OUT_DIR}/ar" ] || [ ! -d "${OUT_DIR}/av" ]; then
    eval_log "FATAL: ${OUT_DIR}/{ar,av} missing; cannot run post-SFT evals."
    exit 2
fi

# 6a. retrieval margin -------------------------------------------------------

RETRIEVAL_JSON="${OUT_DIR}/retrieval_margin.json"
RETRIEVAL_JSONL="${OUT_DIR}/retrieval_per_sample.jsonl"

if [ -f "${RETRIEVAL_JSON}" ]; then
    eval_log "6a retrieval_margin: ${RETRIEVAL_JSON} already exists, skipping."
else
    run_eval_step "6a retrieval_margin" \
        env PYTHONPATH=src $PY scripts/eval/closed_loop_retrieval.py \
            --ckpt-dir         "${OUT_DIR}" \
            --activations-root "${COMBINED_ACT}" \
            --labels-jsonl     "${COMBINED_LBL}/labels.jsonl" \
            --n-samples        "${EVAL_RETRIEVAL_N}" \
            --temperature      0.0 \
            --batch-size       8 \
            --out-json         "${RETRIEVAL_JSON}" \
            --out-jsonl        "${RETRIEVAL_JSONL}"
fi

# 6b. side-by-side dump (eyeball companion) ----------------------------------

AV_SAMPLES_JSONL="${OUT_DIR}/av_samples.jsonl"

if [ -f "${AV_SAMPLES_JSONL}" ]; then
    eval_log "6b dump_av_samples: ${AV_SAMPLES_JSONL} already exists, skipping."
else
    run_eval_step "6b dump_av_samples" \
        env PYTHONPATH=src $PY scripts/eval/dump_av_samples.py \
            --ckpt-dir         "${OUT_DIR}" \
            --activations-root "${COMBINED_ACT}" \
            --labels-jsonl     "${COMBINED_LBL}/labels.jsonl" \
            --per-position     6 \
            --temperatures     0.0 0.7 \
            --out-jsonl        "${AV_SAMPLES_JSONL}"
fi

# 6c. multimodal LLM judge ---------------------------------------------------

JUDGE_JSONL="${OUT_DIR}/llm_judge.jsonl"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    eval_log "==== 6c llm_judge ===="
    eval_log "SKIPPED: OPENAI_API_KEY not set in environment."
elif [ ! -d "${COMBINED_FRAMES}" ]; then
    eval_log "==== 6c llm_judge ===="
    eval_log "SKIPPED: frames cache ${COMBINED_FRAMES} not present."
else
    # The judge resumes from existing JSONL rows, so re-invoking is cheap.
    run_eval_step "6c llm_judge" \
        env PYTHONPATH=src $PY scripts/eval/llm_judge_av_captions.py \
            --ckpt-dir         "${OUT_DIR}" \
            --activations-root "${COMBINED_ACT}" \
            --labels-jsonl     "${COMBINED_LBL}/labels.jsonl" \
            --frames-cache     "${COMBINED_FRAMES}" \
            --video-keys       image wrist_image \
            --per-position     "${EVAL_JUDGE_PER_POSITION}" \
            --concurrency      "${EVAL_JUDGE_CONCURRENCY}" \
            --temperature      0.0 \
            --out-jsonl        "${JUDGE_JSONL}"
fi

# 6d. interpretability leverage sweep ----------------------------------------

LEVERAGE_JSONL="${OUT_DIR}/leverage_sweep.jsonl"

if [ -f "${LEVERAGE_JSONL}" ]; then
    eval_log "6d leverage_sweep: ${LEVERAGE_JSONL} already exists, skipping."
elif [ -z "${GROOT_MODEL_PATH}" ] || [ -z "${LIBERO_DATASET_PATH}" ] || [ -z "${LEVERAGE_TEXT_FILE}" ]; then
    eval_log "==== 6d leverage_sweep ===="
    eval_log "SKIPPED: set GROOT_MODEL_PATH, LIBERO_DATASET_PATH, and LEVERAGE_TEXT_FILE env vars to enable."
elif [ ! -d "${GROOT_MODEL_PATH}" ]; then
    eval_log "==== 6d leverage_sweep ===="
    eval_log "SKIPPED: GROOT_MODEL_PATH=${GROOT_MODEL_PATH} does not exist."
elif [ ! -d "${LIBERO_DATASET_PATH}" ]; then
    eval_log "==== 6d leverage_sweep ===="
    eval_log "SKIPPED: LIBERO_DATASET_PATH=${LIBERO_DATASET_PATH} does not exist."
elif [ ! -f "${LEVERAGE_TEXT_FILE}" ]; then
    eval_log "==== 6d leverage_sweep ===="
    eval_log "SKIPPED: LEVERAGE_TEXT_FILE=${LEVERAGE_TEXT_FILE} does not exist."
else
    run_eval_step "6d leverage_sweep" \
        env PYTHONPATH=src $PY scripts/eval/nla_steer_leverage_sweep.py \
            --model-path       "${GROOT_MODEL_PATH}" \
            --dataset-path     "${LIBERO_DATASET_PATH}" \
            --embodiment-tag   "${LIBERO_EMBODIMENT_TAG}" \
            --ar-dir           "${OUT_DIR}/ar" \
            --text-file        "${LEVERAGE_TEXT_FILE}" \
            --traj-id          0 \
            --step             0 \
            --placements       "image_patch,last_text,anchor" \
            --null-samples     4 \
            --out-jsonl        "${LEVERAGE_JSONL}"
fi

# 6e. unified scorecard ------------------------------------------------------

SCORECARD_JSON="${OUT_DIR}/v3_scorecard.json"

run_eval_step "6e build_v3_scorecard" \
    env PYTHONPATH=src $PY scripts/eval/build_v3_scorecard.py \
        --ckpt-dir  "${OUT_DIR}" \
        --out-json  "${SCORECARD_JSON}"

# Final headline.
if [ -f "${SCORECARD_JSON}" ]; then
    set +e
    OVERALL=$($PY -c "import json,sys; print(json.load(open('${SCORECARD_JSON}')).get('overall','?'))" 2>/dev/null)
    set -e
    eval_log ""
    eval_log "==================================================================="
    eval_log "FINAL: V3 LIBERO SCORECARD overall = ${OVERALL:-?}"
    eval_log "  -> ${SCORECARD_JSON}"
    eval_log "==================================================================="
else
    eval_log "WARN: scorecard JSON not produced at ${SCORECARD_JSON}; check the eval logs above."
fi

exit 0

#!/usr/bin/env bash
#
# V4 post-SFT eval wrapper.
#
# Wraps every existing scripts/eval/* tool into one chained pipeline whose
# output directory (``<SFT_DIR>/post_sft_eval/``) is everything a scorecard
# generator needs. This is the V4 successor to ``run_post_sft_evals.sh``;
# unlike that script (which spreads artifacts at the SFT-dir root and is
# best-effort-continue-on-failure), this wrapper:
#
#   1. Collects ALL artifacts under one ``post_sft_eval/`` subdir.
#   2. HALTS on the first HARD failure (skipped steps don't count as hard).
#   3. Has a ``--dry-run`` flag that prints every command it would run.
#   4. Tolerates a no-GPU box (closed-loop / sim steps WARN+SKIP).
#   5. Writes a one-line PASS/WARN/FAIL ``SUMMARY.txt`` greppable by CI.
#
# Usage:
#   scripts/eval/run_post_sft_evals_v4.sh <SFT_DIR> [--dry-run]
#
# Example:
#   scripts/eval/run_post_sft_evals_v4.sh \
#       data/sft/libero_4suite_v4_consistency_overnight
#
# Optional env-var knobs (sensible defaults baked in).
# Repo .env is sourced when present for keys like OPENAI_API_KEY / GROOT_MODEL_PATH.
#   PY                       venv python (default: .venv/bin/python).
#   ACTIVATIONS_ROOT         default: data/activations/libero_4suite_v4_combined.
#   LABELS_JSONL             default: data/labels/libero_4suite_v4_combined/labels.jsonl.
#   FRAMES_CACHE             default: data/labels/libero_4suite_combined/frames_cache
#                            (V4 re-uses the V3 cache verbatim).
#   EVAL_RETRIEVAL_N         default 256.
#   EVAL_JUDGE_PER_POSITION  default 12.
#   EVAL_JUDGE_CONCURRENCY   default 8.
#   GROOT_MODEL_PATH         enables step (e) sim A/B when set + dir exists.
#   LIBERO_EMBODIMENT_TAG    default LIBERO_PANDA.
#   SIM_EPISODES_PER_ARM     default 10.
#   SIM_N_ENVS               default 5.
#   SIM_PORT                 default 5577.
#   FORCE_NO_GPU=1           treat the box as CPU-only (forces skips).
#   SKIP_GRPO_ORCH_HINTS=1   do not write post_sft_eval/grpo_orchestrator_hints.sh.
#
# The contract:
#   - Exit 0 = wrapper ran end-to-end (some soft skips are fine).
#   - Exit 2 = SFT_DIR missing / unusable.
#   - Exit 3 = a hard step (retrieval, AV dump, scorecard) failed; SUMMARY.txt
#             is still written with OVERALL=FAIL and the failing step labelled.
#   - Exit 64 = bad CLI args.

set -uo pipefail

# Resolve to repo root so relative paths line up regardless of cwd.
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env" || true
    set +a
fi

# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------

DRY_RUN=0
SFT_DIR=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,45p' "$0"
            exit 0
            ;;
        --*)
            echo "ERROR: unknown flag $arg" >&2
            exit 64
            ;;
        *)
            if [ -z "$SFT_DIR" ]; then
                SFT_DIR="$arg"
            else
                echo "ERROR: only one SFT_DIR positional arg allowed; got $arg" >&2
                exit 64
            fi
            ;;
    esac
done

if [ -z "$SFT_DIR" ]; then
    echo "ERROR: SFT_DIR positional arg required." >&2
    echo "Usage: $0 <SFT_DIR> [--dry-run]" >&2
    exit 64
fi

# Trim trailing slash for prettier paths.
SFT_DIR="${SFT_DIR%/}"

if [ ! -d "$SFT_DIR" ]; then
    echo "FATAL: SFT_DIR does not exist: $SFT_DIR" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PY="${PY:-.venv/bin/python}"
ACTIVATIONS_ROOT="${ACTIVATIONS_ROOT:-data/activations/libero_4suite_v4_combined}"
LABELS_JSONL="${LABELS_JSONL:-data/labels/libero_4suite_v4_combined/labels.jsonl}"
FRAMES_CACHE="${FRAMES_CACHE:-data/labels/libero_4suite_combined/frames_cache}"

EVAL_RETRIEVAL_N="${EVAL_RETRIEVAL_N:-256}"
EVAL_JUDGE_PER_POSITION="${EVAL_JUDGE_PER_POSITION:-12}"
EVAL_JUDGE_CONCURRENCY="${EVAL_JUDGE_CONCURRENCY:-8}"

GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-}"
if [[ -z "${GROOT_MODEL_PATH}" ]] && [[ -d "${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_goal" ]]; then
    GROOT_MODEL_PATH="${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_goal"
fi
if [[ -n "${GROOT_MODEL_PATH}" && "${GROOT_MODEL_PATH}" != /* ]]; then
    GROOT_MODEL_PATH="${REPO_ROOT}/${GROOT_MODEL_PATH#./}"
fi
if [[ -z "${GROOT_MODEL_PATH}" ]] && [[ -d "${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_goal" ]]; then
    GROOT_MODEL_PATH="${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_goal"
fi
if [[ -n "${GROOT_MODEL_PATH}" && "${GROOT_MODEL_PATH}" != /* ]]; then
    GROOT_MODEL_PATH="${REPO_ROOT}/${GROOT_MODEL_PATH#./}"
fi
if [[ -z "${GROOT_MODEL_PATH}" ]] && [[ -d "${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_goal" ]]; then
    GROOT_MODEL_PATH="${REPO_ROOT}/checkpoints/GR00T-N1.7-LIBERO/libero_goal"
fi
if [[ -n "${GROOT_MODEL_PATH}" && "${GROOT_MODEL_PATH}" != /* ]]; then
    GROOT_MODEL_PATH="${REPO_ROOT}/${GROOT_MODEL_PATH#./}"
fi
LIBERO_EMBODIMENT_TAG="${LIBERO_EMBODIMENT_TAG:-LIBERO_PANDA}"
SIM_EPISODES_PER_ARM="${SIM_EPISODES_PER_ARM:-10}"
SIM_N_ENVS="${SIM_N_ENVS:-5}"
SIM_PORT="${SIM_PORT:-5577}"

OUT_DIR="${SFT_DIR}/post_sft_eval"
LOG_PATH="${OUT_DIR}/run.log"
SUMMARY_PATH="${OUT_DIR}/SUMMARY.txt"

EXTRACTION_SWEEP_DIR="${SFT_DIR}/extraction_sweep"
EXTRACTION_DIAG="${SFT_DIR}/extraction_diag.json"
METRICS_JSONL="${SFT_DIR}/metrics.jsonl"

EXTRACTION_SCORECARD="${OUT_DIR}/extraction_scorecard.json"
RETRIEVAL_JSON="${OUT_DIR}/retrieval_margin.json"
RETRIEVAL_JSONL="${OUT_DIR}/retrieval_per_sample.jsonl"
AV_SAMPLES_JSONL="${OUT_DIR}/av_samples.jsonl"
JUDGE_JSONL="${OUT_DIR}/llm_judge.jsonl"
SIM_AB_JSON="${OUT_DIR}/sim_ab.json"
SCORECARD_JSON="${OUT_DIR}/v4_sft_scorecard.json"

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$OUT_DIR"
fi

# ---------------------------------------------------------------------------
# Logging + step runners
# ---------------------------------------------------------------------------

log() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] [v4-post-sft] $*"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$msg"
    else
        echo "$msg"
        echo "$msg" >> "$LOG_PATH"
    fi
}

# Pretty-print the command we would run/are running, with each arg on its
# own line for readability (matches the existing run_post_sft_evals.sh
# look-and-feel).
log_cmd() {
    local prefix="     $"
    local first=1
    for tok in "$@"; do
        if [ "$first" -eq 1 ]; then
            log "${prefix} ${tok} \\"
            first=0
        else
            log "       ${tok} \\"
        fi
    done
}

# Hard-failing step: any non-zero exit aborts the whole wrapper after
# writing a FAIL SUMMARY.txt that names the failing step.
run_hard() {
    local label="$1"; shift
    log "==== ${label} (HARD) ===="
    log_cmd "$@"
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi
    set +e
    "$@" 2>&1 | tee -a "$LOG_PATH"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        log "${label} FAILED rc=${rc}; halting V4 post-SFT eval."
        # Best-effort summary so callers don't have to guess.
        echo "OVERALL=FAIL reason=${label}_failed_rc=${rc}" > "$SUMMARY_PATH"
        exit 3
    fi
    log "${label} OK"
}

# Soft step: log the SKIP reason and continue. Used for env-conditional
# stages (no GPU, no API key, no GR00T checkpoint, ...).
skip_step() {
    local label="$1"; shift
    log "==== ${label} (SKIP) ===="
    log "  WARN: $*"
}

# Conditional hard step: if the precondition function returns 0, run hard;
# otherwise log a SKIP with the given reason. This is the workhorse for
# GPU-gated stages.
run_if() {
    local cond_label="$1"; shift
    local cond_status="$1"; shift   # 0 = run, anything else = skip
    local skip_reason="$1"; shift
    local step_label="$1"; shift
    if [ "$cond_status" -eq 0 ]; then
        run_hard "$step_label" "$@"
    else
        skip_step "$step_label" "$skip_reason"
    fi
}

# ---------------------------------------------------------------------------
# Environment probes (cached so we report identical state in dry-run)
# ---------------------------------------------------------------------------

HAS_GPU=1   # 1 = no, 0 = yes (matches shell convention: 0 = success)
if [ "${FORCE_NO_GPU:-0}" = "1" ]; then
    GPU_STATUS_MSG="forced off via FORCE_NO_GPU=1"
elif [ ! -x "$PY" ]; then
    GPU_STATUS_MSG="python not found at $PY; assuming no GPU"
else
    # Use the python probe (it catches the CUDA_VISIBLE_DEVICES="" mask
    # case that ``nvidia-smi`` won't notice). Probe runs even in dry-run
    # so the planned command list reflects real env state.
    if PYTHONPATH=src "$PY" scripts/eval/_post_sft_evals_helpers.py check_gpu \
            >/dev/null 2>&1; then
        HAS_GPU=0
        GPU_STATUS_MSG="cuda available (torch.cuda.is_available()==True)"
    else
        GPU_STATUS_MSG="no usable cuda device (torch probe failed)"
    fi
fi

HAS_OPENAI_KEY=1
if [ -n "${OPENAI_API_KEY:-}" ]; then
    HAS_OPENAI_KEY=0
fi

HAS_FRAMES_CACHE=1
if [ -d "$FRAMES_CACHE" ]; then
    HAS_FRAMES_CACHE=0
fi

HAS_GROOT_MODEL=1
if [ -n "$GROOT_MODEL_PATH" ] && [ -d "$GROOT_MODEL_PATH" ]; then
    HAS_GROOT_MODEL=0
fi

HAS_LABELS=1
if [ -f "$LABELS_JSONL" ]; then
    HAS_LABELS=0
fi

HAS_ACTIVATIONS=1
if [ -d "$ACTIVATIONS_ROOT" ]; then
    HAS_ACTIVATIONS=0
fi

HAS_AR_AV=1
if [ -d "${SFT_DIR}/ar" ] && [ -d "${SFT_DIR}/av" ]; then
    HAS_AR_AV=0
fi

HAS_EXTRACTION_SWEEP=1
if [ -d "$EXTRACTION_SWEEP_DIR" ]; then
    HAS_EXTRACTION_SWEEP=0
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

log "==== V4 post-SFT eval wrapper starting on ${SFT_DIR} ===="
log "  PY:                  ${PY}"
log "  OUT_DIR:             ${OUT_DIR}"
log "  ACTIVATIONS_ROOT:    ${ACTIVATIONS_ROOT} [$( [ $HAS_ACTIVATIONS -eq 0 ] && echo present || echo MISSING )]"
log "  LABELS_JSONL:        ${LABELS_JSONL} [$( [ $HAS_LABELS -eq 0 ] && echo present || echo MISSING )]"
log "  FRAMES_CACHE:        ${FRAMES_CACHE} [$( [ $HAS_FRAMES_CACHE -eq 0 ] && echo present || echo MISSING )]"
log "  ar/, av/ subdirs:    [$( [ $HAS_AR_AV -eq 0 ] && echo present || echo MISSING )]"
log "  extraction_sweep:    [$( [ $HAS_EXTRACTION_SWEEP -eq 0 ] && echo present || echo absent )]"
log "  GPU:                 ${GPU_STATUS_MSG}"
log "  OPENAI_API_KEY:      $( [ $HAS_OPENAI_KEY -eq 0 ] && echo set || echo UNSET )"
log "  GROOT_MODEL_PATH:    ${GROOT_MODEL_PATH:-<unset>} [$( [ $HAS_GROOT_MODEL -eq 0 ] && echo present || echo missing/unset )]"
log "  dry-run:             ${DRY_RUN}"

# ---------------------------------------------------------------------------
# Step (a): recon / extraction scorecard
# ---------------------------------------------------------------------------
#
# There is no separate "FVE eval" script in this repo: the V3 closed-loop
# cosine (a.k.a. closed_greedy_cosine, a FVE-style scalar) is recorded
# inside training metrics.jsonl and surfaced by the final V3 scorecard.
# What CAN exist as a standalone artifact is the V4 extraction sweep
# scorecard (the layer x position-strategy proxy ranking). If an
# extraction_sweep/ dir is present, we roll it up here; otherwise we log
# a SKIP and let step (f) pick up FVE-style signal from metrics.jsonl.

if [ "$HAS_EXTRACTION_SWEEP" -eq 0 ]; then
    EXTRACTION_DIAG_ARG=( )
    if [ -f "$EXTRACTION_DIAG" ]; then
        EXTRACTION_DIAG_ARG=( --diag-json "$EXTRACTION_DIAG" )
    fi
    run_hard "a/recon: build_v4_extraction_scorecard" \
        env PYTHONPATH=src "$PY" scripts/eval/build_v4_extraction_scorecard.py \
            --sweep-root          "$EXTRACTION_SWEEP_DIR" \
            "${EXTRACTION_DIAG_ARG[@]}" \
            --out-json            "$EXTRACTION_SCORECARD" \
            --bootstrap-resamples 100
else
    skip_step "a/recon: build_v4_extraction_scorecard" \
        "no ${EXTRACTION_SWEEP_DIR}/ dir; FVE-style signal will be picked up from metrics.jsonl in step (f)."
fi

# ---------------------------------------------------------------------------
# Step (b): retrieval margin
# ---------------------------------------------------------------------------
#
# GPU-only (closed_loop_retrieval.py needs AV.generate + AR.forward on
# real activations). The V3 retrieval-margin artifact this produces is
# the canonical "is anti-collapse working?" signal consumed by the final
# scorecard.

RETRIEVAL_COND=1
RETRIEVAL_REASON=""
if [ "$HAS_AR_AV" -ne 0 ]; then
    RETRIEVAL_REASON="missing ar/ or av/ subdir in $SFT_DIR"
elif [ "$HAS_GPU" -ne 0 ]; then
    RETRIEVAL_REASON="no GPU available (${GPU_STATUS_MSG})"
elif [ "$HAS_ACTIVATIONS" -ne 0 ]; then
    RETRIEVAL_REASON="ACTIVATIONS_ROOT not found: $ACTIVATIONS_ROOT"
elif [ "$HAS_LABELS" -ne 0 ]; then
    RETRIEVAL_REASON="LABELS_JSONL not found: $LABELS_JSONL"
else
    RETRIEVAL_COND=0
fi

run_if "retrieval-precond" "$RETRIEVAL_COND" "$RETRIEVAL_REASON" \
    "b/retrieval: closed_loop_retrieval" \
    env PYTHONPATH=src "$PY" scripts/eval/closed_loop_retrieval.py \
        --ckpt-dir         "$SFT_DIR" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --labels-jsonl     "$LABELS_JSONL" \
        --n-samples        "$EVAL_RETRIEVAL_N" \
        --temperature      0.0 \
        --batch-size       8 \
        --out-json         "$RETRIEVAL_JSON" \
        --out-jsonl        "$RETRIEVAL_JSONL"

# ---------------------------------------------------------------------------
# Step (c): AV sample dump
# ---------------------------------------------------------------------------
#
# Same GPU + ar/av + activations preconditions as (b). The output is the
# input to step (d) — without av_samples.jsonl the judge has nothing to
# grade.

AV_DUMP_COND=1
AV_DUMP_REASON=""
if [ "$HAS_AR_AV" -ne 0 ]; then
    AV_DUMP_REASON="missing ar/ or av/ subdir in $SFT_DIR"
elif [ "$HAS_GPU" -ne 0 ]; then
    AV_DUMP_REASON="no GPU available (${GPU_STATUS_MSG})"
elif [ "$HAS_ACTIVATIONS" -ne 0 ]; then
    AV_DUMP_REASON="ACTIVATIONS_ROOT not found: $ACTIVATIONS_ROOT"
elif [ "$HAS_LABELS" -ne 0 ]; then
    AV_DUMP_REASON="LABELS_JSONL not found: $LABELS_JSONL"
else
    AV_DUMP_COND=0
fi

run_if "av-dump-precond" "$AV_DUMP_COND" "$AV_DUMP_REASON" \
    "c/av-dump: dump_av_samples" \
    env PYTHONPATH=src "$PY" scripts/eval/dump_av_samples.py \
        --ckpt-dir         "$SFT_DIR" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --labels-jsonl     "$LABELS_JSONL" \
        --per-position     6 \
        --temperatures     0.0 0.7 \
        --out-jsonl        "$AV_SAMPLES_JSONL"

# ---------------------------------------------------------------------------
# Step (d): LLM judge on AV samples (soft skip when key/frames absent)
# ---------------------------------------------------------------------------

JUDGE_COND=1
JUDGE_REASON=""
if [ "$HAS_OPENAI_KEY" -ne 0 ]; then
    JUDGE_REASON="OPENAI_API_KEY not set; skipping multimodal judge"
elif [ "$HAS_FRAMES_CACHE" -ne 0 ]; then
    JUDGE_REASON="FRAMES_CACHE not present: $FRAMES_CACHE"
elif [ "$HAS_AR_AV" -ne 0 ]; then
    JUDGE_REASON="missing ar/ or av/ subdir in $SFT_DIR"
elif [ "$HAS_GPU" -ne 0 ]; then
    JUDGE_REASON="no GPU available (${GPU_STATUS_MSG})"
elif [ "$HAS_ACTIVATIONS" -ne 0 ]; then
    JUDGE_REASON="ACTIVATIONS_ROOT not found: $ACTIVATIONS_ROOT"
elif [ "$HAS_LABELS" -ne 0 ]; then
    JUDGE_REASON="LABELS_JSONL not found: $LABELS_JSONL"
else
    JUDGE_COND=0
fi

run_if "judge-precond" "$JUDGE_COND" "$JUDGE_REASON" \
    "d/judge: llm_judge_av_captions" \
    env PYTHONPATH=src "$PY" scripts/eval/llm_judge_av_captions.py \
        --ckpt-dir         "$SFT_DIR" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --labels-jsonl     "$LABELS_JSONL" \
        --frames-cache     "$FRAMES_CACHE" \
        --video-keys       image wrist_image \
        --per-position     "$EVAL_JUDGE_PER_POSITION" \
        --concurrency      "$EVAL_JUDGE_CONCURRENCY" \
        --temperature      0.0 \
        --out-jsonl        "$JUDGE_JSONL"

# ---------------------------------------------------------------------------
# Step (e): sim A/B steerability (closed_loop_sim_ab.py)
# ---------------------------------------------------------------------------
#
# Requires a real GR00T policy checkpoint + the LIBERO MuJoCo stack. We
# soft-skip if GROOT_MODEL_PATH is unset or the directory is missing; the
# sim_ab.json being absent simply means the final scorecard reports the
# sim bands as NA, falling back to its no-sim required-gate set.

SIM_COND=1
SIM_REASON=""
if [ "$HAS_AR_AV" -ne 0 ]; then
    SIM_REASON="missing ar/ subdir in $SFT_DIR"
elif [ "$HAS_GPU" -ne 0 ]; then
    SIM_REASON="no GPU available (${GPU_STATUS_MSG})"
elif [ "$HAS_GROOT_MODEL" -ne 0 ]; then
    SIM_REASON="GROOT_MODEL_PATH unset or missing (set it to enable sim A/B)"
elif [ "$HAS_LABELS" -ne 0 ]; then
    SIM_REASON="LABELS_JSONL not found: $LABELS_JSONL"
else
    SIM_COND=0
fi

run_if "sim-precond" "$SIM_COND" "$SIM_REASON" \
    "e/sim: closed_loop_sim_ab" \
    env PYTHONPATH=src "$PY" scripts/eval/closed_loop_sim_ab.py \
        --ckpt-dir         "$SFT_DIR" \
        --groot-model-path "$GROOT_MODEL_PATH" \
        --labels-jsonl     "$LABELS_JSONL" \
        --embodiment-tag   "$LIBERO_EMBODIMENT_TAG" \
        --episodes-per-arm "$SIM_EPISODES_PER_ARM" \
        --n-envs           "$SIM_N_ENVS" \
        --port             "$SIM_PORT" \
        --py               "$PY" \
        --work-dir         "${OUT_DIR}/sim_ab_work" \
        --out-json         "$SIM_AB_JSON"

# ---------------------------------------------------------------------------
# Step (f): build final V4 SFT scorecard
# ---------------------------------------------------------------------------
#
# Pure python; always runs. Re-uses build_v3_scorecard.py (bands are
# config-not-code-per the V3 plan, so V4 inherits the same thresholds)
# and points every input override at the OUT_DIR artifacts written above
# plus the SFT_DIR's training metrics.jsonl.
#
# If the V4 plan ever needs distinct bands a copy at
# scripts/eval/build_v4_sft_scorecard.py is the right place; the wrapper
# is structured so swapping the script name is a one-line change.

run_hard "f/scorecard: build_v3_scorecard (V4 inputs)" \
    env PYTHONPATH=src "$PY" scripts/eval/build_v3_scorecard.py \
        --ckpt-dir       "$SFT_DIR" \
        --retrieval-json "$RETRIEVAL_JSON" \
        --judge-jsonl    "$JUDGE_JSONL" \
        --metrics-jsonl  "$METRICS_JSONL" \
        --sim-ab-json    "$SIM_AB_JSON" \
        --out-json       "$SCORECARD_JSON"

# ---------------------------------------------------------------------------
# SUMMARY.txt
# ---------------------------------------------------------------------------

SUMMARY_EXTRA=( )
if [ "$HAS_EXTRACTION_SWEEP" -eq 0 ]; then
    SUMMARY_EXTRA+=( --extraction-scorecard "$EXTRACTION_SCORECARD" )
fi

run_hard "summary: write SUMMARY.txt" \
    env PYTHONPATH=src "$PY" scripts/eval/_post_sft_evals_helpers.py \
        write_summary \
        --scorecard "$SCORECARD_JSON" \
        "${SUMMARY_EXTRA[@]}" \
        --out       "$SUMMARY_PATH"

HINTS_SH="${OUT_DIR}/grpo_orchestrator_hints.sh"
if [ "$DRY_RUN" -eq 0 ] && [ "${SKIP_GRPO_ORCH_HINTS:-0}" != "1" ]; then
    cat >"$HINTS_SH" <<'HINTS'
#!/usr/bin/env bash
# Sourced by scripts/training/orchestrate_v4_to_grpo.sh when present (before GRPO stage 7).
# Only uncommented exports apply. Example 14h wall targeting (orchestrator runs a short probe, then sets --total-steps):
#
# if [[ -z "${PILOT_TARGET_WALL_HOURS+x}" ]]; then
#   export PILOT_TARGET_WALL_HOURS=14
# fi
# if [[ -z "${PILOT_PROBE_STEPS+x}" ]]; then
#   export PILOT_PROBE_STEPS=5
# fi
# if [[ -z "${PILOT_WALL_HEADROOM+x}" ]]; then
#   export PILOT_WALL_HEADROOM=0.90
# fi
#
# To use a fixed step count instead, disable wall targeting:
# export PILOT_TARGET_WALL_HOURS=0
# export PILOT_GRPO_TOTAL_STEPS=1500
HINTS
    chmod a+x "$HINTS_SH" 2>/dev/null || true
    log "Wrote optional orchestrator overrides (source by hand or auto-sourced): ${HINTS_SH}"
fi

log ""
log "==================================================================="
log "DONE: V4 post-SFT eval wrapper complete."
log "  SFT_DIR:  ${SFT_DIR}"
log "  OUT_DIR:  ${OUT_DIR}"
log "  SUMMARY:  ${SUMMARY_PATH}"
log "==================================================================="

exit 0

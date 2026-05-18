#!/usr/bin/env bash
#
# orchestrate_v4_to_grpo.sh
#
# A10 master orchestrator: chains the V4 SFT watcher through the post-SFT eval
# stack into a SimpleVLA-RL flavored GRPO pilot run.
#
# Stages:
#   1. PREFLIGHT       Validate every required script + data file exists.
#   2. WAIT FOR SFT    scripts/training/wait_for_sft_completion.py (6h cap).
#   3. POST-SFT EVAL   scripts/eval/run_post_sft_evals_v4.sh $SFT_DIR.
#   4. IMPROVEMENTS    scripts/eval/build_improvements_report.py + diff_scorecards.py.
#   5. STEER SERVER    scripts/eval/launch_steer_server.sh + health_check_steer_server.py.
#   6. MINI-GRPO       3-step smoke through the steer server.
#   7. PILOT GRPO      ~14h wall budget by default (5-step probe scales --total-steps); backgrounded.
#   8. CLEANUP TRAP    On SIGINT/SIGTERM/error before pilot launch, stop steer + kill GRPO.
#
# Usage:
#   scripts/training/orchestrate_v4_to_grpo.sh <SFT_DIR>
#   scripts/training/orchestrate_v4_to_grpo.sh --dry-run [SFT_DIR]
#   scripts/training/orchestrate_v4_to_grpo.sh --help
#
# Every stage emits a single structured status line of the form
#   [orchestrate] stage=<N> status=<ok|fail|skip> elapsed_s=<int> [extra=...]
# so `tail -f $SFT_DIR/orchestrate.log | grep '\[orchestrate\]'` is sufficient
# for tracking progress.

set -u -o pipefail

# ----------------------------------------------------------------------------
# Constants / paths
# ----------------------------------------------------------------------------

ORCH_VERSION="a10.v2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env" || true
    set +a
fi

DEFAULT_BASELINE_DIR="data/sft/libero_4suite_v3"
DEFAULT_ACTIVATIONS_ROOT="data/activations/libero_4suite_v4_combined"
DEFAULT_CF_PAIRS="data/grpo/libero_goal_counterfactual_pairs.jsonl"
DEFAULT_STEER_PORT_PRIMARY=5556
DEFAULT_STEER_PORT_FALLBACKS=(5557 5558 5559 5560)
SFT_TIMEOUT_S="${SFT_TIMEOUT_S:-32400}"  # 9h default, override via env (gives ~3h slack over current ETA)
STEER_READY_TIMEOUT_S="${STEER_READY_TIMEOUT_S:-300}"
# GR00T policy checkpoint for steer server + sim-eval (prefer .env GROOT_MODEL_PATH).
DEFAULT_GROOT_REL="checkpoints/GR00T-N1.7-LIBERO/libero_goal"
PILOT_SIM_N_WORKERS_FALLBACK="${PILOT_SIM_N_WORKERS_FALLBACK:-4}"
# Overnight-scale GRPO pilot (override anytime before launch).
PILOT_GRPO_TOTAL_STEPS="${PILOT_GRPO_TOTAL_STEPS:-1000}"
PILOT_GRPO_EVAL_EVERY="${PILOT_GRPO_EVAL_EVERY:-50}"
PILOT_GRPO_SAVE_EVERY="${PILOT_GRPO_SAVE_EVERY:-100}"
# Wall-clock targeting (default ~14h): run PILOT_PROBE_STEPS GRPO steps, measure wall time,
# set --total-steps ≈ (hours * 3600 * headroom) / (probe_elapsed / probe_steps).
# Disable with PILOT_TARGET_WALL_HOURS=0 to use fixed PILOT_GRPO_TOTAL_STEPS only.
PILOT_TARGET_WALL_HOURS="${PILOT_TARGET_WALL_HOURS:-14}"
PILOT_PROBE_STEPS="${PILOT_PROBE_STEPS:-5}"
PILOT_WALL_HEADROOM="${PILOT_WALL_HEADROOM:-0.90}"
PILOT_MIN_STEPS="${PILOT_MIN_STEPS:-50}"
PILOT_MAX_STEPS="${PILOT_MAX_STEPS:-200000}"

# Required scripts (relative to repo root). Listed here so preflight can show
# the user EXACTLY which sibling agent's deliverable is missing.
REQUIRED_SCRIPTS=(
    "scripts/training/wait_for_sft_completion.py"            # A6
    "scripts/eval/run_post_sft_evals_v4.sh"                  # A2
    "scripts/eval/build_improvements_report.py"              # A5
    "scripts/eval/diff_scorecards.py"                        # A8
    "scripts/eval/launch_steer_server.sh"                    # A3
    "scripts/eval/stop_steer_server.sh"                      # A3
    "scripts/eval/health_check_steer_server.py"              # A3
    "scripts/training/profile_sim_worker.py"                 # A7
    "scripts/training/run_grpo.py"                           # existing
)

REQUIRED_DATA=(
    "data/grpo/libero_goal_counterfactual_pairs.jsonl"
    "data/activations/libero_4suite_v4_combined"
)

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

DRY_RUN=0
SFT_DIR=""

# ----------------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------------

print_help() {
    cat <<'EOF'
A10 orchestrator: V4 SFT -> evals -> steer server -> GRPO pilot.

Usage:
  scripts/training/orchestrate_v4_to_grpo.sh <SFT_DIR>
  scripts/training/orchestrate_v4_to_grpo.sh --dry-run [SFT_DIR]
  scripts/training/orchestrate_v4_to_grpo.sh --help

Arguments:
  SFT_DIR    Path to the V4 SFT run directory (e.g.
             data/sft/libero_4suite_v4_consistency_overnight). For --dry-run,
             defaults to data/sft/libero_4suite_v3.

Flags:
  --dry-run  Print every command the orchestrator WOULD run and exit 0
             without executing anything (no preflight failure on missing
             sibling-agent deliverables).
  --help     Show this message and exit.

Env overrides:
  PYTHON_BIN                 Python interpreter (default .venv/bin/python).
  STEER_PORT                 Force the steer-server port instead of probing.
  PILOT_SIM_N_WORKERS        Override sim worker count for the pilot.
  BASELINE_DIR               Override v3 baseline SFT dir for diff stages.
  SFT_TIMEOUT_S              Wait timeout for SFT completion (default 32400=9h).
  STEER_READY_TIMEOUT_S      Wait timeout for steer server readiness (default 300s).
  PILOT_GRPO_TOTAL_STEPS     Fallback fixed steps if wall probe is off or fails (default 1000).
  PILOT_GRPO_EVAL_EVERY      Pilot --eval-every (default 50).
  PILOT_GRPO_SAVE_EVERY      Pilot --save-every (default 100).
  PILOT_TARGET_WALL_HOURS    Target wall hours for pilot (default 14). Set to 0 to skip the
                             probe and use PILOT_GRPO_TOTAL_STEPS only.
  PILOT_PROBE_STEPS          Short GRPO run to time seconds/step before the long pilot (default 5).
  PILOT_WALL_HEADROOM        Fraction of target wall time to use (default 0.90 — leaves slack).
  GROOT_MODEL_PATH           GR00T checkpoint dir for steer-server (paths relative to repo root ok).
                             Default: checkpoints/GR00T-N1.7-LIBERO/libero_goal. Repo .env is loaded automatically.
  LIBERO_EMBODIMENT_TAG      Passed to steer server as --embodiment-tag (default LIBERO_PANDA).
  STEER_BOOTSTRAP_FILE      Bullets file for server boot when --ar-dir is set (default scripts/eval/default_steer_boot.txt).

  Optional file (after post-SFT eval): SFT_DIR/post_sft_eval/grpo_orchestrator_hints.sh
  is sourced at stage 7 before the pilot — set exports there to tune GRPO without editing this script.

When V4 SFT finishes, kick off the full chain with:
  nohup bash scripts/training/orchestrate_v4_to_grpo.sh \
      data/sft/libero_4suite_v4_consistency_overnight \
      >/dev/null 2>&1 &
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "[orchestrate] unknown flag: $1" >&2
            print_help >&2
            exit 2
            ;;
        *)
            if [[ -z "$SFT_DIR" ]]; then
                SFT_DIR="$1"
            else
                echo "[orchestrate] unexpected positional arg: $1" >&2
                exit 2
            fi
            shift
            ;;
    esac
done

if [[ "$DRY_RUN" -eq 1 && -z "$SFT_DIR" ]]; then
    SFT_DIR="$DEFAULT_BASELINE_DIR"
fi

if [[ -z "$SFT_DIR" ]]; then
    echo "[orchestrate] ERROR: SFT_DIR is required (see --help)" >&2
    exit 2
fi

# Normalize: strip trailing slash, but keep relative paths relative.
SFT_DIR="${SFT_DIR%/}"

BASELINE_DIR="${BASELINE_DIR:-$DEFAULT_BASELINE_DIR}"
ACTIVATIONS_ROOT="$DEFAULT_ACTIVATIONS_ROOT"
CF_PAIRS_PATH="$DEFAULT_CF_PAIRS"

LIBERO_EMBODIMENT_TAG="${LIBERO_EMBODIMENT_TAG:-LIBERO_PANDA}"

DEFAULT_STEER_BOOT_FILE="${REPO_ROOT}/scripts/eval/default_steer_boot.txt"
STEER_BOOTSTRAP_FILE="${STEER_BOOTSTRAP_FILE:-$DEFAULT_STEER_BOOT_FILE}"

LOG_FILE="${SFT_DIR}/orchestrate.log"

# ----------------------------------------------------------------------------
# Logging primitives (must work BEFORE the log file is set up; in dry-run we
# stay on stdout/stderr only).
# ----------------------------------------------------------------------------

T0_GLOBAL=$(date +%s)

_now_epoch() { date +%s; }

_log() {
    # Plain free-form log line. Tee'd to orchestrate.log via exec redirection.
    local msg="$*"
    printf '[orchestrate %s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$msg"
}

_status() {
    # Structured machine-greppable status line. Usage:
    #   _status <stage> <ok|fail|skip|start> <stage_t0> [extra="..." ...]
    local stage="$1"
    local status="$2"
    local t0="$3"
    shift 3
    local now elapsed
    now=$(_now_epoch)
    elapsed=$((now - t0))
    local extras=""
    if [[ $# -gt 0 ]]; then
        extras=" $*"
    fi
    printf '[orchestrate] stage=%s status=%s elapsed_s=%d%s\n' \
        "$stage" "$status" "$elapsed" "$extras"
}

ensure_groot_model_path() {
    if [[ -z "${GROOT_MODEL_PATH:-}" ]]; then
        local cand="${REPO_ROOT%/}/${DEFAULT_GROOT_REL}"
        if [[ -d "$cand" ]]; then
            export GROOT_MODEL_PATH="$cand"
        fi
    elif [[ "${GROOT_MODEL_PATH}" != /* ]]; then
        export GROOT_MODEL_PATH="${REPO_ROOT%/}/${GROOT_MODEL_PATH#./}"
    fi
}

groot_model_path_ok() {
    [[ -n "${GROOT_MODEL_PATH:-}" && -d "${GROOT_MODEL_PATH}" ]]
}

# ----------------------------------------------------------------------------
# GRPO pilot: optional hints from post-SFT eval + wall-clock step sizing
# ----------------------------------------------------------------------------

apply_grpo_hints_if_present() {
    local f="${SFT_DIR}/post_sft_eval/grpo_orchestrator_hints.sh"
    if [[ ! -f "$f" ]]; then
        return 0
    fi
    _log "sourcing optional GRPO hints: $f"
    set -a
    # shellcheck disable=SC1090
    source "$f"
    set +a
}

pilot_wall_targeting_enabled() {
    [[ -n "${PILOT_TARGET_WALL_HOURS:-}" && "$PILOT_TARGET_WALL_HOURS" != "0" ]]
}

resolve_pilot_total_steps_from_probe() {
    # Sets global PILOT_GRPO_TOTAL_STEPS. Args: nworkers (for sim pool).
    local nworkers="${1:-}"
    local fixed="${PILOT_GRPO_TOTAL_STEPS:-1000}"
    local probe_steps="${PILOT_PROBE_STEPS:-5}"
    local probe_out="${SFT_DIR}/grpo_wall_probe"
    local probe_log="${SFT_DIR}/grpo_wall_probe.log"

    if ! pilot_wall_targeting_enabled; then
        PILOT_GRPO_TOTAL_STEPS="$fixed"
        _log "GRPO pilot sizing: wall targeting OFF → total_steps=$PILOT_GRPO_TOTAL_STEPS"
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        _log "GRPO pilot sizing: would run ${probe_steps}-step wall probe → total_steps ≈ f(~${PILOT_TARGET_WALL_HOURS}h, headroom=${PILOT_WALL_HEADROOM})"
        return 0
    fi

    _log "GRPO wall probe: ${probe_steps} steps → ${probe_out} (log ${probe_log})"
    mkdir -p "$probe_out"
    local t0 t1 elapsed computed rc
    t0=$(_now_epoch)
    set +e
    "$PYTHON_BIN" scripts/training/run_grpo.py \
        --sft-dir "$SFT_DIR" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --output-dir "$probe_out" \
        --batch-size 4 \
        --rollouts-per-activation 8 \
        --total-steps "$probe_steps" \
        --eval-every 999999 \
        --save-every 999999 \
        --sim-reward-weight 0.5 \
        --sim-counterfactual-pairs-path "$CF_PAIRS_PATH" \
        --sim-policy-host localhost \
        --sim-policy-port "$STEER_PORT" \
        --sim-n-workers "$nworkers" \
        --sim-max-steps 100 \
        --dynamic-sampling \
        --use-ppo-clip \
        --disable-kl-anchor \
        --rollout-temperature-high 1.6 \
        --beta 0.0 \
        --seed 0 \
        >"$probe_log" 2>&1
    rc=$?
    set -e
    t1=$(_now_epoch)
    elapsed=$((t1 - t0))

    if [[ $rc -ne 0 ]]; then
        _log "WARN: wall probe failed rc=$rc → fallback total_steps=$fixed (see $probe_log)"
        PILOT_GRPO_TOTAL_STEPS="$fixed"
        return 0
    fi
    if [[ "$elapsed" -lt 1 ]]; then
        _log "WARN: wall probe elapsed=${elapsed}s → fallback total_steps=$fixed"
        PILOT_GRPO_TOTAL_STEPS="$fixed"
        return 0
    fi

    computed=$("$PYTHON_BIN" -c "h=float('${PILOT_TARGET_WALL_HOURS}')
hr=float('${PILOT_WALL_HEADROOM}')
pe=float('${elapsed}')
ps=float('${probe_steps}')
mn=int('${PILOT_MIN_STEPS}')
mx=int('${PILOT_MAX_STEPS}')
fixed=int('${fixed}')
target_s=h*3600.0*hr
sp=pe/ps
v=fixed if sp<=0 else int(target_s/sp)
v=max(mn, min(mx, v))
print(v)")

    PILOT_GRPO_TOTAL_STEPS="$computed"
    local ms_per
    ms_per=$((elapsed * 1000 / probe_steps))
    _log "wall probe: ${elapsed}s / ${probe_steps} steps (~${ms_per}ms/step) → total_steps=${computed} (≈ ${PILOT_TARGET_WALL_HOURS}h * headroom ${PILOT_WALL_HEADROOM})"
}

# ----------------------------------------------------------------------------
# Dry-run command emitter
# ----------------------------------------------------------------------------
#
# All "real" command invocations go through `run_cmd`. In dry-run mode it just
# echoes the command (one per line, shell-safe) and returns 0. In normal mode
# it execs the command and propagates the exit code.

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        local rendered=""
        local arg
        for arg in "$@"; do
            rendered+=" $(printf '%q' "$arg")"
        done
        printf 'DRY-RUN: %s\n' "${rendered# }"
        return 0
    fi
    "$@"
}

run_bg() {
    # Background variant: launches the command with nohup + disown so it
    # survives the orchestrator exit. Echoes the PID. In dry-run mode prints
    # the command and a placeholder PID.
    local log_path="$1"
    shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        local rendered=""
        local arg
        for arg in "$@"; do
            rendered+=" $(printf '%q' "$arg")"
        done
        printf 'DRY-RUN-BG: nohup%s >%s 2>&1 & disown\n' \
            "${rendered}" "$log_path"
        echo "DRYRUN_PID"
        return 0
    fi
    nohup "$@" >"$log_path" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid"
}

# ----------------------------------------------------------------------------
# Cleanup trap
# ----------------------------------------------------------------------------
#
# Only armed AFTER stage 5 succeeds (we have a steer server to tear down) and
# DISARMED at the end of stage 7 once the pilot is safely backgrounded -- we
# explicitly want the steer server and the pilot GRPO to outlive this script
# on the happy path. The trap fires on SIGINT/SIGTERM and on any error exit
# between stages 5 and 7.

STEER_PID=""
STEER_PORT=""
GRPO_PID=""
TRAP_ARMED=0

_cleanup() {
    local sig="${1:-EXIT}"
    if [[ "$TRAP_ARMED" -ne 1 ]]; then
        return 0
    fi
    _log "cleanup invoked (signal=${sig}); tearing down steer server / GRPO"
    if [[ "$DRY_RUN" -ne 1 ]] && [[ -x "scripts/eval/stop_steer_server.sh" ]]; then
        local slog="${SFT_DIR%/}/steer_server_logs"
        bash scripts/eval/stop_steer_server.sh --log-dir "$slog" \
            || _log "warn: stop_steer_server.sh exited nonzero"
    fi
    if [[ -n "$GRPO_PID" && "$GRPO_PID" != "DRYRUN_PID" && "$DRY_RUN" -ne 1 ]]; then
        if kill -0 "$GRPO_PID" 2>/dev/null; then
            _log "killing GRPO pid=$GRPO_PID"
            kill -TERM "$GRPO_PID" 2>/dev/null || true
        fi
    fi
}

arm_trap() {
    TRAP_ARMED=1
    trap '_cleanup INT;  exit 130' INT
    trap '_cleanup TERM; exit 143' TERM
    trap '_cleanup ERR;  exit 1'   ERR
}

disarm_trap() {
    TRAP_ARMED=0
    trap - INT TERM ERR
}

# ----------------------------------------------------------------------------
# Utility: find a free TCP port via `ss` (preferred) or bash /dev/tcp probing.
# ----------------------------------------------------------------------------

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q .
        return $?
    fi
    # Fallback: try to open a TCP connection to localhost:port.
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
    local rc=$?
    exec 3>&- 2>/dev/null || true
    [[ $rc -eq 0 ]]
}

pick_free_port() {
    if [[ -n "${STEER_PORT_OVERRIDE:-}" ]]; then
        echo "$STEER_PORT_OVERRIDE"
        return 0
    fi
    local candidates=("$DEFAULT_STEER_PORT_PRIMARY" "${DEFAULT_STEER_PORT_FALLBACKS[@]}")
    local p
    for p in "${candidates[@]}"; do
        if ! port_in_use "$p"; then
            echo "$p"
            return 0
        fi
    done
    echo ""
    return 1
}

# ----------------------------------------------------------------------------
# STAGE 1: PREFLIGHT
# ----------------------------------------------------------------------------

stage_preflight() {
    local t0
    t0=$(_now_epoch)
    _status 1 start "$t0" "dry_run=${DRY_RUN}"

    # SFT_DIR existence check (skipped in dry-run only when missing AND user
    # specified --dry-run with no positional, since the V3 default is real).
    if [[ ! -d "$SFT_DIR" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            _log "warn: SFT_DIR=$SFT_DIR not found, continuing (dry-run)"
        else
            _log "ERROR: SFT_DIR=$SFT_DIR not found"
            _status 1 fail "$t0" "reason=sft_dir_missing"
            exit 1
        fi
    fi

    local missing=()
    local f
    for f in "${REQUIRED_SCRIPTS[@]}"; do
        if [[ ! -e "$f" ]]; then
            missing+=("script:$f")
        fi
    done
    for f in "${REQUIRED_DATA[@]}"; do
        if [[ ! -e "$f" ]]; then
            missing+=("data:$f")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        _log "preflight: MISSING ${#missing[@]} prereq(s):"
        for f in "${missing[@]}"; do
            _log "  - $f"
        done
        if [[ "$DRY_RUN" -eq 1 ]]; then
            _log "dry-run: continuing despite missing prereqs (commands will still be printed)"
            _status 1 skip "$t0" "missing=${#missing[@]}"
        else
            _log "Halting. Sibling agents (A2/A3/A5/A6/A7/A8) must land their"
            _log "scripts before this orchestrator can run."
            _status 1 fail "$t0" "missing=${#missing[@]}"
            exit 1
        fi
    else
        ensure_groot_model_path
        if [[ "$DRY_RUN" -ne 1 ]] && ! groot_model_path_ok; then
            _log "ERROR: GR00T checkpoint not found for steer/sim (GROOT_MODEL_PATH unset or invalid)."
            _log "  Fix: export GROOT_MODEL_PATH=/abs/path/to/libero_goal checkpoint, or put it in ${REPO_ROOT}/.env"
            _log "  Fallback tried: ${REPO_ROOT}/${DEFAULT_GROOT_REL}"
            _status 1 fail "$t0" "reason=groot_model_missing"
            exit 1
        fi
        if groot_model_path_ok; then
            _log "GROOT_MODEL_PATH=${GROOT_MODEL_PATH}"
        elif [[ "$DRY_RUN" -eq 1 ]]; then
            _log "WARN: GROOT_MODEL_PATH unresolved (dry-run)"
        fi
        _status 1 ok "$t0" "missing=0"
    fi
}

# ----------------------------------------------------------------------------
# STAGE 2: WAIT FOR SFT
# ----------------------------------------------------------------------------

stage_wait_sft() {
    local t0
    t0=$(_now_epoch)
    _status 2 start "$t0"

    run_cmd "$PYTHON_BIN" scripts/training/wait_for_sft_completion.py \
        --sft-dir "$SFT_DIR" \
        --timeout-s "$SFT_TIMEOUT_S"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        _log "wait_for_sft_completion.py exited rc=$rc"
        _status 2 fail "$t0" "rc=$rc"
        exit 1
    fi
    _status 2 ok "$t0"
}

# ----------------------------------------------------------------------------
# STAGE 3: POST-SFT EVAL
# ----------------------------------------------------------------------------

stage_post_sft_eval() {
    local t0
    t0=$(_now_epoch)
    _status 3 start "$t0"

    run_cmd bash scripts/eval/run_post_sft_evals_v4.sh "$SFT_DIR"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        _log "run_post_sft_evals_v4.sh exited rc=$rc"
        _status 3 fail "$t0" "rc=$rc"
        exit 1
    fi
    _status 3 ok "$t0"
}

# ----------------------------------------------------------------------------
# STAGE 4: IMPROVEMENTS REPORT + SCORECARD DIFF
# ----------------------------------------------------------------------------

stage_improvements() {
    local t0
    t0=$(_now_epoch)
    _status 4 start "$t0"

    run_cmd "$PYTHON_BIN" scripts/eval/build_improvements_report.py \
        --sft-dir "$SFT_DIR" \
        --baseline-dir "$BASELINE_DIR"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        _log "build_improvements_report.py exited rc=$rc"
        _status 4 fail "$t0" "rc=$rc"
        exit 1
    fi

    # Pick the most apples-to-apples scorecard available in both dirs. v3 uses
    # the new v4-style extraction scorecard as its comparable artifact.
    local baseline_card="${BASELINE_DIR}/v4_extraction_scorecard.json"
    local candidate_card="${SFT_DIR}/v4_extraction_scorecard.json"
    if [[ "$DRY_RUN" -eq 1 || ( -e "$baseline_card" && -e "$candidate_card" ) ]]; then
        run_cmd "$PYTHON_BIN" scripts/eval/diff_scorecards.py \
            --baseline "$baseline_card" \
            --candidate "$candidate_card" \
            --out "$SFT_DIR/scorecard_diff_v3_vs_v4.md"
        local rc2=$?
        if [[ $rc2 -ne 0 ]]; then
            _log "diff_scorecards.py exited rc=$rc2 (non-fatal)"
        fi
    else
        _log "skipping diff_scorecards.py: missing $baseline_card or $candidate_card"
    fi

    _status 4 ok "$t0" "report=$SFT_DIR/improvements.md"
}

# ----------------------------------------------------------------------------
# STAGE 5: STEER SERVER UP
# ----------------------------------------------------------------------------

stage_steer_up() {
    local t0
    t0=$(_now_epoch)
    _status 5 start "$t0"

    ensure_groot_model_path
    if [[ "$DRY_RUN" -ne 1 ]] && ! groot_model_path_ok; then
        _log "ERROR: steer server needs GROOT_MODEL_PATH (unset after .env load)"
        _status 5 fail "$t0" "reason=no_groot_path"
        exit 1
    fi
    if [[ "$DRY_RUN" -ne 1 ]] && [[ ! -r "$STEER_BOOTSTRAP_FILE" ]]; then
        _log "ERROR: steer server needs steer bootstrap bullets (readable file)."
        _log "  Fix: STEER_BOOTSTRAP_FILE=<path/to.txt>; default=${DEFAULT_STEER_BOOT_FILE}"
        _status 5 fail "$t0" "reason=no_steer_boot_file"
        exit 1
    fi

    local port
    if [[ "$DRY_RUN" -eq 1 ]]; then
        port="$DEFAULT_STEER_PORT_PRIMARY"
    else
        port="$(pick_free_port || true)"
        if [[ -z "$port" ]]; then
            _log "could not find a free port in {$DEFAULT_STEER_PORT_PRIMARY ${DEFAULT_STEER_PORT_FALLBACKS[*]}}"
            _status 5 fail "$t0" "reason=no_free_port"
            exit 1
        fi
    fi
    STEER_PORT="$port"

    local launcher_log="${SFT_DIR}/steer_launcher.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY-RUN-BG: bash scripts/eval/launch_steer_server.sh --sft-dir %q --port %d --ready-timeout %d \\\n' \
            "$SFT_DIR" "$port" "$STEER_READY_TIMEOUT_S"
        printf '       -- --model-path %q --embodiment-tag %q --steer-text-file %q >%q 2>&1 & disown\n' \
            "${GROOT_MODEL_PATH:-${REPO_ROOT}/${DEFAULT_GROOT_REL}}" "$LIBERO_EMBODIMENT_TAG" "${STEER_BOOTSTRAP_FILE}" "$launcher_log"
        STEER_PID="DRYRUN_PID"
    else
        # The launcher script is responsible for daemonizing the actual server
        # and writing its PID; we tail its stdout for the READY signal.
        : > "$launcher_log"
        bash scripts/eval/launch_steer_server.sh \
            --sft-dir "$SFT_DIR" \
            --port "$port" \
            --ready-timeout "$STEER_READY_TIMEOUT_S" \
            -- \
            --model-path "$GROOT_MODEL_PATH" \
            --embodiment-tag "$LIBERO_EMBODIMENT_TAG" \
            --steer-text-file "$STEER_BOOTSTRAP_FILE" \
            >>"$launcher_log" 2>&1 &
        STEER_PID=$!
        disown "$STEER_PID" 2>/dev/null || true
        # Arm cleanup NOW: if READY-wait or health-check fails below we want
        # the trap to tear down the just-launched (potentially orphaned)
        # steer server process.
        arm_trap

        local waited=0
        local ready=0
        while (( waited < STEER_READY_TIMEOUT_S )); do
            if grep -q '\bREADY\b' "$launcher_log" 2>/dev/null; then
                ready=1
                break
            fi
            sleep 2
            waited=$((waited + 2))
        done
        if [[ "$ready" -ne 1 ]]; then
            _log "steer server did not emit READY within ${STEER_READY_TIMEOUT_S}s (see $launcher_log)"
            _status 5 fail "$t0" "reason=ready_timeout port=$port"
            _cleanup READY_TIMEOUT
            exit 1
        fi
    fi

    run_cmd "$PYTHON_BIN" scripts/eval/health_check_steer_server.py --port "$port"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        _log "health_check_steer_server.py rc=$rc"
        _status 5 fail "$t0" "reason=unhealthy port=$port"
        _cleanup HEALTH_FAIL
        exit 1
    fi

    _status 5 ok "$t0" "port=$port pid=$STEER_PID"
}

# ----------------------------------------------------------------------------
# STAGE 6: MINI-GRPO SMOKE (3 steps)
# ----------------------------------------------------------------------------

stage_grpo_smoke() {
    local t0
    t0=$(_now_epoch)
    _status 6 start "$t0"

    local smoke_out="${SFT_DIR}/grpo_smoke"
    local smoke_log="${SFT_DIR}/grpo_smoke.log"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY-RUN: mkdir -p %q\n' "$smoke_out"
    else
        mkdir -p "$smoke_out"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        # Print the full command in --dry-run for transparency.
        printf 'DRY-RUN: %s scripts/training/run_grpo.py --sft-dir %q --activations-root %q --output-dir %q --batch-size 2 --rollouts-per-activation 2 --total-steps 3 --eval-every 100 --save-every 100 --sim-reward-weight 0.5 --sim-counterfactual-pairs-path %q --sim-policy-host localhost --sim-policy-port %d --sim-n-workers 2 --sim-max-steps 50 --dynamic-sampling --use-ppo-clip --disable-kl-anchor --rollout-temperature-high 1.6 --beta 0.0 --seed 0  (tee -> %q)\n' \
            "$PYTHON_BIN" "$SFT_DIR" "$ACTIVATIONS_ROOT" "$smoke_out" \
            "$CF_PAIRS_PATH" "$STEER_PORT" "$smoke_log"
        _status 6 ok "$t0" "dry_run=1"
        return 0
    fi

    set +e
    "$PYTHON_BIN" scripts/training/run_grpo.py \
        --sft-dir "$SFT_DIR" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --output-dir "$smoke_out" \
        --batch-size 2 \
        --rollouts-per-activation 2 \
        --total-steps 3 \
        --eval-every 100 \
        --save-every 100 \
        --sim-reward-weight 0.5 \
        --sim-counterfactual-pairs-path "$CF_PAIRS_PATH" \
        --sim-policy-host localhost \
        --sim-policy-port "$STEER_PORT" \
        --sim-n-workers 2 \
        --sim-max-steps 50 \
        --dynamic-sampling \
        --use-ppo-clip \
        --disable-kl-anchor \
        --rollout-temperature-high 1.6 \
        --beta 0.0 \
        --seed 0 \
        2>&1 | tee "$smoke_log"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -ne 0 ]]; then
        _log "GRPO smoke run failed (rc=$rc). See $smoke_log"
        _status 6 fail "$t0" "rc=$rc log=$smoke_log"
        exit 1
    fi
    _status 6 ok "$t0" "out=$smoke_out"
}

# ----------------------------------------------------------------------------
# STAGE 7: PILOT GRPO (long run, backgrounded; size via PILOT_GRPO_TOTAL_STEPS)
# ----------------------------------------------------------------------------

resolve_pilot_n_workers() {
    # Priority: env override > profile_sim_worker.py recommendation > fallback.
    if [[ -n "${PILOT_SIM_N_WORKERS:-}" ]]; then
        echo "$PILOT_SIM_N_WORKERS"
        return 0
    fi
    if [[ -x "scripts/training/profile_sim_worker.py" || -r "scripts/training/profile_sim_worker.py" ]]; then
        local rec
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "$PILOT_SIM_N_WORKERS_FALLBACK"
            return 0
        fi
        rec="$("$PYTHON_BIN" scripts/training/profile_sim_worker.py --recommend 2>/dev/null \
            | awk '/recommended_n_workers/ {print $2; exit}')"
        if [[ -n "$rec" && "$rec" =~ ^[0-9]+$ ]]; then
            echo "$rec"
            return 0
        fi
    fi
    echo "$PILOT_SIM_N_WORKERS_FALLBACK"
}

stage_grpo_pilot() {
    local t0
    t0=$(_now_epoch)
    _status 7 start "$t0"

    # Optional SFT_DIR/post_sft_eval/grpo_orchestrator_hints.sh — written when stage 3 finishes.
    apply_grpo_hints_if_present

    local pilot_out="${SFT_DIR}/grpo_pilot"
    local pilot_log="${SFT_DIR}/grpo_pilot.log"
    local nworkers
    nworkers="$(resolve_pilot_n_workers)"

    resolve_pilot_total_steps_from_probe "$nworkers"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY-RUN: mkdir -p %q\n' "$pilot_out"
        if pilot_wall_targeting_enabled; then
            printf 'DRY-RUN: (wall probe %s GRPO steps timed first; then pilot total-steps set from ~%sh budget)\n' \
                "$PILOT_PROBE_STEPS" "$PILOT_TARGET_WALL_HOURS"
        fi
        printf 'DRY-RUN-BG: nohup %s scripts/training/run_grpo.py --sft-dir %q --activations-root %q --output-dir %q --batch-size 4 --rollouts-per-activation 8 --total-steps %s --eval-every %s --save-every %s --sim-reward-weight 0.5 --sim-counterfactual-pairs-path %q --sim-policy-host localhost --sim-policy-port %d --sim-n-workers %d --sim-max-steps 100 --dynamic-sampling --use-ppo-clip --disable-kl-anchor --rollout-temperature-high 1.6 --beta 0.0 --seed 0 >%q 2>&1 & disown\n' \
            "$PYTHON_BIN" "$SFT_DIR" "$ACTIVATIONS_ROOT" "$pilot_out" \
            "$PILOT_GRPO_TOTAL_STEPS" "$PILOT_GRPO_EVAL_EVERY" "$PILOT_GRPO_SAVE_EVERY" \
            "$CF_PAIRS_PATH" "$STEER_PORT" "$nworkers" "$pilot_log"
        GRPO_PID="DRYRUN_PID"
        _status 7 ok "$t0" "pid=$GRPO_PID log=$pilot_log workers=$nworkers total_steps=$PILOT_GRPO_TOTAL_STEPS dry_run=1"
        return 0
    fi

    mkdir -p "$pilot_out"

    _log "pilot GRPO: total_steps=$PILOT_GRPO_TOTAL_STEPS eval_every=$PILOT_GRPO_EVAL_EVERY save_every=$PILOT_GRPO_SAVE_EVERY workers=$nworkers"

    nohup "$PYTHON_BIN" scripts/training/run_grpo.py \
        --sft-dir "$SFT_DIR" \
        --activations-root "$ACTIVATIONS_ROOT" \
        --output-dir "$pilot_out" \
        --batch-size 4 \
        --rollouts-per-activation 8 \
        --total-steps "$PILOT_GRPO_TOTAL_STEPS" \
        --eval-every "$PILOT_GRPO_EVAL_EVERY" \
        --save-every "$PILOT_GRPO_SAVE_EVERY" \
        --sim-reward-weight 0.5 \
        --sim-counterfactual-pairs-path "$CF_PAIRS_PATH" \
        --sim-policy-host localhost \
        --sim-policy-port "$STEER_PORT" \
        --sim-n-workers "$nworkers" \
        --sim-max-steps 100 \
        --dynamic-sampling \
        --use-ppo-clip \
        --disable-kl-anchor \
        --rollout-temperature-high 1.6 \
        --beta 0.0 \
        --seed 0 \
        >"$pilot_log" 2>&1 &
    GRPO_PID=$!
    disown "$GRPO_PID" 2>/dev/null || true

    # Give the pilot a couple of seconds to fail fast on argparse / import
    # errors so we can surface that instead of returning a "happy" exit.
    sleep 3
    if ! kill -0 "$GRPO_PID" 2>/dev/null; then
        _log "pilot GRPO died within 3s; see $pilot_log"
        _status 7 fail "$t0" "pid=$GRPO_PID log=$pilot_log"
        exit 1
    fi

    _status 7 ok "$t0" "pid=$GRPO_PID log=$pilot_log workers=$nworkers total_steps=$PILOT_GRPO_TOTAL_STEPS"
    _log "GRPO pilot is RUNNING in background:"
    _log "  pid:  $GRPO_PID"
    _log "  log:  $pilot_log"
    _log "  out:  $pilot_out"
    _log "  port: $STEER_PORT (steer pid=$STEER_PID)"
    _log "Steer server + GRPO pilot will outlive this orchestrator (trap disarmed)."
}

# ----------------------------------------------------------------------------
# main()
# ----------------------------------------------------------------------------

main() {
    # Stage 1 always runs first and decides whether the script may proceed.
    # In real mode we set up tee'ing to $SFT_DIR/orchestrate.log; in dry-run
    # we keep output on stdout only so the validator can scrape it cleanly.
    if [[ "$DRY_RUN" -ne 1 ]]; then
        if [[ ! -d "$SFT_DIR" ]]; then
            echo "[orchestrate] ERROR: SFT_DIR=$SFT_DIR not found" >&2
            exit 1
        fi
        mkdir -p "$(dirname "$LOG_FILE")"
        # Re-exec stdout/stderr through tee -> orchestrate.log so every echo,
        # every child-process output, and every status line is captured.
        exec > >(tee -a "$LOG_FILE") 2>&1
        _log "==== orchestrate_v4_to_grpo.sh ${ORCH_VERSION} starting ===="
        _log "SFT_DIR=$SFT_DIR"
        _log "BASELINE_DIR=$BASELINE_DIR"
        _log "ACTIVATIONS_ROOT=$ACTIVATIONS_ROOT"
        _log "CF_PAIRS_PATH=$CF_PAIRS_PATH"
        _log "log_file=$LOG_FILE"
    else
        _log "==== orchestrate_v4_to_grpo.sh ${ORCH_VERSION} DRY-RUN ===="
        _log "SFT_DIR=$SFT_DIR (no real I/O will occur)"
    fi

    stage_preflight
    stage_wait_sft
    stage_post_sft_eval
    stage_improvements
    stage_steer_up
    stage_grpo_smoke
    stage_grpo_pilot

    # Pilot is backgrounded -- disarm the cleanup trap so a normal `exit 0`
    # below does NOT take down the steer server or the GRPO pilot.
    disarm_trap

    local total
    total=$(($(_now_epoch) - T0_GLOBAL))
    _status 0 done "$T0_GLOBAL" "total_elapsed_s=$total"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        _log "DRY-RUN complete. No commands were executed."
    else
        _log "Orchestrator handing off to backgrounded pilot. Bye."
    fi
    exit 0
}

main "$@"

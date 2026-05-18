#!/usr/bin/env bash
# Iterative smoke-mine -> audit -> tune loop for CF pairs.
#
# Each iteration:
#   1) mine 200 rows into data/grpo/cf_smoke_iter<i>.jsonl with the
#      tuner-suggested flags (seeded by iteration index for reproducibility).
#   2) audit; if gate passes, jump to the production mine.
#   3) tune; if tuner returns NEEDS_EDIT, halt with exit 2 so the parent
#      orchestrator can ask the user.
#
# Caps at 5 iterations. Appends each step to
# data/grpo/cf_mining_iterations.log.
#
# Usage (defaults shown):
#   PYTHONPATH=src bash scripts/training/iter_mine_cf_pairs.sh
#
# Environment overrides:
#   LABELS, OUT_DIR, MAX_ITERS, SMOKE_SIZE, PROD_SIZE
#
# Exits:
#   0 - gate passed AND production mine + audit passed
#   1 - exhausted MAX_ITERS without passing gate
#   2 - tuner reported NEEDS_EDIT that this loop can't satisfy alone
#   3 - production mine failed gate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LABELS="${LABELS:-data/labels/libero_4suite_combined/labels.jsonl}"
OUT_DIR="${OUT_DIR:-data/grpo}"
MAX_ITERS="${MAX_ITERS:-5}"
SMOKE_SIZE="${SMOKE_SIZE:-200}"
PROD_SIZE="${PROD_SIZE:-5000}"
PROD_OUT="${PROD_OUT:-${OUT_DIR}/libero_goal_counterfactual_pairs.jsonl}"
LOG="${OUT_DIR}/cf_mining_iterations.log"

mkdir -p "$OUT_DIR"
: > "$LOG"

# Prefer the project venv if present, otherwise the shell's default python.
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        PY="${REPO_ROOT}/.venv/bin/python"
    else
        PY="python"
    fi
fi
export PYTHONPATH="${PYTHONPATH:-src}"

log() { echo "[iter_cf] $*" | tee -a "$LOG"; }

prev_flags=""
passing_flags=""
passing_iter=""

for ((i=1; i<=MAX_ITERS; i++)); do
    out="${OUT_DIR}/cf_smoke_iter${i}.jsonl"
    audit_json="${out}.audit.json"
    audit_md="${out}.audit.md"

    log "==== iter ${i} ===="
    log "mine flags: ${prev_flags:-<defaults>}"
    # shellcheck disable=SC2086
    "$PY" scripts/training/mine_grpo_counterfactual_pairs.py \
        --labels "$LABELS" \
        --out "$out" \
        --seed "$i" \
        --max-total "$SMOKE_SIZE" \
        ${prev_flags} >>"$LOG" 2>&1

    set +e
    "$PY" scripts/training/audit_cf_pairs.py --pairs "$out" --gate \
        --json-out "$audit_json" --md-out "$audit_md" | tee -a "$LOG"
    gate_rc=${PIPESTATUS[0]}
    set -e

    if [[ "$gate_rc" -eq 0 ]]; then
        log "iter ${i}: GATE PASSED"
        passing_flags="$prev_flags"
        passing_iter="$i"
        break
    fi

    # Ask the tuner for next-iter flags.
    tune_out=$("$PY" scripts/training/_tune_cf_mining.py \
        --audit-json "$audit_json" \
        --prev-flags "$prev_flags")
    log "tuner: ${tune_out//$'\n'/ ; }"
    next_flags=$(printf '%s\n' "$tune_out" | sed -n 's/^NEXT_FLAGS://p')
    needs=$(printf '%s\n' "$tune_out" | sed -n 's/^NEEDS_EDIT://p')

    if [[ -n "$needs" ]]; then
        log "iter ${i}: tuner reported NEEDS_EDIT=${needs}; pausing"
        # We already implemented --max-per-source-task and
        # --balance-target-counts; if neither matches, halt.
        unmet=""
        for tok in ${needs//,/ }; do
            case "$tok" in
                max_per_source_task) ;;          # supported
                balance_target_counts) ;;        # supported
                *) unmet="${unmet}${tok} " ;;
            esac
        done
        if [[ -n "$unmet" ]]; then
            log "iter ${i}: miner missing support for: ${unmet}- halting"
            exit 2
        fi
    fi

    if [[ "$next_flags" == "$prev_flags" ]]; then
        log "iter ${i}: tuner produced no new flags (still failing); halting"
        exit 2
    fi
    prev_flags="$next_flags"
done

if [[ -z "$passing_iter" ]]; then
    log "EXHAUSTED ${MAX_ITERS} iterations without passing the gate"
    exit 1
fi

# Production mine with the validated flags.
log "==== production mine (n=${PROD_SIZE}) with iter ${passing_iter} flags ===="
log "prod flags: ${passing_flags:-<defaults>}"
# shellcheck disable=SC2086
"$PY" scripts/training/mine_grpo_counterfactual_pairs.py \
    --labels "$LABELS" \
    --out "$PROD_OUT" \
    --seed 0 \
    --max-total "$PROD_SIZE" \
    ${passing_flags} >>"$LOG" 2>&1

set +e
"$PY" scripts/training/audit_cf_pairs.py --pairs "$PROD_OUT" --gate \
    --json-out "${PROD_OUT}.audit.json" \
    --md-out "${PROD_OUT}.audit.md" | tee -a "$LOG"
prod_rc=${PIPESTATUS[0]}
set -e
if [[ "$prod_rc" -ne 0 ]]; then
    log "production mine FAILED gate; check ${PROD_OUT}.audit.md"
    exit 3
fi

log "DONE. Production pairs: ${PROD_OUT}"

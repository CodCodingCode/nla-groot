#!/usr/bin/env bash
# Sequential per-suite labeling over all 4 LIBERO suites.
# Phase 4 of the libero 50k recipe c plan.
#
# Per suite: --positions-per-example 2, --guarantee-strata, --concurrency 128
# Uses the hardened production prompt via build_position_prompt.
#
# Pre-req: Phase 3 extraction has populated
#          data/activations/libero_4suite_stride2/libero_<suite>/
#
# Usage:
#   bash scripts/labeling/run_label_all_libero_suites.sh
#
# Outputs:
#   data/labels/libero_4suite_stride2/libero_<suite>/{labels.jsonl,frames_cache/,manifest.json}
#   data/labels/libero_4suite_stride2/label_<suite>.log

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set" >&2
    exit 2
fi

ACT_ROOT="data/activations/libero_4suite_stride2"
LBL_ROOT="data/labels/libero_4suite_stride2"
mkdir -p "${LBL_ROOT}"

SUITES=("goal" "spatial" "object" "10")
POS_PER_EX=2
CONCURRENCY=128

t_total_start=$(date +%s)
for SUITE in "${SUITES[@]}"; do
    ACT_DIR="${ACT_ROOT}/libero_${SUITE}"
    LBL_DIR="${LBL_ROOT}/libero_${SUITE}"
    DS_DIR="third_party/Isaac-GR00T/examples/LIBERO/libero_${SUITE}_no_noops_1.0.0_lerobot"
    LOG_FILE="${LBL_ROOT}/label_${SUITE}.log"

    if [ ! -f "${ACT_DIR}/manifest.json" ]; then
        echo "=== SKIP libero_${SUITE} (no activations manifest at ${ACT_DIR}) ==="
        continue
    fi
    echo "=== START labeling libero_${SUITE} -> ${LBL_DIR} ==="
    t0=$(date +%s)
    PYTHONPATH=src .venv/bin/python scripts/labeling/run_label.py \
        --activations-root "${ACT_DIR}" \
        --dataset-root "${DS_DIR}" \
        --labels-dir "${LBL_DIR}" \
        --positions-per-example "${POS_PER_EX}" \
        --guarantee-strata \
        --concurrency "${CONCURRENCY}" \
        2>&1 | tee "${LOG_FILE}"
    t1=$(date +%s)
    echo "=== DONE libero_${SUITE} in $((t1 - t0))s ==="
done
t_total_end=$(date +%s)
echo "=== ALL SUITES DONE in $((t_total_end - t_total_start))s ==="

echo
echo "=== Per-suite label counts ==="
for SUITE in "${SUITES[@]}"; do
    LBL_DIR="${LBL_ROOT}/libero_${SUITE}"
    if [ -f "${LBL_DIR}/labels.jsonl" ]; then
        N=$(wc -l < "${LBL_DIR}/labels.jsonl")
        echo "  libero_${SUITE}: ${N} labels"
    else
        echo "  libero_${SUITE}: MISSING labels.jsonl"
    fi
done

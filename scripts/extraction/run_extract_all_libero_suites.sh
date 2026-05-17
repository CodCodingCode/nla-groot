#!/usr/bin/env bash
# Sequential per-suite extraction over all 4 LIBERO suites.
# Phase 3 of the libero 50k recipe c plan.
#
# Per suite: --step-stride 2, --steps-per-traj 60 (matches pilot density),
#            --store-input-ids, --compute-stats.
# All trajectories in each suite are extracted (no --traj-ids cap).
#
# Usage:
#   bash scripts/extraction/run_extract_all_libero_suites.sh
#
# Outputs:
#   data/activations/libero_4suite_stride2/libero_<suite>/{manifest.json,index.jsonl,...}
#   data/activations/libero_4suite_stride2/extract_<suite>.log

set -euo pipefail

cd "$(dirname "$0")/../.."

# Pull OPENAI_* + HF_TOKEN if .env is around (extract doesn't need it but harmless)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

OUT_ROOT="data/activations/libero_4suite_stride2"
mkdir -p "${OUT_ROOT}"

SUITES=("goal" "spatial" "object" "10")
STRIDE=2
STEPS_PER_TRAJ=60

t_total_start=$(date +%s)
for SUITE in "${SUITES[@]}"; do
    OUT_DIR="${OUT_ROOT}/libero_${SUITE}"
    LOG_FILE="${OUT_ROOT}/extract_${SUITE}.log"
    if [ -f "${OUT_DIR}/manifest.json" ]; then
        N_DONE=$(jq -r '.num_examples // 0' "${OUT_DIR}/manifest.json" 2>/dev/null || echo 0)
        echo "=== SKIP libero_${SUITE} (manifest already present, num_examples=${N_DONE}) ==="
        continue
    fi
    echo "=== START libero_${SUITE} -> ${OUT_DIR} ==="
    t0=$(date +%s)
    PYTHONPATH=src .venv/bin/python scripts/extraction/run_extract.py \
        --model-path "checkpoints/GR00T-N1.7-LIBERO/libero_${SUITE}" \
        --dataset-path "third_party/Isaac-GR00T/examples/LIBERO/libero_${SUITE}_no_noops_1.0.0_lerobot" \
        --embodiment-tag LIBERO_PANDA \
        --out-root "${OUT_DIR}" \
        --device cuda:0 \
        --step-stride "${STRIDE}" \
        --steps-per-traj "${STEPS_PER_TRAJ}" \
        --store-input-ids \
        --compute-stats \
        2>&1 | tee "${LOG_FILE}"
    t1=$(date +%s)
    echo "=== DONE libero_${SUITE} in $((t1 - t0))s ==="
done
t_total_end=$(date +%s)
echo "=== ALL SUITES DONE in $((t_total_end - t_total_start))s ==="

echo
echo "=== Per-suite row counts ==="
for SUITE in "${SUITES[@]}"; do
    OUT_DIR="${OUT_ROOT}/libero_${SUITE}"
    if [ -f "${OUT_DIR}/manifest.json" ]; then
        N=$(jq -r '.num_examples // 0' "${OUT_DIR}/manifest.json")
        SHARDS=$(jq -r '.num_shards // 0' "${OUT_DIR}/manifest.json")
        echo "  libero_${SUITE}: ${N} examples in ${SHARDS} shards"
    else
        echo "  libero_${SUITE}: MISSING MANIFEST"
    fi
done

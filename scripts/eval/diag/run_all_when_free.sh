#!/usr/bin/env bash
# Orchestrate the AR-vs-AV diagnostic suite once the GPU frees.
# Waits for the running SFT (v9_ar_refine_5h) to release the H100, then:
#   1. m1 + m3 (no sim, fast) for both checkpoints   -> touch DONE_M13
#   2. m2 injection ladder (sim, ~1-2h each) for both -> touch DONE_ALL
# Launch DETACHED (setsid nohup). Robust to SSH/session death.
set -u
cd /lambda/nfs/Natha/nla-groot
export PYTHONPATH=src
PY=.venv/bin/python
D=data/eval/diag
mkdir -p "$D"
CKPTS=(v9_combined_12k v9_ar_refine_5h)

echo "[orch] $(date) waiting for GPU to free (>20GB)..."
# Primary gate: free GPU memory. The training job holds ~73GB; the policy
# server keeps ~7GB. When training exits, free mem jumps well past 20GB.
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE:-0}" -gt 20000 ]; then break; fi
  sleep 30
done
echo "[orch] $(date) GPU free (${FREE} MiB). Starting m1+m3."

run () { echo "[orch] $(date) RUN: $*"; "$@"; echo "[orch] $(date) EXIT=$? : $*"; }

# ---- Stage 1: reconstruction split + intent separation (no sim) ----
for CK in "${CKPTS[@]}"; do
  if [ ! -d "data/sft/$CK/ar" ] || [ ! -d "data/sft/$CK/av" ]; then
    echo "[orch] SKIP $CK (missing av/ or ar/)"; continue
  fi
  run $PY scripts/eval/diag/m1_recon_split.py --sft-dir "data/sft/$CK" --device cuda \
      --out-json "$D/m1_$CK.json"
  run $PY scripts/eval/diag/m3_intent_separation.py --sft-dir "data/sft/$CK" --device cuda \
      --out-dir "$D/m3_$CK"
done
touch "$D/DONE_M13"
echo "[orch] $(date) m1+m3 complete. Starting m2 injection ladder (sim)."

# ---- Stage 2: injection ladder (sim, long) ----
for CK in "${CKPTS[@]}"; do
  if [ ! -d "data/sft/$CK/ar" ]; then echo "[orch] SKIP m2 $CK"; continue; fi
  run $PY scripts/eval/diag/m2_injection_ladder.py --sft-dir "data/sft/$CK" --device cuda \
      --policy-host localhost --policy-port 5556 --n-workers 4 \
      --cache-path data/eval/sim_rollout_cache.jsonl \
      --baseline-results data/eval/v9_combined_12k_n100_cf_strided_cached.json \
      --out-json "$D/m2_ladder_$CK.json"
done
touch "$D/DONE_ALL"
echo "[orch] $(date) ALL DONE."

#!/usr/bin/env bash
# Watch an SFT training log; when "SFT done" lands, automatically run:
#   1. AV caption diagnostic (5 min, gives early intent-awareness read)
#   2. Stop any existing steer server, launch a new one on the SFT checkpoint
#   3. Populate no-steer cache from a prior eval JSON (free 64 of 128 rollouts)
#   4. Run compare_cf_steer_checkpoints.py with image_patch_strided + strided_k=128
#
# Designed to be launched detached (setsid+nohup+disown) so it survives the
# Claude session and SSH disconnect; the SFT process polls the log file path
# so this works without needing a process-tree relationship to the training.
#
# Usage:
#   scripts/eval/auto_cf_eval_after_sft.sh \
#       --sft-log data/sft/v9_combined_12k_launch.log \
#       --sft-dir data/sft/v9_combined_12k \
#       --run-name v9_combined_12k
#
# Tail the auto-eval log to watch progress:
#   tail -F data/eval/<run-name>_auto_eval.log
#
set -euo pipefail

SFT_LOG=""
SFT_DIR=""
RUN_NAME=""
PRIOR_EVAL_JSON="data/eval/v8_cf_steer_goal.json"   # for no-steer cache seed
CACHE_PATH="data/eval/sim_rollout_cache.jsonl"
PORT=5556
N_SAMPLES=32
SIM_MAX_STEPS=100
PAIRS_PATH="data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl"
ACTIVATIONS_ROOT="data/activations/libero_4suite_v4_combined"

usage() {
    grep -E '^#( |$)' "$0" | sed -E 's/^#( |$)//'
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sft-log)         SFT_LOG="$2";          shift 2 ;;
        --sft-dir)         SFT_DIR="$2";          shift 2 ;;
        --run-name)        RUN_NAME="$2";         shift 2 ;;
        --prior-eval-json) PRIOR_EVAL_JSON="$2";  shift 2 ;;
        --cache-path)      CACHE_PATH="$2";       shift 2 ;;
        --port)            PORT="$2";             shift 2 ;;
        --n-samples)       N_SAMPLES="$2";        shift 2 ;;
        --sim-max-steps)   SIM_MAX_STEPS="$2";    shift 2 ;;
        --pairs-path)      PAIRS_PATH="$2";       shift 2 ;;
        -h|--help)         usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "$SFT_LOG" || -z "$SFT_DIR" || -z "$RUN_NAME" ]]; then
    echo "ERROR: --sft-log, --sft-dir, --run-name all required" >&2
    usage
fi

cd /lambda/nfs/Natha/nla-groot

OUT_LOG="data/eval/${RUN_NAME}_auto_eval.log"
OUT_JSON="data/eval/${RUN_NAME}_cf_strided_cached.json"
CAPTIONS_JSON="data/eval/${RUN_NAME}_paired_captions.json"
LAUNCH_LOG="data/eval/${RUN_NAME}_cf_launch.log"

exec > >(tee -a "$OUT_LOG") 2>&1

echo "============================================================"
echo "auto_cf_eval_after_sft starting"
echo "  date(utc)      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  sft_log        = $SFT_LOG"
echo "  sft_dir        = $SFT_DIR"
echo "  run_name       = $RUN_NAME"
echo "  prior_eval     = $PRIOR_EVAL_JSON"
echo "  cache_path     = $CACHE_PATH"
echo "  port           = $PORT"
echo "  n_samples      = $N_SAMPLES"
echo "  sim_max_steps  = $SIM_MAX_STEPS"
echo "============================================================"

# 1. Wait for SFT done -----------------------------------------------------
echo "[$(date -u +%H:%M:%SZ)] waiting for 'SFT done' in $SFT_LOG ..."
ATTEMPTS=0
MAX_ATTEMPTS=4320   # 4320 * 10s = 12h, plenty of headroom past ETA
until grep -q "SFT done" "$SFT_LOG" 2>/dev/null; do
    if (( ATTEMPTS >= MAX_ATTEMPTS )); then
        echo "FATAL: 12h elapsed without SFT done; bailing"
        exit 3
    fi
    sleep 10
    ATTEMPTS=$((ATTEMPTS + 1))
done
echo "[$(date -u +%H:%M:%SZ)] SFT done detected after $((ATTEMPTS * 10))s"

# Brief pause so checkpoint save fully flushes
sleep 30

# 2. Verify checkpoints exist ---------------------------------------------
for f in "$SFT_DIR/ar/adapter_model.safetensors" "$SFT_DIR/av/adapter_model.safetensors"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f"
        exit 4
    fi
done
echo "[$(date -u +%H:%M:%SZ)] checkpoints verified ($SFT_DIR/{ar,av}/adapter_model.safetensors)"

# 3. AV caption diagnostic — paired captions for matched vs mismatched intent
#    Tells us early whether the codec is actually intent-conditional before
#    we burn 1h of sim rollouts.
echo "[$(date -u +%H:%M:%SZ)] running AV caption diagnostic..."
export PYTHONPATH=src
.venv/bin/python -u scripts/eval/av_caption_intent_diff.py \
    --sft-dir "$SFT_DIR" \
    --activations-root "$ACTIVATIONS_ROOT" \
    --pairs-path "$PAIRS_PATH" \
    --n-samples 10 \
    --out-json "$CAPTIONS_JSON" \
    || echo "(caption diagnostic failed -- continuing to CF eval)"

# 4. Stop any existing steer server, launch a new one on the new SFT -----
echo "[$(date -u +%H:%M:%SZ)] cleaning up any existing steer servers..."
scripts/eval/steer_server_status.sh --clean 2>/dev/null || true
# Brute force: kill any python listening on $PORT
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 5

echo "[$(date -u +%H:%M:%SZ)] launching steer server on $SFT_DIR (port $PORT)..."
scripts/eval/launch_steer_server.sh \
    --sft-dir "$SFT_DIR" \
    --port "$PORT" \
    --gpu-id 0 \
    --ready-timeout 360 \
    -- \
    --embodiment-tag LIBERO_PANDA \
    --steer-text-file scripts/eval/default_steer_boot.txt \
    --placement image_patch_all
echo "[$(date -u +%H:%M:%SZ)] steer server ready"

# 5. Populate no-steer cache from prior eval (instant if already populated)
if [[ -f "$PRIOR_EVAL_JSON" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] populating no-steer cache from $PRIOR_EVAL_JSON..."
    .venv/bin/python -u scripts/eval/populate_no_steer_cache.py \
        --prior-eval "$PRIOR_EVAL_JSON" \
        --cache-path "$CACHE_PATH" \
        || echo "(cache populator failed -- continuing without)"
fi

# 6. Run the CF eval ------------------------------------------------------
echo "[$(date -u +%H:%M:%SZ)] launching CF eval (image_patch_strided + strided_k=128)..."
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

setsid nohup .venv/bin/python -u scripts/eval/compare_cf_steer_checkpoints.py \
    --sft-dir "$SFT_DIR" \
    --grpo-av-dir "$SFT_DIR/av" \
    --pairs-path "$PAIRS_PATH" \
    --activations-root "$ACTIVATIONS_ROOT" \
    --n-samples "$N_SAMPLES" \
    --seed 0 \
    --conditions sft_av \
    --intent-arms matched,mismatched_source \
    --causal-arms semantic,no_steer \
    --sim-placement image_patch_strided \
    --strided-k 128 \
    --sim-batch-size 4 \
    --eval-protocol language_swap \
    --require-distinct-intents \
    --deterministic-order \
    --policy-port "$PORT" \
    --sim-max-steps "$SIM_MAX_STEPS" \
    --sim-timeout-s 600 \
    --sim-cache-path "$CACHE_PATH" \
    --out-json "$OUT_JSON" \
    > "$LAUNCH_LOG" 2>&1 < /dev/null &
disown -h $! 2>/dev/null || true
sleep 15
CF_PID=$(pgrep -f "compare_cf_steer.*${RUN_NAME}_cf" | head -1 || pgrep -f "compare_cf_steer_checkpoints.py" | head -1)
echo "$CF_PID" > "data/eval/${RUN_NAME}_cf.pid"
echo "[$(date -u +%H:%M:%SZ)] CF eval started; PID=$CF_PID, out=$OUT_JSON"

# 7. Wait for completion + summarize the final numbers ------------------
echo "[$(date -u +%H:%M:%SZ)] waiting for CF eval to write $OUT_JSON..."
while ! [[ -f "$OUT_JSON" ]]; do
    sleep 30
    if ! kill -0 "$CF_PID" 2>/dev/null; then
        # Process died without producing output -- check for errors
        echo "WARN: CF eval PID $CF_PID is gone but no output JSON; tailing log..."
        tail -20 "$LAUNCH_LOG"
        exit 5
    fi
done

echo "[$(date -u +%H:%M:%SZ)] CF eval finished, summarizing..."
.venv/bin/python <<PY
import json, statistics, math
d = json.load(open("$OUT_JSON"))
samples = d["samples"]
arms = {
    "M_sem":   "sft_av",
    "M_nost":  "sft_av__no_steer",
    "Mm_sem":  "sft_av__mismatched_source",
    "Mm_nost": "sft_av__mismatched_source__no_steer",
}
def get(s, key):
    c = s["conditions"].get(key)
    if c is None or c.get("error") is not None or "skipped_reason" in c:
        return None
    return c.get("r_sim")
sl, sg, lsw = [], [], []
for s in samples:
    vals = {k: get(s, arms[k]) for k in arms}
    if None in vals.values(): continue
    sl.append(vals["M_sem"] - vals["M_nost"])
    sg.append(vals["M_sem"] - vals["Mm_sem"])
    lsw.append(vals["M_nost"] - vals["Mm_nost"])
def stat(xs):
    n=len(xs); m=sum(xs)/n; s=statistics.pstdev(xs); se=s/math.sqrt(n-1) if n>1 else 0
    t=m/se if se>0 else float("inf"); w=sum(1 for x in xs if x>0); l=sum(1 for x in xs if x<0)
    return m,s,se,t,w,l,n
print(f"\n========================================")
print(f"v9_combined_12k CF eval result (n={len(sl)})")
print(f"========================================")
for label, xs in [("steer_lift  (M_sem-M_nost)", sl),
                  ("sem_gap     (M_sem-Mm_sem)", sg),
                  ("lang_swap   (M_nost-Mm_nost)", lsw)]:
    m,s,se,t,w,l,n = stat(xs)
    print(f"  {label:<32} mean={m:+.4f} std={s:.3f} se={se:.4f} t={t:+.2f} w/l={w}/{l}")
ca = (sum(sg)/len(sg)) - (sum(lsw)/len(lsw)) if sg and lsw else 0
print(f"  codec_above_lang                 {ca:+.4f}")
PY

echo "[$(date -u +%H:%M:%SZ)] all done. Result JSON: $OUT_JSON"

#!/usr/bin/env bash
# launch_arm_b_after_grpo.sh
#
# Wait for the currently running Arm A GRPO (PID 523222 by default) to exit,
# then launch Arm B SFT from the no-anchor labels into a new output dir.
#
# Background-safe: invoke with `nohup ... &` so it survives your session.
# Idempotent: refuses to launch a second time if the Arm B output dir already
# has a non-empty metrics.jsonl.
#
# Override env vars:
#   WAIT_PID         PID to poll (default: 523222 = current Arm A GRPO)
#   POLL_INTERVAL_S  Sleep between polls (default: 60)
#   SFT_OUT          Output dir (default: data/sft/libero_4suite_v4_no_anchor)
#   PYTHON_BIN       Python executable (default: .venv/bin/python)

set -euo pipefail
cd /home/ubuntu/nla-groot

WAIT_PID="${WAIT_PID:-523222}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
SFT_OUT="${SFT_OUT:-data/sft/libero_4suite_v4_no_anchor}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

LAUNCH_LOG="${SFT_OUT}_launcher.log"
SFT_LOG="${SFT_OUT}_sft.log"

mkdir -p "$(dirname "$SFT_OUT")"
exec >>"$LAUNCH_LOG" 2>&1

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log "launcher start: wait_pid=$WAIT_PID poll_interval_s=$POLL_INTERVAL_S sft_out=$SFT_OUT"

if [[ -s "${SFT_OUT}/metrics.jsonl" ]]; then
    log "REFUSE: ${SFT_OUT}/metrics.jsonl already exists and is non-empty. Delete the dir to relaunch."
    exit 1
fi

# Poll until WAIT_PID disappears from /proc.
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_INTERVAL_S"
done
log "PID $WAIT_PID is gone; starting Arm B SFT in 30s"
sleep 30   # let the GR00T VRAM settle

# Sanity: refuse to launch if a second SFT is already running.
if pgrep -f "run_sft.py.*$SFT_OUT" >/dev/null 2>&1; then
    log "REFUSE: an Arm B SFT for $SFT_OUT is already running."
    exit 1
fi

mkdir -p "$SFT_OUT"

# Launch Arm B SFT. Mirrors arm_b_sft from configs/ablations/anchor_ablation.yaml.
# Hyperparams match the V4 baseline (libero_4suite_v4_consistency_overnight)
# except:
#   - labels:                 labels_no_anchor.jsonl
#   - hard-neg index:         hard_negatives_no_anchor.jsonl
#   - position mix sampler:   50/50 last_text/image_patch (no anchor stratum)
#   - output dir:             data/sft/libero_4suite_v4_no_anchor
log "launching: PYTHONPATH=src $PYTHON_BIN scripts/training/run_sft.py --output-dir $SFT_OUT (log: $SFT_LOG)"

nohup env PYTHONPATH=src "$PYTHON_BIN" scripts/training/run_sft.py \
    --activations-root data/activations/libero_4suite_v4_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_combined/labels_no_anchor.jsonl \
    --stats-json       data/activations/libero_4suite_v4_combined/stats_pooled.json \
    --output-dir       "$SFT_OUT" \
    --image-patch-pooling mean_pool_image \
    --balance-position-mix \
    --position-mix-json '{"last_text": 0.5, "image_patch": 0.5}' \
    --ar-contrastive-weight 0.5 \
    --ar-nce-hard-negative-source topk_cosine \
    --ar-nce-hard-negative-index-path data/activations/libero_4suite_v4_combined/hard_negatives_no_anchor.jsonl \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --warmup-steps 500 \
    --total-steps 8000 \
    --eval-every 500 \
    --save-every 2000 \
    --log-every 5 \
    --eval-closed-loop \
    --closed-loop-temps 0.0 0.7 \
    --closed-loop-max-batches 64 \
    --ar-av-mix-max 0.4 \
    --ar-av-mix-warmup-frac 0.3 \
    --action-consistency-weight 0.1 \
    --action-consistency-every-n-steps 8 \
    --action-consistency-policy-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
    --action-consistency-embodiment-tag LIBERO_PANDA \
    --action-consistency-dataset-roots '{"goal": "third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot"}' \
    --action-consistency-suites goal \
    >"$SFT_LOG" 2>&1 &

SFT_PID=$!
disown "$SFT_PID" 2>/dev/null || true
log "Arm B SFT spawned: pid=$SFT_PID  log=$SFT_LOG"

# Quick sanity check: did it die within 3s on argparse / import errors?
sleep 3
if ! kill -0 "$SFT_PID" 2>/dev/null; then
    log "FAIL: Arm B SFT pid $SFT_PID died within 3s. Check $SFT_LOG."
    exit 1
fi
log "Arm B SFT alive after 3s; launcher exiting."

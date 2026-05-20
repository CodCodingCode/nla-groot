# V5 overnight guards (3 background scripts)

While you sleep, three watchdogs run in parallel with the main orchestrator:

| Script | Role |
|--------|------|
| `watch_v5_labels_guard.sh` | Retry failed/stalled V5 labeling per suite → validate → expand → merge → hard negs |
| `watch_v5_sft_guard.sh` | After labels ready: **~14h wall-clock SFT** on base `Qwen/Qwen3-4B-Instruct-2507`, up to 3 retries |
| `watch_v5_post_guard.sh` | After SFT: metrics check, AV samples, GRPO with retries |

## Launch

```bash
bash scripts/training/watch/launch_v5_guards.sh
```

PIDs: `logs/v5_guard/{labels,sft,post}_guard.pid`

## Flags (progress)

- `logs/v5_guard/labels_ready.flag` — combined labels + `hard_negatives_v5.jsonl`
- `logs/v5_guard/sft_started.flag` — SFT guard took ownership (orchestrator skips its short SFT)
- `logs/v5_guard/sft_success.flag` — AV+AR checkpoints + ≥500 steps logged
- `logs/v5_guard/pipeline_complete.flag` — GRPO finished (or best-effort)

## Monitor

```bash
tail -f logs/v5_guard/labels_guard.log
tail -f logs/v5_guard/sft_guard.log
tail -f logs/v5_guard/post_guard.log
tail -f logs/v5_overnight.log
```

## Env overrides

- `SFT_WALL_HOURS=14` — target SFT duration (default 14)
- `MAX_SFT_ATTEMPTS=3` — SFT retries on crash/OOM
- `PROBE_STEPS=80` — steps used to estimate `total_steps` from wall budget
- `GRPO_WALL_HOURS=4` — GRPO timeout per attempt

## Morning checklist

1. `cat logs/v5_guard/pipeline_complete.flag`
2. `wc -l data/labels/libero_4suite_v5_combined/labels.jsonl`
3. `tail data/sft/libero_4suite_v5_base_qwen/metrics.jsonl`
4. `ls data/sft/libero_4suite_v5_base_qwen/{av,ar}/`

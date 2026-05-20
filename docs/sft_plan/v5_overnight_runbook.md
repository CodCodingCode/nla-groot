# V5 overnight runbook

Unattended pipeline launched via:

```bash
nohup bash scripts/training/orchestrate_v5_overnight.sh > logs/v5_overnight.boot 2>&1 &
echo $! > logs/v5_overnight.pid
tail -f logs/v5_overnight.log
```

## Stages

1. **V5 step labeling** — 4 suites, `--prompt-mode v5`, `labels_steps.jsonl`
2. **Validate + expand** — 3 position rows/step, suite-prefixed ids
3. **Merge** — `data/labels/libero_4suite_v5_combined/labels.jsonl`
4. **Hard negatives** — `data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl`
5. **SFT** — **fresh** `Qwen/Qwen3-4B-Instruct-2507` LoRA (not V4 checkpoint), V5 prompts, `strided_image_multi` K=8
6. **Metrics check** — `scripts/ci/check_sft_metrics.py`
7. **GRPO** — recon reward, 500 steps, AR frozen

## Outputs

| Artifact | Path |
|----------|------|
| Per-suite steps | `data/labels/libero_4suite_v5/libero_<suite>/labels_steps.jsonl` |
| Combined labels | `data/labels/libero_4suite_v5_combined/labels.jsonl` |
| SFT | `data/sft/libero_4suite_v5_base_qwen/` |
| GRPO | `data/grpo/libero_4suite_v5_base_qwen_grpo/` |

## Env overrides

- `SFT_STEPS` (default 3000)
- `GRPO_STEPS` (default 500)
- `LABEL_CONCURRENCY` (default 64)
- `OPENAI_LABELING_MODEL`

## Note

Labeling ~51k timesteps is API-bound and may take many hours before SFT starts. Monitor `grep '\[orchestrate\]' logs/v5_overnight.log`.

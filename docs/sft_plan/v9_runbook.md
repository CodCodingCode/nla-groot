# v9 Retrain Runbook

v9 is v8's recipe minus the AR-layer truncation, with an option to use a
**magnitude-aware reconstruction loss** instead of plain MSE. Targets the
"high cosine, deeply negative FVE" pattern v8 ended in (val cosine 0.55,
FVE -82, val MSE 350 in raw activation units = ~9% RMS per-vector error
but magnitudes off).

There is no `--recipe v9` in `src/nla/training/recipes.py` yet. v9 is launched
as `--recipe v7` with the v8 deltas plus two new flags:
- `--ar-layers 0` (0 = no truncation, use all Qwen3-4B layers — the v9 lever)
- `--ar-loss-mode decomposed --ar-scale-weight 0.1` (magnitude-aware
  reconstruction; replace `mse` with `decomposed` to target the magnitude
  axis directly)

See:
- [docs/sft_plan/v9_overview.txt](v9_overview.txt) — design rationale
- [docs/sft_plan/v8_runbook.md](v8_runbook.md) — most flag rationale still applies
- [docs/train.md](../train.md) — canonical detached launch pattern

## Single-command launch (canonical)

```bash
cd /lambda/nfs/Natha/nla-groot
export PYTHONPATH=src   # MUST be exported (not inline) for setsid

setsid nohup .venv/bin/python -u scripts/training/run_sft.py \
  --recipe v7 \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v6_with_task/labels.jsonl \
  --output-dir       data/sft/v9_<run_name> \
  --stats-json       data/activations/libero_4suite_v4_combined/stats.json \
  --ar-nce-hard-negative-index-path data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl \
  --ar-spatial-n-positions 128 \
  --image-patch-pooling-strided-k 128 \
  --av-num-image-slots 128 \
  --ar-layers 0 \
  --ar-loss-mode decomposed \
  --ar-scale-weight 0.1 \
  --num-workers 8 \
  --action-consistency-every-n-steps 2 \
  --total-steps 6400 \
  --eval-every 400 \
  --save-every 800 \
  --max-val-items 512 \
  --action-consistency-policy-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
  --action-consistency-embodiment-tag LIBERO_PANDA \
  --action-consistency-dataset-roots '{"goal": "third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot", "object": "third_party/Isaac-GR00T/examples/LIBERO/libero_object_no_noops_1.0.0_lerobot", "spatial": "third_party/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot", "10": "third_party/Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot"}' \
  > data/sft/v9_<run_name>_launch.log 2>&1 < /dev/null &

sleep 5
pgrep -f "scripts/training/run_sft.py.*v9_<run_name>" | tail -1 > data/sft/v9_<run_name>.pid
ps -p $(cat data/sft/v9_<run_name>.pid) -o pid,ppid,tty,sid,stat
# Want: PPID=1 (or own bash that will exit), TT=?, STAT contains 's'.
```

## What changed from v8 — per setting

### AR architecture

| Setting | v8 | v9 | Why |
|---------|----|----|-----|
| `--ar-layers` | 16 | **0** (= no truncation, all Qwen3-4B layers) | Doubles AR's transformer depth, ~20% more trainable LoRA params. Bets that capacity is the bottleneck on the magnitude axis. |

### Reconstruction loss

| Setting | v8 | v9 | Why |
|---------|----|----|-----|
| `--ar-loss-mode` | (implicit `mse`) | **`decomposed`** | Splits direction (cosine) and scale (log-norm²) into separate terms. v8 ended at cosine 0.55 + FVE -82 — direction OK, magnitude badly off. Plain MSE entangles both; decomposed lets us push the magnitude term up explicitly. |
| `--ar-scale-weight` | n/a | **0.1** (default) | Weight on the log-magnitude term. Dial up (0.2 - 0.5) if magnitudes still off after a few hundred steps; dial down if direction collapses. |

### Throughput

| Setting | v8 | v9 (expected) | Why |
|---------|----|---------------|-----|
| AR forward time | baseline | ~2× | Twice the layers means ~2× compute on the AR pass |
| End-to-end step time | ~6 s | ~9-12 s (estimate) | AR is one of several heads; not all 2× hits the critical path |
| 6400-step run | ~12 hr | ~18-22 hr | Plan accordingly |

### Everything else

Unchanged from v8: K=128 spatial AR, v6 labels with `task:` bullet,
`action_consistency_every_n_steps=2`, `num_workers=8`, `--max-val-items
512`, `--eval-every 400`, `--save-every 800`. Hard-neg index reused as-is.

## Order of operations

Identical to v8 with one substitution at step 1:

1. **SFT v9** (~18-22 hr at 6400 steps)
   - Watch `closed_greedy/fve` and `closed_greedy/cosine` — both should
     move *up* simultaneously if the decomposed loss is doing its job.
     v8 had cosine high + FVE deep negative; v9 should narrow that gap.
   - New metric to watch: train `ar_mse` no longer means what it did in
     v8. When `--ar-loss-mode=decomposed`, `ar_mse` in the log is the
     direction + scale-weighted sum, not pure MSE. The val MSE is still
     measured the legacy way for comparability.
2. **Action-effect probe** (~10 min) — see docs/train.md §2. Same gate.
3. **Random-vector control** (~10 min) — see docs/train.md §3. Same gate.
4. **Spatial probe** (~10 min) — `--placement image_patch_strided`.
5. **CF sim eval** (`compare_cf_steer_checkpoints.py --sim-placement image_patch_strided --strided-k 128`).
6. **Position-embedding differentiation** check (instant).
7. **GRPO v9** — still gated on SFT showing nonzero `steer_lift`.

## What "success" means

Same publishable equation as v8:

```
matched_semantic_predicate_rate − matched_no_steer_predicate_rate ≥ +0.05
matched_semantic_predicate_rate − mismatched_predicate_rate       ≥ +0.05
matched_semantic_predicate_rate − matched_null_predicate_rate     ≥ +0.05
```

Plus a new internal check specific to v9:

```
closed_greedy/fve > 0.0   (v8 ended at -82; getting above 0 means we beat the per-dim mean baseline)
val MSE  / α²  < 0.005    (≈ 10% relative error per dim → ≈ 200 in raw units; v8 was ~350)
```

If v9 closes the FVE gap **and** moves `steer_lift > 0`, the headline
story changes from "direction-only codec" to "calibrated codec" — a
cleaner publishable result.

## Common v9 gotchas (additions on top of v8)

| Symptom | Cause | Fix |
|---|---|---|
| OOM at AV/AR load with `--ar-layers 0` | Full 36-layer AR + AV + frozen policy + DiT exceeds 80 GB | Drop AV LoRA rank, reduce batch from 4 → 2, or stay at `--ar-layers 16` and only flip `--ar-loss-mode` |
| AR forward step time jumps to 15+ s/step | Combined effect of full AR depth + `action_consistency_every_n_steps=1` | Keep `action_consistency_every_n_steps=2` (v8 default already) |
| `ar_mse` value at step 0 ~1.0 instead of ~3.0 | Decomposed loss range is ~[0, 2] (cosine ∈ [-1, 1]) vs MSE which depends on activation magnitudes. Expected. | None — sanity-check by watching `closed_greedy/cosine` and `closed_greedy/fve` move in tandem instead. |
| Loss stable but val MSE worse than v8 | `ar_scale_weight` too low; magnitude isn't getting enough gradient | Try `--ar-scale-weight 0.3` or `0.5` for next run |
| Loss explodes after a few hundred steps | `ar_scale_weight` too high — magnitude term dominates and direction collapses | Try `--ar-scale-weight 0.05` |

All v7/v8 gotchas still apply.

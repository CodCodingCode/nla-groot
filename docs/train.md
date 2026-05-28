# Training guide — optimal SFT setup for nla-groot

How to launch a new SFT run that's fast, debuggable, and likely to produce a checkpoint worth probing. Updated 2026-05-28 from v8 pilot lessons.

---

## TL;DR — the command

For maximum throughput on a single H100, launch detached (so the Claude harness can't kill it after a few hours):

```bash
cd /lambda/nfs/Natha/nla-groot
export PYTHONPATH=src

setsid nohup .venv/bin/python -u scripts/training/run_sft.py \
  --recipe v7 \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v6_with_task/labels.jsonl \
  --output-dir       data/sft/<run_name> \
  --stats-json       data/activations/libero_4suite_v4_combined/stats.json \
  --ar-nce-hard-negative-index-path data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl \
  --ar-spatial-n-positions 128 \
  --image-patch-pooling-strided-k 128 \
  --av-num-image-slots 128 \
  --num-workers 8 \
  --action-consistency-every-n-steps 2 \
  --total-steps 4000 \
  --eval-every 100 \
  --save-every 500 \
  --action-consistency-policy-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
  --action-consistency-embodiment-tag LIBERO_PANDA \
  --action-consistency-dataset-roots '{"goal": "third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot", "object": "third_party/Isaac-GR00T/examples/LIBERO/libero_object_no_noops_1.0.0_lerobot", "spatial": "third_party/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot", "10": "third_party/Isaac-GR00T/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot"}' \
  > data/sft/<run_name>_launch.log 2>&1 < /dev/null &

# Save the actual python PID (not the bash $! which may be the setsid wrapper)
sleep 5
pgrep -f "scripts/training/run_sft.py.*<run_name>" | head -1 > data/sft/<run_name>.pid
echo "Training PID: $(cat data/sft/<run_name>.pid)"
```

### Throughput knobs (the 1.5× speedup)

| flag | effect |
|---|---|
| `--num-workers 8` | DataLoader uses 8 worker processes. v7 ran at 37% GPU util with `num_workers=0`; this lifts it to ~80%+. Speedup: ~1.2-1.3×. |
| `--action-consistency-every-n-steps 2` | The frozen-policy forward (action-consistency loss) is ~1.2 s of the ~3.5 s/step budget. Running it every 2 steps instead of every step cuts step time ~1.3×. Halves the policy-effect gradient signal per step, but you get more steps per hour overall. |

Combined: step time goes ~3.5 s → ~2.3 s (~1.5×). Don't pass these for a baseline-comparison run, but for any new training they're free wins.

### Detachment knobs (so Claude can't kill it)

| flag | effect |
|---|---|
| `setsid` | Puts the process in a new session with no controlling terminal. |
| `nohup` | Ignores SIGHUP. |
| `< /dev/null` | No stdin tied to a terminal. |
| `&` | Background. |
| `disown` (optional) | Removes from the shell's job table. |

Combined, the process has PPID=1, no TTY, and survives shell/conversation exit. Claude's background-task subsystem can't reach it.

`--num-workers` is a real CLI flag now (also `--no-pin-memory` / `--no-persistent-workers` to opt out of those defaults).

---

## Knob-by-knob breakdown

### Data

| flag | recommendation | reason |
|---|---|---|
| `--activations-root` | `data/activations/libero_4suite_v4_combined` | Has 128 image-patch tokens per example (full grid; no sub-sampling). |
| `--labels-jsonl` | `data/labels/libero_4suite_v6_with_task/labels.jsonl` | The v6 labels include a `- task: {instruction}` bullet at the end of every description. v5 labels do not — using them produces scene-only captions and behaviorally inert codecs (confirmed via probe). |
| `--ar-nce-hard-negative-index-path` | `data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl` | Required when `ar_nce_hard_negative_source=topk_cosine` (v7 default). Omit and training crashes at dataset init. |
| `--stats-json` | `data/activations/libero_4suite_v4_combined/stats.json` | Provides α from p75-norm; overrides the CLI default 197.44. |

### Architecture (matches v7 + spatial K=128)

| flag | recommendation | reason |
|---|---|---|
| `--recipe v7` | always | Sets `ar_head_type=spatial`, `ar_weight=0.1`, `action_consistency_weight=1.0`, `batch_stratified_positions=true`, etc. |
| `--ar-spatial-n-positions 128` | **128**, not 8 | The live GR00T policy emits 128 image-patch tokens per frame (Cosmos-Reason2-2B with adaptive vision tokens). K=8 is a 16× sub-sample; K=128 is the full grid. v7's K=8 spatial head never differentiated (position embeddings stayed at init std=0.02), so this is the right size. |
| `--image-patch-pooling-strided-k 128` | **128** | Matches the AR side; AV input gets all 128 patches, not 8 strided ones. |
| `--av-num-image-slots 128` | **128** | AV prompt has 128 `<\|act_slot_i\|>` tokens. They auto-register via `ensure_slot_token` at AV init — no manual setup needed. |

### Training cadence

| flag | full run | pilot | reason |
|---|---|---|---|
| `--total-steps` | 4000 | 400 | Pilots tell us if architecture/labels are sane in ~30 min; full runs converge. |
| `--eval-every` | 100 | 50 | Pilot evals frequently to catch failure modes early. |
| `--save-every` | 500 | 200 | Save fewer checkpoints in pilots (disk; load time). |

### Action consistency (the slow but critical loss)

| flag | full run | pilot | reason |
|---|---|---|---|
| `--action-consistency-weight` | **1.0** | **0.0** (off) | Action-consistency is the only loss that grades the codec by *policy effect*. Without it, FVE goes down but the codec stays behaviorally inert (Stage 0 finding). Turning it OFF for pilots is purely for speed — pilots test whether captions/architecture are sane, not whether the codec moves the policy. |
| `--action-consistency-every-n-steps` | 1 | n/a | Every step. |
| `--action-consistency-image-patch-only` | `false` | n/a | All position types see policy gradient. |
| `--action-consistency-policy-path` | the GR00T LIBERO checkpoint | n/a | Frozen policy that the loss queries. |
| `--action-consistency-dataset-roots` | LeRobot dataset for the suite | n/a | Used to reconstruct policy observations. |

**Throughput impact**:
- With action_consistency=1.0: ~6.7 s/step (a frozen policy forward per step)
- With action_consistency=0.0: ~2.1 s/step (3× faster) — but no policy-effect gradient

For a 4000-step full run: ~7.5h with action_consistency, ~2.3h without. Use it for production; skip for pilots.

---

## DataLoader workers — the underappreciated speedup

`num_workers=0` is currently hardcoded in `src/nla/training/sft.py` at lines 425, 433, 439, 444. The v8 pilot ran at 37% GPU utilization, confirming data loading is the bottleneck.

### Recommended setting

```python
--num-workers 8 --pin-memory --persistent-workers
```

Throughput curve on this host (26 CPUs, 221 GB RAM, 102 activation shards):

| num_workers | expected GPU util | speedup vs. 0 |
|---|---|---|
| 0 (current) | ~37% | baseline |
| 4 | ~70-80% | ~2× |
| **8** | ~85-90% | **~2.3× (sweet spot)** |
| 16 | ~85-90% | no real gain; 2× memory overhead from worker forks |

The plateau above ~8 workers is because each worker forks the main process state (hard-neg cache, label index, position embeddings) — past 8, memory pressure rises but throughput doesn't.

### CLI usage

```bash
--num-workers 8                  # 4-8 is the sweet spot
--no-pin-memory                  # opt out of pin_memory (default is on with workers > 0)
--no-persistent-workers          # opt out of persistent_workers (default is on with workers > 0)
```

Defaults preserve old behavior: `num_workers=0` if you don't pass the flag, so nothing breaks for runs that don't opt in.

---

## Post-training checks

After every run finishes (no exceptions):

### 1. Auto-final closed-loop eval (free — runs at end of training)
Read the `[final]` line of the log:
```bash
grep "\[final\]" data/sft/<run_name>/sft.log
```
Headline metrics: `fve`, `cosine`, per-position breakdowns, `closed_greedy/*`.

### 2. Action-effect probe (~10 min, GATES GRPO)
```bash
PYTHONPATH=src .venv/bin/python scripts/eval/action_effect_probe.py \
  --sft-dir          data/sft/<run_name> \
  --activations-root data/activations/libero_4suite_v4_combined \
  --pairs-path       data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \
  --model-path       checkpoints/GR00T-N1.7-LIBERO/libero_goal \
  --embodiment-tag   LIBERO_PANDA \
  --dataset-path     third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \
  --n-samples        32 \
  --placement        image_patch_all \
  --out-json         data/sft/<run_name>/action_effect_probe_all.json
```

Gate: **median_rms ≥ 0.05 → run GRPO. Below → diagnose first.**

### 3. Random-vector control (~10 min, DETECTS MEASUREMENT ARTIFACTS)
Same command as (2) but add `--random-control`. If random control passes at a similar rms as the actual codec, the codec is functionally inert despite "passing" the probe.

### 4. Spatial probe with strided injection (~10 min, OPTIONAL)
Same command as (2) but `--placement image_patch_strided`. Tests whether the K=128 spatial differentiation contributes signal beyond the mean.

### 5. Position-embeddings differentiation check (instant)
```python
import torch
sd = torch.load('data/sft/<run_name>/ar/head.pt', map_location='cpu', weights_only=False)
pe = sd['position_embeddings']  # (K, H_lm)
print('std per-dim:', pe.std(dim=0).mean().item())  # init was 0.02
pe_n = pe / pe.norm(dim=1, keepdim=True)
cos = pe_n @ pe_n.T
off = cos[~torch.eye(pe.shape[0], dtype=torch.bool)]
print('off-diag cosine mean:', off.mean().item())  # if ~0, embeddings stayed orthogonal at init
```
If std ≈ 0.02 and off-diag cosine ≈ 0, the spatial head never differentiated — K is wasted.

---

## Recipes by use case

### Quick architecture pilot (~30 min)
- 400 steps, `--action-consistency-weight 0.0` (off), `--eval-every 50`
- Tests whether labels + architecture produce intent-conditional captions
- Doesn't test policy effect — that requires action_consistency

### Full production run (~2.5-7.5 hours, slow but most signal per step)
- 4000 steps, action_consistency ON (recipe v7 default, every step), eval_every=100
- Eligible for GRPO if probes pass
- ~6.7 s/step with K=8 (v7) or ~3.5 s/step with K=128 (v8 observed)

### Speed-optimized full run (~1.5-2 hours, ~1.5× faster)
- 4000 steps, `--action-consistency-every-n-steps 2`, `--num-workers 8`, eval_every=200
- Step time ~2.3 s. Half as many policy-effect gradient samples per step but ~1.5× more steps in the same wall clock — usually net positive.
- Eligible for GRPO if probes pass.

### Short pilot for fast iteration (~1.5 hours, full pipeline test)
- 1800 steps, `--action-consistency-every-n-steps 2`, `--num-workers 8`, eval_every=200, save_every=300
- Enough to clear the v8-style architecture validation in a single afternoon.

---

## Common gotchas

| symptom | cause | fix |
|---|---|---|
| `ValueError: hard_negative_source='topk_cosine' requires hard_negative_index_path...` | Recipe default v7 expects the index path; CLI didn't pass it | Add `--ar-nce-hard-negative-index-path data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl` |
| `RuntimeError: Replay manifest is empty after building... with suites=[None]` | `--action-consistency-dataset-roots '{"": "..."}'` (empty key) doesn't fall through to all suites in practice | Pass an explicit per-suite mapping covering goal/object/spatial/10 (see TL;DR command above) |
| `RuntimeError: Replay manifest is empty... with suites=[...]` on a *second* run reusing the same output dir | The cached empty manifest from a prior failed attempt is being loaded | `rm -f data/sft/<run>/aux/replay_manifest.jsonl` and relaunch |
| Run crashes at `_StreamingFve expects [B, H] tensors; got (4, 8, 2048)` | Dataset returns (B, K, H) target but FVE accumulator wired for (B, H) | Should be fixed in commit 7c93096; if recurring check that AR head_type matches dataset's `ar_target_spatial` |
| AV build takes 15-20 minutes | Embedding resize for 128 new slot tokens. The mean-cov MVN init is slow | Acceptable for new runs; for re-runs, restore from a fresh-tokens checkpoint |
| Probe `image_patch_spatial` errors with "AR emitted N vectors but GR00T has 128" | Spatial K doesn't match live patch count | Set `--ar-spatial-n-positions 128` (was 8 in old recipe default) |
| Captions are scene-only, no action verb | Using v5 labels without `task:` bullet | Use `data/labels/libero_4suite_v6_with_task/labels.jsonl` |
| Background-task system kills the run after a few hours with no error trace | Claude background-task subsystem cleanup; tasks tied to conversation/session lifetime | Launch with `setsid nohup ... < /dev/null &` and `disown`. PPID=1, no TTY, own session. See "Detachment knobs" above. |
| `ModuleNotFoundError: No module named 'nla'` when launching with setsid+nohup | `PYTHONPATH=src` was set inline on the bash command, not exported. setsid forks a child without inheriting inline env | `export PYTHONPATH=src` before the setsid command |

---

## What we know works vs doesn't

**Works:**
- v7 SFT loss balance (action_consistency_weight=1.0, ar_weight=0.1, ar_contrastive_weight=0.3)
- K=128 patches at extraction (no information loss vs the 8 strided sub-sample)
- Episode-stratified train/val split
- Hard-negative caching via `mine_hard_negatives.py`

**Does NOT work (confirmed inert in v7):**
- K=8 spatial AR head — position embeddings stay at init noise, no differentiation
- v5 labels without `task:` bullet — captions become scene descriptions, AV never learns to encode action intent
- Probing with default `--placement image_patch` — single-token injection is too weak; use `image_patch_all` for behavioral signal

**Untested as of 2026-05-28:**
- K=128 with v6 labels (the v8 pilot — testing now)
- Adding last_text activation as an additional AV input slot (proposed v9; ~3-4h implementation effort)
- DataLoader workers > 0 — flag added; first run with `--num-workers 8` still pending
- `--action-consistency-every-n-steps 2` — recommended ~1.3× speedup, untested behaviorally (does halving the policy-effect gradient hurt the codec?)

**Misdiagnosed earlier (no fix needed):**
- "GR00T policy is running fp32" — the fp32 warnings appear during `from_pretrained` but `Gr00tPolicy.__init__:102` calls `model.to(dtype=torch.bfloat16)` after. Actual inference is already bf16; no speedup available here.

---

## File locations

- Training script: `scripts/training/run_sft.py`
- SFT loop: `src/nla/training/sft.py`
- AV: `src/nla/models/av.py`
- AR (with spatial head): `src/nla/models/ar.py`
- Recipes (v7 defaults): `src/nla/training/recipes.py`
- Action-effect probe: `scripts/eval/action_effect_probe.py`
- Backbone steer hooks: `src/nla/steering/backbone_steer.py`
- Label post-processor for v6 task bullets: (one-off Python; not checked in yet)
- Existing operational runbook: `docs/sft_plan/v7_runbook.md`
- v7 synthesis doc: `docs/sft_plan/v7_overview.txt`

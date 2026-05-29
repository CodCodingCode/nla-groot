# v8 Retrain Runbook

v8 is v7's recipe sized to GR00T's actual patch grid (K=128) on v6 labels
(with a `task:` bullet), launched detached so the harness can't kill it,
and using DataLoader workers + every-other-step action-consistency for
~1.5× throughput.

There is no `--recipe v8` in `src/nla/training/recipes.py` yet. v8 is launched
as `--recipe v7` with the v8 deltas passed explicitly on the CLI. If the
1800-step run clears the action-effect gate, promoting the v8 deltas into a
`V8_SFT_DEFAULTS` block is a 10-minute follow-up.

See:
- [docs/sft_plan/v8_overview.txt](v8_overview.txt) — design synthesis
- [docs/sft_plan/v7_runbook.md](v7_runbook.md) — most flag rationale still applies
- [docs/train.md](../train.md) — canonical launch command + post-training checks

## Single-command launch (canonical)

```bash
cd /lambda/nfs/Natha/nla-groot
export PYTHONPATH=src   # MUST be exported (not inline) for setsid

setsid nohup .venv/bin/python -u scripts/training/run_sft.py \
  --recipe v7 \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v6_with_task/labels.jsonl \
  --output-dir       data/sft/v8_<run_name> \
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
  > data/sft/v8_<run_name>_launch.log 2>&1 < /dev/null &

sleep 5
pgrep -f "scripts/training/run_sft.py.*v8_<run_name>" | head -1 > data/sft/v8_<run_name>.pid
echo "Training PID: $(cat data/sft/v8_<run_name>.pid)"
```

For a shorter test run, swap `--total-steps 4000 --eval-every 100 --save-every 500`
for `--total-steps 1800 --eval-every 400 --save-every 300` (the short-pilot
recipe; ~2 hr instead of ~4.5 hr).

## What changed from v7 — per setting

### Architecture sizing

| Setting | v7 | v8 | Why |
|---------|----|----|-----|
| `--ar-spatial-n-positions` | 8 | **128** | GR00T emits 128 image-patch tokens, not 8. K=8 was a 16× sub-sample; position embeddings never differentiated (std stayed at init=0.02). |
| `--image-patch-pooling-strided-k` | 8 | **128** | AV input gets all 128 patches, not a strided sub-sample |
| `--av-num-image-slots` | 8 | **128** | One `<\|act_slot_i\|>` token per patch (auto-registered via `ensure_slot_token` at AV init) |

### Labels

| Setting | v7 | v8 | Why |
|---------|----|----|-----|
| `--labels-jsonl` | `data/labels/libero_4suite_v5/labels.jsonl` | `data/labels/libero_4suite_v6_with_task/labels.jsonl` | v6 appends `- task: {instruction}` to every example. Without it, AV learns "describe scene"; codecs come out behaviorally inert at action-effect probe. |

### Throughput

| Setting | v7 | v8 | Why |
|---------|----|----|-----|
| `--num-workers` | 0 | **8** | DataLoader workers; GPU util 37% → ~85%. ~1.2–1.3× faster. Recipe defaults to 0 to preserve old behavior; passing this is opt-in. |
| `--action-consistency-every-n-steps` | 1 | **2** | The frozen-policy forward is ~1.2 s of the ~3.5 s/step budget. Every-2 cuts step time ~1.3×. Halves the policy-effect gradient signal per step; net positive in wall clock. |

Combined: step time ~6.7 s → ~4.1 s (observed on `v8_libero_4suite`).

### Detachment

| Setting | v7 | v8 | Why |
|---------|----|----|-----|
| Launch wrapper | `python ... &` | **`setsid nohup ... < /dev/null &`** | v7 runs were silently killed mid-training by Claude's background-task subsystem. PPID=1 + own session + no TTY survives shell/harness exit. |
| `PYTHONPATH=src` | inline | **`export PYTHONPATH=src`** | setsid forks without inheriting inline env. Inline `PYTHONPATH=src setsid nohup ...` yields `ModuleNotFoundError: No module named 'nla'`. |

### Everything else

Unchanged from v7. `action_consistency_weight=1.0`, `ar_weight=0.1`,
`ar_contrastive_weight=0.3`, `ar_av_mix_max=0.7`, `learning_rate=5e-5`,
`balance_position_mix=True`, `batch_stratified_positions=True`,
`split_by=episode`, `eval_closed_loop=True`, etc. — all inherited from the
v7 recipe.

## Required external work

All v7 prerequisites still apply. v8 adds **none**. The v5 hard-neg index
(`data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl`) is
reused as-is.

The one v7 prerequisite that is currently still satisfied without code is
the v6 labels JSONL. The post-processor that produced v6 from v5 (appending
the `- task:` bullet) is a one-off Python script that is not yet checked
in. If v6 labels are ever regenerated, that script needs to land first.

## Order of operations

Identical to v7 with one substitution at step 1:

1. **SFT v8** (~4.5 hr at 4000 steps, ~2 hr at 1800 steps)
   - Watch `data/sft/v8_<run>/metrics.jsonl` for the `action_consistency_loss`
     curve.
   - Watch `closed_greedy/fve` and `closed_greedy/fve/position=image_patch`
     at every val step.
2. **Action-effect probe** (~10 min) — see docs/train.md §2.
   - Gate: `median_rms ≥ 0.05` across n=32 CF pairs with `--placement image_patch_all`.
   - Note: this probe mean-pools the K=128 spatial AR output → 1 vector;
     it's a sanity gate, not the architecturally-correct test.
3. **Random-vector control** (~10 min) — see docs/train.md §3.
   - If random control passes at similar rms as the actual codec, the codec
     is functionally inert despite "passing" the probe.
4. **Spatial probe** (~10 min) — `--placement image_patch_strided`.
   - The architecturally-correct test: K=128 vectors at K=128 patch
     positions, exactly how v8 was trained. Promoted from "optional" after
     commit 354b5ca enabled the full `[B, K, H]` path through the sim
     wrapper.
5. **CF sim eval** with `image_patch_strided` (~30-60 min for n=32).
   - `compare_cf_steer_checkpoints.py --sim-placement image_patch_strided
     --strided-k 128`. Uses all K=128 patch positions end-to-end (no
     mean-pool). Returns `steer_lift_predicate` and `semantic_gap_predicate`.
6. **Position-embedding differentiation check** (instant) — see docs/train.md §5.
   - If `std(pe) ≈ 0.02` and off-diagonal cosine ≈ 0, the spatial head never
     differentiated — K is wasted.
7. **GRPO v8** — *not started.* GRPO is on hold until SFT alone moves
   axis-2 (judge grounding). Stage 0 showed AR injection is inert at
   trained α, so GRPO won't rescue an SFT that didn't shift grounding.

## Common v8 gotchas (additions on top of v7)

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nla'` after setsid+nohup launch | `PYTHONPATH=src` set inline, not exported | `export PYTHONPATH=src` before the setsid command |
| Run silently disappears after ~few hours, no error trace | Claude background-task subsystem cleanup | Launch with `setsid nohup ... < /dev/null &`; verify PPID=1 with `ps -o ppid= <pid>` |
| Probe `image_patch_spatial` errors "AR emitted N vectors but GR00T has 128" | K mismatch | `--ar-spatial-n-positions 128` (was 8 in old recipe) |
| Sim CF eval fails: `steer_h_batch ndarray must be [B,H]; got (B, 128, 2048)` | Sim wrapper used to reject 3D AR output | Fixed in commit 354b5ca. If you see this on an older checkout, pull main or pass `--sim-placement image_patch_all` to force the mean-pool fallback (lossy — discards K=128). |
| Sim CF eval runs but `steer_lift ≈ 0` despite probe passing | Likely running the mean-pool fallback (`image_patch_all`) instead of the architecturally-correct path | Re-run with `--sim-placement image_patch_strided --strided-k 128`. The mean-pool path tests a degraded codec. |
| Captions are scene-only, no action verb | Using v5 labels without `task:` bullet | `--labels-jsonl data/labels/libero_4suite_v6_with_task/labels.jsonl` |
| AV build takes 15–20 minutes on first launch | Embedding resize for 128 new slot tokens; mean-cov MVN init is slow | Acceptable on first launch; reuse a fresh-tokens checkpoint to skip |
| `[hard-neg topk_cosine] 1/95430 anchors have no admissible negatives; fall back to repeating anchor's caption` | Single edge-case row in the index | Harmless; warning only |

All v7 gotchas in [v7_runbook.md](v7_runbook.md) still apply.

## What "success" means in one equation

Unchanged from v7. On held-out CF samples (n ≥ 32):

```
matched_semantic_predicate_rate − matched_no_steer_predicate_rate ≥ +0.05
matched_semantic_predicate_rate − mismatched_predicate_rate       ≥ +0.05
matched_semantic_predicate_rate − matched_null_predicate_rate     ≥ +0.05
```

If v8 hits all three, the codec works and we promote v8 deltas into
`V8_SFT_DEFAULTS` in `recipes.py`. If not, the next iteration changes
the bottleneck — most likely candidates:

- Labels with explicit per-patch spatial bullets (v7_runbook.md §"Image_patch
  caption refresh")
- last_text as an additional AV input slot (the v9 proposal)
- action_consistency at every step (revert the v8 speed knob)

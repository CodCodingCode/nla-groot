# Stage 4 Runbook — Temporal Context Window (conditional)

## When to actually run this

Stage 4 in the plan ([plan file](../../../../home/ubuntu/.claude/plans/lets-exute-the-plan-fizzy-puzzle.md))
is **gated on Stage 3 results**:

> "Only invoke this stage if Stage 3 makes image_patch grounding work but
> trajectories are still inconsistent across time."

Concretely, run Stage 3 first. The decision rule:

| Stage 3 outcome | Stage 4 action |
|---|---|
| `retrieval_margin_image_patch ≥ 0.10` **and** `Δ_cw ≥ +5pp` **and** rollouts look directional across consecutive frames | **Skip Stage 4.** Ship the result. |
| `retrieval_margin_image_patch ≥ 0.10` but rollouts oscillate (jumpy actions, contradictory steers within an episode) | **Run Stage 4.** Temporal mismatch is the next bottleneck. |
| `retrieval_margin_image_patch < 0.10` | **Don't run Stage 4 yet.** The codec still isn't grounded; temporal context will not save it. Re-evaluate Stages 0–3. |

The "jumpy actions" check has a concrete metric in the rollout summaries:
`temporal_consistency_cosine = mean(cos(ĥ_t, ĥ_{t+1}))` over a steered
rollout. If that value is below ~0.85 while single-frame retrieval passes,
Stage 4 is justified.

## What Stage 4 changes (architectural diff sketch)

The current pipeline treats every (h, caption) pair as IID. Stage 4 makes
it a **3-frame window**: `(h_{t-1}, h_t, h_{t+1})` → `(text_{t-1}, text_t, text_{t+1})`
on both AV and AR sides.

### Extraction ([src/nla/extraction/sampler.py](../../src/nla/extraction/sampler.py))

- **Today:** one sample per (episode, step).
- **Stage 4:** one sample per (episode, step) but each sample carries
  `features_window: [T=3, N, D]` instead of `features: [D]`. Edge steps
  (t=0, t=T-1) pad with the boundary frame (repeat) so every window has
  shape `[3, N, D]`.
- New CLI flag: `--temporal-window-size 3` (default 1 = legacy behavior).
- Storage: each row's tensor file becomes 3× larger. Use chunked sharding
  (see [src/nla/extraction/storage.py](../../src/nla/extraction/storage.py))
  to keep individual shard sizes bounded.

### AV input ([src/nla/models/av.py](../../src/nla/models/av.py))

- Today: single activation slot in the prompt.
- Stage 4: **three slots** rendered as `[h_{t-1}] [h_t] [h_{t+1}]` with a
  small text scaffold like "Past:" / "Now:" / "Next:" between them. Re-use
  the existing K-slot machinery (`--av-num-image-slots`) by setting K=3
  for the temporal axis.
- For image_patch + temporal, the prompt holds `3 × K_spatial` slots if
  spatial AR is on. Cap K_spatial at extraction time to avoid prompt blowup.

### AR output ([src/nla/models/ar.py](../../src/nla/models/ar.py))

- Stage 3 introduced `head_type='spatial'` → `(B, N, D)`.
- Stage 4 extends to `head_type='spatial_temporal'` → `(B, T, N, D)` with
  `temporal_window_size: int = 3` on `ARConfig`.
- Loss: per-(t, position) MSE against the windowed target.
- InfoNCE: mean-pool over both T and N for per-row identity (the contrastive
  term scores "this row vs. other rows" — temporal/spatial structure is
  handled by the MSE term).

### Steering ([src/nla/steering/backbone_steer.py](../../src/nla/steering/backbone_steer.py))

- Inject **only the middle frame** at deployment: `ĥ_for_inject = out[:, T//2]`.
- The past/future frames are training-only context that improves what the
  middle frame encodes; they don't get injected because the live policy
  only has one frame of vision at the current step.
- New placement: `image_patch_spatial_temporal` (an extension of Stage 3's
  `image_patch_spatial`) that accepts `(N, D)` and ignores any extra
  temporal dim if the caller passed the raw `(T, N, D)` tensor (slice to
  the middle frame).

### Counterfactual mining ([scripts/training/mine_grpo_counterfactual_pairs.py](../../scripts/training/mine_grpo_counterfactual_pairs.py))

- Today: one CF pair per (source, target) at a single step.
- Stage 4: emit pairs around the chosen step (t-1, t, t+1) so the GRPO
  sim rollout sees a coherent window. The `position_index` field becomes
  the **middle** step's index; `temporal_window_indices: [int, int, int]`
  is added alongside.

## New eval metric

`temporal_consistency_cosine` — extend
[scripts/eval/closed_loop_retrieval.py](../../scripts/eval/closed_loop_retrieval.py)
with a `--temporal-diagnostics` flag (mirroring Stage 3's
`--spatial-diagnostics`). It computes:

```
mean_t cos(ĥ_t, ĥ_{t+1})        # how smoothly the middle-frame prediction
                                  # evolves under tiny text perturbations of
                                  # the same scene
```

Healthy: ≥ 0.85. A spatial-only AR will produce ĥ that flips harshly
between consecutive steered actions even when the steer text is held
constant — the symptom Stage 4 is supposed to fix.

## Risks and reasons to stop short of Stage 4

1. **3× compute** at training time (window expansion).
2. **3× storage** for the extracted activations.
3. The temporal head **can collapse** to mean-over-window (each `t` slot
   outputs the same vector) if the supervision signal is weak. Watch
   `temporal_consistency_cosine_std_across_t` in the spatial-temporal
   diagnostic — near-zero is healthy collapse-resistance, paradoxically.
4. If the underlying issue is **dose** (caught by Stage 0), no amount of
   temporal context will help.

If the team faces a paper deadline and Stage 3 is borderline-PASS, **ship
on Stage 3** and defer Stage 4 to a follow-up.

## Concrete next-action when triggered

1. Add `temporal_window_size` arg to extraction CLI; re-extract activations.
2. Bump `ARConfig.temporal_window_size` and add `head_type='spatial_temporal'`.
3. Train one short pilot (10k steps) on the new windowed data with spatial+temporal
   AR head and the existing image_patch-only filter (Stage 2).
4. Re-run V3 scorecard with both `--spatial-diagnostics` and
   `--temporal-diagnostics`. Check whether the temporal_consistency_cosine
   moves above 0.85.
5. Re-run dose sweep ([scripts/eval/nla_steer_alpha_sweep.py](../../scripts/eval/nla_steer_alpha_sweep.py))
   on the new checkpoint with the same `image_patch_spatial_temporal`
   placement.

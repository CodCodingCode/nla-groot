# 09 — Action-head LoRA, Phase 1: positive steering on libero_goal

**Status:** active follow-on to
[`08_positive_steering_followon.md`](./08_positive_steering_followon.md).
This doc is *Step 4* from doc 08 promoted to Phase 1 — by user request
(see chat 2026-05-17). The other steps in doc 08 remain on the backlog.

**Created:** Sat May 17 2026, in parallel with the V4 SFT launch.

---

## 1. Goal

Cause the LIBERO-goal policy to **actually pick up the re-prompted
object** instead of just suppressing the prior task. Concretely:

```
HEADLINE METRIC
  sim_correct_minus_wrong  on  libero_goal
  current  ≈ 0.0     (V1 and V3, per
                      data/eval/steerability_v1_vs_v3/comparisons/v1_vs_v3_all_seeds.mp4)
  target   ≥ 0.2     (any single libero_goal seed, any AR/AV checkpoint
                      from V4 SFT or later)
```

`sim_correct_minus_wrong` is already produced by the V3 scorecard
(`scripts/eval/build_v3_scorecard.py`); reuse it unchanged.

## 2. Why this and not steps 2/3 from doc 08

Doc 08 ranks counterfactual labels (step 2) and patch-level injection
(step 3) as cheaper. We are jumping to step 4 because:

* The user has watched two A/B videos and the failure mode is
  **the action head ignores the steered activation when the visual
  context disagrees**. That is not a label-quality issue; it is a
  training-distribution issue *of the action head itself*.
* V4 SFT only retrains AR/AV. The action head's weights have never
  seen a `(h_for_caption_X, scene_showing_Y)` mismatched pair, so we
  expect V4 alone to move the needle marginally (per the user's own
  Best-case framing).
* Steps 2 + 3 from doc 08 are still on the backlog and remain
  preferred next-up. Phase 1 here is the **floor**, not the ceiling.

## 3. Scope freezes

| Knob                    | Decision                                  |
| ----------------------- | ----------------------------------------- |
| Suite                   | `libero_goal` only                        |
| Wallclock budget        | One overnight cycle (~8 h on the GPU)     |
| AR/AV checkpoint        | V4 SFT (will be ready by morning)         |
| Backbone                | Frozen, untouched                         |
| AR/AV LoRA              | Frozen during Phase 1                     |
| Action-head LoRA target | `model.transformer_blocks.*.attn2.*` only |
| Tune projector/vlln     | Both frozen                               |
| Sim-eval scope          | The V3 steerability A/B suite, 4 seeds    |
| Acceptance              | `sim_correct_minus_wrong ≥ 0.2` on ≥1 seed|

Anything outside this table is out-of-scope for Phase 1 and is tracked
as a follow-on at the bottom of this doc.

## 4. Pipeline

Four steps. The first two can start *tonight* in parallel with V4 SFT
because they don't need V4 weights. The third blocks on V4 SFT
finishing. The fourth is sim eval.

### 4.1  Cross-task demo pairing (offline, no GPU, ~30 min)

LIBERO-goal has 10 tasks, each with ~50 successful demos. The kitchen
layout is shared, the *target object* differs. We mine synthetic
`(scene_t, caption_alt, action_alt)` triples by aligning timesteps
*across* tasks:

```text
for episode E_A in tasks A in libero_goal:
  for episode E_B in tasks B ≠ A with the same scene_id as E_A:
    align E_A and E_B by length-normalized step index;
    for each aligned (t_A, t_B):
      yield (
        scene_image = E_A.image[t_A],
        instruction = E_A.task_text,            # the actual prompt
        caption_alt = E_B.task_text,            # the "what we want to do" prompt
        action_alt  = E_B.action[t_B],          # the action the OTHER demo took
      )
```

Filtering rules (keep cheap):

* Only pair episodes whose initial gripper pose differs by < 5 cm L2
  (so `action_alt` is approximately realisable from `scene_t`'s arm
  state — same starting condition).
* Skip the last 25% of each demo (the alt action's terminal grasp
  is meaningless if the target object isn't in `scene_t`'s reach
  envelope).
* Drop any caption_alt whose target object is not visually present in
  `scene_t` — checked via the existing V3 attribute probe's object
  vocabulary (`scripts/eval/probe_h_attributes.py` already extracts
  per-frame object tags from the V4 labels).

**Deliverable:** `scripts/training/mine_cross_task_pairs.py` and
`data/synthetic_steering/libero_goal_pairs.jsonl` with one row per
`(scene_t, caption_alt, action_alt, scene_id, source_demos)` tuple.
Target volume: 30–50k rows after filtering.

**Cost:** CPU-only pass over the cached LeRobot LIBERO-goal datasets
at `~/.cache/huggingface/hub/datasets--IPEC-COMMUNITY--libero_goal_no_noops_1.0.0_lerobot`.

### 4.2  Sim-eval baseline rerun (in parallel with V4 SFT, ~1 h GPU)

Before we touch the action head, lock down today's number.

* Run `scripts/eval/closed_loop_sim_ab.py` on the **current** GR00T
  checkpoint with the V3 AR/AV steering vectors, 4 seeds × the
  V3 scorecard's libero_goal pair list.
* Pipe through `scripts/eval/build_v3_scorecard.py --suite libero_goal`.
* Snapshot `sim_correct_minus_wrong` as `baseline_pre_phase1.json`.

This is the "before" number. The acceptance criterion is delta vs this
baseline. If the V3 sim run already shows `> 0.0`, we tighten the
target.

### 4.3  Action-head LoRA fine-tune (blocks on V4 AR; ~3 h GPU)

Once V4 SFT publishes an AR checkpoint:

1. **Materialise targets**: for each row in `libero_goal_pairs.jsonl`,
   compute `ĥ_alt = V4_AR(caption_alt)`. This is one forward pass per
   row; ~10 min on a single GPU at batch 64.
2. **Freeze everything except DiT cross-attention LoRA**:

   ```python
   for name, p in policy.named_parameters():
       p.requires_grad = False
   # then add LoRA adapters via peft on:
   target_modules = [
       "action_head.model.transformer_blocks.*.attn2.to_q",
       "action_head.model.transformer_blocks.*.attn2.to_k",
       "action_head.model.transformer_blocks.*.attn2.to_v",
       "action_head.model.transformer_blocks.*.attn2.to_out.0",
   ]
   # rank=16, alpha=32 — half what AR/AV use; the DiT is wider so the
   # rank-to-dim ratio matches.
   ```

   The DiT block layout is half cross-attention / half self-attention
   when `interleave_self_attention=True` (see
   `third_party/Isaac-GR00T/gr00t/model/modules/dit.py:260`). We
   *only* tune the cross-attention layers, because those are the
   bottleneck between backbone activation `ĥ` and predicted action.

3. **Replace the encoder-hidden-states with `ĥ_alt` *patched into* the
   real backbone output for that frame**: at training time we run the
   normal backbone forward on `scene_t`, then overwrite the
   `image_patch` positions of the resulting `[T, H]` tensor with
   `ĥ_alt` (mean-broadcast over all image-patch tokens, matching the
   V4 mean-pool injection at inference). `last_text` and `anchor`
   tokens are left as the backbone produced them.

   This single training-time choice is the whole reason Phase 1 might
   work: the action head *sees the same residual stream layout it
   sees at steering time*, then is asked to produce the action that
   would be correct for the alt-task. Today the action head has
   never seen this layout during training.

4. **Loss**: the GR00T flow-matching action loss as defined in
   `Gr00tN1d7ActionHead.forward` (see
   `third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py:168`),
   targeting `action_alt`. No auxiliary terms.

5. **Optimiser & schedule**: AdamW, lr=1e-4, warmup 100 steps, cosine
   decay, batch 8, ~5k steps. ~3 h on one H100.

**Deliverable:**
`data/grpo/libero_goal_action_lora_phase1/checkpoint-final/`,
plus a `train_log.jsonl` and a `config.json` recording the exact LoRA
target list.

### 4.4  Sim A/B + scorecard (~1 h GPU)

* Re-run `closed_loop_sim_ab.py` with the LoRA-tuned action head
  loaded on top of V4 AR/AV, same seed list as 4.2.
* Pipe through `build_v3_scorecard.py` again.
* Final acceptance: report `sim_correct_minus_wrong` and
  `sim_correct_success_rate`; the former must clear 0.2 on at
  least one seed for Phase 1 to be called done.
* If it passes, also record `sim_wrong_minus_baseline` so we can
  see whether positive steering came from *adding* correct grasps
  or from *removing* wrong grasps. We want the former.

## 5. Risks and what we'll do about each

| Risk                                                                          | Mitigation                                                                                                    |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Cross-task pairs are nonsense — gripper poses don't actually align            | The < 5 cm initial-pose filter in 4.1. If it kills > 80% of pairs, fall back to same-scene + same-task pairs. |
| LoRA-only DiT can't represent the shift                                       | We measure train loss; if it plateaus high at step 1k, unfreeze `attn1` (self-attn) too as Phase 1.5.         |
| AR's `ĥ_alt` is still degenerate even after V4 mean-pool                      | Doc 08 step 2 (counterfactual labels). We will run a 1k-row spot-check on AR's caption→ĥ quality before 4.3. |
| Sim eval flakes (a known issue on libero_goal seed 3)                         | Run 4 seeds, report best-of-4 for the acceptance call, full distribution for the writeup.                     |
| LoRA changes safety properties (collision rate, gripper-jerk)                 | Sub-band of the scorecard: collisions per episode; gate Phase 2 on `collisions ≤ baseline + 10%`.             |
| V4 SFT itself doesn't finish overnight                                        | Phase 1 4.1 and 4.2 still complete; 4.3/4.4 slip by one day. No work is wasted.                               |

## 6. What we will not do in Phase 1

Each is queued behind Phase 1's outcome:

* Multi-suite generalisation (object/spatial/10). If Phase 1 wins
  on goal, *then* we replicate on object as Phase 2a.
* Full DiT fine-tune (no LoRA). LoRA first, full FT only if Phase 1
  plateaus and we have a budget headroom.
* DiT cross-attention key-bias (doc 08 step 5). Lowest-priority.
* Patch-level injection at *inference* time (doc 08 step 3). Phase 1
  uses pooled injection because it matches the V4 SFT decision; if
  Phase 1 fails we revisit step 3 before step 5.
* Counterfactual *labels* (doc 08 step 2). We rely on cross-task
  *demos* as a coarser substitute for "given this scene, what would
  a different task look like" — cheaper because no LLM labelling
  pass is needed.
* GRPO. Phase 1 must show *supervised* positive steering before we
  spend GRPO compute on it.

## 7. Concrete tonight-to-morning timeline

| Hour      | Step                                                                       |
| --------- | -------------------------------------------------------------------------- |
| 0–0.5     | 4.1 cross-task pair mining (CPU)                                           |
| 0–0.5     | (parallel) Author `mine_cross_task_pairs.py`                               |
| 0.5–1.5   | 4.2 baseline sim A/B + scorecard                                           |
| 0–overnight | V4 SFT continues on its own GPU                                         |
| morning   | V4 AR checkpoint ready                                                     |
| +0–0.5 h  | Caption→ĥ pass for the 30–50k pairs                                        |
| +0.5–3.5 h| 4.3 LoRA training                                                          |
| +3.5–4.5 h| 4.4 sim A/B + scorecard                                                    |
| +4.5–5 h  | Writeup → `data/grpo/libero_goal_action_lora_phase1/REPORT.md`             |

If 4.4 hits the 0.2 bar, we trigger Phase 2a (replicate on object).
If not, we read the failure pattern off the scorecard and decide
between Phase 1.5 (unfreeze DiT self-attn), doc 08 step 3
(patch-level injection at inference), or doc 08 step 2
(counterfactual labels). The decision rule lives in §5.

## 8. Provenance + cross-refs

* Origin: `docs/sft_plan/08_positive_steering_followon.md` step 4.
* Failure mode video:
  `data/eval/steerability_v1_vs_v3/comparisons/v1_vs_v3_all_seeds.mp4`.
* V4 SFT recipe (this Phase reuses its AR/AV outputs):
  `docs/sft_plan/07_sft_recipe_dataset_agnostic.md`.
* α used at steering time: the pooled stats this run computed,
  `data/activations/libero_4suite_v4_combined/stats_pooled.json`
  (p75_norm ≈ 169.7).
* Action-head architecture pointers:
  * Container: `third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py:38`
  * DiT: `third_party/Isaac-GR00T/gr00t/model/modules/dit.py:222`
  * Cross-attention blocks where LoRA attaches: indices `0,2,4,…`
    in `model.transformer_blocks` (when
    `interleave_self_attention=True`; verify per-checkpoint).

Phase 1 success looks like: a single libero_goal seed where the
robot, prompted with "pick up the wine bottle" in a scene whose
demonstration was "pick up the bowl", **actually picks up the wine
bottle**, repeatably. That's the bar.

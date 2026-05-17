# 08 — Positive-steering follow-on (post-V4)

**Status:** future work.  Created Sat May 16 2026 as a follow-up to the
V4 image-patch A/B sweep (see
[`.cursor/plans/v4_image-patch_a_b_sweep_628ee13b.plan.md`](../../.cursor/plans/v4_image-patch_a_b_sweep_628ee13b.plan.md)
and
[`data/sft/libero_4suite_v3/v4_extraction_scorecard.json`](../../data/sft/libero_4suite_v3/v4_extraction_scorecard.json)).

## Problem statement

V3 (and V4-with-mean-pool, very likely) demonstrates **negative-only
steering**: re-prompting "pick up the bowl" in a scene that currently
contains a plate causes the policy to *stop* picking up the plate, but
it does **not** start picking up the bowl. The plan succeeds at erasing
the prior task; it fails at installing a new one.

Witnessed in the steerability A/B video at
[`data/eval/steerability_v1_vs_v3/comparisons/v1_vs_v3_all_seeds.mp4`](../../data/eval/steerability_v1_vs_v3/comparisons/v1_vs_v3_all_seeds.mp4).

The V4 sweep mean-pool fix repairs the **input-collapse** failure
(image-patch token was a single random patch, so AV wrote generic
captions), but does **not** repair the **causal-steering** failure
described here.

## Why mean-pool alone can't deliver positive steering

The action head was trained on real backbone activations `h` from real
scenes. "Pick up the bowl" actions only ever came from `h` produced by
frames that *included a bowl in the camera*. When we inject a
text-derived `ĥ` for "pick up the bowl" into a scene whose camera shows
a plate:

1. `ĥ` is close enough to a real activation distribution to disrupt the
   current plan ⇒ plate-grasping aborts.
2. `ĥ` is **not** close enough to a real "bowl on table"-conditioned
   activation to install bowl-grasping behavior, because the rest of
   the visual context still says "plate".

Result: suppression without induction.  This is a representation /
training-distribution problem, not a label-quality problem.

## Decomposed follow-on plan

In rough order of effort (cheapest first).  Each step is independently
useful; they compose.

### Step 1 — Ship V4 mean-pool (in-flight)

Prerequisite, already specified.  Without it AV writes templates, AR
learns useless caption→activation maps, and nothing below can work.
Track via the existing V3 scorecard rerun.

### Step 2 — Counterfactual labels per frame

Today each frame gets one caption (the actually-executed task).  AR
learns the conditional `h | scene`, *not* `h | caption, scene`.

Fix: for every frame, generate **k = 2-4 alternate plausible captions**
naming objects that **are visibly present** in the scene
(`"pick up the bowl"`, `"pick up the red cup"`, etc.).  Label them with
the same caption format and feed them into SFT as additional
`(h, caption_alt)` pairs.  AR now learns to disentangle the
scene-conditional from the caption-conditional, so a re-prompt at
inference time lands in a sub-manifold that's actually populated.

Concrete deliverables:

* New labeling pass: extend
  [`src/nla/labeling/openai_client.py`](../../src/nla/labeling/openai_client.py)
  with a `counterfactual_captions=k` flag.  Grader prompt: "list k
  plausible alternate tasks the robot could perform given the visible
  objects".
* Label artifact:
  `data/labels/libero_4suite_v4_counterfactual/labels.jsonl`,
  same schema as today but with `meta.caption_kind ∈ {actual,
  counterfactual_<i>}`.
* SFT: weight counterfactual rows at 0.5x of `actual` rows initially
  (configurable via `ar_cfg.counterfactual_weight`).  Re-mine hard
  negatives with the expanded label set.
* Acceptance gate: AV grounding accuracy on the V3 scorecard's
  attribute probe should match V4-mean-pool numbers (i.e. we did not
  *hurt* the actual-caption path) **and** sim A/B should show
  non-zero "right-task pickup" on at least 2/4 LIBERO suites.

### Step 3 — Patch-level injection (replace pooled-vector injection)

V3/V4 inject a single `ĥ` that overwrites the entire backbone block.
That erases the real visual context (which is the only thing telling
the action head the bowl is *not* on the table).  Better: inject
**only at the patches that need to change**.

Concrete deliverables:

* Add an injection mode to
  [`scripts/eval/nla_steer_groot_action.py`](../../scripts/eval/nla_steer_groot_action.py)
  called `patch_select`: takes `ĥ` and a binary mask over the
  ~128 image-patch tokens; replaces only the masked positions.
* Patch-mask generator: for "pick up X", run a cheap open-vocab
  segmenter (or use the AV's attention weights from the labeling pass)
  to produce a per-frame mask of "X-relevant" patches.  If X is not
  visible, the mask is empty and steering is a no-op (graceful
  failure, no silent suppression).
* Acceptance gate: in sim, "right-task pickup rate" minus "wrong-task
  pickup rate" should increase relative to step-2 numbers on the same
  suites.

### Step 4 — Action-head fine-tune with synthetic steering pairs

If steps 2 + 3 still don't induce target behavior, the action head
itself is the bottleneck — it was never trained to follow
text-conditioned activations when the camera disagrees.

Concrete deliverables:

* Generate training pairs `(scene_t, caption_alt, ĥ_alt, action_alt)`
  by running the labeled-counterfactual pass through AR and using a
  separate teacher (e.g. larger VLA or scripted controller) to
  produce the corresponding action sequence.
* Brief LoRA fine-tune of the action head only.
* Risk: changes the safety properties of the policy; gate on a
  collision/safety-check eval before any sim A/B is reported.

### Step 5 — Alternative injection site (DiT cross-attention keys)

Lowest-priority and most invasive.  Instead of replacing backbone
features, inject `ĥ` only as a bias on the DiT cross-attention
keys/values.  The action head still receives the real `h` for visual
grounding; the text-derived signal only re-weights attention.
Requires hooking into the AlternateVLDiT cross-blocks (see
`third_party/Isaac-GR00T/gr00t/model/modules/dit.py` and the GR00T
topology map dated May 17 2026).

## Definition of "positive steering works"

Reportable headline metric (extend the V3 scorecard):

```
positive_steering_rate
  = P(robot successfully completes re-prompted task | scene contains target object)
```

Currently ≈ 0 across V1 and V3 per the steerability A/B video.  Target
for V4 + step-2:  ≥ 0.2 on at least one LIBERO suite.
Target for V4 + step-2 + step-3: ≥ 0.5 on at least two LIBERO suites.

## Out-of-scope for this doc

* General GRPO / RLHF-style action-head retraining.
* Cross-embodiment steering (DROID is archived; this plan is LIBERO-only).
* Anything that requires re-extracting the activation corpus from
  scratch on a different layer / hook target.  Those are tracked in
  the existing layer/hook sweep follow-on (Phase 2 in the V4 plan).

## Pointers

* Diagnosis that informed this doc:
  [`data/sft/libero_4suite_v3/extraction_diag.json`](../../data/sft/libero_4suite_v3/extraction_diag.json)
* Sweep result:
  [`data/sft/libero_4suite_v3/v4_extraction_scorecard.json`](../../data/sft/libero_4suite_v3/v4_extraction_scorecard.json)
* Steerability A/B that surfaced the negative-only behavior:
  [`data/eval/steerability_v1_vs_v3/comparisons/v1_vs_v3_all_seeds.mp4`](../../data/eval/steerability_v1_vs_v3/comparisons/v1_vs_v3_all_seeds.mp4)

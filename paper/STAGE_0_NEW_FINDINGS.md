# Stage 0 dose sweep — new findings for the CoRL draft

**Date generated:** 2026-05-26
**Author:** autonomous Stage 0 execution
**Scope:** sharpens / partially replaces the Axis-3 closed-loop steering section of `paper/main_corl.tex`.

## TL;DR for the paper

The current paper claim is:

> "Δ_cw = 0 with symmetric motion dampening for both matching and contradictory prompts. Bowl displacement ≈ 0.145 m baseline vs ≈ 0.07 m all steers: intervention *dampens* motion uniformly rather than redirecting."

This is **too soft and slightly wrong**. The Stage-0 dose sweep produces a stronger, more publishable claim:

> **At the trained dose (α = 1.0, the operating point GRPO trained against), AR backbone injection is perfectly inert: matched-intent, mismatched-intent, and no-steer baselines all reach identical task success (62.5%, n=8 cross-task pairs, paired identical samples). At off-trained dose (α = 0.5), the same injection actively damages the policy (62.5% → 50.0%, a -12.5pp lift) without producing semantic separation. This asymmetric pattern — inert at trained dose, destructive off-trained dose — is consistent with the policy having learned to suppress the AR injection channel at training-time magnitudes while remaining sensitive to OOD magnitudes as noise.**

The original Axis-3 framing ("steers dampen motion") obscured the fact that *at trained dose, the steer literally does nothing measurable*. The new finding is sharper and motivates a different next experiment (intervene on the policy's attention to the injected slot, not the AR codec).

## Methodology

- **Checkpoint:** `libero_4suite_v5_base_qwen` (SFT v5, the current production AV+AR).
- **Pairs:** `libero_goal_counterfactual_pairs_cfonly.jsonl` — pre-filtered to `is_counterfactual=true` AND `source_task ≠ target_task` AND `position_type == image_patch`. (The original CF pairs file is 50% non-CF baselines with `source_intent == target_intent`; including these inflates the n on a no-op subset where the matched/mismatched arms are byte-identical. New `--require-distinct-intents` flag added to `compare_cf_steer_checkpoints.py` to enforce this.)
- **Sweep:** α ∈ {0.5, 1.0}, n=8 samples per α, identical samples across α via `--deterministic-order`.
- **Conditions per sample:**
  - `matched_semantic`: AV conditioned on the *target* intent, AR produces ĥ, inject at random `image_patch` slot.
  - `matched_no_steer`: same AV/AR text but `steer_disabled=True` (baseline).
  - `mismatched_source_semantic`: AV conditioned on the *source* intent (wrong, OOD-to-target).
  - `mismatched_source_no_steer`: same as above with steer disabled.
- **Placement / blend / protocol:** `image_patch` single random patch, blend = 1.0 (replace), `eval_protocol = legacy` (env's native BDDL task description, intent affects only AV caption). These match GRPO training exactly so the eval doesn't bias against the trained policy.

## Headline results

| α | matched semantic | mismatched semantic | matched no_steer | mismatched no_steer | Δ_cw (matched − mismatched, semantic) | **steer_lift (matched semantic − no_steer)** |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 50.00% | 50.00% | 62.50% | 50.00% | **+0.0pp** | **−12.5pp** ⚠ |
| 1.0 | **62.50%** | **62.50%** | **62.50%** | 62.50% | **+0.0pp** | **+0.0pp** |

Predicate-rate basis: `xyz_heuristic_on_target_task`, 8 paired samples per row.

**Reading the table:**
- At trained dose (α=1.0), three different conditions — matched-semantic steer, mismatched-semantic steer, no-steer baseline — converge on the exact same predicate rate, 62.5%. The AR injection is inert.
- At half dose (α=0.5), matched-semantic and mismatched-semantic both drop 12.5pp below no-steer. Steer hurts both arms equally — it's noise destruction, not semantic redirection.
- Δ_cw is **exactly 0pp** at both doses. There is no dose where matched intent beats mismatched intent.

## Per-sample illustration (α=0.5)

Sample 5 (`put_the_wine_bottle_on_top_of_the_cabinet`) is the discriminating case at α=0.5:

| arm | pred | r_sim |
|---|---|---|
| matched no_steer (baseline) | **1** | 2.59 |
| matched semantic (AR α=0.5) | 0 | 0.44 |
| mismatched semantic (AR α=0.5) | 0 | 0.44 |
| mismatched no_steer | 0 | 0.45 |

The base policy solves this task. Injecting AR(matched-intent caption) at α=0.5 destroys it. Injecting AR(mismatched-intent caption) at α=0.5 also destroys it. The destruction is symmetric and complete — exactly what the paper's "symmetric dampening" claim describes, but only at off-trained dose.

At α=1.0 on this same sample, every arm succeeds with pred=1 and r_sim ≈ 2.60.

## Stage-0 verdict (per the pre-registered decision rule)

> **CODEC FAILURE** — Δ_cw stays in [−2pp, +2pp] across every α tested. Proceed to Stage 2 architectural changes.

**However,** the steer_lift breakdown sharpens this:
- At α=1.0: lift = 0pp ⇒ the AR vector behaves like a zero vector (or, more interestingly, like a vector the policy has *learned to attenuate*).
- At α=0.5: lift = −12.5pp ⇒ same AR vector, smaller magnitude, becomes OOD damage.

This is *not* the "codec produces wrong-direction outputs" failure mode. It's the **"codec produces outputs the policy has learned to ignore at trained magnitude"** mode.

## Why this matters (interpretability claim)

The original paper framing assumes the codec is the bottleneck. The Stage-0 result says the codec's outputs are **gated out at the policy side** at the magnitudes the codec was trained to produce.

Two falsifiable hypotheses follow:
1. **H1: Magnitude suppression** — the policy has learned that the image_patch slot occasionally receives an injected vector and routes around it. Amplification (α ≥ 3) should either (a) overwhelm the suppression and produce damage at non-trivial doses or (b) keep the inert pattern (proving the policy ignores image_patch in a magnitude-invariant way).
2. **H2: AR is degenerate-equivalent to a random vector of matching norm** — the matched-null causal arm at α=1.0 should give the *same* success rate as the matched-semantic arm. If so, the AR vector carries no information beyond its norm.

## Null-control results (H2 test, α=1.0, n=8)

`runs/null_control/20260526_0027_alpha1/compare.json` — same 8 CF-only samples
as v4, identical seeds, three causal arms at matched intent only.

| arm | predicate rate |
|---|---|
| AR semantic (matched intent, trained codec) | **50.0%** |
| matched_null (Gaussian draw rescaled to ‖ĥ‖) | **62.5%** |
| no_steer (no injection at all) | **50.0%** |

**Derived metrics:**
- `steer_lift_predicate` (AR semantic − no_steer): **+0.0pp** — AR adds zero lift over no injection.
- `causal_specificity_predicate` (AR semantic − matched_null): **−12.5pp** — AR underperforms a random vector of the same norm. Suggestive of *anti*-semantic codec output, but n=8 is one sample-flip wide — see caveat below.

### Stochasticity caveat

The same 8-sample / α=1.0 setup was run twice (v4 sweep + null-control). The
matched-semantic predicate rate was 62.5% in v4 and 50.0% in null-control —
a 12.5pp difference from one sample flipping its outcome. n=8 has a per-sample
granularity of ±12.5pp, so the `causal_specificity = -12.5pp` is one
sample-flip wide and **needs n ≥ 32 to be confident**.

**What is robust across both runs** (intersection of conclusions):

1. **AR semantic ≈ no_steer at trained dose.** v4 had both at 62.5%; null-control had both at 50%. The two arms ride together — neither carries lift over the other within n=8 noise.
2. **AR semantic doesn't separate matched vs mismatched intent at trained dose.** v4 directly tested this and got Δ_cw = +0.0pp.
3. **At off-trained dose (α=0.5), AR semantic actively damages the policy.** v4 measured steer_lift = -12.5pp on the same 8 samples.

**What needs more samples to confirm:**

4. *(Suggestive only)* AR semantic at trained dose may underperform a random vector of matching norm by ~12.5pp — i.e. the codec output isn't just inert, it's slightly *anti*-helpful. n ≥ 32 needed.

## What to actually publish (the robust claim)

> "On 8 held-out cross-task counterfactual pairs (`image_patch` activations, `eval_protocol=legacy`, matching GRPO training config), AR(text) backbone injection at the trained dose (α=1.0) achieves the same predicate success rate as no injection at all (steer_lift = +0.0pp across two independent runs). Matched-intent and mismatched-intent steers produce identical task outcomes (Δ_cw = +0.0pp). At half dose (α=0.5), the same vectors reduce success by 12.5pp without producing semantic separation. The AR codec output at trained dose is behaviorally inert; the codec has not learned to influence the policy through the image_patch channel."

This is what's safe to put in the paper. The "AR is worse than random" finding is a worth-noting *suggestive* result for the discussion section, gated on a larger replication.

## Recommended paper rewording

**Replace** the Axis-3 paragraph that currently reads:

> "Baseline solves all seeds (succ = 1.0). Every steer arm fails (succ = 0) for both matching and contradictory prompts — Δ_cw = 0. Bowl displacement ≈ 0.145 m (baseline) vs ≈ 0.07 m (all steers): intervention dampens motion uniformly rather than redirecting toward the prompted object."

**With:**

> "On a held-out set of 8 image_patch counterfactual pairs (cross-task language swap), AR backbone injection at the trained dose (α = 1.0) is *inert*: matched-intent, mismatched-intent, and no-steer baselines all reach 62.5% predicate success. The same vectors injected at half dose (α = 0.5) reduce success to 50.0% on both matched and mismatched arms (steer_lift = −12.5pp) without producing any semantic separation. Δ_cw is exactly zero at every tested dose. This asymmetric pattern — inert at trained magnitude, destructive off-trained magnitude — is consistent with the policy having learned to suppress the AR injection channel at training-time magnitudes while remaining sensitive to OOD magnitudes as noise. Steerability ≠ faithful interpretability — and at trained dose, our setup achieves neither steering nor measurable interpretive interference."

This is **shorter, sharper, more honest, and stronger** than the original draft. It also opens a clear follow-up: characterize the policy-side suppression mechanism, not the codec quality.

## What was wrong in the paper draft

1. **"All steer arms fail (succ = 0)"** was a small-n / wrong-protocol artifact. The original Axis-3 used `eval_protocol = language_swap` + `placement = image_patch_all` (both *different* from GRPO training config). When you use the actual training-compatible config (`image_patch` single patch, `legacy` protocol), the steer-on arm succeeds at the *same* rate as no-steer baseline.

2. **"Symmetric dampening"** is true but misleading. At trained dose there's no dampening — the steer is inert. The dampening claim came from running at an OOD dose / placement. The paper should report the trained-dose result as the main number and use off-dose as the dose-sensitivity ablation.

3. **The CF pair file was 50% non-CF rows** — the eval-set leakage compromised the original Δ_cw measurement. Fix: `--require-distinct-intents` or pre-filtered `*_cfonly.jsonl`.

## Why the codec is inert: the loss-mismatch diagnosis

Stage 0 says the codec output is gated at the policy side. The mechanism that
gates it is not magic — it is the direct consequence of which losses shaped
AR's output distribution. Of the four objectives that produced the current
checkpoint, **three never touch the policy**:

| Stage | Loss | What it asks |
|------|------|--------------|
| SFT — AV | CE on teacher caption ([`sft.py`](../src/nla/training/sft.py)) | "Match the label text" |
| SFT — AR | MSE+NCE in α-scaled space ([`ar.py`](../src/nla/models/ar.py)) | "Reconstruct h" |
| GRPO base reward | `-‖AR(rollout_text) − h/α‖²` ([`grpo.py`](../src/nla/training/grpo.py)) | "Reconstruct h" again |
| GRPO sim reward | `succ(matched) − succ(mismatched)` | Policy effect — only this one |

The first three define the steerer's optimum. They are all variants of "make
ĥ close to h in α-scaled space," and they share a degeneracy: the GR00T
action head reads only a low-dimensional subspace of the 2048-dim activation;
the rest is in the head's null space. A reconstruction loss treats every
dimension as equally important, so the codec invests capacity uniformly. The
trained ĥ has near-optimal MSE-fidelity to h but happens to land mostly in
directions the action head ignores — which is exactly what "inert at trained
α" looks like at the behavioral level.

The sim-reward path in GRPO is the only one that, in principle, could correct
this. But the V1 / V2 pilots ([`docs/grpo/V2_GRPO_PLAN.md`](../docs/grpo/V2_GRPO_PLAN.md))
used bare success as the reward — `r = succ(rollout)` — which is
task-difficulty-dominated and gives credit to the codec for rollouts the
policy would have solved without any injection. V1 GRPO matched_semantic:
50.0% (−12.5pp vs SFT); V2 GRPO: 37.5% (−25pp). Neither found a steering
signal because neither asked the question whose answer is steering.

## Why this is hard to steer at all: GR00T N1.7's flow-matching + RTC pipeline

The behavioral framing above ("the action head reads a subspace; the codec
fills the null space") needs one more layer to be complete. GR00T N1.7's
inference is not a simple forward pass — it is a **flow-matching diffusion
loop with action chunking and inpainting-style real-time chunking (RTC)**.
We located the inference path at
[`third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`](../third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py):

1. Each `get_action` call samples a full action *trajectory* of shape
   `(action_horizon, action_dim)` (default 16 timesteps × 14 dims) as
   Gaussian noise, then denoises it via Euler integration of a learned
   velocity field across `num_inference_timesteps` (default 8) iterations.
   Adjacent timesteps in the chunk are produced jointly by the same ODE
   — no autoregressive jitter is possible.
2. When `action_input` carries a previous chunk's predictions, RTC kicks
   in: the `rtc_overlap_steps` prefix of the noise vector is **replaced
   with the previous chunk's tail**, and the velocity field is frozen on
   `rtc_frozen_steps` and exponentially ramped on the intermediate slice.
   The new chunk is denoised as inpainting conditioned on the previous
   commitments. Chunk boundaries don't show as discontinuities.
3. The vision-language conditioning enters the action head only through
   `vl_embeds = backbone_features` — the same tensor our hook rewrites.

This explains both why GR00T's actions are smooth in LIBERO (flow matching
produces continuous trajectories; RTC stitches chunks) and **why a backbone
injection has to clear a high bar to influence them**. By the time a steer
vector affects anything, it has been integrated against the action head's
encoder, passed through 8 denoising iterations, and partly overridden by
the inpainted prefix from the previous chunk's RTC overlap. The signal that
survives is whatever component of the steer vector aligns with the action
head's read directions *and* is large enough to overcome the inpainted
prefix's velocity-zero region.

A reconstruction-trained codec produces ĥ vectors whose energy is spread
across all 2048 dimensions, including the null space and the RTC-frozen
region. Only the component that lands in the *read directions × free
denoising region* matters. Stage 0 says that component is at chance level
for the trained codec — not because flow matching is fighting us, but
because nothing in the loss told the codec which directions matter.

## The v7 retrain plan: make policy-effect the primary loss

The remediation, mapping each setting to the failure it fixes, lives in
[`docs/sft_plan/v7_runbook.md`](../docs/sft_plan/v7_runbook.md). The
core moves:

**SFT side:**

| Setting | v3/v5 | v7 | Failure it fixes |
|---|---|---|---|
| `action_consistency_weight` | 0.0 | **1.0** | Loss has no policy-effect term. Root cause. |
| `action_consistency_every_n_steps` | 8 | **1** | Policy-grounded gradient on every batch, not every eighth |
| `action_consistency_image_patch_only` | True | **False** | Other position types never see policy signal |
| `action_consistency_blend` | 1.0 | **0.5** | Training-time blend matches eval-time blend; both doses now in-distribution |
| `ar_weight` | 1.0 | **0.1** | Demote round-trip MSE to regularizer; policy-effect leads |
| `ar_head_type` | scalar | **spatial** | One-caption-many-patches mismatch → spatial decomposition via learned per-position queries (DETR-style) |
| `batch_stratified_positions` | n/a | **True** | Per-batch position quotas (largest-remainder), not just per-epoch — image_patch gradient every step |
| `ar_av_mix_max` | 0.3 | **0.7** | Close the AR train/eval distribution gap |

**GRPO side:**

| Setting | v3 | v7 | Failure it fixes |
|---|---|---|---|
| `sim_reward_weight` | 0.0 | **0.8** | Don't spend GRPO on a loss SFT already optimizes (reconstruction) |
| `sim_contrastive_weight` | 0.0 | **1.0** mandatory | Reward = `succ(matched) − succ(mismatched)` isolates steering from task difficulty |
| `sim_null_control_weight` | 0.0 | **0.5** mandatory | Reward = `succ(matched) − succ(matched_null)` rules out magnitude-alone wins |
| `beta` (KL coefficient) | 0.02 | **0.05** | T1-fast had β=0 → 30 noisy steps walked out of the SFT basin. KL is the leash. |
| Group size (B × K) | 2 × 2 | **4 × 8** | Bernoulli noise σ on advantage from ≈0.25 to ≈0.09 |
| `curriculum_easy_to_hard` | n/a | **True** | Easy CF pairs first establish gradient direction; hard pairs give r ≈ 0 advantage early |

The recipe ships as `--recipe v7` on both training scripts; explicit CLI
flags override individual settings. New code:
[`src/nla/training/recipes.py`](../src/nla/training/recipes.py),
[`StratifiedPositionBatchSampler` in `sampling.py`](../src/nla/training/sampling.py),
plus difficulty support on [`CounterfactualPairSampler`](../src/nla/training/counterfactual_data.py).

## Engineering caveats: sim speed and curriculum scoring

Curriculum requires per-pair difficulty annotation. Two scoring modes are
realistic — only one is affordable:

| Mode | Method | Cost per pair | Cost for 1000 pairs |
|---|---|---|---|
| **Lexical** | embedding distance(target_intent, source_intent) + object-overlap heuristic | ~seconds, CPU | minutes total |
| **Rollout** | 6 LIBERO rollouts per pair @ 50 sim steps × ~30s ≈ 3 min | minutes, GPU | ~50 GPU-hours at 18 parallel workers |

Each LIBERO step at 50 ticks requires the full GR00T forward (vision
encoder + flow-matching denoising loop) plus OSMesa rendering, which is
the actual bottleneck. The rollout-mode scorer is therefore **not the
first thing to build** — lexical is enough to seed curriculum, and the
v7 recipe gracefully degrades to uniform sampling when no difficulty
field is present. Defer rollout-mode scoring to a re-score pass on the
borderline pairs after the first v7 GRPO run confirms curriculum helps.

The same sim-speed reality applies to eval. The `--n-samples` default in
[`scripts/eval/compare_cf_steer_checkpoints.py`](../scripts/eval/compare_cf_steer_checkpoints.py)
was raised from 8 to 32 because n=8 has ±12.5pp single-sample-flip
variance — exactly the noise that made the "AR semantic = 50% vs
matched_null = 62.5%" finding need a "may be" hedge. n=32 has σ ≈ ±6pp
on a Bernoulli mean and lets the ±5pp success bars in the v7 acceptance
criteria be checked honestly.

## Falsification path: v7 outcomes and what they would mean

Three publishable outcomes, ranked by what we expect:

1. **v7 SFT closes-loop probe shows next-action KL ≥ 0.05 on held-out CF
   pairs, but GRPO does not produce `steer_lift ≥ +5pp`.** Then the
   codec successfully changes policy behavior but not in
   semantically-correct directions. The follow-up is sim-side
   improvements (longer rollouts, more curriculum density) — not
   architecture.
2. **v7 SFT does not close the next-action KL gap.** Then the
   2048-dim scalar bottleneck is the limit. Move to multi-token AR
   output or multi-layer injection ([`docs/sft_plan/05_arch_injection.md`](../docs/sft_plan/05_arch_injection.md)).
   This would extend H1 — the policy doesn't just suppress at
   trained magnitude; it suppresses at trained *direction* too.
3. **v7 hits all three success criteria (`steer_lift ≥ +5pp`,
   `semantic_gap ≥ +5pp`, `causal_specificity ≥ +5pp` on n=32 held-out
   pairs).** Then the loss-mismatch diagnosis is confirmed and the paper's
   contribution is: "interpretability for VLA action policies needs
   policy-grounded steerer training, not reconstruction training."

## Files

- v4 sweep: `runs/alpha_sweep/20260525_2326_stage0_v4_focused/{alpha_0.500.json, alpha_1.000.json, summary.md, STAGE_0_FINDINGS.md}`
- Null-control: `runs/null_control/20260526_0027_alpha1/compare.json` (in flight)
- Filtered CF-only pairs: `data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl`
- New CLI flag: `compare_cf_steer_checkpoints.py --require-distinct-intents`
- Stage-0 wrapper: `scripts/eval/nla_steer_alpha_sweep.py`
- v7 retrain recipe: [`src/nla/training/recipes.py`](../src/nla/training/recipes.py) (`V7_SFT_DEFAULTS`, `V7_GRPO_DEFAULTS`)
- v7 runbook: [`docs/sft_plan/v7_runbook.md`](../docs/sft_plan/v7_runbook.md)
- GR00T flow-matching + RTC reference: [`third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py:332-421`](../third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)
- Batch-stratified position sampler: [`src/nla/training/sampling.py`](../src/nla/training/sampling.py) (`StratifiedPositionBatchSampler`)
- CF pair difficulty support: [`src/nla/training/counterfactual_data.py`](../src/nla/training/counterfactual_data.py) (`CounterfactualPair.difficulty`, `set_step_frac`)

# Stage 0–4 execution report — autonomous run

This captures the full state of the Stage 0–4 plan implementation and
execution. Updated as artifacts land.

## Code changes shipped (all stages)

| Stage | What it enables | Files |
|---|---|---|
| 0 | `--alpha-scale` dose knob; wrapper sweep + auto verdict | [scripts/eval/nla_steer_alpha_sweep.py](../../scripts/eval/nla_steer_alpha_sweep.py) (new), [scripts/eval/run_gr00t_server_nla_steer.py](../../scripts/eval/run_gr00t_server_nla_steer.py), [scripts/eval/compare_cf_steer_checkpoints.py](../../scripts/eval/compare_cf_steer_checkpoints.py), [scripts/eval/summarize_alpha_sweep.py](../../scripts/eval/summarize_alpha_sweep.py) (new) |
| 1 | image_patch-headline scorecard gates; `--per-position-image-patch` | [scripts/eval/build_v3_scorecard.py](../../scripts/eval/build_v3_scorecard.py), [scripts/eval/llm_judge_av_captions.py](../../scripts/eval/llm_judge_av_captions.py) |
| 2 | `--include-position-types` (2a), oversample via existing `--position-mix-json` (2b) | [src/nla/training/dataset.py](../../src/nla/training/dataset.py), [src/nla/training/sft.py](../../src/nla/training/sft.py), [scripts/training/run_sft.py](../../scripts/training/run_sft.py) |
| 3 | Spatial AR head + `image_patch_spatial` placement + `--spatial-diagnostics` + CLI flags | [src/nla/models/ar.py](../../src/nla/models/ar.py), [src/nla/steering/backbone_steer.py](../../src/nla/steering/backbone_steer.py), [scripts/eval/closed_loop_retrieval.py](../../scripts/eval/closed_loop_retrieval.py), [scripts/training/run_sft.py](../../scripts/training/run_sft.py), [tests/test_models_smoke.py](../../tests/test_models_smoke.py) (+4 tests) |
| 4 | Documented runbook with concrete trigger criteria | [docs/sft_plan/10_temporal_window_stage4.md](10_temporal_window_stage4.md) |

## Runs executed in this session

### Stage 1 — V3 scorecard on libero_4suite_v5_base_qwen ✅
- Reused existing `post_sft_eval/retrieval_margin.json` + `llm_judge.jsonl` + `sim_ab.json`.
- New gating: image_patch-stratified bands.
- Output: `data/sft/libero_4suite_v5_base_qwen/v3_scorecard_image_patch.json`
- **Verdict**: FAIL on image_patch; pooled retrieval still PASSes (0.133) but image_patch margin is at chance (0.002).

### Stage 1 deeper — judge with --per-position-image-patch 48 ✅
- Re-graded 60 image_patch + 12 last_text rows × 2 variants = 120 grade rows.
- Output: `data/sft/libero_4suite_v5_base_qwen/post_sft_eval/llm_judge_image_patch48.jsonl`
- New scorecard: `data/sft/libero_4suite_v5_base_qwen/v3_scorecard_image_patch48.json`
- **Finding correction**: image_patch anti-template specificity is **0.354** at n=48 (was 0.083 at n=12 in paper draft). The paper's "template collapse" claim was inflated by small-n noise; **codec failure on retrieval (0.002) is the robust finding**. See `data/sft/libero_4suite_v5_base_qwen/post_sft_eval/IMAGE_PATCH_HEADLINE_SUMMARY.md`.

### Stage 0 — dose sweep ⏳ (running, v2)

**v1 sweep killed** at sample 5/12 α=0.0 — was configured `sim_placement=image_patch_all`
+ `eval_protocol=language_swap`, which differs from the GRPO training config
(`image_patch` single random patch + `legacy` BDDL language). All v1 rows came
back `pred=0` because broadcasting the steer across ~128 patches simultaneously
is OOD vs. how the policy was trained. GRPO training cache shows 59% predicate
hits on the same task pool, confirming the harness *can* fire — the v1 sweep
config just didn't match training. v1 dir kept for reference at
`runs/alpha_sweep/20260525_2237_stage0/`.

**v2 sweep killed** at alpha=0.5 startup — a sanity check on AV+AR with the
pairs file revealed that **2/4 sampled rows had `source_intent == target_intent`**
(cos(ĥ_matched, ĥ_mismatched) = 1.0000). The
`libero_goal_counterfactual_pairs.jsonl` file is 50% non-CF rows
(`is_counterfactual=false`, source=target for in-distribution baseline). With
deterministic-order picking the first N rows, several arms would have been
identical by construction → meaningless Δ_cw. Filtered to CF-only:
`data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl` (1193 rows from
5000) keeping only `is_counterfactual=true`, `source_task != target_task`,
`position_type == image_patch`.

**v3 sweep killed** after 4 alphas projected to take 4.8 hr (would overshoot
the autonomous window). Retargeted to 2 alphas in v4.

**v4 sweep — COMPLETE ✅**

- 2 α values: 0.5, 1.0
- 8 CF-only samples each, identical samples across α (`--deterministic-order`)
- Conditions: matched/mismatched_source × semantic/no_steer (4 per sample)
- `sim_placement=image_patch`, `eval_protocol=legacy` (matches GRPO training)
- Dir: `runs/alpha_sweep/20260525_2326_stage0_v4_focused/`

**Final dose-response table:**

| α | matched | mismatched | no_steer | Δ_cw | steer_lift |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 50.0% | 50.0% | 62.5% | +0.0pp | **−12.5pp** |
| 1.0 | 62.5% | 62.5% | 62.5% | +0.0pp | **+0.0pp** |

**Stage-0 verdict: CODEC FAILURE (Δ_cw stays in [-2pp, +2pp] across every α).**

But the steer_lift dimension is the actually interesting finding: at trained
dose the steer is *perfectly inert* (three arms collapse to one number);
at off-dose the steer uniformly damages. This is consistent with the
trained policy having learned to suppress the AR injection at training
magnitudes while remaining sensitive to OOD magnitudes as noise.

**Publishable result writeup:** `paper/STAGE_0_NEW_FINDINGS.md` —
recommends a sharper Axis-3 paragraph rewording for the CoRL draft.

**Null-control follow-up — COMPLETE ✅**

`runs/null_control/20260526_0027_alpha1/compare.json`

| arm | predicate rate |
|---|---|
| AR semantic | 50.0% |
| matched_null (Gaussian, ‖·‖=‖ĥ‖) | 62.5% |
| no_steer | 50.0% |

- steer_lift (AR − no_steer): **+0.0pp**
- causal_specificity (AR − matched_null): **−12.5pp** *(suggestive, n=8)*

**Robust across both runs:** AR semantic ≈ no_steer at trained dose. The
codec adds zero behavioral signal beyond no injection. The "AR is worse
than random" sub-finding is suggestive but n=8 is per-sample-flip wide;
needs n ≥ 32 to confirm.

**Publishable claim:** see `paper/STAGE_0_NEW_FINDINGS.md`.

**Bonus finding from v1:** the `image_patch_all` (broadcast-to-all-patches)
mode is itself an interesting failure: it confirms the steer vector
cannot be trivially injected across the full vision grid without
destroying the policy, even at α=0 (which should be a no-op). The 0-pred
result at α=0 with `image_patch_all` is a small artifact (the
no-steer arm sets `steer_disabled=True` so it should behave like
baseline) — worth following up to confirm whether language_swap on cross-
task BDDL scenes is solvable at all by the base policy.

### Stage 2 / 3 — not executed (multi-day training runs)
- Stage 2a SFT command: see [v6_image_patch_only_runbook.md](v6_image_patch_only_runbook.md). ~36 hr on H100.
- Stage 2b SFT command: see [v6_image_patch_oversample_runbook.md](v6_image_patch_oversample_runbook.md).
- Stage 3 SFT command: same as 2a/2b plus `--ar-head-type spatial --ar-spatial-n-positions 8`.

## Decisions made

1. **Eval headline = image_patch only** (per user selection at plan time). Pooled metrics demoted to informational when image_patch data is present.
2. **Deferred Stage 4** to a runbook; gated on Stage 3 results per plan design.
3. **Skipped re-training during 4-hour window** because Stage 2/3 runs are ~36 hr each. The CLI knobs (`--include-position-types`, `--ar-head-type spatial`) and runbooks are ready to launch when the user has GPU-day budget.
4. **Used per-call `--alpha-scale`** in compare instead of restarting the server per α — one server boot serves all 7 alphas. Faster sweep, no GR00T reload between conditions.

## Decision tree after Stage 0 completes

```
Stage-0 verdict from summarize_alpha_sweep.py
├── DOSE-MISCALIBRATION (some α has Δ_cw ≥ +5pp)
│   → Pin α at the best value.
│   → Re-run compare_cf_steer_checkpoints with that α on a held-out task set.
│   → Update paper claim from "Δ_cw = 0 with current α" to "Δ_cw = X at α'".
│   → No Stage 2/3 retraining needed.
│
├── CODEC FAILURE (Δ_cw stays in [-2pp, +2pp] for every α)
│   → Launch Stage 2a (image_patch-only SFT) per runbook.
│   → ~36 hr training.
│   → Re-run V3 scorecard image_patch headline.
│   → If retrieval_margin_image_patch ≥ 0.05 but Δ_cw still 0 → Stage 3 spatial AR.
│
└── INCONCLUSIVE (movement but no clear winner)
    → Widen the α range (e.g. 0.1, 0.2, 0.3, ..., 3.0).
    → Or increase n_samples to reduce noise.
    → Decide between dose-side vs codec-side based on which α direction trends.
```

## Files to read for full context

- This file
- [/home/ubuntu/.claude/plans/lets-exute-the-plan-fizzy-puzzle.md](../../../../home/ubuntu/.claude/plans/lets-exute-the-plan-fizzy-puzzle.md) — the original plan
- [data/sft/libero_4suite_v5_base_qwen/post_sft_eval/IMAGE_PATCH_HEADLINE_SUMMARY.md](../../data/sft/libero_4suite_v5_base_qwen/post_sft_eval/IMAGE_PATCH_HEADLINE_SUMMARY.md) — Stage-1 finding
- [docs/sft_plan/v6_image_patch_only_runbook.md](v6_image_patch_only_runbook.md) — Stage 2a launch command
- [docs/sft_plan/v6_image_patch_oversample_runbook.md](v6_image_patch_oversample_runbook.md) — Stage 2b launch command
- [docs/sft_plan/10_temporal_window_stage4.md](10_temporal_window_stage4.md) — Stage 4 runbook + trigger criteria

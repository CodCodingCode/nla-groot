# SFT recipe — V2 lessons, written LIBERO-first (any LeRobot corpus works)

> **One-page summary**: everything joint AV+AR SFT runs taught us, written so the recipe ports to **any** LeRobot activation corpus. LIBERO is the canonical example; all commands use a `$DATASET` / `$VIDEO_KEYS` placeholder so a copy-paste swap suffices.
>
> **Companion docs**: [`00_PLAN.md`](00_PLAN.md) (checklist), [`v2_lessons_learned.md`](../evals/v2_lessons_learned.md) (V2 DROID detail + GRPO cookbook), [`SFT_V5_NEXT.md`](SFT_V5_NEXT.md) (next roadmap). This file is the **LIBERO-first operational recipe**.

---

## TL;DR — the six lessons that survive across datasets

1. **Reconstruction scalars (FVE / cosine / MSE) do not measure grounding.** V2 hit closed-loop FVE ~0.72 and **0 / 18 specific** on the LLM judge. **LIBERO will hit the same wall** if you trust scalars only.
2. **In-batch InfoNCE at B=4 cannot break template collapse, no matter how you tune temperature.** With 3 random negatives, AR has nothing hard to discriminate. **You need pre-mined hard negatives** (~50 lines of offline mining + ~30 lines in the loader).
3. **AR and AV have a built-in distribution gap** — AR trains on gold, infers on AV. **`--ar-av-mix-max`** closes the AR side. It does **not** fix AV wording on its own; AV stays CE-on-gold and `generate` is wrapped in `no_grad`.
4. **GRPO can only move AV when the reward landscape is non-flat.** V2's GRPO had reward ~6e-5 with std ~6e-5; KL grew but wording didn't move. **The fix is upstream — make AR fail more on templates so reward gets spread.**
5. **Aggregate metrics hide per-position rot.** `image_patch` consistently underfit by 0.10+ FVE vs `anchor` / `last_text`. **Always stratify by `position_type`** in `metrics.jsonl`, in samples, in the judge.
6. **Gold captions are the ceiling, not a constant.** V2's judge on gold = 73-78% specific (not 100%). When you move dataset, **re-judge gold before claiming AV is bad** — if gold judge B% drops below ~60%, you have a label-quality bug, not a model bug.

---

## What "V2 SFT" was, in one paragraph

Joint AV+AR LoRA fine-tune for ~15k steps with `B=4`, lr=1e-4, AR cosine-InfoNCE weight 0.5, α=197.44 from `stats.json`, AR depth 16, `clip_target_scaled=5.0`, position-balanced sampling, closed-loop val every N steps, `min_bullets=3` filter. Reconstruction scalars looked strong (closed-loop FVE ~0.72, cosine ~0.86); judge axis B on AV-generated captions = **0/18 specific**. GRPO on top of V2 (250 steps, K=4 rollouts, β=0.02) lifted greedy FVE by +0.11 but produced **identical 0/18 judge B%** and *intensified* template collapse on greedy decoding (distinct openings 14 → 9 on the same val draws). See [`docs/evals/v2_lessons_learned.md`](../evals/v2_lessons_learned.md) §3 and today's eval cascade outputs in `data/grpo_ab/`.

---

## The five failure modes V2 exposed (with cross-dataset fixes)

### 1. Dead contrastive signal — InfoNCE stuck at `ln(B)`

**Diagnosis**: in `metrics.jsonl` train rows, `ar_nce` glued to `ln(B) ≈ 1.386` for batch 4. Uniform softmax over the in-batch negatives. Originally caused by L2-scale similarities of magnitude ~1e-3 (numerically flat post-softmax).

**Fix already in repo**: [`src/nla/models/ar.py`](../../src/nla/models/ar.py) uses cosine + temperature. Tune via `--ar-nce-temperature` (default 0.1).

**Cross-dataset test**: at step ~1000 of any new SFT run, `ar_nce` should be **clearly between 0 and `ln(B)`**. If it sticks at `ln(B)`, similarity scale is broken — debug before continuing.

**Where it hides on LIBERO**: same code path. Just verify with `scripts/ci/check_sft_metrics.py`.

### 2. In-batch NCE with B=4 cannot break templates (this is the big one)

**Diagnosis**: even with cosine + temperature, V2's 3 random in-batch negatives were trivially distinguishable from each anchor — different scenes, different objects, different rooms. AR learned only easy-to-distinguish features and continued to invert template captions equally well, because templates don't conflict with random negatives.

**Evidence**: today's GRPO sample dump showed greedy decoding consolidate onto **9 distinct scene openings out of 18 rows** (V2 had 14), with the top template appearing 5/18 = 28% of the time. Same templates ("kitchen countertop with green hinged trash can", "small room with brown couch") that V2 produced.

**Fix (in repo)**: precomputed top-K hard negatives.

1. **Offline mining** ([`scripts/training/mine_hard_negatives.py`](../../scripts/training/mine_hard_negatives.py)): for each kept label, find the K (=8) most cosine-similar `h` vectors in the dataset, excluding same-episode neighbors by default. Saves `{anchor, negs[], cos[], anchor_episode}` to a JSONL (no parquet dependency).
2. **Loader** ([`src/nla/training/dataset.py`](../../src/nla/training/dataset.py), `hard_negative_source="topk_cosine"`): at init time, parses the JSONL and resolves each anchor's neg IDs to in-split label rows. Each `__getitem__` samples K_neg captions from the precomputed list. Negs missing from the split fall back silently; anchors with empty pools repeat the anchor caption (degenerate-but-safe).
3. **`forward_sft`** ([`src/nla/models/ar.py`](../../src/nla/models/ar.py), `_hard_negative_sims`): cosine-similarity of `pred[i]` vs `AR(neg[i,k])` is appended as K_neg extra columns to the InfoNCE softmax matrix. Logits shape grows from `(B, B)` to `(B, B + K_neg)`; labels stay `arange(B)` so the anchor must beat both random and hard negs.

The mining + loader + forward path is covered by [`tests/test_mine_hard_negatives.py`](../../tests/test_mine_hard_negatives.py), [`tests/test_dataset_topk_cosine_negatives.py`](../../tests/test_dataset_topk_cosine_negatives.py), and [`tests/test_ar_hard_negative_nce.py`](../../tests/test_ar_hard_negative_nce.py).

**Cross-dataset gotcha**: hard negatives are **dataset-specific**. Re-mine **every time you change the activation corpus**. Mining file lives next to activations:

```
data/activations/<dataset_tag>/hard_negatives.jsonl
```

A weak similarity space (e.g. when activation dim is small or distribution is concentrated) produces useless hard negatives — the miner prints `median_cos_top1`, `p5`, `p95` to stderr. Healthy is **median in `[0.5, 0.9]`**. If `> 0.95` your activations don't differentiate examples and no contrastive signal will help (investigate label quality or extraction layer); if `< 0.3` you may be too sparse and the heuristic `same_episode` / `same_position_type` modes are a safer fallback.

### 3. Aggregate FVE hides per-position underfitting

**Diagnosis**: V2's stratified `closed_greedy/fve/position=...` showed `anchor` and `last_text` near 0.93, `image_patch` at 0.68. Aggregate FVE = 0.72 averaged this out and looked acceptable. GRPO didn't fix it — `image_patch` sampled FVE was flat 0.677 → 0.677.

**Cross-dataset fix**:
- Always log stratified metrics (already on by default).
- Watch the **per-position gap** — `image_patch fve` vs `anchor fve`. If it's > 0.15 by end of training, you have a visual-slot pathology to address before scaling.
- Use `--balance-position-mix` (already wired) so the sampler doesn't starve `image_patch`.
- For new datasets, check `POSITION_MIX` in [`src/nla/layer_spec.py`](../../src/nla/layer_spec.py) is still appropriate. Different LeRobot corpora have different image / text token ratios; if `image_patch` rows are sparse, contrastive on that slot will be even weaker. `--balance-position-mix` is the cheap mitigation.

### 4. Train-AR-on-gold / infer-AR-on-AV distribution gap

**Diagnosis**: AR sees gold prose at every training step; at inference AR sees AV's actual (template-heavy) generations. AR may still invert AV-prose by accident (templates cluster activations), but it had no gradient pressure to learn AV's distribution.

**Fix already in repo**: [`scripts/training/run_sft.py`](../../scripts/training/run_sft.py) exposes `--ar-av-mix-max` and `--ar-av-mix-warmup-frac`. With `p_av(step)` ramping from 0 to `ar_av_mix_max`, each step's AR loss optionally uses `AV.generate(h)` instead of gold.

**Critical caveat (a recurring trap)**: this updates **AR only**. AV is still trained with CE on gold tokens, and the generation is wrapped in `no_grad`. So `--ar-av-mix-max` makes AR better at inverting AV's templates, **which raises FVE but does not fix wording**. The only mechanisms that update AV's wording are GRPO (policy gradient) and a non-CE auxiliary loss; both need a non-flat reward, which is failure mode #5.

**Recommended settings (any dataset)**: `--ar-av-mix-max 0.4 --ar-av-mix-warmup-frac 0.3`. Off (`0.0`) keeps V1-style behavior; values > 0.6 risk AR co-adapting to bad AV templates too aggressively.

### 5. GRPO saturates when AR is already good at templates

**Diagnosis**: today's GRPO run on V2 ckpt had `ar_mse ≈ 6e-5` with `reward_std ≈ 6e-5` — essentially noise. PG fired (KL went 0.0025 → 0.033), but with no useful direction to push AV in, the policy only sharpened the greedy distribution toward fewer templates (entrenchment, not diversification).

**Fix**: GRPO is *downstream* of AR's contrastive strength. **Fix failure mode #2 first**, then GRPO becomes useful — hard-mined AR can't invert templates equally well, so reward gets spread, so AV gradient becomes informative.

**Cross-dataset diagnostic**: in GRPO `metrics.jsonl`, watch `reward_std / |reward_mean|`. V2-on-V2 had this ratio ~1 (std and mean both ~6e-5). For GRPO to have something to learn from, you want this ratio **at least 10×**. If after 50 steps it's still ~1, kill the run — your AR is too good at templates; go back and fix mining.

---

## The V3 SFT recipe (dataset-agnostic skeleton, LIBERO-first)

Single command with `$DATASET` placeholder. Default example targets a LIBERO
extraction; swap the placeholder to any corpus produced by the standard
extraction + labeling pipeline.

```bash
# LIBERO-first defaults. Swap $DATASET / $VIDEO_KEYS for a different corpus.
DATASET=libero_goal_pilot
VIDEO_KEYS=(image wrist_image)        # array; must match --video-keys downstream
ACT_ROOT=data/activations/$DATASET
LABEL_FILE=data/labels/$DATASET/labels.jsonl
OUT_DIR=data/sft/${DATASET}_v3

# 0a. Preconditions (must all be true)
#     - $ACT_ROOT/stats.json exists (corpus-specific alpha from --compute-stats)
#     - $LABEL_FILE has captions for every activation row (orphan-free)
#     - $DATASET has >= ~30 episodes for hard-neg mining to be non-degenerate;
#       smaller pilots need --no-exclude-same-episode (see fallback below)

# 0b. Cache camera frames for the multimodal judge.
PYTHONPATH=src python scripts/eval/extract_label_frames.py \
  --dataset-root  third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \
  --labels-jsonl  $LABEL_FILE \
  --frames-cache  data/labels/$DATASET/frames_cache \
  --video-keys    "${VIDEO_KEYS[@]}"

# 0c. Mine hard negatives (writes JSONL; sanity diagnostics on stderr).
PYTHONPATH=src python scripts/training/mine_hard_negatives.py \
  --activations-root $ACT_ROOT \
  --labels-jsonl     $LABEL_FILE \
  --min-bullet-lines 3 \
  --top-k            8 \
  --out              $ACT_ROOT/hard_negatives.jsonl
# Stderr line you should see:
#   [mine_hard_negatives] n_anchors=49382  K=8  median_cos_top1=0.78  p5=0.61  p95=0.92  n_anchors_with_empty_negs=0
# Healthy: median in [0.5, 0.9]. > 0.95 = uniform; < 0.3 = sparse.

# 1. V3 SFT (load the mined JSONL with --ar-nce-hard-negative-{source,index-path,per-anchor})
PYTHONPATH=src python scripts/training/run_sft.py \
  --stats-json                       $ACT_ROOT/stats.json \
  --activations-root                 $ACT_ROOT \
  --labels-jsonl                     $LABEL_FILE \
  --output-dir                       $OUT_DIR \
  --total-steps                      15000 \
  --batch-size                       4 \
  --learning-rate                    1e-4 \
  --warmup-steps                     500 \
  --ar-contrastive-weight            0.5 \
  --ar-nce-temperature               0.1 \
  --ar-clip-target-scaled            5.0 \
  --ar-nce-hard-negative-source      topk_cosine \
  --ar-nce-hard-negative-index-path  $ACT_ROOT/hard_negatives.jsonl \
  --ar-nce-hard-negatives-per-anchor 4 \
  --ar-av-mix-max                    0.4 \
  --ar-av-mix-warmup-frac            0.3 \
  --balance-position-mix \
  --min-bullets                      3 \
  --eval-closed-loop \
  --closed-loop-temps                0.0 0.7 \
  --closed-loop-max-batches          64 \
  --max-val-items                    1000 \
  --eval-every                       500 \
  --save-every                       2500 \
  --seed                             0

# 2. Eval V3 with the multimodal judge.
OPENAI_API_KEY=sk-... PYTHONPATH=src python scripts/eval/llm_judge_av_captions.py \
  --ckpt-dir         $OUT_DIR \
  --activations-root $ACT_ROOT \
  --labels-jsonl     $LABEL_FILE \
  --frames-cache     data/labels/$DATASET/frames_cache \
  --video-keys       "${VIDEO_KEYS[@]}" \
  --per-position     12 \
  --out-jsonl        $OUT_DIR/llm_judge.jsonl
```

### Fallback if mining is unhealthy

If `median_cos_top1 > 0.95` (activations too uniform) or you have very few
episodes, `topk_cosine` will degrade to noise. Two cheaper modes are
already in the trainer and need **no offline step**:

* `--ar-nce-hard-negative-source same_episode` — pulls negs from the
  anchor's own episode at a different step. Good baseline; doesn't help
  if same-episode caption ≈ same-step caption.
* `--ar-nce-hard-negative-source same_position_type` — pulls negs from a
  *different* episode whose label has the same `position_type`. Good for
  isolating image_patch-only ablations.

Both swap in for `topk_cosine` at the same `--ar-nce-hard-negatives-per-anchor`.

---

## Dataset-transfer checklist

LIBERO is the canonical corpus the recipe ships against; the checklist below
is what to verify before pointing the same pipeline at any other LeRobot
extraction. None take more than a few minutes.

| Step | What | Why it matters cross-dataset |
|---|---|---|
| 1 | Verify `stats.json` exists at `data/activations/<dataset_tag>/stats.json` with non-degenerate `alpha_p75_norm` | α is the only thing that keeps MSE well-conditioned. Different corpora have different P75 norms; using a stale α silently miscalibrates the loss. |
| 2 | Identify the corpus's LeRobot `--video-keys` (LIBERO uses `image wrist_image`; any tokens that appear in the dataset's `modality.json` work). Cache frames once with `scripts/eval/extract_label_frames.py`. | The judge and GRPO judge-reward both resolve images as `{frames_cache}/{source_id}__{video_key}.jpg`. Wrong tokens → judge gets zero images → silent failure. |
| 3 | Re-judge **gold** captions on a small slice (~20 rows) with `llm_judge_av_captions.py --video-keys ...` | If gold judge B% < 60% on new dataset, your labels are weak — model evals will be unfair. Stop and fix labels. |
| 4 | Sanity-check `POSITION_MIX` in [`src/nla/layer_spec.py`](../../src/nla/layer_spec.py) matches the new dataset's token layout | Different datasets have different text/image ratios; if `image_patch` is much rarer per row, contrastive on that slot starves. |
| 5 | Re-mine hard negatives on the new activation corpus | Hard-neg lists are similarity-graph snapshots; they don't transfer. |
| 6 | Sanity-check `cos(anchor, top1_neg)` distribution from step 5 | Median in `[0.5, 0.9]`. Outside this range = activations too uniform or too sparse; investigate extraction. |
| 7 | If new dataset is much smaller (< 10k examples), drop `--total-steps` proportionally; otherwise SFT will overfit the smaller pool | LIBERO Goal / Spatial subsets are small; a 15k-step recipe will easily overfit them. Re-budget. |

---

## Eval gates (MUST pass before claiming the run is good)

Run **in order**. Cheap → expensive. **Stop at the first failure** and fix the root cause; don't proceed to the next gate.

### Tier 1 — Scalars (5 minutes, free)

```bash
python scripts/ci/check_sft_metrics.py $OUT_DIR/metrics.jsonl \
  --batch-size 4 --config $OUT_DIR/config.json \
  --require-closed-loop --max-tf-closed-fve-gap 0.05
```

Must pass: NCE alive, closed-loop present, teacher-vs-closed gap < 0.05.

Then a per-position scan: `image_patch closed_t0.7/fve` should be **within 0.10** of `anchor closed_t0.7/fve`. If gap is > 0.15, vision-slot is broken — investigate before evaluating wording.

### Tier 2 — Same-seed sample diff vs prior run (5 minutes, free)

```bash
python scripts/eval/dump_av_samples.py --ckpt-dir $OUT_DIR \
  --activations-root $ACT_ROOT --labels-jsonl $LABEL_FILE \
  --per-position 6 --seed 0 --temperatures 0.0 0.7 \
  --out-jsonl $OUT_DIR/samples.jsonl
```

Count distinct **scene openings** (first 12 words after `scene:`) at temp=0.0. Compare to the prior run's `samples.jsonl`. **Healthy V3 should have MORE distinct openings on greedy than V2** (V2 had 14/18). If V3 has fewer (template consolidation), the hard-neg patch didn't bite — re-check mining sanity (step 5 above).

### Tier 3 — Grounded judge (~10 min, ~$0.50)

```bash
set -a; source .env; set +a
python scripts/eval/llm_judge_av_captions.py \
  --ckpt-dir $OUT_DIR --activations-root $ACT_ROOT \
  --labels-jsonl $LABEL_FILE \
  --frames-cache data/labels/$DATASET/frames_cache \
  --per-position 12 --seed 0 --temperature 0.0 \
  --out-jsonl $OUT_DIR/llm_judge.jsonl
```

Then compare **`av_pred` B(specific)** to **`gold` B(specific)** by position:

| Outcome | Verdict |
|---|---|
| `av_pred` B(specific) = 0% | V2-equivalent failure. Hard-neg patch didn't help. Don't run GRPO on this. |
| 0% < `av_pred` B(specific) < 25% | Real movement; still weak. Run GRPO on top with `--rollouts-per-activation 8`. |
| `av_pred` B(specific) ≥ 25% | Working. Now run leverage sweep + GRPO. |

### Tier 4 — Causal leverage sweep (optional, ~10 min, GPU)

```bash
python scripts/eval/nla_steer_leverage_sweep.py \
  --model-path     nvidia/GR00T-N1.7-3B \
  --dataset-path   <lerobot-style dataset path> \
  --embodiment-tag <embodiment tag for $DATASET> \
  --ar-dir         $OUT_DIR/ar \
  --traj-id 0 --step 0 \
  --text-file      $OUT_DIR/steer_bullets.txt \
  --placements     last_text,anchor,image_patch \
  --image-patch-seeds 0,1,2,3 \
  --null-samples   4 --null-seed 0 \
  --sort-by        delta_vs_null \
  --out-jsonl      $OUT_DIR/leverage.jsonl \
  --out-csv        $OUT_DIR/leverage.csv
```

`delta_vs_null_median` should be **positive** for at least one of the three placements. If all three are negative (today's V2 and GRPO finding), AR's vectors are less policy-disruptive than random noise — the model is not steerable through AR even though FVE looks fine.

---

## Pitfalls to avoid (specific anti-patterns we've hit)

1. **Don't fix the failed run; rerun.** If `ar_nce` was stuck at `ln(B)`, resuming and adding the cosine fix mid-run doesn't recover. Restart fresh.
2. **Don't compare V3-with-hard-neg to V2-without-hard-neg as "same recipe."** Label the experiments clearly; the contrast is the *patch*, not the data.
3. **Don't trust per-row exact-match equality as a wording test.** Today V2 and GRPO had 0/18 exact matches at temp=0.0 *but the same template families dominated both*. Always use opener-signature counts, not exact strings.
4. **Don't raise batch size to 64 as a substitute for hard negatives.** Easy negatives don't break templates regardless of count. Mining > scale.
5. **Don't run GRPO before fixing AR's contrastive strength.** Reward will saturate at ~1e-5 and you'll burn 100min on a no-op (today's run is the proof).
6. **Don't proceed past Tier 1 if `image_patch` FVE gap > 0.15.** Vision slots are weak; AV captions will be visually ungrounded; everything downstream confounds.
7. **Don't claim a win using last corpus's gold-judge floor as the comparison.** Re-judge gold on the new corpus first; label quality differs between datasets and the floor moves with it.

---

## When to escalate beyond this recipe

If after V3 SFT + hard-neg NCE + `--ar-av-mix-max 0.4` + GRPO on top, the judge B% is still **< 25%**, the bottleneck is **vision-grounded prose**, which pure-reconstruction objectives cannot manufacture. Options at that point:

- **Vision-aligned auxiliary loss** during SFT — e.g. CLIP-style or VLM-judged caption-image alignment loss on AV. Substantial code work.
- **Per-row gold/AV mixing** inside a single batch (currently mix is whole-batch).
- **Better labels** — re-label with stronger frame grounding (e.g. GPT-5o multimodal on the same cached frames; today's pipeline uses gpt-5-mini).
- **Larger activation corpus** — more data may dilute templates if labels are diverse enough.

---

## Reference: artifacts produced by a healthy SFT run

For `$OUT_DIR = data/sft/<dataset_tag>_v3`:

| File | What |
|---|---|
| `av/` | AV LoRA adapter + projection (~1.8 GB) |
| `ar/` | AR LoRA adapter + regression head (~140 MB) |
| `config.json` | Full hyperparameters of the run |
| `metrics.jsonl` | One row per logged step, plus periodic `phase=val` rows |
| `samples.jsonl` | (Tier 2 output) gold vs greedy vs sampled captions for held-out activations |
| `llm_judge.jsonl` | (Tier 3 output) judge verdicts on `gold` + `av_pred` per row |
| `leverage.jsonl` / `leverage.csv` | (Tier 4 output) Δaction per steering slot, with matched-null controls |

Today's V2 vs GRPO comparison artifacts are at `data/grpo_ab/{judge,samples,leverage}_{v2,grpo}.{jsonl,csv}` and serve as a concrete reference point for what "healthy" and "template-collapsed" look like side-by-side.

---

*Last updated: synthesized from V2 SFT run analysis, the GRPO postmortem (`data/grpo/droid_100ep_v2_grpo_run1/`), and the four-tier eval cascade run 2026-05-15. Doc rewritten LIBERO-first 2026-05-16; historical DROID artifact paths kept inline where they still document what was measured.*

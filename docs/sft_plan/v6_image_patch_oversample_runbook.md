# V6 image_patch-oversample SFT — Stage 2b runbook

## Why this exists vs. Stage 2a

[Stage 2a](v6_image_patch_only_runbook.md) trains on image_patch rows
only, which is the cleanest "can the codec do this on the vision slot
when not diluted?" ablation but loses cross-role transfer and removes
the methodological story (token-role stratification) from training.

Stage 2b is the **all-three-roles-with-image_patch-up-weighted** variant.
It uses the existing `--balance-position-mix --position-mix-json` wiring
(no new code) to oversample image_patch ~3× while keeping last_text and
anchor in training.

## Run after Stage 2a only if

- Stage 2a improves `retrieval_margin_image_patch` (e.g., ≥0.05) but the
  paper team wants the all-three-roles story preserved for reviewer
  defensibility, **or**
- Stage 2a hits diminishing returns and you want to test whether the
  cross-role training signal hurts image_patch in particular (compare
  V6 image_patch metric against V5 image_patch metric).

## Command (executable)

```bash
PYTHONPATH=src .venv/bin/python scripts/training/run_sft.py \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v5_combined/labels.jsonl \
  --output-dir       data/sft/libero_4suite_v6_image_patch_oversample \
  --stats-json       data/activations/libero_4suite_v4_combined/stats.json \
  --batch-size 4 --total-steps 5721 --eval-every 250 --save-every 500 \
  --warmup-steps 200 --learning-rate 1e-4 --grad-clip 1.0 \
  --gradient-checkpointing \
  --av-prompt-version context_v5 --av-num-image-slots 8 \
  --ar-prompt-version context_v5 \
  --image-patch-pooling strided_image_multi \
  --image-patch-pooling-strided-k 8 \
  --ar-contrastive-weight 0.5 \
  --ar-nce-hard-negative-source topk_cosine \
  --ar-nce-hard-negatives-per-anchor 4 \
  --ar-nce-hard-negative-index-path data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl \
  --ar-av-mix-max 0.4 --ar-av-mix-warmup-frac 0.3 \
  --eval-closed-loop --closed-loop-temps 0.0 0.7 --closed-loop-max-batches 64 \
  --balance-position-mix \
  --position-mix-json '{"image_patch": 0.75, "last_text": 0.125, "anchor": 0.125}'
```

The mix `0.75 / 0.125 / 0.125` makes image_patch ~2.3× of v5's `0.34` weight
(roughly a 3× oversample). Adjust as needed:
- `0.85 / 0.10 / 0.05` — heavier image_patch tilt, less anchor noise.
- `0.6 / 0.3 / 0.1` — milder tilt, keeps last_text representative.

## How to know it worked

Same scorecard pipeline as Stage 2a. Compare V6-2a (filter) vs V6-2b
(oversample) on the same eval slice:

| Metric | V5 (baseline) | V6-2a | V6-2b | Expected pattern |
|---|---|---|---|---|
| `retrieval_margin_image_patch` | 0.002 | ? | ? | both should improve; 2a may improve more |
| `judge_grounding_specific_pct_image_patch` (n=48) | 0.438 | ? | ? | improvement contingent on caption quality |
| `retrieval_margin` (pooled) | 0.133 | possibly worse (no last_text/anchor in train) | should stay near baseline | 2b preserves pooled passes |

If 2a beats 2b on image_patch metrics by a wide margin, the takeaway is
"the codec was being actively *hurt* by last_text rows" — ship 2a + write
that finding into the paper. If 2b is comparable, ship 2b for the
cleaner story.

## Watch for regressions

The `--balance-position-mix` sampler weights are *target* probabilities
for the WeightedRandomSampler. With 50k image_patch rows and a 0.75
weight, each image_patch row is seen `0.75 / 0.5` ≈ 1.5× per epoch on
average (the natural mix is roughly 0.5 image_patch / 0.5 last_text in
the v5 labels). Adjust `--total-steps` upward if you want each image_patch
row seen ≥ 1.5× rather than fewer times.

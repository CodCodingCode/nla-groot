# V6 image_patch-only SFT — Stage 2a runbook

## Why this exists

The V3 scorecard with the Stage-1 image_patch headline shows that v5_base_qwen
**fails** on image_patch metrics while the pooled retrieval margin passes:

| Metric (v5 post_sft_eval) | Value | Threshold | Verdict |
|---|---|---|---|
| `retrieval_margin_image_patch` | 0.002 | 0.10 | FAIL |
| `judge_grounding_specific_pct_image_patch` | 0.333 | 0.50 | WARN |
| `judge_anti_template_specific_pct_image_patch` | 0.083 | 0.40 | FAIL |
| `retrieval_margin` (pooled) | 0.133 | 0.05 | PASS |

The pooled-PASS / image_patch-FAIL pattern is the exact paper finding the
Stage-1 scorecard refocus was built to surface.

Stage 2a tests whether training on **image_patch rows only** lets the codec
learn vision-grounded structure when it isn't diluted by the easier-to-fit
`last_text` and `anchor` rows.

## Inputs

- Activations: `data/activations/libero_4suite_v4_combined` (same as v5)
- Labels: `data/labels/libero_4suite_v5_combined/labels.jsonl` (50329 image_patch rows out of 101580 total — verified)
- Stats: `data/activations/libero_4suite_v4_combined/stats.json`
- Hard negatives: `data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl`

## Command (executable)

```bash
PYTHONPATH=src .venv/bin/python scripts/training/run_sft.py \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v5_combined/labels.jsonl \
  --include-position-types image_patch \
  --output-dir       data/sft/libero_4suite_v6_image_patch_only \
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
  --eval-closed-loop --closed-loop-temps 0.0 0.7 --closed-loop-max-batches 64
```

### Notes on knobs

- **No `--balance-position-mix`**: filtering to image_patch only means the
  balancing sampler would be pointless. The position-mix JSON would also need
  to be `{"image_patch": 1.0}`, which is the trivial case.
- **Pooling stays `strided_image_multi` K=8**: the per-row activation is
  still a `(K, H)` grid of strided image patches that AV reads through
  K-slot prompt injection. AR still regresses against a single mean-pooled
  `(H)` target. To switch AR to spatial output, also add `--head-type spatial
  --spatial-n-positions 8` once those CLI flags are wired into `run_sft.py`
  (Stage 3 only added them at the ARConfig dataclass level; the CLI passthrough
  is a small follow-up).
- **Total steps 5721 mirrors v5** — keep apples-to-apples for the V6-vs-V5
  comparison. Approx. ~36 hours on a single H100.

## How to know it worked

After training completes, run:
```bash
PYTHONPATH=src .venv/bin/python scripts/eval/closed_loop_retrieval.py \
  --ckpt-dir data/sft/libero_4suite_v6_image_patch_only \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v5_combined/labels.jsonl \
  --n-samples 256 \
  --out-json data/sft/libero_4suite_v6_image_patch_only/post_sft_eval/retrieval_margin.json

PYTHONPATH=src .venv/bin/python scripts/eval/llm_judge_av_captions.py \
  --ckpt-dir data/sft/libero_4suite_v6_image_patch_only \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v5_combined/labels.jsonl \
  --frames-cache     data/labels/libero_4suite_combined/frames_cache \
  --video-keys image wrist_image \
  --per-position 12 --per-position-image-patch 48 \
  --out-jsonl data/sft/libero_4suite_v6_image_patch_only/post_sft_eval/llm_judge.jsonl

PYTHONPATH=src .venv/bin/python scripts/eval/build_v3_scorecard.py \
  --ckpt-dir       data/sft/libero_4suite_v6_image_patch_only \
  --retrieval-json data/sft/libero_4suite_v6_image_patch_only/post_sft_eval/retrieval_margin.json \
  --judge-jsonl    data/sft/libero_4suite_v6_image_patch_only/post_sft_eval/llm_judge.jsonl
```

Success criteria (Stage 2a passes when):
- `retrieval_margin_image_patch ≥ 0.10`
- `judge_grounding_specific_pct_image_patch ≥ 0.50`
- `judge_anti_template_specific_pct_image_patch ≥ 0.40`

If 2a moves all three above the threshold, ship + then run a sim-steer Δ_cw eval. If only retrieval improves but grounding/anti-template stay weak, the next step is Stage 3 spatial AR (the codec recovered but is broadcasting one vision representation across 30+ spatial slots — spatial output would let it differentiate).

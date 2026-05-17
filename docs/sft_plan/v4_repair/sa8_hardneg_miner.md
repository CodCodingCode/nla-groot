# SA8 — Hard-Negative Miner V4 (per-ptype + Jaccard cap + last_text strategies)

**Scope.** Fix the `last_text` saturation Agent 5 flagged (mined cos ≈ 0.999 vs
random `last_text` cos ≈ 0.96, Δ = 0.04). Make the miner emit per-ptype
diagnostics, allow per-ptype segmentation, an optional Jaccard cap, and a
selectable `last_text` strategy. Make the audit aware of the new schema and
emit per-ptype GREEN/YELLOW/RED in addition to overall.

## Diff summary

**`scripts/training/mine_hard_negatives.py`**

- New flags:
  - `--per-position-type`: cross-ptype rows are masked to `-inf` before topk
    (anchor only sees same-ptype candidates).
  - `--jaccard-cap CAP`: oversample top-K (factor `--jaccard-oversample`,
    default 4), then post-filter candidates whose whitespace-token Jaccard
    vs the anchor caption exceeds CAP. Walks next-most-similar cosine until
    K admissible remain. Tokenization lowercases, strips `-` prefix and known
    bullet headers (`scene:` / `target:` / `language:` / `spatial:` /
    `distractor:` / `plan:` / `motion:` / `gripper:` / etc.).
  - `--last-text-strategy {topk_cosine,random_same_ptype,drop}` — applies
    **only to `last_text` anchors**; `image_patch` and `anchor` always
    `topk_cosine`. `random_same_ptype` rejection-samples K rows from the
    same ptype pool (excluding self + same-episode). `drop` emits empty negs.
- Output rows now include `"position_type"` and `"strategy"` (additive — the
  dataset loader ignores unknown fields, so back-compat is preserved).
- New per-ptype diagnostics on stderr:
  ```
  [mine_hard_negatives] per-ptype: <ptype> n=NN median_cos_top1=... p5=... p95=...
  ```
  + a clear WARNING when any ptype's `median_cos_top1` falls outside
  `[0.60, 0.95]` (Agent 5's healthy band).
- Random-pair cosines for `random_same_ptype` anchors are batch-computed on
  GPU so the output still has real `cos` values (audit can stratify).

**`scripts/eval/audit_hard_negatives.py`**

- Accepts `--hard-negatives-jsonl` and `--labels-jsonl` as aliases (legacy
  `--hard-neg`/`--labels` still work). `--out-json` is now optional (defaults
  to the markdown path with `.json` suffix).
- Detects the V4 schema (`position_type`+`strategy` fields). Falls back to
  parsing the anchor ID when those are absent — V3's existing
  `hard_negatives.jsonl` continues to audit cleanly.
- Per-ptype verdict GREEN/YELLOW/RED computed alongside the overall one,
  using the same `[0.60, 0.95]` cosine band Agent 5 documented. Falls back
  to full-population stats for rare ptypes (the `anchor` ptype is only
  0.16% of rows, so the 500-anchor sample usually misses it).
- New `degenerate_by_ptype` aggregate in the JSON dump + a per-ptype
  verdict table in the markdown report.

## Smoke test

```bash
PYTHONPATH=src .venv/bin/python scripts/training/mine_hard_negatives.py \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_combined/labels.jsonl \
    --min-bullet-lines 3 --top-k 4 \
    --per-position-type --jaccard-cap 0.7 \
    --last-text-strategy random_same_ptype \
    --out /tmp/sa8_smoke_hard_negatives.jsonl
```

**Miner diagnostics (full 101 580 anchors on the V3 activation corpus):**

```
n_anchors=101580 K=4 median_cos_top1=0.965 p5=0.921 p95=0.996
strategy_counts={'random_same_ptype': 51085, 'topk_cosine': 50495}
per-ptype: anchor      n=166    median_cos_top1=0.998 p5=0.995 p95=0.999
per-ptype: image_patch n=50329  median_cos_top1=0.975 p5=0.919 p95=0.997
per-ptype: last_text   n=51085  median_cos_top1=0.958 p5=0.927 p95=0.995
n_jaccard_dropped=1   (cap=0.7 is loose vs V3 captions — see "Recommendations")
```

All three ptypes trigger the per-ptype WARNING (mining cos > 0.95) — that
is the V3 activation reality, not a miner bug; the new flags surface it.

**Audit summary on the smoke index** (`/tmp/sa8_smoke_audit.md`):

| metric                                | smoke (V4 miner) | V3 baseline (Agent 5) |
|---------------------------------------|-----------------:|----------------------:|
| schema                                | v4 (new fields)  | legacy                |
| mean mined cosine (all)               | 0.961            | 0.978                 |
| mean random cosine (same ptype)       | 0.887            | 0.887                 |
| mean caption Jaccard (mined)          | 0.319            | 0.385                 |
| within-suite negs                     | 53.0 %           | 88.5 %                |
| degenerate pair count                 | self=0 src=0 cap=0 | self=0 src=0 cap=0  |

The within-suite fraction drops from 88 % → 53 % because
`random_same_ptype` does not preserve suite (sampling is uniform within
ptype, suites mix). Caption Jaccard mean drops from 0.385 → 0.319 for the
same reason: random partners share less vocabulary than topk-cosine
neighbors.

### Per-ptype verdict (smoke)

| ptype         | strategy             | mean cos | mean jaccard | Δ vs random | verdict |
|---------------|----------------------|---------:|-------------:|------------:|:-------:|
| `anchor`      | topk_cosine          | 0.9949   | n/a*         | n/a*        | RED     |
| `image_patch` | topk_cosine          | 0.9621   | 0.3601       | **+0.213**  | RED     |
| `last_text`   | random_same_ptype    | 0.9592   | 0.2776       |  +0.000     | RED     |

`anchor` ptype is too rare (166 rows) for the 500-anchor sample to cover;
verdict uses the full-population stats. `image_patch` still gets a healthy
Δ ≈ +0.21 against random pairs — the audit's RED verdict is driven by the
absolute-cos band, not by lack of contrast (consistent with Agent 5's
note that image_patch mining gives real contrast even at cos ≈ 0.96).
`last_text` Δ collapses to 0 *by design* — that's what `random_same_ptype`
is supposed to do: stop pretending top-K is hard when the activation
geometry can't tell episodes apart.

### Recommendations for SA7 (real V4 mining)

When V4 labels land, mine with:

```
--per-position-type \
--top-k 8 \
--jaccard-cap 0.55 \
--last-text-strategy random_same_ptype
```

- **`--per-position-type` ON.** Cross-ptype masking is required for clean
  InfoNCE math; cosines across `last_text` ↔ `image_patch` are not
  comparable. Costs nothing.
- **`--top-k 8`** matches the V3 corpus shape and the SFT loader's
  `ar_nce_hard_negatives_per_anchor=4` sub-sampling.
- **`--jaccard-cap 0.55`** (tighter than the 0.7 used in this smoke). V3
  audit p90 of mined Jaccard was 0.48, mean 0.385 — cap = 0.7 dropped only
  1 candidate across all 200k topk_cosine pairs. Cap = 0.55 should drop
  ~5–10 % of `image_patch` candidates, biting the long tail where mined
  Jaccard reaches 0.65. If SA10's regression audit shows Jaccard p90 still
  > 0.5 after V4 captions land, drop the cap further to 0.5.
- **`--last-text-strategy random_same_ptype`.** This is the explicit
  Agent-5 recommendation (mine `last_text` from random within ptype until
  an earlier-layer extraction exists). If SA10 prefers a zero-noise
  setting, `drop` is also acceptable — the dataset loader's empty-pool
  fallback handles it. Do **not** ship `topk_cosine` for `last_text` until
  someone re-extracts at an earlier hidden layer (deferred work).
- **Layer change (deferred).** The actual fix is mining `last_text` from
  an earlier hidden layer; the current corpus is the last decoder layer.
  Out of scope for V4 dataset repair, but worth tracking — once available,
  switch `--last-text-strategy back to topk_cosine` and rerun.

### Breaking changes for the training-side loader

None.

- `LabeledPositionDataset._load_topk_cosine_index` reads only
  `anchor` + `negs` + `cos`; the new `position_type` / `strategy` fields
  are silently ignored.
- Anchors with `negs=[]` already fall back to the loader's "repeat-self"
  behavior in `_sample_hard_negatives` (so `--last-text-strategy drop`
  is safe even without code changes on the training side).
- The V3 `hard_negatives.jsonl` continues to audit cleanly: the audit's
  back-compat path infers ptype from the anchor ID when the V4 fields
  are absent.

**Optional future work (NOT required for V4 mining):**

- Per-ptype temperatures or z-score normalization in `_hard_negative_sims`
  inside `src/nla/training/sft.py` — Agent 5 recommendation (3.a).
  Currently every ptype shares the same `tau`; the smoke run shows
  `image_patch` cos ≈ 0.96 and `last_text` cos ≈ 0.96 are now numerically
  closer (because last_text is now random), so the asymmetry is smaller
  but still present.

## Artifacts

- `/tmp/sa8_smoke_hard_negatives.jsonl` — 101 580 rows, V4 schema.
- `/tmp/sa8_smoke_audit.md` + `/tmp/sa8_smoke_audit.json` — per-ptype audit.
- `/tmp/sa8_smoke.log` — full miner stderr.
- `/tmp/sa8_legacy_audit.md` — back-compat run on the existing V3
  `data/activations/libero_4suite_combined/hard_negatives.jsonl` to prove
  the audit changes don't regress legacy schema (mean cos 0.9782 matches
  the original Agent 5 number).

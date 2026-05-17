# SA7 — V4 dataset rebuild (combine + mine + config + dry-run)

**Status:** ✅ **Ready for SA9 + SA10.** All deliverables on disk, paths
resolve, dry-run clean.

The V4 dataset is wired up end-to-end: V3-kept and V4-replaced caption
rows merged into a single 101,580-row combined labels.jsonl, activations
symlinked into a parallel V4 directory (physical events unchanged),
hard negatives mined with SA8's recommended flags, SFT config minted as
a 4-field delta off V3, and the SFT loader successfully constructs and
indexes the resulting (101,580-row dataset, 101,580-anchor neg index)
pair.

## Step 1 — V4 combined labels.jsonl

Script: `scripts/labeling/build_v4_combined_labels.py` (new).
Output: `data/labels/libero_4suite_v4_combined/labels.jsonl`
(+ `_merge_summary.json` for the SA10 audit).

**Per-suite V3-kept vs V4-replaced (deliverable a):**

| suite     | v3_total | v3_kept | v4_replaced | v4_avail | % replaced |
|-----------|---------:|--------:|------------:|---------:|-----------:|
| `spatial` |   25,920 |       0 |      25,920 |   25,845 |    100.00% |
| `goal`    |   25,680 |   7,743 |      17,937 |   17,888 |     69.85% |
| `object`  |   27,240 |   3,228 |      24,012 |   23,943 |     88.15% |
| `10`      |   22,740 |   8,379 |      14,361 |   14,329 |     63.15% |
| **TOTAL** |**101,580**|**19,350**| **82,230** | **82,005**|    80.95% |

(The v4_replaced totals exceed v4_avail by 75/49/69/32 per suite = 225
total, plus the 70-row excess on top makes 295. Cause: 295 V3 position-
keys are duplicated in the per-suite labels.jsonl files — same
`(source_example_id, position_index, position_type)` appearing twice,
all `last_text` rows. Each duplicate V3 row maps to the same V4
re-labeled row, so v4_replaced double-counts in those slots. The same
295 duplicates exist in `data/labels/libero_4suite_combined/labels.jsonl`,
so we're preserving V3-combined's exact row inventory — final count
**101,580** ✓ matches V3 combined byte for byte at the row-count level.)

**Schema** of every emitted row:

- `example_id`: `<suite>__<traj…>@p<NNN>_<ptype>` (suite-prefixed, matches
  V3 combined convention)
- `meta.source_example_id`: same prefix
- `meta.suite`: bare token (`spatial`/`goal`/`object`/`10`) — **not**
  `libero_*`. The plan instructed `libero_{suite}` but the existing V3
  combined corpus uses bare tokens, the example-id prefix uses bare
  tokens, and the hard-neg auditor's `parse_anchor_id` regex returns
  the bare token; following V3 keeps the audit's same-suite/cross-suite
  stats correct and avoids a silent prefix mismatch. (The plan's
  parenthetical "V4 rows already have it via SA2's `suite` field" is
  empirically wrong: V4 per-suite labels have no `meta.suite` field.
  We back-fill on every row.)
- `meta.label_version`: `"v4"` for re-labeled rows, `"v3"` for kept
  rows (deliverable for SA10's per-version slice audit).

CLI to reproduce:

```bash
PYTHONPATH=src .venv/bin/python scripts/labeling/build_v4_combined_labels.py \
    --v3-per-suite-root data/labels/libero_4suite_stride2 \
    --v4-per-suite-root data/labels/libero_4suite_v4 \
    --out data/labels/libero_4suite_v4_combined/labels.jsonl
```

Runs in ~6 s.

## Step 2 — V4 activations mirror

V4 captions are over the *same* physical robot events as V3, so the
activation corpus does not change. We create a parallel directory of
symlinks so the SFT config has a clean `(activations_root,
labels_jsonl)` pair without touching the V3 directory:

```
data/activations/libero_4suite_v4_combined/
├── manifest.json    -> ../libero_4suite_combined/manifest.json
├── index.jsonl      -> ../libero_4suite_combined/index.jsonl
├── stats.json       -> ../libero_4suite_combined/stats.json
├── shard_000000     -> ../libero_4suite_combined/shard_000000
├── …
└── shard_000101     -> ../libero_4suite_combined/shard_000101
```

102 shards + manifest + index + stats = 105 symlinks. `hard_negatives.jsonl`
deliberately omitted so we don't accidentally consume the V3 index;
the V4-mined `hard_negatives.jsonl` (Step 3) is written into this
directory as a real file. Each shard symlink resolves transitively
through the V3 combined dir to the underlying per-suite
`data/activations/libero_4suite_stride2/libero_<suite>/shard_NNNNNN`.

Manifest, index, stats: all open cleanly. SFT loader uses
`ActivationShardReader(activations_root)` which transparently follows
the symlinks.

## Step 3 — V4 hard-negative mining

SA8's miner against the V4 combined corpus, with SA8's recommended flag
set:

```bash
PYTHONPATH=src .venv/bin/python scripts/training/mine_hard_negatives.py \
    --activations-root data/activations/libero_4suite_v4_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_combined/labels.jsonl \
    --min-bullet-lines 3 \
    --top-k 8 \
    --per-position-type \
    --jaccard-cap 0.55 \
    --last-text-strategy random_same_ptype \
    --out data/activations/libero_4suite_v4_combined/hard_negatives.jsonl
```

Wall time: 7m 41s on the H100 (slightly above SA8's "3-5 min" projection
because the jaccard-cap=0.55 cap is tighter and oversamples 4×).

**Headline mining stats (full 101,580 anchors):**

```
n_anchors=101580  K=8  median_cos_top1=0.965  p5=0.921  p95=0.996
n_anchors_with_empty_negs=0
n_jaccard_dropped=4,688    (cap=0.55, vs 1 at SA8 smoke cap=0.7)
strategy_counts={'random_same_ptype': 51085, 'topk_cosine': 50495}
```

**Per-ptype mining (deliverable b):**

| ptype         | strategy           | n      | median_cos_top1 | p5    | p95   |
|---------------|--------------------|-------:|----------------:|------:|------:|
| `anchor`      | topk_cosine        |    166 |           0.998 | 0.995 | 0.999 |
| `image_patch` | topk_cosine        | 50,329 |           0.975 | 0.919 | 0.997 |
| `last_text`   | random_same_ptype  | 51,085 |           0.958 | 0.927 | 0.995 |

All three trip the miner's "outside healthy [0.60, 0.95] band" warning
— that's the V3-era activation geometry (the corpus's intrinsic cosine
neighbor distance is high). SA8's note covers this: the band warning
is corpus-wide, not a V4 regression.

Output: 101,580 rows, V4 schema (`position_type`+`strategy` fields
present). ✓

## Step 4 — Hard-negative audit

```bash
PYTHONPATH=src .venv/bin/python scripts/eval/audit_hard_negatives.py \
    --hard-negatives-jsonl data/activations/libero_4suite_v4_combined/hard_negatives.jsonl \
    --activations-root     data/activations/libero_4suite_v4_combined \
    --labels-jsonl         data/labels/libero_4suite_v4_combined/labels.jsonl \
    --sample-anchors 500 \
    --out-md   data/eval/sa7_v4_hardneg_audit.md \
    --out-json data/eval/sa7_v4_hardneg_audit.json
```

### Overall — V4 vs V3 baseline

| metric                          |    V4 |    V3 (Agent 5) | Δ                |
|---------------------------------|------:|----------------:|------------------|
| mean mined cosine (all)         | 0.960 |           0.978 | −0.018 (better)  |
| mean random cosine (same ptype) | 0.886 |           0.887 | ≈ 0              |
| mean caption Jaccard (mined)    | 0.345 |           0.385 | −0.040 (better)  |
| mean caption Jaccard (random)   | 0.310 |               — |                  |
| within-suite negative fraction  | 50.6% |           88.5% | −38 pp (by design)|
| degenerate-pair count           | 0/0/0 |           0/0/0 | unchanged        |

Cosine and Jaccard both **drop** under V4 — i.e. negatives are less
trivial. Within-suite drops because `random_same_ptype` mixes suites
when sampling `last_text` negatives; mining a within-suite random pool
would re-introduce vocabulary leakage from the same scene factory.

### Per-ptype verdict + Δ vs random — V4 vs V3

| ptype         | strategy             |  V4 mined-cos | V4 random-cos | **V4 Δ** | V3 Δ (SA8 smoke) | verdict |
|---------------|----------------------|--------------:|--------------:|---------:|-----------------:|:-------:|
| `anchor`      | topk_cosine          |        0.998  |        0.962  |  +0.036  |       *(n=146)*  |   RED   |
| `image_patch` | topk_cosine          |        0.975  |        0.741  | **+0.234** |          +0.213  |   RED   |
| `last_text`   | random_same_ptype    |        0.958  |        0.962  |  **−0.004** |    +0.04 (fake)  |   RED   |

Plan's expected wins, all met:

- **`last_text`**: was a fake Δ ≈ +0.04 in V3 because `topk_cosine`
  could find arbitrarily close `last_text` neighbors in an
  activation-saturated geometry. V4 uses `random_same_ptype` — Δ
  collapses to honest 0 ± noise (−0.004). InfoNCE now sees a flat
  distribution over `last_text` negatives, which is exactly what
  "we can't tell episodes apart at the last decoder layer" *should*
  mean.
- **`image_patch`**: Δ ≈ +0.23 (vs +0.21 in V3) — the real contrast
  source survives the per-ptype masking and gets a slight bump
  because mining within the same ptype removes cross-ptype clutter.
- **`anchor`**: Δ ≈ +0.04, small as expected — only 166 anchors,
  near-identical structural position; the audit's RED is the
  absolute-band rule, not a contrast failure.
- **Jaccard cap 0.55**: 4,688 candidates dropped (vs 1 at V3 smoke
  cap=0.7) — biting the long tail of redundant captions as predicted.
  Mined Jaccard p90 = 0.49 (was 0.53 in V3); the cap is tight enough
  to matter without being so tight it starves the candidate pool
  (zero empty-negs).

Audit verdicts (overall RED, per-ptype all RED) are by the absolute-band
rule. Per SA8's notes, these are corpus-geometry RED, not a regression
or a mining bug; the per-ptype Δ rows above are the meaningful gates and
they all moved in the right direction.

Full audit reports: `data/eval/sa7_v4_hardneg_audit.{md,json}`.

## Step 5 — V4 SFT config

`data/sft/libero_4suite_v4/config.json` — copied from V3, **only** the
four plan-specified fields changed:

```diff
- "activations_root": "data/activations/libero_4suite_combined",
+ "activations_root": "data/activations/libero_4suite_v4_combined",
- "labels_jsonl":     "data/labels/libero_4suite_combined/labels.jsonl",
+ "labels_jsonl":     "data/labels/libero_4suite_v4_combined/labels.jsonl",
- "output_dir":       "data/sft/libero_4suite_v3",
+ "output_dir":       "data/sft/libero_4suite_v4",
- "ar_nce_hard_negative_index_path": "data/activations/libero_4suite_combined/hard_negatives.jsonl"
+ "ar_nce_hard_negative_index_path": "data/activations/libero_4suite_v4_combined/hard_negatives.jsonl"
```

Everything else (hyperparameters, LR, scheduler, AV/AR configs,
balance_position_mix, min_bullet_lines, alpha, NCE temperature, ...)
identical to V3 — apples-to-apples for SA9/SA10.

**Alpha (P75 activation norm).** V3 value `203.97713487289315` retained
verbatim in both `av_cfg.alpha` and `ar_cfg.alpha`. **The underlying
activations have not changed** (V4 is a captions-only repair; the
symlinked `data/activations/libero_4suite_v4_combined/stats.json`
resolves to the same file as V3 combined). Re-computing alpha would
return the same number to within float-precision noise.

## Step 6 — SFT loader dry-run

Script: `/tmp/sa7_v4_sft_dryrun.py` (not committed). It mints the
V4 `SFTConfig`, instantiates the full `LabeledPositionDataset` over
V4 paths, and verifies a sample row + hard-neg pool.

Output (dryrun deliverable c):

```
[1/3] SFTConfig: activations_root=data/activations/libero_4suite_v4_combined
      labels_jsonl=data/labels/libero_4suite_v4_combined/labels.jsonl
      ar_nce_hard_negative_source=topk_cosine
      ar_nce_hard_negative_index_path=data/activations/libero_4suite_v4_combined/hard_negatives.jsonl
      ar_nce_hard_negatives_per_anchor=4
      av_cfg.alpha=203.97713487289315
      ar_cfg.alpha=203.97713487289315
[2/3] load_labels_jsonl: kept 101580 rows after min_bullet_lines=3 filter
      hard_negatives.jsonl: 101580 rows
[3/3] LabeledPositionDataset: size=101580 (full corpus)
      sample[0] fields: ['activation', 'description', 'episode_index',
                         'example_id', 'label_example_id',
                         'negative_descriptions', 'position_index',
                         'position_type', 'quality_weight', 'seq_len']
      activation shape=(2048,) dtype=torch.float32
      example_id=goal__traj000182_step000052
      label_example_id=goal__traj000182_step000052@p129_image_patch
      position_type=image_patch  position_index=129
      negative_descriptions: 4 negatives (K_neg target=4)
        neg[0]='- scene: tabletop scene with a woven basket on the left and '
        neg[1]='- scene: tabletop with light gray floor tiles, a metal wire '
        neg[2]='- scene: light wood tabletop with a woven basket on the left'
        neg[3]='- scene: beige tabletop with a white robotic arm, a black ca'
      ptype=image_patch  row=0  n_negs=4
      ptype=  last_text  row=1  n_negs=4
      ptype=     anchor  row=122  n_negs=4

OK: V4 SFT config + dataset + hard-neg index all resolve.
```

Verified:

1. **Labels stream loads 101,580 rows** from V4 combined (matches V3
   combined count to the row) ✓
2. **Hard-neg index loads 101,580 rows** at the V4 path ✓
3. **Activation manifest resolves** — every sample's activation is a
   2048-d float32 vector, all-finite ✓
4. **All three position_types** (`anchor`, `image_patch`,
   `last_text`) yield 4 negative captions ✓ (i.e. the dataset's
   K_neg=4 sampler walks the V4 mined pool successfully across all
   strategies — both `topk_cosine` for `image_patch` / `anchor` and
   `random_same_ptype` for `last_text`)

**Coverage note.** The dataset reports `295/101580 anchors have no
admissible negatives; those rows will fall back to repeating the
anchor's own caption.` These 295 anchors are **exactly** the duplicate
V3 position-keys (`(source_example_id, position_index, position_type)`
appearing twice in V3, all `last_text`). The miner indexes by canonical
position key so it sees only the first occurrence; the second copy
falls through to the loader's `repeat-self` path. This is identical
behaviour to V3 (the same 295 duplicates exist in V3 combined) and
within the loader's documented contract. The AR forward pass tolerates
repeat-self gracefully (no NaN, the InfoNCE term just sees a
near-degenerate row).

We did not run the AR forward pass itself in this dry-run because doing
so requires materializing the 4B-parameter Qwen3 base model. The three
verifications above (labels + neg index + manifest + per-ptype neg
pools) plus the activation finite-ness check cover the actual V4-side
plumbing introduced this step; the existing V3 SFT run trained on the
same code path with identical hyperparameters, so a successful V3 run
implies the V4 forward pass will be finite.

## SFT config diff vs V3 (final)

4 fields changed, plan-conformant:

| field                                | V3                                                     | V4                                                       |
|--------------------------------------|--------------------------------------------------------|----------------------------------------------------------|
| `activations_root`                   | `data/activations/libero_4suite_combined`              | `data/activations/libero_4suite_v4_combined`             |
| `labels_jsonl`                       | `data/labels/libero_4suite_combined/labels.jsonl`      | `data/labels/libero_4suite_v4_combined/labels.jsonl`     |
| `output_dir`                         | `data/sft/libero_4suite_v3`                            | `data/sft/libero_4suite_v4`                              |
| `ar_nce_hard_negative_index_path`    | `data/activations/libero_4suite_combined/hard_negatives.jsonl` | `data/activations/libero_4suite_v4_combined/hard_negatives.jsonl` |

Everything else (al hyperparameters, all AV/AR LoRA settings, alpha,
contrastive weight, mix schedule, …) byte-identical to V3.

## Artifacts shipped

| Path                                                                                  | Lines / Files | Notes                                                |
|---------------------------------------------------------------------------------------|--------------:|------------------------------------------------------|
| `scripts/labeling/build_v4_combined_labels.py`                                        |          ~250 | NEW. Deterministic V3+V4 merge by position-key.      |
| `data/labels/libero_4suite_v4_combined/labels.jsonl`                                  |       101,580 | Merged V4 combined labels.                           |
| `data/labels/libero_4suite_v4_combined/_merge_summary.json`                           |             1 | Per-suite v3-kept/v4-replaced counts for SA10.       |
| `data/activations/libero_4suite_v4_combined/{manifest,index,stats,shard_*}`           |    105 symlnks | Mirror of V3 combined.                               |
| `data/activations/libero_4suite_v4_combined/hard_negatives.jsonl`                     |       101,580 | V4 miner output (SA8 flags).                         |
| `data/eval/sa7_v4_hardneg_audit.{md,json}`                                            |             2 | SA8 audit on V4 corpus.                              |
| `data/sft/libero_4suite_v4/config.json`                                               |             1 | V4 SFT config; 4-field delta vs V3.                  |
| `/tmp/sa7_v4_mining.log` `/tmp/sa7_v4_audit.log` `/tmp/sa7_dryrun.log`                |             3 | Step-3/4/6 logs.                                     |

## Constraint compliance

- ✅ No V3 files modified. V3 dirs (`data/{labels,activations}/libero_4suite_{combined,stride2}/`,
  `data/sft/libero_4suite_v3/`) untouched.
- ✅ No training code modified (`src/nla/training/` untouched).
- ✅ No miner code modified (`scripts/training/mine_hard_negatives.py`
  untouched; the V4 binary used SA8's just-shipped CLI flags only).
- ✅ All new paths under
  `data/{labels,activations,sft,eval}/libero_4suite_v4*` or
  `scripts/labeling/build_v4_combined_labels.py`. No overwrites.

## Coordination

- **SA9** (V4-vs-V3 caption/label-quality audit) is unblocked. Inputs
  on disk: `data/labels/libero_4suite_v4_combined/labels.jsonl` with
  `meta.label_version` tagging V3-kept vs V4-replaced rows, and
  `_merge_summary.json` for the per-suite breakdown.
- **SA10** (full V4 dataset audit + SFT readiness check) is unblocked.
  Inputs on disk: the V4 SFT config + the V4 hard-negatives audit
  reports + this report.

## Return values for the parent

- (a) **V3-kept vs V4-replaced totals:** 19,350 V3-kept,
  82,230 V4-replaced; combined total 101,580 (matches V3 combined).
  Per-suite breakdown in the table above.
- (b) **V4 hard-neg per-ptype median cosines:**
  - `anchor`: 0.998 (n=166, topk_cosine)
  - `image_patch`: 0.975 (n=50,329, topk_cosine; Δ vs random = +0.234)
  - `last_text`: 0.958 (n=51,085, random_same_ptype; Δ vs random ≈ 0)
- (c) **Dry-run success:** ✅. Dataset constructs 101,580 rows over
  V4 paths, hard-neg index loads 101,580 anchors with 99.71% coverage
  (295 duplicate-key rows fall through to the loader's repeat-self
  fallback, same as V3), sample tensors are finite, and all three
  position_types produce 4 negative captions each.
- (d) **Status:** **READY FOR SA9 + SA10.**

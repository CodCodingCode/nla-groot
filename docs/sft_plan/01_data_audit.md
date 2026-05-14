# 01 – Warm-start label data audit (pre-SFT)

Audit run: 2026-05-14, against the three on-disk `labels.jsonl` files and
their matching `data/activations/<run>/index.jsonl`. All numbers come from
direct file reads; the audit script lives at `/tmp/audit_labels.py` and the
machine-readable output at `/tmp/audit_out/{droid_smoke,droid_ep1,droid_100ep}.json`.

The reference target distribution is the repo's
`POSITION_MIX = {last_text: 0.40, image_patch: 0.40, anchor: 0.20}` from
`src/nla/layer_spec.py`.

---

## 1. Coverage

### 1.1 Row counts and provenance

| run          | label rows | act. examples | label/example | manifest planned | gap     |
|--------------|-----------:|--------------:|--------------:|-----------------:|--------:|
| droid_smoke  |          4 |             4 |          1.00 |                4 |       0 |
| droid_ep1    |      1,064 |           266 |          4.00 |            1,064 |       0 |
| droid_100ep  |     99,968 |        25,084 |          3.99 |          100,336 |   −368  |

- `droid_100ep` ran with `positions_per_example=4` against 25,084 source
  examples; manifest says 100,336 planned, 100,336 completed, but the live
  `labels.jsonl` is **368 rows short** (0.37% gap). Likely lost during a
  rerun: `labels.jsonl.bak` (May 12) and the current `labels.jsonl` (May 14)
  have different sizes, so part of the file was rewritten.
- `droid_ep1` has `4.00` labels per example (266 ex × 4 positions = 1,064 rows).
  3 rows are exact duplicates of an existing `(source_example_id,
  position_index, position_type)` key — minor noise, not blocking.
- `droid_100ep` has **0 duplicate position keys** across 99,968 rows.
- `droid_smoke` is just the 4-row API smoke test.

### 1.2 Activation join health

For every row, `meta.source_example_id` was checked against the activations
`index.jsonl`:

| run          | rows missing matching `example_id` | seq_len mismatches |
|--------------|-----------------------------------:|-------------------:|
| droid_smoke  |                                  0 |                  0 |
| droid_ep1    |                                  0 |                  0 |
| droid_100ep  |                                  0 |                  0 |

Every label is recoverable. The dataset code (`LabeledPositionDataset` in
`src/nla/training/dataset.py`) will not silently drop rows.

### 1.3 Errors / empties

| run          | rows with `error` | rows with empty `description` |
|--------------|------------------:|------------------------------:|
| droid_smoke  |                 0 |                             0 |
| droid_ep1    |                 0 |                             0 |
| droid_100ep  |                 0 |                             0 |

Zero failed labeling calls in all three files. Either the OpenAI runs were
cleanly retried away or failures were dropped — but `LabeledPositionDataset`
already filters `error` and empty `description`, so this is not a risk for
SFT.

### 1.4 Episode coverage (droid_100ep)

- 100 distinct `episode_index` values, ranging **1 → 127** (with gaps —
  several episodes were skipped during extraction).
- Rows per episode: **min 357, p25 625, p50 857, p75 1244, max 3329, mean
  ≈ 1000**. Long-tail: a few long episodes (124, 127) contribute >3000
  rows each. Episode-stratified holdout will produce uneven val sets but
  the existing `_split_episode_aware` is already designed for this; no
  blocking issue.

---

## 2. Position-type distribution vs `POSITION_MIX`

The repo's intended mix is `{last_text: 0.40, image_patch: 0.40, anchor: 0.20}`.
What we actually have:

| run          | image_patch | last_text | anchor   |
|--------------|------------:|----------:|---------:|
| target       |       0.400 |     0.400 |    0.200 |
| droid_smoke  |       0.250 |     0.500 |    0.250 |
| droid_ep1    |       0.750 |     0.178 |    0.072 |
| droid_100ep  | **0.7525**  | **0.1638**| **0.0837** |

Severe skew toward `image_patch` in both real runs. Why this happens:

- `sample_positions_per_example(..., n_per_example=4)` draws **without
  replacement** within an example.
- A sequence has only **1 `last_text` and 1 `anchor`** but ~256 image-patch
  tokens (per `image_patch_meta=[k, 256]`). Once `last_text` and `anchor`
  are picked, the remaining 2 of 4 picks must fall back to `image_patch`.
- Worse, the sampler often fails to pick `last_text`/`anchor` even on its
  weighted draws (likely because of the no-replacement loop biasing toward
  the larger pool).

Per-example breakdown of which position types each source example got
labeled at (droid_100ep):

| positions/example | sources        |
|-------------------|---------------:|
| 4                 |        24,717  |
| 3                 |           366  |
| 2                 |             1  |

| coverage condition                     | sources | %     |
|----------------------------------------|--------:|------:|
| has all 3 position types               |     270 | 1.1%  |
| missing `last_text` label              |   8,711 | 34.7% |
| missing `anchor` label                 |  16,713 | 66.6% |

So **66.6% of source examples have no `anchor` label at all**, and
**34.7% have no `last_text` label**. This is not a join bug; it's a
sampling bug in how `sample_positions_per_example` allocates the 4 picks.

Effective totals available for SFT:

- `image_patch`: 75,224
- `last_text`:   16,373
- `anchor`:       8,371

These are still large absolute counts — even 8,371 `anchor` rows is more
than enough for warm-start SFT — so we're not under-data, we're under-
*weighted* relative to the documented mix.

---

## 3. Description quality

### 3.1 Style / format conformance (droid_100ep)

- Mean description length: **828 chars** (min 460, max 1,261). Comparable
  to Anthropic NLA paper warm-start length.
- Bullet count per row:

  | bullets | rows  |
  |--------:|------:|
  |       1 |     3 |
  |       3 |     1 |
  |       4 | 1,022 |
  |       5 | 98,942 |

  99% of rows use the requested 5-bullet format; 1% use 4 bullets;
  4 anomalies have ≤3 bullets and should be filtered out before SFT.

- **Canonical-bullet category presence** (fraction of rows that contain
  this category at least once):

  | category      | rows   | fraction |
  |---------------|-------:|---------:|
  | scene         | 99,964 | 99.996%  |
  | target        | 99,164 | 99.20%   |
  | image_region  | 76,640 | 76.66%   |
  | distractor    | 68,358 | 68.38%   |
  | spatial       | 67,680 | 67.70%   |
  | gripper       | 53,501 | 53.52%   |
  | language      | 18,424 | 18.43%   |
  | plan          |  7,306 |  7.31%   |
  | motion        |    161 |  0.16%   |

  `scene` and `target` are nearly universal — good. `motion` is essentially
  absent (0.16%) and `plan` is very rare (7.3%). The teacher is tracking
  *what's there* (scene/target/distractor/spatial/image_region/gripper)
  more than *what to do next* (plan/motion). That's a noticeable gap from
  the prompt's stated goal of "what features the model is internally
  tracking *to choose the next action*".

- **Non-canonical compound categories** (top, droid_100ep): the prompt
  lists 9 allowed categories; ~6,800 rows (≈6.8%) contain at least one
  out-of-vocab category, dominated by `gripper/spatial` (3,734),
  `spatial/plan` (504), `gripper/motion` (442), `distractor/spatial` (410).
  These are merged-bullets like `- gripper/spatial: ...` rather than truly
  invented categories. They would show up to AV as novel category tokens.
  The `build_strict_position_prompt` in `prompts.py` was already added to
  re-label this kind of row, but it has not been re-run on `droid_100ep`.

### 3.2 Qualitative review (20 random rows from droid_100ep)

Sampled 20 rows (seed 0) across episodes 8, 15, 22, 32, 39, 45, 49, 62,
67, 73, 75, 85, 88, 91, 92, 100, 106, 127. Position types: 13
image_patch, 5 last_text, 1 anchor, 1 fallback (matches the sampling
skew).

**What is working well:**

- The `target` bullet correctly identifies the task object in
  ≈18/20 rows, even on diverse DROID scenes (blue block, knife, cap,
  curtain pull cord, scissors, medicine bottle, dishwasher plate).
- `scene` bullets describe surface, lighting, surrounding objects in
  concrete terms ("white round table", "kitchen countertop with stove
  to the right"), not boilerplate.
- `gripper` bullets are usually grounded in the visible end-effector
  ("black two-finger parallel gripper descending into the bowl").
- For `last_text` and `anchor` positions the labeler does insert a
  task-grounded `language:` or `plan:` last bullet ("instruction has
  been read; goal is to grasp …"), as the prompt requested.

**Failure modes / weaknesses observed:**

1. **Confabulated `image_region` content.** The prompt tells the model
   "image patch k of 256" but does not actually highlight which patch.
   The labeler responds with plausible-sounding "this patch contains
   the gripper finger above the can" descriptions that are guessed
   from k as a coarse position index. Sample 3 (image patch 89/256)
   describes "lower-left quadrant of the close-up image patch" — the
   model is inferring patch location from the index, not from any
   visual cue. **This is a teacher-side hallucination by construction**
   and the AV will learn to do the same.
2. **Compound-category bullets.** Lines like `- gripper/spatial: ...`
   bypass the 9-category vocabulary (≈6.8% of rows). Not catastrophic
   but the AV will inherit them.
3. **Near-duplicate descriptions across consecutive timesteps.** Many
   episodes share the same instruction and very similar frames; the
   teacher tends to produce paraphrases of the same 4–5 bullets across
   neighboring steps (e.g. 90% of episode 1's "Put the blue block in
   the green bowl" rows look near-identical). Not a bug, but the AV
   may overfit to per-task wordings.
4. **Almost no `motion` / `plan`.** Even on `last_text` positions where
   the model presumably is committing to "reach left" or "release",
   the labeler stays at the goal level and rarely articulates the
   imminent motion direction. `plan` shows up in only 7.3% of rows.
5. **Occasional unicode/em-dash glitches** (`—`, `’`) — already in
   examples; harmless for tokenization but cosmetic.

No GPT-style preamble ("Here is the description:") was observed in the
20-sample slice, and the system prompt explicitly forbids it. No
clearly-confabulated objects were observed in the sample (the targets
match the instructions in 18/20 cases; sample 4 shows a teal cup the
gripper is actually holding, which is grounded).

---

## 4. Position mix vs paper / repo plan

- The repo plan and `POSITION_MIX` say 40/40/20 — the current label set is
  ≈75/16/8, almost the inverse weighting between `last_text` and
  `image_patch`.
- Two ways to interpret this:
  - *Match the documented mix.* Subsample `image_patch` rows down to ~16k
    so we get ~16k/16k/8k = 40/40/20. Loses ~58k labeled rows (≈58% of
    the file) but no extra spend.
  - *Match natural distribution.* The extraction stats say
    `image_token_fraction = 0.912`, so most positions in real GR00T
    sequences are image patches anyway. A 75/16/8 mix is closer to that.
    Argument for keeping all rows: train the AV on the actual position
    statistics it'll see at RL/inference time.
- The Anthropic NLA paper trains on a deliberately *flat* mix per layer
  to avoid token-frequency bias. The repo plan agrees. So strictly we
  should rebalance.

`LabeledPositionDataset` does **not** rebalance by `position_type` — it
uses each row uniformly. So whatever skew is in the file is the skew the
AV sees. If we want 40/40/20 we must either:

- Filter the file to a balanced subset, or
- Add a `WeightedRandomSampler` keyed on `position_type` to the
  DataLoader, or
- Re-run labeling with **balanced positions per example** (the cheapest
  way is to call `sample_positions_per_example(..., n_per_example=K)`
  with `K=2` and then top up `last_text`/`anchor` only — but it requires
  another OpenAI spend of ≈30k rows worth).

For the **first** SFT run, the cheapest choice is the in-loader weighted
sampler — implementation cost is one new dataloader, no new OpenAI
spend, and behavior is reversible.

---

## 5. Recommendation for the first real SFT run

**Use `data/labels/droid_100ep/labels.jsonl` as-is, with two cheap fixes
in the data loader.** Justification and concrete fixes follow.

### Why droid_100ep is ready enough

- 99,968 grounded `(h, description)` pairs, **0 errors, 0 missing
  activation joins, 0 duplicate position keys, 25,084 unique source
  examples across 100 episodes**. That is comfortably more than the
  warm-start budgets in the NLA paper for this scale of model.
- Quality is decent: 99% conform to the 5-bullet format; `scene`/
  `target` are present in ≥99% of rows; the `target` bullet matches the
  instruction in 18/20 sampled rows.
- The activations side is sound (P75 norm 197.4 → α candidate ≈ 200,
  91% image-token fraction, 49 shards, 25,084 examples) and joins
  perfectly.

### What we should NOT do

- **Don't relabel from scratch.** The qualitative issues (compound
  categories, missing `motion`, confabulated `image_region`) are not
  blocking for warm-start; the AV's job is to imitate this distribution
  and we're then planning to RL-correct in the GRPO phase.
- **Don't drop `image_patch` rows just to hit 40/40/20.** Throwing away
  60% of the data on a first run is wasteful when the DataLoader can
  reweight at sample time.
- **Don't bother with `droid_smoke` (n=4) or `droid_ep1` (n=1064, single
  task / single instruction)** for a real SFT run. Use `droid_ep1` only
  for unit/integration tests of the SFT loop.

### Cheap pre-SFT fixes (≤1 day, no new OpenAI spend)

1. **Filter pathological rows.** In `load_labels_jsonl`, additionally
   drop rows with `bullet_count < 4`. Affects 4 rows in droid_100ep
   (`{1: 3, 3: 1}`); trivial.
2. **Rebalance by position type at sample time.** Add a
   `WeightedRandomSampler` to the SFT DataLoader keyed on
   `position_type`, using class weights `{last_text: 0.40/0.1638,
   image_patch: 0.40/0.7525, anchor: 0.20/0.0837}`. This matches the
   intended `POSITION_MIX` without re-labeling. Document the choice in
   the SFT config.
3. **(Optional) Re-relabel the 6.8% non-canonical-category rows** with
   the existing `build_strict_position_prompt` in
   `src/nla/labeling/prompts.py`. ≈6,800 calls to gpt-5-mini —
   essentially free, and removes the AV's exposure to compound-category
   bullets like `gripper/spatial`. Worth it if we're already touching
   the labeling pipeline; skip for the first run otherwise.
4. **Keep `episode-stratified` holdout** (the dataset default). With 100
   episodes and `held_out_fraction=0.05–0.10`, val will see 5–10
   distinct episodes never used in train.

### What to investigate AFTER the first SFT run (don't gate on it now)

- Whether the AV inherits the teacher's `image_region` confabulation
  pattern; if yes, it's a clean target for the GRPO reconstruction
  reward to correct.
- Whether under-representing `motion` in SFT shows up as missing
  reconstruction signal for high-action timesteps. If so, do a small
  top-up labeling pass with a `motion`-mandatory prompt variant.
- Whether bridging in the `bridge_pilot` activations (the directory is
  empty today) materially improves out-of-task generalization. As of
  this audit there is **no** `bridge` extraction or labels on disk.

---

## 6. TL;DR

`droid_100ep`'s label file is **production-ready as warm-start data**:
99,968 rows, zero error/missing-join/duplicate issues, 99%+ format
conformance, plausible content. Its one real defect is the position-type
distribution (75/16/8 vs the intended 40/40/20), caused by the
`positions_per_example=4` no-replacement sampler exhausting `last_text`
and `anchor` after one pick each. Fix at the DataLoader (weighted
sampler), not by re-labeling. Filter the 4 sub-4-bullet rows and ship
the first SFT run on this file.

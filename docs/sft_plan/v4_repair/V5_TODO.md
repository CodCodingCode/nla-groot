# V5 TODO — beyond content quality

V4 is a **content-quality fix**: it kills hallucinations, scaffold leakage, motor
imperatives, and position-type blur. It does **not** address the optimization
side or the activation-layer issues that bound how much detail the model can
actually learn. This is the queue for V5.

Reference framework: a dataset is good only if **(1) inputs vary with detail**,
**(2) labels encode detail**, **(3) optimization punishes vagueness**, plus the
four properties — specificity, anti-mode diversity, slot-aligned supervision,
quality grading that bites.

V4 scoreboard against that framework:

| Dimension                       | V4 status                  | Action needed for V5         |
|---------------------------------|----------------------------|------------------------------|
| 1. Specificity-conditioned      | ~85% (libero_10 weak)      | Item 4 below                 |
| 2. Anti-mode diversity          | Shuffled, not killed       | Items 1, 5 below             |
| 3. Slot-aligned supervision     | Mostly yes                 | Item 2 (last_text layer)     |
| 4. Quality grading that bites   | Infrastructure unused      | Item 3 below                 |
| (A) Input varies with detail    | Not addressed              | Item 2 below                 |
| (B) Label encodes detail        | Strong on target/scene, weak on plan | Item 1 below        |
| (C) Optim punishes vagueness    | Partial                    | Items 3, 5, 6 below          |

---

## Priority queue (do in order; each unblocks the next)

### 1. Defuse the new `plan`-bullet template collapse  *(prompt fix, ~$5)*

V4 traded the V3 "phase active" template (26.8% DF) for a new one:
`over the next 3 timesteps` / `phase over the` / `before placing on` at
47–69% document frequency in `plan:` bullets. Top-phrase DF actually went
**up** (V3 49.8% `robot arm/scene` → V4 71.5% `the next/plan`).

- Edit `_LAST_BULLET_BY_POSITION_TYPE["last_text"]` in
  [`src/nla/labeling/prompts.py`](../../src/nla/labeling/prompts.py) — drop
  the prescriptive "over the next 3 timesteps" phrasing; instead require
  the bullet to name **one specific motion + one specific object** without
  any temporal connector boilerplate.
- Add the new boilerplate phrases to the forbidden-phrase tuple in
  [`scripts/eval/audit_prompt_hardening.py`](../../scripts/eval/audit_prompt_hardening.py).
- Selective re-label of rows containing the new boilerplate (≈55k rows by
  current DF, ≈$40).

**Target:** top-phrase DF in any single bullet ≤ 15%; no phrase > 25% in
`plan`.

### 2. Re-extract `last_text` from an earlier hidden layer  *(GPU, few hours)*

Agent 5 confirmed and SA8 reproduced: the final-layer hidden state at the
last text token is **saturated** across episodes (cos 0.96 between random
same-suite/same-ptype rows). InfoNCE on this signal is structurally
degenerate; V4 worked around it via `random_same_ptype` mining, but the
underlying input variation just isn't there.

- Extract `last_text` `h` from layer −4 and layer −8 (in
  [`scripts/extraction/run_extract.py`](../../scripts/extraction/run_extract.py),
  there is a `--layer` flag — or add one if it isn't yet wired).
- Re-run [`scripts/eval/audit_hard_negatives.py`](../../scripts/eval/audit_hard_negatives.py)
  on the new shards; pick the layer whose random-pair cosine sits in
  [0.5, 0.8] (the "healthy" band Agent 5 documented).
- Re-mine V4 hard negatives against the chosen layer using SA8's
  [`scripts/training/mine_hard_negatives.py`](../../scripts/training/mine_hard_negatives.py)
  with `--per-position-type --top-k 8`.

**Target:** `last_text` mined-vs-random cosine Δ ≥ 0.12 (Agent 5's
"healthy" threshold).

### 3. Turn on quality weights  *(half day, ~$30 for grading)*

[`SFTConfig.use_quality_weights`](../../src/nla/training/sft.py) exists but is
`false`. We have SA9's 500-row judge JSONL. The infrastructure to downweight
templated / non-grounded rows is already wired — just disconnected.

- Grade a larger stratified sample (~5k rows, ~$50) with `gpt-5.1` using
  [`scripts/eval/verify_libero_label_quality.py`](../../scripts/eval/verify_libero_label_quality.py).
- Build a small script `scripts/training/build_quality_weights.py` that
  joins judge verdicts to `(source_example_id, position_index, position_type)`
  and emits a `quality_weights.jsonl` (anchors with `B=specific` → weight 1.0,
  `B=somewhat_specific` → 0.5, `B=non_specific` → 0.0 or 0.1).
- For ungraded rows (the vast majority): inherit per-suite, per-ptype mean
  weight from the graded subset.
- Set `use_quality_weights=true` and `quality_weights_path=...` in
  [`data/sft/libero_4suite_v4/config.json`](../../data/sft/libero_4suite_v4/config.json)
  (or a sibling `_quality_weighted.json`).

**Target:** AR cosine on held-out V4-judged-`specific` rows ≥ 5pp above
mean cosine — i.e., the weighting actually concentrates learning on good
rows.

### 4. Apply spatial-style suite rules to `libero_10`  *(prompt fix, ~$10)*

`libero_10` (long-horizon composite tasks) regressed −14.75pp in SA9's V4
judge run: `10/last_text −18.60`, `10/anchor −13.89`, `10/image_patch
−11.63`. The pattern matches what `libero_spatial` looked like before SA2's
SP-1..SP-7 addendum — instruction-anchored hallucination of object/phase
state for tasks the labeler can't visually verify (multi-step kitchen
sequences).

- Author `_V4_SUITE_ADDENDA["libero_10"]` in
  [`src/nla/labeling/prompts.py`](../../src/nla/labeling/prompts.py)
  mirroring SP-6/SP-7's "visually verify every object before naming it"
  pattern, plus a "describe the CURRENT sub-task, not the goal" rule.
- Pilot judge on ~50 V3-failure rows in `libero_10`; iterate if needed
  (SA2's iter-0 → iter-1 found the SP-6/SP-7 lever was the bigger one
  vs the original SP-1..SP-5).
- Selective re-label of `libero_10` rows where V4 judge B ≠ `specific`
  (~3k rows, ~$3).

**Target:** `libero_10` overall B ≥ 88% (matches the other healthy
suites).

### 5. Decide the `language:` bullet contract  *(prompt fix, no API cost)*

V4 collapsed `language:` to 0.52% on `last_text` rows. The V4 prompt
half-asks (marked OPTIONAL), and the labeler effectively never delivers.
This means the audits keep flagging it but it's expected behavior.

Two clean options:

- **Option A:** make `language:` REQUIRED on `last_text` rows, with a
  template ("language: instruction parses as: '<verb> <noun> on/with
  <reference>'") and re-label all `last_text` rows missing it (~14k rows,
  ~$10).
- **Option B:** drop `language:` from `V4_BULLET_CATEGORIES` entirely and
  remove the audit checks. Documents what the prompt actually produces.

**Target:** whichever option, the audit reports 100% (present) or 100%
(absent) per the declared contract.

### 6. Add anti-template loss term  *(training fix, ~1 day)*

Items 1 + 3 attack the template problem from data side. The optimization
side complement: add an explicit penalty for emitting the dominant
phrase.

- In [`src/nla/training/sft.py`](../../src/nla/training/sft.py), add a
  config `anti_template_phrases: list[str] = []` and
  `anti_template_weight: float = 0.0`.
- During AV training, scan the greedy decode for any of the top-15
  V4-DF phrases (output of
  [`scripts/eval/audit_diversity.py`](../../scripts/eval/audit_diversity.py));
  multiply the per-step CE by `(1 + anti_template_weight)` for every
  token inside a matched phrase.
- Sweep `anti_template_weight` in {0.0, 0.5, 1.0, 2.0} on a small SFT;
  pick the value that drops top-phrase DF in val captions ≤ 15% without
  raising CE > 10%.

**Target:** AV greedy-decode top-phrase DF ≤ 15%, CE within 10% of
baseline.

### 7. (Optional) Patch SFT FVE magnitude pathology  *(5-line patch)*

Documented in
[`docs/sft_plan/v4_repair/sa_scale_audit.md`](sa_scale_audit.md). AR's
NCE term gives no magnitude pressure, so predicted `h` grows 5.3× the
target norm. Cosine 0.36 stays the same; FVE crashes to −23.

Pick ONE:

- **eval-time renorm:** in
  [`src/nla/training/sft.py`](../../src/nla/training/sft.py) eval loop,
  rescale `pred_unscaled` to `‖h_target‖` before computing FVE / MSE.
  Honest reporting; doesn't change training.
- **training-time magnitude term:** add a small
  `λ * (‖pred_scaled‖ - 1.0)²` term to the AR loss. Forces pred norm to
  match target norm during training.
- **lower NCE weight:** halve `ar_contrastive_weight` from 0.5 → 0.25 so
  MSE has more pull on magnitude.

Run V4 SFT once with the eval-time renorm first to see how much of the
FVE = −23 disaster was pure magnitude vs actual signal failure.

---

## Cost summary

| Item                                  | API $   | GPU h | Engineering  |
|---------------------------------------|--------:|------:|--------------|
| 1. Defuse plan template               |  ~$40   |     0 | half day     |
| 2. Re-extract `last_text` early layer |    $0   |   2–4 | half day     |
| 3. Quality weights                    |  ~$50   |     0 | half day     |
| 4. libero_10 suite rules              |  ~$10   |     0 | half day     |
| 5. `language:` contract decision      |  ~$10   |     0 | 1 hr         |
| 6. Anti-template loss                 |    $0   |   1–2 | 1 day        |
| 7. FVE magnitude patch                |    $0   |   0   | 30 min       |
| **Total V5**                          |  ~$110  |   3–6 | ~3 dev days  |

---

## Stopping rule for V5

V5 succeeds if **both** hold:

- Multimodal judge B ≥ 95% overall AND ≥ 90% on every suite (paper bar
  V4 missed by 4.8pp).
- SFT AR cosine ≥ 0.55 on held-out V5 rows (FVE gate from V3 scorecard).

If V5 fails the SFT gate but passes the judge gate, the bottleneck is
optimization / architecture, not data — graduate to a separate `architecture_v2`
plan rather than V6 dataset.

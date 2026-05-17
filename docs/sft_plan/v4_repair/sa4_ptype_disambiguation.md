# SA4 — image_patch ↔ last_text disambiguation + Jaccard delta metric

Status: green (30/30 tests in `tests/test_labeling_smoke.py`); V3 baseline
captured at `data/eval/sa4_v3_baseline_ptype_jaccard.json`.

## 1. What changed in `prompts.py`

Edits are scoped exclusively to (a) the `_LAST_BULLET_BY_POSITION_TYPE`
dict (SA1's hook) and (b) one new rule appended at the bottom of
`_V4_EXTRA_RULES`. No other prompt code was touched, no builder or
constant was renamed, SA2/SA3/SA5 surfaces are untouched.

### 1a. `_LAST_BULLET_BY_POSITION_TYPE` — diff summary

| ptype | V3 / SA1 baseline | SA4 refinement |
|---|---|---|
| `image_patch` | "MUST be a `target:` or `scene:` bullet describing what is visible in the attached camera frame at THIS exact moment. Do NOT restate the task instruction…" | (a) name a specific object/region currently visible; (b) **Do NOT restate the task instruction**, paraphrase, or quote its content words; (c) **no temporal/predictive phrasing** (`is about to`, `will then`, `next step`, `over the next … timesteps`) and **no plan-phase verbs as the main predicate** (`pickup`, `place`, `release`, …); (d) MUST use `visible in this frame: <object> <observable state>`. Canonical example: `- target: black wine bottle upright on the wooden tabletop next to the gripper.` |
| `last_text` | "MUST be a `plan:` bullet describing the next ~3 timesteps of motion as a specific phase…" | (a) name a phase from `V4_PLAN_PHASES`; (b) **explicit temporal connector** (one of `over the next 3 timesteps`, `over the next ~3 timesteps`, `before releasing`, `before placing`, `before retracting`, `until the gripper closes`); (c) **no pixel/region/patch vocabulary** (`upper-left`, `patch`, `region`, `visible in this frame`, …); (d) references the parsed instruction verbatim/near-verbatim. Canonical example: `- plan: pickup phase over the next 3 timesteps: gripper closes on the wine bottle, then lifts before placing on the rack.` |
| `anchor` | "MUST be a `plan:` bullet describing the overall trajectory phase…" | Clarified to describe the **OVERALL trajectory phase** **AND remaining steps** of the trajectory; explicitly NOT just the next single step (last_text's job) and NOT just the current frame (image_patch's job). Canonical example: `- plan: approach trajectory; arm staged above the table, remaining steps: reach over the bowl, close on the wine bottle, lift, and place on the rack.` |
| `fallback` | "either `plan:` (preferred) or `target:`…" | Same default, plus an explicit pointer: if `plan:` is chosen, follow the LAST-TEXT temporal-connector convention; if `target:` is chosen, follow the IMAGE-PATCH perceptual convention. |

The phrases `Do NOT restate the task instruction`, `next ~3 timesteps`,
and `overall trajectory phase` are preserved verbatim so SA1's existing
`test_v4_position_prompt_position_type_conditioning` keeps asserting on a
stable contract.

### 1b. V4-LEAK-1 — new rule in `_V4_EXTRA_RULES`

Appended after the existing `Plan-bullet diversity` block:

> **Rule V4-LEAK-1 — Position-type discipline (anti-cross-leak):**
> Do NOT write image_patch-style perceptual bullets (`visible in this
> frame: …`, `in this frame: …`, `<object> upright on the tabletop next
> to the gripper`) on last_text or anchor rows, and do NOT write
> last_text-style temporal-plan bullets (`over the next 3 timesteps: …`,
> `over the next ~3 timesteps: …`, `before releasing …`, `before placing
> …`) on image_patch rows. The position_type clause at the bottom of the
> user prompt specifies which style applies to this row; obey it
> strictly. The same task instruction and the same frame must produce
> DIFFERENT last bullets for image_patch vs last_text vs anchor.

This rule lives inside `_V4_POSITION_SYSTEM` (via `_V4_EXTRA_RULES`), so
it is shared by every ptype, not appended conditionally.

## 2. V3 baseline (the cells SA10 should beat)

From `data/eval/sa4_v3_baseline_ptype_jaccard.json`
(`scripts/eval/audit_ptype_disambiguation.py` over
`data/labels/libero_4suite_stride2/`, 101,580 rows; seed=0; n_pairs=2000).

### Top-30 unigram Jaccard (image_patch vs last_text), per suite

| suite | target | scene | spatial | plan | verdict |
|---|---|---|---|---|---|
| libero_10 | 0.76 ⚠️ | 0.76 ⚠️ | 0.76 ⚠️ | 0.67 ⚠️ | RED |
| libero_goal | 0.82 ⚠️ | 0.82 ⚠️ | 0.76 ⚠️ | 0.76 ⚠️ | RED |
| libero_object | 0.76 ⚠️ | 0.67 ⚠️ | 0.82 ⚠️ | 0.71 ⚠️ | RED |
| libero_spatial | 0.82 ⚠️ | 0.76 ⚠️ | 0.82 ⚠️ | 0.62 ⚠️ | RED |
| **overall** | **0.76** | **0.76** | **0.71** | **0.71** | **RED** |

These line up with Agent 4's V3 audit (top-30 Jaccard 0.71-0.76 for
`target/scene`); same tokenization conventions as
`scripts/eval/audit_bullet_informativeness.py`.

### Overall top-30 Jaccard mean: **0.740**

### Pairwise Jaccard (random image_patch × last_text rows per suite)

| suite | n_pairs | mean | p10 | p50 | p90 |
|---|---|---|---|---|---|
| libero_10 | 7,465 | 0.28 | 0.10 | 0.25 | 0.50 |
| libero_goal | 7,919 | 0.27 | 0.11 | 0.25 | 0.46 |
| libero_object | 7,842 | 0.28 | 0.11 | 0.25 | 0.50 |
| libero_spatial | 7,719 | 0.30 | 0.12 | 0.27 | 0.50 |
| **overall** | 7,729 | 0.28 | 0.11 | 0.26 | 0.50 |

(`n_pairs` per suite exceeds the 2000 target because the sampler emits
one Jaccard per matched bullet inside each pair; the score per pair
averages over up to 4 bullets.)

### Last-bullet mix per ptype (overall)

| ptype | target | scene | plan | spatial |
|---|---|---|---|---|
| image_patch | 6.2% | 0.0% | **93.8% ⚠️** | 0.0% |
| last_text | 0.0% | 0.0% | **100.0%** | 0.0% |
| anchor | 0.0% | 0.0% | 100.0% | 0.0% |

`image_patch.plan = 93.8%` is the V3 failure mode the V4 prompt
targets — the labeler emits `plan:` last bullets even for image_patch
rows, contradicting the position type. V4 wants `image_patch.plan ≤ 30%`
with the bulk going to `target:`/`scene:`.

### V3 token entropy (overall, bits)

| ptype | target | scene | spatial | plan |
|---|---|---|---|---|
| image_patch | 7.58 | 6.43 | 7.32 | 6.93 |
| last_text | 7.42 | 6.69 | 7.36 | 6.72 |

Entropy is similar across ptypes (deltas <0.3 bits) — confirming the
templated-caption diagnosis: the labeler isn't varying vocabulary by
ptype, it's just emitting the same caption with different bullet
headers.

## 3. V4 targets

| metric | V3 baseline | V4 target | V4 delta |
|---|---|---|---|
| top-30 Jaccard (target) | 0.76 | ≤ 0.45 | -0.31 |
| top-30 Jaccard (scene) | 0.76 | ≤ 0.45 | -0.31 |
| top-30 Jaccard (spatial) | 0.71 | ≤ 0.45 | -0.26 |
| top-30 Jaccard (plan) | 0.71 | ≤ 0.45 | -0.26 |
| top-30 Jaccard mean | 0.740 | ≤ 0.45 | **-0.29** |
| last_bullet_mix[image_patch].plan | 93.8% | ≤ 30% | -64 pp |
| last_bullet_mix[image_patch].(target ∪ scene) | 6.2% | ≥ 60% | +54 pp |
| last_bullet_mix[last_text].plan | 100.0% | ≥ 60% (already met) | n/a |
| pairwise Jaccard mean | 0.28 | ≤ 0.20 (soft) | -0.08 |

The headline number SA10 watches is the top-30 Jaccard mean: **0.740 →
≤ 0.45** (~0.30 absolute drop). If V4 lands the prompt clauses but
not the last-bullet mix, the script will still flag the run RED via
the `image_patch.plan > 30%` cell.

## 4. How SA10 calls the script (V4 regression gate)

Same CLI as the V3 baseline; just point it at the V4 labels root:

```bash
cd /home/ubuntu/nla-groot && PYTHONPATH=src .venv/bin/python \
    scripts/eval/audit_ptype_disambiguation.py \
    --labels-root data/labels/libero_4suite_stride2_v4 \
    --out-json data/eval/sa10_v4_ptype_jaccard.json \
    --out-md docs/sft_plan/v4_repair/sa10_v4_ptype_jaccard.md
```

Optional `--suite libero_spatial` restricts to one suite (handy for
mid-run spot checks while V4 labels are still streaming in).

Exit-code rules SA10 should enforce:

- **GREEN** verdict in `summary["overall"]["verdict"]` → V4 ships.
- **YELLOW** (1-2 cells) → escalate to a human reviewer; the cell that
  failed is named in `summary["overall"]["top30_jaccard"]` and
  `summary["overall"]["last_bullet_mix"]`.
- **RED** (≥3 cells) → block the V4 promotion; investigate the prompt
  via re-running on a 1% subsample to see whether the issue is
  per-suite-localised.

The script prints a 5-line stdout summary suitable for direct
copy-pasting into a parent-agent report:

```
=== ptype disambiguation summary ===
verdict=… violations=… rows=…
top30_jaccard target=…, scene=…, spatial=…, plan=…
top30_jaccard_mean=…
mean_pairwise=… (p10=…, p50=…, p90=…)
last_bullet_mix.image_patch.plan=…% (target <=30%)
last_bullet_mix.last_text.plan=…% (target >=60%)
```

## 5. Tests added

`tests/test_labeling_smoke.py`:

- `test_v4_last_bullet_image_patch_vs_last_text_differs` — builds prompts
  for `image_patch` and `last_text` from the same instruction/frame, then
  asserts (a) user prompts differ, (b) system prompts differ, (c) the
  image_patch clause contains the perceptual phrasing + canonical
  wine-bottle example, and (d) the last_text clause contains the
  temporal connector + canonical `pickup phase over the next 3
  timesteps` example and does NOT itself contain the word
  "perceptual".
- `test_v4_leak_rule_present` — asserts the `Rule V4-LEAK-1 — Position-
  type discipline` block (and its specific anti-cross-leak phrasings)
  are baked into the V4 system prompt regardless of ptype.

Run: `cd /home/ubuntu/nla-groot && PYTHONPATH=src .venv/bin/python -m
pytest tests/test_labeling_smoke.py -x -q` → **30 passed**.

## 6. Coordination notes

- SA1's `_LAST_BULLET_BY_POSITION_TYPE` contract is preserved
  bit-for-bit — `Do NOT restate the task instruction`, `next ~3
  timesteps`, and `overall trajectory phase` still appear verbatim so
  SA1's `test_v4_position_prompt_position_type_conditioning` keeps
  passing.
- SA2 owns `_V4_SUITE_ADDENDA["libero_spatial"]` and
  `PositionLabelInput.suite`; this work doesn't touch either. The
  Jaccard script runs orthogonally on whatever corpus SA2's addendum
  produces.
- SA3 owns `scripts/eval/audit_prompt_hardening.py` and the forbidden-
  phrase tuples; the new Jaccard script does **not** import them — it
  is intentionally tokenization-only so SA3's regex churn cannot
  destabilise this regression gate.
- SA5 wires `NLA_POSITION_PROMPT_MODE=v4` through the pipeline; once
  V4 labels exist, SA10 calls the script as in §4.

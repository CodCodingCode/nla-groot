# SA3 — Motor-imperative / scaffold-leakage / non-canonical-header audit

Extends `scripts/eval/audit_prompt_hardening.py` (originally Agent 2 of
the V3 audit) with three V4-regression failure modes. SA10 will run the
extended audit against V4 and regression-gate it against the frozen V3
baseline at `data/eval/sa3_v3_baseline_summary.json`.

## Status: green

- 4/4 new tests pass (`tests/test_audit_prompt_hardening.py`).
- Full test sweep `tests/test_labeling_smoke.py +
  tests/test_audit_prompt_hardening.py` → 34 passed.
- V3 baseline summary frozen at `data/eval/sa3_v3_baseline_summary.json`.

## Diff summary

### `scripts/eval/audit_prompt_hardening.py`

- New imports: `argparse`, `sys`, and (after a `sys.path` insert so the
  script's `nla.labeling.prompts` import resolves without
  `PYTHONPATH=src`) the canonical V4 phrase tuples:

  ```python
  from nla.labeling.prompts import (
      V4_BULLET_CATEGORIES,
      V4_FORBIDDEN_HEADERS,
      V4_MOTOR_IMPERATIVE_PHRASES,
      V4_SCAFFOLD_FORBIDDEN_PHRASES,
  )
  ```

  This is the single source of truth — `test_audit_uses_canonical_constants`
  pins `is`-identity so the audit cannot silently drift from the V4
  prompt ban list.

- New regexes (case-insensitive word-boundary alternation built with
  `_phrase_regex`, plus a line-anchored header detector):
  - `V4_MOTOR_IMPERATIVE_RE` (built from the 13 imperative phrases)
  - `V4_SCAFFOLD_LEAKAGE_RE` (built from the 9 scaffold phrases)
  - `V4_NONCANON_HEADER_RE` (matches `^\s*-?\s*(gripper|motion|image_region)\s*:`)

- New `SuiteStats` fields:
  - Row-level: `n_v4_motor`, `n_v4_scaffold`, `n_v4_noncanon_header`
  - Per-bullet-type breakdown: `bt_total`, `bt_motor_hits`,
    `bt_scaffold_hits`
  - Per-forbidden-header breakdown: `noncanon_header_hits`

- New scan logic in `_scan_row` runs in the same single CPU pass:
  - Row-level motor / scaffold detection on the entire description
  - Per-bullet motor / scaffold detection on bullet bodies
  - Line-level forbidden-header detection

- New markdown sections in the report:
  - "Motor imperatives (V4 regression mode)" — per-suite table + per
    bullet-type table
  - "Scaffold leakage (V4 regression mode)" — per-suite table + per
    bullet-type table
  - "Non-canonical bullet headers (V4 regression mode)" — per-suite
    table + per-header breakdown

- New JSON keys (under `v4_failure_modes`):

  ```json
  {
    "motor_imperative_pct": {
      "goal": ..., "spatial": ..., "object": ..., "10": ..., "overall": ...,
      "by_bullet_type": {"plan": ..., "target": ..., ...}
    },
    "scaffold_leakage_pct": { same shape },
    "noncanonical_header_pct": {
      "goal": ..., "spatial": ..., "object": ..., "10": ..., "overall": ...,
      "header_breakdown_overall": {"gripper": N, "motion": N, "image_region": N}
    }
  }
  ```

  The original `v3_overall.rates_pct` block also gains three flat keys
  (`v4_motor_imperative`, `v4_scaffold_leakage`, `v4_noncanonical_header`)
  so any caller already keying off `rates_pct` picks them up without
  reshaping the schema.

- New verdict logic (extends the V3 rubric without replacing it):
  - V4 motor-imperative > 2% in any suite → RED for that suite
  - V4 scaffold-leakage > 5% → YELLOW; > 15% → RED
  - V4 non-canonical headers > 0.5% → YELLOW; > 2% → RED

- CLI now takes `--labels-root`, `--out-json`, `--out-md`, plus
  `--skip-baselines` for V4 regression runs. Defaults preserve the
  V3-corpus path; the JSON summary defaults to a *new* path
  (`data/eval/audit_prompt_hardening_summary.json`) so the frozen
  Agent-2 V3 baseline at `data/eval/agent2_summary.json` is never
  overwritten.

### `tests/test_audit_prompt_hardening.py` (new)

Four tests, all green:

1. `test_audit_detects_motor_imperative` — `- plan: grasp the bowl`
   trips both the regex and the row-level counter; lands in
   `bt_motor_hits["plan"]`.
2. `test_audit_detects_scaffold_leakage` — `- plan: action head selects
   the next motion` trips both the regex and the row-level counter;
   lands in `bt_scaffold_hits["plan"]`. Verifies motor counter does
   not also fire on this row.
3. `test_audit_detects_noncanonical_header` — a row containing a `-
   gripper:` bullet trips the non-canonical-header detector and lands
   in `noncanon_header_hits["gripper"]`.
4. `test_audit_uses_canonical_constants` — the audit's
   `V4_MOTOR_IMPERATIVE_PHRASES` / `V4_SCAFFOLD_FORBIDDEN_PHRASES` /
   `V4_FORBIDDEN_HEADERS` are `is`-identical to the ones exported from
   `nla.labeling.prompts`. Each individual phrase is also exercised
   against its own regex to catch escaping mistakes.

## V3 baseline numbers (frozen reference for SA10)

Run on `data/labels/libero_4suite_stride2/` (n=101,580 across the four
suites).

### Motor imperatives

| Suite | rows hit | % rows |
|---|---|---|
| libero_goal | 13,203 / 25,680 | **51.41%** |
| libero_spatial | 20,108 / 25,920 | **77.58%** |
| libero_object | 22,332 / 27,240 | **81.98%** |
| libero_10 | 7,714 / 22,740 | **33.92%** |
| **overall** | 63,357 / 101,580 | **62.37%** |

Per-bullet-type concentration on V3-overall (top entries):

| Bullet type | n bullets | motor hits | rate |
|---|---|---|---|
| `plan` | 101,415 | 59,687 | **58.85%** |
| `language` | 20,516 | 8,912 | 43.44% |
| `motion` (V4-forbidden) | 4,526 | 1,753 | 38.73% |
| `gripper` (V4-forbidden) | 5,769 | 1,696 | 29.40% |
| `spatial` | 99,441 | 1,157 | 1.16% |
| `target` | 113,134 | 310 | 0.27% |
| `scene` | 100,378 | 5 | 0.005% |
| `distractor` | 62,654 | 0 | 0.000% |

Plan-bullet concentration matches what Agent 4 found: imperatives are
overwhelmingly in `plan:`. The 43% rate on `language:` is the
labeler quoting the raw instruction (`language: the user says
"grasp the bowl"`); V4 should paraphrase this.

### Scaffold leakage

| Suite | rows hit | % rows |
|---|---|---|
| libero_goal | 7,676 / 25,680 | **29.89%** |
| libero_spatial | 7,718 / 25,920 | **29.78%** |
| libero_object | 7,129 / 27,240 | **26.17%** |
| libero_10 | 7,718 / 22,740 | **33.94%** |
| **overall** | 30,241 / 101,580 | **29.77%** |

Per-bullet-type:

| Bullet type | n bullets | scaffold hits | rate |
|---|---|---|---|
| `plan` | 101,415 | 27,989 | **27.60%** |
| `target` | 113,134 | 1,800 | 1.59% |
| `motion` | 4,526 | 146 | 3.23% |
| `language` | 20,516 | 244 | 1.19% |
| `image_region` | 58 | 40 | 68.97% |
| `scene` | 100,378 | 0 | 0.000% |

Agent 3 estimated 11-17% on `plan` bullets; the canonical
V4 phrase tuple (which adds `transformer` / `embedding` /
`hidden state` / `residual stream`) raises that to **27.6%**. The
delta is the model's tendency to describe activations directly
("the embedding shifts toward …", "the hidden state reflects …").

### Non-canonical headers

| Suite | rows hit | % rows |
|---|---|---|
| libero_goal | 2,746 / 25,680 | **10.69%** |
| libero_spatial | 3,071 / 25,920 | **11.85%** |
| libero_object | 997 / 27,240 | **3.66%** |
| libero_10 | 3,538 / 22,740 | **15.56%** |
| **overall** | 10,352 / 101,580 | **10.19%** |

Header breakdown (V3-overall):

| Forbidden header | # rows hit |
|---|---|
| `gripper:` | 5,769 |
| `motion:` | 4,526 |
| `image_region:` | 57 |

Matches Agent 2's first-pass numbers exactly (5,769 / 4,526). V4 folds
all three into `plan` / `target` / `spatial` via the
`V4_FORBIDDEN_HEADERS` constant.

### V3 verdict

**RED** — every suite and the overall aggregate trips at least one V4
regression-mode RED bar. Concretely: V4 must drive all three modes
down by 1-2 orders of magnitude before SA10 can flip the verdict to
GREEN.

## Expected V4 targets (post-SA6 re-label)

| Failure mode | V3 overall | V4 target (GREEN) | V4 ceiling (YELLOW) |
|---|---|---|---|
| motor-imperative | 62.37% | **< 0.5%** | < 2% |
| scaffold-leakage | 29.77% | **< 1%** | < 5% |
| non-canonical-header | 10.19% | **< 0.5%** | < 2% |

GREEN targets correspond to the rubric in the SA3 spec; YELLOW
ceilings correspond to the RED-trigger thresholds in the audit's
`compute_verdict`. Anything above the YELLOW ceiling is an automatic
SA10 regression failure.

## How SA10 regression-gates V4 against V3

Two passes — one to refresh the V3 baseline (idempotent; should match
`sa3_v3_baseline_summary.json` exactly), one to score V4:

```bash
cd /home/ubuntu/nla-groot

# 1. V3 baseline (frozen; run if sanity-checking).
PYTHONPATH=src .venv/bin/python scripts/eval/audit_prompt_hardening.py \
    --labels-root data/labels/libero_4suite_stride2 \
    --out-json data/eval/sa3_v3_baseline_summary.json \
    --out-md   /tmp/sa3_v3_baseline.md

# 2. V4 audit (path filled in by SA5/SA6).
PYTHONPATH=src .venv/bin/python scripts/eval/audit_prompt_hardening.py \
    --labels-root data/labels/libero_4suite_v4 \
    --out-json data/eval/sa10_v4_audit_summary.json \
    --out-md   docs/sft_plan/v4_repair/sa10_v4_audit.md \
    --skip-baselines

# 3. Regression-gate comparison (SA10 owns this script).
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
v3 = json.load(open("data/eval/sa3_v3_baseline_summary.json"))["v4_failure_modes"]
v4 = json.load(open("data/eval/sa10_v4_audit_summary.json"))["v4_failure_modes"]
for mode, target, ceiling in [
    ("motor_imperative_pct", 0.5, 2.0),
    ("scaffold_leakage_pct", 1.0, 5.0),
    ("noncanonical_header_pct", 0.5, 2.0),
]:
    v3_overall = v3[mode]["overall"]
    v4_overall = v4[mode]["overall"]
    delta = v3_overall - v4_overall
    status = "GREEN" if v4_overall < target else ("YELLOW" if v4_overall < ceiling else "RED")
    print(f"{mode:30s} V3={v3_overall:6.3f}%  V4={v4_overall:6.3f}%  Δ={delta:+6.3f}%  {status}")
PY
```

The audit's own `verdict` field already encodes this gate
(`compute_verdict` walks per-suite stats), but SA10 should also do
the explicit V3-vs-V4 delta check above because a regression that
moves V4 from 0.05% to 0.40% (still below target) is something to
flag in the SA10 report even though both verdicts read GREEN.

## Per-suite numbers in one table

For completeness, here is everything SA10 needs in one place:

| Mode | goal | spatial | object | 10 | overall |
|---|---|---|---|---|---|
| motor-imperative % | 51.41 | 77.58 | 81.98 | 33.92 | **62.37** |
| scaffold-leakage % | 29.89 | 29.78 | 26.17 | 33.94 | **29.77** |
| non-canonical-header % | 10.69 | 11.85 | 3.66 | 15.56 | **10.19** |

## Return values for the parent agent

- (a) V3 motor-imperative % overall: **62.37%**
- (b) V3 scaffold-leakage % overall: **29.77%**
- (c) Test pass count: **4 / 4** new tests in
  `tests/test_audit_prompt_hardening.py` (plus 30 pre-existing
  labeling-smoke tests still green; 34 total).

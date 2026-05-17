# SA6 — V4 selective re-label orchestration

**Status: BUDGET PAUSE — awaiting parent agent approval before kicking off
the production run.** Queue builder + driver are written, dry-run is clean,
10-row smoke run is green. The hard-budget gate fires because the V3
failure heuristics catch more rows than the plan author projected.

## Headline numbers

| Suite | policy | V3 rows | queued | % flagged | est. cost |
|---|---|---|---|---|---|
| `libero_spatial` | full | 25,920 | **25,845** | 99.7% | $18.09 |
| `libero_goal`    | selective | 25,680 | 17,888 | 69.7% | $12.52 |
| `libero_object`  | selective | 27,240 | 23,943 | 87.9% | $16.76 |
| `libero_10`      | selective | 22,740 | 14,329 | 63.0% | $10.03 |
| **TOTAL**        | | **101,580** | **82,005** | 80.7% | **$57.40** |

(see `data/labels/v4_relabel_queue/_summary.json` for the full JSON.)

The plan estimated `~$30-40` total; the actual selective queue is `$57.40`
because the `motor_imperative` heuristic alone catches 33-82% of V3 rows
per suite (already known from SA3's V3 baseline: motor-imperative overall
was 62.37%). Almost every V3 `plan:` bullet contains some imperative
phrase from `V4_MOTOR_IMPERATIVE_PHRASES` ("grasp the", "reach toward",
"approach the", "place it"…); when the heuristic is OR'd against
scaffold + forbidden-header it sweeps up most of the corpus.

## Hit-rate breakdown (heuristics overlap; sums > queued)

| Suite | motor | scaffold | forbidden hdr | < 3 bullets | error |
|---|---:|---:|---:|---:|---:|
| `libero_spatial` (full) | 20,045 | 7,694 | 3,059 | 0 | 0 |
| `libero_goal` | 13,162 | 7,651 | 2,730 | 0 | 0 |
| `libero_object` | 22,264 | 7,111 | 991 | 0 | 0 |
| `libero_10` | 7,699 | 7,691 | 3,527 | 0 | 0 |

`error` and `<3 bullets` are both zero — V3 labeling itself was clean,
the issue is purely the V4 phrasing/header bans against V3 prose.

## Why the heuristic is "over-eager"

`motor_imperative` and `scaffold_leakage` are V4 *prompt-effectiveness*
levers, not V3 *labeling-quality* failures:

* **Motor imperatives** (`grasp the`, `place it`, `reach over`…) — V4
  switches the prompt to forbid second-person robot addressing, but V3
  imperatives are *descriptive*, just stylistically wrong-axis. The
  caption "plan: grasp the wine bottle" is informative; V4 wants
  "plan: pickup; gripper closing on the wine bottle".
* **Scaffold leakage** (`embedding`, `transformer`, `action head`,
  `hidden state`) — V4 forbids these as scaffolding echoes; in V3 they
  appear in 26-34% of rows.
* **Forbidden headers** (`gripper:`, `motion:`, `image_region:`) — V4
  drops these categories; V3 emitted them in 4-16% of rows.

The grounding axis (object identity, spatial relations vs the actual
frame) is the V3 quality failure that motivated the spatial pilot
re-label; that one is not in the heuristic for the selective suites
because SA1's V3 audit reported `libero_goal/object/10` had healthy
B-pass (>=91%). For those three suites V4's job is style/phrasing
hardening, not grounding repair.

## Smoke-test result (driver works)

```
queue: 10 rows -> 10 PositionLabelInputs -> 10 V4 labels in 14s, errors=0
forbidden_hdr=0 motor=0 scaffold=0 across the 10 smoke rows
```

(Out at `/tmp/sa6_smoke_v4_goal/labels.jsonl`; full row 1 written below.)

The driver picks up:

* `NLA_POSITION_PROMPT_MODE=v4` set before importing `openai_client` →
  `_select_position_builder` dispatches to `build_v4_position_prompt`
  (SA5's wiring).
* `--suite libero_goal` stamped onto every `PositionLabelInput.suite`
  → `build_v4_position_prompt` would activate the
  `_V4_SUITE_ADDENDA["libero_goal"]` if one were registered (currently
  `libero_goal/object/10` are no-ops; only `libero_spatial` has SP-1..7).
* Frames cache reuses `data/labels/libero_4suite_stride2/<suite>/frames_cache/`
  (~25k pre-extracted JPEGs per suite); no re-extraction.
* Resume key matches on `(source_example_id, position_index, position_type)`
  exactly per `openai_client._position_resume_key_from_row`.
* Cost log row written to `_cost_log.jsonl` per completion.

Sample V4 output (image_patch row from `libero_goal`):

```
- scene: beige tabletop with a robot arm, a white square base, a black cabinet, and two camera views of the same workspace.
- target: gray speckled bowl visible in this frame near the front edge of the table.
- distractor: white plate with red concentric rings lies on the tabletop below the bowl.
- spatial: bowl and plate are separated, with the bowl closer to the gripper side and the plate below it in the overhead view.
- target: visible in this frame: gray bowl upright on the tabletop, with the plate nearby as the placement surface.
```

(All 10 smoke rows are forbidden-phrase clean. One stylistic nit: 3 of
10 rows ended with a duplicate `target:` last bullet instead of a
distinct ptype-specific bullet — that's per the SA4 last-bullet contract
on `image_patch` ptype, but hits the "duplicate header" pattern. Not a
blocker; flagged for SA9/SA10's broader audit.)

## Decision required from parent agent — three options

### Option A — proceed at $57.40 (14.8% over $40 default, 14.8% over the $50 hard cap)

Run the full queue as-is. Maximum corpus coverage; everything that fails
ANY V4 phrasing rule is rewritten. Total spend `$57.40` plus the SA2
pilot's already-paid `~$0.05`.

```bash
for SUITE in spatial goal object 10; do
    PYTHONPATH=src .venv/bin/python scripts/labeling/run_v4_relabel.py \
        --queue-jsonl data/labels/v4_relabel_queue/libero_${SUITE}.jsonl \
        --activations-root data/activations/libero_4suite_stride2/libero_${SUITE} \
        --dataset-root third_party/Isaac-GR00T/examples/LIBERO/libero_${SUITE}_no_noops_1.0.0_lerobot \
        --suite libero_${SUITE} \
        --out-dir data/labels/libero_4suite_v4/libero_${SUITE} \
        --concurrency 32 \
        2>&1 | tee data/labels/libero_4suite_v4/relabel_${SUITE}.log
done
```

### Option B (recommended) — drop motor-imperative-only matches, keep scaffold + forbidden-header + full-spatial → ~$37.50

Hardest-and-cheapest scope: keep `libero_spatial` full re-label
(grounding-driven; SA2 pilot already proved this is needed); for the
other three suites only re-label rows whose V3 description trips
`scaffold_leakage` OR `forbidden_header`. This drops the motor-only
contingent (which is V4's stylistic ban, not a grounding problem) but
catches the V4-incompatible header structure and the prompt-scaffolding
echoes.

| Suite | rows under tighter heuristic | est. cost |
|---|---:|---:|
| `libero_spatial` (unchanged, full) | 25,845 | $18.09 |
| `libero_goal` (scaffold OR hdr) | 9,693 | $6.79 |
| `libero_object` (scaffold OR hdr) | 7,921 | $5.55 |
| `libero_10` (scaffold OR hdr) | 10,108 | $7.08 |
| **TOTAL** | **53,567** | **$37.50** |

(numbers cross-checked from the V3 corpus directly; would re-run the
queue builder with a `--no-motor` flag I'd add.)

Risk: V4 vs V3 motor-imperative drop on `libero_goal/object/10` would be
smaller than under Option A. SA3's V4 GREEN target is motor-imperative
< 0.5% overall; under Option B those three suites would inherit V3's
~50-80% motor-imperative rate on the rows we kept un-relabeled, blended
across the full corpus (because SA7 will combine V3-kept + V4-rewritten).
Rough math: blended motor-imperative% ≈ V3% × (kept fraction) + V4% ×
(re-labeled fraction). For `libero_object` that's ~82% × 71% + ~5% ×
29% ≈ 60% motor-imperative remaining → **fails SA3's < 0.5% green
target**. SA10's V3-vs-V4 audit would still see a real drop, but the
verdict gate would land RED on motor-imperative.

So Option B keeps cost in budget but trades the SA3 motor-imperative
gate.

### Option C — re-scope inside the budget by capping per-suite

Keep all three V3-failure heuristics, but cap `libero_object` (the
$16.76 selective bucket) and `libero_10` to whatever fits the $40
ceiling once `libero_spatial` ($18.09) and `libero_goal` ($12.52) are
in. Under a $40 budget that leaves $9.39 for `libero_object + libero_10`
combined ≈ 13,400 rows split between them. Risk: arbitrary downsample
of V3-flagged rows; biased toward early example_ids unless we add
sampling logic; SA10 audit gate would land somewhere between A and B
depending on stratification.

## Recommendation

**Option A** if the additional `$17` over the default budget is
acceptable. The plan's $30-40 estimate was a projection error (the
heuristic's actual hit rate is 1.5-2× higher than the plan author
expected); the V4 audit gates SA10 enforces (motor-imperative < 0.5%,
scaffold-leakage < 1%, non-canonical header < 0.5%) effectively require
re-labeling every flagged row, otherwise the blended V3+V4 corpus will
fail those gates by tens of percentage points. Option B saves $20 but
ships a corpus that flunks the motor-imperative gate.

**If parent says "stay at $40 max":** Option B with my caveat. SA8's
hard-neg miner uses captions for retrieval contrastive — partial
re-label still improves spatial; the regression on motor-imperative is
the lever to highlight to SA9/SA10 in the eval report.

**If parent says "go":** I will execute Option A as-is. Sequential per
suite, ~25-40 min per suite at concurrency=32 → ~2-3h total. Cost log
at `data/labels/libero_4suite_v4/libero_<suite>/_cost_log.jsonl`; will
abort and re-report if cumulative spend exceeds the approved cap.

## Deliverables shipped this turn (regardless of option)

* `scripts/labeling/build_v4_relabel_queue.py` — queue builder, V4
  failure-heuristic-based; CLI `--v3-labels-root`, `--out-dir`,
  `--max-per-suite`, `--suites`. Dry-run logs per-suite reason
  histograms and prints the total-cost banner. Imports the canonical
  V4 phrase tuples (`V4_FORBIDDEN_HEADERS`,
  `V4_MOTOR_IMPERATIVE_PHRASES`, `V4_SCAFFOLD_FORBIDDEN_PHRASES`)
  directly from `nla.labeling.prompts` so the heuristic cannot drift
  from the V4 prompt's ban list.
* `scripts/labeling/run_v4_relabel.py` — driver. Reads the queue;
  sets `NLA_POSITION_PROMPT_MODE=v4` before importing
  `nla.labeling.openai_client` (SA5 dispatch); builds
  `PositionLabelInput`s only for queued rows via
  `ActivationShardReader.iter_examples(record_filter=...)` so we don't
  pay frame-extraction on un-queued examples; resume by canonical
  position key; one cost-log row per completion to
  `<out-dir>/_cost_log.jsonl` with cumulative-cost rolling sum.
* `data/labels/v4_relabel_queue/{libero_spatial,libero_goal,libero_object,libero_10}.jsonl`
  — 82,005 queued rows total.
* `data/labels/v4_relabel_queue/_summary.json` — per-suite counts,
  reason histograms, cost estimate.
* `/tmp/sa6_smoke_v4_goal/{labels,_cost_log,manifest}.jsonl` — 10-row
  smoke output (will not be promoted to `data/labels/libero_4suite_v4/`;
  `out-dir=/tmp/...` so the production runs land in clean directories).

## Coordination

* SA7 / SA8 / SA9 / SA10 are blocked on the production run output. I
  will not start the production run until the parent picks an option.
* SA8 (hard-neg miner refactor) is independent and can land in parallel
  with this pause; it consumes V4 labels post-merge by SA7.

## Return values for the parent (current)

* (a) **Total cost (queue estimate):** `$57.40` (vs `$40` default,
  `$50` hard cap).
* (b) **Per-suite re-labeled row counts (planned, Option A):**
  spatial 25,845, goal 17,888, object 23,943, 10 14,329; total 82,005.
* (c) **Per-suite error/leak counts:** smoke run is 10/10 clean
  (motor=0, scaffold=0, forbidden_hdr=0); production-run leak counts
  available only after the production run.
* (d) **Status:** **NOT ready for SA7** — production run is paused
  pending budget decision.

# SA5 — V4 prompt pipeline wiring

Status: green. 29 / 30 tests passing in `tests/test_labeling_smoke.py`
(see "Test status" below — the one failure is a pre-existing `prompts.py`
issue in SA4's territory and unrelated to my changes).

5 V4 labels written end-to-end with a real OpenAI call to
`/tmp/sa5_smoke_v4/labels.jsonl`. V3 production behavior is untouched
(default `--prompt-mode v3`, default env var unset → `v3`).

## Files touched

| file | summary |
|---|---|
| `src/nla/labeling/openai_client.py` | New `_select_position_builder(mode)` dispatcher; new `_call_position_builder(builder, inp)` helper that introspects the builder's signature with `inspect.signature` and threads `inp.suite` only when the builder accepts it. `_build_messages` now goes through both. V3 / strict callers see no change. |
| `src/nla/labeling/pipeline.py` | `run_labeling(...)` and `run_labeling_sync(...)` now accept `suite: str \| None = None`. When `None`, infer from `dataset_root` via `_infer_suite_from_dataset_root` (matches `libero_(spatial\|object\|goal\|10)` against the lowercased path string; first match wins). Suite is threaded into `build_position_inputs` and recorded on the manifest's `extra` block alongside the active `prompt_mode`. |
| `src/nla/labeling/context.py` | `build_position_inputs(...)` accepts `suite=None`; when set it stamps both `extra["suite"]` and the new `PositionLabelInput.suite` field on every yielded input. |
| `scripts/labeling/run_label.py` | New `--prompt-mode {v3,v3_strict,v4}` CLI flag; when set it overrides `NLA_POSITION_PROMPT_MODE` for the run (set *before* the pipeline import). New `--suite` flag plumbed through to `run_labeling_sync`. Docstring updated with a V4 example invocation. |
| `tests/test_labeling_smoke.py` | New `test_pipeline_dispatches_v4_when_mode_set` (pure unit test for `_select_position_builder` covering `v4`, `V4_position`, `v4-position`, `strict`, `v3_strict`, `v3`, unknown-mode, and env-var dispatch). New `test_pipeline_threads_suite_through_to_v4_builder` (asserts `_build_messages` returns a system prompt containing `LIBERO-SPATIAL addendum` + `Scaffold-leakage ban` when `inp.suite="libero_spatial"` and `NLA_POSITION_PROMPT_MODE=v4`). |

No edits to `prompts.py`, `prompt_variants.py`, audit scripts, or the
labeling I/O / resume logic.

## CLI surface

V3 production (back-compat — unchanged):

```bash
PYTHONPATH=src python scripts/labeling/run_label.py \
    --activations-root data/activations/libero_4suite_stride2/libero_goal \
    --dataset-root     third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \
    --labels-dir       data/labels/libero_goal_v3 \
    --concurrency 16
```

V4 single-suite re-label (the SA6 production command — see "For SA6" below):

```bash
NLA_POSITION_PROMPT_MODE=v4 PYTHONPATH=src python scripts/labeling/run_label.py \
    --activations-root data/activations/libero_4suite_stride2/libero_goal \
    --dataset-root     third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \
    --labels-dir       data/labels/libero_goal_v4 \
    --prompt-mode v4 --suite libero_goal \
    --positions-per-example 1 --concurrency 16
```

`--prompt-mode` and the env var are interchangeable; the CLI flag
sets the env var before pipeline import. `--suite` is optional; when
omitted, the pipeline auto-infers from the dataset path
(`libero_goal_no_noops_1.0.0_lerobot` → `libero_goal`), as confirmed
in the smoke run logs:

```
INFO [nla.labeling.pipeline] Inferred suite=libero_goal from dataset_root
```

## Smoke run output

Command (exact, as run):

```bash
cd /home/ubuntu/nla-groot && set -a && source .env && set +a && \
NLA_POSITION_PROMPT_MODE=v4 PYTHONPATH=src .venv/bin/python scripts/labeling/run_label.py \
    --activations-root data/activations/libero_4suite_stride2/libero_goal \
    --dataset-root third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \
    --labels-dir /tmp/sa5_smoke_v4 \
    --max-examples 5 --concurrency 4 --positions-per-example 1
```

Pipeline log (tail):

```
Inferred suite=libero_goal from dataset_root
12840 examples
Sampling 1 position(s) per example (seed=0, guarantee_strata=False)
5 sampled positions across 5 examples
5 inputs ready (with frames)
Labeling: 5 new, 0 example_ids done, 0 position keys done -> /tmp/sa5_smoke_v4/labels.jsonl
Labeling done. New: 5. Total in /tmp/sa5_smoke_v4/labels.jsonl: 5.
```

5 V4 labels (full text, from `/tmp/sa5_smoke_v4/labels.jsonl`):

### Row 1 — `traj000000_step000002@p142_last_text` (ptype=last_text)

```
- scene: tabletop workspace with a white plate, a metal bowl, a black bottle, and nearby robot arm.
- target: bowl centered near the plate, with the plate directly below it in the task area.
- spatial: bowl and plate are aligned for a transfer; bottle, blue rectangular item, and dark blocks remain clear distractors.
- language: "put the bowl on the plate" binds the bowl as the object and the plate as the destination.
- plan: place phase over the next ~3 timesteps: bowl moves onto the plate, then gripper opens before releasing the bowl onto the plate.
```

### Row 2 — `traj000000_step000006@p142_anchor` (ptype=anchor)

```
- scene: tabletop workspace with a robot arm, a gray bowl, a white plate with red rings, a black bottle, and a blue rectangular item.
- target: gray bowl is the object to transfer onto the plate.
- spatial: bowl sits below the plate in the lower camera view, with the plate directly above it and the bottle nearby.
- distractor: black bottle and blue rectangular item are not the transfer target.
- plan: approach trajectory; remaining steps: stage over the bowl, lift it, carry it to the plate, and place it centered on the plate.
```

### Row 3 — `traj000000_step000000@p071_image_patch` (ptype=image_patch)

```
- scene: tabletop workspace with a white plate, a gray bowl, a black bottle, a blue rectangular object, and a large black container.
- target: gray bowl with a speckled interior sitting below the plate.
- distractor: white plate with red rings and the black bottle are separate from the bowl.
- spatial: bowl is centered under the plate and left of the black container in the top view.
- plan: approach; bowl and plate are both visible, with the bowl positioned for transfer onto the plate.
```

### Row 4 — `traj000000_step000004@p142_last_text` (ptype=last_text)

```
- scene: tabletop with a gray robot arm above a light wood surface, a white square burner-like base, a red-rimmed plate, and a metal bowl.
- target: the bowl is the object paired with the plate for the requested placement.
- distractor: a black bottle, a blue rectangular item, a black-handled tool, and cabinet edges remain on the table.
- spatial: the plate sits below and slightly right of the bowl in one view, with the bowl centered between the plate and the black bottle.
- plan: place phase over the next ~3 timesteps: bowl aligned with the plate, then lowered for placement on the plate.
```

### Row 5 — `traj000000_step000008@p142_anchor` (ptype=anchor)

```
- scene: tabletop with a gray bowl, a red-rimmed plate, a black bottle-shaped object, and the robot arm above the work surface.
- target: gray bowl centered near the plate, with the plate just above it in the overhead view.
- distractor: black bottle and small blue rectangular object sit beside the bowl and plate, with a dark cabinet edge and metal handle on the right.
- spatial: bowl and plate are aligned closely, with the bowl positioned directly below the plate in the overhead frame.
- plan: approach trajectory; remaining steps: move over the bowl, lift it, carry it to the plate, and place it centered on the plate.
```

## Forbidden-phrase audit (5/5 smoke labels)

| row | ptype | bullets | scaffold leaks | motor imperatives | forbidden header |
|---|---|---|---|---|---|
| 1 | last_text  | 5 | — | — | — |
| 2 | anchor     | 5 | — | **`carry it`, `place it`** | — |
| 3 | image_patch| 5 | — | — | — |
| 4 | last_text  | 5 | — | — | — |
| 5 | anchor     | 5 | — | **`carry it`, `place it`** | — |

Bullet headers (V4 categories only): `scene`, `target`, `spatial`,
`language`, `plan`, `distractor`. No `gripper:`, `motion:`, or
`image_region:` headers — the V4 builder's header restriction holds.
Scaffold leakage is fully clean across all 5 rows ("action head",
"transformer", "embedding", "patch carries", etc. are absent).

### Leaks observed (for SA6 / SA2 awareness)

Both `anchor` rows ended their `plan:` bullet with the second-person
imperative pair **"carry it to the plate, and place it centered on the
plate"**. `"carry it"` and `"place it"` are both listed in
`V4_MOTOR_IMPERATIVE_PHRASES`, so the V4 system text *does* call them
out — the model is regressing on long `plan:` bullets that itemize the
remaining trajectory in second person. Two-of-five is a 40% leak rate
on `anchor` (one ptype, n=2), 0% on `last_text` and `image_patch`.

This is **not** a pipeline-wiring issue (the V4 system prompt is being
delivered correctly — see the `Scaffold-leakage ban` test pass and the
clean scaffold column above). It is a prompt-effectiveness issue and
falls under SA2 / SA3 / SA4 scope. Per the task spec I am flagging it
but not patching the prompts.

Recommended follow-up for whichever SA owns plan-bullet phrasing
hardening: tighten the `_LAST_BULLET_BY_POSITION_TYPE["anchor"]` and
`_V4_MOTOR_IMPERATIVE_PHRASES` interaction so that the
"itemize remaining trajectory steps" pattern in `anchor` plan bullets
must be third-person ("the arm carries the bowl... the gripper places
the bowl..."). Re-running this exact 5-row smoke after the prompt fix
is the cheapest way to verify.

## Test status

```bash
cd /home/ubuntu/nla-groot && PYTHONPATH=src .venv/bin/python -m pytest tests/test_labeling_smoke.py -q
# 1 failed, 29 passed in 15.13s
```

- **29 passing**, including both new SA5 tests
  (`test_pipeline_dispatches_v4_when_mode_set` ✅,
  `test_pipeline_threads_suite_through_to_v4_builder` ✅) and all
  existing SA1 V4-prompt tests, the V3 prompt tests, the suite
  auto-inference test, the libero_spatial addendum tests, the message
  builder tests, and the end-to-end async-runner tests with the mocked
  OpenAI client.
- **1 failing** — `test_v4_position_prompt_position_type_conditioning`:
  expects `"Do NOT restate the task instruction"` in the V4 image_patch
  system prompt; that string is currently absent from `prompts.py`.
  This test pins SA4's ptype-disambiguation deliverable
  (`docs/sft_plan/v4_repair/sa4_ptype_disambiguation.md`) and is not
  affected by anything in this task. **No edits made to `prompts.py`.**

## For SA6 — production V4 re-label

Use this command (one per suite, in parallel or serial):

```bash
cd /home/ubuntu/nla-groot && set -a && source .env && set +a && \
PYTHONPATH=src python scripts/labeling/run_label.py \
    --activations-root data/activations/libero_4suite_stride2/<SUITE> \
    --dataset-root     third_party/Isaac-GR00T/examples/LIBERO/<SUITE>_no_noops_1.0.0_lerobot \
    --labels-dir       data/labels/<SUITE>_v4 \
    --prompt-mode v4 --suite <SUITE> \
    --positions-per-example 1 --concurrency 16
```

Where `<SUITE>` ∈ `libero_goal`, `libero_spatial`, `libero_object`,
`libero_10`. `--suite` is technically optional (the dataset-path
inference catches all four suite names) but passing it explicitly
keeps the manifest unambiguous and avoids surprises if the activation
or dataset path is renamed.

The manifest at `<labels-dir>/manifest.json` will record both
`extra.suite` and `extra.prompt_mode` so downstream readers can
distinguish V3 / V4 corpora without re-reading every bullet.

## Return values for the parent

- **CLI for SA6 V4 re-label** — see above (`--prompt-mode v4 --suite <SUITE>`).
- **Test pass count** — 29 passing, 1 pre-existing fail in SA4 territory.
- **Leaks observed in 5 smoke labels** — 0 scaffold-leakage hits, 0
  forbidden-header hits, 2/5 rows leaked motor-imperative phrases
  (`"carry it"` and `"place it"`) in long `anchor` `plan:` bullets;
  flagged for SA2 / SA3 / SA4 prompt-effectiveness follow-up.

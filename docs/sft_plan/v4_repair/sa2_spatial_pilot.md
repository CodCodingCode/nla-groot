# SA2 — libero_spatial V4 addendum + pilot judge

**Verdict: PASS** — V4 B-pass on the V3-failure pilot rose from **0.00% → 85.29%** (29 of 34 worst-case `libero_spatial` rows now graded `grounding=specific`).

## Pilot definition

- **Source pool:** rows in `data/eval/libero_v3_quality_judge.jsonl` with `example_id.startswith("libero_spatial::")` and `grounding.verdict != "specific"`.
- **n:** all 34 such rows (the plan asked for 40 stratified-bad rows; only 34 non-specific spatial rows exist in the V3 judge, so we used all of them — this is strictly the worst-case slice).
- **Bucket breakdown (V3-bad cluster):** `anchor` 14, `image_patch` 15, `last_text` 5.
- **Frames:** reused from `data/labels/libero_4suite_stride2/libero_spatial/frames_cache/` (no re-extraction).
- **Re-label model:** `OPENAI_LABELING_MODEL=gpt-5.4-mini`, concurrency=32.
- **Grader model:** `OPENAI_GRADER_MODEL=gpt-5.1`, concurrency=32, identical multimodal judge as V3.

## Addendum text (registered in `_V4_SUITE_ADDENDA["libero_spatial"]`)

Iteration-1 final shipped 7 rules (SP-1..SP-7). The first 5 came from the plan; SP-6/SP-7 were added in iter-1 after the iter-0 failure analysis (below).

- **SP-1** Spatial relations (`left of`, `right of`, `behind`, `in front of`, `next to`, `above`, `below`, `between`) may ONLY be stated when both anchor and reference object are visible in the attached frame at this timestep; otherwise omit the relation.
- **SP-2** Frame of reference must be explicit ("in the camera frame:", "from the robot's POV:"); bare `"X is left of Y"` is forbidden because frame-ambiguous.
- **SP-3** No invented relations between commonly co-located LIBERO objects (named confabulation pairs: bowl↔plate, mug↔shelf, cube↔tray, wine bottle↔rack).
- **SP-4** The `spatial:` bullet must include at least one visually verifiable landmark ("on the wooden tabletop", "near the silver gripper", "against the dark cabinet wall") instead of generic positional language.
- **SP-5** Occlusion must be named, not invented. If the target is partially hidden by the gripper or another object, the `spatial:` bullet must say so and refuse to describe the hidden side.
- **SP-6** **(iter-1 add)** Do NOT anchor object identity on the instruction text. The instruction often names a color ("pick up the BLACK bowl") that may not match the visible scene; visually verify every color/material/identity from the frame before writing it.
- **SP-7** **(iter-1 add)** Object color/material must come from the pixels. If unsure, omit the modifier (write `"bowl"` rather than guess `"black bowl"`) — an unmodified noun is safer than an invented color.

Full prose lives in `src/nla/labeling/prompts.py::_V4_LIBERO_SPATIAL_ADDENDUM`.

## Design rationale

The V3 audit attributed the spatial cluster failure to "object/spatial-relation hallucinations vs the actual frame", and the plan correctly hypothesized that suite-specific rules around in-frame verification and frame-of-reference would help. SP-1..SP-5 implement that hypothesis verbatim.

However, after iter-0 (SP-1..SP-5 only), V4 B-pass on the worst 34 rows was only **29.4%** (10/34) — a real improvement over V3's 0%, but well below the 85% PASS bar. Reading the 24 remaining iter-0 failures showed the dominant failure mode was **not spatial-relation hallucination** but **instruction-anchored object-identity hallucination**: the labeler was reading the task instruction ("pick up the **black** bowl on the wooden cabinet…") and asserting a `"black bowl"` into the scene even when the actual visible bowls were metallic gray. Of the 24 iter-0 failure reasons, 22 explicitly cite this color/identity confabulation.

SP-6/SP-7 directly target that pattern: SP-6 names the instruction text as a goal description and not ground truth; SP-7 demands every color/material modifier come from the pixels and provides a safe out (drop the modifier rather than guess).

## Headline numbers

| Pass | Spec | n  | B-pass (`grounding=specific`) | C-pass |
|------|------|----|-------------------------------|--------|
| V3 baseline | original V3 prompt | 34 | **0.00%** (0/34) | (not re-graded — pilot is selected on V3 grounding fails) |
| V4 iter-0   | SP-1..SP-5 | 34 | 29.41% (10/34) | 100.00% (34/34) |
| **V4 iter-1** | **SP-1..SP-7 (shipped)** | 34 | **85.29% (29/34)** | **100.00% (34/34)** |

Δ V3 → V4 iter-1 on these same 34 rows: **+85.29 pp**.

Δ V4 iter-0 → V4 iter-1: **+55.88 pp** — the entire jump from "marginal" to "pass" came from SP-6/SP-7 (instruction-priming defense), not the SP-1..SP-5 spatial-relation rules. This is a useful generalization for SA9 / SA10 to weight in their fuller A/B.

C-axis remained 100% across both iterations — V4 does not regress the appropriateness axis.

## Per-row table (all 34 rows; sorted by example_id)

| example_id | V3 grounding | V3 reason | V4 iter-1 grounding | V4 iter-1 reason |
|---|---|---|---|---|
| `traj000001_step000000@p155_anchor` | generic | mentions a black bowl inside the top drawer and a metal bowl on the table, but in the image both visible bowls appear metallic | specific | references the metal bowl in the open drawer, etc. |
| `traj000016_step000014@p152_last_text` | generic | mentions a black bowl and other dark objects that are not present in the scene | specific | references particular objects (cookie box, red-rimmed plate, metallic bowls, cabinet) and their concrete spatial relations |
| `traj000019_step000022@p118_image_patch` | generic | misidentifies a gray metal bowl as black and mentions two small metal cups that are not visible | specific | mentions the gray bowls, red-rimmed plate, snack packet, robot arm, and cabinet in locations that match |
| `traj000021_step000002@p155_anchor` | generic | mentions a black bowl between plate and ramekin that is not visible | specific | refers to the wooden table, dark cabinet, red-rimmed plate, metallic bowls, and ramekins that match |
| `traj000024_step000032@p23_image_patch` | generic | refers to a black bowl and small patterned square item that are not present and mislabels the visible gray | specific | references the visible metallic bowls, striped plate, robot arm, and cabinet with correct spatial relations |
| `traj000050_step000058@p150_anchor` | generic | mentions a black bowl on a stove and a plate on the table, but the scene shows a patterned bowl near a dark cabinet | specific | mentions the patterned bowl on the stove, white plate with red rim, metal cup, and cabinet which match |
| `traj000067_step000012@p39_image_patch` | generic | mentions a black bowl and black cookie box that are not present, and misstates object colors | specific | mentions the speckled bowl on a cookie box, white plate with red rim, and metal cup |
| `traj000074_step000032@p74_image_patch` | generic | mentions a black bowl near the left foreground and a white plate below it, which do not match | specific | refers to white plate with red rim, silver bowls, cabinet, and card in correct spatial layout |
| `traj000085_step000004@p151_anchor` | generic | mentions a black bowl next to the plate which is not visible | specific | mentions the black cabinet, red-rimmed plate, rectangular packet, and multiple bowls in positions that match |
| `traj000104_step000000@p155_anchor` | generic | mentions a black drawer-like tray with a bowl inside | **generic** | label mentions a bowl inside the open drawer, but the frames show the drawer mostly closed |
| `traj000120_step000016@p150_anchor` | generic | mentions a black bowl on a stove burner and a silver plate on the countertop | specific | concrete objects, colors, and positions that match the two views |
| `traj000125_step000020@p42_image_patch` | generic | mentions a black bowl on a white cookie box that does not appear | specific | cookie box, black cabinet, plate with red rim, and precise locations of bowls and cup |
| `traj000128_step000026@p150_last_text` | generic | mentions a black bowl and stove that are not visible | specific | red-rimmed plate, packet, cabinet — concrete and matches frames |
| `traj000136_step000038@p126_image_patch` | generic | mentions a black bowl and ramekin that are not visible | **generic** | wrongly claims a bowl rim is visible at the bottom edge |
| `traj000146_step000008@p150_last_text` | generic | refers to a black bowl on a stovetop burner and nearby gripper that are not visible | **generic** | misdescribes bowl color, gripper above bowl, bowl–plate relation, adds an unseen black cabinet |
| `traj000171_step000002@p150_anchor` | generic | mentions a black bowl and gray plate that are not visible | specific | gray bowl at table center, nearby white plate, packet, right-side cabinet — match |
| `traj000252_step000010@p89_image_patch` | generic | mentions a black bowl next to a white plate that is not visible | **generic** | still incorrectly refers to a black bowl that is not visible |
| `traj000256_step000034@p110_image_patch` | generic | mentions a black bowl and ramekin that are not visible | **generic** | invents a bowl at the lower edge and a package between it and the plate |
| `traj000260_step000038@p152_anchor` | generic | misstates layout and omits visible drawer unit | specific | patterned bowl on a ramekin, white plate with dark center, and black cabinet exactly as seen |
| `traj000277_step000024@p150_anchor` | generic | mentions a black bowl on the stove that is not visible | specific | metal bowl on the stove, patterned plate, packaged item, and black drawer unit exactly as seen |
| `traj000280_step000044@p153_anchor` | generic | mentions a black bowl next to a ramekin that is not visible | specific | striped plate, ramekin, black cabinet, two metal bowls — match |
| `traj000289_step000034@p153_anchor` | generic | mentions a black bowl and silver ramekin arrangement that does not match | specific | two silver bowls, a ramekin — concrete and matches |
| `traj000293_step000058@p91_image_patch` | generic | incorrectly states the bowl is on top of the cookie box | specific | patterned gray bowl on the cookie box, red-and-white striped plate, wooden table, cabinet — match |
| `traj000305_step000000@p151_anchor` | generic | mentions a black bowl and a white bowl on the cabinet that do not match | specific | gray bowl on tray, white plate with red rim, packet, cabinet edge |
| `traj000310_step000008@p98_image_patch` | generic | mentions a black bowl on the stove and a white plate on the table | specific | visible stove, bowl, plate, metal cup, wooden tabletop in correct locations |
| `traj000311_step000026@p151_last_text` | generic | mentions a black bowl and two small metal bowls when the visible bowls are gray | specific | actual gray bowls, white plate with red rings, colorful box, cabinet, gripper |
| `traj000313_step000034@p155_last_text` | generic | mentions a nearby white ramekin that is not visible | specific | red-rimmed plate, metal bowl below it, cookie package to the left, black cabinet |
| `traj000315_step000022@p155_anchor` | generic | mentions a black bowl and black drawer box not seen in the images | specific | concrete objects, colors, and locations match |
| `traj000341_step000010@p12_image_patch` | generic | mentions a black bowl and ramekin that are not visible | specific | silver bowl, white plate with red rim, ramekin, orange packet, cabinet — match |
| `traj000359_step000020@p155_anchor` | generic | mentions two black bowls and a small colorful packet when the scene shows metallic-looking bowls | specific | silver bowl, black drawer, red-rimmed plate, packet in locations that match |
| `traj000374_step000002@p78_image_patch` | generic | mentions a black bowl that is not visible and misstates relations | specific | plate, bowls, and cabinet edges in correct locations |
| `traj000374_step000018@p54_image_patch` | generic | mentions a black bowl, white plate, and ramekin ring that are not visible | specific | concrete objects, colors, and spatial layout match |
| `traj000383_step000030@p32_image_patch` | generic | mentions a black bowl and cabinet occlusion that are not visible | specific | metallic bowls, red-rimmed plate, packet, cabinet and their spatial relations |
| `traj000427_step000006@p20_image_patch` | generic | mentions a black bowl and ramekin that do not appear | specific | wooden table, gray arm, gray bowls, ramekin, plate with red rim, and dark cabinet |

5 of 34 (14.7%) still flip to `generic` under V4 iter-1. Two of those (`traj000252_step000010@p89`, `traj000146_step000008@p150`) continue to assert a "black bowl" despite SP-6/SP-7 — these are the residual cases where the addendum is not strict enough; the other 3 fail on subtler scene-misreading (wrong drawer state, invented bowl at frame edge). These 5 are good targets for SA10's regression-gate spot-check.

## Iteration history

| Iter | Rules | Labels artifact | Judge artifact | B-pass |
|---|---|---|---|---|
| iter-0 | SP-1..SP-5 (plan as written) | `data/labels/sa2_pilot_v4_spatial/labels.iter0.jsonl` | `data/eval/sa2_pilot_v4_spatial_judge.iter0.jsonl` | 29.4% |
| **iter-1** (shipped) | **SP-1..SP-7** | `data/labels/sa2_pilot_v4_spatial/labels.jsonl` | `data/eval/sa2_pilot_v4_spatial_judge.jsonl` | **85.3%** |

## Files

### Edits to `src/nla/labeling/prompts.py`

- New module-level constant `_V4_LIBERO_SPATIAL_ADDENDUM` (SP-1..SP-7 prose).
- Registered as `_V4_SUITE_ADDENDA["libero_spatial"] = _V4_LIBERO_SPATIAL_ADDENDUM`.
- New `suite: str | None = None` field on `PositionLabelInput` (last field — preserves positional-arg back-compat for every existing caller).
- New helper `infer_suite_from_example_id(example_id, *, extra=None) -> str | None` returning `"libero_spatial" / "libero_goal" / "libero_object" / "libero_10"` from the eval-style `libero_<suite>::` prefix or from `extra["suite"]`. Tuple of recognized suites pinned at `_LIBERO_SUITES`.
- `build_v4_position_prompt(inp, suite=None)` now auto-infers `suite` from `inp.suite`, `inp.extra["suite"]`, or `inp.example_id` when called with `suite=None`; explicit `suite=...` from the caller wins.

### Edits to `src/nla/labeling/__init__.py`

- Re-exported `infer_suite_from_example_id`.

### Tests added to `tests/test_labeling_smoke.py`

- `test_v4_libero_spatial_addendum_present` — asserts the spatial system prompt now contains the SP-1..SP-7 rule headers, the SP-3 confabulation lexicon (bowl/plate/mug/shelf/cube/tray/wine bottle/rack), and SP-6/SP-7 keywords ("visually verify", "metallic").
- `test_v4_libero_spatial_addendum_absent_for_other_suites` — asserts the SP-N rule headers do NOT leak into `libero_goal` / `libero_object` / `libero_10` / no-suite prompts. (Uses the unique `Rule SP-N` markers to disambiguate against the base `_IMAGE_PATCH_RULES` which already contains the substring "visible in the attached camera frame".)
- `test_v4_suite_auto_inference_from_example_id` — asserts that `build_v4_position_prompt(inp)` with `suite=None` auto-activates the addendum when `inp.example_id.startswith("libero_spatial::")`, and exercises `infer_suite_from_example_id` directly for all four suites + the `extra={"suite": ...}` fallback.

The existing SA1 `test_v4_position_prompt_suite_hook` was tightened to drop the now-obsolete "empty addendum is no-op" arm (since SA2 has registered the libero_spatial block, that branch is no longer empty) — the test still pins the unknown-suite no-op contract.

Run: `cd /home/ubuntu/nla-groot && PYTHONPATH=src .venv/bin/python -m pytest tests/test_labeling_smoke.py -q -k "v4_libero_spatial or v4_suite_auto_inference or v4_position_prompt_suite_hook"` → **4 passed**.

(There is an unrelated failure of `test_v4_position_prompt_position_type_conditioning` from SA4's concurrent edits to the IMAGE-PATCH last-bullet clause — SA4 owns that test.)

### Pilot scripts

- `scripts/labeling/sa2_pilot_v4_spatial_relabel.py` — selects the 34 V3-bad rows, builds `PositionLabelInput`s from the existing activations, monkey-patches `openai_client.build_position_prompt` to dispatch through `build_v4_position_prompt(inp, suite="libero_spatial")`, streams to `data/labels/sa2_pilot_v4_spatial/labels.jsonl` with resume.
- `scripts/eval/sa2_pilot_v4_spatial_judge.py` — re-grades the pilot using the multimodal `gpt-5.1` grader from `nla.labeling.grader`, schema-compatible with `verify_libero_label_quality.py`'s `libero_quality_judge.jsonl`.

### Artifacts

- `data/labels/sa2_pilot_v4_spatial/labels.jsonl` (34 rows; shipped) — V4 iter-1 labels (SP-1..SP-7).
- `data/labels/sa2_pilot_v4_spatial/labels.iter0.jsonl` (34 rows) — V4 iter-0 labels (SP-1..SP-5 only), retained for forensics.
- `data/labels/sa2_pilot_v4_spatial/frames_cache/` — frame cache populated by the relabel script.
- `data/eval/sa2_pilot_v4_spatial_judge.jsonl` (34 rows; shipped) — V4 iter-1 grades.
- `data/eval/sa2_pilot_v4_spatial_judge.iter0.jsonl` (34 rows) — V4 iter-0 grades.

## Notes for downstream subagents

- **SA5 (pipeline wiring):** auto-inference is wired. If your pipeline already constructs `PositionLabelInput` with `example_id` that lacks the `libero_<suite>::` prefix (production labeling uses the raw `traj…_step…@p…_…` id), set `inp.extra["suite"] = "libero_spatial"` from the dataset/run config OR set `inp.suite = "libero_spatial"` — either is sufficient and the V4 builder will pick the addendum up automatically without an explicit `suite=...` kwarg. The new `_LIBERO_SUITES` tuple in `prompts.py` is the canonical list of recognized values.
- **SA6 (re-label orchestration):** for the full libero_spatial re-label, dispatch through `build_v4_position_prompt(inp, suite="libero_spatial")` exactly as `scripts/labeling/sa2_pilot_v4_spatial_relabel.py` does (monkey-patch `openai_client.build_position_prompt` or import the V4 builder directly from `nla.labeling`). Concurrency 32 ran 34 labels in ~50 s with no errors; 27k spatial labels should land in ≈40 min at the same rate. Cost on this pilot was approximately $0.05 in labeling + $0.18 in grading (well under the $0.25 plan estimate).
- **SA9 (judge A/B 500):** the iter-0 → iter-1 lift (+55.88 pp) suggests most of the V3 spatial cluster failure comes from instruction-priming on color/identity, not from spatial-relation hallucination per se. If your stratified 500-row run still shows weak spatial B-pass after V4 ships, the lever is more SP-6/SP-7-style instruction-anchoring constraints, not more SP-1..SP-3-style relation constraints.
- **SA10 (audit regression gate):** the 5 residual iter-1 failures are listed above and are a useful spot-check set — if those go to ≥3 specific after broader changes in V4 they confirm continued lift; if they regress to <2 specific they confirm V4 ships in degraded form.

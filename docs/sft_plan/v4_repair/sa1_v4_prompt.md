# SA1 — V4 prompt scaffolding

Status: green (23/23 tests in `tests/test_labeling_smoke.py`).

## What changed

- New builder `build_v4_position_prompt(inp, suite=None)` in
  `src/nla/labeling/prompts.py`. Does **not** touch
  `build_position_prompt` / `build_strict_position_prompt`; V3 audits and
  existing tests keep using them unchanged.
- New module-level constants (also re-exported from
  `nla.labeling.__init__`):
  - `V4_BULLET_CATEGORIES` — `scene, target, distractor, spatial, plan, language`.
  - `V4_FORBIDDEN_HEADERS` — `gripper, motion, image_region`.
  - `V4_SCAFFOLD_FORBIDDEN_PHRASES`, `V4_MOTOR_IMPERATIVE_PHRASES`, `V4_PLAN_PHASES`.
- New system-prompt blocks: `_V4_STRICT_BLOCK`, `_V4_EXTRA_RULES`,
  `_V4_POSITION_SYSTEM`. The V4 system text inherits `_POSITION_SYSTEM`
  (style + image-patch guards + anthropomorphic-phrasing bans) and layers
  the V4 rules on top.
- Position-type-conditional last-bullet rule is a dict, not an if/elif:
  `_LAST_BULLET_BY_POSITION_TYPE: dict[str, str]` keyed by
  `image_patch | last_text | anchor | fallback`. SA4 can override one
  ptype without forking the builder.
- Suite hook: `_V4_SUITE_ADDENDA: dict[str, str]` (initially empty).
  `suite=None` and unknown suites are silent no-ops.

## Forbidden phrases now in the V4 system text

- Scaffold leakage (9): `"action head"`, `"this patch carries"`,
  `"token carries"`, `"the patch carries"`, `"carries the"`,
  `"transformer"`, `"embedding"`, `"hidden state"`, `"residual stream"`.
- Motor imperatives (13): `"grasp the"`, `"reach toward"`, `"reach over"`,
  `"align the gripper"`, `"lower the gripper"`, `"raise the gripper"`,
  `"move toward"`, `"approach the"`, `"carry it"`, `"place it"`,
  `"release the"`, `"open the gripper"`, `"close the gripper"`.
- Bullet headers (3): `gripper`, `motion`, `image_region` (folded into
  `plan` / `target` / `spatial`).

## Position-type-conditional last bullet (the Agent-4 lever)

| position_type | last bullet must be | rationale |
|---|---|---|
| `image_patch` | `target:` or `scene:` — perceptual description of THIS frame; must NOT restate the instruction | drives down `image_patch`↔`last_text` Jaccard |
| `last_text`  | `plan:` — next ~3 timesteps, named phase from `V4_PLAN_PHASES`, third-person | binds instruction → motion |
| `anchor`     | `plan:` — overall trajectory phase | matches the trajectory-level signal |
| `fallback`   | `plan:` (preferred) or `target:` | safe default |

`language:` is explicitly marked OPTIONAL on `image_patch` and `anchor`;
4-bullet image_patch responses are now canonical (resolves the Agent-2
strict-conformance ambiguity).

## How SA2-SA5 extend this

- **SA2** (libero_spatial spatial-grounding): register
  `_V4_SUITE_ADDENDA["libero_spatial"] = "..."` in `prompts.py`; may also
  add a `suite: str | None` field to `PositionLabelInput` and a small
  inference helper that maps `example_id` → suite. The
  `build_v4_position_prompt(inp, suite=...)` signature already accepts the
  hook; no changes to the builder needed.
- **SA3** (motor / scaffold audit regex): import
  `V4_MOTOR_IMPERATIVE_PHRASES` and `V4_SCAFFOLD_FORBIDDEN_PHRASES`
  directly from `nla.labeling.prompts` (or the `nla.labeling` package) and
  build the audit regex from those tuples — no need to re-derive from
  prose.
- **SA4** (ptype disambiguation + Jaccard delta): mutate
  `_LAST_BULLET_BY_POSITION_TYPE[ptype]` to refine the per-ptype clause.
  No builder edits required.
- **SA5** (pipeline wiring): thread an env flag
  (`NLA_POSITION_PROMPT_MODE=v4`) through
  `src/nla/labeling/openai_client.py` and `scripts/labeling/run_label.py`
  that dispatches to `build_v4_position_prompt`. Suite-aware call sites
  should pass `suite=...` derived from the dataset/run config so SA2's
  addendum activates automatically.

## Tests added (`tests/test_labeling_smoke.py`)

- `test_v4_position_prompt_forbids_scaffold_leakage`
- `test_v4_position_prompt_forbids_motor_imperatives`
- `test_v4_position_prompt_position_type_conditioning`
- `test_v4_position_prompt_suite_hook`

Run: `cd /home/ubuntu/nla-groot && PYTHONPATH=src .venv/bin/python -m pytest tests/test_labeling_smoke.py -x -q` → **23 passed**.

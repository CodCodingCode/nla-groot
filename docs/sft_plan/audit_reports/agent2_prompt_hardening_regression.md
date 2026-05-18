# Agent 2 — Prompt-Hardening Regression Scan (V3 LIBERO)

Single-pass scan over 101,580 V3 LIBERO labels (`libero_4suite_stride2/`) plus the V2 DROID and LIBERO-goal pilot baselines, checking whether the hardened labeling prompt in `src/nla/labeling/prompts.py` eliminated the failure modes documented in `docs/sft_plan/01_data_audit.md`. Run via `PYTHONPATH=src .venv/bin/python scripts/eval/audit_prompt_hardening.py`.

## Verdict

**YELLOW**

Reasons:
- position-aware bullet conformance=63.59% (<99% GREEN bar)
- strict 'all 5 prefixes always' conformance=17.06% — but the hardened prompt only asks last_text rows for `language:`, so this strict metric is partly schema-design, not labeler failure

## Failure-mode rates

| Failure mode | goal | spatial | object | 10 | V3-overall | V2-DROID | Pilot |
|---|---|---|---|---|---|---|---|
| n (rows) | 25,680 | 25,920 | 27,240 | 22,740 | 101,580 | 100,336 | 243 |
| Anthropomorphic phrasing | 5 (0.019%) | 2 (0.008%) | 0 (0.000%) | 0 (0.000%) | 7 (0.007%) | 5,675 (5.656%) | 37 (15.23%) |
| Numerical confabulation | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 1 (0.004%) | 1 (0.001%) | 14 (0.014%) | 0 (0.000%) |
| image_region bullets | 19 (0.074%) | 9 (0.035%) | 10 (0.037%) | 20 (0.088%) | 58 (0.057%) | 666 (0.664%) | 100 (41.15%) |
| 'reads/has read the instruction' | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 30 (0.030%) | 0 (0.000%) |
| 'understands/comprehends the goal' | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| 'ready to execute' | 2 (0.008%) | 0 (0.000%) | 1 (0.004%) | 0 (0.000%) | 3 (0.003%) | 280 (0.279%) | 0 (0.000%) |
| Empty / degenerate (<50 chars or <3 bullets) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
|     of which: <50 chars | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
|     of which: <3 bullets | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| Error rows (non-null error) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

## Bullet-prefix conformance

% of rows that contain *each* expected bullet prefix, plus the two aggregate metrics:
- **strict** = all five of `- language:`, `- target:`, `- scene:`, `- spatial:`, `- plan:` present;
- **relaxed (position-aware)** = `last_text` rows need all five; `image_patch` / `anchor` rows only need target+scene+spatial+plan, because the hardened prompt explicitly steers those positions away from a `language:` bullet (see `_IMAGE_PATCH_RULES` in `src/nla/labeling/prompts.py`).

| Bullet present | goal | spatial | object | 10 | V3-overall | V2-DROID | Pilot |
|---|---|---|---|---|---|---|---|
| - language: | 16.85% | 21.51% | 23.30% | 18.76% | 20.20% | 18.32% | 50.21% |
| - target: | 99.98% | 100.00% | 100.00% | 99.98% | 99.99% | 100.00% | 100.00% |
| - scene: | 99.35% | 98.23% | 99.70% | 97.00% | 98.63% | 100.00% | 100.00% |
| - spatial: | 99.87% | 99.31% | 99.59% | 91.93% | 97.87% | 68.72% | 74.49% |
| - plan: | 98.72% | 95.93% | 96.40% | 97.32% | 97.07% | 7.277% | 11.11% |
| **All 5 prefixes present (strict)** | 16.07% | 19.18% | 22.60% | 9.112% | 17.06% | 0.098% | 0.412% |
| **Position-aware conformance (relaxed)** | 64.47% | 64.75% | 68.67% | 55.20% | 63.59% | 4.502% | 8.230% |

### Position-type-conditioned conformance (V3 aggregate)

Separates 'labeler skipped a prescribed bullet' from 'prompt did not ask for that bullet here'.

| position_type | n rows | language: % | target: % | scene: % | spatial: % | plan: % | all-5 (strict) % | relaxed-conformant % |
|---|---|---|---|---|---|---|---|---|
| last_text | 51,085 | 40.10% | 99.99% | 97.28% | 96.21% | 100.00% | 33.87% | 33.87% |
| image_patch | 50,329 | 0.010% | 99.99% | 100.00% | 99.56% | 94.09% | 0.008% | 93.65% |
| anchor | 166 | 15.66% | 100.00% | 98.80% | 98.19% | 100.00% | 13.25% | 96.99% |

### Non-conformant examples

These rows fail the **strict** all-5-prefixes test (we list them for completeness; many of these are simply image_patch rows missing only `language:`, which the prompt expected).

### libero_goal (64.47% position-aware conformant)

**traj000002_step000006@p100_image_patch** — matched: `missing: language`

```
- scene: tabletop workspace with a robot arm, a black wine bottle, two circular plates, a bowl, and a wooden rack.
- target: the black wine bottle stands upright near the center of the workspace, separate from the rack.
- distractor: a black handled utensil-like object lies near the lower-left area, and a blue rectangular item sits near the plates.
- spatial: the wooden rack is adjacent to the large dark box on the right, with open slats visible for placing the bottle.
- plan: bottle-to-rack placement phase active; grasp the upright bottle and move it toward the slatted rack opening.
```

**traj000000_step000032@p120_image_patch** — matched: `missing: language`

```
- scene: tabletop workspace with a gray robot arm, a white plate with red rings, and a gray bowl on the table.
- target: bowl and plate are both visible; the bowl sits below the plate in the camera view.
- spatial: the plate is clear and unobstructed on the light wood surface, ready to receive the bowl.
- distractor: a blue rectangular object and dark fixture sit near the right-side furniture, away from the placement area.
- plan: transfer phase active; align the gripper with the bowl, then carry it to the plate.
```

**traj000001_step000054@p111_image_patch** — matched: `missing: language`

```
- scene: tabletop workspace with a robot arm, a black wine bottle, a wooden rack, a bowl, a plate, and a phone.
- target: the wine bottle stands upright near the center of the table, next to the gray bowl.
- distractor: a brown wooden rack with slats and metal supports sits on the right side, separate from the bottle.
- spatial: the arm is positioned above the table, with the gripper near the bottle and the rack reachable to the right.
- plan: grasp the wine bottle and move it toward the rack for placement.
```

**traj000000_step000028@p010_image_patch** — matched: `missing: language`

```
- scene: tabletop workspace with a robot arm, a white plate with red rings, and a speckled bowl on a light wood surface.
- target: the bowl is the object to move, positioned below the plate in the close-up view.
- distractor: a blue rectangular item lies near the plate and is not part of the placement task.
- spatial: the plate is clear and accessible for receiving the bowl, with the bowl currently separated from it.
- plan: reach toward the bowl and carry it to the plate; this patch carries bowl-and-plate placement context into the action head.
```

**traj000000_step000014@p088_image_patch** — matched: `missing: language`

```
- scene: tabletop workspace with a white plate, a gray bowl, a black bottle, a blue card, and a red-ringed dish.
- target: gray bowl is the object to be moved onto the white plate.
- distractor: black bottle stands between the bowl and the plate; blue card sits near the bowl.
- spatial: bowl is below and left of the plate in the closer camera view, with both resting on the beige surface.
- plan: grasp the gray bowl and carry it toward the plate; this patch carries the bowl-and-plate relation into the next action.
```

### libero_spatial (64.75% position-aware conformant)

**traj000001_step000058@p155_last_text** — matched: `missing: scene`

```
- language: pick up the black bowl and place it on the plate.
- target: black bowl in the top drawer, with patterned interior visible.
- target: round plate on the tabletop, white center with red rim.
- spatial: bowl is close to the open drawer edge and the plate sits clear on the table surface.
- plan: reach into the drawer for the bowl, then carry it over the plate for placement.
```

**traj000000_step000038@p152_last_text** — matched: `missing: language`

```
- scene: tabletop with a robot arm, a black bowl, a black cookie box, and two plates visible.
- target: black bowl with patterned interior near the cookie box, positioned in front of the arm.
- distractor: cookie box is open at the right side and is not the grasp target.
- spatial: bowl is next to the cookie box and above the plate area in the workspace.
- plan: pick-up phase for the black bowl and carry it toward the plate.
```

**traj000000_step000016@p152_last_text** — matched: `missing: scene`

```
- language: pick up the black bowl and place it on the plate.
- target: black bowl is the main grasp target, with the larger dark bowl near the cookie box.
- distractor: cookie box sits beside the bowl, and the white plate with red rings is the placement destination.
- spatial: bowl, cookie box, and plate are arranged in a left-to-right workspace cluster on the tabletop.
- plan: reach to the black bowl, grasp it, then carry it to the plate.
```

**traj000000_step000050@p152_last_text** — matched: `missing: scene`

```
- language: pick up the black bowl and place it on the plate.
- target: black bowl with patterned interior and yellow rim in the close camera view.
- distractor: cookie box is the dark rectangular box with silver handles on the right.
- spatial: bowl is next to the cookie box on the tabletop; plate is the red ring-shaped target on the left.
- plan: grasp the bowl, lift it clear of the table, and carry it toward the plate for placement.
```

**traj000001_step000034@p155_last_text** — matched: `missing: language`

```
- scene: tabletop workspace with a wooden cabinet drawer open and the robot arm beside it.
- target: black bowl is visible inside the top drawer, centered in the close-up frame.
- distractor: a white plate with a red rim sits on the table, and a small packet lies nearby.
- spatial: gripper is positioned above the drawer opening near the bowl, aligned for a downward reach.
- plan: pick-and-place phase active; reach into the top drawer for the black bowl, then move it toward the plate.
```

### libero_object (68.67% position-aware conformant)

**traj000001_step000032@p082_image_patch** — matched: `missing: language`

```
- scene: tabletop arrangement with a white woven basket, several condiment bottles, and a blue can on a tiled floor.
- target: ketchup bottle with red cap and red label standing near the center-right of the scene.
- distractor: green bottle, brown bottle, and blue can remain separate from the ketchup.
- spatial: basket sits left of the ketchup; the robot gripper is above the objects and not yet aligned to grasp.
- plan: reach toward the ketchup bottle, grasp it, and carry it toward the basket.
```

**traj000000_step000002@p147_last_text** — matched: `missing: language`

```
- scene: tabletop setup with a woven basket on the left and several bottles clustered on the floor.
- target: orange juice carton standing near the center-right, with a bright orange label and green top.
- distractor: red bottle in front-left, green bottle behind the carton, and a dark red bottle on the right.
- spatial: basket is left of the orange juice carton; the carton is between the basket and the right-side bottle cluster.
- plan: reach toward the orange juice carton and carry it toward the basket for placement.
```

**traj000000_step000038@p042_image_patch** — matched: `missing: language`

```
- scene: tabletop or floor-level grasping setup with a robot arm hovering above three bottles and a woven basket.
- target: orange juice carton with orange-and-green label centered in the workspace, directly below the gripper.
- distractor: red bottle on the left and dark bottle on the right remain unselected beside the target.
- spatial: basket sits to the left of the bottles; the orange juice stands between the basket and the robot hand.
- plan: target bottle and nearby gripper alignment are carried forward for a pick-up then basket-placement action.
```

**traj000000_step000006@p147_last_text** — matched: `missing: language`

```
- target: orange juice carton with orange-green label is the object to grasp and place.
- scene: wicker basket sits on the left side of the workspace on the tile floor.
- spatial: orange juice carton stands near the center-right of the object cluster, with the basket separated to its left.
- gripper: robot gripper is aligned above the workspace and ready to descend toward the bottle cluster.
- plan: pick up the orange juice carton first, then carry it to the basket for placement.
```

**traj000000_step000014@p088_image_patch** — matched: `missing: language`

```
- scene: tabletop with gray tiled floor, basket on the left, and several bottle-like objects arranged near the center.
- target: orange juice carton stands near the middle among brown and green bottles.
- distractor: basket is empty and separate from the drink objects.
- spatial: gripper hovers above the central cluster, aligned over the orange juice carton area.
- plan: grasp the orange juice carton and carry it toward the basket; this patch carries forward the pickup target and reach location.
```

### libero_10 (55.20% position-aware conformant)

**traj000000_step000034@p155_last_text** — matched: `missing: language`

```
- scene: wooden tabletop with two small white plates with red rings at the left and right edges.
- target: white mug with a yellow handle sits near the left plate.
- target: yellow-and-white mug stands near the center under the robot arm.
- spatial: left plate is closest to the white mug; right plate is closest to the yellow-and-white mug.
- plan: move the white mug onto the left plate first, then transfer the yellow-and-white mug onto the right plate.
```

**traj000000_step000024@p036_image_patch** — matched: `missing: language`

```
- scene: tabletop with two plates and two mugs under the robot gripper.
- target: white mug stands left of the red patterned mug; yellow-handled mug sits near the left plate.
- distractor: red patterned mug is centered under the gripper and is not the first placement target.
- spatial: left plate is open for the white mug; right plate is open for the yellow-and-white mug.
- plan: grasp the white mug and place it onto the left plate before handling the yellow-and-white mug.
```

**traj000002_step000006@p100_image_patch** — matched: `missing: language`

```
- scene: tabletop workspace with a microwave at the left edge and a robot arm behind the table.
- target: yellow mug on the wooden surface, matching the yellow-and-white mug in the instruction.
- distractor: gray mug with a white handle sits farther to the right and is not the target.
- spatial: microwave door is open, creating a clear opening for placing the mug inside.
- plan: grasp the yellow mug and move it toward the microwave opening before closing the door.
```

**traj000000_step000004@p086_image_patch** — matched: `missing: language`

```
- scene: tabletop with two plates at the left and right edges, a yellow mug on the left side, a red patterned mug in the center, and a gray-white mug on the right.
- target: the left plate is the destination for the white mug, visible as the pale plate near the left edge of the table.
- target: the right plate is the destination for the yellow-and-white mug, visible as the pale plate near the right edge of the table.
- spatial: the red patterned mug stands between the two plates, with the yellow mug left of center and the gray-white mug right of center.
- plan: transfer phase active; place the ...
```

**traj000001_step000042@p154_last_text** — matched: `missing: scene,spatial`

```
- language: put the white mug on the plate and place the chocolate pudding to the right of the plate.
- target: white mug is the object to lift, with the gripper near its handle.
- target: round white plate with red rim is visible on the table as the placement surface.
- target: chocolate pudding cup is the small brown rectangular dessert container on the table.
- plan: move the mug onto the plate, then place the pudding to the right side of the plate.
```


## Per-bullet length distribution (V3 aggregate, tokens per bullet body)

| Bullet | n bullets | mean tok | p10 | p50 | p90 |
|---|---|---|---|---|---|
| language | 20,516 | 13.5 | 9 | 14 | 18 |
| target | 113,134 | 16.7 | 13 | 16 | 21 |
| scene | 100,378 | 20.0 | 15 | 20 | 25 |
| spatial | 99,441 | 19.0 | 15 | 19 | 23 |
| plan | 101,415 | 17.3 | 13 | 16 | 24 |
| distractor | 62,654 | 17.6 | 14 | 17 | 22 |
| image_region | 58 | 18.4 | 15 | 18 | 22 |
| _other (uncategorised) | 10,304 | 15.2 | 12 | 15 | 18 |

## Non-canonical bullet categories (V3 aggregate)

Bullets whose category is not in the prompt's allowed set (`scene, target, distractor, spatial, plan, language, image_region`). Counts are per-bullet, not per-row.

| Category | # bullets |
|---|---|
| `gripper` | 5,769 |
| `motion` | 4,526 |
| `action_head` | 5 |
| `last_text` | 2 |
| `action_link` | 1 |
| `action` | 1 |
| **total non-canonical bullets** | 10,304 |

These bullets are not flagged by the user's failure-mode list, but they do count against bullet-prefix conformance because they crowd out a prescribed category. `gripper:` and `motion:` together account for >99% of the non-canonical volume and explain a non-trivial fraction of the image_patch rows that are missing `plan:` or `spatial:`.

## Top anthropomorphic phrase hits (V3 aggregate)

| Phrase | # rows hit |
|---|---|
| `the model` | 4 |
| `the policy` | 2 |
| `prepares to` | 1 |

## Examples for V3 failure modes (>0.5% incidence)

_(No V3 failure mode exceeded 0.5%; no example listing needed.)_

## Cross-reference: Agent 1 (multimodal gpt-5.1 judge, 500-row sample)

Agent 1 read 500 V3 rows with a multimodal judge (suite breakdown: {'libero_10': 122, 'libero_goal': 125, 'libero_object': 127, 'libero_spatial': 126}).

- Grounding pass: **91.0%** (9.0% C-grounding fails). These are visual-misidentification failures ("misstates layout", "misidentifies the visible can"). My regex scan **does not catch any of these** — they require pixels to detect.
- Appropriateness pass: **98.2%** (1.8% appropriateness fails). These fails are dominated by low-level motor commands ("grasp the bowl and carry it", "align the gripper, lift, place") rather than the anthropomorphic phrasing the V2/Pilot baselines showed. **My anthropomorphic regex is 0.007% but Agent 1's judge flags ~1.8% — they're catching a distinct C-failure mode** (actuator-level imperative phrasing) that the hardened prompt did not explicitly forbid.

Sample Agent 1 appropriateness-fail reasons:
  - Label includes actuator-level instructions about aligning the gripper, lifting, and placing rather than higher-level visual-plan content.
  - Last bullet gives low-level motor instructions about moving, lifting, and placing rather than higher-level scene or plan state.
  - Last bullet gives actuator-level grasp-and-move commands rather than higher-level state or plan.

**Implication**: the hardened prompt eliminated the *old* C-failure mode (anthropomorphic / cognitive-state phrasing) but the labeler has shifted to a *new* one (low-level motor imperatives in the `plan:` bullet, often co-occurring with the non-canonical `motion:` / `gripper:` bullets called out above).

## Recommendations

1. **New C-failure mode**: Agent 1's multimodal judge flagged 1.8% appropriateness fails — dominated by low-level motor-imperative phrasing ('grasp the X', 'align the gripper, lift, place') rather than the anthropomorphic phrasing the original hardened prompt targeted. **Action**: extend `_FORBIDDEN_PHRASING` in `src/nla/labeling/prompts.py` with imperative-verb patterns ('grasp', 'lift', 'reach', 'place', 'carry it') when they appear inside the `plan:` bullet, then re-label the ~6% of rows that fail. Alternatively, add a scrub step that rewrites imperative `plan:` bullets to a neutral 'plan: <phase> active; <observable state>' template.
2. **Non-canonical bullets**: V3 contains 10,304 bullets whose category is outside the prompt's allowed set (top: `gripper` (5,769), `motion` (4,526), `action_head` (5)). The hardened `build_strict_position_prompt` already enforces the closed vocabulary; the issue is that production labeling used `build_position_prompt`, which lists categories without forbidding others. **Action**: switch the default labeling entrypoint to `build_strict_position_prompt`, *or* run a category-rewrite scrub that maps `gripper:` → `target:` (when gripper state is the topic) and `motion:` → `plan:` before SFT.
3. image_region: bullets at 0.057% (58 rows; pilot was 41%, V2 DROID was 0.66%). Run `scripts/labeling/strip_hallucinated_image_region.py --match patch_or_layout --mode strip` over each of the four V3 labels.jsonl files; that single sweep finishes the job without touching the prompt.

## Method notes

- Counts are per-label (one bullet hit = one row hit). Rates are denominated against total rows in each file (including any rows with empty descriptions / non-null `error`).
- Anthropomorphic phrasing uses case-insensitive substring matching against the prompt-hardening phrase list documented in the user task. The same row can match multiple phrases; the `top anthropomorphic phrase hits` table breaks them out individually.
- Numerical confabulation uses the regex shipped in the user task (measurement units mm/cm/m/in/°/rad/kg/g/N, with optional decimal). It is intentionally narrower than `scripts/labeling/scrub_fabricated_measurements.py` (which also catches `5-8` ranges with hedging words); the spec asked for the tighter regex.
- `image_region` detection is a bullet-prefix match (one or more lines starting with `- image_region:` or `image region:`).
- Bullet conformance accepts both exact-match (`- language:`) and prefix-match (`- language/state:`) for each canonical category. Categories not in the canonical set are bucketed under `_other` in the length-distribution table.


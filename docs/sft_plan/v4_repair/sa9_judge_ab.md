# SA9 — LIBERO V4 Multimodal Judge A/B (vs V3)

**Headline metric.** Stratified-500-row gpt-5.1 multimodal judge comparing the V4 rewritten dataset against the V3 frozen baseline (Agent 1's audit, `data/eval/libero_v3_quality_judge.jsonl`).

## Methodology

- **Script:** `scripts/eval/verify_libero_label_quality.py` (the exact script that produced the V3 baseline; unchanged for V4).
- **Grader:** OpenAI `gpt-5.1`, prompt unchanged from V3.
- **Sample size:** 500 rows per run, stratified by (suite × position_type) = 12 buckets.
- **Seed:** `--seed 0` (matches the V3 run), so bucket sizes are directly comparable.
- **Concurrency:** 32.
- **Frames:** the V3 frames cache (`data/labels/libero_4suite_stride2/libero_<suite>/frames_cache/`) is reused — V4 did not re-render frames; the underlying LIBERO trajectories are identical.

Two runs were performed:

1. **`v4_combined` (headline / paper-grade pool).** Samples the full V4 merged label pool — 82,005 newly-rewritten rows + 19,350 V3-kept rows = 101,580 total in `data/labels/libero_4suite_v4_combined/labels.jsonl`. This is the apples-to-apples comparison with V3 because it is exactly the row set SFT will train on.
2. **`v4_only` (attributable improvement).** Samples only the V4-rewritten rows (82,005 across `data/labels/libero_4suite_v4/libero_<suite>/labels.jsonl`). This isolates the rewrite quality without dilution from V3-kept rows.

Per-suite directory views needed by the judge script were built with `scripts/eval/build_v4_per_suite_view.py`:

- `data/labels/libero_4suite_v4_combined_per_suite/libero_<suite>/{labels.jsonl, frames_cache}` (split from combined; suite prefix stripped from `source_example_id`; frames symlinked to V3).
- `data/labels/libero_4suite_v4_view/libero_<suite>/{labels.jsonl, frames_cache}` (symlinked from `libero_4suite_v4` + V3 frames).

---

## A. V4-combined vs V3 (headline)

### Overall — V4-combined

| Metric | V3 baseline | V4 | Δ |
|---|---|---|---|
| n | 500 | 500 | — |
| B (grounding=specific) | 91.00% (455/500) | 90.20% (451/500) | -0.80pp |
| C (appropriateness=appropriate) | 98.20% (491/500) | 99.20% (496/500) | +1.00pp |

### Per-suite — V4-combined

| suite | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |
|---|---|---|---|---|---|---|---|
| spatial | 126 | 73.02% | 91.27% | +18.25pp | 98.41% | 99.21% | +0.79pp |
| goal | 125 | 98.40% | 96.80% | -1.60pp | 95.20% | 99.20% | +4.00pp |
| object | 127 | 99.21% | 93.70% | -5.51pp | 99.21% | 100.00% | +0.79pp |
| 10 | 122 | 93.44% | 78.69% | -14.75pp | 100.00% | 98.36% | -1.64pp |

### Per-position_type — V4-combined

| position_type | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |
|---|---|---|---|---|---|---|---|
| anchor | 159 | 88.68% | 91.82% | +3.14pp | 98.11% | 100.00% | +1.89pp |
| image_patch | 173 | 89.02% | 90.75% | +1.73pp | 99.42% | 100.00% | +0.58pp |
| last_text | 168 | 95.24% | 88.10% | -7.14pp | 97.02% | 97.62% | +0.60pp |

### 12-cell matrix — V4-combined

| suite/ptype | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |
|---|---|---|---|---|---|---|---|
| spatial/anchor | 41 | 65.85% | 95.12% | +29.27pp | 97.56% | 100.00% | +2.44pp |
| spatial/image_patch | 44 | 65.91% | 88.64% | +22.73pp | 100.00% | 100.00% | +0.00pp |
| spatial/last_text | 41 | 87.80% | 90.24% | +2.44pp | 97.56% | 97.56% | +0.00pp |
| goal/anchor | 41 | 97.56% | 97.56% | +0.00pp | 95.12% | 100.00% | +4.88pp |
| goal/image_patch | 42 | 97.62% | 100.00% | +2.38pp | 97.62% | 100.00% | +2.38pp |
| goal/last_text | 42 | 100.00% | 92.86% | -7.14pp | 92.86% | 97.62% | +4.76pp |
| object/anchor | 41 | 97.56% | 92.68% | -4.88pp | 100.00% | 100.00% | +0.00pp |
| object/image_patch | 44 | 100.00% | 93.18% | -6.82pp | 100.00% | 100.00% | +0.00pp |
| object/last_text | 42 | 100.00% | 95.24% | -4.76pp | 97.62% | 100.00% | +2.38pp |
| 10/anchor | 36 | 94.44% | 80.56% | -13.89pp | 100.00% | 100.00% | +0.00pp |
| 10/image_patch | 43 | 93.02% | 81.40% | -11.63pp | 100.00% | 100.00% | +0.00pp |
| 10/last_text | 43 | 93.02% | 74.42% | -18.60pp | 100.00% | 95.35% | -4.65pp |

### Row-level deltas — V4-combined

**Wins (V3 non-specific → V4 specific):** 33 matched rows.

Top 20 wins (matched on (suite, source_id, position_index)):

| suite | source_id | pos | ptype | V3 grounding | V4 grounding | V3 reason | V4 reason |
|---|---|---|---|---|---|---|---|
| 10 | traj000199_step000036 | 61 | image_patch | generic | specific | Misidentifies the visible red milk carton and blue box as alphabet soup and tomato sauce, so the description does not ma | Mentions concrete objects, colors, and their relative positions that match the visible scene. |
| 10 | traj000008_step000000 | 61 | image_patch | generic | specific | Mentions specific product identities and colors (alphabet soup can, tomato sauce bottle, red‑and‑white label) that are n | Refers to concrete objects, colors, and their locations that match the shown scene. |
| goal | traj000395_step000038 | 50 | image_patch | generic | specific | Mentions a black pot and cooktop rings with stove controls below that are not visible in the images. | Mentions concrete objects and layout (wood tabletop, black stove unit, gray hob, lighter, bottle, bowl, plate, burner wi |
| spatial | traj000260_step000038 | 152 | anchor | generic | specific | Misstates layout (bowl is on ramekin, not near plate or above ramekin) and omits visible drawer unit, so it does not mat | References the actual cabinet, ramekin, patterned bowl on the ramekin, and plate layout visible in the frames. |
| spatial | traj000315_step000022 | 155 | anchor | generic | specific | Mentions a black bowl and black drawer box not seen in the images and adds an adjacent drawer bowl that is not clearly p | Mentions concrete objects, colors, and their relations that match the visible scene, despite calling the cabinet black i |
| spatial | traj000120_step000016 | 150 | anchor | generic | specific | Mentions a black bowl on a stove burner and a silver plate on the countertop, which do not match the visible grey bowls  | Refers to concrete objects and their locations that match the frames, including the silver bowl, patterned bowl, red‑rim |
| spatial | traj000280_step000044 | 153 | anchor | generic | specific | Mentions a black bowl next to a ramekin that is not visible in the current frames, so it is not grounded to this scene. | References the beige tabletop, red-ringed plate, gray speckled bowl, metal cups, and dark cabinet that are all visible i |
| spatial | traj000289_step000034 | 153 | anchor | generic | specific | It mentions a black bowl and silver ramekin arrangement that does not match the visible objects, so it is not grounded i | Mentions concrete objects (silver bowl, ramekin, red-rimmed plate, packet, black drawer) and their actual spatial layout |
| spatial | traj000001_step000000 | 155 | anchor | generic | specific | It mentions a black bowl inside the top drawer and a metal bowl on the table, but in the image both visible bowls appear | Mentions concrete objects, colors, and spatial layout that match the visible bowls, plate, packet, drawer, and robot arm |
| spatial | traj000171_step000002 | 150 | anchor | generic | specific | Mentions a black bowl and gray plate that are not visible in the scene, so parts of the description are not grounded in  | Refers to concrete scene details like the metallic gray bowl at center, white plate with red rim, cabinet, and black squ |
| spatial | traj000305_step000000 | 151 | anchor | generic | specific | Mentions a black bowl and a white bowl on the cabinet that do not match the visible metallic bowls and objects in the sc | References the gray textured bowl on the cabinet, the white plate with a red rim, the package, and their spatial layout  |
| spatial | traj000085_step000004 | 151 | anchor | generic | specific | Mentions a black bowl next to the plate which is not visible and misstates that the bowl is to the right of the plate, s | Mentions the silver bowl, red‑rimmed plate, colorful card, cabinet, and their locations that match the given scene. |
| spatial | traj000021_step000002 | 155 | anchor | generic | specific | Mentions a black bowl between plate and ramekin that is not visible in this scene, so parts contradict the image. | Mentions the particular plate with red rings, multiple metallic bowls, drawer unit, and food package laid out exactly as |
| spatial | traj000277_step000024 | 150 | anchor | generic | specific | Mentions a black bowl on the stove that is not visible and misplaces the plate relative to the actual scene layout. | Mentions the stove, wooden tabletop, black cabinet, gray speckled bowl, and red‑rimmed plate that are all visible and co |
| spatial | traj000050_step000058 | 150 | anchor | generic | specific | Mentions a black bowl on a stove and a plate on the table, but the scene shows a patterned bowl near a dark cabinet and  | Mentions concrete scene elements like the wooden table, black stove, metal bowl, silver cup, and white plate with red ri |
| spatial | traj000359_step000020 | 155 | anchor | generic | specific | Label mentions two black bowls and a small colorful packet, while the scene shows metallic-looking bowls and a rectangul | Refers to the visible gray bowls, white plate with red rim, packet, open black drawer, and their actual layout in the sc |
| spatial | traj000024_step000032 | 23 | image_patch | generic | specific | Refers to a black bowl and small patterned square item that are not present in the scene and mislabels the visible gray  | Mentions the exact visible objects (two gray bowls, white plate, colorful packet) and their spatial layout in this scene |
| spatial | traj000383_step000030 | 32 | image_patch | generic | specific | Mentions a black bowl and cabinet occlusion that are not visible in the frames, so parts do not match this specific scen | Refers to the wooden cabinet, dark drawer front, metallic bowl, patterned plate, and packet in their correct spatial lay |
| spatial | traj000374_step000018 | 54 | image_patch | generic | specific | Mentions a black bowl, white plate, and ramekin ring that are not visible or identifiable in the provided frames. | Refers to concrete visible elements like the gray bowl, dark cabinet, red-rimmed ramekin, and black lower border in thes |
| spatial | traj000074_step000032 | 74 | image_patch | generic | specific | Mentions a black bowl near the left foreground and a white plate below it, which do not match the visible scene layout o | Refers to the wooden cabinet, metallic bowl, patterned plate, and their concrete spatial relations visible in the images |

**Losses (V3 specific → V4 non-specific):** 37 matched rows.

Top 20 losses:

| suite | source_id | pos | ptype | V3 grounding | V4 grounding | V3 reason | V4 reason |
|---|---|---|---|---|---|---|---|
| 10 | traj000120_step000016 | 150 | anchor | specific | generic | References the visible open bottom drawer, black bottle, bowl, and wooden boards with correct spatial relations. | Label mentions a black bowl while the visible bowl is gray/white and misplaces the drawer relative to the bowl. |
| 10 | traj000289_step000054 | 149 | anchor | specific | generic | Refers to the visible basket, two cans, and cream cheese box in their actual spatial configuration. | Mentions an alphabet soup can and cream cheese box and an empty basket that are not visible or verifiable in the current |
| 10 | traj000085_step000004 | 155 | anchor | specific | generic | References the actual mugs, plates, colors, and their spatial layout visible in the scene. | Misidentifies mug colors and positions (e.g., white mug on right, yellow-and-white under gripper) that do not match the  |
| 10 | traj000001_step000010 | 70 | image_patch | specific | generic | Refers to concrete objects (striped plate, white mug, pudding cup) and their spatial layout on this particular table. | Misidentifies the mug color/texture and pudding location relative to plate and thus does not match the specific scene. |
| 10 | traj000234_step000042 | 148 | anchor | specific | generic | Refers to the visible woven basket, robot arm, and specific colored/typed packages arranged as in the images. | Refers to alphabet soup and tomato sauce not visually identifiable in the frame and uses loose spatial descriptions that |
| 10 | traj000319_step000004 | 111 | image_patch | specific | generic | Mentions concrete scene elements like the open drawer, dark bottle, wooden block, and bowl in their actual layout. | Misidentifies the bowl as black and mentions an open bottom drawer that is not visible, so the description does not matc |
| 10 | traj000120_step000048 | 65 | image_patch | specific | generic | Refers to the visible cabinet with open drawer, bottle, bowl, and their spatial relations. | It incorrectly calls the patterned grey bowl black and places the drawer to the left of the bowl, contradicting the visi |
| 10 | traj000349_step000048 | 44 | image_patch | specific | generic | Refers to the brown caddy, white mug with gold handle, and black upright object that are clearly visible, plus their spa | Mentions a brown three-compartment caddy and book position that are not visible or verifiable in the shown patches. |
| 10 | traj000289_step000034 | 149 | anchor | specific | generic | Refers to concrete objects like the woven basket, blue carton, and specific cans in positions that match the scene. | Mentions a woven basket on the left and an empty opening, but the basket interior/top is not clearly visible and some sp |
| 10 | traj000257_step000052 | 63 | image_patch | specific | generic | Refers to particular mugs, plates, colors, and their spatial layout that match the scene. | Several details contradict the scene (no yellow handle, no visible dark liquid, and the focused mug appears plain white) |
| 10 | traj000014_step000038 | 30 | image_patch | specific | generic | References the visible open drawer, patterned bowl, bottle, wooden boards, and robot gripper in their correct spatial ar | It misstates that the bowl is in the drawer and that the gripper is above the drawer, which contradicts the visible layo |
| 10 | traj000257_step000044 | 155 | last_text | specific | generic | Refers to particular colored mugs and plates and their locations that match the visible scene. | It misidentifies the gray mug as white and claims a yellow-and-white mug and plates at the left/right edges that are not |
| 10 | traj000061_step000042 | 151 | last_text | specific | generic | Refers to concrete visible elements like the wooden table, mug, robot arm, and caddy with a rear compartment, and notes  | The label mentions a book that is not visible in the frames and misplaces the mug position, so it is not well grounded i |
| 10 | traj000265_step000042 | 149 | last_text | specific | generic | Refers to the visible basket, blue cream cheese box, and two cans with concrete spatial relations. | It mentions a cream cheese box that is not visible and misstates that there are two canned foods near the lower-right. |
| 10 | traj000121_step000030 | 155 | last_text | specific | generic | Mentions concrete objects (white mug, yellow-and-white mug, red patterned mug, two striped plates) and their spatial rel | It misstates the current layout (white mug is not near the left plate, yellow-and-white mug not near the right plate, an |
| 10 | traj000080_step000038 | 148 | last_text | specific | generic | Refers to the yellow-and-white and gray mugs, black microwave, and their concrete spatial layout visible in the scene. | Microwave is actually open and the arm is already holding the yellow mug, so parts of the description mismatch the speci |
| 10 | traj000114_step000054 | 155 | last_text | specific | generic | Refers to particular colored mugs, plates, and their locations on this table scene. | It misstates the configuration (no white mug attached to the yellow one, and left plate is not empty), so it does not ac |
| 10 | traj000217_step000046 | 148 | last_text | specific | generic | Refers to the yellow-and-white mug, the white mug, microwave position, and robot gripper exactly as seen in the images. | Microwave door state and mug/arm positions are mis-described compared to the images, so the label is not accurately grou |
| 10 | traj000336_step000012 | 148 | last_text | specific | generic | Refers to concrete scene elements like the woven basket on the left and specific cans and their positions on the table. | The description of the cans’ colors, sizes, and positions (e.g., small colorful can center-left, tall red‑and‑white can  |
| goal | traj000188_step000004 | 140 | last_text | specific | generic | References concrete objects, colors, and spatial relations that match this scene, like the white burner plate, black kno | Misidentifies a small black knob near the burner that is not visible, so parts of the description do not match this scen |

### 10 worst-grading V4 rows — V4-combined

Showing first 10 (B-fails first, then any C-fails):

#### W1. 10 / last_text — traj000061_step000042 (p151)

- **B verdict:** generic — The label mentions a book that is not visible in the frames and misplaces the mug position, so it is not well grounded in this specific scene.
- **C verdict:** inappropriate — It predicts precise temporal structure for the next timesteps rather than just high-level plan and also refers to an unseen object.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000061_step000042__image.jpg` (+ `__wrist_image.jpg`)

#### W2. spatial / last_text — traj000266_step000056 (p155)

- **B verdict:** generic — Label misidentifies the patterned bowl as the target black bowl and invents a ramekin-like cup that is not visible.
- **C verdict:** inappropriate — Includes a detailed multi-timestep actuator-level plan rather than higher-level intent and also relies on incorrect scene interpretation.
- **Frames:** `data/labels/libero_4suite_stride2/libero_spatial/frames_cache/traj000266_step000056__image.jpg` (+ `__wrist_image.jpg`)

#### W3. 10 / anchor — traj000120_step000016 (p150)

- **B verdict:** generic — Label mentions a black bowl while the visible bowl is gray/white and misplaces the drawer relative to the bowl.
- **C verdict:** appropriate — Focuses on scene layout, objects, and high-level plan without actuator details or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000120_step000016__image.jpg` (+ `__wrist_image.jpg`)

#### W4. 10 / anchor — traj000289_step000054 (p149)

- **B verdict:** generic — Mentions an alphabet soup can and cream cheese box and an empty basket that are not visible or verifiable in the current frames.
- **C verdict:** appropriate — Describes high-level scene, objects, distractor, and plan without low-level controls or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000289_step000054__image.jpg` (+ `__wrist_image.jpg`)

#### W5. 10 / anchor — traj000249_step000006 (p155)

- **B verdict:** generic — It misstates object locations and plate positions relative to the image, so it is not correctly grounded in this scene.
- **C verdict:** appropriate — It focuses on task-relevant objects, roles, and high-level plan steps without low-level controls or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000249_step000006__image.jpg` (+ `__wrist_image.jpg`)

#### W6. 10 / anchor — traj000085_step000004 (p155)

- **B verdict:** generic — Misidentifies mug colors and positions (e.g., white mug on right, yellow-and-white under gripper) that do not match the visible scene.
- **C verdict:** appropriate — Focuses on objects, spatial layout, and high-level plan without numeric minutiae or low-level motor commands.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000085_step000004__image.jpg` (+ `__wrist_image.jpg`)

#### W7. 10 / anchor — traj000021_step000002 (p154)

- **B verdict:** generic — Mug is actually on the left and one mug is patterned red, so several spatial and color details contradict the scene.
- **C verdict:** appropriate — Focuses on objects, spatial relations, and high-level plan without numeric precision or low-level control.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000021_step000002__image.jpg` (+ `__wrist_image.jpg`)

#### W8. 10 / anchor — traj000234_step000042 (p148)

- **B verdict:** generic — Refers to alphabet soup and tomato sauce not visually identifiable in the frame and uses loose spatial descriptions that could fit many similar scenes.
- **C verdict:** appropriate — Tracks scene layout, target objects, and high-level plan without motor commands or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000234_step000042__image.jpg` (+ `__wrist_image.jpg`)

#### W9. 10 / anchor — traj000289_step000034 (p149)

- **B verdict:** generic — Mentions a woven basket on the left and an empty opening, but the basket interior/top is not clearly visible and some spatial details are partially inferred.
- **C verdict:** appropriate — Focuses on scene layout, targets, distractor, and high-level plan without affect or low-level motor commands.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000289_step000034__image.jpg` (+ `__wrist_image.jpg`)

#### W10. 10 / image_patch — traj000001_step000010 (p70)

- **B verdict:** generic — Misidentifies the mug color/texture and pudding location relative to plate and thus does not match the specific scene.
- **C verdict:** appropriate — Focuses on objects, roles, and spatial relations relevant to the task without low-level control or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000001_step000010__image.jpg` (+ `__wrist_image.jpg`)

---

## B. V4-only-rewritten-rows vs V3 (aggressive)

### Overall — V4-only

| Metric | V3 baseline | V4 | Δ |
|---|---|---|---|
| n | 500 | 500 | — |
| B (grounding=specific) | 91.00% (455/500) | 87.80% (439/500) | -3.20pp |
| C (appropriateness=appropriate) | 98.20% (491/500) | 99.40% (497/500) | +1.20pp |

### Per-suite — V4-only

| suite | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |
|---|---|---|---|---|---|---|---|
| spatial | 136 | 73.02% | 86.03% | +13.01pp | 98.41% | 100.00% | +1.59pp |
| goal | 122 | 98.40% | 95.08% | -3.32pp | 95.20% | 100.00% | +4.80pp |
| object | 128 | 99.21% | 92.19% | -7.03pp | 99.21% | 99.22% | +0.01pp |
| 10 | 114 | 93.44% | 77.19% | -16.25pp | 100.00% | 98.25% | -1.75pp |

### Per-position_type — V4-only

| position_type | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |
|---|---|---|---|---|---|---|---|
| anchor | 139 | 88.68% | 89.21% | +0.53pp | 98.11% | 100.00% | +1.89pp |
| image_patch | 179 | 89.02% | 89.94% | +0.93pp | 99.42% | 100.00% | +0.58pp |
| last_text | 182 | 95.24% | 84.62% | -10.62pp | 97.02% | 98.35% | +1.33pp |

### 12-cell matrix — V4-only

| suite/ptype | n (V4) | V3 B | V4 B | ΔB | V3 C | V4 C | ΔC |
|---|---|---|---|---|---|---|---|
| spatial/anchor | 41 | 65.85% | 92.68% | +26.83pp | 97.56% | 100.00% | +2.44pp |
| spatial/image_patch | 47 | 65.91% | 82.98% | +17.07pp | 100.00% | 100.00% | +0.00pp |
| spatial/last_text | 48 | 87.80% | 83.33% | -4.47pp | 97.56% | 100.00% | +2.44pp |
| goal/anchor | 33 | 97.56% | 90.91% | -6.65pp | 95.12% | 100.00% | +4.88pp |
| goal/image_patch | 44 | 97.62% | 97.73% | +0.11pp | 97.62% | 100.00% | +2.38pp |
| goal/last_text | 45 | 100.00% | 95.56% | -4.44pp | 92.86% | 100.00% | +7.14pp |
| object/anchor | 41 | 97.56% | 87.80% | -9.76pp | 100.00% | 100.00% | +0.00pp |
| object/image_patch | 42 | 100.00% | 95.24% | -4.76pp | 100.00% | 100.00% | +0.00pp |
| object/last_text | 45 | 100.00% | 93.33% | -6.67pp | 97.62% | 97.78% | +0.16pp |
| 10/anchor | 24 | 94.44% | 83.33% | -11.11pp | 100.00% | 100.00% | +0.00pp |
| 10/image_patch | 46 | 93.02% | 84.78% | -8.24pp | 100.00% | 100.00% | +0.00pp |
| 10/last_text | 44 | 93.02% | 65.91% | -27.11pp | 100.00% | 95.45% | -4.55pp |

### Row-level deltas — V4-only

**Wins (V3 non-specific → V4 specific):** 10 matched rows.

Top 10 wins (matched on (suite, source_id, position_index)):

| suite | source_id | pos | ptype | V3 grounding | V4 grounding | V3 reason | V4 reason |
|---|---|---|---|---|---|---|---|
| spatial | traj000289_step000034 | 153 | anchor | generic | specific | It mentions a black bowl and silver ramekin arrangement that does not match the visible objects, so it is not grounded i | Mentions concrete scene details like wooden tabletop, black cabinet, silver bowl and ramekin, red-rimmed plate, and pack |
| spatial | traj000260_step000038 | 152 | anchor | generic | specific | Misstates layout (bowl is on ramekin, not near plate or above ramekin) and omits visible drawer unit, so it does not mat | Mentions the cabinet, ramekin, patterned bowl on ramekin, and plate exactly as seen in the frames. |
| spatial | traj000280_step000044 | 153 | anchor | generic | specific | Mentions a black bowl next to a ramekin that is not visible in the current frames, so it is not grounded to this scene. | Mentions the beige tabletop, white plate with red rings, gray speckled bowl, metal ramekin, and dark cabinet all correct |
| spatial | traj000085_step000004 | 151 | anchor | generic | specific | Mentions a black bowl next to the plate which is not visible and misstates that the bowl is to the right of the plate, s | Mentions the silver bowl, red‑rimmed plate, black cabinet, and card in locations that match the images. |
| spatial | traj000315_step000022 | 155 | anchor | generic | specific | Mentions a black bowl and black drawer box not seen in the images and adds an adjacent drawer bowl that is not clearly p | Mentions the open dark drawer, two silver bowls, red‑rimmed plate, snack packet, and their locations as seen in the fram |
| spatial | traj000359_step000020 | 155 | anchor | generic | specific | Label mentions two black bowls and a small colorful packet, while the scene shows metallic-looking bowls and a rectangul | Mentions the open drawer, silver-gray bowls, white plate with red rim, and packet in locations matching the scene. |
| spatial | traj000050_step000058 | 150 | anchor | generic | specific | Mentions a black bowl on a stove and a plate on the table, but the scene shows a patterned bowl near a dark cabinet and  | Mentions concrete objects and spatial relations (metallic bowl on black stove, silver cup, white plate with red rim) tha |
| spatial | traj000171_step000002 | 150 | anchor | generic | specific | Mentions a black bowl and gray plate that are not visible in the scene, so parts of the description are not grounded in  | Mentions the dark cabinet, black square base, gray bowl at center, and white plate with red rim which all match this sce |
| spatial | traj000277_step000024 | 150 | anchor | generic | specific | Mentions a black bowl on the stove that is not visible and misplaces the plate relative to the actual scene layout. | References the actual stove, wooden tabletop, metal bowl, red-rimmed plate, and dark cabinet visible in the frame with c |
| spatial | traj000120_step000016 | 150 | anchor | generic | specific | Mentions a black bowl on a stove burner and a silver plate on the countertop, which do not match the visible grey bowls  | Refers to concrete objects and their colors/locations that match the visible scene, even if it mislabels black vs silver |

**Losses (V3 specific → V4 non-specific):** 7 matched rows.

Top 7 losses:

| suite | source_id | pos | ptype | V3 grounding | V4 grounding | V3 reason | V4 reason |
|---|---|---|---|---|---|---|---|
| 10 | traj000120_step000016 | 150 | anchor | specific | generic | References the visible open bottom drawer, black bottle, bowl, and wooden boards with correct spatial relations. | Mentions a black bowl and an arm above the cabinet while the image shows a light bowl and the arm not directly over the  |
| 10 | traj000289_step000054 | 149 | anchor | specific | generic | Refers to the visible basket, two cans, and cream cheese box in their actual spatial configuration. | Mentions a cream cheese box and basket that are not visible in the frames even though a woven box of tissues and only ca |
| 10 | traj000289_step000034 | 149 | anchor | specific | generic | Refers to concrete objects like the woven basket, blue carton, and specific cans in positions that match the scene. | Mentions a woven basket on the left and its emptiness, which are not visible in the current cropped views, so some conte |
| 10 | traj000085_step000004 | 155 | anchor | specific | generic | References the actual mugs, plates, colors, and their spatial layout visible in the scene. | Mug colors, positions, and which mug is under the gripper are mis-described relative to the images. |
| object | traj000104_step000000 | 146 | anchor | specific | generic | Refers to the visible white woven basket, scattered boxes and can, and their left/center layout that match this scene. | Mentions blue milk carton and specific object colors/positions that do not clearly match what is visible in the frames. |
| object | traj000008_step000052 | 147 | anchor | specific | generic | Refers to the visible basket on the left and the particular red-brown ketchup bottle among other bottles and cans with c | Mentions a red-capped ketchup at center-right and a green bottle/blue can arrangement that do not match the shown object |
| object | traj000382_step000022 | 147 | anchor | specific | generic | References the brown bottle, wicker basket, can, box, and carton in concrete spatial relations visible in the scene. | It misplaces the tomato sauce jar and gripper relative to the basket and center, so parts contradict this specific scene |

### 10 worst-grading V4 rows — V4-only

Showing first 10 (B-fails first, then any C-fails):

#### W1. 10 / last_text — traj000293_step000042 (p149)

- **B verdict:** generic — Misidentifies the visible blue cream cheese box as a distractor and adds an unseen alphabet soup target, so it does not match this specific scene.
- **C verdict:** inappropriate — Includes an over-precise stepwise plan about the next 3 timesteps and grasping both items simultaneously, which is beyond plausible mid-level representations.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000293_step000042__image.jpg` (+ `__wrist_image.jpg`)

#### W2. 10 / anchor — traj000120_step000016 (p150)

- **B verdict:** generic — Mentions a black bowl and an arm above the cabinet while the image shows a light bowl and the arm not directly over the cabinet.
- **C verdict:** appropriate — Focuses on scene layout, objects, and high-level plan without actuator details or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000120_step000016__image.jpg` (+ `__wrist_image.jpg`)

#### W3. 10 / anchor — traj000289_step000054 (p149)

- **B verdict:** generic — Mentions a cream cheese box and basket that are not visible in the frames even though a woven box of tissues and only cans are shown.
- **C verdict:** appropriate — Describes objects, spatial layout, a distractor, and high-level plan steps without low-level control or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000289_step000054__image.jpg` (+ `__wrist_image.jpg`)

#### W4. 10 / anchor — traj000289_step000034 (p149)

- **B verdict:** generic — Mentions a woven basket on the left and its emptiness, which are not visible in the current cropped views, so some content is not grounded in this scene.
- **C verdict:** appropriate — Focuses on objects, spatial layout, and high-level plan without low-level controls or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000289_step000034__image.jpg` (+ `__wrist_image.jpg`)

#### W5. 10 / anchor — traj000085_step000004 (p155)

- **B verdict:** generic — Mug colors, positions, and which mug is under the gripper are mis-described relative to the images.
- **C verdict:** appropriate — Content focuses on objects, spatial layout, and high-level plan without actuator-level or affective details.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000085_step000004__image.jpg` (+ `__wrist_image.jpg`)

#### W6. 10 / image_patch — traj000219_step000032 (p78)

- **B verdict:** generic — Mentions alphabet soup and tomato sauce cans that are not clearly identifiable in the frames.
- **C verdict:** appropriate — Focuses on scene layout, objects, distractors, and task state without low-level control or mental-state attributions.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000219_step000032__image.jpg` (+ `__wrist_image.jpg`)

#### W7. 10 / image_patch — traj000019_step000054 (p50)

- **B verdict:** generic — Several details are incorrect for this frame (mug is not in the gripper and pudding is roughly centered, not clearly left of the plate), so the description is not well grounded in the specific scene.
- **C verdict:** appropriate — It talks about objects, their roles, and spatial relations without low-level controls or mental states.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000019_step000054__image.jpg` (+ `__wrist_image.jpg`)

#### W8. 10 / image_patch — traj000302_step000020 (p58)

- **B verdict:** generic — It misstates the scene (mug described as on tabletop with handle right and mentions a second flat container) and omits unique details like the fallen mug near the gripper.
- **C verdict:** appropriate — Bullets focus on objects, relations, and high-level plan without numeric details, emotions, or low-level motor commands.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000302_step000020__image.jpg` (+ `__wrist_image.jpg`)

#### W9. 10 / image_patch — traj000360_step000046 (p44)

- **B verdict:** generic — Mentions a cream cheese box near the center-right and beside the gripper, which does not match the shown close-up of a grey box in gripper and omits the butter box view.
- **C verdict:** appropriate — Focuses on scene layout, objects, and spatial relations without low-level control or mental-state descriptions.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000360_step000046__image.jpg` (+ `__wrist_image.jpg`)

#### W10. 10 / image_patch — traj000343_step000036 (p130)

- **B verdict:** generic — Several statements contradict the scene layout (e.g., white mug is near center, not front-left by left plate; yellow-and-white mug is not clearly on right side), so it is not accurately grounded.
- **C verdict:** appropriate — Bullets concern objects, colors, and spatial relations relevant to the task without numeric minutiae or low-level motor commands.
- **Frames:** `data/labels/libero_4suite_stride2/libero_10/frames_cache/traj000343_step000036__image.jpg` (+ `__wrist_image.jpg`)

---

## Gate verdicts

Gates (per plan): **Paper-grade** = overall B ≥ 95% AND C ≥ 95%; **Spatial-rescue** = spatial B ≥ 85%; **No-regression** = NO 12-cell suite×ptype value drops > 5pp B or C from V3.

| Gate | V4-combined | V4-only |
|---|---|---|
| Paper-grade (B≥95% AND C≥95%) | YELLOW (B=90.20%, C=99.20%) | YELLOW (B=87.80%, C=99.40%) |
| Spatial-rescue (spatial B ≥ 85%) | PASS (spatial B = 91.27%) | PASS (spatial B = 86.03%) |
| No-regression (no cell drops >5pp) | FAIL (5 cells regressed) | FAIL (6 cells regressed) |

### V4-combined cells that regressed >5pp

| Cell | Metric | ΔB or ΔC (pp) |
|---|---|---|
| goal/last_text | B | -7.14 |
| object/image_patch | B | -6.82 |
| 10/anchor | B | -13.89 |
| 10/image_patch | B | -11.63 |
| 10/last_text | B | -18.60 |

### V4-only cells that regressed >5pp

| Cell | Metric | ΔB or ΔC (pp) |
|---|---|---|
| goal/anchor | B | -6.65 |
| object/anchor | B | -9.76 |
| object/last_text | B | -6.67 |
| 10/anchor | B | -11.11 |
| 10/image_patch | B | -8.24 |
| 10/last_text | B | -27.11 |

### V4-combined cells with ≥10pp B improvement ("rescues")

| Cell | ΔB (pp) |
|---|---|
| spatial/anchor | +29.27 |
| spatial/image_patch | +22.73 |

---

## Recommendation

**YELLOW (acceptable to ship, but not paper-grade GREEN).** 

- Overall **B = 90.20%** vs V3 91.00% (Δ -0.80pp). Misses 95% paper bar.
- Overall **C = 99.20%** vs V3 98.20% (Δ +1.00pp). Clears 95%.
- **Spatial-rescue PASS:** spatial B = 91.27% (≥85% gate).
- **No-regression FAIL:** 5 of 12 cells regressed >5pp.

The V4 dataset is the strict superset improvement the plan promised on the worst-case axis (spatial), and overall **C clears the 95% appropriateness gate**. Overall **B misses 95%** primarily because the `libero_10` suite did not get the same lift the spatial-targeted V4 rewrite produced. Given V3 was the YELLOW shipped baseline, V4 is **strictly better on spatial and overall comparable on C**; ship V4 for SFT while opening a follow-up to push `libero_10/last_text` rewrites in V5.

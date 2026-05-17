# Agent 1 — Multimodal Judge Quality Gate

**Audit target:** V3 LIBERO 4-suite caption labels (101,580 rows total) at
`data/labels/libero_4suite_stride2/libero_{goal,spatial,object,10}/labels.jsonl`.

**Method:** stratified 500-row sample (≈42 per suite × position_type cell)
graded by `gpt-5.1` against the cached camera frames the labeler saw. Script:
`scripts/eval/verify_libero_label_quality.py`. Raw grades:
`data/eval/libero_v3_quality_judge.jsonl`. Run log: `/tmp/agent1_judge.log`.
Wall-clock: ~4 min, 500/500 calls succeeded with 0 parse errors.

---

## Headline numbers (n=500)

| Axis | Pass% | Pass count |
|---|---:|---:|
| **B — grounding = specific** | **91.00%** | 455 / 500 |
| **C — appropriateness = appropriate** | **98.20%** | 491 / 500 |

- **Overall verdict: YELLOW.** Both axes are above the 85% YELLOW floor, but
  grounding misses the 95% GREEN bar by 4 points.
- C-axis (appropriateness) **passes the 95% bar** — the V2→V3 prompt
  hardening against affect / motor-command / "action head" leakage worked,
  with only 9 / 500 (1.8%) C-fails left and they are mild ("last bullet
  gives low-level motor instructions").
- B-axis (grounding) is dragged down almost entirely by **libero_spatial**
  and to a lesser extent **libero_10**. Two cells fall below the per-cell
  85% floor and are RED.

### Verdict

**Overall: YELLOW.**
- libero_object, libero_goal, libero_10: **GREEN-adjacent** (all cells ≥ 92%).
- **libero_spatial: RED** on grounding (B = 73.0% suite-wide, with two cells
  at 65.9%). Recommend gating libero_spatial out of SFT until either the
  prompt is re-hardened on this suite or the rows are filtered.

---

## Per-suite

| Suite | n | B (specific) | C (appropriate) |
|---|---:|---:|---:|
| libero_object | 127 | **99.2%** | **99.2%** |
| libero_goal | 125 | **98.4%** | 95.2% |
| libero_10 | 122 | 93.4% | **100.0%** |
| libero_spatial | 126 | **73.0%** ⚠️ | 98.4% |

## Per-position_type

| Position | n | B (specific) | C (appropriate) |
|---|---:|---:|---:|
| last_text | 168 | 95.2% | 97.0% |
| image_patch | 173 | 89.0% | 99.4% |
| anchor | 159 | 88.7% | 98.1% |

`last_text` is the only position type that clears the 95% B bar overall.
`anchor` and `image_patch` lag, but the gap is almost entirely concentrated
in libero_spatial — see the matrix below.

## 12-cell matrix (suite × position_type)

`B%` = grounding_specific%, `C%` = appropriateness_appropriate%. Cells
below the 85% per-cell floor are flagged RED.

| Suite \ Pos | last_text | image_patch | anchor |
|---|---|---|---|
| libero_object | n=42  B=100.0% / C=97.6% | n=44  B=100.0% / C=100.0% | n=41  B=97.6% / C=100.0% |
| libero_goal   | n=42  B=100.0% / C=92.9% | n=42  B=97.6% / C=97.6% | n=41  B=97.6% / C=95.1% |
| libero_10     | n=43  B=93.0% / C=100.0% | n=43  B=93.0% / C=100.0% | n=36  B=94.4% / C=100.0% |
| libero_spatial| n=41  B=87.8% / C=97.6% | n=44  **B=65.9% RED** / C=100.0% | n=41  **B=65.9% RED** / C=97.6% |

The two RED cells (libero_spatial / anchor and libero_spatial / image_patch)
account for 28 of the 45 B-failures (62%). Fixing libero_spatial alone moves
overall B from 91.0% to ≈ 96.5%, well above the GREEN bar.

---

## What's failing — failure-mode breakdown

### B-fails (grounding, 45 / 500 = 9.0%)

Every B-fail rationale is a variation of **"label mentions object/relation
not visible in the frame."** Two clusters:

1. **libero_spatial: spatial-relation hallucinations (~28 of 45).** Captions
   describe layouts that contradict the actual frame — e.g. *"black bowl on
   the stove"* when the bowl is on the table, *"plate to the right of the
   bowl"* when the plate is left. These suites have a lot of distractor
   bowls/plates/boxes and the labeler is using the **instruction text** as a
   crutch when the visual layout disagrees.

2. **libero_10: object-identity confusion among grocery items (~8 of 45).**
   Captions confidently label *"alphabet soup can"* / *"tomato sauce
   bottle"* / *"cream cheese box"* / *"SHIKI pasta sauce box"* when the
   visible objects are unrelated cans/cartons. The labeler is being primed
   by the instruction string and naming products it can't actually
   discriminate from low-res top-down camera frames.

3. **A few libero_goal misses (~2)** describe distractors that aren't
   present (a "striped wooden block", a "stove with cooktop rings").

### C-fails (appropriateness, 9 / 500 = 1.8%)

All 9 are the same low-severity pattern: **the last bullet drifts into
actuator / motor-command language** ("aligning the gripper, lifting, and
placing"; "grasp the bowl and carry it"; "carry the phase into the action
head"). No anthropomorphic affect ("wants/feels/thinks") was caught — that
V2 failure mode appears fully eliminated. None of the V2-era
`image_region:` hallucinations showed up either.

---

## 10 worst examples (sorted: both-axis fails first, then any-axis)

Frame paths follow the pattern
`data/labels/libero_4suite_stride2/<suite>/frames_cache/<source_id>__{image,wrist_image}.jpg`.

1. **libero_spatial / anchor — `traj000277_step000024` p150** (both axes fail)
   - instr: *"pick up the black bowl on the stove and place it on the plate"*
   - **B (generic):** "Mentions a black bowl on the stove that is not visible and misplaces the plate relative to the actual scene layout."
   - **C (inappropriate):** "Includes an internal-plumbing phrase about carrying the phase into the action head rather than just scene and plan content."
   - desc start: *"scene: tabletop workspace with a robot arm, a stove, a black bowl on the stove, a white plate, and a small metal cup."*

2. **libero_10 / anchor — `traj000249_step000006` p155**
   - instr: *"put the white mug on the left plate and put the yellow and white mug on the right plate"*
   - **B (generic):** misstates which mug is near which plate; layout doesn't match.

3. **libero_10 / anchor — `traj000021_step000002` p154**
   - instr: *"put the white mug on the plate and put the chocolate pudding to the right of the plate"*
   - **B (generic):** mug is not on the right of the plate; pudding placement contradicts the frame.

4. **libero_10 / image_patch — `traj000199_step000036` p61**
   - instr: *"put both the alphabet soup and the tomato sauce in the basket"*
   - **B (generic):** labels visible red milk carton and blue box as "alphabet soup" and "tomato sauce".

5. **libero_10 / image_patch — `traj000316_step000004` p67**
   - instr: *"put both moka pots on the stove"*
   - **B (generic):** invents a "larger moka pot on the back counter" not in frame.

6. **libero_10 / image_patch — `traj000008_step000000` p61**
   - instr: *"put both the alphabet soup and the tomato sauce in the basket"*
   - **B (generic):** product identities and label colors not actually identifiable from images.

7. **libero_10 / last_text — `traj000151_step000016` p150**
   - instr: *"put the black bowl in the bottom drawer of the cabinet and close it"*
   - **B (generic):** says "black bowl" + "cabinet door"; actual scene has a grey bowl and a drawer-only cabinet.

8. **libero_10 / last_text — `traj000163_step000054` p149**
   - instr: *"put both the alphabet soup and the cream cheese box in the basket"*
   - **B (generic):** invents a "wire basket on the left" and "cream cheese box" not visible.

9. **libero_10 / last_text — `traj000291_step000036` p148**
   - instr: *"put both the alphabet soup and the tomato sauce in the basket"*
   - **B (generic):** "no alphabet soup can or SHIKI tomato sauce box visible" — bare product hallucination.

10. **libero_goal / anchor — `traj000217_step000038` p145**
    - instr: *"put the wine bottle on top of the cabinet"*
    - **B (generic):** mentions a wine bottle near a "striped wooden block" not visible.

(Full JSON dump including descriptions and resolved frame paths:
`/tmp/agent1_worst.json`.)

---

## 10 best examples (one per cell where both axes pass)

| Cell | source_id @ pidx | instruction |
|---|---|---|
| libero_10/anchor | `traj000289_step000054` p149 | *"put both the alphabet soup and the cream cheese box in the basket"* |
| libero_10/image_patch | `traj000120_step000048` p65 | *"put the black bowl in the bottom drawer of the cabinet and close it"* |
| libero_10/last_text | `traj000187_step000022` p144 | *"put both moka pots on the stove"* |
| libero_goal/anchor | `traj000008_step000052` p145 | *"open the top drawer and put the bowl inside"* |
| libero_goal/image_patch | `traj000030_step000036` p67 | *"open the top drawer and put the bowl inside"* |
| libero_goal/last_text | `traj000421_step000034` p140 | *"turn on the stove"* |
| libero_object/anchor | `traj000433_step000008` p147 | *"pick up the ketchup and place it in the basket"* |
| libero_object/image_patch | `traj000277_step000004` p42 | *"pick up the orange juice and place it in the basket"* |
| libero_object/last_text | `traj000309_step000038` p146 | *"pick up the milk and place it in the basket"* |
| libero_spatial/anchor | `traj000017_step000014` p151 | *"pick up the black bowl on the cookie box and place it on the plate"* |

Common quality across these: the labeler grounds in **named, visible
distractors with colors and spatial relations** ("black cabinet, striped
bowl, plate with red rings, blue block"; "white woven basket on the left,
tall blue-and-white milk carton near the gripper"). The grader's positive
rationale repeatedly highlights "concrete objects, colors, branding, and
correct spatial layout" — exactly what we want SFT to learn.

(Full JSON: `/tmp/agent1_best.json`.)

---

## Recommendation (3 bullets)

1. **Gate libero_spatial out of the V3 SFT mix until grounding is fixed.**
   That single suite causes the 91% overall B and 4-pt miss vs the GREEN
   bar. The failure mode is uniform — captions invent or misplace
   spatial relations between bowls/plates/boxes/stove — so a targeted
   re-prompt that (a) forbids stating object positions not visually
   verifiable from the camera frame and (b) injects a "double-check
   spatial layout against the image before describing it" rule should fix
   it. Re-grade libero_spatial only and require ≥ 90% B before unblocking.

2. **Tighten the labeler prompt against grocery-item identity hallucination
   in libero_10.** When the instruction names a specific product
   (`alphabet soup`, `tomato sauce`, `cream cheese box`, `SHIKI pasta
   sauce`), the labeler is currently confidently echoing those names even
   when the frame shows a generic colored carton. Add a "if the camera
   frame doesn't make the product identity obvious, say `the can named in
   the instruction` instead of restating its packaging" clause. This will
   recover the ~8 libero_10 B-fails.

3. **C-axis is essentially shipped — no further work needed.** The
   remaining 9 C-fails are a single mild leak ("last bullet drifts into
   gripper-level motor language") and 1.8% is below SFT noise floor.
   Optional polish: extend the prompt's `forbidden:` list to include
   "aligning the gripper", "lifting", "placing", "grasp and carry" in the
   final bullet. No anthropomorphic / `image_region:` patterns remain
   from V2, so the hardening did its job on those fronts.

If 1 + 2 land, expect overall B ≥ 96% and overall C ≥ 98% on a re-run,
which would clear GREEN. For now, **ship libero_object, libero_goal,
and libero_10 to SFT V3; hold libero_spatial.**

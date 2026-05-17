# V3 Bullet Informativeness Audit (Agent 4)

Per-bullet grounding, redundancy, and informativeness for the V3 5-bullet captions on LIBERO 4-suite (stride-2). Scope: the five scoped bullet types `language / target / scene / spatial / plan`. Other categories (`distractor`, `motion`, `gripper`, `image_region`) are recognised by the bullet parser so they do not bleed into scoped-bullet content, but are not reported here.

**Corpus**: 101,580 labels across libero_goal=25,680, libero_spatial=25,920, libero_object=27,240, libero_10=22,740.

## 1. Length, presence, concrete-noun (per suite)

Length cells are `p10/p50/p90 (μ=mean)` whitespace tokens, including the bullet header words.

| bullet | len[libero_goal] | len[libero_spatial] | len[libero_object] | len[libero_10] | presence | concrete-noun |
|---|---|---|---|---|---|---|
| **language** | 7/10/14 (μ=10.4) | 11/14/18 (μ=14.7) | 11/14/16 (μ=13.4) | 12/15/19 (μ=15.4) |  20.2% |  98.3% |
| **target** | 12/17/21 (μ=17.0) | 13/16/22 (μ=17.4) | 13/16/22 (μ=17.2) | 14/20/37 (μ=23.4) | 100.0% |  99.8% |
| **scene** | 16/21/26 (μ=21.1) | 15/19/25 (μ=19.8) | 15/19/24 (μ=19.5) | 15/19/25 (μ=19.8) |  98.6% | 100.0% |
| **spatial** | 16/19/23 (μ=19.4) | 15/18/23 (μ=18.6) | 16/19/23 (μ=19.1) | 15/19/24 (μ=19.2) |  97.9% |  99.8% |
| **plan** | 13/16/23 (μ=17.4) | 13/17/25 (μ=18.2) | 13/15/23 (μ=16.7) | 14/18/26 (μ=19.4) |  97.1% |  98.4% |


**Presence anomaly**: the following bullets are missing from a non-trivial share of labels, suggesting the prompt template does not require them strongly enough:
- `language` present in only 20.2% of rows

Per-suite concrete-noun rate (denominator = present bullets):

| bullet | libero_goal | libero_spatial | libero_object | libero_10 |
|---|---|---|---|---|
| **language** |  94.1% |  99.6% |  99.5% |  99.1% |
| **target** |  99.9% | 100.0% |  99.8% |  99.2% |
| **scene** | 100.0% | 100.0% | 100.0% |  99.9% |
| **spatial** |  99.6% | 100.0% |  99.8% |  99.9% |
| **plan** |  96.9% |  99.0% |  99.7% |  97.8% |

## 2. Filler-bullet rate (bullet covers ≥80% of instruction content words)

Filler means the bullet's content-word set covers the instruction's content-word set almost completely — i.e., it just paraphrases the task text without adding visual or plan information. Computed only on rows where the instruction is non-empty.

| bullet | libero_goal | libero_spatial | libero_object | libero_10 | overall |
|---|---|---|---|---|---|
| **language** |  43.2% |  43.2% |  83.7% |  39.0% |  54.9% |
| **target** |   3.6% |   1.4% |   0.2% |   7.5% |   3.0% |
| **scene** |   0.0% |   0.0% |   0.0% |   1.2% |   0.3% |
| **spatial** |   0.9% |   0.1% |   0.0% |   1.7% |   0.6% |
| **plan** |  12.0% |   9.6% |   5.8% |  13.6% |  10.1% |

## 3. Cross-bullet Jaccard redundancy (mean over rows where both bullets present)

Token-set Jaccard on whitespace tokens (after lowercase, punctuation strip). Diagonals are 1.00 by definition; only upper triangle is computed and mirrored.

| | language | target | scene | spatial | plan |
|---|---|---|---|---|---|
| **language** | 1.00 | 0.18 | 0.12 | 0.17 | 0.21 |
| **target** | 0.18 | 1.00 | 0.13 | 0.17 | 0.16 |
| **scene** | 0.12 | 0.13 | 1.00 | 0.15 | 0.11 |
| **spatial** | 0.17 | 0.17 | 0.15 | 1.00 | 0.16 |
| **plan** | 0.21 | 0.16 | 0.11 | 0.16 | 1.00 |

Pairs ranked by mean Jaccard (descending):

| pair | mean Jaccard | flag (>0.4) |
|---|---|---|
| language ↔ plan | 0.214 |  |
| language ↔ target | 0.177 |  |
| language ↔ spatial | 0.174 |  |
| target ↔ spatial | 0.167 |  |
| spatial ↔ plan | 0.162 |  |
| target ↔ plan | 0.161 |  |
| scene ↔ spatial | 0.153 |  |
| target ↔ scene | 0.131 |  |
| language ↔ scene | 0.122 |  |
| scene ↔ plan | 0.109 |  |

## 4. Plan-bullet phase taxonomy (per suite)

Each plan bullet is matched against a curated keyword list; a single bullet may contribute to multiple phases (e.g., `pick-and-place phase active; reach over the bowl, then place` → `pick-and-place`, `reach`, `place`). `(other)` = no keyword matched.

| suite | plan-bullets | top phases (count, %) |
|---|---|---|
| libero_goal | 25352 | place (13530, 53%); grasp (9937, 39%); carry (9158, 36%); reach (6383, 25%); lift (4706, 19%); align (3271, 13%); (other) (2789, 11%); pick-and-place (2257, 9%); approach (539, 2%); open (271, 1%) |
| libero_spatial | 24864 | grasp (19994, 80%); carry (16978, 68%); place (14026, 56%); reach (10759, 43%); pick-and-place (6455, 26%); lift (5463, 22%); align (533, 2%); approach (246, 1%); open (49, 0%); (other) (44, 0%) |
| libero_object | 26258 | grasp (21688, 83%); carry (20997, 80%); place (14732, 56%); reach (11887, 45%); pick-and-place (6820, 26%); lift (3111, 12%); align (647, 2%); approach (386, 1%); release (249, 1%); (other) (34, 0%) |
| libero_10 | 22130 | place (16106, 73%); grasp (9213, 42%); carry (6539, 30%); pick-and-place (6240, 28%); reach (4176, 19%); align (1862, 8%); close (1321, 6%); lift (1144, 5%); (other) (634, 3%); approach (187, 1%) |

Overall plan-phase share (all suites combined):

| phase | count | share |
|---|---|---|
| grasp | 60832 | 61.7% |
| place | 58394 | 59.2% |
| carry | 53672 | 54.4% |
| reach | 33205 | 33.7% |
| pick-and-place | 21772 | 22.1% |
| lift | 14424 | 14.6% |
| align | 6313 | 6.4% |
| (other) | 3501 | 3.6% |
| approach | 1358 | 1.4% |
| close | 1321 | 1.3% |
| release | 674 | 0.7% |
| open | 384 | 0.4% |
| idle | 93 | 0.1% |
| retract | 4 | 0.0% |

## 5. Position-type sensitivity (image_patch vs last_text)

Top content unigrams per bullet for each position type. If `image_patch` bullets do their job (visually-grounded), their top n-grams should be more concrete (objects, colors, parts) than the matched `last_text` bullets.

Per-position bullet counts (summed across bullets): last_text=221494, image_patch=198120, anchor=685.

**Quantitative summary**: Jaccard overlap of top-N content unigrams between `last_text` and `image_patch` per bullet. 1.00 = identical vocabulary; high values mean the labeler is producing similar text regardless of position type.

| bullet | top-10 Jaccard | top-30 Jaccard |
|---|---|---|
| **language** | 0.33 | 0.26 |
| **target** | 0.82 | 0.76 |
| **scene** | 0.82 | 0.76 |
| **spatial** | 0.67 | 0.71 |
| **plan** | 0.67 | 0.71 |

### `language` bullet — top 10 content unigrams

| rank | last_text | image_patch | anchor |
|---|---|---|---|
| 1 | parsed (12864) | bowl (4) | place (18) |
| 2 | place (10982) | place (3) | pick (16) |
| 3 | instruction (9749) | turn (2) | basket (16) |
| 4 | pick (9156) | stove (2) | parsed (15) |
| 5 | plate (8797) | drawer (2) | plate (12) |
| 6 | basket (8302) | pick (2) | instruction (7) |
| 7 | bowl (8025) | black (2) | task (7) |
| 8 | black (5691) | table (2) | bowl (6) |
| 9 | task (5054) | center (2) | chocolate (6) |
| 10 | specifies (3641) | plate (2) | pudding (6) |

Concrete-vocab share in top-30 unigrams — last_text: 13/30 concrete; image_patch: 7/30 concrete; anchor: 13/30 concrete

### `target` bullet — top 10 content unigrams

| rank | last_text | image_patch | anchor |
|---|---|---|---|
| 1 | bowl (23196) | bowl (24051) | bowl (75) |
| 2 | object (19006) | black (18289) | black (62) |
| 3 | black (18204) | visible (15540) | object (57) |
| 4 | sits (12081) | object (15126) | plate (45) |
| 5 | plate (10907) | sits (13782) | bottle (42) |
| 6 | visible (10375) | plate (10673) | visible (40) |
| 7 | bottle (10272) | center (10581) | table (39) |
| 8 | table (9972) | bottle (10444) | center (38) |
| 9 | center (8589) | view (9742) | sits (35) |
| 10 | box (7627) | table (8730) | box (33) |

Concrete-vocab share in top-30 unigrams — last_text: 15/30 concrete; image_patch: 14/30 concrete; anchor: 15/30 concrete

### `scene` bullet — top 10 content unigrams

| rank | last_text | image_patch | anchor |
|---|---|---|---|
| 1 | tabletop (34455) | tabletop (44301) | tabletop (139) |
| 2 | robot (25474) | robot (34388) | robot (97) |
| 3 | workspace (23000) | arm (30804) | workspace (93) |
| 4 | arm (21749) | workspace (29544) | arm (91) |
| 5 | left (21437) | black (24405) | left (78) |
| 6 | black (21157) | white (20454) | black (77) |
| 7 | white (20796) | small (16596) | white (77) |
| 8 | basket (16818) | left (15908) | plate (57) |
| 9 | plate (16382) | basket (15774) | right (57) |
| 10 | cabinet (14591) | plate (15673) | basket (55) |

Concrete-vocab share in top-30 unigrams — last_text: 15/30 concrete; image_patch: 16/30 concrete; anchor: 15/30 concrete

### `spatial` bullet — top 10 content unigrams

| rank | last_text | image_patch | anchor |
|---|---|---|---|
| 1 | bowl (23897) | bowl (23850) | plate (80) |
| 2 | plate (22713) | plate (20791) | bowl (76) |
| 3 | basket (17428) | basket (17103) | basket (70) |
| 4 | left (12281) | left (13851) | left (47) |
| 5 | sits (11077) | sits (12243) | positioned (39) |
| 6 | robot (10732) | robot (11669) | robot (39) |
| 7 | positioned (9926) | positioned (10642) | bottle (36) |
| 8 | cabinet (8956) | gripper (10222) | front (35) |
| 9 | separated (8887) | open (9279) | surface (35) |
| 10 | right (8835) | cabinet (9083) | table (35) |

Concrete-vocab share in top-30 unigrams — last_text: 12/30 concrete; image_patch: 12/30 concrete; anchor: 11/30 concrete

### `plan` bullet — top 10 content unigrams

| rank | last_text | image_patch | anchor |
|---|---|---|---|
| 1 | phase (25985) | toward (32959) | phase (85) |
| 2 | bowl (21871) | grasp (20404) | toward (77) |
| 3 | active (21374) | bowl (20196) | plate (73) |
| 4 | plate (19948) | carry (18244) | carry (66) |
| 5 | grasp (19917) | phase (18141) | active (64) |
| 6 | carry (19755) | reach (17218) | basket (63) |
| 7 | toward (19320) | plate (15944) | bowl (62) |
| 8 | move (18763) | basket (15204) | grasp (60) |
| 9 | basket (17823) | carries (14643) | carries (54) |
| 10 | placement (15669) | active (14211) | action (48) |

Concrete-vocab share in top-30 unigrams — last_text: 9/30 concrete; image_patch: 10/30 concrete; anchor: 10/30 concrete

## 6. Examples

### Well-grounded (all 5 bullets present, all concrete, no filler)

- **traj000001_step000038@p143_last_text** (libero_goal) — instruction: _put the wine bottle on the rack_
    - language: instruction specifies placing the wine bottle onto the rack.
    - target: black wine bottle stands near the center of the tabletop, upright beside the bowl.
    - scene: wooden bottle rack is on the right side of the workspace, attached to the dark cabinet.
    - spatial: bottle, rack, bowl, and coiled red-and-white cord are all on the table surface within reach of the arm.
    - plan: pickup-and-place phase active; grasp the wine bottle, lift it, and align it with the rack opening.

- **traj000000_step000040@p142_last_text** (libero_goal) — instruction: _put the bowl on the plate_
    - language: command parsed as placing the bowl onto the plate.
    - scene: tabletop workspace with a white plate marked by red rings and a bowl below the robot gripper.
    - target: bowl is the object to be moved; plate is the destination surface.
    - spatial: bowl sits near the lower edge of the view, while the plate lies above it in the camera frame.
    - plan: pick-and-place phase active; reach for the bowl, lift it, then align over the plate for placement.

- **traj000001_step000014@p143_last_text** (libero_goal) — instruction: _put the wine bottle on the rack_
    - language: instruction specifies placing the wine bottle onto the rack.
    - scene: tabletop workspace with a black wine bottle, a wooden bottle rack, a gray bowl, a red cable ring, and a white hotplate.
    - target: black wine bottle standing near the center of the table below the gripper.
    - spatial: rack sits on the right side of the table, adjacent to the black cabinet, with open slats for bottle placement.
    - plan: bottle-to-rack placement phase active; move to the bottle, grasp it, and align it with the rack opening.

- **traj000000_step000012@p142_last_text** (libero_goal) — instruction: _put the bowl on the plate_
    - language: instruction parsed as putting the bowl onto the plate.
    - scene: a metal bowl sits below the robot arm beside a striped plate on the tabletop.
    - target: the gray bowl is the object to be grasped and moved.
    - spatial: the bowl is close to the plate, with the plate directly above it in the close-up view.
    - plan: pick the bowl, lift it, and place it onto the plate.

- **traj000000_step000050@p142_last_text** (libero_goal) — instruction: _put the bowl on the plate_
    - language: command parsed as putting the bowl onto the plate.
    - scene: tabletop workspace with a robotic arm, a patterned bowl, and a striped plate.
    - target: the bowl is the object to grasp, with the plate as the placement surface.
    - spatial: the bowl sits below the plate in the camera view, aligned for a lift-and-place move.
    - plan: pick the bowl first, then carry it over the plate for placement.

### Poorly-grounded (≤3 bullets, or ≥2 filler bullets, or ≤1 concrete-hit)

- **traj000001_step000026@p066_image_patch** (libero_goal) — present=3 concrete=3 filler=0 — instruction: _put the wine bottle on the rack_
    - scene: tabletop workspace with a white hotplate, a clear bowl, a plate, a black bottle, and a wooden rack structure beside a dark cabinet.
    - target: wine bottle standing upright near the center of the table, separate from the rack and ready to be moved.
    - distractor: black knob-like object on the left, small blue rectangular item, and the gray bowl are unrelated objects on the surface.
    - spatial: rack opening and slatted wooden rails are adjacent to the cabinet edge, creating the destination area for the bottle.
    - target: bottle and nearby placement area are visible in this patch; the action head carries forward the bottle-to-rack placement relation.

- **traj000003_step000012@p145_last_text** (libero_goal) — present=5 concrete=5 filler=2 — instruction: _open the top drawer and put the bowl inside_
    - language: open the top drawer and put the bowl inside.
    - scene: tabletop workspace with a black cabinet drawer unit, a speckled gray bowl, a red-rimmed plate, a dark bottle, and a white cooktop with a black coil.
    - target: the gray bowl is the object to pick up and place into the drawer.
    - spatial: the drawer front is visible on the black cabinet at the right, with the bowl sitting on the table in front of it.
    - plan: open the top drawer, then carry the bowl to the open drawer cavity and set it inside.

- **traj000004_step000016@p145_last_text** (libero_goal) — present=5 concrete=5 filler=2 — instruction: _open the top drawer and put the bowl inside_
    - language: open the top drawer and put the bowl inside parsed as the active task phrase.
    - target: gray bowl centered below the gripper, ready for a grasp or placement.
    - scene: black drawer cabinet at the right edge with the top drawer open and visible interior.
    - spatial: bowl sits between the cabinet drawer and the robot base, with a dark bottle above it and a coiled ring below it.
    - plan: carry the bowl to the open top drawer and place it inside; this token forwards the pick-and-place phase into the action head.

- **traj000004_step000012@p145_last_text** (libero_goal) — present=5 concrete=5 filler=2 — instruction: _open the top drawer and put the bowl inside_
    - language: open the top drawer and put the bowl inside is the active task phrase.
    - scene: tabletop workspace with a drawer cabinet on the right, a white bowl, and the robot arm above the table.
    - target: gray bowl on the table is the object to grasp and move.
    - spatial: the drawer front and handle are on the right side, with the bowl positioned in front of the cabinet.
    - plan: reach for the bowl, open the top drawer, then place the bowl inside.

- **traj000005_step000046@p084_image_patch** (libero_goal) — present=3 concrete=3 filler=0 — instruction: _put the cream cheese in the bowl_
    - scene: tabletop workspace with a robot arm, a bowl of cream cheese, a dark bottle, a red-striped ring, and a black bin.
    - target: cream cheese sits in the small gray bowl near the center of the work area.
    - distractor: a dark bottle stands behind the bowl, and a black rectangular container occupies the right side.
    - spatial: the bowl is directly in front of the robot gripper area, with the ring lying below it on the table.
    - target: this patch carries the bowl-and-gripper vicinity needed to guide the reach toward the cream cheese in the bowl.

## 7. Verdict

**Verdict: YELLOW** (1 issue(s))
- plan-phase concentrated: 'grasp' = 62%

**Overall concrete-noun rate (mean across 5 bullets)**: 99.2%

## 8. Top recommendations for prompt tightening

1. **Force `language` to add information beyond the instruction.** 55% of `language` bullets cover ≥80% of the instruction's content words — i.e., they are paraphrases of the task text. Add a prompt rule: '`language` must include at least one property NOT in the instruction text (color, spatial relation, gripper state, or next-step verb).'

2. **Diversify plan-phase taxonomy.** Plan bullets are dominated by 'grasp' (62%). The labeler isn't tracking step-within-episode. Either (a) give the labeler the step index out of total, or (b) explicitly enumerate phases in the prompt (approach / grasp / lift / carry / place / release / retract / idle) and ask which one is active NOW.

3. **Make `image_patch` bullets visually distinct from `last_text` bullets.** Top-30 unigram Jaccard between `last_text` and `image_patch` is very high for: `target` top-30 Jaccard=0.76, `scene` top-30 Jaccard=0.76, `spatial` top-30 Jaccard=0.71, `plan` top-30 Jaccard=0.71 (see §5) -- the labeler is producing essentially the same caption regardless of which token is highlighted. Add a prompt rule that, for `image_patch` positions, the relevant bullet(s) must describe the object/region visible *at the gripper or directly under the current frame attention*, while for `last_text` positions the same bullets describe the parsed plan phase or instruction-level intent.

## 9. Coordination notes

- **Agent 3 (diversity)**: if they flag a boilerplate plan phrase (e.g., `pickup-and-place phase active`) appearing in a large share of plan bullets, that complements §4 here — plan-phase concentration and plan-bullet boilerplate are the same failure mode.
- **Agent 2 (forbidden phrasing)**: our filler detection (§2) is orthogonal to their phrasing scan. A bullet can be filler without tripping forbidden phrasing, and vice versa.

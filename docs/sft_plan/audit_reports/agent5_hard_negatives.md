# Agent 5 — Hard-Negative Mining Quality

**Verdict:** `RED`

**Source:** `data/activations/libero_4suite_combined/hard_negatives.jsonl`
**Mining script:** `scripts/training/mine_hard_negatives.py` (top-K cosine, same-episode mask).
**This audit:** `scripts/eval/audit_hard_negatives.py` (read-only).

## Headline numbers

| metric | value | healthy band |
|---|---:|---|
| mean mined cosine (all ptypes) | **0.9782** | [0.60, 0.95] |
| mean random-pair cosine (same ptype) | 0.8872 | — |
| mean caption Jaccard (mined) | **0.3850** | < 0.40 |
| mean caption Jaccard (random) | 0.2846 | — |
| within-suite negative fraction | 88.47% | mostly within-suite |
| degenerate-pair count | self=0, same-src=0, identical-caption=0 | all ≈ 0 |

## 1. Schema summary

- Total anchors: **101,580** (matches `manifest.json` num_examples × 2 position types? manifest reports `num_examples=50_790`; 50_790 × 2 = 101_580 ✓)
- Negatives per anchor distribution: `{8: 101580}`  → every anchor has exactly **8** negatives.
  - The mining script emits **top-K=8** (see `scripts/training/mine_hard_negatives.py`); the SFT config `ar_nce_hard_negatives_per_anchor=4` (see `SFTConfig` in `src/nla/training/sft.py`) sub-samples 4 of those 8 per anchor at each training step.
  - Mining excludes self and same-`episode_index` rows (the dataset uses `--exclude-same-episode` by default — verified by mining script source).
- Stored field `cos` is the cosine sim used during mining. Spot-check: max |stored − recomputed| over 4000 sampled pairs = `1.19e-06` (≈ 0 → stored sims are trustworthy and the audit can rely on them).
- **Position-type inventory:** three values appear in the anchor IDs: `last_text` (~50%), `image_patch` (~50%), and a small `anchor` slice (~1.5% of all rows — `grep '@p[0-9]\+_anchor'` on the JSONL returns 1,567). Mining mostly keeps `last_text→last_text` and `image_patch→image_patch` (99.66% of pairs), but it does NOT explicitly mask cross-ptype negatives: a small number of `last_text↔anchor` swaps appear (≈2,700 pairs total). InfoNCE will treat those as legitimate negatives even though the activation positions are different kinds of summary token — probably fine, but worth knowing.

## 2. Cosine-similarity distribution: mined vs random

Mined cosines come from the stored `cos` field (full top-8 list per sampled anchor). Random-pair cosines were computed by sampling 500 random (anchor_a, anchor_b) pairs from the same position_type and loading their activations.

- **Mined (sampled top-8, all ptypes):** mean=0.9782  p10=0.9314  p50=0.9976  p90=0.9992  min=0.8801  max=0.9995  n=4000
- **Random pairs (same ptype):** mean=0.8872  p10=0.7119  p50=0.9481  p90=0.9847  min=0.5565  max=0.9975  n=500
- **All anchors top-1 (full 101k):** mean=0.9840  p10=0.9486  p50=0.9984  p90=0.9993  min=0.7611  max=0.9997  n=101580

**Mined cosine** (per position_type):

| ptype | mean | p10 | p50 | p90 | count |
|---|---|---|---|---|---|
| `image_patch` | 0.9577 | 0.9126 | 0.9602 | 0.9939 | 2008 |
| `last_text` | 0.9989 | 0.9984 | 0.9990 | 0.9993 | 1992 |

**Random-pair cosine** (per position_type):

| ptype | mean | p10 | p50 | p90 | count |
|---|---|---|---|---|---|
| `anchor` | 0.9610 | 0.9425 | 0.9557 | 0.9921 | 146 |
| `image_patch` | 0.7494 | 0.6678 | 0.7526 | 0.8161 | 173 |
| `last_text` | 0.9592 | 0.9374 | 0.9563 | 0.9890 | 181 |

### Mined-cosine histogram
```
  [0.000,0.050)      0 |
  [0.050,0.100)      0 |
  [0.100,0.150)      0 |
  [0.150,0.200)      0 |
  [0.200,0.250)      0 |
  [0.250,0.300)      0 |
  [0.300,0.350)      0 |
  [0.350,0.400)      0 |
  [0.400,0.450)      0 |
  [0.450,0.500)      0 |
  [0.500,0.550)      0 |
  [0.550,0.600)      0 |
  [0.600,0.650)      0 |
  [0.650,0.700)      0 |
  [0.700,0.750)      0 |
  [0.750,0.800)      0 |
  [0.800,0.850)      0 |
  [0.850,0.900)     76 |
  [0.900,0.950)    699 |########
  [0.950,1.000)   3225 |########################################
```
### Random-pair cosine histogram
```
  [0.000,0.050)      0 |
  [0.050,0.100)      0 |
  [0.100,0.150)      0 |
  [0.150,0.200)      0 |
  [0.200,0.250)      0 |
  [0.250,0.300)      0 |
  [0.300,0.350)      0 |
  [0.350,0.400)      0 |
  [0.400,0.450)      0 |
  [0.450,0.500)      0 |
  [0.500,0.550)      0 |
  [0.550,0.600)      1 |
  [0.600,0.650)      5 |
  [0.650,0.700)     30 |#####
  [0.700,0.750)     47 |#######
  [0.750,0.800)     50 |########
  [0.800,0.850)     37 |######
  [0.850,0.900)      6 |#
  [0.900,0.950)     86 |##############
  [0.950,1.000)    238 |########################################
```

## 3. Caption-similarity (token Jaccard)

Token Jaccard between the anchor caption and each of its 8 mined negative captions.
- **Mined pairs:** mean=0.3850  p10=0.2871  p50=0.3837  p90=0.4824  min=0.1091  max=0.6557  n=4000
- **Random pairs (same ptype, no mining):** mean=0.2846  p10=0.2105  p50=0.2738  p90=0.3766  min=0.1429  max=0.5158  n=500

**Mined Jaccard** (per position_type):

| ptype | mean | p10 | p50 | p90 | count |
|---|---|---|---|---|---|
| `image_patch` | 0.3532 | 0.2584 | 0.3516 | 0.4487 | 2008 |
| `last_text` | 0.4169 | 0.3302 | 0.4146 | 0.5000 | 1992 |

### Mined-Jaccard histogram
```
  [0.000,0.050)      0 |
  [0.050,0.100)      0 |
  [0.100,0.150)      2 |
  [0.150,0.200)     21 |
  [0.200,0.250)    126 |####
  [0.250,0.300)    367 |##############
  [0.300,0.350)    770 |#############################
  [0.350,0.400)   1034 |########################################
  [0.400,0.450)    878 |#################################
  [0.450,0.500)    521 |####################
  [0.500,0.550)    204 |#######
  [0.550,0.600)     60 |##
  [0.600,0.650)     15 |
  [0.650,0.700)      2 |
  [0.700,0.750)      0 |
  [0.750,0.800)      0 |
  [0.800,0.850)      0 |
  [0.850,0.900)      0 |
  [0.900,0.950)      0 |
  [0.950,1.000)      0 |
```
### Random-Jaccard histogram
```
  [0.000,0.050)      0 |
  [0.050,0.100)      0 |
  [0.100,0.150)      1 |
  [0.150,0.200)     28 |######
  [0.200,0.250)    123 |#############################
  [0.250,0.300)    169 |########################################
  [0.300,0.350)    100 |#######################
  [0.350,0.400)     54 |############
  [0.400,0.450)     17 |####
  [0.450,0.500)      7 |#
  [0.500,0.550)      1 |
  [0.550,0.600)      0 |
  [0.600,0.650)      0 |
  [0.650,0.700)      0 |
  [0.700,0.750)      0 |
  [0.750,0.800)      0 |
  [0.800,0.850)      0 |
  [0.850,0.900)      0 |
  [0.900,0.950)      0 |
  [0.950,1.000)      0 |
```

## 4. Cross-suite distribution

- Total (anchor, neg) pairs: **812,640**
- Within-suite negatives: **718,932** (88.47%)
- Cross-suite negatives:  **93,708** (11.53%)

Top (anchor_suite → neg_suite) pairs:

| pair | count |
|---|---|
| `object->object` | 201,922 |
| `goal->goal` | 183,767 |
| `spatial->spatial` | 182,474 |
| `10->10` | 150,769 |
| `spatial->goal` | 16,807 |
| `goal->spatial` | 13,516 |
| `10->object` | 12,654 |
| `10->goal` | 9,662 |
| `10->spatial` | 8,835 |
| `object->10` | 7,312 |
| `object->goal` | 4,729 |
| `spatial->10` | 4,532 |
| `goal->10` | 4,414 |
| `object->spatial` | 3,957 |
| `goal->object` | 3,743 |
| `spatial->object` | 3,547 |

## 5. Cross-position-type distribution

- Pairs with same position_type as anchor: **809,897** (99.66%)

| pair | count |
|---|---|
| `last_text->last_text` | 407,258 |
| `image_patch->image_patch` | 402,632 |
| `last_text->anchor` | 1,422 |
| `anchor->last_text` | 1,321 |
| `anchor->anchor` | 7 |

## 6. Degenerate cases

- Self-matches (anchor in its own neg list): **0** (0.0000%) — should be 0
- Same `source_example_id` as anchor: **0** (0.0000%)
- Identical caption text: **0** (0.0000%)
- Pairs missing a caption in labels: **0**

## 7. Example (anchor, negs) blocks

Showing first 4 negs (matches `ar_nce_hard_negatives_per_anchor=4`); each row shows cosine and Jaccard.

### Example 1

**Anchor** `10__traj000372_step000048@p046_image_patch`
> - scene: tabletop workspace with a gray robot arm, a dark three-compartment caddy, and a white mug on a wooden surface.
- target: the caddy’s back compartment is the placement slot, with the book-sized dark rectangular object nearby.
- dist...

- **Neg 1** `10__traj000027_step000044@p045_image_patch`  (cos=0.9519, jaccard=0.468)
  > - scene: wooden tabletop with a brown three-compartment caddy, a white mug, and a black book standing near the front edge.
- target: the book is the relevant object to pick up; it is a dark rectangular object with a visible spine and pages....
- **Neg 2** `10__traj000299_step000018@p046_image_patch`  (cos=0.9481, jaccard=0.468)
  > - scene: wooden tabletop with a brown caddy, a silver mug with a gold handle, a black pen-like object, and the robot arm above the table.
- target: the book is not visible in this patch; the caddy’s open back compartment is the placement de...
- **Neg 3** `10__traj000368_step000028@p046_image_patch`  (cos=0.9442, jaccard=0.368)
  > - scene: Wooden tabletop with a black-and-brown caddy, a black book standing upright, and a white mug with yellow liquid.
- target: The book is the graspable object matching the instruction; the caddy’s rear compartment is the placement goa...
- **Neg 4** `10__traj000028_step000044@p046_image_patch`  (cos=0.9414, jaccard=0.416)
  > - scene: tabletop workspace with a robot arm above a wooden table and a brown three-compartment caddy.
- target: a black book-like object lies near the white mug, separate from the caddy.
- spatial: the caddy’s back compartment is the far s...

### Example 2

**Anchor** `object__traj000293_step000046@p147_last_text`
> - scene: a wicker basket with a white liner sits on the left, and a robot arm hovers over a tiled floor with several small items.
- target: the alphabet soup can is the multicolored can on the floor near the lower center of the workspace.
-...

- **Neg 1** `object__traj000006_step000050@p147_last_text`  (cos=0.9993, jaccard=0.389)
  > - scene: tabletop setup with a white woven basket on the left and a cluster of food items on the right.
- target: canned soup with a red label and green top sits in front of the other objects.
- distractor: green bottle stands near the cent...
- **Neg 2** `object__traj000364_step000044@p147_last_text`  (cos=0.9992, jaccard=0.519)
  > - language: the parsed task is pick up alphabet soup and place it in the basket.
- target: the red-and-green soup can sits on the floor in front of the robot.
- scene: a wicker basket lined with white fabric is positioned to the left of the...
- **Neg 3** `object__traj000069_step000036@p147_last_text`  (cos=0.9992, jaccard=0.366)
  > - scene: tabletop workspace with a woven basket on the left, a soup can near the front, and other small blocks and containers near the robot.
- target: the alphabet soup can with a red label and green top sits closest to the reach area on t...
- **Neg 4** `object__traj000345_step000044@p147_last_text`  (cos=0.9992, jaccard=0.400)
  > - scene: tabletop work area with a wicker basket lined in white paper, a soup can, a green bottle, a small orange block, a blue block, and a red carton under the robot arm.
- target: alphabet soup can is the grasp target, standing near the ...

### Example 3

**Anchor** `10__traj000041_step000034@p149_last_text`
> - scene: wooden tabletop with a woven basket on the left and a robot arm centered over two food items.
- target: an alphabet soup can and a cream cheese box are the two objects named in the instruction.
- spatial: the cream cheese box is un...

- **Neg 1** `10__traj000373_step000048@p149_last_text`  (cos=0.9989, jaccard=0.429)
  > - scene: wooden table with a woven basket on the left and a white robot arm over the workspace.
- target: alphabet soup can and cream cheese box are the two objects named by the instruction.
- target: the soup can with colorful label and th...
- **Neg 2** `10__traj000373_step000052@p149_last_text`  (cos=0.9988, jaccard=0.453)
  > - target: basket sits on the left side of the tabletop as the destination container.
- target: alphabet soup can and cream cheese box are the two manipulable items on the right side near the gripper.
- scene: wooden table surface with the r...
- **Neg 3** `10__traj000074_step000058@p149_last_text`  (cos=0.9987, jaccard=0.457)
  > - scene: tabletop workspace with a wicker basket on the left, a white robot arm over the table, and two food items on the right.
- target: alphabet soup can with red-and-green labeling sits near the robot gripper on the wooden surface.
- ta...
- **Neg 4** `10__traj000318_step000054@p149_last_text`  (cos=0.9987, jaccard=0.400)
  > - language: active instruction is to put both the alphabet soup and the cream cheese box into the basket.
- target: alphabet soup can is on the table near the robot gripper, with a red-and-green label visible.
- target: cream cheese box is ...

### Example 4

**Anchor** `10__traj000271_step000020@p105_image_patch`
> - scene: tabletop with a gray robot arm over a wooden surface, a white plate, a white mug, a red patterned mug, and a chocolate pudding cup.
- target: white mug stands near the plate in the front camera, while the red mug hangs under the gr...

- **Neg 1** `10__traj000157_step000040@p098_image_patch`  (cos=0.9123, jaccard=0.321)
  > - scene: wooden tabletop with two pink-rimmed plates and two mugs beneath a white robot arm.
- target: white mug with a yellow handle sits left of center near the left plate.
- target: yellow-and-white mug with a white handle sits under the...
- **Neg 2** `10__traj000210_step000006@p104_image_patch`  (cos=0.9104, jaccard=0.438)
  > - scene: wooden tabletop with a white plate, a white mug, a red patterned mug, and a chocolate pudding cup.
- target: white mug sits to the right of the plate and is the object to be moved onto it.
- target: chocolate pudding cup lies below...
- **Neg 3** `10__traj000000_step000054@p106_image_patch`  (cos=0.9060, jaccard=0.329)
  > - scene: tabletop workspace with two plates, a white mug with yellow handle, a red patterned mug, and a white ribbed mug under the robot arm.
- target: white mug on the left side of the table is positioned nearest the left plate.
- target: ...
- **Neg 4** `10__traj000108_step000024@p103_image_patch`  (cos=0.9036, jaccard=0.600)
  > - scene: tabletop workspace with a red patterned mug, a white mug, a white plate, and a chocolate pudding cup.
- target: white mug is the object to place on the plate; chocolate pudding cup is the second object to place beside the plate.
- ...

### Example 5

**Anchor** `10__traj000213_step000050@p151_last_text`
> - scene: wooden tabletop with a black caddy, a mug, and a black book-like object near the robot gripper.
- target: the black rectangular book stands upright below the gripper and is the object to pick up.
- spatial: the caddy’s back compart...

- **Neg 1** `10__traj000348_step000050@p151_last_text`  (cos=0.9987, jaccard=0.418)
  > - scene: tabletop workspace with a gray robot arm, a brown two-compartment caddy, a white mug, and a black book standing near the front edge.
- target: the black book is the object to grasp; it is upright beside the mug and in front of the ...
- **Neg 2** `10__traj000159_step000044@p151_last_text`  (cos=0.9987, jaccard=0.417)
  > - scene: tabletop workspace with a gray robot arm, a black book-like object, a white mug, and a brown caddy with three compartments.
- target: the book is the black rectangular object standing near the left side of the table, separated from...
- **Neg 3** `10__traj000311_step000054@p151_last_text`  (cos=0.9987, jaccard=0.425)
  > - scene: wooden tabletop with a black robotic gripper arm, a brown caddy with three compartments, and a light-colored mug nearby.
- target: the book is visible as a thin black rectangular object standing near the gripper and table edge.
- s...
- **Neg 4** `10__traj000279_step000034@p151_last_text`  (cos=0.9986, jaccard=0.551)
  > - scene: tabletop workspace with a wooden surface, a black caddy with multiple compartments, a white robot arm, a mug, and a black rectangular book.
- target: the black book stands upright near the mug, separate from the caddy and ready to ...

### Example 6

**Anchor** `goal__traj000003_step000008@p145_last_text`
> - scene: tabletop workspace with a bowl, patterned plate, black bottle, blue rectangular item, drawers, and the robot arm overhead.
- target: gray bowl is the object to grasp and place into the top drawer.
- spatial: bowl sits between the b...

- **Neg 1** `goal__traj000119_step000008@p145_last_text`  (cos=0.9993, jaccard=0.315)
  > - scene: tabletop workspace with a drawer cabinet, a black appliance base, a white hotplate, and a metal bowl on the desk.
- target: the bowl is centered in the workspace and is the object to be placed into the drawer.
- distractor: the ope...
- **Neg 2** `goal__traj000119_step000010@p145_last_text`  (cos=0.9992, jaccard=0.346)
  > - language: instruction parsed as open the top drawer and place the bowl inside.
- scene: tabletop workspace with a black cabinet drawer unit, white coil burner, wooden rack, bottle, bowl, plate, and blue carton.
- target: gray bowl on the ...
- **Neg 3** `goal__traj000363_step000010@p145_last_text`  (cos=0.9991, jaccard=0.407)
  > - scene: tabletop workspace with a white induction cooktop, a black bottle, a gray bowl, a white plate with red rings, and a cabinet/drawer unit beside the robot arm.
- target: the gray bowl is the object to be moved; it sits near the plate...
- **Neg 4** `goal__traj000119_step000012@p145_last_text`  (cos=0.9991, jaccard=0.449)
  > - scene: tabletop workspace with a dark cabinet drawer on the right, a white stove-like surface on the left, and a robot arm above the center.
- target: gray bowl sits near the center of the table, below a dark bottle and above a striped pl...

### Example 7

**Anchor** `10__traj000281_step000048@p149_last_text`
> - scene: wooden tabletop with a woven basket at left, a blue box near center, and a striped soup can at lower right.
- target: alphabet soup can is the striped can on the table; cream cheese box is the small blue carton near the basket.
- s...

- **Neg 1** `10__traj000208_step000040@p149_last_text`  (cos=0.9988, jaccard=0.544)
  > - scene: wooden tabletop with a woven basket on the left and a robot arm over objects on the right.
- target: alphabet soup can with red and green stripes sits near the lower right of the main view.
- target: cream cheese box is the small b...
- **Neg 2** `10__traj000221_step000052@p149_last_text`  (cos=0.9983, jaccard=0.472)
  > - scene: tabletop workspace with a wicker basket on the left and two canned food items on the wooden surface.
- target: alphabet soup can stands near the robot gripper, and a cream cheese box sits farther back near the basket.
- spatial: th...
- **Neg 3** `10__traj000289_step000040@p149_last_text`  (cos=0.9982, jaccard=0.425)
  > - language: put both items into the basket; two-object pick-and-place task active.
- target: alphabet soup can and cream cheese box are the goal objects on the tabletop.
- scene: wicker basket sits on the left side of the wooden table, empt...
- **Neg 4** `10__traj000083_step000050@p149_last_text`  (cos=0.9982, jaccard=0.355)
  > - language: task parses as moving both the alphabet soup can and the cream cheese box into the basket.
- scene: tabletop workspace with a wicker basket on the left and a white robot arm over a brown wooden surface.
- target: a cream cheese ...

### Example 8

**Anchor** `spatial__traj000205_step000010@p030_image_patch`
> - scene: tabletop workspace with a robot arm above a light wood table and multiple bowls, a plate, and a small packet.
- target: the black bowl is the bowl centered in the lower camera view, with a dark rim and pale interior.
- distractor: ...

- **Neg 1** `spatial__traj000165_step000006@p030_image_patch`  (cos=0.9918, jaccard=0.541)
  > - scene: tabletop workspace with a robot arm above a light wood table and multiple dishes scattered around.
- target: black bowl is visible near the center of the table in the lower camera view.
- distractor: white plate with a red rim sits...
- **Neg 2** `spatial__traj000327_step000008@p030_image_patch`  (cos=0.9868, jaccard=0.321)
  > - scene: tabletop workspace with a robot arm, several small bowls, a white plate with red rings, and a dark cabinet edge.
- target: black bowl on the table near a small silver ramekin; both are visible among the other dishes.
- distractor: ...
- **Neg 3** `spatial__traj000415_step000024@p030_image_patch`  (cos=0.9792, jaccard=0.515)
  > - scene: beige tabletop with a red-rimmed plate, a black bowl, a small silver ramekin, and a dark cabinet at the right.
- target: black bowl on the table near the silver ramekin, matching the instruction to pick up the bowl.
- distractor: r...
- **Neg 4** `spatial__traj000123_step000000@p030_image_patch`  (cos=0.9791, jaccard=0.373)
  > - scene: kitchen work surface with a stove at the top view and a light wood table below.
- target: black bowl sits on the stove area beside a small white tray, matching the object to pick up.
- distractor: shiny metal cups, a red-and-white ...

### Example 9

**Anchor** `goal__traj000413_step000030@p054_image_patch`
> - scene: tabletop workspace with a gray robot arm, a dark cabinet, a white appliance base, a blue rectangular object, a red ring, and a gray bowl on a light wood surface.
- target: cabinet drawer handle visible on the dark cabinet; the cabi...

- **Neg 1** `goal__traj000252_step000022@p054_image_patch`  (cos=0.9733, jaccard=0.440)
  > - scene: tabletop workspace with a cabinet, a robot arm, a white burner-like plate, a glass bowl, a black bottle, a blue rectangular object, and a coiled cable.
- target: dark cabinet drawer front with a silver vertical handle is visible be...
- **Neg 2** `goal__traj000089_step000018@p062_image_patch`  (cos=0.9728, jaccard=0.322)
  > - scene: tabletop workspace with a black cabinet at the right, a white burner at the left, and a robot arm above the work area.
- target: a dark wine bottle stands near the center of the table, below the robot arm and left of the cabinet.
-...
- **Neg 3** `goal__traj000023_step000008@p054_image_patch`  (cos=0.9708, jaccard=0.381)
  > - scene: tabletop workspace with a gray bowl, a white plate with red rings, a black bottle-like object, a blue rectangular item, and a dark cabinet.
- target: gray bowl is the object to move for the cabinet placement task.
- spatial: bowl s...
- **Neg 4** `goal__traj000333_step000030@p054_image_patch`  (cos=0.9678, jaccard=0.360)
  > - scene: tabletop workspace with a gray robotic arm, a black cabinet on the right, a patterned bowl, a circular plate, and a small blue object on the wood surface.
- target: the bowl is the relevant object, with a light outer rim and dark p...

### Example 10

**Anchor** `goal__traj000271_step000050@p008_image_patch`
> - scene: tabletop workspace with a robot arm, a black bowl-like container on the right, a white plate with red stripes, and a small black bottle near the center.
- target: cream cheese container is present as a blue-labeled package in the l...

- **Neg 1** `spatial__traj000010_step000042@p007_image_patch`  (cos=0.9713, jaccard=0.298)
  > - scene: tabletop workspace with a robot arm, a white plate, a black drawer unit, a pink-rimmed ramekin, and a patterned black bowl.
- target: the black bowl sits on the ramekin in the lower center of the scene, directly matching the object...
- **Neg 2** `goal__traj000060_step000048@p007_image_patch`  (cos=0.9696, jaccard=0.394)
  > - scene: tabletop workspace with a robot arm, a black rectangular bin/tray, a clear bowl with a red rim, and two dark bottles standing near the arm.
- target: cream cheese container is the light blue rectangular package with a circular logo...
- **Neg 3** `spatial__traj000069_step000056@p007_image_patch`  (cos=0.9690, jaccard=0.347)
  > - scene: beige tabletop with a robot arm, a white plate with red rings, and a patterned black bowl.
- target: the black bowl is the object named by the instruction and is visible near the plate.
- spatial: the bowl sits below the plate in t...
- **Neg 4** `spatial__traj000044_step000050@p008_image_patch`  (cos=0.9669, jaccard=0.317)
  > - scene: tabletop workspace with two robot arms, a black cabinet on the right, and several small dishes on a light wood surface.
- target: black bowl sits near the bottom of the close camera view, with a silver patterned interior and pale r...

### Example 11

**Anchor** `spatial__traj000234_step000036@p046_image_patch`
> - scene: beige tabletop with a white plate, a black bowl, a small silver ramekin, and a dark cabinet-like object.
- target: the black bowl is the object to lift, positioned near the ramekin on the table.
- distractor: the silver ramekin and...

- **Neg 1** `spatial__traj000025_step000036@p046_image_patch`  (cos=0.9839, jaccard=0.408)
  > - scene: tabletop workspace with a black bowl, a silver ramekin, a white plate with red rings, and a black cabinet/drawer unit.
- target: the black bowl sits next to the ramekin on the table and is the object to lift.
- spatial: the plate l...
- **Neg 2** `spatial__traj000175_step000008@p046_image_patch`  (cos=0.9778, jaccard=0.447)
  > - scene: tabletop with two camera views of the same workspace, robot arm above a light wood surface and a dark cabinet on the right.
- target: black bowl on the table near a small ramekin, with another gray bowl nearby as a distractor.
- di...
- **Neg 3** `spatial__traj000394_step000008@p046_image_patch`  (cos=0.9743, jaccard=0.333)
  > - scene: light wood tabletop with a black cabinet at the right edge and the robot arm above the workspace.
- target: black bowl filled with white rice sits near the lower-right area of the table in the main view.
- distractor: a small metal...
- **Neg 4** `spatial__traj000375_step000004@p046_image_patch`  (cos=0.9734, jaccard=0.410)
  > - scene: kitchen counter with a stove at the back, a plate on the counter, and a black drawer unit on the right.
- target: black bowl sits in the stove area, with the bowl rim visible against the dark burner surface.
- distractor: a small m...

### Example 12

**Anchor** `object__traj000129_step000044@p060_image_patch`
> - scene: tabletop workspace with a white basket on the left, a robot arm in the center, and small containers near the gripper.
- target: cream cheese carton visible in the lower camera frame, tilted upright with a blue-and-white label.
- di...

- **Neg 1** `object__traj000391_step000030@p060_image_patch`  (cos=0.9958, jaccard=0.356)
  > - scene: tabletop workspace with a basket on the left and several small food items on the floor-like surface near the robot.
- target: cream cheese carton is the blue rectangular package near the robot base, matching the pickup instruction....
- **Neg 2** `object__traj000249_step000058@p060_image_patch`  (cos=0.9958, jaccard=0.344)
  > - scene: tabletop or floor workspace with a white basket on the left and several small grocery items spread out on the tiles.
- target: salad dressing bottle is the green bottle near the center, standing upright beside other items.
- distra...
- **Neg 3** `object__traj000003_step000032@p060_image_patch`  (cos=0.9957, jaccard=0.383)
  > - scene: tabletop workspace with a white basket on the left and a robot arm centered above the floor tiles.
- target: cream cheese is the blue rectangular package on the floor near the robot base.
- distractor: two cylindrical cans sit to t...
- **Neg 4** `object__traj000415_step000042@p060_image_patch`  (cos=0.9956, jaccard=0.398)
  > - scene: tabletop/worksurface with a silver robot arm, a woven basket, and several small food items on the floor-like surface.
- target: a green bottle-shaped dressing container is positioned near the center beneath the gripper area.
- dist...

### Example 13

**Anchor** `goal__traj000161_step000038@p143_last_text`
> - scene: tabletop workspace with a robot arm over a light wood surface, a black bowl, a black bottle, a coiled red-and-white cable, and a blue carton.
- target: cream cheese container is the blue carton with a white label near the bottom ce...

- **Neg 1** `goal__traj000202_step000034@p143_last_text`  (cos=0.9988, jaccard=0.368)
  > - language: instruction specifies placing the cream cheese into the bowl.
- scene: tabletop workspace with a robot arm, a pink-rimmed bowl, a blue cream-cheese box, a dark bottle, and a black appliance.
- target: the blue cream-cheese packa...
- **Neg 2** `goal__traj000231_step000032@p143_last_text`  (cos=0.9987, jaccard=0.315)
  > - scene: tabletop workspace with a robot arm, a dark bowl-like container, a black bottle, a red-and-white coil ring, and a small blue item near the front edge.
- target: cream cheese is the wooden slotted block/packet resting near the right...
- **Neg 3** `goal__traj000085_step000032@p143_last_text`  (cos=0.9986, jaccard=0.392)
  > - language: instruction parsed as placing cream cheese into the bowl.
- scene: tabletop workspace with a robot arm, a dark bowl, a black cabinet, a bottle, a striped plate, and a small cream-cheese package.
- target: cream-cheese package si...
- **Neg 4** `goal__traj000253_step000050@p143_last_text`  (cos=0.9986, jaccard=0.463)
  > - scene: tabletop workspace with a gray robot arm over a light wood surface, a white plate with red rings, a clear bowl, a dark bottle, and a black cabinet.
- target: cream cheese container is the small dark object at the gripper area near ...

### Example 14

**Anchor** `10__traj000019_step000004@p154_last_text`
> - language: parsed instruction links white mug to the plate and chocolate pudding to the right of the plate.
- scene: tabletop workspace with a white plate centered low in view, a gray mug on the right, and a chocolate pudding package near ...

- **Neg 1** `10__traj000068_step000008@p154_last_text`  (cos=0.9993, jaccard=0.493)
  > - language: two-step placement task parsed; white mug onto the plate, chocolate pudding to the right of the plate.
- scene: wooden tabletop with a white-and-pink plate, a gray mug on the right, a red patterned mug under the robot, and a cho...
- **Neg 2** `10__traj000062_step000006@p154_last_text`  (cos=0.9993, jaccard=0.368)
  > - language: task specifies two placements: white mug onto the plate, then chocolate pudding to the right of the plate.
- target: white mug is the pale blue-gray cup on the right side of the table, separate from the plate.
- target: chocolat...
- **Neg 3** `10__traj000356_step000000@p154_last_text`  (cos=0.9993, jaccard=0.392)
  > - language: parsed instruction specifies two placements, mug onto plate and pudding to the right of the plate.
- scene: white mug, striped plate, chocolate pudding snack bar, and red patterned mug are visible on the wooden table.
- spatial:...
- **Neg 4** `10__traj000068_step000002@p154_last_text`  (cos=0.9992, jaccard=0.369)
  > - language: put the white mug on the plate and place the chocolate pudding to the right of the plate.
- target: white mug stands to the right of the plate, while the red patterned mug is under the gripper.
- target: small chocolate pudding ...

### Example 15

**Anchor** `object__traj000055_step000048@p026_image_patch`
> - scene: tabletop workspace with a robot arm centered above scattered small items on a gray surface.
- target: cream cheese is the blue rectangular carton near the gripper in the close-up frame.
- distractor: a woven basket sits to the left...

- **Neg 1** `object__traj000296_step000048@p026_image_patch`  (cos=0.9742, jaccard=0.323)
  > - scene: tabletop workspace with a basket on the left, cream cheese box in the center, and the robot arm over the work area.
- target: blue Cream O'Land cream cheese package is visible on the table and also close in the gripper camera view....
- **Neg 2** `object__traj000096_step000038@p026_image_patch`  (cos=0.9631, jaccard=0.351)
  > - scene: tabletop workspace with a woven basket on the left and a robot arm centered over a gray floor tile surface.
- target: cream cheese carton is visible among several small packages, including a blue-and-white carton in the lower camer...
- **Neg 3** `object__traj000423_step000058@p026_image_patch`  (cos=0.9578, jaccard=0.295)
  > - scene: tabletop or floor-level workspace with a white basket on the left and several bottles clustered near the robot arm.
- target: bbq sauce bottle with dark reddish-brown liquid and orange cap among the nearby containers.
- distractor:...
- **Neg 4** `object__traj000258_step000056@p026_image_patch`  (cos=0.9502, jaccard=0.294)
  > - scene: tabletop workspace with tiled floor and a metal basket on the left.
- target: brown tomato-sauce bottle stands near the robot arm beside other condiment packages.
- distractor: colorful cartons and a second small box sit to the rig...

### Example 16

**Anchor** `spatial__traj000374_step000022@p152_last_text`
> - language: parsed task specifies lifting the black bowl from the ramekin and placing it onto the plate.
- target: black bowl sitting in a small gray ramekin near the center-left of the workspace.
- scene: white plate with a red patterned r...

- **Neg 1** `spatial__traj000365_step000026@p152_last_text`  (cos=0.9987, jaccard=0.434)
  > - scene: tabletop workspace with a robot arm over a light wood surface, a black cabinet on the right, and a striped plate visible near the top.
- target: black bowl sits on a small ramekin-like stand near the center of the table.
- distract...
- **Neg 2** `spatial__traj000365_step000016@p152_last_text`  (cos=0.9982, jaccard=0.473)
  > - language: parsed task is pick up the black bowl resting on the ramekin, then place it on the plate.
- target: black bowl sits directly on a small square ramekin; the bowl is the pickup object.
- scene: white plate with a dark circular cen...
- **Neg 3** `spatial__traj000208_step000044@p152_last_text`  (cos=0.9982, jaccard=0.303)
  > - scene: tabletop workspace with a robot arm above a light wood surface, a black bowl, a white plate, and a red ring-shaped ramekin.
- target: black bowl sits on the ramekin area and is the object to be lifted; the plate is nearby for place...
- **Neg 4** `spatial__traj000127_step000026@p152_last_text`  (cos=0.9982, jaccard=0.417)
  > - scene: tabletop workspace with a robot arm over a light wood surface, a black cabinet on the right, a red-rimmed plate, and a small black ramekin.
- target: black bowl with patterned interior sits directly on the ramekin near the center o...

### Example 17

**Anchor** `spatial__traj000302_step000026@p083_image_patch`
> - scene: tabletop workspace with a robot arm, a white plate with red rings, a black bowl, and a small ramekin-like object.
- target: black bowl sits to the right of the plate and near the ramekin, matching the object to be lifted.
- spatial...

- **Neg 1** `spatial__traj000142_step000022@p082_image_patch`  (cos=0.9781, jaccard=0.384)
  > - scene: tabletop workspace with a robot arm at the back and several dishes arranged on a light wood surface.
- target: black bowl is visible on the right side of the table in the close-up frame, near the plate.
- distractor: a red-rimmed w...
- **Neg 2** `goal__traj000407_step000032@p081_image_patch`  (cos=0.9669, jaccard=0.361)
  > - scene: tabletop kitchen workspace with a gray robot arm, black stove appliance, white plate with red rings, and a metal bowl.
- target: bowl is the object named by the instruction and is visible on the table near the plate.
- spatial: sto...
- **Neg 3** `spatial__traj000275_step000036@p079_image_patch`  (cos=0.9666, jaccard=0.354)
  > - scene: tabletop workspace with a silver robot arm, a white plate with red rings, a dark cabinet, and a black bowl.
- target: black bowl sits left of the plate in the front camera, with a round opening and pale textured interior.
- spatial...
- **Neg 4** `goal__traj000020_step000036@p073_image_patch`  (cos=0.9636, jaccard=0.357)
  > - scene: tabletop workspace with a cabinet, robot arm, metal bowl, coiled cable, phone, and round striped plate on a light wood surface.
- target: black cabinet drawer front and gray horizontal handle are visible as the relevant object for ...

### Example 18

**Anchor** `spatial__traj000232_step000030@p091_image_patch`
> - scene: tabletop workspace with a robot arm, a white plate with red rings, a black bowl, and a ramekin area.
- target: black bowl with gray patterned interior sits below the plate in the camera view.
- spatial: the bowl is positioned betwe...

- **Neg 1** `spatial__traj000259_step000018@p075_image_patch`  (cos=0.9422, jaccard=0.301)
  > - scene: light wooden cabinet surface with two camera views showing a black cabinet drawer unit, a white plate with red rim, a small metal cup, and a dark bowl.
- target: black bowl sits on the cabinet near the drawer unit and is the object...
- **Neg 2** `spatial__traj000134_step000040@p084_image_patch`  (cos=0.9418, jaccard=0.447)
  > - scene: tabletop workspace with a gray robot arm, a white plate with red rings, and a black bowl with a patterned interior.
- target: the black bowl sits between the plate and the dark rectangular ramekin-like object.
- spatial: the bowl i...
- **Neg 3** `goal__traj000079_step000052@p073_image_patch`  (cos=0.9341, jaccard=0.353)
  > - scene: tabletop workspace with a robot arm, a black cabinet, a wooden slatted rack, a blue packet, and a striped bowl on the table.
- target: the bowl is the task object, patterned white with dark gray floral motifs and a yellow rim.
- sp...
- **Neg 4** `spatial__traj000167_step000030@p106_image_patch`  (cos=0.9328, jaccard=0.382)
  > - scene: tabletop workspace with a robot arm above a light wood table and several objects spread out.
- target: black bowl with patterned interior is visible near the lower-left area of the close-up frame.
- distractor: red-and-white stripe...

### Example 19

**Anchor** `10__traj000254_step000042@p024_image_patch`
> - scene: kitchen counter with an open cabinet drawer on the left, a black/gray robotic gripper above the work surface, and a gray bowl on the wooden tabletop.
- target: the black bowl sits on the counter near the cabinet drawer opening and ...

- **Neg 1** `spatial__traj000389_step000026@p021_image_patch`  (cos=0.9551, jaccard=0.202)
  > - scene: tabletop workspace with a robot arm, a black cookie box, a black bowl, and a white plate with red rings.
- target: black bowl sits on top of the cookie box in the upper camera view and is the object to grasp.
- distractor: a second...
- **Neg 2** `object__traj000373_step000030@p024_image_patch`  (cos=0.9549, jaccard=0.252)
  > - scene: tabletop workspace with a gray robot arm, a woven basket on the left, and several small food containers on the floor in front of the arm.
- target: a tomato sauce package is visible among the objects near the center-right of the wo...
- **Neg 3** `10__traj000018_step000058@p025_image_patch`  (cos=0.9545, jaccard=0.214)
  > - scene: wooden tabletop with two pink-rimmed plates at the left and right edges, a white mug near the center-left, and a red patterned mug held by the robot gripper.
- target: the white mug is the object matched to the left-plate part of t...
- **Neg 4** `goal__traj000175_step000038@p024_image_patch`  (cos=0.9544, jaccard=0.224)
  > - scene: tabletop workspace with a robot arm centered above a light wood surface and a black cabinet at the right edge.
- target: wine bottle object is present as a dark bottle-shaped item on the table near the lower center of the view.
- d...

### Example 20

**Anchor** `10__traj000000_step000028@p010_image_patch`
> - scene: tabletop with two plates and three mugs under a robot arm.
- target: white mug with a yellow handle on the left side of the table.
- target: yellow-and-white mug near the center, plus a red patterned mug and a gray mug nearby.
- sp...

- **Neg 1** `10__traj000013_step000026@p010_image_patch`  (cos=0.9874, jaccard=0.224)
  > - scene: tabletop workspace with a woven basket on the left and the robot arm hovering above the table.
- target: alphabet soup can and tomato sauce carton are visible among several boxed/canned groceries.
- distractor: orange juice carton,...
- **Neg 2** `10__traj000062_step000038@p010_image_patch`  (cos=0.9873, jaccard=0.446)
  > - scene: tabletop workspace with brown wooden surface, plate, two mugs, and a chocolate pudding bar wrapper.
- target: white mug sits to the right of the plate under the robot gripper.
- distractor: red patterned mug stands near the center-...
- **Neg 3** `10__traj000079_step000028@p010_image_patch`  (cos=0.9863, jaccard=0.191)
  > - scene: tabletop workspace with a metal basket on the left and several grocery items spread on the wooden surface.
- target: alphabet soup can and tomato sauce carton are both present among the items to be collected.
- distractor: blue mil...
- **Neg 4** `object__traj000167_step000018@p010_image_patch`  (cos=0.9844, jaccard=0.168)
  > - scene: tabletop workspace with a woven basket on the left image and several small cartons and cans arranged on the floor-like surface.
- target: milk carton is visible among the objects, with a blue carton in the lower view and a red-and-...


## 8. Verdict

**RED**

Bands used:
- GREEN: mean mined cosine in [0.6, 0.95], mean Jaccard < 0.4, degenerate fraction < 0.1%
- YELLOW: exactly one of those off
- RED: degenerate fraction > 1% OR mean cosine outside [0.6, 0.95]

Inputs to the verdict:
- mean mined cosine = 0.9782
- mean Jaccard = 0.3850
- self-match pct = 0.0000%
- same-source pct = 0.0000%
- identical-caption pct = 0.0000%

## 9. Recommendations

1. **`last_text` activations are saturated** (mean mined cos = 0.9989; random `last_text` pairs also sit at 0.9592 → mining gains only Δ=0.0396). The final-token hidden state after a templated prompt is nearly identical across episodes, so top-K cosine over the final layer is mining *uniform noise*: any 'hard' negative is interchangeable with a random one. **Action:** mine `last_text` from an earlier layer (current activations are the last decoder layer; try `layer -4` or `-8`), or project the activation through a learned scene-encoder head and mine in that subspace, or drop `last_text` from the InfoNCE term entirely and keep hard-neg only on `image_patch` anchors (where mining is healthier — see slot 3).
2. **Caption Jaccard is moderate-high** (mean = 0.385 vs random pairs 0.285). Mining genuinely tightens caption similarity (Δ = +0.100), which is the desired effect — negatives describe similar but non-identical scenes. Watch the p90 (0.48); if it creeps above 0.6 the InfoNCE objective will start penalising legitimately similar captions. Optional: add a hard cap (`reject neg if jaccard(anchor, neg) > 0.7`) to the miner.
3. **Position-type asymmetry: `image_patch` mining gives real contrast, `last_text` does not.** image_patch mined-cos = 0.9577 vs random 0.7494 (Δ = +0.2083 — top-K is meaningfully tighter than random). last_text mined-cos = 0.9989 vs random 0.9592 (Δ = +0.0396 — mining barely tightens over random). The training-time `tau` has to compromise between two distributions of very different sharpness. **Action:** (a) apply z-score normalisation per `position_type` to the cosines before the softmax in `_hard_negative_sims`, or use separate temperatures `tau_last_text`, `tau_image_patch`; (b) better, drop `last_text` from InfoNCE (its mining signal is nonexistent — see slot 1) and let `image_patch` carry the contrastive term.
4. **Cross-suite balance is good** (88.5% within-suite; 11.5% cross). Within-suite negatives dominate (giving hard contrast) while 11.5% cross-suite provides breadth. No change.

## 10. Cross-references to other audit_reports outputs

- **Agent 1 (multimodal judge)**: B-axis grounding 91% (YELLOW), dragged down by `libero_spatial`. Implication for this audit: if spatial captions are less grounded, mined within-spatial negatives may share vague language → boosts Jaccard. Spatial-cell within-suite mining count here is `182,474` (~22.4% of all mined pairs). If spatial captions are filtered, the hard-neg index must be re-mined (the dataset loader rebuilds the candidate set against the in-split rows only — see `_build_topk_cosine_index` in `src/nla/training/dataset.py`).
- **Agent 2 (prompt-hardening regression)**: position-aware bullet conformance only 63.59% (RED) — significant for hard-neg mining because lower-quality captions = noisier Jaccard signal. The mining itself is unaffected (it uses activations, not captions), but the InfoNCE objective's *meaningfulness* depends on caption quality.
- **Agent 4 (bullet informativeness)**: `language:` bullet present in only 20.2% of labels; `language` is also the highest filler bullet (54.9%). This is consistent with the moderate Jaccard (~0.38) observed here: many bullets are scene/spatial/plan which are similar across nearby scenes.
- **Agent 3 (caption diversity)**: report not present at audit time. If it surfaces with high near-duplicate rate, expect Jaccard p90 to climb above 0.5; re-check this audit after their findings.
# Agent 3 — V3 LIBERO Caption Diversity Audit

Streaming audit of the 101,580 V3 LIBERO captions in `data/labels/libero_4suite_stride2/`, with V2 DROID and the LIBERO pilot as diversity baselines. Per-bullet vocabulary, n-gram, duplicate, and cross-suite distinguishability metrics.

## 1. Corpus inventory
| source | ≈records | total bullets | bullet types observed |
|---|---|---|---|
| libero_goal | 25676 | 126205 | action_head, action_link, distractor, gripper, image_region, language, motion, plan, scene, spatial, target |
| libero_spatial | 25920 | 127574 | action, action_head, distractor, gripper, image_region, language, motion, plan, scene, spatial, target |
| libero_object | 27240 | 134894 | distractor, gripper, image_region, language, motion, plan, scene, spatial, target |
| libero_10 | 22735 | 104642 | action_head, distractor, gripper, image_region, language, last_text, motion, plan, scene, spatial, target |
| droid_100ep_v2 | 100336 | 424094 | anchor, anchor (plan), anchor/plan, distractor, distractor/destination, distractor/gripper, distractor/gripper/spatial, distractor/image_region, distractor/motion, distractor/obstacle, distractor/secondary target, distractor/spatial, distractor/target, distractor/tool, gripper, gripper / distractor, gripper / spatial, gripper and motion, gripper/brush, gripper/distractor, gripper/image_region, gripper/motion, gripper/motion/plan, gripper/motion/spatial, gripper/plan, gripper/planning, gripper/spatial, gripper/spatial/motion, gripper/spatial/plan, gripper/tool, image_region, image_region / gripper, image_region/distractor, image_region/gripper, image_region/spatial, language, language (anchor), language/anchor, language/plan, motion, motion/gripper, motion/plan, motion/spatial, orange object, plan, plan (anchor), plan/gripper, plan/language, plan/motion, plan/spatial, scene, spatial, spatial/distractor, spatial/gripper, spatial/image_region, spatial/motion, spatial/motion/gripper, spatial/plan, target, target (destination), target (placement), target container, target/container, target/distractor, tool/gripper |
| libero_goal_pilot | 243 | 1211 | anchor, distractor, gripper, gripper/spatial, image_region, language, language/plan, plan, scene, spatial, target |

## 2. Per-(suite, bullet_type) vocabulary stats

TTR = unique unigrams / total tokens. Lower TTR + low unique n-grams = heavier template re-use.

| source | bullet | n_bullets | tokens | uniq 1g | uniq 2g | uniq 3g | TTR | exact dup % | near dup % | avg tok |
|---|---|---|---|---|---|---|---|---|---|---|
| libero_goal | language | 4327 | 46,196 | 323 | 1,591 | 3,026 | 0.0070 | 64.0% | 69.5% | 10.7 |
| libero_goal | target | 25676 | 430,701 | 979 | 10,157 | 29,056 | 0.0023 | 8.9% | 12.4% | 16.8 |
| libero_goal | scene | 25512 | 545,181 | 702 | 7,422 | 23,414 | 0.0013 | 5.1% | 6.6% | 21.4 |
| libero_goal | spatial | 25646 | 502,077 | 1,065 | 13,274 | 41,503 | 0.0021 | 1.2% | 2.1% | 19.6 |
| libero_goal | plan | 25352 | 456,283 | 870 | 9,568 | 28,265 | 0.0019 | 12.4% | 16.0% | 18.0 |
| libero_goal | action_head | 2 | 43 | 32 | 39 | 38 | 0.7442 | 0.0% | 0.0% | 21.5 |
| libero_goal | action_link | 1 | 14 | 14 | 13 | 12 | 1.0000 | 0.0% | 0.0% | 14.0 |
| libero_goal | distractor | 16943 | 315,495 | 917 | 9,324 | 29,340 | 0.0029 | 0.4% | 0.8% | 18.6 |
| libero_goal | gripper | 1071 | 16,646 | 351 | 1,597 | 2,969 | 0.0211 | 0.7% | 1.3% | 15.5 |
| libero_goal | image_region | 19 | 366 | 118 | 237 | 289 | 0.3224 | 0.0% | 0.0% | 19.3 |
| libero_goal | motion | 1656 | 26,362 | 491 | 2,976 | 5,776 | 0.0186 | 1.0% | 1.1% | 15.9 |
| libero_spatial | language | 5576 | 83,164 | 240 | 1,060 | 2,144 | 0.0029 | 64.7% | 73.5% | 14.9 |
| libero_spatial | target | 25920 | 433,543 | 838 | 8,440 | 24,571 | 0.0019 | 11.1% | 15.5% | 16.7 |
| libero_spatial | scene | 25461 | 509,327 | 630 | 5,950 | 18,537 | 0.0012 | 8.8% | 10.0% | 20.0 |
| libero_spatial | spatial | 25741 | 486,108 | 955 | 11,123 | 33,015 | 0.0020 | 2.8% | 4.2% | 18.9 |
| libero_spatial | plan | 24864 | 476,196 | 563 | 4,921 | 13,908 | 0.0012 | 30.2% | 39.4% | 19.2 |
| libero_spatial | action | 1 | 18 | 16 | 17 | 16 | 0.8889 | 0.0% | 0.0% | 18.0 |
| libero_spatial | action_head | 2 | 44 | 30 | 37 | 37 | 0.6818 | 0.0% | 0.0% | 22.0 |
| libero_spatial | distractor | 16937 | 308,507 | 816 | 7,933 | 24,506 | 0.0026 | 0.8% | 1.5% | 18.2 |
| libero_spatial | gripper | 2209 | 32,870 | 320 | 1,632 | 3,352 | 0.0097 | 2.5% | 4.5% | 14.9 |
| libero_spatial | image_region | 9 | 174 | 71 | 118 | 132 | 0.4080 | 0.0% | 0.0% | 19.3 |
| libero_spatial | motion | 854 | 13,011 | 271 | 1,293 | 2,467 | 0.0208 | 4.4% | 5.6% | 15.2 |
| libero_object | language | 6346 | 86,885 | 266 | 1,360 | 2,923 | 0.0031 | 66.8% | 74.7% | 13.7 |
| libero_object | target | 27240 | 479,787 | 893 | 9,258 | 28,670 | 0.0019 | 4.4% | 6.8% | 17.6 |
| libero_object | scene | 27158 | 533,135 | 501 | 4,444 | 13,136 | 0.0009 | 9.9% | 12.2% | 19.6 |
| libero_object | spatial | 27128 | 523,813 | 822 | 9,735 | 31,108 | 0.0016 | 0.8% | 1.7% | 19.3 |
| libero_object | plan | 26258 | 462,818 | 578 | 5,298 | 15,462 | 0.0012 | 26.2% | 33.1% | 17.6 |
| libero_object | distractor | 19767 | 358,428 | 711 | 7,555 | 27,161 | 0.0020 | 0.5% | 0.9% | 18.1 |
| libero_object | gripper | 638 | 9,902 | 281 | 1,177 | 2,196 | 0.0284 | 0.2% | 0.3% | 15.5 |
| libero_object | image_region | 10 | 200 | 83 | 140 | 160 | 0.4150 | 0.0% | 0.0% | 20.0 |
| libero_object | motion | 349 | 5,635 | 244 | 901 | 1,567 | 0.0433 | 0.0% | 0.0% | 16.1 |
| libero_10 | language | 4267 | 67,825 | 315 | 1,686 | 3,557 | 0.0046 | 40.9% | 51.3% | 15.9 |
| libero_10 | target | 22735 | 414,781 | 1,127 | 12,888 | 37,086 | 0.0027 | 5.8% | 8.3% | 18.2 |
| libero_10 | scene | 22057 | 442,218 | 751 | 7,193 | 19,859 | 0.0017 | 9.7% | 12.5% | 20.0 |
| libero_10 | spatial | 20905 | 404,987 | 1,083 | 12,862 | 37,882 | 0.0027 | 3.5% | 5.0% | 19.4 |
| libero_10 | plan | 22130 | 451,824 | 953 | 10,959 | 33,242 | 0.0021 | 11.4% | 15.5% | 20.4 |
| libero_10 | action_head | 1 | 26 | 18 | 22 | 22 | 0.6923 | 0.0% | 0.0% | 26.0 |
| libero_10 | distractor | 9007 | 155,246 | 939 | 7,826 | 19,886 | 0.0060 | 9.0% | 12.1% | 17.2 |
| libero_10 | gripper | 1851 | 28,731 | 452 | 2,387 | 4,958 | 0.0157 | 0.2% | 0.8% | 15.5 |
| libero_10 | image_region | 20 | 392 | 122 | 246 | 295 | 0.3112 | 0.0% | 0.0% | 19.6 |
| libero_10 | last_text | 2 | 39 | 26 | 37 | 35 | 0.6667 | 0.0% | 0.0% | 19.5 |
| libero_10 | motion | 1667 | 26,795 | 487 | 2,927 | 5,871 | 0.0182 | 0.4% | 0.9% | 16.1 |
| droid_100ep_v2 | language | 18346 | 412,507 | 1,196 | 11,267 | 29,071 | 0.0029 | 15.1% | 20.5% | 22.5 |
| droid_100ep_v2 | target | 100336 | 2,382,184 | 3,758 | 89,376 | 330,653 | 0.0016 | 0.3% | 0.3% | 23.7 |
| droid_100ep_v2 | scene | 100336 | 2,859,747 | 3,203 | 73,676 | 286,606 | 0.0011 | 0.0% | 0.1% | 28.5 |
| droid_100ep_v2 | spatial | 68045 | 2,076,850 | 3,267 | 78,114 | 291,267 | 0.0016 | 0.0% | 0.0% | 30.5 |
| droid_100ep_v2 | plan | 7251 | 187,297 | 1,309 | 12,197 | 31,027 | 0.0070 | 0.5% | 0.9% | 25.8 |
| droid_100ep_v2 | anchor | 323 | 7,938 | 444 | 1,914 | 3,423 | 0.0559 | 0.0% | 0.0% | 24.6 |
| droid_100ep_v2 | anchor (plan) | 1 | 21 | 18 | 19 | 19 | 0.8571 | 0.0% | 0.0% | 21.0 |
| droid_100ep_v2 | anchor/plan | 23 | 582 | 166 | 357 | 454 | 0.2852 | 0.0% | 0.0% | 25.3 |
| droid_100ep_v2 | distractor | 68852 | 1,742,214 | 3,816 | 82,348 | 285,722 | 0.0022 | 0.0% | 0.0% | 25.3 |
| droid_100ep_v2 | distractor/destination | 3 | 81 | 48 | 72 | 73 | 0.5926 | 0.0% | 0.0% | 27.0 |
| droid_100ep_v2 | distractor/gripper | 143 | 4,061 | 539 | 1,765 | 2,655 | 0.1327 | 0.0% | 0.0% | 28.4 |
| droid_100ep_v2 | distractor/gripper/spatial | 1 | 42 | 34 | 41 | 40 | 0.8095 | 0.0% | 0.0% | 42.0 |
| droid_100ep_v2 | distractor/image_region | 26 | 729 | 267 | 535 | 649 | 0.3663 | 0.0% | 0.0% | 28.0 |
| droid_100ep_v2 | distractor/motion | 1 | 32 | 24 | 30 | 30 | 0.7500 | 0.0% | 0.0% | 32.0 |
| droid_100ep_v2 | distractor/obstacle | 1 | 24 | 20 | 23 | 22 | 0.8333 | 0.0% | 0.0% | 24.0 |
| droid_100ep_v2 | distractor/secondary target | 1 | 29 | 20 | 25 | 27 | 0.6897 | 0.0% | 0.0% | 29.0 |
| droid_100ep_v2 | distractor/spatial | 410 | 12,515 | 1,029 | 5,057 | 8,169 | 0.0822 | 0.0% | 0.0% | 30.5 |
| droid_100ep_v2 | distractor/target | 5 | 149 | 83 | 125 | 137 | 0.5570 | 0.0% | 0.0% | 29.8 |
| droid_100ep_v2 | distractor/tool | 8 | 219 | 77 | 136 | 169 | 0.3516 | 0.0% | 0.0% | 27.4 |
| droid_100ep_v2 | gripper | 53171 | 1,281,862 | 2,223 | 36,713 | 128,890 | 0.0017 | 0.0% | 0.0% | 24.1 |
| droid_100ep_v2 | gripper / distractor | 1 | 33 | 31 | 32 | 31 | 0.9394 | 0.0% | 0.0% | 33.0 |
| droid_100ep_v2 | gripper / spatial | 1 | 35 | 25 | 33 | 33 | 0.7143 | 0.0% | 0.0% | 35.0 |
| droid_100ep_v2 | gripper and motion | 1 | 26 | 24 | 25 | 24 | 0.9231 | 0.0% | 0.0% | 26.0 |
| droid_100ep_v2 | gripper/brush | 10 | 251 | 72 | 155 | 193 | 0.2869 | 0.0% | 0.0% | 25.1 |
| droid_100ep_v2 | gripper/distractor | 115 | 3,261 | 481 | 1,554 | 2,288 | 0.1475 | 0.0% | 0.0% | 28.4 |
| droid_100ep_v2 | gripper/image_region | 196 | 5,459 | 446 | 1,883 | 3,138 | 0.0817 | 0.0% | 0.0% | 27.9 |
| droid_100ep_v2 | gripper/motion | 436 | 11,676 | 648 | 3,361 | 6,152 | 0.0555 | 0.0% | 0.0% | 26.8 |
| droid_100ep_v2 | gripper/motion/plan | 2 | 68 | 48 | 63 | 64 | 0.7059 | 0.0% | 0.0% | 34.0 |
| droid_100ep_v2 | gripper/motion/spatial | 2 | 64 | 47 | 58 | 59 | 0.7344 | 0.0% | 0.0% | 32.0 |
| droid_100ep_v2 | gripper/plan | 229 | 7,281 | 489 | 2,276 | 4,031 | 0.0672 | 0.0% | 0.0% | 31.8 |
| droid_100ep_v2 | gripper/planning | 1 | 39 | 30 | 37 | 37 | 0.7692 | 0.0% | 0.0% | 39.0 |
| droid_100ep_v2 | gripper/spatial | 3743 | 109,102 | 1,251 | 12,184 | 30,620 | 0.0115 | 0.0% | 0.0% | 29.1 |
| droid_100ep_v2 | gripper/spatial/motion | 7 | 220 | 99 | 176 | 196 | 0.4500 | 0.0% | 0.0% | 31.4 |
| droid_100ep_v2 | gripper/spatial/plan | 12 | 417 | 146 | 299 | 360 | 0.3501 | 0.0% | 0.0% | 34.8 |
| droid_100ep_v2 | gripper/tool | 27 | 623 | 109 | 265 | 385 | 0.1750 | 0.0% | 0.0% | 23.1 |
| droid_100ep_v2 | image_region | 666 | 17,467 | 994 | 5,522 | 9,756 | 0.0569 | 0.0% | 0.0% | 26.2 |
| droid_100ep_v2 | image_region / gripper | 1 | 24 | 21 | 23 | 22 | 0.8750 | 0.0% | 0.0% | 24.0 |
| droid_100ep_v2 | image_region/distractor | 2 | 67 | 43 | 59 | 63 | 0.6418 | 0.0% | 0.0% | 33.5 |
| droid_100ep_v2 | image_region/gripper | 45 | 1,220 | 243 | 648 | 896 | 0.1992 | 0.0% | 0.0% | 27.1 |
| droid_100ep_v2 | image_region/spatial | 10 | 278 | 120 | 216 | 247 | 0.4317 | 0.0% | 0.0% | 27.8 |
| droid_100ep_v2 | language (anchor) | 9 | 231 | 98 | 172 | 195 | 0.4242 | 0.0% | 0.0% | 25.7 |
| droid_100ep_v2 | language/anchor | 3 | 86 | 46 | 67 | 72 | 0.5349 | 0.0% | 0.0% | 28.7 |
| droid_100ep_v2 | language/plan | 32 | 1,026 | 201 | 482 | 640 | 0.1959 | 0.0% | 0.0% | 32.1 |
| droid_100ep_v2 | motion | 160 | 3,848 | 482 | 1,733 | 2,608 | 0.1253 | 0.0% | 0.0% | 24.1 |
| droid_100ep_v2 | motion/gripper | 47 | 1,200 | 244 | 661 | 911 | 0.2033 | 0.0% | 0.0% | 25.5 |
| droid_100ep_v2 | motion/plan | 38 | 1,006 | 290 | 651 | 812 | 0.2883 | 0.0% | 0.0% | 26.5 |
| droid_100ep_v2 | motion/spatial | 46 | 1,277 | 297 | 745 | 995 | 0.2326 | 0.0% | 0.0% | 27.8 |
| droid_100ep_v2 | orange object | 1 | 25 | 22 | 24 | 23 | 0.8800 | 0.0% | 0.0% | 25.0 |
| droid_100ep_v2 | plan (anchor) | 4 | 111 | 58 | 91 | 95 | 0.5225 | 0.0% | 0.0% | 27.8 |
| droid_100ep_v2 | plan/gripper | 12 | 387 | 145 | 289 | 338 | 0.3747 | 0.0% | 0.0% | 32.2 |
| droid_100ep_v2 | plan/language | 2 | 56 | 36 | 45 | 45 | 0.6429 | 0.0% | 0.0% | 28.0 |
| droid_100ep_v2 | plan/motion | 16 | 457 | 182 | 340 | 389 | 0.3982 | 0.0% | 0.0% | 28.6 |
| droid_100ep_v2 | plan/spatial | 20 | 648 | 212 | 461 | 567 | 0.3272 | 0.0% | 0.0% | 32.4 |
| droid_100ep_v2 | spatial/distractor | 44 | 1,298 | 324 | 840 | 1,068 | 0.2496 | 0.0% | 0.0% | 29.5 |
| droid_100ep_v2 | spatial/gripper | 217 | 6,627 | 506 | 2,235 | 3,806 | 0.0764 | 0.0% | 0.0% | 30.5 |
| droid_100ep_v2 | spatial/image_region | 16 | 477 | 180 | 359 | 420 | 0.3774 | 0.0% | 0.0% | 29.8 |
| droid_100ep_v2 | spatial/motion | 125 | 3,477 | 432 | 1,585 | 2,405 | 0.1242 | 0.0% | 0.0% | 27.8 |
| droid_100ep_v2 | spatial/motion/gripper | 1 | 43 | 32 | 39 | 39 | 0.7442 | 0.0% | 0.0% | 43.0 |
| droid_100ep_v2 | spatial/plan | 499 | 15,938 | 784 | 4,752 | 8,696 | 0.0492 | 0.0% | 0.0% | 31.9 |
| droid_100ep_v2 | target (destination) | 3 | 71 | 45 | 61 | 62 | 0.6338 | 0.0% | 0.0% | 23.7 |
| droid_100ep_v2 | target (placement) | 1 | 21 | 18 | 20 | 19 | 0.8571 | 0.0% | 0.0% | 21.0 |
| droid_100ep_v2 | target container | 1 | 21 | 20 | 20 | 19 | 0.9524 | 0.0% | 0.0% | 21.0 |
| droid_100ep_v2 | target/container | 1 | 30 | 24 | 29 | 28 | 0.8000 | 0.0% | 0.0% | 30.0 |
| droid_100ep_v2 | target/distractor | 5 | 146 | 78 | 122 | 131 | 0.5342 | 0.0% | 0.0% | 29.2 |
| droid_100ep_v2 | tool/gripper | 1 | 20 | 18 | 19 | 18 | 0.9000 | 0.0% | 0.0% | 20.0 |
| libero_goal_pilot | language | 121 | 2,413 | 160 | 445 | 707 | 0.0663 | 7.4% | 7.4% | 19.9 |
| libero_goal_pilot | target | 243 | 5,252 | 374 | 1,606 | 2,597 | 0.0712 | 0.0% | 0.0% | 21.6 |
| libero_goal_pilot | scene | 243 | 7,702 | 345 | 1,720 | 3,249 | 0.0448 | 0.0% | 0.0% | 31.7 |
| libero_goal_pilot | spatial | 181 | 5,192 | 422 | 1,842 | 3,129 | 0.0813 | 0.0% | 0.0% | 28.7 |
| libero_goal_pilot | plan | 27 | 656 | 111 | 256 | 352 | 0.1692 | 0.0% | 0.0% | 24.3 |
| libero_goal_pilot | anchor | 1 | 34 | 27 | 33 | 32 | 0.7941 | 0.0% | 0.0% | 34.0 |
| libero_goal_pilot | distractor | 219 | 5,555 | 386 | 1,681 | 2,798 | 0.0695 | 0.0% | 0.0% | 25.4 |
| libero_goal_pilot | gripper | 72 | 1,654 | 187 | 613 | 950 | 0.1131 | 0.0% | 0.0% | 23.0 |
| libero_goal_pilot | gripper/spatial | 3 | 85 | 49 | 74 | 79 | 0.5765 | 0.0% | 0.0% | 28.3 |
| libero_goal_pilot | image_region | 100 | 3,247 | 290 | 1,061 | 1,720 | 0.0893 | 0.0% | 0.0% | 32.5 |
| libero_goal_pilot | language/plan | 1 | 29 | 23 | 28 | 27 | 0.7931 | 0.0% | 0.0% | 29.0 |

## 3. Top-20 most common phrases per bullet type (V3 LIBERO combined)

Combined across the 4 V3 suites. Phrase % is *document frequency* (fraction of bullets containing the phrase at least once). Items with **bold** % cross the 5% boilerplate threshold.

### language  (n_bullets=20,516)
**Top-20 bigrams (document frequency)**

| bigram | DF count | % of bullets |
|---|---|---|
| and place | 9,987 | **48.7%** |
| pick up | 9,480 | **46.2%** |
| place it | 8,900 | **43.4%** |
| up the | 8,079 | **39.4%** |
| the basket | 7,814 | **38.1%** |
| in the | 6,485 | **31.6%** |
| the plate | 6,154 | **30.0%** |
| parsed as | 6,036 | **29.4%** |
| black bowl | 5,657 | **27.6%** |
| on the | 5,362 | **26.1%** |
| it in | 5,338 | **26.0%** |
| the black | 4,606 | **22.5%** |
| it on | 3,769 | **18.4%** |
| instruction parsed | 3,652 | **17.8%** |
| parsed instruction | 3,412 | **16.6%** |
| cream cheese | 2,605 | **12.7%** |
| into the | 2,457 | **12.0%** |
| bowl and | 2,403 | **11.7%** |
| as pick | 2,259 | **11.0%** |
| instruction specifies | 2,195 | **10.7%** |

**Top-20 trigrams (document frequency)**

| trigram | DF count | % of bullets |
|---|---|---|
| and place it | 8,613 | **42.0%** |
| pick up the | 7,706 | **37.6%** |
| in the basket | 6,030 | **29.4%** |
| it in the | 5,336 | **26.0%** |
| place it in | 5,200 | **25.3%** |
| the black bowl | 4,601 | **22.4%** |
| on the plate | 4,300 | **21.0%** |
| it on the | 3,756 | **18.3%** |
| up the black | 3,749 | **18.3%** |
| instruction parsed as | 3,411 | **16.6%** |
| place it on | 3,386 | **16.5%** |
| black bowl and | 2,399 | **11.7%** |
| as pick up | 2,257 | **11.0%** |
| parsed as pick | 2,143 | **10.4%** |
| bowl and place | 1,978 | **9.6%** |
| into the basket | 1,695 | **8.3%** |
| is pick up | 1,500 | **7.3%** |
| to pick up | 1,409 | **6.9%** |
| cream cheese box | 1,334 | **6.5%** |
| parsed task is | 1,316 | **6.4%** |

### target  (n_bullets=101,571)
**Top-20 bigrams (document frequency)**

| bigram | DF count | % of bullets |
|---|---|---|
| near the | 45,802 | **45.1%** |
| is the | 45,000 | **44.3%** |
| of the | 35,993 | **35.4%** |
| black bowl | 25,696 | **25.3%** |
| on the | 25,418 | **25.0%** |
| the center | 21,920 | **21.6%** |
| the object | 19,507 | **19.2%** |
| in the | 16,943 | **16.7%** |
| with a | 15,180 | **14.9%** |
| the table | 14,899 | **14.7%** |
| object to | 14,829 | **14.6%** |
| bowl sits | 13,459 | **13.3%** |
| bowl is | 12,677 | **12.5%** |
| center of | 12,618 | **12.4%** |
| the plate | 9,408 | **9.3%** |
| the workspace | 8,898 | **8.8%** |
| the lower | 8,866 | **8.7%** |
| the robot | 8,354 | **8.2%** |
| is visible | 8,124 | **8.0%** |
| cream cheese | 8,017 | **7.9%** |

**Top-20 trigrams (document frequency)**

| trigram | DF count | % of bullets |
|---|---|---|
| near the center | 21,491 | **21.2%** |
| is the object | 17,063 | **16.8%** |
| the object to | 13,798 | **13.6%** |
| center of the | 12,610 | **12.4%** |
| black bowl sits | 10,765 | **10.6%** |
| the center of | 10,643 | **10.5%** |
| bowl is the | 10,391 | **10.2%** |
| of the workspace | 7,998 | **7.9%** |
| of the table | 7,360 | **7.2%** |
| on the table | 6,925 | **6.8%** |
| sits near the | 6,809 | **6.7%** |
| the center right | 5,872 | **5.8%** |
| black bowl is | 5,474 | **5.4%** |
| near the lower | 5,284 | **5.2%** |
| bottle is the | 5,020 | 4.9% |
| in the close | 5,016 | 4.9% |
| bowl sits on | 4,998 | 4.9% |
| right of the | 4,994 | 4.9% |
| black bowl with | 4,879 | 4.8% |
| the object named | 4,767 | 4.7% |

### scene  (n_bullets=100,188)
**Top-20 bigrams (document frequency)**

| bigram | DF count | % of bullets |
|---|---|---|
| with a | 83,393 | **83.2%** |
| and a | 52,211 | **52.1%** |
| robot arm | 49,878 | **49.8%** |
| on the | 48,805 | **48.7%** |
| workspace with | 46,457 | **46.4%** |
| tabletop workspace | 42,336 | **42.3%** |
| a black | 36,962 | **36.9%** |
| the left | 35,180 | **35.1%** |
| a white | 33,966 | **33.9%** |
| a robot | 28,592 | **28.5%** |
| tabletop with | 22,653 | **22.6%** |
| basket on | 20,267 | **20.2%** |
| the right | 19,116 | **19.1%** |
| woven basket | 17,838 | **17.8%** |
| arm a | 16,773 | **16.7%** |
| and several | 16,681 | **16.6%** |
| left and | 16,510 | **16.5%** |
| white plate | 16,029 | **16.0%** |
| a gray | 15,363 | **15.3%** |
| light wood | 15,187 | **15.2%** |

**Top-20 trigrams (document frequency)**

| trigram | DF count | % of bullets |
|---|---|---|
| workspace with a | 44,184 | **44.1%** |
| tabletop workspace with | 42,155 | **42.1%** |
| on the left | 31,428 | **31.4%** |
| a robot arm | 26,832 | **26.8%** |
| basket on the | 20,224 | **20.2%** |
| tabletop with a | 19,317 | **19.3%** |
| with a robot | 18,283 | **18.2%** |
| with a white | 15,810 | **15.8%** |
| robot arm a | 15,783 | **15.8%** |
| the left and | 15,531 | **15.5%** |
| robot arm above | 14,493 | **14.5%** |
| on the right | 13,566 | **13.5%** |
| a white plate | 13,333 | **13.3%** |
| a black cabinet | 13,120 | **13.1%** |
| a light wood | 10,884 | **10.9%** |
| woven basket on | 10,698 | **10.7%** |
| a woven basket | 9,965 | **9.9%** |
| with a gray | 9,574 | **9.6%** |
| and a black | 9,461 | **9.4%** |
| left and several | 9,380 | **9.4%** |

### spatial  (n_bullets=99,420)
**Top-20 bigrams (document frequency)**

| bigram | DF count | % of bullets |
|---|---|---|
| of the | 40,812 | **41.1%** |
| with the | 29,554 | **29.7%** |
| to the | 28,382 | **28.5%** |
| on the | 27,496 | **27.7%** |
| the plate | 25,493 | **25.6%** |
| the bowl | 23,900 | **24.0%** |
| the basket | 23,673 | **23.8%** |
| the robot | 20,675 | **20.8%** |
| from the | 19,054 | **19.2%** |
| and the | 17,836 | **17.9%** |
| left of | 17,684 | **17.8%** |
| bowl is | 17,421 | **17.5%** |
| in the | 14,196 | **14.3%** |
| the right | 13,659 | **13.7%** |
| near the | 13,476 | **13.6%** |
| separated from | 13,415 | **13.5%** |
| the table | 13,316 | **13.4%** |
| the cabinet | 11,755 | **11.8%** |
| above the | 11,489 | **11.6%** |
| between the | 11,335 | **11.4%** |

**Top-20 trigrams (document frequency)**

| trigram | DF count | % of bullets |
|---|---|---|
| left of the | 16,775 | **16.9%** |
| separated from the | 11,471 | **11.5%** |
| in front of | 9,847 | **9.9%** |
| front of the | 9,189 | **9.2%** |
| to the right | 8,231 | **8.3%** |
| is separated from | 8,049 | **8.1%** |
| on the table | 7,963 | **8.0%** |
| the bowl is | 7,099 | **7.1%** |
| right of the | 6,943 | **7.0%** |
| adjacent to the | 6,749 | **6.8%** |
| the basket is | 6,691 | **6.7%** |
| the right of | 6,169 | **6.2%** |
| the robot arm | 5,782 | **5.8%** |
| with the plate | 5,596 | **5.6%** |
| the plate is | 5,311 | **5.3%** |
| the robot gripper | 5,208 | **5.2%** |
| the basket and | 5,190 | **5.2%** |
| of the robot | 5,020 | **5.0%** |
| from the basket | 5,012 | **5.0%** |
| is left of | 4,769 | 4.8% |

### plan  (n_bullets=98,604)
**Top-20 bigrams (document frequency)**

| bigram | DF count | % of bullets |
|---|---|---|
| toward the | 47,048 | **47.7%** |
| carry it | 32,624 | **33.1%** |
| the basket | 31,775 | **32.2%** |
| to the | 31,314 | **31.8%** |
| into the | 28,843 | **29.3%** |
| the plate | 28,151 | **28.5%** |
| and place | 27,967 | **28.4%** |
| grasp the | 26,699 | **27.1%** |
| phase active | 26,433 | **26.8%** |
| it toward | 26,043 | **26.4%** |
| carries the | 22,691 | **23.0%** |
| the bowl | 21,823 | **22.1%** |
| the black | 20,574 | **20.9%** |
| black bowl | 20,212 | **20.5%** |
| and carry | 20,205 | **20.5%** |
| it to | 19,339 | **19.6%** |
| place phase | 18,721 | **19.0%** |
| pick and | 17,906 | **18.2%** |
| action head | 17,072 | **17.3%** |
| the action | 16,761 | **17.0%** |

**Top-20 trigrams (document frequency)**

| trigram | DF count | % of bullets |
|---|---|---|
| it toward the | 26,037 | **26.4%** |
| the black bowl | 20,032 | **20.3%** |
| and carry it | 19,392 | **19.7%** |
| it to the | 19,222 | **19.5%** |
| and place phase | 17,897 | **18.2%** |
| pick and place | 17,869 | **18.1%** |
| the action head | 16,454 | **16.7%** |
| reach toward the | 15,266 | **15.5%** |
| carry it to | 15,077 | **15.3%** |
| carry it toward | 14,917 | **15.1%** |
| into the action | 14,805 | **15.0%** |
| place phase active | 13,937 | **14.1%** |
| toward the basket | 13,535 | **13.7%** |
| then carry it | 12,670 | **12.8%** |
| toward the plate | 11,742 | **11.9%** |
| this patch carries | 11,692 | **11.9%** |
| patch carries the | 10,913 | **11.1%** |
| and move it | 9,478 | **9.6%** |
| token carries the | 9,286 | **9.4%** |
| to the plate | 9,003 | **9.1%** |

## 4. Boilerplate phrases (content n-grams ≥5% of V3 bullets per type)

Filtered to **content-bearing** n-grams only (bigrams with ≥1 non-function-word token, trigrams with ≥2). Stopword bigrams like `with a`/`on the`/`of the` are excluded from this table — they are covered in Section 3.

| bullet | n-gram type | phrase | DF count | % of bullets |
|---|---|---|---|---|
| scene | 2-gram | robot arm | 49,878 | 49.8% |
| language | 2-gram | and place | 9,987 | 48.7% |
| plan | 2-gram | toward the | 47,048 | 47.7% |
| scene | 2-gram | workspace with | 46,457 | 46.4% |
| language | 2-gram | pick up | 9,480 | 46.2% |
| target | 2-gram | near the | 45,802 | 45.1% |
| language | 2-gram | place it | 8,900 | 43.4% |
| scene | 2-gram | tabletop workspace | 42,336 | 42.3% |
| scene | 3-gram | tabletop workspace with | 42,155 | 42.1% |
| language | 2-gram | up the | 8,079 | 39.4% |
| language | 2-gram | the basket | 7,814 | 38.1% |
| language | 3-gram | pick up the | 7,706 | 37.6% |
| scene | 2-gram | a black | 36,962 | 36.9% |
| scene | 2-gram | the left | 35,180 | 35.1% |
| scene | 2-gram | a white | 33,966 | 33.9% |
| plan | 2-gram | carry it | 32,624 | 33.1% |
| plan | 2-gram | the basket | 31,775 | 32.2% |
| language | 2-gram | the plate | 6,154 | 30.0% |
| language | 2-gram | parsed as | 6,036 | 29.4% |
| scene | 2-gram | a robot | 28,592 | 28.5% |
| plan | 2-gram | the plate | 28,151 | 28.5% |
| plan | 2-gram | and place | 27,967 | 28.4% |
| language | 2-gram | black bowl | 5,657 | 27.6% |
| plan | 2-gram | grasp the | 26,699 | 27.1% |
| scene | 3-gram | a robot arm | 26,832 | 26.8% |
| plan | 2-gram | phase active | 26,433 | 26.8% |
| plan | 2-gram | it toward | 26,043 | 26.4% |
| spatial | 2-gram | the plate | 25,493 | 25.6% |
| target | 2-gram | black bowl | 25,696 | 25.3% |
| spatial | 2-gram | the bowl | 23,900 | 24.0% |
| spatial | 2-gram | the basket | 23,673 | 23.8% |
| plan | 2-gram | carries the | 22,691 | 23.0% |
| scene | 2-gram | tabletop with | 22,653 | 22.6% |
| language | 2-gram | the black | 4,606 | 22.5% |
| language | 3-gram | the black bowl | 4,601 | 22.4% |
| plan | 2-gram | the bowl | 21,823 | 22.1% |
| target | 2-gram | the center | 21,920 | 21.6% |
| target | 3-gram | near the center | 21,491 | 21.2% |
| plan | 2-gram | the black | 20,574 | 20.9% |
| spatial | 2-gram | the robot | 20,675 | 20.8% |
| plan | 2-gram | black bowl | 20,212 | 20.5% |
| plan | 2-gram | and carry | 20,205 | 20.5% |
| plan | 3-gram | the black bowl | 20,032 | 20.3% |
| scene | 2-gram | basket on | 20,267 | 20.2% |
| target | 2-gram | the object | 19,507 | 19.2% |
| scene | 2-gram | the right | 19,116 | 19.1% |
| plan | 2-gram | place phase | 18,721 | 19.0% |
| language | 3-gram | up the black | 3,749 | 18.3% |
| plan | 3-gram | and place phase | 17,897 | 18.2% |
| plan | 2-gram | pick and | 17,906 | 18.2% |
| plan | 3-gram | pick and place | 17,869 | 18.1% |
| language | 2-gram | instruction parsed | 3,652 | 17.8% |
| scene | 2-gram | woven basket | 17,838 | 17.8% |
| spatial | 2-gram | left of | 17,684 | 17.8% |
| spatial | 2-gram | bowl is | 17,421 | 17.5% |
| plan | 2-gram | action head | 17,072 | 17.3% |
| plan | 2-gram | the action | 16,761 | 17.0% |
| plan | 2-gram | move it | 16,540 | 16.8% |
| scene | 2-gram | arm a | 16,773 | 16.7% |
| plan | 3-gram | the action head | 16,454 | 16.7% |
| language | 3-gram | instruction parsed as | 3,411 | 16.6% |
| language | 2-gram | parsed instruction | 3,412 | 16.6% |
| scene | 2-gram | and several | 16,681 | 16.6% |
| scene | 2-gram | left and | 16,510 | 16.5% |
| scene | 2-gram | white plate | 16,029 | 16.0% |
| scene | 3-gram | robot arm a | 15,783 | 15.8% |
| plan | 2-gram | reach toward | 15,377 | 15.6% |
| plan | 3-gram | reach toward the | 15,266 | 15.5% |
| scene | 2-gram | a gray | 15,363 | 15.3% |
| scene | 2-gram | light wood | 15,187 | 15.2% |
| plan | 3-gram | carry it toward | 14,917 | 15.1% |
| scene | 2-gram | arm above | 14,811 | 14.8% |
| target | 2-gram | the table | 14,899 | 14.7% |
| target | 2-gram | object to | 14,829 | 14.6% |
| plan | 2-gram | pick up | 14,429 | 14.6% |
| scene | 3-gram | robot arm above | 14,493 | 14.5% |
| scene | 2-gram | black cabinet | 14,494 | 14.5% |
| plan | 2-gram | for placement | 14,220 | 14.4% |
| plan | 3-gram | place phase active | 13,937 | 14.1% |
| scene | 2-gram | a light | 13,935 | 13.9% |

## 5. Cross-suite distinguishability

TF-IDF (1-2 gram) + LogReg classifier, fit on a balanced sample of bullets per suite (cap 2,000 per suite per bullet), 75/25 split. Chance = 0.25 with 4 suites. Higher accuracy ⇒ the labeler is writing suite-specific content; near-chance ⇒ boilerplate that ignores the underlying task.

| bullet_type | n_samples | accuracy | macro_F1 | baseline |
|---|---|---|---|---|
| language | 8000 | 0.998 | 0.998 | chance=0.250 |
| target | 8000 | 0.962 | 0.961 | chance=0.250 |
| scene | 8000 | 0.958 | 0.957 | chance=0.250 |
| spatial | 8000 | 0.946 | 0.946 | chance=0.250 |
| plan | 8000 | 0.977 | 0.976 | chance=0.250 |

**Confusion matrix — `language`** (rows=true, cols=pred, order=['libero_10', 'libero_goal', 'libero_object', 'libero_spatial'])

|  | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---|---|---|---|
| libero_10 | 499 | 0 | 1 | 0 |
| libero_goal | 0 | 499 | 1 | 0 |
| libero_object | 0 | 0 | 500 | 0 |
| libero_spatial | 1 | 0 | 1 | 498 |

**Confusion matrix — `target`** (rows=true, cols=pred, order=['libero_10', 'libero_goal', 'libero_object', 'libero_spatial'])

|  | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---|---|---|---|
| libero_10 | 462 | 12 | 8 | 18 |
| libero_goal | 8 | 481 | 4 | 7 |
| libero_object | 11 | 2 | 487 | 0 |
| libero_spatial | 0 | 7 | 0 | 493 |

**Confusion matrix — `scene`** (rows=true, cols=pred, order=['libero_10', 'libero_goal', 'libero_object', 'libero_spatial'])

|  | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---|---|---|---|
| libero_10 | 477 | 4 | 17 | 2 |
| libero_goal | 3 | 477 | 0 | 20 |
| libero_object | 9 | 0 | 491 | 0 |
| libero_spatial | 1 | 28 | 1 | 470 |

**Confusion matrix — `spatial`** (rows=true, cols=pred, order=['libero_10', 'libero_goal', 'libero_object', 'libero_spatial'])

|  | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---|---|---|---|
| libero_10 | 463 | 14 | 23 | 0 |
| libero_goal | 6 | 463 | 2 | 29 |
| libero_object | 1 | 3 | 496 | 0 |
| libero_spatial | 2 | 28 | 0 | 470 |

**Confusion matrix — `plan`** (rows=true, cols=pred, order=['libero_10', 'libero_goal', 'libero_object', 'libero_spatial'])

|  | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---|---|---|---|
| libero_10 | 492 | 2 | 5 | 1 |
| libero_goal | 1 | 486 | 1 | 12 |
| libero_object | 1 | 0 | 499 | 0 |
| libero_spatial | 1 | 23 | 0 | 476 |

## 6. V3 vs V2 DROID vs Pilot — diversity comparison

Aggregated per bullet type. V3 columns combine the four LIBERO suites; DROID and Pilot are reported as-is.

| bullet | source | n_bullets | uniq 1g | uniq 2g | uniq 3g | TTR | near dup % |
|---|---|---|---|---|---|---|---|
| language | V3 LIBERO | 20,516 | 594 | 4,104 | 9,690 | 0.0021 | 68.4% |
| language | V2 DROID | 18,346 | 1,196 | 11,267 | 29,071 | 0.0029 | 20.5% |
| language | Pilot | 121 | 160 | 445 | 707 | 0.0663 | 7.4% |
| target | V3 LIBERO | 101,571 | 1,680 | 25,686 | 89,783 | 0.0010 | 10.8% |
| target | V2 DROID | 100,336 | 3,758 | 89,376 | 330,653 | 0.0016 | 0.3% |
| target | Pilot | 243 | 374 | 1,606 | 2,597 | 0.0712 | 0.0% |
| scene | V3 LIBERO | 100,188 | 1,128 | 16,169 | 58,094 | 0.0006 | 10.5% |
| scene | V2 DROID | 100,336 | 3,203 | 73,676 | 286,606 | 0.0011 | 0.1% |
| scene | Pilot | 243 | 345 | 1,720 | 3,249 | 0.0448 | 0.0% |
| spatial | V3 LIBERO | 99,420 | 1,629 | 28,882 | 105,904 | 0.0008 | 3.1% |
| spatial | V2 DROID | 68,045 | 3,267 | 78,114 | 291,267 | 0.0016 | 0.0% |
| spatial | Pilot | 181 | 422 | 1,842 | 3,129 | 0.0813 | 0.0% |
| plan | V3 LIBERO | 98,604 | 1,320 | 19,473 | 68,644 | 0.0007 | 26.4% |
| plan | V2 DROID | 7,251 | 1,309 | 12,197 | 31,027 | 0.0070 | 0.9% |
| plan | Pilot | 27 | 111 | 256 | 352 | 0.1692 | 0.0% |

## 7. Top near-duplicate clusters (10 examples, 3 members each)

Picked from the most heavily reused normalized `plan` bullets — the bullet most prone to template collapse. Each cluster shares the same content-token set (after stopword removal). **Note:** even the largest cluster is <1% of all `plan` bullets, yet structural n-gram templates (e.g. `phase active`, `pick and place`, `carry it toward the …`) blanket 15–27% of the corpus — meaning the labeler swaps object names but locks in the same scaffold, so hash-bucket clusters under-state the true template collapse.

**Cluster #1** — appears in 163 `plan` bullets (0.2% of all V3 plan bullets)

- reach toward the black bowl, grasp it, and carry it to the plate.
- reach toward the black bowl, grasp it, and carry it to the plate.
- reach toward the black bowl, grasp it, then carry it to the plate.

**Cluster #2** — appears in 111 `plan` bullets (0.1% of all V3 plan bullets)

- grasp the black bowl first, then carry it toward the plate for placement.
- grasp the black bowl first, then carry it toward the plate for placement.
- grasp the black bowl first, then carry it toward the plate for placement.

**Cluster #3** — appears in 110 `plan` bullets (0.1% of all V3 plan bullets)

- grasp the black bowl and carry it toward the plate for placement.
- grasp the black bowl and carry it toward the plate for placement.
- grasp the black bowl and carry it toward the plate for placement.

**Cluster #4** — appears in 110 `plan` bullets (0.1% of all V3 plan bullets)

- pick-up-and-place phase active; grasp the black bowl and move it onto the plate.
- pick-and-place phase active; grasp the black bowl and move it onto the plate.
- pick-and-place phase active; grasp the black bowl and move it onto the plate.

**Cluster #5** — appears in 104 `plan` bullets (0.1% of all V3 plan bullets)

- grasp the black bowl first, then carry it to the plate for placement.
- grasp the black bowl first, then carry it to the plate for placement.
- grasp the black bowl first, then carry it to the plate for placement.

**Cluster #6** — appears in 98 `plan` bullets (0.1% of all V3 plan bullets)

- pick up the black bowl and carry it toward the plate for placement.
- pick up the black bowl and carry it toward the plate for placement.
- pick up the black bowl and carry it toward the plate for placement.

**Cluster #7** — appears in 98 `plan` bullets (0.1% of all V3 plan bullets)

- pick-and-place phase active; reach to the black bowl, lift it, and move it onto the plate.
- pick-and-place phase active; reach for the black bowl, lift it, and move it onto the plate.
- pick-and-place phase active; reach to the black bowl, lift it, then move it onto the plate.

**Cluster #8** — appears in 97 `plan` bullets (0.1% of all V3 plan bullets)

- reach toward the black bowl and lift it for placement onto the plate.
- reach toward the black bowl and lift it for placement onto the plate.
- reach toward the black bowl and lift it for placement onto the plate.

**Cluster #9** — appears in 89 `plan` bullets (0.1% of all V3 plan bullets)

- pick-and-place phase active; grasp the black bowl, then carry it to the plate.
- pick-and-place phase active; grasp the black bowl, then carry it to the plate.
- pick-and-place phase active; grasp the black bowl and carry it onto the plate.

**Cluster #10** — appears in 86 `plan` bullets (0.1% of all V3 plan bullets)

- reach toward the bbq sauce bottle, grasp it, and carry it to the basket.
- reach toward the bbq sauce bottle, grasp it, then carry it to the basket.
- reach toward the bbq sauce bottle, grasp it, then carry it into the basket.

## 8. Cross-check with Agent 2 (forbidden phrases & motor-imperative regression)

✅ No V3 LIBERO top-20 n-gram matched the classical anthropomorphic heuristic (`wants`, `decides`, `thinks`, `believes`, `feels`, …). The hardened prompt successfully scrubbed the V2-era cognitive-state phrasing.

Agent 2 also reported a *new* C-failure mode: the hardened prompt eliminated cognitive-state phrasing but introduced **low-level motor imperatives** in the `plan:` bullet. The boilerplate signals below from my n-gram analysis quantify the prevalence of that regression — these phrases should be **double-flagged** against Agent 2's appropriateness fail set:

| bullet | imperative phrase | DF count | % of bullets |
|---|---|---|---|
| plan | carry it | 32,624 | 33.1% |
| plan | and place | 27,967 | 28.4% |
| plan | grasp the | 26,699 | 27.1% |
| plan | phase active | 26,433 | 26.8% |
| plan | and carry it | 19,392 | 19.7% |
| plan | and place phase | 17,897 | 18.2% |
| plan | pick and place | 17,869 | 18.1% |
| plan | move it | 16,540 | 16.8% |
| plan | reach toward | 15,377 | 15.6% |
| plan | reach toward the | 15,266 | 15.5% |
| plan | carry it to | 15,077 | 15.3% |
| plan | carry it toward | 14,917 | 15.1% |
| plan | pick up | 14,429 | 14.6% |
| plan | for placement | 14,220 | 14.4% |
| plan | place phase active | 13,937 | 14.1% |

## 9. Verdict

**Overall: RED**

Decision criteria (GREEN ≅ V2 DROID diversity & no phrase >10%; YELLOW ≅ mild repetition / phrase 10-25%; RED ≅ severe collapse / single phrase >25%).

- Worst single phrase: `robot arm` in **49.8%** of V3 `scene` bullets (49,878 hits).
- V3 avg near-dup rate (across 5 canonical bullets): 23.9%
- V2 DROID avg near-dup rate: 4.3%
- Unique bigrams per 1k bullets — V3 220.5 vs V2 DROID 1013.9 ⇒ V3 is **much less diverse** than V2 DROID.
- 29 distinct phrase(s) exceed the 25% RED threshold.
- 96 distinct phrase(s) fall in the 10-25% YELLOW band.

## 10. Top 3 recommendations

1. **Rewrite the labeler system prompt** to break the template `robot arm` (currently in 49.8% of V3 `scene` bullets) and switch to a free-form sentence schema with example variations per bullet. In particular, `phase active` shows up in 26.8% of `plan` bullets — strip the literal `<task>-phase active;` scaffold from the prompt.
2. **Increase decoding diversity for the re-label**: bump `temperature` to 0.9–1.0 and add `top_p=0.95`, or rotate the labeler model across `gpt-5.4-mini` / `gpt-5.5-mini`. Current V3 near-dup rate (23.9%) is 5.5× the V2 DROID baseline (4.3%).
3. **Strip prompt-scaffold leakage from outputs**: the labeler is writing prompt internals into the caption (`action head` (17.3% of `plan`), `the action head` (16.7% of `plan`), `this patch` (13.0% of `plan`)). These are non-grounded artifacts that hurt AV training. Either post-filter such lines, or remove the `action_head` / `image_patch_token` cues from the labeler's user message.

### 10a. Detected prompt-scaffold leakage

Phrases that look like labeler prompt internals (the `action head` / image-patch token vocabulary) leaking into the caption body:

| bullet | phrase | DF count | % of bullets |
|---|---|---|---|
| plan | action head | 17,072 | 17.3% |
| plan | the action head | 16,454 | 16.7% |
| plan | this patch | 12,849 | 13.0% |
| plan | patch carries | 12,286 | 12.5% |
| plan | this patch carries | 11,692 | 11.9% |
| plan | patch carries the | 10,913 | 11.1% |
| plan | token carries | 10,416 | 10.6% |
| plan | token carries the | 9,286 | 9.4% |
| plan | this token carries | 8,312 | 8.4% |
| plan | basket this patch | 3,751 | 3.8% |
| plan | plate this patch | 3,111 | 3.2% |

## 11. Five-line summary

- V3 overall vocabulary (unigrams summed over 5 canonical bullets): **9,514** unique types.
- Top-1 most-common bullet phrase: `robot arm` in `scene` (49.8%).
- V3 near-duplicate rate (avg over canonical bullets): **23.9%**.
- V3 vs V2 DROID diversity: **much less diverse**.
- Final verdict: **RED**.

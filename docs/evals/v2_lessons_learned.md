# V2 SFT post-mortem: lessons learned & GRPO A/B cookbook

> **[V3 LIBERO results superseding this doc]** The V2 DROID failure modes
> described below were the motivation for the V3 LIBERO 4-suite rerun.
> Authoritative pass/fail signal for V3 lives in
> `data/sft/libero_4suite_v3/v3_scorecard.json`, produced by
> [`scripts/eval/build_v3_scorecard.py`](../../scripts/eval/build_v3_scorecard.py).
> The V3 eval pipeline (retrieval margin, 3-axis LLM judge, closed-loop
> sim A/B, live AV captioning) is documented in
> [`v3_libero_eval_refactor.plan.md`](../../.cursor/plans/v3_libero_eval_refactor_c499993c.plan.md)
> and runs automatically as Phase 6 of the SFT watcher
> ([`scripts/eval/run_post_sft_evals.sh`](../../scripts/eval/run_post_sft_evals.sh)).
> All raw DROID artifacts referenced below have been archived to
> `data/_archive_droid/` via
> [`scripts/migration/archive_droid.sh`](../../scripts/migration/archive_droid.sh)
> — read the manifest at `data/_archive_droid/MANIFEST.txt` for the
> origin → destination map.

> **Start here for “what do we run next?”** [`docs/sft_plan/06_v2_postmortem_v3_rerun.md`](../sft_plan/06_v2_postmortem_v3_rerun.md) (overnight checklist, judge B, **`--ar-av-mix-max`**, GRPO). **SFT-plan sibling:** [`04_v2_lessons_learned.md`](../sft_plan/04_v2_lessons_learned.md) (same run, plan-folder wording). **Repo overview:** [`README.md`](../../README.md).

This note summarizes what **`data/sft/droid_100ep_v2_nce`** taught us (quantitative vs qualitative failure modes), what we changed in code afterward, how we evaluate honestly, and a **simple ~1-hour A/B** to test whether **GRPO** moves the needle on **template / “bag of phrases” collapse**.

Artifacts referenced below live under:

- Run snapshot: `data/sft/droid_100ep_v2_nce/`
  - `config.json`, `metrics.jsonl`, `av/`, `ar/`
  - Qualitative exports (may have been produced post-training): `samples.jsonl`, `llm_judge.jsonl`

---

## 1. What V2 was trying to do

Joint warm-start **SFT** for:

- **AV** (Activation Verbalizer): activation-conditioned causal LM; CE on **gold** caption tokens.
- **AR** (Activation Reconstructor): caption \(\rightarrow\) \(\hat h\) regression in \(\alpha\)-scaled space; optional **InfoNCE** to discourage captions that reconstruct **everyone’s** activation.

Representative **snapshot hyperparameters** (see `config.json`):

| Knob | V2 value (approx.) |
|------|---------------------|
| `total_steps` | 15000 |
| `learning_rate` | 1e-4 |
| `warmup_steps` | 500 |
| `batch_size` | 4 |
| `ar_contrastive_weight` | 0.5 |
| `truncate_to_n_layers` (AR) | 16 |
| `alpha` | ~197.44 from extraction stats |
| `clip_target_scaled` | 5.0 |
| `balance_position_mix` | true |
| `min_bullet_lines` | 3 |
| `eval_closed_loop` | true (greedy + `T=0.7`), capped batches |

---

## 2. What looked “good” on scalar metrics

By late training, **teacher-forced** validation showed strong aggregate **FVE / cosine**, and **closed-loop** `h → AV.generate → AR → ĥ` tracked teacher-forced **closely** on aggregates — suggesting **little numerical exposure bias** in reconstruction scores at convergence.

Stratified metrics flagged **weaker `image_patch`** heads versus **`anchor` / `last_text`**: aggregate scores can hide **visual-slot underfitting**.

**Interpretation trap:** high reconstruction quality means \(\hat h\) aligns with \(h\) **under your decoding pipeline**. It does **not** imply that AV language matches **pixels**, instructions, or human-readable semantics.

---

## 3. What was catastrophically bad anyway (qualitative)

Held-out qualitative evaluation using **`scripts/eval/llm_judge_av_captions.py`** (GPT multimodal judge on the **same cached frames** used during labeling; axes **grounding** + **appropriateness**) showed:

- **`gold`** captions passed grounding (**specific**) on a majority of sampled rows — sanity ceiling \(\approx\) **73%** in one sweep (**not** 100%; references themselves drift vs frames/time).
- **`av_pred`** (**AV greedy generations**) failed grounding (**generic**) on **every** sampled row in that artifact (**0 / N**) despite plausible prose — systematic **wrong-scene** descriptions (“templates”: tabletop/green bowl, trash-can kitchen, couch/hoodie, socks-on-bed, etc.) reused across **different** tasks/visuals.

This matches **`scripts/eval/dump_av_samples.py`**: gold vs generated captions diverged semantically while **closed-loop FVE** could still look respectable — classic **“shorthand collapse”** / discrete verbal clusters acting like IDs that AR can invert.

---

## 4. Root causes (not mutually exclusive)

### 4.1 Dead InfoNCE during V2 (implementation pathology)

Training **`metrics.jsonl`** showed **`ar_nce`** glued at **`ln(batch_size)` \(\approx\) 1.386** for batch **B = 4** — diagnostic of **uniform softmax over negatives** (no discriminative contrastive gradient).

**Cause:** earlier similarity definition produced numerically tiny logits → softmax \(\approx\) uniform.

**Fix in repo:** cosine similarities scaled by **`nce_temperature`** (CLI **`--ar-nce-temperature`**) so InfoNCE is numerically alive.

### 4.2 Objective mismatch with semantic grounding

SFT never receives **pixels**. Faithfulness to scenes enters **only** via whatever GR00T’s \(h\) already encodes **plus** weak coupling:

- AV learns CE toward labels conditioned on \(h\).
- AR pulls captions toward invertibility under reconstruction (+ contrastive structure).

Nothing explicitly aligns caption tokens to **observable attributes**. Shortcut captions can occupy low-loss reconstruction pockets **especially when contrastive pressure is absent**.

### 4.3 Train/eval gap on caption generators — partially structural

**AR** trains largely on **gold** strings; at inference **AV.generate** feeds AR — **`scripts/training/run_sft.py`** now exposes **`--ar-av-mix-*`** so AR sometimes trains on **`AV.generate(h)`** (**scheduled sampling on AR inputs only**).  
Important limitation: **no gradient through discrete generation into AV** on that branch — fixing wording generally requires **RL / GRPO**, multimodal losses, or better labeled coupling — **not** `p_av` alone.

---

## 5. Code & tooling improvements landed post‑diagnosis (inventory)

| Area | Purpose |
|------|---------|
| `src/nla/models/ar.py` | Cosine InfoNCE + `nce_temperature`; optional `clip_target_scaled` |
| `scripts/training/run_sft.py` | `--stats-json`, `--ar-nce-temperature`, closed-loop flags, **`--ar-av-mix-*`**, balancing / bullets |
| `src/nla/training/sft.py` | Closed-loop eval, position sampler, scheduled AR mixing, **`p_av` / `ar_mix_used`** logging |
| `scripts/eval/dump_av_samples.py` | Gold vs greedy/sample AV & TF vs closed-loop scalars |
| `scripts/eval/llm_judge_av_captions.py` | Grounded judge overlay vs cached wrist/exterior frames (`GRADE_SYSTEM`) |

---

## 6. Honest evaluation playbook (minimal bar before trusting captions)

Run **both**:

1. **Scalars:** teacher-forced + closed-loop (`--eval-closed-loop`), stratified by **`position_type`**.
2. **Qualitative:** `dump_av_samples.py` + at least a **spot-check judge bundle**.

Judge negatives (**gold failures**) imply cleaning labels/frame alignment — otherwise AV blame is inflated.

---

## 7. GRPO: ~1-hour simple A/B to probe template collapse

**Scientific question:** Does short **on-policy** RL (**GRPO**) reward \(\mathrm{AR}(\text{sampled caption})\approx h\) shift AV away from a **small reusable phrase bag**, improving **grounded diversity**, faster than longer vanilla SFT?

**Mechanism:** `scripts/training/run_grpo.py` samples captions \(y\) from the **current** AV policy and pushes \(\log \pi(y\mid h)\) weighted by **group-relative advantages** vs KL anchor \(\beta\) to **frozen** reference AV — gradients hit **`generate`** paths unlike CE-only SFT.

**Outcome signals (“simple”)**:

| Signal | Baseline interpretation |
|--------|-------------------------|
| **Gap greedy vs sampled FVE** (logged via **`--eval-temperatures`**) | Memorized / low-entropy verbalizers often show huge gaps after shortcut collapse. GRPO may modestly rebalance if sampling improves diversity **without** wrecking greedy reconstruction. |
| **Distinct phrases / templates** on fixed held-out draws (`dump_av_samples.py`) | Few dominant templates \(\Rightarrow\) shortcut regime; growth \(\Rightarrow\) exploratory improvement (not sufficient for truthfulness). |
| **Micro judge slice** (`llm_judge_av_captions.py`, small `--per-position`) | Slow/costly — optional cap **10–15** judge pairs end-of-hour. |

### 7.1 Arms

| Arm | What it is |
|-----|------------|
| **A — Control** | Frozen **`data/sft/droid_100ep_v2_nce`** (no GRPO training time). Run eval harness only (`dump_av_samples.py`, optional judge). |
| **B — Treatment** | Same **`--sft-dir`**, **GRPO** training **~45–55 minutes wall-clock** (leave slack for eval IO). |

Keep **`--seed`** identical where supported so qualitative draws match **except** for checkpoint changes.

### 7.2 Example arms (after calibrating `--total-steps` once)

Run **Arm A** first (cheap):

```bash
cd /path/to/nla-groot && source .venv/bin/activate

export PYTHONPATH=src

# Arm A — qualitative baseline from V2 SFT (no training)
python scripts/eval/dump_av_samples.py \
  --ckpt-dir         data/sft/droid_100ep_v2_nce \
  --activations-root data/activations/droid_100ep \
  --labels-jsonl     data/labels/droid_100ep/labels.jsonl \
  --per-position     6 \
  --seed             0 \
  --out-jsonl        data/grpo_ab/arm_a_v2_baseline/samples.jsonl

# Optional micro judge (needs OPENAI_API_KEY)
# python scripts/eval/llm_judge_av_captions.py ...
```

**Arm B — conservative GRPO smoke (~lr sens.)**

Pick **`total_steps`** empirically so **`elapsed_s`** at last logged train row \(\approx\) target budget on **your** GPU (GRPO cost \(\sim O(B \cdot K)\) generations per step).

Starter recipe — **tune `--total-steps` down/up after one timing probe**:

```bash
python scripts/training/run_grpo.py \
  --sft-dir             data/sft/droid_100ep_v2_nce \
  --activations-root    data/activations/droid_100ep \
  --output-dir          data/grpo_ab/arm_b_grpo_1h_smoke \
  --batch-size          4 \
  --rollouts-per-activation 4 \
  --rollout-temperature     1.0 \
  --rollout-top-p           0.95 \
  --rollout-max-new-tokens  160 \
  --beta                    0.02 \
  --learning-rate           3e-6 \
  --warmup-steps            20 \
  --total-steps             120 \
  --eval-every              15 \
  --save-every              60 \
  --grad-clip               1.0 \
  --eval-temperatures       0.0,0.7,1.0 \
  --seed                    0
```

**Variant B2 — AR tracks evolving AV wording** (if Arm B scalars move but language stays stale):

```bash
# Same as B but add e.g. small → medium co-training weight after smoke stability:
# --ar-co-train-weight 0.1
```

(Watch **`train/` diagnostics** in **`metrics.jsonl`** for instability.)

Then mirror Arm A dump against **`data/grpo_ab/arm_b_grpo_1h_smoke`**:

```bash
python scripts/eval/dump_av_samples.py \
  --ckpt-dir         data/grpo_ab/arm_b_grpo_1h_smoke \
  --activations-root data/activations/droid_100ep \
  --labels-jsonl     data/labels/droid_100ep/labels.jsonl \
  --per-position     6 \
  --seed             0 \
  --out-jsonl        data/grpo_ab/arm_b_grpo_1h_smoke/samples.jsonl
```

### 7.3 Reading results quickly

1. **`metrics.jsonl`:** compare **`val/fve_*`** across **`eval_temperatures`** — shrinking greedy/sample gap **may** indicate healthier entropy / less memorized verbal shortcuts (not guaranteed truthfulness).
2. **`samples.jsonl`:** scan whether **same canned openings** dominate Arm B vs Arm A.
3. Optional **10-row judge** → fraction **`specific`** on `av_pred`.

### 7.4 Failure modes to expect in a 1-hour slice

- **No visible language gain** — RL horizon too short or \(\beta\) too high vs LR too low.
- **Reward hacking without grounding** — reconstruction improves but judge generic rate unchanged (needs multimodal loss / richer reward).
- **Instability** — lower **`--learning-rate`**, raise **`--beta`**, reduce **`rollout-temperature`**, or temporarily disable **`--ar-co-train-weight`**.

---

## 7.5 Intervention leverage sweep (which slots matter?)

Aggregate reconstruction quality does not tell us **which token slots** \((L, p)\) actually move the action head. **`scripts/eval/nla_steer_leverage_sweep.py`** repeats the existing single-shot causal probe (`scripts/eval/nla_steer_groot_action.py`) over a **grid** of `SteerSpec` placements (`last_text`, `anchor`, `image_patch` × seeds, `fixed` × token range), with **matched-norm Gaussian null controls** so rankings are not confounded by replacement magnitude alone. It is **open-loop** — one `(traj, step)` observation, no sim rollout — and writes a ranked JSONL/CSV of `|Δaction|` per condition.

```bash
PYTHONPATH=src python scripts/eval/nla_steer_leverage_sweep.py \
  --model-path     nvidia/GR00T-N1.7-3B \
  --dataset-path   /path/to/lerobot_dataset \
  --embodiment-tag OXE_DROID_EEP \
  --ar-dir         data/sft/droid_100ep_v2_nce/ar \
  --traj-id 0 --step 0 \
  --text-file      steer_bullets.txt \
  --placements     last_text,anchor,image_patch \
  --image-patch-seeds 0,1,2,3,4 \
  --fixed-token-range 0::16 \
  --null-samples   4 \
  --sort-by        delta_vs_null \
  --out-jsonl      data/sft/droid_100ep_v2_nce/intervention_leverage.jsonl \
  --out-csv        data/sft/droid_100ep_v2_nce/intervention_leverage.csv
```

Each row carries `effect.global_max_abs` (real steer) and `null_global_max_abs_median` plus `null_global_max_abs_p95`; `delta_vs_null_median` is the recommended ranking key when judging **which slots are causally important** rather than which slots merely accept large vectors.

---

## 8. Bottom-line checklist before claiming interpretability

- [ ] InfoNCE **not** stuck at **`ln(B)`** during training.
- [ ] Closed-loop vs teacher gap monitored **per `position_type`**.
- [ ] Qualitative **dump + grounded judge** pass on held-out episodes.
- [ ] If captions still lie but reconstruct: acknowledge **steerability \(\neq\)** **faithful interpretability** — push **GRPO** / **vision-aligned objectives** / **reference QA**.

### Local gating helper (CLI-only)

For fast, repeatable pass/fail checks over SFT logs, use:

```bash
python scripts/ci/check_sft_metrics.py \
  data/sft/my_run/metrics.jsonl \
  --batch-size 4 \
  --config data/sft/my_run/config.json \
  --require-closed-loop \
  --max-tf-closed-fve-gap 0.05
```

What it checks (configurable):

- **NCE alive:** fails when mean train `ar_nce` in the recent tail sits near
  `ln(B)` while `ar_contrastive_weight > 0` (dead contrastive signal).
- **Closed-loop presence:** fails if no val/final row has `closed_*/fve`.
- **Teacher-vs-closed gap:** optional threshold on `fve - closed_greedy/fve`.

Exit code is **0** on pass, **1** on failed checks, **2** on bad inputs.

---

## Document history

- **2026-05-15:** Draft from V2 metrics/judge analysis & repo audit fixes.

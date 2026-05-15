# V2 postmortem: lessons learned (SFT on `droid_100ep`)

> **Canonical rerun checklist:** [`06_v2_postmortem_v3_rerun.md`](06_v2_postmortem_v3_rerun.md). **Expanded metrics + GRPO cookbook (evals folder):** [`docs/evals/v2_lessons_learned.md`](../evals/v2_lessons_learned.md). **Repo overview:** [`README.md`](../../README.md).

This note captures what we learned from the **`droid_100ep_v2_nce`** joint AV+AR SFT run (~15k steps), why aggregate metrics misled us, what we changed in code afterward, and how to run a **short GRPO A/B** aimed at template / “bag-of-scenes” collapse.

---

## 1. What V2 was trying to do

- **AV (Activation Verbalizer):** causal LM with activation injected at a slot; trained with **teacher-forced CE** on gold captions from `labels.jsonl`.
- **AR (Activation Reconstructor):** text → activation map; trained with **MSE** in α-scaled space toward the **same-row** activation, plus **InfoNCE** (`ar_contrastive_weight`) to discourage captions that reconstruct many activations equally well.
- **Extras enabled on V2:** `--balance-position-mix`, `--min-bullets 3`, `--eval-closed-loop` (greedy + `t=0.7`), `--closed-loop-max-batches 64`, `--max-val-items 1000`, `--stats-json` / aligned α, AR depth **16**, **`clip_target_scaled: 5`**.

Checkpoint layout: `data/sft/droid_100ep_v2_nce/` with `av/`, `ar/`, `config.json`, `metrics.jsonl`.

---

## 2. What looked “good” numerically

From **`metrics.jsonl`** (final + progression):

- **Teacher-forced FVE / cosine** improved to strong-looking values by late training (e.g. aggregate **FVE ~0.72**, **cosine ~0.86** region — exact numbers live in the run’s final `val` / `final` rows).
- **Closed-loop** metrics (`h → AV.generate → AR → ĥ`) **tracked teacher-forced closely by the end** — greedy vs `t=0.7` were similar to TF within a small gap on aggregate.
- **Position stratification:** **`image_patch` FVE stayed materially weaker** than **`anchor`** / **`last_text`** even when the aggregate looked healthy — a recurring sign that **vision-heavy positions** are harder and can hide under macro averages.

**Lesson:** Strong **reconstruction-style** metrics do **not** imply **faithful natural-language descriptions** of the scene or task.

---

## 3. What failed qualitatively (“shorthand / template collapse”)

### 3.1 Symptoms

- **`dump_av_samples.py`** (`samples.jsonl`): AV-generated captions were often **diverse-looking** but **wrong**: repeated **fictitious scenes** (e.g. green bowl + star toy, kitchen trash can + paper towel, couch + hoodie, socks on mattress, …) applied across **different** instructions and gold captions.
- **`llm_judge_av_captions.py`** (`llm_judge.jsonl`): multimodal judge on **cached wrist + exterior frames**, grading **grounding** (axis B) and **appropriateness** (axis C).
  - **`av_pred`:** **0 / 30** rows labeled **specific** (100% **generic** grounding failure on that slice).
  - **`gold`:** ~**22 / 30** specific (~73%) — the reference labels are **not** a perfect ceiling (frame sync, stale layout, patch mismatch).

### 3.2 Interpretation

The AV learned captions that function partly as **discrete cluster IDs** AR can invert (“bag of templates”), **without** tracking pixel-grounded content the judge can verify.

---

## 4. Root causes (not mutually exclusive)

### 4.1 Dead contrastive signal during V2

- Training logs showed **`ar_nce` locked at ln(batch_size) ≈ ln(4)** for the entire run — classic signature of a **uniform softmax over negatives** (no learning gradient through contrastive term).
- **Cause (since fixed in code):** InfoNCE similarities based on **tiny MSE-scale logits** → numerically flat distribution.
- **Fix in repo:** **cosine similarity + temperature** in `ActivationReconstructor.forward_sft`, CLI **`--ar-nce-temperature`**.

### 4.2 Objective mismatch: reconstruction ≠ semantic fidelity

- SFT **never sees pixels** in this pipeline; faithfulness to the **world** only enters **via whatever GR00T’s activation already encodes** and via **gold text**.
- If AR+MSE can explain variance with **shortcuts**, CE + reconstruction pressure need **not** produce visually grounded prose.

### 4.3 Train/eval text distribution gap for AR

- Historically AR trains mostly on **gold** prose while inference uses **AV-generated** text — **`--ar-av-mix-*`** (scheduled sampling for AR inputs only) targets AR robustness; it **does not** gradient-fix AV wording by itself.

### 4.4 Reference noise

- ~27% gold grounding failures on the judged slice implies **some** “AV looks wrong” comparisons are **label/frame drift**, not only model failure — qualitative eval must report **gold baseline**.

---

## 5. What we implemented afterward (code map)

| Topic | Where |
|--------|--------|
| α from `stats.json` | `scripts/training/run_sft.py` `--stats-json` |
| AR truncation default **16** | `ARConfig`, `--ar-layers` |
| Cosine InfoNCE + temperature | `src/nla/models/ar.py`, `--ar-nce-temperature` |
| Optional AR target clip | `--ar-clip-target-scaled` |
| Position rebalance | `--balance-position-mix`, `sft.py` WeightedRandomSampler |
| Min bullets filter | `--min-bullets`, `dataset.py` |
| Closed-loop val | `--eval-closed-loop`, `--closed-loop-temps`, `--closed-loop-max-batches` |
| AR trains on AV text sometimes | `--ar-av-mix-*`, `sft.py` (AV CE still gold-only) |
| Train logs `p_av`, `ar_mix_used` | `metrics.jsonl` train rows |
| Qualitative dump | `scripts/eval/dump_av_samples.py` |
| Frame-grounded judge | `scripts/eval/llm_judge_av_captions.py` |
| GRPO + AR co-training option | `scripts/training/run_grpo.py`, `GRPOConfig.ar_co_train_weight` |

---

## 6. Evaluation playbook (minimum viable)

1. **Scalars:** teacher-forced **and** closed-loop FVE/MSE/cosine, **stratified by `position_type`**.
2. **Contrastive health:** `ar_nce` **should move off ln(B)** when InfoNCE is learning; log **NCE accuracy / margins** if debugging.
3. **Qualitative:** `dump_av_samples.py` on held-out val — inspect **template reuse** across **different** episodes/instructions.
4. **Grounding:** `llm_judge_av_captions.py` — compare **`gold` vs `av_pred`** on **axis B** with fixed `per-position` * N.

---

## 7. Open directions (research)

- **GRPO** (policy gradient on **sampled** captions, reconstruction reward): optimizes **AV** under **on-policy** text — primary knob for moving **wording** without adding vision.
- **Vision-aligned auxiliary losses** if captions must match frames **by construction**.
- **Per-row AR mixing**, larger batches for NCE, **hard negatives / queues**.
- **Label QA:** fix temporal/frame mismatch so gold is a trustworthy ceiling.

---

## 8. One-hour GRPO A/B protocol (“bag of words / templates”)

**Goal:** In ~**60 minutes** wall-clock, get a **paired** signal whether **short GRPO** improves **diversity + grounding proxies** vs the **frozen V2 SFT** baseline — **not** full convergence.

**Warm-start:** use the V2 tree you want to improve (e.g. `data/sft/droid_100ep_v2_nce` — must contain `av/` and `ar/`).

### 8.1 Arm A — baseline (no GRPO time budget)

- **Checkpoint:** `data/sft/droid_100ep_v2_nce` (or your current best SFT).
- **Eval only:** same harness as §6 — especially **`dump_av_samples.py`** + optional **`llm_judge_av_captions.py`** on a **fixed seed** and **`--per-position`** count.

This consumes near-zero training time; almost the full hour goes to **Arm B** + **shared post-eval**.

### 8.2 Arm B — GRPO run (~most of the hour)

**Why GRPO targets templates:** reward uses **current** sampled caption \(y\) and **`AR(y)` vs \(h\)**. Within an activation group, rollouts that **share generic boilerplate** get **similar rewards** and **lower contrast** under group normalization — pressure exists to find **text that tracks \(h\)** better across samples; KL to **ref AV** stops collapse-to-nonsense when \(\beta\) is sane.

**Suggested starter hyperparameters** (tune after a 5-step timing probe):

```bash
# 0) Calibrate: note seconds/step from logs, then set --total-steps ≈ 3600 / step_time

PYTHONPATH=src python scripts/training/run_grpo.py \
  --sft-dir          data/sft/droid_100ep_v2_nce \
  --activations-root data/activations/droid_100ep \
  --output-dir       data/grpo/ab_v2_vs_grpo_B \
  --batch-size       4 \
  --rollouts-per-activation 4 \
  --rollout-temperature     1.0 \
  --rollout-top-p           0.95 \
  --rollout-max-new-tokens  160 \
  --beta             0.02 \
  --learning-rate    3e-6 \
  --warmup-steps     20 \
  --total-steps      REPLACE_ME \
  --eval-every       10 \
  --save-every       REPLACE_ME \
  --eval-max-examples 64 \
  --eval-temperatures 0.0,0.7
```

**Optional “close the loop harder” variant (Arm B′):** add **`--ar-co-train-weight 0.1`** (or `0.2`) so AR **tracks policy wording** while AV trains — more drift risk; watch val FVE.

**If GPU memory allows:** `batch_size 6` or `rollouts-per-activation 6` increases GRPO group diversity per step (often helps template pathology), at higher cost per step — reduce **`total_steps`** to fit the hour.

### 8.3 Shared post-processing (both arms, identical settings)

```bash
SEED=0
PER=8

for CKPT in data/sft/droid_100ep_v2_nce data/grpo/ab_v2_vs_grpo_B; do
  PYTHONPATH=src python scripts/eval/dump_av_samples.py \
    --ckpt-dir "$CKPT" \
    --activations-root data/activations/droid_100ep \
    --labels-jsonl     data/labels/droid_100ep/labels.jsonl \
    --per-position "$PER" \
    --seed "$SEED" \
    --out-jsonl "${CKPT}/ab_samples_seed${SEED}.jsonl"
done
```

Optional (costs API): run **`llm_judge_av_captions.py`** on each checkpoint into **`${CKPT}/ab_judge.jsonl`** with the **same** `--seed` / `--per-position`.

### 8.4 What to compare (simple, template-focused)

| Metric | Arm A (V2 SFT) | Arm B (GRPO) | Interpretation |
|--------|----------------|--------------|----------------|
| **Judge axis B pass rate** (`specific`) | baseline | ↑ desired | Grounded-to-frame proxy |
| **Distinct templates** in `generated` | baseline | ↑ usually good | Eyeball + count recurring opening clauses |
| **Within-row rollout reward variance** (from GRPO train diagnostics if logged) | — | ↑ modest | Same \(h\), different \(y\) — if variance → 0, policy stuck |
| **Greedy vs sampled val FVE gap** (`eval_temperatures`) | baseline | smaller gap sometimes | Memorization / entropy symptom (from `grpo.py` doc intent) |

**Success bar for a 1h smoke:** any **consistent** improvement on **judge grounding** or clear **reduction in cross-example template reuse** without **KL explosion** or **val FVE collapse**. Negatives are still informative (publishable as “what didn’t move in 1h”).

### 8.5 Pitfalls

- **Unfair step counts:** GRPO is **heavier per step** than SFT (K rollouts). Calibrate wall-clock before committing **`total_steps`**.
- **Comparing to cosine-fixed SFT:** V2 was trained with **broken NCE**; comparing GRPO-on-V2 vs **V3 SFT** (cosine NCE + mix) is a **different** experiment — label axes clearly in notes.
- **AR co-training:** can help distribution match but may **mask** AV problems if AR absorbs junk text — keep an Arm B **without** `ar_co_train_weight` first.

---

## 9. References in-repo

- SFT loop: `src/nla/training/sft.py`
- GRPO: `src/nla/training/grpo.py`, `scripts/training/run_grpo.py`
- V2 snapshot: `data/sft/droid_100ep_v2_nce/config.json`, `metrics.jsonl`, `llm_judge.jsonl`, `samples.jsonl`

---

*Last updated: aligned with repo state after V2 analysis and SFT/GRPO tooling described above.*

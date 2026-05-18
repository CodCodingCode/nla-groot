# Interpretability Evaluation Protocol

> **Two tracks:** (1) **SFT / scene grounding** — caption vs **camera** (`llm_judge_av_captions.py`, axes B/C); (2) **Interp panel** below — caption behavior under **counterfactual `h` edits**. Do not conflate them; see **`docs/evals/v2_lessons_learned.md`**, **`07_sft_recipe_dataset_agnostic.md`**.

**Project:** `nla-groot` — AV/AR on GR00T activations (`PYTHONPATH=src`). High **FVE** does not substitute for track (1); that is the main **V2** lesson.

This document is the runbook for the structured interpretability evaluation
pipeline. It describes **what** is being measured, **how** to run the
end-to-end pipeline, and **how** to read the resulting numbers.

The pipeline lives entirely under `scripts/eval/` and consists of five
scripts plus this protocol and `judge_prompt.md`.

---

## What we are evaluating

We score **interpretability claims** — not raw FVE, not just judge vibes.
Each evaluation case is a single (example, token) position where we have:

- An original activation `h` from the GR00T backbone.
- A pre-registered hypothesis describing what the AV should say about `h`
  and how its description should change under a specific counterfactual edit.
- A control: a random matched-magnitude perturbation that should *not*
  produce the same change.

For each case we collect three AV explanations (baseline / edited / control)
and compute deterministic metrics + a constrained LLM-judge rubric. The
final composite score is a weighted blend of the two.

### Claim taxonomy (paper-shaped)

| Claim                       | Captured by                                  |
| --------------------------- | -------------------------------------------- |
| **Faithfulness (causal)**   | `direction_match`, `normalized_effect_size` |
| **Specificity**             | rubric `specificity_0_3`                     |
| **Robustness / Stability**  | `seed_stability` over paraphrase samples     |
| **Confabulation control**   | rubric `confabulation_0_3` + verbatim quote validation |
| **Counterfactual response** | rubric `consistency_0_3`, `direction_match` |

---

## File-flow diagram

```mermaid
flowchart TD
    A["build_eval_cases.py\n(eval_cases.jsonl)"] --> B["run_interp_panel.py\n(panel_rows.jsonl)"]
    B --> C["score_panel.py\n(auto metrics)"]
    B --> D["run_llm_judge.py\n(judge_rows.jsonl)"]
    D --> E["score_panel.py\n(merge + composite)"]
    C --> E
    E --> F["scores.json\nscores_by_case.jsonl"]
```

What scores what:

- **Deterministic auto metrics** (machine, no LLM):
  `direction_match`, `normalized_effect_size`, `seed_stability`. These are
  the strict scientific signal — every paper figure with error bars uses
  these.
- **LLM-as-judge** (structured, anchored):
  scores rubric dimensions (`specificity`, `consistency`, `confabulation`,
  `overall_faithfulness`) against the **anchored bins** in `rubric.py` and
  must quote evidence verbatim.
- **Composite**: `composite = w_auto * auto_score + w_judge * judge_score`
  (defaults `0.7 / 0.3` — auto-metric-dominant by design).

---

## Pre-registration

Before any AV/AR change can be claimed in the paper, you must:

1. Freeze the eval set once: `python scripts/eval/build_eval_cases.py ...`
   produces `eval_cases.jsonl` and you commit it (or hash it).
2. Hand-edit the per-case `hypothesis` and `expected_direction` fields, then
   freeze again.
3. Only then run the panel + judge + score. Re-running with the same seeds
   reproduces the same case set and the same auto metrics bit-for-bit.

If you change the eval set you start over with a new file name; never edit
in place.

---

## Reproducibility guarantees

| Source of variance        | Pinned by                                   |
| ------------------------- | ------------------------------------------- |
| Case sampling             | `--seed` on `build_eval_cases.py`           |
| Token-position selection  | Same `TokenPositionSampler(seed=...)` used  |
| Counterfactual edit noise | `--seed` on `run_interp_panel.py` (torch RNG) |
| Swap partner pairing      | `--swap-seed` on `run_interp_panel.py`      |
| AV decoding               | `--greedy` recommended for the eval row     |
| LLM judge                 | `temperature=0`, `seed=<arg>`, JSON schema  |
| Auto metric math          | Pure Python on materialized vectors         |

The judge model itself is *not* fully deterministic across deployments; that
is why the judge weight defaults to 0.3 and the auto-metric weight is 0.7.

---

## End-to-end runbook

The four scripts share an output directory. Pick one and stick to it:

```bash
EVAL_DIR=runs/eval/groot_av_v1
mkdir -p "$EVAL_DIR"
```

### 1. Freeze the eval set

```bash
python scripts/eval/build_eval_cases.py \
  --activations-root data/activations/libero_goal_pilot \
  --out "$EVAL_DIR/eval_cases.jsonl" \
  --n-cases 16 \
  --seed 0
```

Then **manually** edit `hypothesis` / `expected_direction` per row (or write
a one-shot script that fills them from your task DSL). Commit / archive the
result.

### 2. Run the intervention panel

```bash
python scripts/eval/run_interp_panel.py \
  --cases "$EVAL_DIR/eval_cases.jsonl" \
  --activations-root data/activations/libero_goal_pilot \
  --av-dir runs/sft/groot_av_v1/av \
  --ar-dir runs/sft/groot_av_v1/ar \
  --out "$EVAL_DIR/panel_rows.jsonl" \
  --max-new-tokens 80 \
  --greedy \
  --n-stability-samples 2 \
  --seed 0
```

This loads the AV (and optionally AR) and writes one row per case with
baseline + edited + control explanations and the auto-metric inputs.

### 3. Run the LLM judge

```bash
export OPENAI_API_KEY=...
python scripts/eval/run_llm_judge.py \
  --cases "$EVAL_DIR/eval_cases.jsonl" \
  --panel "$EVAL_DIR/panel_rows.jsonl" \
  --out "$EVAL_DIR/judge_rows.jsonl" \
  --model gpt-4o-2024-08-06 \
  --seed 0
```

For dual-judge agreement add `--dual-judge-model <other-model>`. Use
`--resume` if a run is interrupted.

### 4. Aggregate

```bash
python scripts/eval/score_panel.py \
  --cases "$EVAL_DIR/eval_cases.jsonl" \
  --panel "$EVAL_DIR/panel_rows.jsonl" \
  --judge "$EVAL_DIR/judge_rows.jsonl" \
  --out-by-case "$EVAL_DIR/scores_by_case.jsonl" \
  --out-summary "$EVAL_DIR/scores.json" \
  --w-auto 0.7 --w-judge 0.3
```

`scores.json` is the file you cite in the paper. `scores_by_case.jsonl` is
the per-case audit trail.

---

## Reporting format (for the paper / appendix)

For each evaluation table report:

- **Mean and std** of `composite`, `auto_score_01`, `judge_score_01` across
  all cases.
- **Per-stratum means** by `position_type` (`last_text` / `image_patch` /
  `anchor`).
- **Per-edit-kind means** (`noise` / `swap` / `null` / `paraphrase`).
- `judge_agreement` (mean inter-judge agreement, when dual-judge).
- `confabulation_score` (fraction of judge quotes that survived verbatim
  validation).
- Number of failed cases (`n_failed`) if any.

A single-row "headline" should always include the auto-metric mean
**separately** from the judge mean — never collapse them so reviewers can
audit which signal is doing the work.

---

## Failure modes to watch for

- **Direction match near 0 across all cases**: the AR isn't sensitive to the
  text difference between baseline and edited. Either AR is undertrained or
  the edit is too small.
- **High judge score but low auto score**: the judge likes the prose but the
  AR can't reconstruct the predicted shift. Suspect confabulation.
- **High `seed_stability` (~1.0)**: AV is deterministic regardless of input —
  re-check that injection is actually happening (this is the failure mode
  the paper warned about).
- **Many dropped quotes (`_warnings`)**: judge invented evidence. Treat the
  judge's score for that case as low-trust.

---

## Pass/fail bands (suggested)

These are starting points, not absolutes; tune as you accumulate data:

| Composite | Meaning                                                   |
| --------- | --------------------------------------------------------- |
| ≥ 0.70    | Strong: edit is detectable and the explanation is faithful |
| 0.55-0.70 | Acceptable: most signals positive, paper-grade with caveats |
| 0.40-0.55 | Weak: report as honest negative result                    |
| < 0.40    | Failed; do not claim interpretability for this regime     |

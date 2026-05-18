# First real SFT run — consolidated plan

This document merges `docs/sft_plan/01–05.md` into one actionable checklist. The parallel subagents did not write files; this folder was populated manually from codebase + NFS data (May 2026).

> **Read first (post–droid_100ep V2):** **`07_sft_recipe_dataset_agnostic.md`** (operational recipe) and **`docs/evals/v2_lessons_learned.md`** (V2 DROID quantitative failure modes + GRPO cookbook). **Next roadmap:** **`SFT_V5_NEXT.md`**. The numbered docs below remain valid for data/hparams/architecture.

---

## Preconditions (tick before GPU time)

1. **Data locked:** `data/activations/droid_100ep/` + `data/labels/droid_100ep/labels.jsonl` (99 968 labels, **0 orphans** vs index).
2. **α:** **197.4447** from `data/activations/droid_100ep/stats.json` (`p75_norm`).
3. **Sanity skim:** read ~20 label rows; confirm bullets look sane (optional: spot hallucinations).
4. **Disk / HF:** model weights cache reachable; `PYTHONPATH=src` works.

---

## Hyperparameters (non-negotiables — v1)

These match the recipe table in `02_hyperparams.md` §4 and are now also the
code defaults after the audit fixes (`--alpha 197.44`, `--ar-layers 16`).

| Knob | Value |
|------|-------|
| `--alpha` | **197.44** (or pass `--stats-json data/activations/droid_100ep/stats.json` to read α=197.4447 from `p75_norm`) |
| `--ar-layers` | **16** (now the default) |
| `--learning-rate` | **1e-4** (LoRA-rank-32 anchor from `droid_100ep_dirty`; v2 ablation: 3e-5 / 5e-5 vs paper's 1e-5) |
| `--batch-size` | **4** (raise if VRAM allows) |
| `--grad-accum-steps` | **1** (effective batch 4; bump to 2 only if CE noise warrants) |
| `--total-steps` | **3000** (prior run was still improving at 2000) |
| `--warmup-steps` | **200** |
| `--split-by` | **episode** |
| `--held-out-fraction` | **0.05** |
| `--eval-every` | **250** (with `--max-val-items 1000`) |
| `--save-every` | **500** |
| `ar-contrastive-weight` | **0** first run; enable if AR cheats generic recon |

Optional v1 audit flags (see §"Audit-fix flags" below):

- `--stats-json PATH` — load α from a Phase-1 extraction `stats.json` (overrides `--alpha`).
- `--balance-position-mix` — rebalance training draws toward `layer_spec.POSITION_MIX` (40/40/20).
- `--min-bullets N` — drop labels with fewer than `N` markdown bullet lines.
- `--eval-closed-loop` (+ `--closed-loop-temps`, `--closed-loop-max-batches`) — add `h → AV.generate → AR → ĥ` metrics alongside the teacher-forced eval.
- `--ar-clip-target-scaled V` — clamp the α-scaled AR target to ±V during `forward_sft` (e.g. 5.0).

Full paste-ready command: see **`02_hyperparams.md` §5**.

---

## Training procedure

1. Run **`scripts/training/run_sft.py`** with args above → output e.g. `data/sft/droid_100ep_v2/`.
2. Watch **`metrics.jsonl`** + TensorBoard in `output_dir/log/`.
3. Every eval: insist on **stratified FVE** by `position_type` (implement extra logging if missing — see **`03_eval_harness.md`**).
4. Save **best checkpoint** by **val FVE** (manual or script).

---

## Evaluation after SFT

**Immediate:** val **FVE / MSE / cosine** (stratified).

**Quick qualitative:** greedy-generate on a **fixed short list** of positions; paste snippets into a note (no full overlay required).

**Week 1 probes:** instruction-binding MCQ + optional gripper agreement (see **`03_eval_harness.md`**).

**Defer:** full paper eval suite, SAE consistency, dense overlay videos per checkpoint.

---

## Known risks / accepted deviations

| Risk | Mitigation |
|------|------------|
| **FVE looks good, captions don’t match cameras (V2)** | Run **`llm_judge_av_captions.py`**; read **`docs/evals/v2_lessons_learned.md`** and **`07_sft_recipe_dataset_agnostic.md`**. Don’t ship on reconstruction alone. |
| Labels ~75% **image_patch** | Monitor per-type CE/FVE; pass `--balance-position-mix` to draw closer to `POSITION_MIX` (40/40/20) |
| AV is **Qwen3-4B**, not Cosmos | Accepted for v1; largest architectural gap vs paper |
| AR `--ar-layers` was 10 | Default is now **16** (matches `SELECT_LAYER`) |
| Teacher-forced eval hides AV→AR pipeline regressions | Add `--eval-closed-loop` to log `h → AV.generate → AR → ĥ` stratified FVE/cosine |
| α drifts from `stats.json` | Pass `--stats-json` so AV/AR α come from the same Phase-1 dump |

---

## After SFT succeeds

1. **`run_grpo.py`** from `sft_dir` with `activations-root` **without** labels.
2. Consider **`--ar-co-train-weight > 0`** so AR tracks AV drift (paper-style).
3. α **unchanged**.

---

## Source docs

| File | Topic |
|------|-------|
| `01_data_audit.md` | Label counts, joins, position skew |
| `02_hyperparams.md` | LR, batch, paste-ready CLI |
| `03_eval_harness.md` | Metrics + robotics probes |
| `04_layer_alpha.md` | Hook layer, α math, GRPO reward |
| `05_arch_injection.md` | Qwen vs Cosmos, AR depth, templates |
| `07_sft_recipe_dataset_agnostic.md` | **Dataset-agnostic SFT recipe** (hard negs, mix, eval) |
| `docs/evals/v2_lessons_learned.md` | **V2 DROID lessons + GRPO A/B cookbook** |
| `SFT_V5_NEXT.md` | **V5 roadmap** (optimization, layers, quality weights) |

See also repo-wide **`docs/NLA_AGENT_KNOWLEDGE.md`** and root **`README.md`** (layout + quick start).

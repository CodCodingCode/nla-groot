# First real SFT run — consolidated plan

This document merges `docs/sft_plan/01–05.md` into one actionable checklist. The parallel subagents did not write files; this folder was populated manually from codebase + NFS data (May 2026).

---

## Preconditions (tick before GPU time)

1. **Data locked:** `data/activations/droid_100ep/` + `data/labels/droid_100ep/labels.jsonl` (99 968 labels, **0 orphans** vs index).
2. **α:** **197.4447** from `data/activations/droid_100ep/stats.json` (`p75_norm`).
3. **Sanity skim:** read ~20 label rows; confirm bullets look sane (optional: spot hallucinations).
4. **Disk / HF:** model weights cache reachable; `PYTHONPATH=src` works.

---

## Hyperparameters (non-negotiables)

| Knob | Value |
|------|-------|
| `--alpha` | **197.4447** |
| `--ar-layers` | **16** (was default 10 — update for backbone-depth parity) |
| `--learning-rate` | **1e-5** (paper-ish; not 1e-4 default) |
| `--batch-size` | **4** (raise if VRAM allows) |
| `--grad-accum-steps` | **16** (effective batch ≈ 64) |
| `--total-steps` | **4000** (adjust after first plateau) |
| `--split-by` | **episode** |
| `--held-out-fraction` | **0.05** |
| `ar-contrastive-weight` | **0** first run; enable if AR cheats generic recon |

Full command: see **`02_hyperparams.md`**.

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
| Labels ~75% **image_patch** | Monitor per-type CE/FVE; optional rebalance or top-up labeling |
| AV is **Qwen3-4B**, not Cosmos | Accepted for v1; largest architectural gap vs paper |
| AR `--ar-layers` was 10 | **Fix to 16** this run |
| Label skew vs `POSITION_MIX` | SFT sees file distribution; GRPO later uses sampler |

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

See also repo-wide **`docs/NLA_AGENT_KNOWLEDGE.md`**.

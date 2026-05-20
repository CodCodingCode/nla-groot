# nla-groot

Implementation of **Natural Language Autoencoder (NLA)**–style tooling for **robotics VLA activations** (GR00T backbone): an **activation verbalizer (AV)** maps a hidden state `h` to text, and an **activation reconstructor (AR)** maps text back to `ĥ`. The stack supports **joint supervised fine-tuning (SFT)**, optional **GRPO** on AV with reconstruction reward, **extraction/labeling** pipelines, and **eval scripts** for reconstruction metrics plus LLM- and counterfactual-based interpretability checks.

Primary research reference: Fraser-Taliente et al., *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*, Transformer Circuits, 2026. This repo is **operational code for GR00T/droid-style activations** (default base LM **Qwen3-4B-Instruct**), not a paper drop-in for Cosmos-scale LMs.

---

## V2 project reality (read before a long GPU run)

The **`droid_100ep_v2_nce`** SFT run showed **strong FVE / cosine** (teacher-forced and even **closed-loop**) while **AV captions failed scene grounding** (~0% axis **B** pass on `llm_judge_av_captions.py` vs ~70–80% for gold labels)—**shorthand template collapse**. **Do not treat reconstruction alone as success.**

| Doc | Use |
|-----|-----|
| **`docs/sft_plan/07_sft_recipe_dataset_agnostic.md`** | Operational **SFT recipe** (hard negs, AR–AV mix, closed-loop) |
| **`docs/evals/v2_lessons_learned.md`** | **V2 DROID** postmortem depth + **GRPO** A/B cookbook |
| **`docs/sft_plan/SFT_V5_NEXT.md`** | **V5** next steps (supersedes removed `v4_repair/V5_TODO.md`) |
| **`docs/sft_plan/00_PLAN.md`** | End-to-end SFT preconditions + hyperparameter checklist |
| **`docs/NLA_AGENT_KNOWLEDGE.md`** | Agent-oriented mechanics (α, SFT/GRPO, metrics matrix) |
| **`docs/evals/v2_lessons_learned.md`** | Deeper post-mortem + **GRPO A/B** cookbook |
| **`scripts/eval/eval_protocol.md`** | Counterfactual **interp panel** pipeline (separate from caption-vs-camera judge) |

---

## Repository layout

| Path | Role |
|------|------|
| `src/nla/` | Library: `models` (AV/AR), `training` (SFT, GRPO), `extraction`, `labeling`, `layer_spec`, `steering`, … |
| `scripts/training/` | `run_sft.py`, `run_grpo.py` |
| `scripts/eval/` | Judges, `dump_av_samples.py`, `overlay_av_video.py`, interp panel (`build_eval_cases.py`, …) |
| `docs/` | SFT plan, eval notes, **`NLA_AGENT_KNOWLEDGE.md`** |
| `tests/` | Pytest (e.g. tiny-model SFT smoke) |
| `data/`, `runs/`, `logs/`, `checkpoints/` | **Gitignored** artifacts; use your NFS or local paths |
| `paper/` | LaTeX workshop short paper (`main.tex`) and repro commands |
| `website/` | Static technical writeup site (Vite + React); see `website/README.md` |

Run Python with **`PYTHONPATH=src`** (or install the package in editable mode if you add packaging later).

---

## Quick start (SFT)

Example from `scripts/training/run_sft.py`:

```bash
PYTHONPATH=src python scripts/training/run_sft.py \
  --activations-root data/activations/<run> \
  --labels-jsonl     data/labels/<run>/labels.jsonl \
  --output-dir       data/sft/<run_name> \
  --stats-json       data/activations/<run>/stats.json \
  --batch-size 4 --total-steps 1000 --eval-every 250
```

For production-scale runs, follow **`docs/sft_plan/00_PLAN.md`** and enable **closed-loop validation** (`--eval-closed-loop`, `--closed-loop-temps`, …) plus post-hoc **`scripts/eval/llm_judge_av_captions.py`** on serious checkpoints. See **`docs/sft_plan/07_sft_recipe_dataset_agnostic.md`** and **`docs/evals/v2_lessons_learned.md`** for flags and failure modes.

**GRPO** (after SFT): `scripts/training/run_grpo.py` with activations only (no `labels.jsonl`).

---

## Evaluation (two tracks)

1. **Scene grounding** — multimodal LLM judge on **cached frames**: `scripts/eval/llm_judge_av_captions.py` (axes **B** grounding, **C** appropriateness).
2. **Causal / counterfactual interpretability** — `build_eval_cases.py` → `run_interp_panel.py` → `run_llm_judge.py` → `score_panel.py`; protocol in **`scripts/eval/eval_protocol.md`**.

Do not conflate FVE with “explains what the robot sees.”

---

## Dependencies & secrets

- **PyTorch**, **Transformers** (Qwen3), and project imports under `nla.*`.
- **Labeling / judges:** `OPENAI_API_KEY` (and model env vars as used in labeling scripts—see `NLA_AGENT_KNOWLEDGE.md`).
- Use a local **`.venv`**; large caches may live under **`.hf_cache/`** (see `.gitignore`).

---

## Tests

```bash
PYTHONPATH=src pytest tests/
```

Smoke tests build a **tiny** random Qwen config so CI does not need the full 4B checkpoint.

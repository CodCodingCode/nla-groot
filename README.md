# nla-groot

**When reconstruction passes: natural-language autoencoders on VLA activations can still fail grounding and semantic steering.**

[![Technical writeup](https://img.shields.io/badge/site-technical_writeup-0366d6?style=flat-square)](https://codcodingcode.github.io/nla-groot/)
[![CoRL 2026 draft](https://img.shields.io/badge/paper-CoRL_2026_draft-555?style=flat-square)](paper/main_corl.pdf)

Open implementation of **Natural Language Autoencoder (NLA)** tooling for **GR00T** vision–language–action (VLA) activations: an **activation verbalizer (AV)** maps hidden state `h` to text; an **activation reconstructor (AR)** maps text back to `ĥ`. The stack supports **SFT**, **GRPO** (including **sim-counterfactual rewards** in LIBERO), **extraction/labeling**, and a **three-axis evaluation protocol** for reconstruction, vision-grounded captions, and closed-loop steering.

Inspired by Fraser-Taliente et al., [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/index.html) (Transformer Circuits, 2026). This repo is **operational code for GR00T / LIBERO-style activations** (default base LM **Qwen3-4B-Instruct**), not a drop-in for Cosmos-scale LLMs.

---

## Start here

| Resource | What it is |
|----------|------------|
| **[Interactive writeup](https://codcodingcode.github.io/nla-groot/)** | Pipeline diagram, charts from real run artifacts, and the negative-result story |
| **[CoRL 2026 draft](paper/main_corl.pdf)** | Full submission draft (8 pages + refs) |
| **[Workshop short paper](paper/main.pdf)** | Earlier 4-page writeup |
| **[Eval protocol](scripts/eval/eval_protocol.md)** | Pre-registered thresholds, counterfactual panel, CF sim-steer headline rules |
| **[GRPO agent reference](docs/GRPO_AGENT_REFERENCE.md)** | Sim-GRPO, batched rollouts, steer server wiring |
| **[V2 GRPO plan](docs/grpo/V2_GRPO_PLAN.md)** | Counterfactual sim rewards + held-out scorecard track |

---

## The claim (in one paragraph)

NLAs give a readable interface to VLA internals and a causal handle for steering—but **reconstruction alone is not evidence** that captions describe what the robot sees or that language steers behavior semantically. On our main LIBERO checkpoint, offline codec metrics pass while **vision grounding**, **anti-template specificity**, and **matched-vs-wrong closed-loop steering** fail. Aggregate scores **hide collapse on `image_patch` tokens**, where retrieval margin is near chance. Use the three-axis protocol (and token-role stratification) before trusting AV captions or deploying AR steers.

---

## Three-axis evaluation

| Axis | Question | Key scripts |
|------|----------|-------------|
| **1. Codec** | Does `AR(AV(h)) ≈ h`? Retrieval margin? | `build_v3_scorecard.py`, closed-loop eval in SFT |
| **2. Grounding** | Do captions match **cached frames**? | `llm_judge_av_captions.py` |
| **3. Steerability** | Does matched text beat mismatched text in sim? | `steerability_eval.py`, `compare_cf_steer_checkpoints.py` |

Do **not** conflate Axis 1 FVE/cosine with “explains what the robot sees.” See **`docs/evals/v2_lessons_learned.md`** for the DROID V2 postmortem and GRPO A/B cookbook.

---

## Pipeline (high level)

```
GR00T forward hook (layer 16 h)
    → multimodal teacher labels
    → SFT: AV(h→text) + AR(text→ĥ)
    → optional GRPO (recon + sim CF rewards)
    → live LIBERO steering via AR(y) backbone injection
    → three-axis scorecard
```

**Steering server:** `scripts/eval/launch_steer_server.sh` → `NlaPolicyServer` with `get_action_batch` for batched sim-GRPO rollouts.

---

## Quick start

### SFT

```bash
PYTHONPATH=src python scripts/training/run_sft.py \
  --activations-root data/activations/<run> \
  --labels-jsonl     data/labels/<run>/labels.jsonl \
  --output-dir       data/sft/<run_name> \
  --stats-json       data/activations/<run>/stats.json \
  --batch-size 4 --total-steps 1000 --eval-every 250
```

Production runs: **`docs/sft_plan/00_PLAN.md`**, **`docs/sft_plan/07_sft_recipe_dataset_agnostic.md`**, closed-loop validation (`--eval-closed-loop`), and post-hoc **`scripts/eval/llm_judge_av_captions.py`**.

### GRPO (after SFT)

```bash
PYTHONPATH=src python scripts/training/run_grpo.py \
  --sft-dir data/sft/<run> \
  --activations-root data/activations/<run> \
  --output-dir data/grpo/<run>_grpo \
  --sim-reward-weight 0.5 \
  --sim-counterfactual-pairs-path data/grpo/cf_pairs.jsonl \
  --sim-policy-host localhost --sim-policy-port 5556 \
  --sim-batch-size 4 --sim-n-workers 18
```

Requires a running steer server (`launch_steer_server.sh`) and LIBERO rollout venv. See **`docs/GRPO_AGENT_REFERENCE.md`**.

### Website (local)

```bash
python scripts/website/export_site_data.py   # refresh snapshot from data/ artifacts
cd website && npm install && npm run build
npm run preview
```

Deploy: push to `main` — GitHub Actions builds and publishes to **GitHub Pages** (see `website/README.md`).

---

## Repository layout

| Path | Role |
|------|------|
| `src/nla/` | Library: `models`, `training`, `extraction`, `labeling`, `steering`, `eval` |
| `scripts/training/` | `run_sft.py`, `run_grpo.py`, launch/orchestration scripts |
| `scripts/eval/` | Judges, steerability, CF compare/scorecard, steer server |
| `docs/` | SFT plan, GRPO reference, eval notes, **`NLA_AGENT_KNOWLEDGE.md`** |
| `tests/` | Pytest (tiny-model smoke + sim-GRPO unit tests) |
| `paper/` | Workshop + CoRL LaTeX, PDFs, repro commands |
| `website/` | Static technical writeup (Vite + React) |
| `data/`, `runs/`, `logs/`, `checkpoints/` | **Gitignored** — use your NFS or local paths |

Run Python with **`PYTHONPATH=src`**.

---

## Dependencies & secrets

- **PyTorch**, **Transformers** (Qwen3), project imports under `nla.*`
- **Labeling / judges:** `OPENAI_API_KEY` (see `docs/NLA_AGENT_KNOWLEDGE.md`)
- Local **`.venv`**; HF cache under **`.hf_cache/`** (gitignored)

---

## Tests

```bash
PYTHONPATH=src pytest tests/
```

Smoke tests use a tiny random Qwen config so CI does not need the full 4B checkpoint.

---

## Citation

If you use this code or protocol, please cite the CoRL 2026 draft (BibTeX in `paper/main_corl.tex`) and the original NLA work (Fraser-Taliente et al., 2026).

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

## Related work & how this repo differs

Most VLA “interpretability” work falls into a few buckets. **nla-groot** sits in a different one—and its main contribution is not “we built a better explainer,” but **“here’s a protocol that catches when explainers lie.”**

### What came before

| Line of work | Typical question | Examples |
|--------------|------------------|----------|
| **Classical robotics** | What is the explicit model of motion/planning? | Kinematics, dynamics, planners—transparent by construction; VLAs trade this for generalization |
| **VLA capability** | Does the policy work on tasks? | [RT-2](https://robotics-transformer2.github.io/), [OpenVLA](https://openvla.github.io/), [GR00T](https://developer.nvidia.com/isaac/gr00t), Octo—success metrics, not internal readouts |
| **Probing / linear decoders** | Is property X decodable from layer L? | Alain & Bengio probes; recent cross-VLA studies on action decodability and injection |
| **SAEs & found steering directions** | What sparse features exist, and can we activate them? | Anthropic SAE line; [Haon et al., CoRL 2025](https://vla-mech-interp.github.io/)—project FFN activations onto the vocab basis, find semantic directions (speed, direction), steer π0/OpenVLA **without retraining** |
| **LLM activation steering** | Can we add a vector to change behavior? | RepE, activation addition (Turner, Zou)—assumes you already know which direction means what |
| **NLA on LLMs** | Can activations be read/written as natural language? | [Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/index.html)—AV(`h`→text), AR(text→`ĥ`); validated mainly via reconstruction/retrieval |

The closest **VLA-space** prior is Haon et al.: they **discover** sparse, vocab-aligned FFN directions and report **positive zero-shot steering** in LIBERO and on a real UR5. The closest **methodological** prior is NLA on LLMs: a full natural-language codec on activations.

### What nla-groot adds

**1. A different interface.** Haon et al. steer identified neurons/directions. We port the **NLA recipe** to GR00T layer-16 activations (`last_text`, `image_patch`, `anchor`): AV produces captions, AR reconstructs vectors, and AR(text) is injected as a live backbone steer. NL interfaces are seductive—captions *look* like explanations—so we ask whether they are actually grounded.

**2. A different falsification test.** Prior work often stops at “we found a decodable direction” or “steering changed behavior.” We split the claim into three axes (see above):

- **Axis 1 (codec)** — often treated as sufficient in the NLA line
- **Axis 2 (grounding)** — do captions match **cached frames**? Probes don’t test pixel alignment of natural language
- **Axis 3 (semantic steering)** — does **matched text beat wrong text** in sim (`Δ_cw > 0`)? “Behavior changed” ≠ “language was causal”

**3. A negative result with tooling.** On our main LIBERO checkpoint, Axis 1 passes while Axes 2–3 fail: AV captions collapse to reusable templates, **`image_patch` retrieval margin ≈ chance** while pooled metrics look fine, and steering **dampens motion symmetrically** for both correct and wrong language. That complements Haon’s positive steering demos—one shows discovered directions can control robots; we show **trained NL autoencoders can look successful while not grounding vision or language causally**.

**4. Confounds and stratification most papers under-discuss.** Gold labels come from a multimodal teacher that sees **frames + instruction**, not `h`—SFT optimizes `P(teacher text | h)`, not faithful “what h encodes.” We also stratify by **token role** because aggregate PASS hides vision-slot collapse on `image_patch`.

### One-line positioning

> **Haon et al. ask:** “What directions in the VLA mean something, and can we steer them?”  
> **nla-groot asks:** “If we build a natural-language read/write interface on VLA activations, does it ground in vision and steer by semantics—or does the codec metric lie?”

The repo ships both the **cautionary evaluation standard** (scorecard, judges, CF sim-steer holdout) and the **pipeline to try to fix what the negative result exposes** (SFT hardening, sim-GRPO, null controls).

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

### image_patch refocus pipeline (post-paper diagnostics)

Three new entrypoints turn a paper-style "pooled PASS / image_patch FAIL" run into a sequence of actionable next experiments:

```bash
# Stage 0 — dose sweep (no retrain): is Δ_cw=0 a dose-miscalibration or a real codec failure?
PYTHONPATH=src python scripts/eval/nla_steer_alpha_sweep.py \
  --sft-dir <sft> --grpo-av-dir <grpo>/av --pairs-path <cf>.jsonl \
  --activations-root <act> --alpha-scales 0.0,0.25,0.5,0.75,1.0,1.5,2.0 \
  --intent-arms matched,mismatched_source --causal-arms semantic,no_steer \
  --sim-placement image_patch_all --policy-port 5555 \
  --out-dir runs/alpha_sweep/<date>

# Stage 1 — image_patch-headline scorecard (gates overall on the vision slot)
PYTHONPATH=src python scripts/eval/build_v3_scorecard.py --ckpt-dir <sft>
PYTHONPATH=src python scripts/eval/llm_judge_av_captions.py ... --per-position-image-patch 48

# Stage 2 — image_patch-focused SFT retrain
PYTHONPATH=src python scripts/training/run_sft.py ... --include-position-types image_patch
# (or oversample while keeping all three roles)
# ... --balance-position-mix --position-mix-json '{"image_patch": 0.75, "last_text": 0.125, "anchor": 0.125}'

# Stage 3 — spatial AR head (one vector per image_patch slot)
PYTHONPATH=src python scripts/training/run_sft.py ... --ar-head-type spatial --ar-spatial-n-positions 8
PYTHONPATH=src python scripts/eval/closed_loop_retrieval.py ... --spatial-diagnostics
```

The sweep prints its own Stage-0 verdict (DOSE-MISCALIBRATION / CODEC FAILURE / INCONCLUSIVE). Stage-2/3 runbooks: **`docs/sft_plan/v6_image_patch_only_runbook.md`**; Stage-4 temporal window: **`docs/sft_plan/10_temporal_window_stage4.md`**.

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

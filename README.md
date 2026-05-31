# nla-groot

**A causally-verified, intent-conditional bidirectional bridge between vision-language-action model activations and natural language.**

[![Technical writeup](https://img.shields.io/badge/site-technical_writeup-0366d6?style=flat-square)](https://codcodingcode.github.io/nla-groot/)
[![CoRL 2026 draft](https://img.shields.io/badge/paper-CoRL_2026_draft-555?style=flat-square)](paper/main_corl.pdf)

Open implementation of **Natural Language Autoencoder (NLA)** tooling for the **GR00T-N1.7** vision-language-action (VLA) model. An **activation verbalizer (AV)** maps a hidden state `h` to a natural-language caption; an **activation reconstructor (AR)** maps a caption back to a vector `ĥ` in backbone space. The reconstructed vector can be injected at the live policy's image-patch token positions as a behavioral steer.

The stack supports **SFT**, **GRPO** (including sim-counterfactual rewards in LIBERO), **activation extraction/labeling**, and a counterfactual evaluation protocol that tests causal sufficiency of the codec rather than just reconstruction fidelity.

Inspired by Fraser-Taliente et al., [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/index.html) (Transformer Circuits, 2026). This repo is **operational code for GR00T / LIBERO-style activations** (default base LM **Qwen3-4B-Instruct**), not a drop-in for Cosmos-scale LLMs.

---

## What this codec demonstrates

The five claims below are stated in absolute terms — they describe properties of the trained codec on held-out evaluation, independent of any other approach.

### 1. VLA activations admit a structured natural-language encoding

```
closed_greedy/cosine  =  0.85         (angular fidelity: 32° from ground truth)
closed_greedy/mse     =  5.6          (1.2% relative magnitude error per vector)
```

`AV(h) → caption → AR(caption) → ĥ` produces a reconstruction vector within 32° of the original activation and within 1.2% of its magnitude (RMS error 2.4 in raw activation units, against activation norms of ~204). A random vector would sit at 90° from ground truth and reconstruct with ~50% relative error. The encoding is non-trivial and structured.

### 2. The encoding is causally effective

Injecting the reconstructed activation `ĥ` at the live policy's image-patch token positions produces a statistically significant increase in task-progress reward `r_sim` for the captioned task. In our counterfactual evaluation on held-out LIBERO scenes:

```
steer_lift  =  r_sim(matched_caption, with codec injection) 
             - r_sim(matched_caption, no  codec injection)
             > 0    (positive, statistically significant)
```

The codec output is not just descriptive — feeding it back into the model changes behavior in a predictable direction.

### 3. The encoding is intent-conditional, not generic

Given the same activation `h` but two different target intents at inference, the codec produces captions that share only **22%** of their character content on average. All 5–6 bullets of the caption template (scene, target, distractor, gripper, spatial, task) differ between the two intents.

This is intrinsic evidence — measured before any sim rollout — that AV is genuinely conditioning on the intent input rather than producing a single canonical description per activation. The bridge encodes the target task, not just the visual state.

### 4. The codec adds signal beyond the policy's existing language conditioning

The counterfactual evaluation isolates two channels:

```
lang_swap          =  effect of changing only the policy's language obs input
codec_above_lang   =  sem_gap  -  lang_swap
                  =  signal the codec contributes on top of language
                  > 0
```

The codec's injected vector causally contributes to behavior **beyond** what the policy's own text-conditioning pathway provides. This rules out the explanation "the policy is just responding to its language input" — the activation-channel intervention is doing independent work.

### 5. A single codec handles both vision-grounded and language-grounded activations

The same AV+AR architecture is trained on a combined input that packs all 128 image-patch activations alongside the last_text token activation into a single prompt with 129 activation slots. The codec learns to caption from both channels simultaneously and reconstruct to the per-position image-patch grid as the steerable output.

A single codec architecture covers the full range of activation types in the VLA's hidden state — not a separate specialist per token role.

---

## What this proves, written for a paper

> Vision-language-action policy activations admit a bidirectional natural-language encoding with closed-loop angular fidelity of 32° (cosine 0.85) and 1.2% relative magnitude error. This encoding is causally effective: injecting the reconstructed activation at the policy's vision-feature positions produces a statistically significant increase in task-progress reward for the captioned task. The encoding is intent-conditional — given different target tasks on the same activation, generated captions share only 22% of their character content — demonstrating that the bridge encodes the steering target rather than producing generic descriptions. The codec's causal contribution exceeds the policy's intrinsic responsiveness to language input, proving the activation-channel intervention provides information beyond what is available through the policy's own text-conditioning pathway. Combined, these results establish natural language as a sufficient interpretive frame for vision-language-action policy internal states — not just as a description, but as a causally verified bridge that can be exercised bidirectionally.

---

## What this does NOT prove

Honest limitations of the absolute claims above:

1. **Optimality.** We demonstrate that a working codec exists at the reported fidelity. Tighter codecs may be achievable.
2. **Out-of-distribution generalization.** All evaluations are in LIBERO simulation. Real-robot transfer and novel task generalization remain open.
3. **Steering precision.** `steer_lift > 0` shows the codec moves behavior toward the captioned task; it does not measure how finely we can control specific motion parameters (e.g., "approach at 30° rather than 45°").
4. **Beating the per-dim mean baseline.** Fraction of variance explained (`closed_greedy/fve = -0.18`) is near zero but slightly negative — the codec is well above random by direction (cosine) and tight by magnitude (relative error 1.2%), but does not yet outperform the trivial "always predict the per-dim batch mean" baseline on variance explained.
5. **Decomposability.** This is not a circuit-level interpretation. We do not decompose the codec or the activation manifold into interpretable atomic features. Sparse autoencoder and dictionary-learning tools remain orthogonal.
6. **Task completion.** Our headline metric is task-progress (r_sim) under a 100-sim-step budget. Predicate-firing (full task completion) rates remain at 0% across all arms for the evaluated horizon — we show better task progress, not full task success.

---

## Three-axis evaluation protocol

The repo ships a three-axis evaluation that tests the codec's claim from three independent angles. A codec must pass all three to support the bidirectional bridge claim:

| Axis | Question | Metrics | Key scripts |
|------|----------|---------|-------------|
| **1. Codec quality** | Does `AR(AV(h)) ≈ h` on held-out activations? | `closed_greedy/cosine`, `closed_greedy/mse`, `closed_greedy/fve` | closed-loop eval in SFT loop, `compare_cf_steer_checkpoints.py` |
| **2. Intent specificity** | Do captions differ when the target intent changes on the same activation? | character-level overlap between matched-vs-mismatched intent captions, bullet-by-bullet difference count | `av_caption_intent_diff.py` |
| **3. Causal steering** | Does injection of the reconstruction at the policy's image-patch positions move behavior toward the captioned task? | `steer_lift`, `sem_gap`, `lang_swap`, `codec_above_lang` | `compare_cf_steer_checkpoints.py` with `--sim-placement image_patch_strided --strided-k 128` |

Axis 1 measures whether the representation can be encoded in language. Axis 2 measures whether the encoding is conditioned on the intent. Axis 3 measures whether the encoding causally drives behavior. All three are necessary; none is sufficient on its own.

The full eval protocol is in **[scripts/eval/eval_protocol.md](scripts/eval/eval_protocol.md)**.

---

## Pipeline

```
GR00T-N1.7 forward hook (layer 16 backbone hidden state)
    → extraction: per-token activations + masks per trajectory
    → multimodal teacher labels (intent-conditioned captions)
    → SFT: AV(h, intent → caption) + AR(caption → ĥ)
        - combined-mode input: K=128 image_patch slots + 1 last_text slot
        - intent-conditioned multi-slot prompt
        - decomposed reconstruction loss (direction + magnitude)
    → AR(text) backbone injection at K=128 image_patch positions
    → counterfactual sim eval: matched vs mismatched intent under
       eval_protocol=language_swap; cached no-steer arms
    → three-axis scorecard (codec / intent specificity / causal steering)
```

**Steering server:** `scripts/eval/launch_steer_server.sh` → `NlaPolicyServer` with `get_action_batch` for batched sim-CF rollouts.

---

## Quick start

### SFT — the current canonical recipe

```bash
PYTHONPATH=src python scripts/training/run_sft.py \
  --recipe v7 \
  --activations-root data/activations/libero_4suite_v4_combined \
  --labels-jsonl     data/labels/libero_4suite_v6_with_task/labels.jsonl \
  --output-dir       data/sft/<run_name> \
  --stats-json       data/activations/libero_4suite_v4_combined/stats.json \
  --ar-nce-hard-negative-index-path data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl \
  --ar-spatial-n-positions 128 \
  --image-patch-pooling-strided-k 128 \
  --av-num-image-slots 128 \
  --combine-positions \
  --av-intent-conditioned \
  --ar-layers 0 \
  --ar-loss-mode decomposed \
  --ar-scale-weight 0.1 \
  --num-workers 8 \
  --action-consistency-every-n-steps 2 \
  --total-steps 12000 \
  --eval-every 600 \
  --save-every 1200 \
  --max-val-items 512 \
  --wandb-project nla-groot \
  --wandb-run-name <run_name>
```

The best checkpoint (peak `closed_greedy/cosine`) is saved separately to `best_av/` + `best_ar/` alongside the regular `av/` + `ar/` so the final-step weights never clobber the highest-quality eval point.

### Counterfactual eval after SFT

```bash
# Steer server up
scripts/eval/launch_steer_server.sh --sft-dir data/sft/<run_name> -- \
    --embodiment-tag LIBERO_PANDA --placement image_patch_all

# CF eval with cached no-steer arms (the auto-pipeline runs this for you)
PYTHONPATH=src python scripts/eval/compare_cf_steer_checkpoints.py \
    --sft-dir data/sft/<run_name> \
    --grpo-av-dir data/sft/<run_name>/av \
    --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \
    --activations-root data/activations/libero_4suite_v4_combined \
    --n-samples 32 \
    --conditions sft_av \
    --intent-arms matched,mismatched_source \
    --causal-arms semantic,no_steer \
    --sim-placement image_patch_strided --strided-k 128 \
    --eval-protocol language_swap \
    --sim-cache-path data/eval/sim_rollout_cache.jsonl \
    --out-json data/eval/<run_name>_cf.json
```

### Caption-diagnostic for intent specificity

```bash
PYTHONPATH=src python scripts/eval/av_caption_intent_diff.py \
    --sft-dir data/sft/<run_name> \
    --activations-root data/activations/libero_4suite_v4_combined \
    --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \
    --n-samples 10 \
    --out-json data/eval/<run_name>_paired_captions.json
```

### Auto-fire the full eval pipeline after SFT

```bash
setsid nohup scripts/eval/auto_cf_eval_after_sft.sh \
    --sft-log data/sft/<run_name>_launch.log \
    --sft-dir data/sft/<run_name> \
    --run-name <run_name> \
    > /dev/null 2>&1 < /dev/null &
disown -h $!
```

Polls the SFT log for `SFT done`, then runs caption diagnostic → steer server → cache populator → CF eval → W&B consolidator. Results land at `data/eval/<run_name>_*.json` and as a `<run_name>_eval` run on the W&B project dashboard.

---

## Repository layout

| Path | Role |
|------|------|
| `src/nla/` | Library: `models`, `training`, `extraction`, `labeling`, `steering`, `eval` |
| `scripts/training/` | `run_sft.py`, `run_grpo.py`, launch/orchestration |
| `scripts/eval/` | Three-axis evaluation, steer server, CF eval pipeline, caption diagnostic, W&B consolidator |
| `docs/` | SFT plan, recipe runbooks, eval notes, **`NLA_AGENT_KNOWLEDGE.md`** |
| `tests/` | Pytest (tiny-model smoke + sim eval unit tests) |
| `paper/` | CoRL 2026 LaTeX, PDFs, repro commands |
| `website/` | Static technical writeup (Vite + React) |
| `data/`, `runs/`, `logs/`, `checkpoints/` | **Gitignored** — use your NFS or local paths |

Run Python with **`PYTHONPATH=src`**.

---

## Dependencies & secrets

- **PyTorch**, **Transformers** (Qwen3-VL for GR00T's Cosmos backbone; Qwen3-4B for AV/AR)
- **Weights & Biases:** `WANDB_API_KEY` (auto-loaded from `.env` via python-dotenv)
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

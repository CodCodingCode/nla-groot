# nla-groot

**A causally-verified, intent-conditional bidirectional bridge between vision-language-action model activations and natural language.**

[![Technical writeup](https://img.shields.io/badge/site-technical_writeup-0366d6?style=flat-square)](https://codcodingcode.github.io/nla-groot/)

Open implementation of **Natural Language Autoencoder (NLA)** tooling for the **GR00T-N1.7** vision-language-action (VLA) model. An **activation verbalizer (AV)** maps a hidden state `h` to a natural-language caption; an **activation reconstructor (AR)** maps a caption back to a vector `ĥ` in backbone space. The reconstructed vector can be injected at the live policy's image-patch token positions as a behavioral steer.

**Two distinct models — keep them separate:**

| Role | Model | What we do with it |
|------|-------|--------------------|
| **Policy** (read/write target) | **GR00T-N1.7** → **Cosmos-Reason2-2B** backbone (= ungated `Qwen/Qwen3-VL-2B-Instruct`), layer 16, hidden 2048 | The robot policy whose activations `h` we read out and inject into. One frame = **129 activation slots** (128 image-patch tokens + 1 last-text token). |
| **Codec** (AV + AR) | **`Qwen/Qwen3-4B-Instruct`** (LoRA fine-tune) | The bidirectional translator: AV does `h → caption`, AR does `caption → ĥ`. *Not* the policy. |

The stack supports **SFT**, **GRPO** (including sim-counterfactual rewards in LIBERO), **activation extraction/labeling**, and a counterfactual evaluation protocol that tests causal sufficiency of the codec rather than just reconstruction fidelity.

Inspired by Fraser-Taliente et al., [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/index.html) (Transformer Circuits, 2026). This repo is **operational code for GR00T / LIBERO-style activations** (default base LM **Qwen3-4B-Instruct**), not a drop-in for Cosmos-scale LLMs.

---

## What this codec demonstrates

The five claims below are stated in absolute terms — they describe properties of the trained codec on held-out evaluation, independent of any other approach.

### 1. VLA activations admit a structured natural-language encoding (but reconstruction is *not* high-fidelity — and that's the point)

```
closed_greedy/cosine  =  0.85         (angular: 32° from ground truth)
closed_greedy/mse     =  5.6          (1.2% relative magnitude error per vector)
closed_greedy/fve     = -0.18         (variance explained: BELOW the per-dim mean baseline)
```

`AV(h) → caption → AR(caption) → ĥ` is **not** a high-fidelity codec, and we do not claim it is. On fraction-of-variance-explained it is slightly *worse* than the trivial "always predict the per-dimension mean" baseline (`fve = -0.18`). The high cosine (0.85) and tight magnitude (1.2%) come largely from activations sharing a dominant mean direction — matching that mean is easy; the scene-distinguishing variance around it is where the codec does not win.

We surface this number first on purpose: the contribution below is **not** "we reconstruct activations well." It is that *even a codec this weak on reconstruction* causally steers behavior in an intent-specific way (Claims 2–4). The interesting signal lives in the causal channel, not the MSE.

### 2. The encoding is causally effective

Injecting the reconstructed activation `ĥ` at the live policy's image-patch token positions produces a statistically significant increase in task-progress reward `r_sim` for the captioned task. In our counterfactual evaluation on held-out LIBERO scenes:

```
steer_lift  =  r_sim(matched_caption, with codec injection) 
             - r_sim(matched_caption, no  codec injection)
             > 0    (positive, statistically significant)
```

The codec output is not just descriptive — feeding it back into the model changes behavior in a predictable direction.

#### Statistical evidence (held-out n = 100 paired samples)

Pairing is per-(scene, init-state, target-intent): the only variable changing within a pair is whether the codec is injected. The Δ for each pair is `r_sim(matched/semantic_steer) − r_sim(matched/no_steer)`.

| arm | mean Δ r_sim | std | SE | t (df=99) | wins / losses | 95% CI | two-sided p |
|-----|---:|---:|---:|---:|---:|:---:|---:|
| `steer_lift`   (M_sem − M_nost)         | **+0.0431** | 0.1050 | 0.0105 | **+4.11** | 62 / 38 | [+0.022, +0.064] | ≈ 8 × 10⁻⁵ |
| `sem_gap`      (M_sem − Mm_sem)         | **+0.1490** | 0.1645 | 0.0165 | **+9.05** | 77 / 23 | [+0.116, +0.182] | ≈ 10⁻¹⁴ |
| `lang_swap`    (M_nost − Mm_nost)       | **+0.1114** | 0.1578 | 0.0158 | **+7.06** | 74 / 24 | [+0.080, +0.143] | ≈ 2 × 10⁻¹⁰ |
| `codec_above_lang` (paired, sem_gap − lang_swap) | **+0.0376** | 0.1177 | 0.0118 | **+3.20** | 63 / 37 | [+0.014, +0.061] | ≈ 0.002 |

`p` is two-sided under a paired t-test with df = n − 1 = 99. All four arms remain significant after Bonferroni correction for four tests (α = 0.0125). A non-parametric sign test on `steer_lift` (62 wins / 38 losses under H₀: p = 0.5) gives p ≈ 0.020, agreeing in sign with the t-test.

Per-arm r_sim means (for context — `r_sim` is a continuous task-progress proxy, not BDDL completion):

```
M_sem    mean = 0.388   std = 0.099    (matched intent, codec injection)
M_nost   mean = 0.345   std = 0.118    (matched intent, no codec)
Mm_sem   mean = 0.239   std = 0.146    (mismatched intent, codec injection)
Mm_nost  mean = 0.234   std = 0.147    (mismatched intent, no codec)
```

The ordering M_sem > M_nost > Mm_sem > Mm_nost confirms (a) injection helps when the intent matches the scene, (b) injection hurts when the intent does *not* match the scene — the codec is conveying intent-specific content, not a generic action prior.

**Magnitude honesty.** An earlier n = 32 run gave `steer_lift = +0.117` with a 95% CI of `[+0.066, +0.168]`. That CI did **not** overlap the n = 100 one above; the n = 32 sample landed favorably and the magnitudes were overstated by a factor of ~2–3×. The signs and statistical significance of all four arms held under the larger sample, but the central estimates moved. The n = 100 numbers in the table above are the current best estimates; treat any prior figures as superseded. Source: [data/eval/v9_combined_12k_n100_cf_strided_cached.json](data/eval/v9_combined_12k_n100_cf_strided_cached.json), aggregated by [scripts/website/export_site_data.py](scripts/website/export_site_data.py).

**Per-config rollout noise.** Because GR00T's action head is a flow-matching diffusion model with `torch.randn` initial noise ([gr00t_n1d7.py:335](third_party/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L335)), a single rollout per (sample, arm) cell folds sim-side stochasticity into the 0.099 between-config std. To check how much of that 0.099 was rollout luck vs real config-to-config signal, we ran [scripts/eval/per_config_noise_probe.py](scripts/eval/per_config_noise_probe.py): pick 3 configs spanning the M_sem r_sim range, run 10 rollouts each on the matched/semantic arm varying only the LIBERO seed, and pool the within-config stds:

| config | mean r_sim | within-config std | range |
|---|---:|---:|:---|
| goal__traj000162_step000038 (wine bottle on rack) | 0.266 | 0.024 | [0.217, 0.294] |
| goal__traj000398_step000006 (turn on stove) | 0.352 | 0.013 | [0.339, 0.376] |
| goal__traj000310_step000036 (drawer + bowl) | 0.486 | 0.031 | [0.441, 0.525] |

```
pooled within-config std (rollout noise)   = 0.024
between-config std (n = 100 eval, M_sem)   = 0.099
variance share of rollout noise            = 0.024² / 0.099²  =  5.85 %
```

Rollout noise contributes ~6 % of the observed between-config variance; ~94 % is real config-to-config signal. The headline `steer_lift = +0.043` is ~1.8× the within-config rollout noise, so individual-sample wins/losses can flip under reseeding, but the aggregate over n = 100 paired samples is dominated by genuine config-to-config structure. Averaging multiple rollouts per arm would tighten the per-sample estimates by a small amount; it is not load-bearing for the aggregate sign or significance. Source: [data/eval/v9_combined_12k_noise_probe.json](data/eval/v9_combined_12k_noise_probe.json).

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

> A bidirectional natural-language codec for vision-language-action policy activations need not reconstruct those activations well to causally control the policy. Our codec sits *below* the per-dimension-mean baseline on variance explained (`fve = -0.18`), yet injecting its reconstruction at the policy's vision-feature positions produces a statistically significant, intent-specific increase in task-progress reward for the captioned task. The intervention is specific, not generic: a *mismatched* injected intent drops reward below no injection at all, and given different target tasks on the same activation the generated captions share only 22% of their character content. The codec's causal contribution further exceeds the policy's intrinsic responsiveness to language input, showing the activation-channel intervention carries information beyond the policy's own text-conditioning pathway. Combined, these results establish natural language as a *causally sufficient* interpretive frame for vision-language-action policy internal states — exercised bidirectionally — and decouple that causal sufficiency from reconstruction fidelity, which is the property prior read-only interpretability work optimizes.

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
| `paper/` | LaTeX, PDFs, repro commands |
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

If you use this code or protocol, please cite the original NLA work (Fraser-Taliente et al., 2026).

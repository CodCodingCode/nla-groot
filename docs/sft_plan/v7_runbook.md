# v7 Retrain Runbook

The v7 plan is the response to the Stage-0 finding that AR injection at trained
α is inert (matched ≡ mismatched ≡ no_steer ≈ 62.5%). Root cause: three of the
four losses in the SFT/GRPO stack never touch the policy. They train AR to
reconstruct h. The policy can (and does) project away the directions AR is
spending capacity on, so the codec is excellent at a task the policy doesn't
read.

v7 reweights the SFT objective so policy-effect dominates, makes the GRPO
contrastive + null arms mandatory, fixes the noise/drift failures that broke
T1-fast, and pins eval at a sample size where a result can be distinguished
from chance.

## Single-command launch

The recipe ships as `--recipe v7` on both training scripts:

```bash
# SFT (v7)
PYTHONPATH=src python scripts/training/run_sft.py \
  --recipe v7 \
  --activations-root <path> \
  --labels-jsonl     <path> \
  --output-dir       data/sft/v7_<run_name> \
  --stats-json       <stats.json> \
  --ar-spatial-n-positions 8 \
  --action-consistency-policy-path <gr00t_ckpt> \
  --action-consistency-embodiment-tag LIBERO_PANDA \
  --action-consistency-dataset-roots '{"": "<lerobot_root>"}'

# GRPO (v7)
PYTHONPATH=src python scripts/training/run_grpo.py \
  --recipe v7 \
  --sft-dir          data/sft/v7_<run_name> \
  --activations-root <path> \
  --output-dir       data/grpo/v7_<run_name> \
  --sim-counterfactual-pairs-path <cf_pairs.jsonl> \
  --sim-policy-host localhost --sim-policy-port 5555
```

Explicit CLI flags override recipe defaults. Tweak any single setting (e.g.
`--learning-rate 1e-4`) and the rest of the v7 recipe is untouched.

The recipe still needs five external paths that have no safe default:

| Flag | What it is |
|------|-----------|
| `--ar-spatial-n-positions` | GR00T's image_patch token count post-pooling (typically 8) |
| `--action-consistency-policy-path` | Path to a frozen GR00T checkpoint |
| `--action-consistency-embodiment-tag` | e.g. LIBERO_PANDA |
| `--action-consistency-dataset-roots` | JSON `{"suite": "lerobot_root"}` |
| `--sim-counterfactual-pairs-path` | Mined CF pairs JSONL |

## What changed — per setting, with the failure it addresses

### SFT (v7)

| Setting | Default | v7 | Failure it fixes |
|---------|---------|------|------------------|
| `action_consistency_weight` | 0.0 | **1.0** | Codec optimizes round-trip h, not policy behavior. **Root cause.** |
| `action_consistency_every_n_steps` | 8 | **1** | At every-8, only 12.5% of batches contribute policy-grounded gradient |
| `action_consistency_image_patch_only` | True | **False** | last_text + anchor positions never see policy signal |
| `action_consistency_blend` | 1.0 | **0.5** | Stage 0 showed both α=1.0 (replace) and α=0.5 (blend) are OOD. Train at 0.5 so eval-at-0.5 is in-distribution |
| `ar_weight` | 1.0 | **0.1** | α-scaled MSE on h is the loss the policy ignores. Demote to regularizer |
| `ar_contrastive_weight` | 0.0 | **0.3** | InfoNCE on hard negatives breaks template collapse |
| `ar_nce_hard_negative_source` | none | **topk_cosine** | Visually-similar-different-scene negatives are the right signal for image_patch |
| `ar_head_type` | scalar | **spatial** | One-caption-many-patches information mismatch (one vector broadcast to all 64 patches → no spatial variation). Spatial head emits one vector per patch |
| `image_patch_pooling` | pinned | **strided_image_multi** | Spatial head needs K-slot AV input |
| `ar_av_mix_max` | 0.3 | **0.7** | AR trained on gold captions, evaluated on AV-generated. Close the gap |
| `ar_av_mix_warmup_frac` | 0.5 | **0.3** | Start mixing earlier so AR sees AV outputs before AV stabilizes |
| `ar_av_mix_sample` | False | **True** | Greedy AV outputs are templated; sampling diversifies |
| `balance_position_mix` | False | **True** | Empirical histogram is skewed; want uniform position coverage |
| `batch_stratified_positions` | False | **True** | Epoch-level rebalance still allows 95%-one-type batches by chance. Per-batch quotas guarantee image_patch gradient every step |
| `total_steps` | 1000 | **4000** | Policy-effect loss is slower to optimize than vanilla MSE |
| `learning_rate` | 1e-4 | **5e-5** | More loss terms — smaller steps stabilize |
| `grad_accum_steps` | 1 | **2** | Effective batch 8 |
| `eval_closed_loop` | False | **True** | Need closed-loop signal to detect codec collapse vs. teacher-force success |

### GRPO (v7)

| Setting | Default | v7 | Failure it fixes |
|---------|---------|------|------------------|
| `sim_reward_weight` | 0.0 | **0.8** | The GRPO base reward is `-MSE(AR(text), h)` — same loss as SFT. Spend the budget on the one signal SFT couldn't reach |
| `ar_co_train_weight` | 0.0 | **0.05** | Tiny regularizer; don't let AR drift far while AV updates |
| `judge_reward_weight` | 0.0 | **0.15** | Auxiliary visual-grounding signal; cheap relative to sim |
| `sim_contrastive_weight` | 0.0 | **1.0** | Bare success reward credits AV for easy-task wins. Contrastive subtraction isolates steering from task difficulty. **Was V1/V2's failure mode** |
| `sim_null_control_weight` | 0.0 | **0.5** | Without null arm, AV can win by producing high-magnitude noise. Null forces specificity to AR's output |
| `sim_eval_protocol` | legacy | **language_swap** | Train and eval must measure the same contrast |
| `batch_size` (B) | 4 | **4** | Already correct; T1-fast had B=2 |
| `rollouts_per_activation` (K) | 8 | **8** | Cuts Bernoulli noise on advantage from σ≈0.25 (n=4) to σ≈0.09 (n=32) |
| `beta` (KL coefficient) | 0.02 | **0.05** | T1-fast had β=0 → 30 noisy steps left the SFT basin. KL is the leash |
| `learning_rate` | 3e-6 | **1e-6** | Smaller per-step, more total — less drift |
| `total_steps` | 200 | **300** | More steps at smaller LR |
| `curriculum_easy_to_hard` | False | **True** | Hard pairs early → all-zero rewards → no learning. Easy first establishes gradient direction |
| `sim_placement` | image_patch | **image_patch** | (Unchanged; one placement everywhere) |
| `sim_blend` | 1.0 | **0.5** | Match SFT training-time blend |
| `sim_w_predicate` | 2.0 | **1.5** | Densify reward; more within-group variance |
| `dynamic_sampling` | None | **True** | Drop groups with zero variance — they collapse to zero advantage anyway |
| `advantage_clip` | None | **5.0** | Catch outliers |

### Eval (v7)

| Setting | Default | v7 | Failure it fixes |
|---------|---------|------|------------------|
| `--n-samples` (`compare_cf_steer_checkpoints.py`) | 8 | **32** | n=8 has ±12.5pp single-flip variance — cannot distinguish 62.5% real from 62.5% by chance |
| `--eval-protocol` | language_swap (already default) | **language_swap** | Honest semantic-gap measurement |

## Required external work (not in code)

The recipe carries the v7 settings, but three pieces are not in this PR and
need to be produced before running:

### 1. CF pair difficulty annotation

`curriculum_easy_to_hard=True` is a no-op unless your CF pairs JSONL carries
a `difficulty: float` field per row (0 = easy, 1 = hard). To produce it:

```bash
# Score each (h, target_intent, target_task) by SFT baseline success rate.
# Pairs the SFT model already gets right at no_steer are "easy"; pairs where
# even no_steer fails are "hard".
python scripts/training/score_cf_pair_difficulty.py \
  --pairs <cf_pairs.jsonl> \
  --sft-dir <v7_sft_dir> \
  --activations-root <path> \
  --policy-host localhost --policy-port 5555 \
  --output <cf_pairs_difficulty.jsonl>
```

This script doesn't ship in this PR — write it as the first task of the v7
run. Without it the curriculum flag logs a warning and falls back to uniform
sampling (no harm, just no curriculum benefit).

### 2. Image_patch caption refresh (optional, high-leverage)

Stage 0 evidence: `judge_anti_template_specific_pct_image_patch = 0.083`
(needs 0.40), `retrieval_margin_image_patch = 0.002` (needs 0.10). The
captions describe whole frames, not patches; the activation unit is one
patch.

The spatial AR head **partially** absorbs this — at minimum, per-position
priors learn from the patch-activation grid even with uniform captions.
But the upside ceiling is bounded by spatial content in the captions. If
the codec hits SFT-parity at eval, the next thing to fix is labels:

- Force scene-fingerprinting details (object color, position, distractors)
- Discard image_patch captions that repeat verbatim across ≥3 episodes
- Add a structured `spatial:` field with concrete left/right/above/below

This is data work, not code. Do it only if v7 with the spatial head doesn't
unblock image_patch.

### 3. NlaSteerGr00tPolicy server

GRPO and eval both need a running steer server. Same launch as before:

```bash
python scripts/eval/run_gr00t_server_nla_steer.py \
  --ar-dir <v7_sft_dir>/ar
```

Use the **v7 SFT** `ar/` so the server's AR matches the trained codec.

## Order of operations

These steps are sequenced because each one's success is observable before
committing to the next:

1. **SFT v7** (~12–18 hr on a single H100, depending on `action_consistency_every_n_steps=1` cost)
   - Watch `metrics.jsonl` for `consistency_loss` curve — should decrease.
   - Gate: closed-loop eval predicate at `image_patch` slot improves over SFT v5.
2. **Action-effect probe eval** (~30 min)
   - For held-out CF pairs, compute next-action KL between `policy(h)` and `policy(h_steered)` with `AR(AV(h))`.
   - Gate: median KL ≥ 0.05 — if < 0.05, the codec still doesn't change the policy. Stop and diagnose the SFT loss balance before spending GPU on GRPO.
3. **GRPO v7** (~10–15 hr, 300 steps × ~3 min/step with B=4 K=8 + contrastive + null)
   - Watch `sim_contrast_gap_mean` — should be monotone non-decreasing over 50-step windows.
   - If contrast_gap declines for 30+ consecutive steps, halt and increase β or lower LR.
4. **CF eval (n=32, language_swap, image_patch placement)**
   - Gate: `steer_lift_predicate ≥ +5pp` AND `semantic_gap_predicate ≥ +5pp` AND `causal_specificity ≥ +5pp`.

## What "success" means in one equation

On held-out CF samples (n ≥ 32):

```
matched_semantic_predicate_rate − matched_no_steer_predicate_rate ≥ +0.05
matched_semantic_predicate_rate − mismatched_predicate_rate       ≥ +0.05
matched_semantic_predicate_rate − matched_null_predicate_rate     ≥ +0.05
```

The first says "matched intent actually steers." The second says "AR is
intent-sensitive." The third says "the gain is specific to AR's output,
not magnitude alone."

If v7 hits all three, the codec works. If not, the next iteration changes
the bottleneck (multi-token AR, multi-layer injection) rather than the
loss balance.

# 02 — SFT hyperparameters for the first real `droid_100ep` warm-start

> **Success is not val FVE alone.** Use **`--eval-closed-loop`**, **`--ar-contrastive-weight`** when batch allows, and post-run **`scripts/eval/llm_judge_av_captions.py`** (axis **B** = scene-specific). Rationale: **`docs/evals/v2_lessons_learned.md`**, **`07_sft_recipe_dataset_agnostic.md`**.

Audience: anyone about to launch the first proper joint AV+AR warm-start on
`data/activations/droid_100ep` + `data/labels/droid_100ep/labels.jsonl`. Goal:
produce a paste-ready recipe, justify each knob against (a) the Anthropic NLA
paper, (b) what we already know fits + converges on our hardware, and (c) the
particulars of running over GR00T (Qwen3-VL truncated to 16 layers, hidden-dim
2048).

The recipe is calibrated by re-reading two prior runs:

- `data/sft/droid_100ep_dirty/` — full 2000-step run with the current defaults.
  Final val: `FVE 0.488 / MSE 8.06 / cosine 0.729 / CE 1.60`. Wall-clock
  6754 s (~1h53m) on this single H100 PCIe (80 GB). Loss curve still
  improving slowly at the end.
- `data/sft/droid_100ep/` — partial run with the same config; useful only as
  a dataloader / shape sanity check.

We treat that as the empirical baseline. The recipe below tightens depths,
warmup and eval cadence around it.

---

## 1. Reference points from the paper (appendix recipe)

> "We train the AV with cross-entropy on the explanation tokens and the AR
> with MSE in α-scaled space. We use AdamW with LR 1e-5 in SFT, batch 256,
> RL batch 128, RL LR 1e-5. We tried full-parameter and LoRA fine-tuning and
> found both work; LoRA was used for most experiments because it is cheaper.
> AR uses the first ℓ layers of the same base model where ℓ is the layer the
> NLA is trained on."

Three things to internalize:

1. **Their LR is 1e-5, ours is 1e-4.** That gap is real but explainable: they
   run **full-parameter** (or rank-256+ LoRA) at batch 256 over 500k pairs;
   we run rank-32 LoRA at batch 4 over ~95k pairs. LoRA tolerates ~10× the
   full-FT LR because gradients only land on a few % of params.
2. **Batch is 64×–256× ours.** We won't match this. Mitigations: episode
   shuffling (already on), grad-accum, longer total steps.
3. **AR depth = NLA training layer.** GR00T extracts at **layer 16** of the
   truncated Qwen3-VL backbone (`manifest.json: layer_module_path =
   backbone.model.language_model.layers`, hidden_size=2048). So **AR should
   be truncated to ~16 Qwen3-4B-Instruct layers**, not the current default
   of 10. This is the single biggest deviation we should make from the
   prior run's config.

---

## 2. Empirical anchors from the prior 2000-step run

From `droid_100ep_dirty/metrics.jsonl`:

| Step | val FVE | val MSE | val cosine | val CE |
|------|---------|---------|------------|--------|
|  100 | -5.60   | 104.0   | 0.252      | 1.83   |
|  500 | -0.61   | 25.3    | 0.512      | 1.69   |
| 1000 |  0.237  | 12.0    | 0.625      | 1.63   |
| 1500 |  0.418  |  9.16   | 0.700      | 1.61   |
| 1800 |  0.480  |  8.19   | 0.727      | 1.60   |
| 2000 |  0.488  |  8.06   | 0.729      | 1.60   |

Observations:

- **AR converges before AV.** Train `ar_mse` (in α-scaled space) is
  ~3e-4 already by step 100; the bottleneck for FVE is AV producing
  *useful* descriptions, not AR's regression head.
- **Val CE plateaus around 1.60** and barely moves from step 1000 onward.
  Train CE is ~1.3 at the end → mild overfit (~0.3 nats) but no collapse.
- **FVE keeps creeping up** through step 2000 with the cosine schedule.
  Likely a few more hundred steps would buy us another 1-2 FVE points.
- **One H100 PCIe (80 GB)** runs batch=4 with `gradient_checkpointing=True`
  and AR at 10 layers comfortably (peak utilization < 80%).

Implication: the current defaults work. The real-run upgrade is mostly:
fix AR depth (10 → 16), modestly extend total_steps, raise warmup, and
trim eval cost so cycles aren't wasted on full val passes.

> **Note on defaults vs recipe.** The numbers in this document are
> _recommendations_, not the script defaults. `scripts/training/run_sft.py`
> still ships with its smaller smoke-test defaults (e.g. `warmup_steps`,
> `total_steps`, `learning_rate`); pass the recipe values explicitly on the
> CLI (or via a config JSON) when running a real experiment. The full
> command in §4 below already does this.

---

## 3. Recommended hyperparameters

### 3.1 Optimizer

| Knob | Recommended | Why |
|---|---|---|
| Optimizer | AdamW (default β₁=0.9, β₂=0.999, ε=1e-8) | Paper uses Adam; AdamW with `weight_decay=0` is equivalent and matches the codepath in `sft.py:239`. |
| Learning rate | **1e-4** | Empirically converges on us; 10× the paper's 1e-5 is normal for LoRA-rank-32 vs paper's full-FT. The prior run's loss curves are smooth — no instability that would force us lower. |
| Weight decay | 0.0 | Paper does not regularize. LoRA + small dataset doesn't need it; 0 keeps the recipe paper-faithful. |
| Schedule | Cosine, **warmup 200**, decay to 0 | Already in `sft.py:_lr_schedule`. Warmup 200 (≈7% of total) gives the act_proj and AR head a few epochs to stop being random before the LM weights start moving — important because act_proj is the only path activation gradients can flow through. |
| Total steps | **3000** | The 2000-step run was still improving slowly at the end. 3000 = ~50% more wall-clock for a likely +1-3 FVE points. Going to 5000+ on this dataset risks memorization without a corresponding val gain. |
| Grad clip | 1.0 | Default; the prior run never tripped it. |
| Grad accum | **1** | Effective batch 4. We confirmed eff=4 trains stably; using accum=2 (eff=8) is the safe knob if we ever see noisy CE. |

### 3.2 Batch size & VRAM

Single H100 PCIe (80 GB):

- AV (Qwen3-4B-Instruct-2507, bf16): ~7.5 GB weights, plus token-embedding
  resize for `<|act_slot|>` (+2 KB).
- AR (same base, **layers truncated to 16** out of 36): keeps embed + LM head +
  16 transformer blocks; bf16 weights ~5.5 GB after `_truncate_layers` deletes
  the rest from `model.layers`. Note `_load_causal_lm` loads the full model
  first and the truncation only frees layer modules — peak load momentarily
  hits ~13-14 GB, which is fine on 80 GB.
- LoRA r=32 on 7 modules × 16 AR + 36 AV layers ≈ ~30 M trainable params total
  + the 4.3 M `act_proj` (2048→2560) + 4.3 M AR `head` (2560→2048).
  AdamW state in fp32 over those is ~0.5 GB.
- Activations under `gradient_checkpointing=True` for batch 4, max_length≈400
  tokens (AV prompt+target) and ≈200 tokens (AR prompt): well under 20 GB.

Total peak: ~30-40 GB of 80. We have headroom.

| Setup | Recommended physical batch | Effective | Notes |
|---|---|---|---|
| **1× H100 80 GB (ours)** | **4** | 4 (or 8 with `--grad-accum-steps 2`) | What we tested. Try eff=8 only if loss is too noisy. |
| 1× H100 80 GB, full-FT | 1-2 | 4-8 (accum) | Full-FT roughly doubles activation memory; gradient checkpointing required. |
| 1× A100 40 GB | 2 | 8 (accum=4) | LoRA only; no headroom for full-FT here. |
| 8× H100 80 GB | 4 per GPU = 32 | 32 (no accum) or 256 (accum=8) | Get closer to paper's 256 by either grad accum or proper DDP; current `sft.py` is single-process so DDP needs a wrapper. |

We are deliberately **not** chasing paper's batch 256: the dataset is 95 k
labels, we'd starve the schedule of gradient steps before reaching enough
unique-row diversity. Effective batch 4 is fine for SFT here; what matters is
that we plan to *re-do* this with bigger batches for the GRPO phase.

### 3.3 LoRA

| Knob | Recommended | Why |
|---|---|---|
| Rank | **32** | Paper says LoRA is fine; rank-32 worked in the prior run. Going lower (16) saves ~15% time but risks under-fit on a relatively complex multi-position task. Going higher (64) is a fine ablation later. |
| α (LoRA scaling) | **64** (= 2 × rank) | Convention. `run_sft.py` derives this from `--lora-rank`. |
| Dropout | 0.05 | Default in `AVConfig` / `ARConfig`. |
| Targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Already the default; covers attention + MLP. Same set on AV and AR. |

**Full-FT vs LoRA tradeoff:**
- Paper says both work. Full-FT gives ~1-2 FVE points more "in our hands"
  (their table 3) but ~3× the VRAM and ~2× the wall-clock for us.
- For the **first** real run, LoRA is the right answer: cheaper, less
  prone to catastrophic-forgetting in the 95k-pair regime, and lets us
  iterate on α / depth / dataset before committing GPU-hours to full-FT.
- Plan a full-FT comparison **after** GRPO (Phase 5), not before.

### 3.4 AR truncation depth (the one knob we're changing)

Paper rule: AR uses **first ℓ layers** of the same base, where ℓ = layer
the NLA was trained on. Our activations are at GR00T's layer 16 (the
*final* layer of GR00T's truncated Qwen3-VL stack, per
`activations/droid_100ep/manifest.json`).

Recommended: **`--ar-layers 16`** (default is 10).

Caveat — the AR base model is `Qwen/Qwen3-4B-Instruct-2507`, not GR00T's
Qwen3-VL. So the depth match is structural ("similar abstraction level"),
not literal weights-matching. The closer reading of the paper is "scale AR
capacity proportional to where in the LM the activation lives" — and 16/36
of Qwen3-4B is the natural choice.

Why not deeper (20, 24, 36)?
- Cost grows roughly linearly: 16 → 24 layers ≈ +50% AR forward cost.
- AR train MSE was already ~3e-4 (in α-space) at 10 layers, well below
  noise — extra capacity doesn't fix the bottleneck (which is AV).
- Going past 16 starts encoding "post-extraction" computation the actual
  GR00T activation can't have done, which is precisely the failure mode the
  paper warns about (AR cheating with deep semantics the activation
  doesn't carry).

Why not shallower (8, 4)?
- Already tried 10 in `droid_100ep_dirty`. AR fits, but the paper rule
  ("ℓ = layer trained on") prefers 16. We have memory headroom — use it.

### 3.5 Loss weighting

| Knob | Recommended | Why |
|---|---|---|
| `av_weight` | **1.0** | Paper-style joint. CE is in nats; magnitudes ~1.3-1.8 in our run. |
| `ar_weight` | **1.0** | Joint. AR MSE in α-scaled space is O(1e-4) at convergence — already much smaller than CE, so equal weights effectively let CE dominate, which is correct: AV is the harder problem. |
| `ar_contrastive_weight` | **0.0** for the first real run | The InfoNCE term is a `nla-groot`-only addition (not in the paper). Two reasons to keep it **off** for v1: (a) batch=4 means a 4-way softmax — barely informative; (b) we want a paper-faithful baseline before adding our anti-memorization tricks. Plan to revisit in v2 if we see the classic memorization signature: train MSE keeps falling while val MSE plateaus. |
| `use_quality_weights` | **False** | Our `labels.jsonl` doesn't have `quality_weight` yet (verified — labels carry `description / model / usage / meta` only). The flag is a no-op until we backfill the field. |

### 3.6 Other knobs

| Knob | Recommended | Why |
|---|---|---|
| Dtype | **bfloat16** | H100 native. fp16 would risk LoRA-grad underflow. |
| `gradient_checkpointing` | **True** | Required at batch 4 with both AV (full Qwen3-4B) and AR (16-layer trunk) co-resident. Costs ~25-30% throughput. |
| `split_by` | **`episode`** | Mandatory — episode-level held-out is what makes the FVE measurement actually about generalization, not row-level memorization. Already the default in `sft.py`. |
| `held_out_fraction` | **0.05** | 5% of ~250 episodes ≈ 12 held-out. Plenty for stable val; smaller if we want more train. |
| `eval_every` | **250** | Each eval over the full val loader took ~250-260 s in the prior run; eval-every-100 spent ~25% of wall-clock on val. 250 reduces that to ~10%. |
| `save_every` | **500** | Six checkpoints over a 3000-step run — enough to cherry-pick but not so frequent that disk thrashes. |
| `log_every` | **10** | Default. CE/MSE log lines are tiny. |
| `max_val_items` | **1000** | Caps val passes at ~1000/4 = 250 batches × ~0.2 s = ~50 s per eval (5× faster than the 5000-item full pass). With episode-level split this still covers most held-out episodes. |
| `seed` | **0** | Reproducibility. Bump per re-run if seeding ablations. |
| α | **197.44** | From the `droid_100ep` extraction stats (P75 of ‖h‖₂). Override the legacy default 196.15 (which was from a smoke run) explicitly via `--alpha 197.44`. |

### 3.7 Wall-clock estimate (1× H100 PCIe, 80 GB)

Empirical baseline: 2000 steps × batch 4 × AR-10-layers + 19 evals (every 100)
ran in **6754 s (~1h53m)**. Decomposition:
- Train: ~2000 × 3.4 s/step ≈ 6800 s? Actually evaluating shows ~3.0-3.5 s/step
  net of eval; eval pauses cost ~250 s × 19 ≈ 4750 s. Net training ≈ 2000 s.
- Eval-heavy: ~70% of wall-clock was eval at 100-step cadence.

For the recommended config (3000 steps, AR 16 layers, eval every 250,
max_val_items=1000):
- Train: 3000 × ~3.8 s/step (≈ +12% from extra AR layers) ≈ **11 400 s (3.2 h)**.
- Eval: 12 evals × ~50 s ≈ **600 s (10 min)**.
- **Total ≈ 3.4 h.**

Plan for ~3.5 h on a single H100; budget 4 h to be safe.

---

## 4. Recipe summary table

| Group | Knob | Value |
|---|---|---|
| Optim | learning_rate | 1e-4 |
| Optim | weight_decay | 0.0 |
| Optim | warmup_steps | 200 |
| Optim | total_steps | 3000 |
| Optim | grad_clip | 1.0 |
| Optim | schedule | cosine (linear-warmup → cosine-decay-to-0) |
| Batch | batch_size | 4 |
| Batch | grad_accum_steps | 1 |
| Batch | effective_batch | 4 |
| LoRA  | lora_rank | 32 |
| LoRA  | lora_alpha | 64 |
| LoRA  | lora_dropout | 0.05 |
| LoRA  | lora_targets | q,k,v,o,gate,up,down |
| AR    | truncate_to_n_layers | **16** *(was 10)* |
| Loss  | av_weight | 1.0 |
| Loss  | ar_weight | 1.0 |
| Loss  | ar_contrastive_weight | 0.0 |
| α     | alpha | **197.44** *(P75 ‖h‖, droid_100ep; pass `--stats-json` to read it directly)* |
| Data  | split_by | episode |
| Data  | held_out_fraction | 0.05 |
| Data  | max_val_items | 1000 |
| Sys   | dtype | bfloat16 |
| Sys   | gradient_checkpointing | True |
| Sys   | device | cuda |
| Sys   | seed | 0 |
| Eval  | eval_every | 250 |
| Eval  | save_every | 500 |
| Eval  | log_every | 10 |

---

## 5. Paste-ready first-real-run command

```bash
cd /home/ubuntu/nla-groot
PYTHONPATH=src python scripts/training/run_sft.py \
  --activations-root data/activations/droid_100ep \
  --labels-jsonl     data/labels/droid_100ep/labels.jsonl \
  --output-dir       data/sft/droid_100ep_v1 \
  --base-model       Qwen/Qwen3-4B-Instruct-2507 \
  --stats-json       data/activations/droid_100ep/stats.json \
  --ar-layers        16 \
  --lora-rank        32 \
  --dtype            bfloat16 \
  --batch-size       4 \
  --grad-accum-steps 1 \
  --learning-rate    1e-4 \
  --warmup-steps     200 \
  --total-steps      3000 \
  --av-weight        1.0 \
  --ar-weight        1.0 \
  --ar-contrastive-weight 0.0 \
  --split-by         episode \
  --held-out-fraction 0.05 \
  --max-val-items    1000 \
  --eval-every       250 \
  --save-every       500 \
  --log-every        10 \
  --seed             0 \
  --device           cuda \
  --log-level        INFO \
  2>&1 | tee logs/sft_droid_100ep_v1.log
```

(Optional: prepend `nohup` and append `&` to detach. Memory-safety wrapper:
`CUDA_VISIBLE_DEVICES=0` if other jobs share the box.)

Audit-fix knobs added in May 2026 (all optional, all default-off so the
recipe above is paper-faithful by default):

- `--stats-json data/activations/droid_100ep/stats.json` — read α from the
  Phase-1 dump (`p75_norm`); overrides `--alpha` for both AV and AR.
- `--balance-position-mix` — draw training rows with a `WeightedRandomSampler`
  so per-batch frequencies approximate `POSITION_MIX` (40/40/20). Use when
  the labels file is skewed (today: ~75% `image_patch`).
- `--min-bullets N` — drop labels whose description has fewer than `N`
  bullet (`-`-prefixed) lines. Use to cull degenerate captions.
- `--eval-closed-loop --closed-loop-temps 0.0 [0.7] [--closed-loop-max-batches 64]`
  — log `h → AV.generate → AR → ĥ` stratified FVE/cosine alongside the
  default teacher-forced eval. Slow; cap batches on large val sets.
- `--ar-clip-target-scaled 5.0` — clamp the α-scaled AR target inside
  `forward_sft` to tame heavy tails; inference path is unaffected.

After it finishes, the deliverables to look at first:

- `data/sft/droid_100ep_v1/metrics.jsonl` — final val FVE / MSE / cosine /
  CE per `position_type`. We expect roughly: FVE ≥ 0.50, cosine ≥ 0.74,
  CE ≤ 1.60. Anything materially worse means AR-16 hurt (then revert to 10);
  anything materially better is a green light into GRPO.
- `data/sft/droid_100ep_v1/{av,ar}/` — LoRA adapters + heads, ready for
  GRPO bring-up.
- `data/sft/droid_100ep_v1/log/` — TensorBoard scalars; eyeball
  train CE for the noise envelope before deciding on grad_accum=2 in v2.

---

## 6. Deviations from the paper (flag list)

1. **Effective batch 4 vs paper 256.** ~64× smaller. Mitigations: episode
   shuffling, longer warmup, slightly higher LR (already baked in), and
   not using InfoNCE in v1 since 4-way contrast is too small to be useful.
2. **Dataset size: ~95k labels vs paper's 500k+.** We are firmly in the
   regime where memorization is plausible. Episode-level split is the
   primary defense; v2 may re-add the InfoNCE term.
3. **LR 1e-4 vs paper 1e-5.** Justified by LoRA-rank-32 vs full-FT, but a
   future ablation should sanity-check 5e-5 and 3e-5 once dataset is
   bigger.
4. **AR base ≠ activation source model.** Paper assumes AR and activations
   come from the same M; ours doesn't (AR is plain Qwen3-4B-Instruct,
   activations are from GR00T's Qwen3-VL truncated to 16 layers). Depth
   matching at 16 is a heuristic; weight-match would require AR on
   Qwen3-VL with the same 16-layer truncation, which is doable but adds
   moving parts and isn't worth deferring v1.
5. **No InfoNCE in v1**, despite the codebase supporting it. Reasoning in
   §3.5; revisit if val MSE flatlines while train MSE keeps falling.
6. **Single GPU, no DDP.** Paper is silent on parallelism but at batch 256
   they obviously trained on >1 device. Plan for a multi-GPU rerun once we
   have a verified single-GPU baseline.
7. **Quality-weighted loss disabled.** Labels don't carry the field yet;
   when we add (model_confidence, validator_score, …) we should compare
   weighted vs unweighted in a v3.

---

## 7. V3 defaults (May 2026 — Workstream D)

Three cheap recipe lifts ship as **default-on** for V3 SFT (and for the
next GRPO run). All three are CLI-overridable so legacy commands remain
reproducible by passing the old values explicitly.

| Knob | V2 default | V3 default | Code path | Rationale |
|---|---|---|---|---|
| `--ar-nce-hard-negative-source` | _flag did not exist_ | `none` (opt-in) | [`src/nla/training/dataset.py` `LabeledPositionDataset` hard-neg miner](../../src/nla/training/dataset.py) + [`src/nla/models/ar.py` `forward_sft(negative_explanations=...)`](../../src/nla/models/ar.py) | Pure InfoNCE with batch=4 only contrasts each anchor against 3 random in-batch captions. Hard-neg mining injects K_neg captions sampled from `same_episode` (different step) or `same_position_type` (different episode), so the contrastive denominator gains visually-similar-but-wrong distractors. Off by default to keep paper-faithful baseline reproducible; pass `--ar-nce-hard-negative-source same_episode` to turn on. |
| `--ar-av-mix-max` | `0.0` | **`0.3`** | [`src/nla/training/sft.py` `_ar_av_mix_p` + the train-loop call](../../src/nla/training/sft.py) | V2 postmortem (`docs/evals/v2_lessons_learned.md`) traced the AR shortcut: AR trained on *gold* captions but evaluated on *AV-generated* text. Mixing ~30% AV-gen into AR’s input post-warmup closes the gap. Pass `--ar-av-mix-max 0` for legacy behavior. |
| `GRPOConfig.rollouts_per_activation` (`--rollouts-per-activation`) | `4` | **`8`** | [`src/nla/training/grpo.py`](../../src/nla/training/grpo.py) | Group-relative advantage normalization is noisy with K=4: each rollout's advantage is std-normalized over only 3 peers, and the rare lucky/unlucky tail dominates. K=8 doubles rollout cost but halves the per-group advantage variance. The live `droid_100ep_v2_grpo_run1` is unaffected (its `config.json` froze K=4); the next GRPO run picks up K=8 unless the user passes `--rollouts-per-activation 4`. |

The test coverage for the first two lives in
[`tests/test_ar_hard_negative_nce.py`](../../tests/test_ar_hard_negative_nce.py)
and
[`tests/test_dataset_hard_negative_mining.py`](../../tests/test_dataset_hard_negative_mining.py);
see also [`tests/test_sft_smoke.py::test_run_sft_logs_ar_mix_and_nce`](../../tests/test_sft_smoke.py)
for the AR-AV mix integration smoke.

---

## 8. v2 / v3 ablation candidates (don't run yet — capture for later)

- Full-FT vs LoRA-32 vs LoRA-64 (all on top of v1's other knobs).
- AR depth sweep: 10 / 16 / 20 / 24 layers.
- LR sweep: 3e-5 / 5e-5 / 1e-4 / 2e-4.
- α robustness: 100 / 197 / 400 (paper says ~10× tolerance).
- Effective batch via grad_accum: 4 / 8 / 16 / 32.
- InfoNCE on/off at batch 8+ (where the contrastive term becomes useful).
- Episode-vs-row split as a memorization probe.

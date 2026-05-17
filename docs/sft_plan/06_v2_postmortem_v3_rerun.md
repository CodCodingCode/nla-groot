# V2 postmortem & V3 overnight rerun

Audience: anyone picking up after **droid_100ep_v2_nce** (or similar). Use this so we do not repeat the same blind spots on a long GPU run.

**Repo context:** root **`README.md`** (layout, `PYTHONPATH=src`, eval tracks); **`docs/NLA_AGENT_KNOWLEDGE.md`** (AV/AR/GRPO mechanics for agents).

---

## 1. What V2 taught us (the important mistakes)

### 1.1 High reconstruction ≠ readable scene truth

Val **FVE / cosine** can look strong (including **closed-loop** `closed_greedy/*`, `closed_t0.7/*`) while captions are **useless for humans**.

The failure mode was **shorthand collapse**: the AV reused a **small set of vague templates** (cluster IDs). AR co-adapted so those strings still inverted **approximately** to the right **activation family**. Reconstruction stayed up; **semantics vs camera** did not.

### 1.2 Teacher-forced AR path hides AV shortcuts

Joint SFT trains:

- **AV** with CE on **gold** captions (injected activation → text).
- **AR** with MSE (± InfoNCE) on **gold** captions → activation.

So AR becomes expert at **label prose**, while **inference** is **AV prose → AR**. If those distributions diverge, you only see it if you **measure** it.

### 1.3 The LLM judge is not optional for “is this model good?”

Script: `scripts/eval/llm_judge_av_captions.py`.

- **Axis B (grounding / specific)** — does the caption describe **this** frame, not a generic scene?
- **Axis C (appropriateness)** — style/tone sanity.

V2 pattern to avoid: **gold ~70–80% B pass**, **av_pred ~0% B pass**. That means: **eval pipeline is fine; AV is not grounded**.

**Do not** ship a “success” story on FVE alone.

### 1.4 InfoNCE helped diversity but is not a full fix

`--ar-contrastive-weight > 0` (cosine InfoNCE in `AR.forward_sft`) mainly **sharpens AR under the captions in the batch** (gold, during SFT). It **does not** by itself force AV to match the **camera**; it also **does not** directly train AV (no grad from NCE into AV).

To attack shortcuts end-to-end you want **closed-loop training** next: **GRPO** and/or **AR loss on `AV(h)`** (scheduled sampling / `ar_co_train_weight` in GRPO) — see §3.

---

## 2. Overnight rerun checklist (SFT “V3 baseline”)

Use the **same activations + labels + split** as V2 when comparing.

### 2.1 Preconditions (same as `00_PLAN.md`)

- `data/activations/droid_100ep/stats.json` → **α** via `--stats-json` (or match stored `av_config.json`).
- `labels.jsonl` + activations **orphan-free** vs index.
- `PYTHONPATH=src`, GPU, HF cache.

### 2.2 Training flags — do not drop these on a long run

| Goal | Flag / setting |
|------|----------------|
| Stratified metrics | default `_evaluate` + `StratifiedFve` |
| Closed-loop val | `--eval-closed-loop --closed-loop-temps 0.0 0.7 --closed-loop-max-batches 64` |
| AR stability | `--ar-clip-target-scaled 5.0` (if you used it in V2) |
| Label quality | `--min-bullets 3` (if culling short labels) |
| Position balance | `--balance-position-mix` if file is skewed vs `POSITION_MIX` |
| Anti-generic AR (batch contrast) | `--ar-contrastive-weight` tuned (V2 used **0.5**); **raise batch size / accum** if you can so negatives are meaningful |
| Val ceiling | `--max-val-items 1000` for speed |

### 2.3 After the run (same night or next morning)

1. **Last row of `metrics.jsonl`** — `fve` vs `closed_greedy/fve` vs `closed_t0.7/fve` (should be in the same ballpark; huge gap → pathologies).
2. **`llm_judge_av_captions.py`** on the **final** `av/` — require **B pass** trending toward **gold**, not ~0%.
3. Optional eyeball: `scripts/eval/dump_av_samples.py` and/or `overlay_av_video.py`.

### 2.4 Implemented vs still optional

**Implemented (train AR on AV text sometimes):** `run_sft.py` exposes **`--ar-av-mix-max`**, **`--ar-av-mix-warmup-frac`**, etc.; `SFTConfig` + `sft.py` log **`p_av`** and **`ar_mix_used`**. Default **`ar_av_mix_max=0`** keeps legacy behavior; set **> 0** to ramp scheduled mixing. **Gradients:** AR gets MSE/NCE on mixed text; **AV** still learns only from **gold CE** on that step (generation is **`no_grad`**).

**Still not in repo (optional):** automated **CI gates** on `metrics.jsonl` thresholds; **per-row** gold/AV mix inside one batch (current code mixes **whole batch** or not); **val `ar_nce`**; any **vision** training loss.

---

## 3. After SFT: GRPO (when you want AV to change, not only AR)

`scripts/training/run_grpo.py`:

- Reward **−‖h − AR(y)‖²** on **sampled** captions `y` from AV — gradient into **AV** via policy gradient.
- **`--ar-co-train-weight > 0`** — AR also trains on sampled rollouts so AR tracks the **live** AV distribution (see `grpo.py` docstring).

**Caveat:** if shortcuts still reconstruct well, pure reconstruction reward can **reinforce** them. Then you need **stronger discriminative signal** (contrastive on rollouts, more rollouts, or future **vision-grounded** terms).

---

## 4. Steerability (how it relates to this project)

### 4.1 NLA-style steering (paper recipe, adapted mentally to VLA)

1. Choose token/layer (here: **GR00T residual** at the **same layer** as extraction).
2. **Baseline explanation** `z₀ = AV(h)`.
3. **Edited explanation** `z₁` (you change words in the natural-language bottleneck).
4. **AR** maps text → activation-shaped vector: `ĥ₀ = AR(z₀)`, `ĥ₁ = AR(z₁)`.
5. **Steering direction** `Δ = ĥ₁ - ĥ₀` (or normalized). **Add** `α Δ` to the model’s residual at that position (scale `α` like the paper).

Interpretation: you are **pushing internal state** toward “what would be consistent with this **edited** story,” without retraining.

### 4.2 “Same sentence for 5 seconds, then a new one” — that is **not** automatic

Activations **`h` change every timestep** (new frame, new state). There is no single English string that stays “true” for 5s unless:

- the **underlying cognition** is actually stable (e.g. same subtask), **or**
- you **choose** to **hold** the last steering vector / last caption **fixed** for a **window** (application policy).

Practical patterns:

| Strategy | Effect |
|----------|--------|
| **Re-run AV every step** | Most faithful to **current** `h`; text may **flicker**. |
| **Re-run every N steps or on threshold** | Recompute when ‖hₜ − hₜ₋₁‖ or phase detector says **phase changed**. |
| **Hold caption/Δ for W ms** | **Stable** overlay or stable steering; can be **stale** if the robot is already doing something else. |

So: **yes**, you often want a **new** sentence when the **task phase** changes; exactly *when* is a **control / segmentation** choice on top of the NLA stack, not something the base SFT run guarantees.

---

## 5. One-line summary

**V2:** recon OK, **grounding failed** → measure **judge B** + **closed-loop** every time.  
**V3 overnight:** same data discipline + **contrastive + closed-loop eval**; enable **`--ar-av-mix-max`** when you want AR to see **AV** text during SFT; add **GRPO + co-train** for **AV** policy gradients beyond CE.
**Steering:** text edit → **AR difference** → **add to residual**; **temporal smoothing** is a **product** decision (when to refresh vs hold).

---

## 6. Next GRPO run with multimodal-judge reward

Once **`droid_100ep_v2_grpo_run1`** finishes (DO NOT launch this in parallel with it), the next GRPO experiment should turn on the optional **multimodal-judge** reward term so the AV is pressured toward **camera-grounded** text rather than only toward whatever string happens to invert through AR.

The reward becomes (with `w = --judge-reward-weight`):

\[
  r = (1 - w) \cdot \mathrm{zscore}(r_\text{recon}) + w \cdot r_\text{judge},
  \qquad r_\text{judge} \in \{-1.5, -0.5, +0.5, +1.5\}
\]

where `r_judge = b_score + c_score` from the same GPT-5.1 judge `scripts/eval/llm_judge_av_captions.py` already uses:

- `b_score = +1` if grounding=`specific`, else `-1`
- `c_score = +0.5` if appropriateness=`appropriate`, else `-0.5`

### Flags (all default-off; `--judge-reward-weight 0` is byte-identical to current code)

| Flag | Recommended starter | Notes |
|------|---------------------|-------|
| `--judge-reward-weight` | `0.3` | Start small; 1.0 = pure judge, 0 = pure recon (current). |
| `--judge-concurrency` | `16` | Max in-flight OpenAI calls **per GRPO step**. |
| `--frames-cache` | `data/labels/droid_100ep/frames_cache` | Same dir the labeler / `llm_judge_av_captions.py` use. |
| `--judge-cache-path` | `data/grpo/judge_cache.jsonl` | Append-only `sha1(source_id:text)` cache; persists across runs / resumes. |
| `--judge-model` | (unset → `gpt-5.1`) | Override `OPENAI_GRADER_MODEL` if you need to A/B graders. |

Run validation requires both `--frames-cache` and `OPENAI_API_KEY` when `--judge-reward-weight > 0`; otherwise the script aborts early.

### Recommended launch command (next GRPO run)

```bash
OPENAI_API_KEY=sk-... PYTHONPATH=src python scripts/training/run_grpo.py \
    --sft-dir          data/sft/droid_100ep_v2_nce \
    --activations-root data/activations/droid_100ep \
    --output-dir       data/grpo/droid_100ep_v2_grpo_run2_judge \
    --batch-size 4 --rollouts-per-activation 4 \
    --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-max-new-tokens 160 \
    --beta 0.02 --learning-rate 3e-6 --warmup-steps 20 --total-steps 250 \
    --eval-every 25 --save-every 50 --grad-clip 1.0 \
    --ar-co-train-weight 0.1 --eval-temperatures 0.0,0.7,1.0 --eval-max-examples 64 \
    --gradient-checkpointing --seed 0 --device cuda \
    --judge-reward-weight 0.3 \
    --judge-concurrency   16 \
    --frames-cache        data/labels/droid_100ep/frames_cache \
    --judge-cache-path    data/grpo/judge_cache.jsonl
```

### Caveats

- **Per-step latency.** Even with `--judge-concurrency 16`, the OpenAI grader adds wallclock to every step that has cache misses; expect ~1–3 s/step overhead during the first epoch and progressively less as the cache fills (the cache is keyed by `sha1(source_id + ":" + rollout_text)` and persists across resumes / runs).
- **DO NOT launch this while `droid_100ep_v2_grpo_run1` is still training** — both runs would compete for the same GPU + filesystem locks under `data/grpo/`. Wait for it to finish, snapshot its metrics, then start the judge-on run from the same SFT checkpoint.
- The judge term blends with **z-scored** reconstruction reward (current `r_recon` is on the order of `-0.005`, judge is in `[-1.5, +1.5]`), so the two terms are on comparable scales after blend.
- If the judge API errors transiently, the rollout's `r_judge` falls back to 0 (neutral) for that step — never propagates a NaN.


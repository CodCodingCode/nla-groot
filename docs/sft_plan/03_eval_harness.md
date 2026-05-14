# 03 — First-run SFT eval harness

> Scope: design for the **first** real SFT run of the GR00T NLA. Be opinionated:
> ship the cheapest evals that let us answer "did the AV/AR pair learn anything
> useful, and is it generalizing?" Defer paper-style breadth until v2.
>
> Reference: Fraser-Taliente et al., *Natural Language Autoencoders Produce
> Unsupervised Explanations of LLM Activations*, Transformer Circuits 2026.

---

## 0. TL;DR — must-have vs nice-to-have

### Must-have for run #1 (gate go/no-go)

| # | Metric | What it answers | Where it lives |
|---|---|---|---|
| 1 | **Closed-loop FVE** (`h → AV → ŷ → AR → ĥ`) stratified by `position_type`, on val | "Did the autoencoder learn anything?" | new `nla.eval.recon.closed_loop_fve` (refactor of `grpo._evaluate_fve`) |
| 2 | **Teacher-forced FVE** (`description → AR → ĥ` against `h`) on val | "Is AR sane / are labels self-consistent?" | already in `sft._evaluate` — **rename keys** |
| 3 | **CE on val** (token-level NLL of gold description given `h`) | "Did the AV head fit at all?" | already in `sft._evaluate` |
| 4 | **Train vs val FVE/CE gap** | Memorization signal (overall + per-episode-stratum) | new — add `train_fve_subsample` to `_evaluate` |
| 5 | **Greedy vs sampled FVE gap** (T ∈ {0.0, 0.7, 1.0}) | Entropy / memorization signal at the generative head | already exists in `grpo._evaluate_fve(temperatures=…)`; **port to SFT** |
| 6 | **Stratified FVE by `position_type`** (`image_patch` / `last_text` / `anchor`) | "Does it work where it matters (image patches)?" | already in `StratifiedFve` |
| 7 | **Generation samples dump** (~32 rollouts, gold + AV greedy + 1 sample, per `position_type`) | Eyeball regression check | new `samples.jsonl` written every `eval_every` |

### Nice-to-have for run #1 (cheap, ship if time permits)

| # | Metric | Notes |
|---|---|---|
| 8 | **Behavioral probe: target object MCQ** | Re-uses gold image + a small object pool from the dataset; ~20 lines of code on top of the existing `grader.py` |
| 9 | **Behavioral probe: gripper open/closed** | Boolean classification from the AV caption; tiny grader prompt or regex on the `gripper:` bullet |
| 10 | **Behavioral probe: next action class (reach / grasp / lift / place)** | 4-way classification from `plan:` / `motion:` bullets |
| 11 | **Confabulation spot-check** (re-use `grader.GPT51Grader` on N=30 AV outputs) | Same pass/fail axes (B grounding, C appropriateness) as labeling QA — drop-in |
| 12 | **Paraphrase-FVE drop** (lite) | Round-trip AV output through a paraphraser, recompute FVE, compare; one number, big interpretability signal |

### Defer (v2+)

- Full Anthropic probe suite (Suffix prediction, CoT hints, Safety sandbagging,
  User modeling, classification proxies). Not all map to GR00T; build only the
  ones that do, and only after the autoencoder works.
- **SAE consistency** — we don't have GR00T SAEs.
- **Recurrence-across-tokens trust score** — needs sequence-level rollouts on
  full trajectories; not in our SFT eval loop.
- **Full shuffle / translate steganography sweep** — confounded by GR00T
  captions being short bullets; defer until we have longer captions.
- **Writing-quality / factuality classifier panels.**
- **End-to-end policy steering causal eval** (`nla.steering`).

---

## 1. Reconstruction metrics

### 1.1 What `sft.py::_evaluate` already produces

Per `src/nla/training/sft.py:194` (`_evaluate`):

```200:218:src/nla/training/sft.py
    av.eval()
    ar.eval()
    ce_sum = 0.0
    ce_n = 0
    fve_acc = StratifiedFve(group_name="position")
    for batch in val_loader:
        acts = batch["activations"].to(device)
        out = av.forward_sft(
            activations=acts,
            position_types=batch["position_type"],
            target_texts=batch["description"],
        )
        ce_sum += float(out.loss.item()) * acts.shape[0]
        ce_n += acts.shape[0]
        pred_scaled = ar(batch["description"], device=device)
        pred_unscaled = pred_scaled.detach().float() * alpha
        fve_acc.update(acts.float(), pred_unscaled, batch["position_type"])
```

Produced keys (after `StratifiedFve.compute()`):

- `ce`
- `fve`, `mse`, `cosine` (over the whole val set)
- `fve/position=<image_patch|last_text|anchor>`, plus `mse/...`, `cosine/...`

### 1.2 Gaps to close before the first real run

The current `_evaluate` measures **AR fed the gold description** — i.e., it's a
**teacher-forced** FVE. That's fine as a sanity check on AR / on labeling
quality, but the headline NLA metric is **closed-loop**: `h → AV → text → AR
→ ĥ`. The two numbers are very different. The teacher-forced one can stay high
even if the AV is talking to itself.

Three concrete changes:

1. **Rename teacher-forced metrics** with a `tf/` prefix (`tf/fve`,
   `tf/mse`, `tf/cosine`, `tf/fve/position=...`) so they don't shadow the
   closed-loop metric we're about to add.
2. **Add closed-loop FVE** as the headline metric. Port `grpo._evaluate_fve`
   into a shared module so both SFT and GRPO call the same code.
3. **Sweep temperatures** (`{0.0, 0.7, 1.0}`) on the closed-loop pass so we get
   the greedy/sampled gap "for free."

Concrete refactor:

- New file: `src/nla/eval/recon.py`
  ```python
  def closed_loop_fve(
      av, ar, val_loader, device, *,
      max_examples: int,
      temperatures: tuple[float, ...] = (0.0, 0.7, 1.0),
      max_new_tokens: int = 160,
      top_p: float = 0.95,
  ) -> dict[str, float]: ...

  def teacher_forced_fve(
      av, ar, val_loader, device, *, alpha: float,
  ) -> dict[str, float]: ...   # CE + FVE on AR(gold)
  ```
- `sft._evaluate` becomes a thin wrapper that calls both and merges keys,
  prefixing teacher-forced output with `tf/`.
- `grpo._evaluate_fve` is replaced by `closed_loop_fve` — single source of
  truth, identical key shape across phases.

### 1.3 Output keys after the refactor

```
val/ce
val/tf/fve, val/tf/mse, val/tf/cosine
val/tf/fve/position=image_patch, ...
val/cl/fve/temp=0.0, val/cl/mse/temp=0.0, val/cl/cosine/temp=0.0
val/cl/fve/temp=0.7, ...
val/cl/fve/temp=1.0, ...
val/cl/fve/position=image_patch/temp=0.0, ...
```

Slash-delimited keys flatten cleanly into both JSONL columns and TB scalars.

---

## 2. Behavioral probes (robotics-flavored)

Anthropic's probes test whether AV captions carry information that downstream
LLM tasks can exploit. We replace those with **caption-conditioned multiple-
choice questions about the robot scene**: a frozen grader gets only the AV
caption (not the image), and answers an MCQ whose ground truth comes from the
same step's metadata. If the AV caption is informative, accuracy beats
random + a label-ablated baseline (see §2.4).

All three probes share infrastructure: a frozen judge LLM (default
`gpt-5.1-mini`, configurable via `OPENAI_PROBE_MODEL`), structured JSON
output, and a single eval-set sampler.

### 2.1 Target-object MCQ

**Question to grader.** "Given this 4-bullet caption of what a robot model is
attending to, which of the following is the most likely manipulation target?"
Choices: 4 distractors drawn from the same dataset's task vocabulary
(`task_text` field on `ExampleRecord`) plus the gold target.

**Data source.** `meta.task_text` from extraction + parsed target object from
the gold caption's `target:` bullet (or, when available, the LeRobot dataset's
`task` field). Cache the parsed targets to `data/eval/probe_target_pool.json`
once.

**Grader.** Re-use `nla.labeling.grader.GPT51Grader`'s message scaffolding;
add a new `probe_target` system prompt that returns `{"answer": "A"|"B"|...}`.

**Baseline.** Same MCQ given **only** the instruction (no caption). If our AV
caption can't beat instruction-alone, we're not learning grounded content
beyond the prompt.

**Acceptance.** AV-caption accuracy > (instruction-only baseline + 0.05) on
val, on at least the `image_patch` slice (where the instruction-only baseline
is weakest). Last-text positions will likely beat instruction-only by epsilon
because the captions paraphrase the instruction.

### 2.2 Gripper-state probe

**Question.** "From the caption alone, is the gripper currently `open`,
`closed`, or `transitioning`?"

**Data source.** Robot state vector at the labeled step (`PositionLabelInput.state`
already plumbs `gripper` as the last entry for this dataset). Threshold the
gripper joint (e.g. `> 0.7` = open, `< 0.3` = closed, else `transitioning`)
to get the ground-truth label per `(example_id, step_index)`.

**Grader.** Lightweight regex first (search the AV caption for
`gripper:` bullet content matching `open|closed|transition`); fall back to
`gpt-5.1-mini` for ambiguous bullets.

**Baseline.** Majority-class accuracy (typically ~50–60% for our dataset
mix) and a "shuffled caption" baseline where we hand the grader a random
other caption from the same val set.

**Acceptance.** AV ≥ majority + 0.10 with confusion-matrix dump per
position type.

### 2.3 Next-action-class probe

**Question.** "Predict the next action class: `reach | grasp | lift | place
| other`."

**Data source.** Heuristic on robot trajectories: end-effector velocity sign +
gripper transitions over the next few steps. Gross-grain enough that we don't
need a separate model. Implement once in `src/nla/eval/action_classes.py`.

**Grader.** GPT-5.1-mini with the AV caption + the action class taxonomy
in the system prompt.

**Baseline.** Frequency-based prior + "instruction-only" caption (same as
§2.1) so we can see whether the probe is reading the *plan* bullet vs just
copying the instruction.

**Acceptance.** AV beats both baselines on `image_patch` positions. (We
genuinely don't expect this to be high on first run — it's a stretch
metric.)

### 2.4 Wiring

```
src/nla/eval/
    __init__.py
    recon.py          # FVE, teacher-forced + closed-loop (§1)
    probes.py         # MCQ runner, prompt templates, baselines
    action_classes.py # heuristic ground-truth labeler
    confab.py         # §3
```

Probes run **off the SFT loop** as a separate post-hoc step
(`scripts/eval/run_probes.py`) on the saved `av/`+`ar/` checkpoints. They
are too slow to put in `_evaluate` (they call OpenAI). Caching is keyed
on `(checkpoint_hash, val_split_hash, probe_id, example_id)` so reruns
are cheap.

---

## 3. Confabulation / robustness checks (light)

Three small additions; only the spot-grader is "must-have" for the first run.

### 3.1 Spot-grader on AV outputs (must-have)

Re-use `nla.labeling.grader.grade_many_async` on a stratified sample of
**generated** AV captions (not the gold labels). Same B/C axes (grounding,
appropriateness) — a quick sanity check that the AV isn't producing
uninterpretable junk.

**Sample size.** 30 per `position_type` (`last_text`, `image_patch`, `anchor`)
= 90 grader calls per checkpoint. Cheap.

**Wiring.** Wrap the existing `GradeInput` builder so it accepts a
`(activation_record, av_caption)` instead of a `(activation_record, label_row)`.
No new prompt needed.

**Reported metrics.** `pass_rate_b_av`, `pass_rate_c_av` per position type
(plus the existing `pass_rate_b_llm`, `pass_rate_c_llm` on the gold labels
for direct comparison).

### 3.2 Paraphrase-FVE drop (nice-to-have)

A light steganography proxy. Procedure:

1. Run AV greedily on val; store captions `y_i`.
2. Send `y_i` through `gpt-5.1-mini` with the prompt "Paraphrase this
   robot-state description in different wording but preserve all factual
   content." Get `y_i'`.
3. Compute closed-loop FVE on `y_i'` against the same `h_i`. Call it
   `cl/fve/paraphrased`.

**Metric.** `paraphrase_fve_drop = cl/fve/temp=0.0  −  cl/fve/paraphrased`.

Large drop (>0.15 absolute FVE) ⇒ AR is keying on surface form / token-level
artifacts of the AV (steg-like behavior), not semantic content. Small drop ⇒
the explanation is mostly carried by content words and we're closer to the
NLA paper's "semantic channel" regime.

**Cost.** ~$0.50 per checkpoint at N=128 val examples. Skip if hot path.

### 3.3 Recurrence-across-tokens (defer)

The paper computes a per-claim trust score by checking whether a claim made
at token *t* recurs across many nearby tokens of the same generation. Our
captions are 4–5 bullets per single token position, so the within-caption
recurrence signal is too weak. Re-introduce in v2 if/when we run AV across
sliding windows of token positions for a single trajectory.

---

## 4. Memorization vs generalization

We already do the right thing structurally (episode-stratified split is the
default; `split_by="episode"`). What's missing is **reporting** the
generalization gap.

### 4.1 Train-FVE vs val-FVE

`sft._evaluate` runs only on `val_loader`. Add a `_subsample_train_loader`
of fixed size (default 256 examples, sharing the seed) and run the same
`closed_loop_fve` + `teacher_forced_fve` on it. Two extra rows per eval
step in `metrics.jsonl`:

```json
{"step": 200, "phase": "train_subsample", "cl/fve/temp=0.0": 0.81, ...}
{"step": 200, "phase": "val",             "cl/fve/temp=0.0": 0.62, ...}
```

The headline scalar to plot is `gap = train_subsample_fve − val_fve`.

### 4.2 Greedy vs sampled FVE gap

This is the cheap entropy proxy from §1: we already get
`cl/fve/temp=0.0` and `cl/fve/temp=1.0`. Derive
`mem_gap_temp = cl/fve/temp=0.0 − cl/fve/temp=1.0`.

A model that has memorized a small set of high-FVE captions will have a
large `mem_gap_temp`; a model that has internalized a generative process
will have a small one. (Rule of thumb from the paper: <0.05 = healthy,
>0.15 = check what got memorized.)

### 4.3 Per-episode breakdown (nice-to-have)

Aggregate val FVE by `episode_index` (already on the batch) and report
mean ± std across episodes. Helps catch the "one episode dominates val"
failure mode.

---

## 5. Logging — `metrics.jsonl` schema + tensorboard

### 5.1 metrics.jsonl

One row per logged event. Keep it **flat** (no nested dicts) so it's
trivial to load with pandas / parquetize later.

**Common fields (every row).**

```
step          int      training step
phase         str      "train" | "val" | "train_subsample" | "final"
elapsed_s     float    seconds since trainer start
```

**Train rows** (frequency `cfg.log_every`, default 5):

```
ce, ar_mse, ar_nce, qw_mean, loss, lr
```

(matches today's schema; no change.)

**Val / train_subsample rows** (frequency `cfg.eval_every`):

```
ce
tf/fve, tf/mse, tf/cosine
tf/fve/position=image_patch, tf/fve/position=last_text, tf/fve/position=anchor
tf/mse/position=...               (ditto)
tf/cosine/position=...
cl/fve/temp=0.0, cl/mse/temp=0.0, cl/cosine/temp=0.0
cl/fve/temp=0.7, ...
cl/fve/temp=1.0, ...
cl/fve/position=image_patch/temp=0.0, ...
mem_gap_temp                       (cl/fve/temp=0.0 − cl/fve/temp=1.0)
gen_gap                            (train_subsample cl/fve − val cl/fve, val rows only)
n_examples                         (so we can sanity-check "did we run the full val pass?")
```

**Final row** (`phase="final"`): same as a val row, plus a snapshot of the
checkpoint hash and the active git SHA.

### 5.2 samples.jsonl (new)

Don't dump generated text to `metrics.jsonl` — it bloats rows and breaks
columnar tools. Instead, sister file `samples.jsonl`:

```
{
  "step": 200,
  "example_id": "traj000017_step000042@p042",
  "position_type": "image_patch",
  "instruction": "pick up the blue cube",
  "gold": "- scene: ...\n- target: ...\n...",
  "av_greedy": "- scene: ...\n- target: ...\n...",
  "av_sample_t07": "- scene: ...\n...",
  "tf_fve": 0.78,
  "cl_fve_t0": 0.61,
  "cl_fve_t07": 0.55
}
```

Sample 8 examples per `position_type` (24 per eval step), deterministic
seed so the same val examples are retraced through training. Easy to grep
and diff across checkpoints; also drives the spot-grader (§3.1).

### 5.3 tensorboard

- `train/*` — every scalar from train rows (already wired).
- `val/*` and `train_subsample/*` — every scalar from those rows. Use the
  same `/`-delimited keys as JSONL; TB collapses them into nested groups
  in the UI.
- `gap/*` — derived gaps (`gap/mem_temp`, `gap/gen`).
- **No text** to TB. Keep the `samples.jsonl` flow above for that —
  TB's text widget is awkward for bullet lists.
- **No images**: defer the spatial-NLA-map visualization (paper Fig. 6
  analog) to v2 / `nla.viz`.

### 5.4 Where to look in the existing code

- Schema is enforced in `_write_jsonl_row` (`src/nla/training/sft.py:188` and
  `src/nla/training/grpo.py:159`).
- Tensorboard wiring lives in the same files (search `SummaryWriter`).
- Samples dumping should be a new helper in `src/nla/eval/samples.py`
  called at the end of each `_evaluate` block, taking `(av, ar, val_loader,
  out_path, step)` and appending one block per step.

---

## 6. Concrete file plan

### New files

```
src/nla/eval/__init__.py
src/nla/eval/recon.py          # closed_loop_fve, teacher_forced_fve
src/nla/eval/samples.py        # samples.jsonl writer
src/nla/eval/probes.py         # behavioral probes (post-hoc)
src/nla/eval/action_classes.py # heuristic next-action labeler (§2.3)
src/nla/eval/confab.py         # spot-grader + paraphrase-FVE
scripts/eval/run_probes.py     # post-hoc CLI on a checkpoint
scripts/eval/run_confab.py     # post-hoc CLI for §3
```

### Existing files to extend

- `src/nla/training/sft.py::_evaluate` — call `closed_loop_fve` +
  `teacher_forced_fve`, dump samples.
- `src/nla/training/grpo.py::_evaluate_fve` — replace with the shared
  `closed_loop_fve`. (No semantic change; just dedupes.)
- `src/nla/training/sft.py::SFTConfig` — add:
  - `eval_temperatures: tuple[float, ...] = (0.0, 0.7, 1.0)`
  - `eval_max_examples: int = 256`
  - `eval_train_subsample: int = 256`
  - `eval_n_samples: int = 8`           (per position type, dumped to samples.jsonl)
  - `eval_max_new_tokens: int = 160`
- `src/nla/__init__.py` — un-stub the docstring's promise of a real
  `nla.eval` submodule.

### Default values for the first run

- `eval_every = 100` (less frequent than today's 50; closed-loop
  generation is ~10× slower than teacher-forced AR forward).
- `eval_max_examples = 256`. Stratified across position types via
  `SampledPositionDataset`-style sampling on the val labels list.
- `eval_temperatures = (0.0, 0.7, 1.0)`.
- `eval_train_subsample = 256`. Same seed every eval so the subsample
  is stable.

### Smoke-test acceptance gate (for the first real run)

After 1k steps on the full label set we want, on val:

- `cl/fve/temp=0.0` > `0.30` overall (vs. `~0` from a random AV/AR pair).
- `cl/fve/temp=0.0/position=image_patch` > `0.20` (this is the slice that
  matters; SAEs have no native readout there).
- `tf/fve` > `cl/fve/temp=0.0` by < 0.20 (small AV gap = AV is matching
  the labeling distribution).
- `mem_gap_temp` < `0.15`.
- `gen_gap` < `0.10`.
- Spot-grader B (grounding) pass rate ≥ 0.80 on AV outputs (the gold
  labels run at ≥ 0.95 in current QA).

Any one of these failing is a fixable diagnosis, not a model-is-broken
signal — we expect to iterate.

---

## 7. Out-of-scope for run #1 (revisit only if must-have suite passes)

- Anthropic-style suffix prediction / CoT hints / sandbagging probes —
  none have direct GR00T analogs that we trust yet.
- SAE consistency comparisons — no GR00T SAEs.
- Cross-token recurrence trust score — captions too short.
- Translate / shuffle steganography sweep — confounded by short bullet
  format; revisit when we move to longer free-form captions.
- Causal steering eval (`nla.steering`) — not even on the same axis as
  reconstruction. That's a separate v2 deliverable.

---

## 8. Summary of code changes ranked by ROI

1. **Refactor FVE eval into `nla.eval.recon`** and call the **closed-loop**
   variant from `sft._evaluate`. This single change is the difference
   between "we have an autoencoder metric" and "we have an AR-quality
   metric." Highest ROI of anything on this list.
2. **Add the train-subsample eval pass** — three extra lines in
   `_evaluate`, gives us memorization signal with no new infra.
3. **Add `samples.jsonl`** — debug / spot-check / spot-grade pipeline runs
   off this file forever after.
4. **Spot-grader on AV outputs** — re-uses 100% of the labeling QA stack;
   ~30 lines of glue.
5. **Behavioral probes** — implement in priority order (target-object →
   gripper → next-action), behind a `--with-probes` flag on a post-hoc
   eval CLI. Don't block the train loop on them.


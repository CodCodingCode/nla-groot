# V4 pipeline runbook

A single-page recipe for the V4 upgrade. Each stage states (a) what it needs,
(b) the exact command, (c) approximate wall time / cost, and (d) the pass gate.
Stages are ordered by dependency: do not run stage N until stage N-1 passes.

The full reasoning behind each stage lives in
[`v4_training_recon_audit.md`](v4_training_recon_audit.md). The audit doc is
the "why"; this file is the "how to run."

## Status (as of this PR)

- Stage 0 (FVE fix): **completed in this PR**. See
  [`src/nla/training/fve.py`](../../src/nla/training/fve.py) and
  [`tests/test_fve.py`](../../tests/test_fve.py).
- Stage 1 (re-mine hard negatives): **completed in this PR**. Output at
  [`data/activations/libero_4suite_combined/hard_negatives_v4.jsonl`](../../data/activations/libero_4suite_combined/hard_negatives_v4.jsonl).
  Audit at
  [`data/activations/libero_4suite_combined/hard_negatives_v4_audit.md`](../../data/activations/libero_4suite_combined/hard_negatives_v4_audit.md).
- Stage 2 (scorecard tightening): **not done — small follow-up code change**.
- Stage 3 (V4 labels): **paused on budget approval** per
  [`sa6_relabel.md`](v4_repair/sa6_relabel.md) line 233.
- Stages 4–7: **gated on stage 3** (need V4 labels) and on GPU / sim access.

## Stage 0 — FVE definition (completed in this PR)

Streaming FVE now matches `fve_per_token` (per-dim batch-mean baseline)
exactly, regardless of chunking. Pre-fix `fve` values (everything written to
`data/sft/libero_4suite_v3/metrics.jsonl`) used a global scalar mean and are
not directly comparable to post-fix values. `mse` and `cosine` are unchanged.

Verify locally:

```bash
cd /home/ubuntu/nla-groot
PYTHONPATH=src .venv/bin/python -m pytest tests/test_fve.py -q
```

## Stage 1 — Re-mine hard negatives (completed in this PR)

What changed vs V3 mining:

- `--per-position-type` so `image_patch` anchors only ever see `image_patch`
  candidates (V3 mining mixed ptypes).
- `--last-text-strategy random_same_ptype` so the trainer stops being shown
  fake-hard `last_text` neighbors that are actually shuffle noise.
- `--jaccard-cap 0.55` to drop near-duplicate captions from the negative set.

Re-run if needed:

```bash
cd /home/ubuntu/nla-groot
PYTHONPATH=src .venv/bin/python scripts/training/mine_hard_negatives.py \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_combined/labels.jsonl \
    --out              data/activations/libero_4suite_combined/hard_negatives_v4.jsonl \
    --per-position-type \
    --top-k 8 \
    --jaccard-cap 0.55 \
    --last-text-strategy random_same_ptype \
    --min-bullet-lines 3 \
    --device cuda --dtype float32
```

Audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval/audit_hard_negatives.py \
    --hard-negatives-jsonl data/activations/libero_4suite_combined/hard_negatives_v4.jsonl \
    --activations-root     data/activations/libero_4suite_combined \
    --labels-jsonl         data/labels/libero_4suite_combined/labels.jsonl \
    --sample-anchors 500 --random-pairs 500 \
    --out-md   data/activations/libero_4suite_combined/hard_negatives_v4_audit.md \
    --out-json data/activations/libero_4suite_combined/hard_negatives_v4_audit.json
```

**Gate:** `image_patch` mined mean cosine **minus** random same-ptype mean
cosine must be **≥ 0.10**. Current value: **+0.21**. The absolute-cosine RED
verdict in the audit script is expected at this layer choice (see audit doc
§7 step 3 caveat).

## Stage 2 — Scorecard tightening (small follow-up code change, not in this PR)

In [`scripts/eval/build_v3_scorecard.py`](../../scripts/eval/build_v3_scorecard.py):

- Move `closed_greedy_cosine` and `retrieval_at_1` from `INFORMATIONAL` to
  `REQUIRED_FOR_PASS_*` (lines 120–139).
- Make a missing judge artifact a build error rather than a silent skip
  (`_read_judge` lines 161–213).
- Add `random_baseline = k / N` to the retrieval rows for visibility.

This is intentionally not included in the V4 audit PR — it changes the
contract of an existing script and deserves its own focused review.

## Stage 3 — V4 labels (paused on budget)

Per [`sa6_relabel.md`](v4_repair/sa6_relabel.md), the queue is built and the
driver is ready. The recommended scope is **Option A full relabel** at
~$57 (current `$40` default cap is too tight). Pre-flight smoke test:

```bash
cd /home/ubuntu/nla-groot
export OPENAI_API_KEY=...
PYTHONPATH=src .venv/bin/python scripts/labeling/run_v4_relabel.py \
    --queue-dir   data/labels/v4_relabel_queue \
    --out-dir     /tmp/sa6_smoke_run \
    --suite       libero_goal \
    --max-rows    10 \
    --concurrency 4
```

If smoke is clean, kick off production (~3 hours, ~$57):

```bash
PYTHONPATH=src .venv/bin/python scripts/labeling/run_v4_relabel.py \
    --queue-dir   data/labels/v4_relabel_queue \
    --out-dir     data/labels/libero_4suite_v4 \
    --concurrency 32 \
    --cost-cap    70
```

Track spend in `data/labels/libero_4suite_v4/_cost_log.jsonl`.

Then merge V3-kept + V4-rewritten rows into a single
`data/labels/libero_4suite_v4_merged/labels.jsonl`. The merge script
belongs to SA7 (not in this repo today) — for V4 you can do it with a
small one-off:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, pathlib
v3 = pathlib.Path("data/labels/libero_4suite_combined/labels.jsonl")
v4_root = pathlib.Path("data/labels/libero_4suite_v4")
out = pathlib.Path("data/labels/libero_4suite_v4_merged/labels.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
# Build the set of canonical position keys that got V4-rewritten.
v4_rows = {}
for f in v4_root.glob("libero_*/labels.jsonl"):
    for line in f.open():
        r = json.loads(line)
        k = (r["meta"]["source_example_id"], r["meta"]["position_index"], r["meta"]["position_type"])
        v4_rows[k] = r
# Emit V4 rows where available, else fall back to V3.
n_v3_kept, n_v4 = 0, 0
with out.open("w") as fout:
    for line in v3.open():
        r = json.loads(line)
        k = (r["meta"]["source_example_id"], r["meta"]["position_index"], r["meta"]["position_type"])
        if k in v4_rows:
            fout.write(json.dumps(v4_rows[k]) + "\n"); n_v4 += 1
        else:
            fout.write(line); n_v3_kept += 1
print(f"merged labels.jsonl: v4={n_v4}, v3_kept={n_v3_kept}, total={n_v4+n_v3_kept}")
PY
```

After merge, re-mine hard negatives **against the merged label set** so the
caption-Jaccard cap uses V4 captions:

```bash
PYTHONPATH=src .venv/bin/python scripts/training/mine_hard_negatives.py \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_merged/labels.jsonl \
    --out              data/activations/libero_4suite_combined/hard_negatives_v4_merged.jsonl \
    --per-position-type --top-k 8 --jaccard-cap 0.55 \
    --last-text-strategy random_same_ptype --min-bullet-lines 3 \
    --device cuda --dtype float32
```

## Stage 4 — V4 SFT (~14 GPU-hours)

A direct A/B vs V3 SFT, fixed seed, with the two changes that matter most:
`--grad-accum-steps 4` to 4× the effective NCE batch, and the new
hard-negative index.

```bash
cd /home/ubuntu/nla-groot
PYTHONPATH=src .venv/bin/python scripts/training/run_sft.py \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_merged/labels.jsonl \
    --stats-json       data/activations/libero_4suite_combined/stats.json \
    --output-dir       data/sft/libero_4suite_v4 \
    --base-model       Qwen/Qwen3-4B-Instruct-2507 \
    --ar-layers        16 \
    --lora-rank        32 \
    --lora-alpha       64 \
    --lora-dropout     0.05 \
    --dtype            bfloat16 \
    --batch-size       4 \
    --grad-accum-steps 4 \
    --learning-rate    1e-4 \
    --warmup-steps     500 \
    --total-steps      15000 \
    --av-weight        1.0 \
    --ar-weight        1.0 \
    --ar-contrastive-weight 0.5 \
    --ar-nce-hard-negative-source     topk_cosine \
    --ar-nce-hard-negatives-per-anchor 4 \
    --ar-nce-hard-negative-index-path data/activations/libero_4suite_combined/hard_negatives_v4_merged.jsonl \
    --ar-av-mix-max         0.3 \
    --ar-av-mix-warmup-frac 0.3 \
    --balance-position-mix \
    --min-bullets 3 \
    --split-by    episode \
    --held-out-fraction 0.05 \
    --max-val-items 1000 \
    --eval-every  500 \
    --save-every  2500 \
    --eval-closed-loop --closed-loop-temps 0.0 0.7 --closed-loop-max-batches 64 \
    --seed 0
```

In a side terminal, tail health checks:

```bash
watch -n 300 'PYTHONPATH=src .venv/bin/python scripts/ci/check_sft_metrics.py \
    data/sft/libero_4suite_v4/metrics.jsonl \
    --require-closed-loop --max-tf-closed-fve-gap 0.10'
```

## Stage 5 — V4 evaluation (mandatory before any "ship V4" claim)

```bash
cd /home/ubuntu/nla-groot

# 5a. closed-loop retrieval (~5 min)
PYTHONPATH=src .venv/bin/python scripts/eval/closed_loop_retrieval.py \
    --ckpt-dir         data/sft/libero_4suite_v4 \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_merged/labels.jsonl \
    --n-samples 256 --batch-size 8 --seed 0

# 5b. multimodal judge — ~$5 OpenAI spend, ~10 min wall (concurrency 8)
export OPENAI_API_KEY=...
PYTHONPATH=src .venv/bin/python scripts/eval/llm_judge_av_captions.py \
    --ckpt-dir         data/sft/libero_4suite_v4 \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_merged/labels.jsonl \
    --frames-cache     data/labels/libero_4suite_combined/frames_cache \
    --video-keys       image wrist_image \
    --per-position     24 \
    --concurrency      8 \
    --out-jsonl        data/sft/libero_4suite_v4/llm_judge.jsonl

# 5c. qualitative samples for human spot-check
PYTHONPATH=src .venv/bin/python scripts/eval/dump_av_samples.py \
    --ckpt-dir         data/sft/libero_4suite_v4 \
    --activations-root data/activations/libero_4suite_combined \
    --labels-jsonl     data/labels/libero_4suite_v4_merged/labels.jsonl \
    --per-position     8 --temperatures 0.0 0.7 \
    --out-jsonl        data/sft/libero_4suite_v4/av_samples.jsonl

# 5d. scorecard (assumes Stage 2 has landed; otherwise still run V3-style scorecard)
PYTHONPATH=src .venv/bin/python scripts/eval/build_v3_scorecard.py \
    --ckpt-dir data/sft/libero_4suite_v4
```

**Pass targets (calibrated to V3 baseline):**

| metric                                | V3       | V4 target |
|---------------------------------------|---------:|----------:|
| teacher-forced AR cosine (overall)    | 0.364    | > 0.50    |
| `image_patch` AR cosine               | 0.327    | > 0.45    |
| closed-greedy cosine                  | 0.388    | > 0.50    |
| retrieval@1 (overall, N=178)          | 0.039    | > 0.20    |
| retrieval margin                      | 0.124    | > 0.20    |
| judge axis B (`av_pred`)              | 0.4375   | > 0.55    |
| judge anti-template (`av_pred`)       | 0.25     | > 0.50    |
| judge appropriateness (`av_pred`)     | 0.969    | ≥ 0.90    |

If 5+ of the 8 targets are met, V4 is shippable.

## Stage 6 — Steerability validation (gated on Stage 5 PASS)

Two-terminal flow per [`docs/evals/sim_steer_rollout.md`](../evals/sim_steer_rollout.md).

**Offline single-step probe first** (no sim, ~minutes):

```bash
cd /home/ubuntu/nla-groot
PYTHONPATH=src .venv/bin/python scripts/eval/nla_steer_groot_action.py \
    --model-path     checkpoints/GR00T-N1.7-LIBERO/libero_goal \
    --dataset-path   third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \
    --embodiment-tag LIBERO_PANDA \
    --ar-dir         data/sft/libero_4suite_v4/ar \
    --traj-id 0 --step 50 \
    --placement image_patch --blend 1.0 \
    --text-file prompts/steer_pickup_bottle.txt \
    --out-json data/eval/v4_steer_offline_pickup_bottle.json
```

Sweep `placement ∈ {image_patch, anchor, image_patch_all}` ×
`blend ∈ {0.25, 0.5, 1.0}`. Acceptance: non-trivial `|Δaction|` vs baseline
AND meaningful change between prompt A and prompt B.

**Closed-loop sim (two terminals):**

Terminal A — steered server:

```bash
cd /home/ubuntu/nla-groot
source third_party/Isaac-GR00T/.venv/bin/activate
export HF_TOKEN=...
PYTHONPATH=src python scripts/eval/run_gr00t_server_nla_steer.py \
    --model-path        checkpoints/GR00T-N1.7-LIBERO/libero_goal \
    --embodiment-tag    LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --ar-dir            data/sft/libero_4suite_v4/ar \
    --steer-text-file   prompts/steer_pickup_bottle.txt \
    --placement         image_patch \
    --blend             1.0 \
    --port 5555
```

Terminal B — LIBERO rollout client (separate `libero_uv` venv):

```bash
third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
    third_party/Isaac-GR00T/gr00t/eval/rollout_policy.py \
    --n-episodes 20 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/put_the_bowl_on_the_plate \
    --n-action-steps 8 \
    --n-envs 5
```

Baseline = same client, server with `--steer-off`. Per-condition harness:

```bash
PYTHONPATH=src python scripts/eval/steerability_eval.py \
    --config scripts/eval/steerability_v4.yaml
```

(Author `steerability_v4.yaml` as a clone of
[`steerability_v1_vs_v3.yaml`](../../scripts/eval/steerability_v1_vs_v3.yaml)
with `ar_dirs: [data/sft/libero_4suite_v4/ar]`.)

Acceptance: at least one prompt where success rate or trajectory class
differs significantly between the steered arm and baseline.

## Stage 7 — GRPO (optional, only if SFT plateaus)

Reserved for post-V4 SFT. Reconstruction-only GRPO can reinforce template
collapse per [`docs/evals/v2_lessons_learned.md`](../evals/v2_lessons_learned.md)
lines 81–88, so always include the judge reward path:

```bash
cd /home/ubuntu/nla-groot
export OPENAI_API_KEY=...
PYTHONPATH=src .venv/bin/python scripts/training/run_grpo.py \
    --sft-dir          data/sft/libero_4suite_v4 \
    --activations-root data/activations/libero_4suite_combined \
    --output-dir       data/grpo/libero_4suite_v4 \
    --batch-size 4 \
    --rollouts-per-activation 8 \
    --beta 0.02 \
    --total-steps 1000 \
    --eval-every 100 --save-every 250 \
    --eval-max-examples 128 --eval-temperatures 0.0,0.7 \
    --ar-co-train-weight 0.1 \
    --judge-reward-weight 0.3 \
    --frames-cache     data/labels/libero_4suite_combined/frames_cache \
    --judge-video-keys image wrist_image
```

Wall cost ~5–30× per-step SFT, plus OpenAI judge spend; budget accordingly.

---

## Quick "what changed in this PR" summary

| File | Change |
|------|--------|
| [`src/nla/training/fve.py`](../../src/nla/training/fve.py) | `_StreamingFve` now uses per-dim batch-mean baseline (matches docstring + `fve_per_token`); fp64 accumulators; lazy hidden-dim allocation; rejects non-2D / shape / dim-mismatch inputs |
| [`tests/test_fve.py`](../../tests/test_fve.py) | New — 15 tests pinning streaming-vs-batch invariance under chunking and strata |
| [`docs/sft_plan/03_eval_harness.md`](03_eval_harness.md) | Note that FVE definition changed 2026-05; pre-fix runs are not comparable to post-fix runs |
| [`docs/sft_plan/v4_training_recon_audit.md`](v4_training_recon_audit.md) | New — full V4 audit and upgrade plan |
| [`docs/sft_plan/v4_pipeline_runbook.md`](v4_pipeline_runbook.md) | New — this file |
| [`data/activations/libero_4suite_combined/hard_negatives_v4.jsonl`](../../data/activations/libero_4suite_combined/hard_negatives_v4.jsonl) | New — V4 hard negatives (101,580 anchors, K=8, per-ptype + Jaccard 0.55 + `last_text` random_same_ptype) |
| [`data/activations/libero_4suite_combined/hard_negatives_v4_audit.{md,json}`](../../data/activations/libero_4suite_combined/hard_negatives_v4_audit.md) | New — audit of the above |

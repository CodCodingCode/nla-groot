# NLA / nla-groot — agent knowledge base

Concise reference distilled from project code and the Anthropic NLA paper (Transformer Circuits, 2026). Use this so other agents don’t re-derive basics.

> **Project reality (post–droid_100ep V2):** Val **FVE** (teacher-forced or **closed-loop**) can look **good** while **`llm_judge_av_captions.py` axis B** tanks — **template collapse**. Narrative + GRPO cookbook: **`docs/evals/v2_lessons_learned.md`**. Operational recipe: **`docs/sft_plan/07_sft_recipe_dataset_agnostic.md`**. Next work queue: **`docs/sft_plan/SFT_V5_NEXT.md`**. Older bullets below remain true for **mechanics**.

---

## Repo & doc map (this codebase)

- **Root overview:** `README.md` — layout, quick start, V2 pointers.
- **SFT runbook:** `docs/sft_plan/00_PLAN.md` (checklist); **`07_sft_recipe_dataset_agnostic.md`** (recipe); **`SFT_V5_NEXT.md`** (V5 roadmap).
- **V2 detail / GRPO A/B:** **`docs/evals/v2_lessons_learned.md`**.
- **Library code:** `src/nla/` (`models`, `training`, `extraction`, `labeling`, `steering`, …).
- **Entrypoints:** `scripts/training/run_sft.py`, `run_grpo.py`; `scripts/eval/*.py`.
- **Artifacts:** `data/`, `runs/`, `logs/` are typically **gitignored**; paths in docs assume NFS/local mirrors.

---

## What we’re building

- **Activation verbalizer (AV):** `h` (vector at one layer/token) → **text**.
- **Activation reconstructor (AR):** **text** → `ĥ` (vector).
- **Warm-start (SFT):** supervised `(h, description)` from `labels.jsonl`.
- **RL (GRPO):** improve AV using reconstruction reward via **frozen** AR by default; optional AR co-training.

**Direction:** Data flow is **`h → AV → text → AR → ĥ`**. Never “AV generates `h`.”

---

## Alpha (α)

- **What:** Fixed **scalar** — **not** learned during SFT/GRPO in the default recipe.
- **Role:**
  - **AV injection:** project `h` → L2-normalize direction → **multiply by α** so injected norm matches a band the LM tolerates.
  - **AR / rewards:** compare **`h/α`** and **`ĥ`** (scaled space) — nicer numerics than raw huge norms.
- **Does α filter examples or “detail”?** **No.** Same α for all rows; it **does not** drop activations. It’s **volume / units calibration**, not filtering.
- **How chosen:** **Recommended:** compute from extraction dump — **α ≈ P75 of ‖h‖₂** over many valid positions (`nla.extraction.stats` → e.g. `stats.json`). That’s “typical-on-the-large-side” magnitude, **not** perfection or optimality. Paper/repo note ~**one order of magnitude** slack around that value is usually OK.
- **CLI:** `--alpha` can override the default number from stats.

---

## Labeling → `labels.jsonl`

- **Not live sim:** Frames come from **LeRobot MP4s** on disk (`EpisodeFrameLoader`, PyAV); cached JPEGs for API calls.
- **OpenAI multimodal API:** System + user **text** + **full camera JPEGs** per timestep (base64 / image_url). Model field often `gpt-5-mini` / configurable via `OPENAI_LABELING_MODEL`.
- **GPT never sees raw `h` floats** in default labeling; bullets describe scene/task given **pixels + instruction + token metadata**.
- **`description`:** Bullet text from **`build_position_prompt`** (categories: scene, target, spatial, …); **not** copying the instruction verbatim.
- **`meta`:** Join keys — `source_example_id`, `position_index`, `position_type`, etc.

**Caveat:** Teacher can mention scene facts **weakly present in `h`** — classic warm-start confound.

---

## SFT (joint warm-start)

- **Both AV and AR** updated **in the same steps** on **same batches** (not “week of AV only”).
- **AV loss:** **cross-entropy** on `description` tokens with **`h` injected** (teacher forcing).
- **AR loss:** **MSE** in **`h/α`** space from `description` → `ĥ` (optional **InfoNCE** — **AR only**, not AV).
- **Not GRPO** — pure supervised gradients until RL phase.

**Distribution gap:** AR default trains on **gold** text; at inference **AV** feeds AR — use **`--ar-av-mix-max`** and/or **GRPO** / grounding-aware losses. See **`07_sft_recipe_dataset_agnostic.md`** and **`docs/evals/v2_lessons_learned.md`**.

---

## GRPO (RL phase)

- **Data:** `SampledPositionDataset` over **extraction only** — **no `labels.jsonl`**.
- **Reward:** Reconstruction — **high** when **`ĥ = AR(y)`** close to **`h`** in scaled space (negative mean squared error over dims).
- **Per batch:** `B` activations, **`K` rollouts** per `h`; rewards compared **within each group** → **advantages**; policy gradient on AV **log-probs** + **KL** to frozen **reference AV** (SFT copy).
- **Default (`ar_co_train_weight=0`):** **AR frozen** — acts as reward model only.
- **Optional:** `ar_co_train_weight > 0` — **AR also gets MSE** on sampled captions (tracks AV’s evolving wording; paper-style simultaneous AR regression).

**Why RL on AV, not AR:** AR is **differentiable** text→`ĥ` with known **`h`** → **MSE + Adam** is direct. AV **samples discrete text**; optimizing expected reconstruction needs **policy gradients** (GRPO), not “same as AR.”

---

## Extraction vs index size

- **`index.jsonl`:** One row per **`(traj_id, step_idx)`** written by extraction — **not** “every frame in the universe.” Count follows **`traj_ids`**, **`steps_per_traj`**, **`step_stride`**.

---

## Cameras / modality

- Count comes from **dataset `meta/modality.json`** (`video_keys`). Example demo: **two** streams — not a GR00T hard cap.
- Labeling attaches **all** `video_keys` by default (unless overridden).
- **`image_patch_meta (k, n)`:** **`k`‑th image token** among **`n`** total in the **fused sequence** — not an explicit “camera id” field; camera follows **model token layout**.

---

## Overlay slowness (`overlay_av_video.py`)

- Per step: **`av.generate`** up to **`max_new_tokens`** — **many LM forwards**, not realtime video speed. Output MP4 **fps** matches dataset metadata; **generation does not.**

---

## Metrics

- **Primary reconstruction:** **FVE / MSE** (and codebase adds **cosine** as auxiliary). NLA paper emphasizes **FVE/MSE**, not cosine as headline.

### What “good” means here (don’t skip)

| Signal | What it catches | Caveat |
|--------|------------------|--------|
| **Teacher-forced** `fve` / `cosine` in `sft._evaluate` | AR can invert **gold** captions | **Not** inference path |
| **Closed-loop** `closed_greedy/*`, `closed_t*/*` (`--eval-closed-loop`) | `h → AV.generate → AR → ĥ` | Still **not** “matches camera”; shortcuts can score |
| **`llm_judge_av_captions.py`** axis **B** (specific) / **C** (appropriate) | Caption vs **cached frames** | Needs `OPENAI_API_KEY`; **this** is the human-facing bar |
| **GRPO** `_evaluate_fve` (multi-temp) | Policy + collapse | After RL |

**Rule:** Never claim “interpretability works” from **FVE alone**. Compare **`av_pred`** judge B to **`gold`** judge B on the same rows.

### Scripts (quick index)

| Script | Role |
|--------|------|
| `scripts/eval/llm_judge_av_captions.py` | Gold + AV vs frames (B/C) |
| `scripts/eval/dump_av_samples.py` | Gold vs greedy/sampled + per-row TF vs closed-loop |
| `scripts/eval/build_eval_cases.py` → `run_interp_panel.py` | Counterfactual **h** edits (different question than B) |
| `scripts/eval/overlay_av_video.py` | MP4 with AV text overlay (demo) |
| `scripts/eval/nla_steer_overlay_video.py` | MP4: frames + **baseline vs backbone-steer** action deltas (needs GR00T + Cosmos HF access) |
| `scripts/eval/nla_steer_groot_action.py` | Prints numeric **baseline vs steer** `get_action` diff (same deps) |
| `scripts/eval/nla_steer_quant_probe.py` | **Math probe**: one timestep, **two** AR prompts → Δactions + numeric previews + JSON (**clear steerability stats**) |
| `scripts/eval/nla_steer_ar_smoke.py` | AR→ĥ + hook on toy backbone only (**no** GR00T) |
| `scripts/eval/run_gr00t_server_nla_steer.py` | Launch a GR00T policy server with `NlaSteerGr00tPolicy` so any Isaac sim client (LIBERO/SimplerEnv) hits a steered backbone — runbook: `docs/evals/sim_steer_rollout.md`; closed-loop LIBERO Goal pilot results: `docs/evals/libero_goal_pilot_results.md` |
| `scripts/eval/steerability_eval.py` | **Steerability eval interface**: config-driven harness comparing N `(AR dir, steer text)` rows on shared LIBERO env+seeds; optional **multi-hold-out** AV grading via ``av_eval.datasets``; optional ``patch_scorecard`` merges sim + IND-judge outcomes into ``v3_scorecard.json``. See `scripts/eval/steerability_v1.yaml`, `scripts/eval/steerability_v1_vs_v3.yaml` (v1 vs `libero_4suite_v3` head-to-head, 3 seeds). Artifacts typically under `data/eval/steerability_v1/` or `data/eval/steerability_v1_vs_v3/` |

### Local CI-style contract (no GitHub Action required)

- **Gate script:** `scripts/ci/check_sft_metrics.py` reads `metrics.jsonl` and fails on:
  - dead NCE (`ar_nce` tail near `ln(B)` when contrastive is on),
  - missing closed-loop metrics (if requested),
  - excessive `fve - closed_greedy/fve` gap (if threshold supplied).
- **Smoke test:** `tests/test_sft_smoke.py` includes a tiny-run check that logs
  `p_av`, `ar_mix_used`, and finite `ar_nce` under `ar_av_mix_*` + contrastive.
- Use this pair to keep “is this implemented?” answers consistent across chats.

---

## External advice (injection / SFT)

- **α** robust ~**10×** around P75-style choice; **SFT cheap** — rerun/sweep reasonable.
- **LoRA** used in repo; compatible with pipeline.

---

## Paths worth knowing

- Activations: `data/activations/<run>/` (+ `index.jsonl`, shards).
- Labels: `data/labels/<run>/labels.jsonl` (often **gitignored**; check NFS paths if symlinked).
- SFT out: `data/sft/<run>/` with `av/`, `ar/`, `config.json`.

---

## Citations

- Paper: Fraser-Taliente et al., *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*, Transformer Circuits, 2026.
- This file is **project operational knowledge**, not a substitute for the paper.

# NLA / nla-groot — agent knowledge base

Concise reference distilled from project code and the Anthropic NLA paper (Transformer Circuits, 2026). Use this so other agents don’t re-derive basics.

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
- **AR loss:** **MSE** in **`h/α`** space from `description` → `ĥ`.
- **Not GRPO** — pure supervised gradients until RL phase.

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

# SFT V5 — what to do next

**Status:** V4 focused on **label / content quality** (hallucinations, scaffold leakage, motor imperatives, position-type blur). V5 is the queue for **optimization**, **layer choice**, **template pressure in training**, and **making quality grading actually bite**.

**Supersedes:** the former `docs/sft_plan/v4_repair/V5_TODO.md` (removed; this file is the single roadmap).

**Reference framework:** A dataset is useful only if **(1) inputs vary with detail**, **(2) labels encode detail**, **(3) optimization punishes vagueness**, plus: specificity, anti-mode diversity, slot-aligned supervision, and quality grading that affects gradients.

| Dimension | V4 status | V5 action |
|-----------|-----------|-----------|
| Specificity-conditioned | ~85% (libero_10 weak) | Item 4 |
| Anti-mode diversity | Shuffled, not killed | Items 1, 5 |
| Slot-aligned supervision | Mostly yes | Item 2 (earlier `last_text` layer) |
| Quality grading that bites | Infra exists, mostly unused | Item 3 |
| Input varies with detail | Not fully | Item 2 |
| Label encodes detail | Strong target/scene, weaker plan | Item 1 |
| Optimization punishes vagueness | Partial | Items 3, 5, 6 |

---

## Priority queue (recommended order)

### 1. Defuse the new `plan:` template collapse (prompt fix)

V4 replaced one dominant template with another (`over the next 3 timesteps` / phase boilerplate). Top phrase document frequency went **up** vs V3.

- **Edit** `_LAST_BULLET_BY_POSITION_TYPE["last_text"]` in `src/nla/labeling/prompts.py` — require **one specific motion + one specific object** without prescriptive temporal connector boilerplate.
- **Extend** forbidden phrases in `scripts/eval/audit_prompt_hardening.py`.
- **Selective re-label** rows with the new boilerplate (budgeted).

**Target:** top-phrase DF in any single bullet ≤ 15%; no phrase > 25% in `plan:`.

### 2. Re-extract `last_text` from an earlier hidden layer (GPU)

Final-layer `last_text` hidden states can be **saturated** (very high cosine between unrelated rows), making InfoNCE structurally weak.

- **Add or use** a layer selector on `scripts/extraction/run_extract.py` (manifest / CLI — verify current wiring before relying on it).
- **Re-run** `scripts/eval/audit_hard_negatives.py` on new shards; pick a layer whose random-pair cosine sits in a “healthy” band (e.g. ~0.5–0.8 per internal audits).
- **Re-mine** hard negatives (`scripts/training/mine_hard_negatives.py`) for that layer.

**Target:** `last_text` mined-vs-random cosine margin ≥ internal healthy threshold (e.g. Δ ≥ 0.12).

### 3. Turn on quality weights (data + small builder script)

`SFTConfig.use_quality_weights` exists; labels often lack per-row weights.

- Grade a **stratified sample** with `scripts/eval/verify_libero_label_quality.py` (or successor).
- **Add** `scripts/training/build_quality_weights.py`: join judge verdicts to `(source_example_id, position_index, position_type)` → emit `quality_weights.jsonl` (e.g. B=specific → 1.0, somewhat_specific → 0.5, non_specific → 0.0 / 0.1); ungraded rows inherit suite/ptype means.
- **Enable** in real configs / CLI for production SFT.

**Target:** AR cosine on held-out *judge-specific* rows measurably above mean — weights must visibly concentrate learning.

### 4. Spatial-style rules for `libero_10` (prompt fix)

Long-horizon suite can mirror pre-fix `libero_spatial` failure modes (instruction-anchored guesses).

- **Author** suite addendum in `src/nla/labeling/prompts.py` (visually verify objects; describe current sub-task not distant goal).
- **Pilot judge** on failure rows; **selective re-label** as needed.

**Target:** `libero_10` judge axis B in line with healthier suites.

### 5. Decide the `language:` bullet contract (prompt / audit)

Either require `language:` on `last_text` with a clear template, or **remove** it from the closed vocabulary and audits so reports match the contract.

**Target:** audits report 100% consistent with the declared contract.

### 6. Anti-template loss term (training)

Complement data-side fixes with an **explicit** AV penalty for dominant phrases.

- **Add** `anti_template_phrases` / `anti_template_weight` to `SFTConfig` and the AV step in `src/nla/training/sft.py`.
- **Source** phrase list from `scripts/eval/audit_diversity.py` / diversity reports.
- **Sweep** weight on a small SFT before large runs.

**Target:** greedy top-phrase DF capped without blowing CE.

### 7. (Optional) FVE magnitude / norm pathology

NCE can ignore magnitude; predicted ‖h‖ may drift while cosine looks fine.

Pick one: **eval-time renorm** before FVE in the eval loop; **small magnitude regularizer** on AR; or **lower NCE weight** so MSE pulls norms. See `docs/sft_plan/v4_repair/sa_scale_audit.md`.

---

## Infra & research-process backlog (from codebase review)

Do **not** block V5 dataset work on these, but track them so papers and ops stay honest.

| Issue | Direction |
|-------|-----------|
| `paper/repro/canonical_commands.sh` | Keep flags in sync with `scripts/training/run_sft.py` (e.g. `--closed-loop-temps`, `--ar-nce-temperature`). |
| SFT val vs GRPO val | Same `seed` ≠ same episodes; document leakage if GRPO / counterfactual mining use full corpora. |
| Scorecard vs narrative | PASS gates emphasize retrieval + judge; closed-loop cosine may be informational — align docs with `build_v3_scorecard.py`. |
| Doc 09 (action-head LoRA) vs shipped **action consistency** loss | Same motivation, different mechanism — pick one story for the paper. |
| `mine_cross_task_pairs.py` stubs | `scene_id` / visibility **TODO**s; do not treat as production-safe until implemented. |
| `action_head_consistency.py` cache | Unbounded baseline cache — fix before large consistency runs. |
| Per-row vs whole-batch AR–AV mix | Documented optional future; current code is whole-batch coin flip. |

---

## Cost / effort (rough)

| Item | API $ | GPU h | Engineering |
|------|-------|------|-------------|
| 1. Plan template | ~$40 | 0 | ½ day |
| 2. Layer re-extract | $0 | 2–4 | ½ day |
| 3. Quality weights | ~$50 | 0 | ½ day |
| 4. libero_10 rules | ~$10 | 0 | ½ day |
| 5. `language:` contract | ~$10 | 0 | 1 h |
| 6. Anti-template loss | $0 | 1–2 | 1 day |
| 7. FVE magnitude patch | $0 | 0 | 30 min |

---

## Stopping rule for V5

V5 succeeds if **both**:

- Multimodal judge **B** ≥ target band overall and per suite (set numerically vs your paper bar).
- SFT **AR cosine** on held-out V5 rows meets the FVE / cosine gate you use in the scorecard.

If judge passes but SFT metrics fail, treat the bottleneck as **optimization / architecture** (not another V4-style relabel-only round).


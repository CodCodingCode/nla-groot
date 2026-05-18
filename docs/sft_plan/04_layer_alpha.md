# 04 — Layer selection and α calibration audit

> **Interpretability warning:** Correct α and layer choice do **not** guarantee **scene-accurate** AV text. See **`docs/evals/v2_lessons_learned.md`** (shorthand collapse vs FVE).

**Reference paper.** Fraser-Taliente et al., *Natural Language Autoencoders Produce
Unsupervised Explanations of LLM Activations*, Transformer Circuits, 2026.

**Stats source.** `data/activations/droid_100ep/stats.json` (n=2,001,414 valid
positions; image_token_fraction=0.912):
P50=185.014, P75=197.445, P90=209.647, P99=233.278, mean=249.523, std=1070.391.

This document audits whether the repo's α and layer-selection setup faithfully
follows the paper's injection recipe and recommends concrete settings for the
first SFT run.

---

## 1. Where activations are hooked

The hook is wired exactly where the paper wants it, but on a stack that GR00T
has already physically truncated.

- `src/nla/layer_spec.py:45` sets `SELECT_LAYER = 16`. The header comment
  (`src/nla/layer_spec.py:18-28`) documents that GR00T physically pops decoder
  layers off the Cosmos-Reason2-2B Qwen3-VL backbone:
  ```22:23:src/nla/layer_spec.py
  while len(self.model.language_model.layers) > select_layer:
      self.model.language_model.layers.pop(-1)
  ```
  After this surgery the backbone is **only 16 layers deep**, and there is
  literally no layer 17+ to hook.
- `src/nla/layer_spec.py:84-90` defines the canonical hook target as
  `backbone_features` (pre-vlln, pre-`vl_self_attention`):
  ```83:90:src/nla/layer_spec.py
  TARGET_BACKBONE_FEATURES = HookTarget(
      name="backbone_features_pre_vlln",
      module_path="backbone.model.language_model.layers",
      pre_vlln=True,
      notes="Output of Qwen3-VL decoder layer (SELECT_LAYER-1), shape [B,T,2048]. "
            "Equivalent to `outputs.hidden_states[-1]` returned by Qwen3Backbone.forward.",
  )
  ```
- `src/nla/extraction/hook.py:103-138` registers a `forward_hook` on the
  `Qwen3Backbone` *wrapper* (not on individual decoder layers). It reads the
  `BatchFeature` returned by the wrapper:
  ```103:106:src/nla/extraction/hook.py
  features = output["backbone_features"]
  attention_mask = output["backbone_attention_mask"]
  image_mask = output["image_mask"]
  ```
  with a hard assertion on `hidden_size == BACKBONE_EMBEDDING_DIM = 2048`
  (`src/nla/extraction/hook.py:110-113`). This guarantees we capture the output
  of the surviving last layer and not, e.g., post-vlln `vl_embeds`.
- `scripts/extraction/run_extract.py:291` attaches the hook to
  `policy.model.backbone` and at line 183 explicitly calls
  `policy.model.backbone(backbone_inputs)`, bypassing the DiT entirely. There is
  one capture per step; `assert captured.batch_size == 1`
  (`scripts/extraction/run_extract.py:310-312`).

**Verdict.** Plumbing is correct: we capture `hidden_states[-1]` of the
truncated 16-layer backbone in fp32, with attention/image masks, exactly once
per forward.

### Is "layer 16" the right call?

The paper recommends roughly 2/3 depth (the "middle-to-late" band) for an
*untruncated* base LM, because that's where features are abstract enough to be
verbalizable but not yet collapsed to next-token logits. Mapping that advice
naively onto our 16-layer stack would point at **layer ~10–11**, not layer 16.

Two reasons we should still hook layer 16:

1. **Layer 16 is the only policy-relevant tap.** GR00T's action head consumes
   exactly the output of the last surviving Qwen3-VL layer (run through
   `vlln` → `vl_self_attention` → DiT cross-attention). Anything earlier is an
   internal representation that the action head never sees. The whole point of
   running NLA on GR00T (versus on a generic LM) is to verbalize *what the
   policy is committing to*, which by construction lives at layer 16.
2. **The paper's depth heuristic is about LM-internal usefulness, not about
   downstream consumers.** When you build NLAs over a base LM there is no
   downstream consumer; you just want a layer where the residual stream is
   semantically rich. Here we have a hard downstream consumer (the DiT), and
   it observes layer 16. Picking layer 11 would optimize the wrong objective:
   we'd verbalize features that may be perfectly interpretable but are a
   detour rather than the policy's actual decision substrate.

The action-head-sees-it-after-`vlln` nuance is also why the codebase exposes
`TARGET_VL_EMBEDS` (`src/nla/layer_spec.py:93-98`) as an ablation target, and
`TARGET_DIT_MID` (`src/nla/layer_spec.py:102-107`) as a motor-stream ablation.
Those are the right *next* sweeps; the *first* run should be layer 16.

The `LAYER_SWEEP = (8, 12, 16)` constant (`src/nla/layer_spec.py:56`) gives us
the intermediate taps if we ever want to test the "middle-to-late" claim
directly. That sweep is for later — it requires re-extraction with the hook
re-pointed at intermediate layers.

---

## 2. α injection in AV (`src/nla/models/av.py`)

The injection follows the paper's recipe in order:

```212:217:src/nla/models/av.py
def _project_activation(self, activation: torch.Tensor, embed_dtype: torch.dtype) -> torch.Tensor:
    """Project, L2-normalize, scale by α. Returns ``(B, H)`` in ``embed_dtype``."""
    proj = self.act_proj(activation.to(self.act_proj.weight.dtype))
    proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    proj = proj * self.cfg.alpha
    return proj.to(embed_dtype)
```

Order: **project → L2-normalize → multiply by α**, identical to the paper. The
output norm is *exactly* `α` (modulo the `1e-6` eps) regardless of the input
activation magnitude. The slot embedding is then overwritten in-place:

```282:296:src/nla/models/av.py
def _embed_with_injection(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    slot_indices: list[int],
    activations: torch.Tensor,
) -> torch.Tensor:
    embed_module = self._embed()
    embeds = embed_module(input_ids)              # (B, T, H)
    proj = self._project_activation(activations, embeds.dtype)
    idx_b = torch.arange(embeds.shape[0], device=embeds.device)
    idx_t = torch.tensor(slot_indices, device=embeds.device, dtype=torch.long)
    embeds = embeds.clone()
    embeds[idx_b, idx_t] = proj
    return embeds
```

### Slot token mechanism

Sane and idempotent. By default the constructor adds a fresh special token
`<|act_slot|>` and resizes the LM's embedding table:

```121:140:src/nla/models/av.py
def ensure_slot_token(tokenizer, base_model, slot_str: str) -> int:
    """Add ``slot_str`` to the tokenizer (if missing) and resize embeddings.
    ...
    """
    existing = tokenizer.encode(slot_str, add_special_tokens=False)
    if isinstance(existing, list) and len(existing) == 1:
        return int(existing[0])
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [slot_str]}
    )
    if n_added > 0 and base_model is not None:
        base_model.resize_token_embeddings(len(tokenizer))
    ids = tokenizer.encode(slot_str, add_special_tokens=False)
    if not (isinstance(ids, list) and len(ids) == 1):
        raise RuntimeError(...)
    return int(ids[0])
```

Per-row slot index resolution then locates that single id inside the rendered
prompt (`src/nla/models/av.py:241-254`), and the row-and-column index
(`idx_b`, `idx_t`) overwrite is an exact, single-position replacement. There is
no risk of collision with text content because the special token is registered
explicitly.

### Deviation from the paper: learnable `act_proj`

The paper describes a *fixed* (often weight-tied or random orthogonal)
projection, then α scaling. The repo uses a **learnable `nn.Linear` with bias**:

```189:194:src/nla/models/av.py
self.act_proj = nn.Linear(cfg.activation_dim, hidden_size, bias=True)
if cfg.activation_dim == hidden_size:
    nn.init.eye_(self.act_proj.weight)
else:
    nn.init.xavier_uniform_(self.act_proj.weight)
nn.init.zeros_(self.act_proj.bias)
```

GR00T's backbone is 2048-d and Qwen3-4B-Instruct-2507's hidden size is 2560,
so we are in the `xavier_uniform_` branch (no identity init). This is a
conscious deviation, justified in the file's docstring as "the only path by
which activation gradients reach the LM" (`src/nla/models/av.py:39-40, 199-200`).

Functionally the behavior is still paper-faithful: regardless of what
`act_proj` does, the L2-normalize step strips the magnitude before α is
applied, so the *scale* of the injected vector is always α. What `act_proj`
learns is the **direction** in LM-embedding space.

**Recommendation.** Keep the learnable affine for now. It's an order-of-mag
freer than the paper's fixed projection but the SFT signal needs *some* path
from `h` to LM embedding space; without it we'd have to weight-tie or
orthogonalize, which adds engineering risk for the first run. Track
`act_proj.weight.norm()` in the training metrics to detect runaway scaling
(unlikely given the post-projection L2-normalize, but cheap to monitor).

---

## 3. α in AR (`src/nla/models/ar.py`)

The AR is paper-faithful: it predicts and is trained in `h/α` space, and
`predict()` re-multiplies by α on the way out.

- Target is α-scaled before MSE (`src/nla/models/ar.py:190-194`):
  ```190:194:src/nla/models/ar.py
  pred_scaled = self.forward(explanations, device=target_activations.device)
  target_scaled = (target_activations / self.cfg.alpha).to(pred_scaled.dtype)
  mse = nn.functional.mse_loss(pred_scaled, target_scaled)
  if not return_nce:
      return mse, pred_scaled
  ```
- `forward(...)` returns the scaled prediction directly (no α multiplication),
  so chained losses keep working in scaled space
  (`src/nla/models/ar.py:140-158`).
- `predict(..., unscale=True)` is the only place where α is multiplied back
  in:
  ```160:163:src/nla/models/ar.py
  def predict(self, explanations: list[str], *, unscale: bool = True) -> torch.Tensor:
      with torch.no_grad():
          pred_scaled = self.forward(explanations)
      return pred_scaled * self.cfg.alpha if unscale else pred_scaled
  ```
  Default `unscale=True` is the right call: external consumers of the
  reconstruction (steering, FVE diagnostics that compare to the original `h`)
  should always see un-α'd vectors.

The InfoNCE auxiliary in `forward_sft(..., return_nce=True)` lives in scaled
space too (`src/nla/models/ar.py:196-203`), which is the right convention.

**Verdict.** AR α handling is correct.

---

## 4. α propagation through GRPO (`src/nla/training/grpo.py`)

The GRPO reward and AR co-training term both live in α-scaled space, matching
the paper:

```253:267:src/nla/training/grpo.py
if ar_train_weight > 0.0:
    ar.train()
    pred_scaled = ar(rollout_texts, device=device)                   # (B*K, H_act)  WITH grad
    target_scaled = (acts_rep / ar.cfg.alpha).to(pred_scaled.dtype)
    rewards = -((pred_scaled.detach() - target_scaled.detach()) ** 2).mean(dim=-1).float()
    ar_mse = ((pred_scaled - target_scaled) ** 2).mean()             # scalar, with grad
else:
    ar.eval()
    with torch.no_grad():
        pred_scaled = ar(rollout_texts, device=device)
        target_scaled = (acts_rep / ar.cfg.alpha).to(pred_scaled.dtype)
        rewards = -((pred_scaled - target_scaled) ** 2).mean(dim=-1).float()
```

Reward = `-‖h/α − AR(y)‖²` (mean over hidden), exactly what the audit asked
for. The eval path also matches: scaled prediction is multiplied by `ar.cfg.alpha`
*after* the MSE-equivalent for FVE (`src/nla/training/grpo.py:387-390`):

```387:390:src/nla/training/grpo.py
pred_scaled = ar(rollout["text"], device=device)
pred_unscaled = pred_scaled.float() * ar.cfg.alpha
fve_acc.update(acts.float(), pred_unscaled, ptypes)
```

FVE is computed in raw `h` space (the right space for variance-explained
reporting), but the loss/reward stay in scaled space. Group-relative advantage
normalization (`src/nla/training/grpo.py:269-275`) is itself scale-invariant,
so α only affects how clipped rewards look in absolute units, not the gradient
direction.

**Verdict.** GRPO α propagation is correct.

The same `alpha`-scaled MSE convention also flows through SFT
(`src/nla/training/sft.py:194-218`), where `_evaluate(...)` un-α's only at the
final FVE step.

---

## 5. Outlier handling (the heavy tail)

The droid_100ep stats are alarmingly heavy-tailed:

| metric | value |
|---|---|
| P50 | 185.0 |
| P75 (= α) | **197.44** |
| P90 | 209.6 |
| P99 | 233.3 |
| mean | 249.5 |
| std | **1070.4** |

A standard deviation of 1070 with a P99 of 233 means a tiny number of positions
have norms in the thousands or tens-of-thousands range. A back-of-envelope: if
P99 is ~1.18·α and `mean − P99 ≈ 16` while `std ≈ 1070`, then ~1% of positions
sit in a long tail extending several hundred α away. Combined with
`image_token_fraction = 0.912`, almost all of those outliers are image-patch
tokens.

### What the code does today

- **AV injection is fully outlier-robust by construction.** The L2-normalize
  step (`src/nla/models/av.py:215`) maps every input to unit length before
  α-scaling, so an `‖h‖ = 50,000` activation injects a slot embedding of norm
  α just like any other. There is no AV-side action item.
- **AR training is *not* robust.** `target_scaled = h / α` directly inherits
  the heavy tail. An outlier with `‖h‖ = 50α ≈ 9870` produces
  `‖target_scaled‖ ≈ 50` (per-vector) and a per-element MSE contribution
  thousands of times larger than a typical position. Plain `mse_loss` is a mean
  over a batch, so a single outlier can dwarf 99 typical examples in one step.
- **GRPO reward inherits the same problem.** `-‖h/α − AR(y)‖²` for outlier `h`
  yields hugely negative rewards; group-relative normalization
  (`src/nla/training/grpo.py:269-275`) helps locally, but if every rollout in a
  group has a huge negative reward, the within-group `std` is also huge and
  advantages collapse. (Std-based normalization actually masks this: variance
  in advantages is preserved by construction; it's the *cross-group* signal
  that gets distorted.)
- **`SampledPositionDataset` does no norm filtering.** Quick search confirms
  there is no `clamp`, `clip`, `outlier`, or `max_norm` in
  `src/nla/training/dataset.py`. Positions are sampled by token type only.

### Recommendations

For the first SFT run, do the cheapest of the following (in order):

1. **Add an AR-side target clip.** In `ActivationReconstructor.forward_sft`,
   after computing `target_scaled`, add an optional clip (default: clip per-element
   to `[-K, K]` with `K = 5`, i.e. allow up to ~5× α in raw norm before
   saturating). Five α is roughly 990, which preserves P99 (~233) untouched
   while bounding the contribution of true outliers to ~25× a typical
   element's MSE rather than thousands of times.
   ```python
   target_scaled = (target_activations / self.cfg.alpha).to(pred_scaled.dtype)
   if self.cfg.target_clip is not None:
       target_scaled = target_scaled.clamp(-self.cfg.target_clip, self.cfg.target_clip)
   ```
2. **Optionally, also filter outlier positions out of the sampling pool.** Add
   a pre-pass to `SampledPositionDataset` that drops positions where
   `‖h_t‖ > 3α` (roughly 592, comfortably above P99=233). This loses
   ~1–2% of positions but removes the contamination at the source. Keep this
   *off* by default; turn on if AR loss diverges or FVE oscillates.
3. **Do NOT clamp before AV's L2-normalize.** That would change the *direction*
   of the injected vector (toward zero for outliers) without any benefit;
   L2-normalize already strips magnitude.
4. **Track outlier frequency in metrics.** Cheap: log fraction of batch
   positions with `‖h‖ > 3α` and `> 5α` per step. If this fraction is
   substantially nonzero (say, > 5% per step in expectation) it justifies
   turning on filter (2).

These are *additive* to the paper recipe, not departures from it: the paper
assumed a roughly Gaussian norm distribution where P75-vs-P99 was a tame
multiplier. GR00T's image-token tail breaks that assumption; targeted clipping
restores well-conditioned MSE without changing the fundamental scaling.

---

## 6. Final recommendation

| Setting | Recommended value | Rationale |
|---|---|---|
| `α` (AV + AR) | **197.44** (P75 from `droid_100ep/stats.json`) | Paper's prescribed `α = P75(‖h‖₂)`; current code defaults to `196.15` (close but stale). Update `AVConfig.alpha` and `ARConfig.alpha` to load from `stats.json` rather than baking in a literal. |
| Hook layer | **layer 16** (`SELECT_LAYER - 1`, i.e. last surviving Qwen3-VL decoder layer) | This is the policy-relevant representation that GR00T's DiT actually consumes. The paper's "2/3 depth" advice optimizes a different objective (LM-internal interpretability) and would point at layer ~11; that's a worthwhile *follow-up sweep* but a strictly worse first-run choice for an action-policy NLA. |
| `act_proj` | Keep learnable affine (current code) | A fixed projection is closer to the paper but gives no SFT path from `h` to LM embedding space. The post-`act_proj` L2-normalize neutralizes any magnitude blow-up; only the direction is learned. |
| Slot token | Keep `<|act_slot|>` + `resize_token_embeddings` (current code) | Idempotent, single-id guaranteed. |
| Outlier handling | **Add per-element clip on AR target_scaled** (e.g. `clamp(±5)`); leave AV injection as-is | Heavy tail (std=1070 vs P75=197) will dominate AR MSE without it. Filtering outlier positions from the pool is a stronger optional follow-up. |
| GRPO reward | Already correct (`-‖h/α − AR(y)‖²`) | No change. |

### Pre-SFT TODO checklist

- [ ] Update `AVConfig.alpha` / `ARConfig.alpha` defaults to **197.44** (or, better, plumb `stats.json` through the SFT runner so α is never hard-coded).
- [ ] Add `ARConfig.target_clip: float | None = 5.0` and apply it inside `forward_sft` and (symmetrically) inside the GRPO reward computation in `grpo_step`. Keep the reward and the loss in the same clipped space so the reward signal AV optimizes is the same one AR is trained against.
- [ ] (Optional) Add a `‖h‖ ≤ max_norm_alphas * α` filter knob to `SampledPositionDataset`; default `None` (off).
- [ ] Confirm hook is still firing on `policy.model.backbone` after any GR00T version bump (`src/nla/extraction/_compat.py` patches drift; if `select_layer` changes upstream, `BACKBONE_EMBEDDING_DIM` and the wrapper output dict shape must still match).
- [ ] Log per-step outlier fractions (`> 3α`, `> 5α`) for the first 100 SFT steps to validate the assumed regime.

---

## Summary

The repo is paper-faithful where it matters most: AV injection is
project → L2-normalize → α (`src/nla/models/av.py:212-217`), AR is trained and
predicts in `h/α` space with `unscale=True` returning raw `h`
(`src/nla/models/ar.py:160-194`), and GRPO's reward is the scaled-space negative
MSE (`src/nla/training/grpo.py:253-267`). Layer 16 is the right hook target
despite the paper's "2/3 depth" rule of thumb, because GR00T physically
truncates Qwen3-VL to 16 layers (`src/nla/layer_spec.py:18-23, 45`) and layer
16 is the only representation the action head actually consumes — interpreting
anything earlier would explain a detour rather than the policy. For the first
SFT run, use **α = 197.44** (P75 from `droid_100ep/stats.json`) and **layer
16**, and add a per-element clip on AR's `target_scaled` (e.g. `clamp(±5)`) to
tame the heavy image-token tail (std=1070 vs P75=197) before it dominates the
MSE; AV injection itself needs no outlier handling because L2-normalize already
strips magnitude.

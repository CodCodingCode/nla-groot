# 05 — Architecture audit: AV / AR / injection vs. the NLA paper

> **Training vs semantics:** Injection math here is orthogonal to the **V2** issue (high reconstruction, **generic captions**). Closing that gap is training/eval — see **`docs/evals/v2_lessons_learned.md`**, **`07_sft_recipe_dataset_agnostic.md`**.

Audit of `src/nla/models/{av.py, ar.py, templates.py}` against
Fraser-Taliente et al., *Natural Language Autoencoders Produce Unsupervised
Explanations of LLM Activations* (Transformer Circuits, 2026), specialized
to GR00T N1.7-3B (Cosmos-Reason2-2B backbone, `SELECT_LAYER=16`,
hidden=2048 per `src/nla/layer_spec.py:36-46`).

Each item ends with **keep as-is for first run** or **fix before SFT**.

---

## 0. Architecture summary (what we actually have)

| Component | Code                                                          | Value                                                     |
| --------- | ------------------------------------------------------------- | --------------------------------------------------------- |
| Target    | `src/nla/layer_spec.py:36-46`                                 | GR00T-N1.7-3B / Cosmos-Reason2-2B, layer 16, hidden 2048  |
| AV base   | `src/nla/models/av.py:70` (`AVConfig.base_model`)             | `Qwen/Qwen3-4B-Instruct-2507` — hidden 2560, 36 layers    |
| AR base   | `src/nla/models/ar.py:52` (`ARConfig.base_model`)             | `Qwen/Qwen3-4B-Instruct-2507`, **truncated to 10 layers** |
| Projector | `src/nla/models/av.py:189`                                    | `nn.Linear(2048, 2560, bias=True)` → L2-norm → ×α         |
| AR head   | `src/nla/models/ar.py:93`                                     | `nn.Linear(hidden, 2048, bias=True)`                      |
| LoRA      | `src/nla/models/av.py:60-63`, `src/nla/models/ar.py:44-47`    | r=32, α=64, dropout=0.05, all 7 standard Qwen3 modules    |
| Slot      | `src/nla/models/av.py:90-99,121-140` (`<\|act_slot\|>`)       | Added via `add_special_tokens` + `resize_token_embeddings`|
| α         | `AVConfig.alpha=196.15` / `ARConfig.alpha=196.15`             | Fixed scalar, P75 of ‖h‖₂ from extraction stats           |

The relevant paper reference points:
* AV is a copy of the **target model**, single-slot injection at one prompt
  position (paper §3.1).
* AR is the target model **truncated to the first ℓ layers**, where ℓ is the
  NLA training layer, plus a learnable affine head on the last hidden state
  (paper §3.2).
* AV prompt: "You are a meticulous AI researcher … `<concept>{slot}</concept>` …".
* AR prompt: `Summary of the following text: <text>{explanation}</text> <summary>`.
* "Inject α × direction(`proj h`)" — direction normalized, scale by α (P75).

---

## 1. AV base model choice — `src/nla/models/av.py:70`

```70:71:src/nla/models/av.py
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    activation_dim: int = 2048              # GR00T backbone hidden size (Phase 1 confirmed)
```

**Tradeoff.** Paper inits AV from a copy of the *target* model so the
single-slot injection lands on a representation manifold the LM already
knows how to interpret (it is, after all, that model's own internal
activation at layer ℓ). We use Qwen3-4B-Instruct, which is from the same
*family* as Cosmos-Reason2-2B (both Qwen3-derived, same tokenizer, same
attention bias / norm) but **not** the target itself. Consequences:

* Hidden size **does not match** (2560 vs 2048), so `act_proj` is forced to
  be a non-square learnable linear; the identity-init shortcut at
  `av.py:190-194` is dead in our config. The projector has to do *all*
  cross-model alignment, not just dimension matching.
* The activation is no longer "Qwen3-4B's own layer-N output," so layers
  downstream of the slot do not see something they natively know how to
  consume. They have to learn to from LoRA + projector + warm-start labels.

**Is Qwen3-4B a reasonable proxy?** It is a reasonable *engineering* proxy
(ungated, same family, more capable text decoder than the 2B), but **not a
paper-faithful one**. The paper-faithful drop-in is the **public** base of
Cosmos-Reason2-2B: `Qwen/Qwen3-VL-2B-Instruct` (already documented as
`QWEN3_VL_BASE_REPO` at `src/nla/layer_spec.py:38`). Its text decoder has
hidden=2048, 28 layers, 8 KV heads — i.e. it *matches Cosmos exactly* on
the text-only sub-architecture. With Qwen3-VL-2B-Instruct's text decoder:

* Projector becomes a hidden=2048 → hidden=2048 map → can be identity-init
  (`av.py:190-191` path activates) → "raw injection ≈ paste activation into
  its own layer 16."
* No vision tower needed for AV; you simply call the text-only path.

**Recommendation.** **Fix before SFT (soft).** For a first real SFT run,
Qwen3-4B-Instruct will train and produce captions — keep it if you have
already invested compute. But the cleanest paper-faithful swap is to set
`AVConfig.base_model = "Qwen/Qwen3-VL-2B-Instruct"` and route through the
text decoder only. Doing this *before* a long SFT run is much cheaper than
re-running the experiment later. Cosmos-Reason2-2B itself is gated; the
public Qwen3-VL-2B-Instruct base is the next-best paper-faithful choice
and is already referenced in our own constants.

---

## 2. AR architecture — `src/nla/models/ar.py:52-55`

```52:55:src/nla/models/ar.py
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    activation_dim: int = 2048
    alpha: float = 196.15
    truncate_to_n_layers: int = 10
```

**Paper:** AR = target LM truncated to its first **ℓ** layers (where ℓ is
the layer NLA is trained on) + learned affine head. Intuition: AR's
last hidden state should mirror what the *target* would have at layer ℓ
reading the explanation text, then a tiny head maps that into the
activation space we are reconstructing.

**Ours:**
* AR base is Qwen3-4B (36-layer), not the target. Truncating Qwen3-4B's
  first N layers gives an *approximation* of "the target's first N layers"
  only insofar as Qwen3-4B's stack is comparable to Cosmos's. As with the
  AV, the cleanest fix is to switch to Qwen3-VL-2B-Instruct's text decoder.
* `truncate_to_n_layers=10` < `SELECT_LAYER=16`. Paper recipe says truncate
  to **the** training layer. Layer 16 is exactly the layer our activations
  come from (`layer_spec.py:45` and `extraction/hook.py:5`). Truncating to
  10 means AR reads explanation through *fewer* layers than the target
  used to produce the activation, then the affine head has to make up the
  difference. This will probably still learn — there is no theoretical
  reason a 10-layer Qwen3-4B + linear head cannot regress to 2048-D
  Cosmos activations — but it is a quiet deviation from the paper.

**Recommendation.** **Fix before SFT.** Set `ARConfig.truncate_to_n_layers
= 16` to match `SELECT_LAYER`. This is a one-line change and aligns AR
with the paper's recipe. (If you also switch AR base to Qwen3-VL-2B's text
decoder per Item 1, "first 16 layers" is even more semantically faithful
since Cosmos itself is Qwen3-VL truncated to 16.) Keeping the lower
truncation is only justifiable if VRAM forces it; on H100/H200 it should
not.

---

## 3. AV injection slot mechanism — `src/nla/models/av.py:90-296`

What the code does (verified by reading):

1. Registers `<|act_slot|>` as a new special token at construction
   (`av.py:90-99`, `121-140,181-187`) via
   `tokenizer.add_special_tokens(...)` + `base.resize_token_embeddings(...)`.
2. At forward time, in `_tokenize_prompts` (`av.py:241-279`), the
   placeholder `<<ACTIVATION_SLOT>>` is substituted with the real
   single-token slot string, encoded once, and the **position** of the
   slot id within `prompt_ids` is recorded per row.
3. `_embed_with_injection` (`av.py:282-296`) does:
   - `embeds = embed_module(input_ids)` — normal token-embedding lookup.
   - `proj = self._project_activation(activations, embeds.dtype)` —
     Linear → L2-norm → ×α.
   - `embeds = embeds.clone()` — important, breaks the input-embedding's
     in-place autograd path; injection writes into a *new* tensor.
   - `embeds[idx_b, idx_t] = proj` — overwrite only the slot position.

**Checks.**
* Overwrite happens **at runtime, not at __init__**: the input embedding
  row for the new slot id is never trained directly. Whatever
  `resize_token_embeddings` initialized for that row is irrelevant because
  every forward pass replaces it.
* **Only the marked position** is replaced (single `(b, t)` index per
  row), not a span.
* The `clone()` before write means gradients flow back through `act_proj`
  only, not through the embedding table at the slot row — correct.
* `tie_word_embeddings=True` for Qwen3 means the LM head shares weights
  with the input embedding. The output side still has a row for slot_id,
  but slot_id never appears in target labels (only in the prompt section,
  which is masked with `-100` in `_tokenize_prompts:255`), so no
  gradient pressure on that row from CE. Safe.

**Recommendation.** **Keep as-is for first run.** This is correct, matches
paper §3.1, and the implementation is hygienic.

---

## 4. Projector — `src/nla/models/av.py:189-217`

```189:194:src/nla/models/av.py
        self.act_proj = nn.Linear(cfg.activation_dim, hidden_size, bias=True)
        if cfg.activation_dim == hidden_size:
            nn.init.eye_(self.act_proj.weight)
        else:
            nn.init.xavier_uniform_(self.act_proj.weight)
        nn.init.zeros_(self.act_proj.bias)
```

```212:217:src/nla/models/av.py
    def _project_activation(self, activation: torch.Tensor, embed_dtype: torch.dtype) -> torch.Tensor:
        proj = self.act_proj(activation.to(self.act_proj.weight.dtype))
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = proj * self.cfg.alpha
        return proj.to(embed_dtype)
```

* Single `nn.Linear` with bias = **learnable affine** (matches paper's
  "weakly recommended" choice).
* L2-normalize then ×α = paper's injection-norm calibration.
* **Bias**: kept on, zero-init. The bias does contribute *direction*
  (since renormalization is applied after the linear, a non-zero bias
  shifts the direction toward itself when ‖`W x`‖ is small). In practice
  with our 2048→2560 mapping and α=196.15 the bias is a small DC term;
  letting it learn cannot hurt and matches the paper's affine
  recommendation. Turning it off would only be defensible if you wanted
  to enforce strict linearity through the origin.
* With our 2048 → 2560 mismatch, the `eye_` branch never fires; xavier-init
  is used. If we switch AV base to Qwen3-VL-2B-Instruct (Item 1), the
  identity-init branch activates and is the *right* warm start.

**Recommendation.** **Keep as-is for first run.** The projector is a
learnable affine + spherical projection + α scale, which is exactly the
paper's stance. Bias on is fine.

---

## 5. LoRA setup — `src/nla/models/av.py:60-77,196-200`

```60:63:src/nla/models/av.py
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
```

```196:200:src/nla/models/av.py
        if apply_lora:
            self.base = _wrap_lora(self.base, cfg)

        for p in self.act_proj.parameters():
            p.requires_grad = True
```

* Targets cover **all attention projections + all MLP projections** — the
  standard "saturated" LoRA target set for Qwen2/Qwen3 architectures.
  Sensible for SFT.
* `act_proj` is constructed **before** `_wrap_lora`, so it is a sibling of
  the PEFT-wrapped base, not inside the PEFT graph. Its parameters
  therefore train as full-precision fp32/bf16 params, independent of the
  LoRA adapters.
* The explicit `requires_grad = True` loop is defensive and correct — PEFT
  freezes all base params by default, but `act_proj` is *not* a base param,
  so it would be trainable anyway. The loop makes the invariant explicit
  in case someone wraps it differently in the future.
* `AVConfig.lora_alpha = 64 = 2 × lora_rank` is the standard recipe.

**Recommendation.** **Keep as-is for first run.** Confirmed `act_proj`
receives gradients per the docstring promise.

Minor note (no action): the AR uses
`task_type="FEATURE_EXTRACTION"` (`ar.py:283`) while AV uses
`task_type="CAUSAL_LM"` (`av.py:519`). Both are correct for their roles
(AR returns hidden states from the backbone via `_run_transformer`, never
the LM head; AV uses the full LM head for generation).

---

## 6. Prompt template — `src/nla/models/templates.py`

### AR template (lines 47)

```47:47:src/nla/models/templates.py
AR_PROMPT_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
```

**Verbatim match to paper.** Pick-off position is the last non-pad token,
which lands at the closing `>` of `<summary>`. ✓

**Recommendation.** **Keep as-is.**

### AV template (lines 31-43)

```31:43:src/nla/models/templates.py
AV_PROMPT_TEMPLATE = (
    "You are interpretability tooling for the GR00T N1.7 vision-language-action "
    "robot model. You are shown a single internal activation vector taken from "
    "one token position inside the backbone, plus a short hint indicating where "
    "in the input the position sits.\n"
    "Position type: {position_type}.\n"
    "Activation: " + AV_SLOT_PLACEHOLDER + "\n"
    "Describe, in 4-5 bullet points (one per line, '- <category>: <content>.'), "
    "what features the model is internally tracking at this position to predict "
    "its next action. The last bullet should describe what this exact position "
    "encodes.\n"
    "Bullets:"
)
```

**Deviates from paper.** Paper's AV prompt is open-ended ("describe the
concept"), framed as an interpretability researcher inspecting a token.
Ours adds two domain hooks (GR00T/robotics framing, position-type hint)
and one *structural* constraint (4-5 bullets, `- <category>: <content>.`).

Pros:
* Bullet format matches `labels.jsonl` (`openai_client.py` produces the
  same structure), so SFT teacher-forcing is consistent.
* Position-type hint lets the model condition style on
  `last_text` / `image_patch` / `anchor`.

Cons / risks:
* The 4-5-bullets formatting tax eats output tokens that AR will then
  have to ingest — longer, more boilerplate-y prompts may make AR's MSE
  reconstruction harder than necessary if the relevant signal is sparse
  in the bullets.
* During GRPO, the AV may discover that *abandoning* the bullet structure
  reconstructs `h` better. KL to a bullet-formatted reference will fight
  this, possibly capping reconstruction.
* "Predict its next action" framing biases captions toward
  action-relevant features; this *is* the right inductive bias for our
  use case, but it is not what the paper trains on.

**Recommendation.** **Keep as-is for first run** (the bullet format
matches our labels; throwing it out before the first run means
re-generating labels). After the first SFT/eval cycle, run a one-off
ablation with a paper-style open-ended prompt (no bullets,
`<concept>{slot}</concept>` framing) and compare AR FVE.

---

## 7. Single biggest risk (the one architectural flag)

**Both AV and AR use Qwen3-4B-Instruct, not the target model (or its
public sibling Qwen3-VL-2B-Instruct text decoder).**

This single choice cascades:

* The paper's central trick — "inject the activation back into the same
  model that produced it, ask the model to verbalize itself" — does not
  apply. The projector is the only mechanism aligning Cosmos's 2048-D
  layer-16 manifold to Qwen3-4B's 2560-D layer-* representations. LoRA
  has to do the rest by fine-tuning Qwen3-4B's stack to act *as if* the
  injected vector were a native intermediate state.
* On the AR side, the assumption that "AR's first ℓ layers ≈ target's
  first ℓ layers" collapses to "AR's first 10 (or 16) layers of Qwen3-4B
  ≈ Cosmos's first 16 layers." This will almost certainly work as
  regression, but it loses the paper's clean theoretical story and you
  cannot interpret AR's behavior as "completing the target's
  computation."
* Hidden-dim mismatch (2048 → 2560 in projector, 2560 → 2048 in head)
  also costs the identity-init shortcut on the projector and the natural
  symmetry on the head.

**Cheapest paper-faithful fix** before SFT: switch
`AVConfig.base_model` and `ARConfig.base_model` to
`Qwen/Qwen3-VL-2B-Instruct` (already documented as the public base of
Cosmos-Reason2-2B at `src/nla/layer_spec.py:38`) and call its text
decoder only. Hidden dim becomes 2048 across the board; projector
identity-inits; AR truncate-to-16 becomes "exactly the layers Cosmos
used." Cosmos itself is gated and cannot be relied on for reproducible
training; Qwen3-VL-2B-Instruct is the maximally faithful ungated
substitute.

If you keep Qwen3-4B for this first SFT run for compute / capacity
reasons, treat reconstruction FVE numbers as a *lower bound* relative to
a future paper-faithful run.

---

## TL;DR action list before first SFT

| #   | Issue                                          | Action            |
| --- | ---------------------------------------------- | ----------------- |
| 1   | AV base = Qwen3-4B (not target)                | Fix before SFT (soft): swap to `Qwen/Qwen3-VL-2B-Instruct` text decoder; otherwise keep but flag |
| 2   | `ARConfig.truncate_to_n_layers = 10` (≠ 16)    | **Fix before SFT** — set to 16 to match `SELECT_LAYER` |
| 3   | Slot injection mechanism                       | Keep as-is        |
| 4   | Projector (linear + L2 + ×α, bias on)          | Keep as-is        |
| 5   | LoRA targets + `act_proj` trainability         | Keep as-is        |
| 6a  | AR prompt — verbatim paper                     | Keep as-is        |
| 6b  | AV prompt — robotics-flavored, bulleted        | Keep for first run; ablate later |
| 7   | AV/AR base ≠ target (THE big one)              | Fix before SFT (soft, see #1) |

---

## Summary (4 sentences)

Implementation of slot injection, LoRA, projector (learnable affine + L2 +
α-scale), and the AR prompt template all match the NLA paper recipe and
are safe for a first SFT run. The biggest architectural deviation is that
both AV and AR are initialized from `Qwen/Qwen3-4B-Instruct-2507` rather
than from the GR00T backbone (Cosmos-Reason2-2B) or its public sibling
`Qwen/Qwen3-VL-2B-Instruct`, which breaks the paper's "AV is a copy of
the target" assumption and forces the projector and LoRA to absorb all
cross-model alignment. The other clearly fixable item is
`ARConfig.truncate_to_n_layers = 10` while extraction uses layer 16
(`SELECT_LAYER`); paper recipe is to truncate AR to *exactly* the NLA
layer, so this should be bumped to 16 before SFT. Recommended actions:
flip `ARConfig.truncate_to_n_layers` to 16, and (compute permitting)
switch both AV and AR `base_model` to `Qwen/Qwen3-VL-2B-Instruct`
(text-decoder-only) for paper-faithful initialization and an
identity-init projector.

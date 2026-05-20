"""Activation Reconstructor (AR).

A LoRA fine-tune of a *truncated* causal LM (default first 16 layers of
``Qwen/Qwen3-4B-Instruct``, matching GR00T's ``SELECT_LAYER`` so AR depth
mirrors where the activation lives in the language model) that ingests an
English explanation, runs it through the truncated transformer, and predicts
the original activation via a learned affine head on the last non-pad hidden
state.

Template (verbatim from the plan / paper)::

    Summary of the following text: <text>{explanation}</text> <summary>

The head's "pick-off" position is the last non-pad token in this template,
which the tokenizer turns into the literal closing characters of
``<summary>``.  We don't bother adding it as a special token: the position is
stable as the *last real token*, which is what we use.

α scaling
---------

The plan calls for "α scaling baked in".  Concretely:

- AR predicts the activation in α-scaled space:  ``pred_scaled = head(last_hidden)``.
- During training the target is also α-scaled:   ``target_scaled = target / α``.
- The MSE loss lives in α-scaled space (so a P75-norm activation has unit
  scale, making the loss well-conditioned regardless of α).
- ``predict(text)`` returns the *unscaled* prediction (``pred_scaled * α``)
  by default so callers don't have to remember the convention.  An
  ``unscale=False`` flag preserves the scaled output for chained losses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from nla.models.av import _hidden_size, _load_causal_lm, _load_tokenizer
from nla.models.templates import PositionType, PromptVersion, render_ar_prompt


DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass
class ARConfig:
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    activation_dim: int = 2048
    alpha: float = 197.44
    truncate_to_n_layers: int = 16
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    dtype: str = "bfloat16"
    max_length: int = 1024
    # Optional symmetric clamp on the α-scaled target tensor inside
    # ``forward_sft`` to tame heavy tails (a P75-normalized activation has unit
    # scale, but outliers in std are well above that). ``None`` disables; e.g.
    # 5.0 clamps to ±5 in α-scaled space. Does NOT affect inference / predict.
    clip_target_scaled: float | None = None
    # Temperature for the InfoNCE contrastive term in ``forward_sft``.  Lower
    # temperature = sharper softmax = stronger contrastive gradient.  The
    # similarity matrix uses cosine (in [-1, 1]); dividing by 0.1 maps that to
    # [-10, 10] before softmax, which gives well-scaled gradients even when AR
    # predictions are nearly identical across rows (the failure mode that
    # produced mode collapse with the legacy negative-L2 sims at α-scaled
    # magnitudes ~1e-3).
    nce_temperature: float = 0.1
    # V5 conditioned AR prompt (prepends position / timestep / instruction).
    ar_prompt_version: PromptVersion = "legacy"
    ar_include_step_index: bool = True
    ar_include_instruction: bool = True


class ActivationReconstructor(nn.Module):
    """Truncated causal LM + LoRA + affine head."""

    def __init__(
        self,
        cfg: ARConfig,
        *,
        tokenizer=None,
        base_model=None,
        apply_lora: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(cfg.base_model)
        self._pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id or 0)
        )

        if base_model is None:
            base_model = _load_causal_lm(cfg.base_model, dtype=cfg.dtype)
        base_model = _truncate_layers(base_model, cfg.truncate_to_n_layers)
        hidden_size = _hidden_size(base_model)
        self.base = base_model

        if apply_lora:
            self.base = _wrap_lora_ar(self.base, cfg)

        self.head = nn.Linear(hidden_size, cfg.activation_dim, bias=True)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ utils

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def _render_prompts(
        self,
        explanations: list[str],
        *,
        position_types: list[PositionType] | None = None,
        step_indices: list[int | None] | None = None,
        instructions: list[str | None] | None = None,
    ) -> list[str]:
        B = len(explanations)
        version = self.cfg.ar_prompt_version
        if version == "legacy":
            return [render_ar_prompt(e, prompt_version="legacy") for e in explanations]
        if position_types is None:
            position_types = ["fallback"] * B
        if len(position_types) != B:
            raise ValueError(
                f"position_types length {len(position_types)} != batch size {B}."
            )
        step_list = step_indices if step_indices is not None else [None] * B
        instr_list = instructions if instructions is not None else [None] * B
        if len(step_list) != B or len(instr_list) != B:
            raise ValueError(
                "step_indices and instructions must match batch size when provided."
            )
        out: list[str] = []
        for i, expl in enumerate(explanations):
            step_arg = step_list[i] if self.cfg.ar_include_step_index else None
            instr_arg = instr_list[i] if self.cfg.ar_include_instruction else None
            out.append(
                render_ar_prompt(
                    expl,
                    position_type=position_types[i],
                    step_index=step_arg,
                    instruction=instr_arg,
                    prompt_version="context_v5",
                )
            )
        return out

    def _tokenize(
        self,
        explanations: list[str],
        *,
        position_types: list[PositionType] | None = None,
        step_indices: list[int | None] | None = None,
        instructions: list[str | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        rendered = self._render_prompts(
            explanations,
            position_types=position_types,
            step_indices=step_indices,
            instructions=instructions,
        )
        return self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            add_special_tokens=False,
        )

    def _run_transformer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns last-layer hidden states ``(B, T, H)`` from the (truncated) backbone.

        We bypass the LM head by calling the underlying model's transformer
        block (``base.model`` in HF causal LMs), which returns hidden states.
        With PEFT wrappers we go through ``base_model.model`` to reach the
        same place.
        """
        backbone = _resolve_backbone(self.base)
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        return out.last_hidden_state

    def _pickoff(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Take the last real (non-pad) hidden state per row."""
        seq_lens = attention_mask.sum(dim=1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, seq_lens]

    # ------------------------------------------------------------------ public

    def forward(
        self,
        explanations: list[str],
        *,
        device: torch.device | None = None,
        position_types: list[PositionType] | None = None,
        step_indices: list[int | None] | None = None,
        instructions: list[str | None] | None = None,
    ) -> torch.Tensor:
        """Return α-scaled predictions ``(B, activation_dim)``.

        Use ``predict(...)`` if you want unscaled outputs (the default for
        consumers of the activation).
        """
        toks = self._tokenize(
            explanations,
            position_types=position_types,
            step_indices=step_indices,
            instructions=instructions,
        )
        if device is None:
            device = self.device
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)
        hidden = self._run_transformer(input_ids, attention_mask)
        last = self._pickoff(hidden, attention_mask)
        return self.head(last.to(self.head.weight.dtype))

    def predict(
        self,
        explanations: list[str],
        *,
        unscale: bool = True,
        position_types: list[PositionType] | None = None,
        step_indices: list[int | None] | None = None,
        instructions: list[str | None] | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred_scaled = self.forward(
                explanations,
                position_types=position_types,
                step_indices=step_indices,
                instructions=instructions,
            )
        return pred_scaled * self.cfg.alpha if unscale else pred_scaled

    def forward_sft(
        self,
        explanations: list[str],
        target_activations: torch.Tensor,
        *,
        return_nce: bool = False,
        negative_explanations: list[list[str]] | None = None,
        position_types: list[PositionType] | None = None,
        step_indices: list[int | None] | None = None,
        instructions: list[str | None] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Returns ``(mse, pred_scaled)``, or ``(mse, nce, pred_scaled)`` if
        ``return_nce=True``.

        ``target_activations`` is the *unscaled* original activation; we
        divide by α inside so the loss lives in well-conditioned space.

        Contrastive (InfoNCE) term
        --------------------------
        When ``return_nce=True`` we also compute an InfoNCE-style contrastive
        loss over the batch.  For each row ``i`` the logit at column ``j`` is
        ``-||AR(y_i) - h_j/α||²`` (mean over hidden), the labels are the
        diagonal.  Intuitively: AR should reconstruct *its own row's*
        activation better than any other row's.  This penalizes generic
        descriptions that reconstruct equally well to many activations -- the
        precise failure mode that lets memorization slip through plain MSE on
        small training corpora.  With batch size 1 (smoke tests) we return a
        zero NCE term silently.

        Hard-negative augmentation
        --------------------------
        ``negative_explanations`` is optional.  When provided as a
        ``list[list[str]]`` of shape ``[B][K_neg]`` (K_neg may vary across
        rows, but typically uniform), we tokenize each row's negatives,
        compute AR predictions on them, and append per-row similarity
        columns to the (B, B) softmax matrix, yielding a (B, B+K_neg)
        logit matrix.  Each new column at ``[i, B+k]`` is the cosine
        similarity ``cos(pred_scaled[i], pred_neg_scaled[i, k])``: AR
        should reconstruct its own caption to an embedding that does
        *not* coincide with what it would compute for a hard-negative
        caption.  Standard cross-entropy with labels ``arange(B)``.

        When ``negative_explanations is None`` the code path is byte-identical
        to the random-in-batch-only baseline.
        """
        pred_scaled = self.forward(
            explanations,
            device=target_activations.device,
            position_types=position_types,
            step_indices=step_indices,
            instructions=instructions,
        )
        target_scaled = (target_activations / self.cfg.alpha).to(pred_scaled.dtype)
        if self.cfg.clip_target_scaled is not None:
            clip = float(self.cfg.clip_target_scaled)
            target_scaled = target_scaled.clamp(-clip, clip)
        mse = nn.functional.mse_loss(pred_scaled, target_scaled)
        if not return_nce:
            return mse, pred_scaled

        B = pred_scaled.shape[0]
        if B > 1:
            # Cosine similarity with temperature.  Cosine lives in [-1, 1]
            # regardless of the tiny per-dim magnitudes of α-scaled
            # activations (~0.03 RMS), which is critical: the legacy
            # ``sims = -(diffs**2).mean(-1)`` implementation produced sims of
            # magnitude ~1e-3 on this dataset, so softmax over the 4-way
            # diagonal-vs-off-diagonal contrast was numerically uniform and
            # the InfoNCE term collapsed to ``ln(B)`` with zero gradient.
            # Cosine + temperature 0.1 puts sims in [-10, 10], giving a real
            # contrastive signal even at batch size 4.
            sims = nn.functional.cosine_similarity(
                pred_scaled.unsqueeze(1),                 # (B, 1, H)
                target_scaled.unsqueeze(0),               # (1, B, H)
                dim=-1,
            )                                             # (B, B), in [-1, 1]
            if negative_explanations is not None:
                sims_neg = self._hard_negative_sims(
                    pred_scaled=pred_scaled,
                    negative_explanations=negative_explanations,
                    position_types=position_types,
                    step_indices=step_indices,
                    instructions=instructions,
                )
                sims = torch.cat([sims, sims_neg], dim=1)  # (B, B+K_neg)
            temp = max(1e-6, float(self.cfg.nce_temperature))
            sims = sims / temp
            labels = torch.arange(B, device=pred_scaled.device)
            nce = nn.functional.cross_entropy(sims.float(), labels)
        else:
            nce = torch.zeros((), device=pred_scaled.device, dtype=mse.dtype)
        return mse, nce, pred_scaled

    def _hard_negative_sims(
        self,
        *,
        pred_scaled: torch.Tensor,
        negative_explanations: list[list[str]],
        position_types: list[PositionType] | None = None,
        step_indices: list[int | None] | None = None,
        instructions: list[str | None] | None = None,
    ) -> torch.Tensor:
        """Return ``(B, K_neg)`` cosine sims of pred[i] vs AR(negative[i,k]).

        Requires every row to provide the same number of negatives so we can
        run a single batched AR forward; we validate the shape here to fail
        loudly rather than silently mis-aligning rows in the cat below.
        """
        B = pred_scaled.shape[0]
        if len(negative_explanations) != B:
            raise ValueError(
                "negative_explanations must have one list per batch row; "
                f"got {len(negative_explanations)} rows for batch size {B}."
            )
        k_per_row = {len(row) for row in negative_explanations}
        if len(k_per_row) != 1:
            raise ValueError(
                "negative_explanations must be rectangular (same K_neg for "
                f"every row); got row-lengths {sorted(k_per_row)}."
            )
        K_neg = next(iter(k_per_row))
        if K_neg == 0:
            return pred_scaled.new_zeros((B, 0))
        flat = [neg for row in negative_explanations for neg in row]
        neg_position_types: list[PositionType] | None = None
        neg_step_indices: list[int | None] | None = None
        neg_instructions: list[str | None] | None = None
        if position_types is not None:
            neg_position_types = [
                position_types[i] for i in range(B) for _ in range(K_neg)
            ]
        if step_indices is not None:
            neg_step_indices = [
                step_indices[i] for i in range(B) for _ in range(K_neg)
            ]
        if instructions is not None:
            neg_instructions = [
                instructions[i] for i in range(B) for _ in range(K_neg)
            ]
        pred_neg_flat = self.forward(
            flat,
            device=pred_scaled.device,
            position_types=neg_position_types,
            step_indices=neg_step_indices,
            instructions=neg_instructions,
        )  # (B*K_neg, H)
        pred_neg = pred_neg_flat.view(B, K_neg, pred_neg_flat.shape[-1])
        # cos(pred[i], pred_neg[i, k]) for each (i, k).
        return nn.functional.cosine_similarity(
            pred_scaled.unsqueeze(1),                 # (B, 1, H)
            pred_neg,                                 # (B, K_neg, H)
            dim=-1,
        )

    # ------------------------------------------------------------------ save/load

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(str(path))
        torch.save(self.head.state_dict(), str(path / "head.pt"))
        cfg = {**self.cfg.__dict__}
        cfg["lora_targets"] = list(cfg["lora_targets"])
        (path / "ar_config.json").write_text(json.dumps(cfg, indent=2))

    def load_head(self, path: str | Path) -> None:
        sd = torch.load(str(Path(path) / "head.pt"), map_location="cpu")
        self.head.load_state_dict(sd)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _truncate_layers(model, keep: int):
    """Trim the model's decoder stack to the first ``keep`` layers, in place.

    Works for HF Qwen2/Qwen3-style architectures where layers live at
    ``model.model.layers``.  Returns the same ``model`` for chaining.
    """
    backbone = _resolve_backbone(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise AttributeError(
            f"Model backbone {type(backbone).__name__} has no `.layers`; "
            "AR layer truncation cannot proceed without manual handling."
        )
    if keep <= 0:
        raise ValueError(f"truncate_to_n_layers must be > 0; got {keep}.")
    if keep >= len(layers):
        return model
    # ModuleList does not support slice assignment directly; rebuild.
    new_list = nn.ModuleList([layers[i] for i in range(keep)])
    backbone.layers = new_list
    # Some configs cache num_hidden_layers; keep it consistent.
    cfg = getattr(backbone, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = keep
    return model


def _resolve_backbone(model):
    """Return the bare transformer (the one whose call returns hidden_states).

    Handles three wrapper layouts we care about:
    - bare ``Qwen3ForCausalLM`` (``.model``)
    - PEFT-wrapped ``Qwen3ForCausalLM`` (``.base_model.model.model``)
    - already-bare ``Qwen3Model`` (returned unchanged)
    """
    # PEFT wrapper:  PeftModel.base_model.model is the original ForCausalLM
    inner = getattr(model, "base_model", None)
    if inner is not None:
        inner = getattr(inner, "model", inner)
        # If we now point at the *CausalLM*, descend one more level to the
        # transformer; if we point at the transformer already, return.
        sub = getattr(inner, "model", None)
        return sub if sub is not None else inner
    # Bare ForCausalLM:
    sub = getattr(model, "model", None)
    if sub is not None:
        return sub
    return model


def _wrap_lora_ar(model, cfg: ARConfig):
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, lora_cfg)

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
from nla.models.templates import render_ar_prompt


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

    def _tokenize(self, explanations: list[str]) -> dict[str, torch.Tensor]:
        rendered = [render_ar_prompt(e) for e in explanations]
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
    ) -> torch.Tensor:
        """Return α-scaled predictions ``(B, activation_dim)``.

        Use ``predict(...)`` if you want unscaled outputs (the default for
        consumers of the activation).
        """
        toks = self._tokenize(explanations)
        if device is None:
            device = self.device
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)
        hidden = self._run_transformer(input_ids, attention_mask)
        last = self._pickoff(hidden, attention_mask)
        return self.head(last.to(self.head.weight.dtype))

    def predict(self, explanations: list[str], *, unscale: bool = True) -> torch.Tensor:
        with torch.no_grad():
            pred_scaled = self.forward(explanations)
        return pred_scaled * self.cfg.alpha if unscale else pred_scaled

    def forward_sft(
        self,
        explanations: list[str],
        target_activations: torch.Tensor,
        *,
        return_nce: bool = False,
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
        """
        pred_scaled = self.forward(explanations, device=target_activations.device)
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
            temp = max(1e-6, float(self.cfg.nce_temperature))
            sims = sims / temp
            labels = torch.arange(B, device=pred_scaled.device)
            nce = nn.functional.cross_entropy(sims.float(), labels)
        else:
            nce = torch.zeros((), device=pred_scaled.device, dtype=mse.dtype)
        return mse, nce, pred_scaled

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

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
from typing import Literal

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
    # 0 (sentinel) or any value >= the model's actual num_hidden_layers
    # = no truncation, i.e. use all transformer blocks. Positive < depth
    # truncates to the first N blocks (v8 default 16, paying compute /
    # capacity for memory + speed). v9 recipe drops this to 0.
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

    # Reconstruction loss formulation (v9 lever for "cosine high, FVE
    # negative" pattern). Plain MSE entangles direction + magnitude error
    # under one gradient; decomposed splits them so we can weight each.
    # "mse"               : standard ``F.mse_loss(pred_scaled, target_scaled)`` -- byte-identical to legacy.
    # "decomposed"        : (1 - cosine(pred, target)).mean() + ar_scale_weight * (log||pred|| - log||target||)^2.mean()
    # "huber"             : ``F.smooth_l1_loss(pred_scaled, target_scaled)`` -- smoother gradient than MSE for outliers.
    ar_loss_mode: Literal["mse", "decomposed", "huber"] = "mse"
    # Weight on the log-magnitude term in the decomposed loss. Ignored
    # unless ``ar_loss_mode == "decomposed"``. Dial up to push magnitudes
    # closer to target.
    ar_scale_weight: float = 0.1

    # Stage-3 plan: spatial AR output. ``scalar`` (default, V3/V4/V5 byte-
    # identical) returns a single ``(B, H)`` vector per text input that gets
    # broadcast across image_patch slots at inject time. ``spatial`` returns
    # a ``(B, N, H)`` grid; one vector per image_patch position. Pair with
    # placement="image_patch_spatial" at steer time so the k-th predicted
    # vector lands in the k-th image_patch slot, restoring the spatial
    # variation that real GR00T vision tokens carry.
    head_type: Literal["scalar", "spatial"] = "scalar"
    # Number of spatial positions emitted by the spatial head. Must match
    # the count of image_patch tokens in the GR00T forward (post-pooling).
    # Ignored when head_type=="scalar". 0 (default) is a sentinel — set it
    # explicitly when head_type=="spatial" or AR will raise at init.
    spatial_n_positions: int = 0
    # Hidden width of the small MLP that produces the per-position vectors.
    # Only consulted when head_type=="spatial". Defaults to the AR base
    # model's hidden size so the head adds no extra hyperparameters when
    # left at 0.
    spatial_head_hidden: int = 0


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

        if cfg.head_type == "spatial":
            if cfg.spatial_n_positions <= 0:
                raise ValueError(
                    "ARConfig.head_type='spatial' requires spatial_n_positions > 0; "
                    f"got {cfg.spatial_n_positions}. Set it to the number of "
                    "image_patch tokens GR00T emits at layer 16 (post-pooling)."
                )
            self.head = SpatialReconstructionHead(
                hidden_size=hidden_size,
                activation_dim=cfg.activation_dim,
                n_positions=int(cfg.spatial_n_positions),
                hidden=int(cfg.spatial_head_hidden) or hidden_size,
            )
        else:
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
        """Return α-scaled predictions.

        - head_type='scalar': ``(B, activation_dim)`` (legacy)
        - head_type='spatial': ``(B, N, activation_dim)`` (Stage-3)

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
        head_dtype = self._head_param_dtype()
        return self.head(last.to(head_dtype))

    def _head_param_dtype(self) -> torch.dtype:
        """Pick the dtype the head's first parameter uses, working for both
        the legacy nn.Linear and the spatial decoder."""
        for p in self.head.parameters():
            return p.dtype
        return torch.float32

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
        if pred_scaled.dim() != target_activations.dim():
            raise ValueError(
                f"AR head_type={self.cfg.head_type!r} produces "
                f"{pred_scaled.dim()}D predictions (shape {tuple(pred_scaled.shape)}) "
                f"but target_activations is {target_activations.dim()}D "
                f"(shape {tuple(target_activations.shape)}). For "
                "head_type='spatial' the dataset must supply per-position "
                "targets of shape (B, N, H); for 'scalar' a (B, H) tensor."
            )
        target_scaled = (target_activations / self.cfg.alpha).to(pred_scaled.dtype)
        if self.cfg.clip_target_scaled is not None:
            clip = float(self.cfg.clip_target_scaled)
            target_scaled = target_scaled.clamp(-clip, clip)
        # ``mse`` is the name we keep for the *training* loss for back-compat
        # with the rest of the codebase (caller logs as ``ar_mse``). The
        # underlying formula depends on ``cfg.ar_loss_mode``. The returned
        # ``pred_scaled`` is the raw AR prediction either way, so eval / NCE /
        # ĥ-injection paths are unchanged.
        mse = _ar_recon_loss(
            pred_scaled, target_scaled, self.cfg,
        )
        if not return_nce:
            return mse, pred_scaled

        # For the contrastive term we collapse spatial predictions to a single
        # per-row vector via mean-pool over N. This keeps the existing InfoNCE
        # path intact: AR's per-row identity should still differ across rows
        # regardless of head shape.
        if pred_scaled.dim() == 3:
            pred_for_nce = pred_scaled.mean(dim=1)
            target_for_nce = target_scaled.mean(dim=1)
        else:
            pred_for_nce = pred_scaled
            target_for_nce = target_scaled

        B = pred_for_nce.shape[0]
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
                pred_for_nce.unsqueeze(1),                # (B, 1, H)
                target_for_nce.unsqueeze(0),              # (1, B, H)
                dim=-1,
            )                                             # (B, B), in [-1, 1]
            if negative_explanations is not None:
                sims_neg = self._hard_negative_sims(
                    pred_scaled=pred_for_nce,
                    negative_explanations=negative_explanations,
                    position_types=position_types,
                    step_indices=step_indices,
                    instructions=instructions,
                )
                sims = torch.cat([sims, sims_neg], dim=1)  # (B, B+K_neg)
            temp = max(1e-6, float(self.cfg.nce_temperature))
            sims = sims / temp
            labels = torch.arange(B, device=pred_for_nce.device)
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
        )  # (B*K_neg, H) for scalar head, (B*K_neg, N, H) for spatial.
        if pred_neg_flat.dim() == 3:
            # Mean-pool spatial predictions; the contrastive term scores
            # row identity, not per-position structure.
            pred_neg_flat = pred_neg_flat.mean(dim=1)
        pred_neg = pred_neg_flat.view(B, K_neg, pred_neg_flat.shape[-1])
        # cos(pred[i], pred_neg[i, k]) for each (i, k). pred_scaled is the
        # mean-pooled (B, H) view passed in from forward_sft.
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

def _ar_recon_loss(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    cfg: "ARConfig",
) -> torch.Tensor:
    """Reconstruction loss dispatched on ``cfg.ar_loss_mode``.

    All three modes operate in α-scaled space (same as legacy MSE) so the
    loss magnitude is comparable to v8 runs at startup and the existing
    LR / warmup schedules transfer.

    - ``mse``        : ``F.mse_loss(pred, target)`` (byte-identical to legacy).
    - ``huber``      : ``F.smooth_l1_loss(pred, target)``. Same as MSE for
                       small errors; linear for large errors. Mitigates
                       outlier-dominated gradients.
    - ``decomposed`` : separates direction and scale errors so each can be
                       weighted. Direction term is ``(1 - cos(pred, tgt))``
                       averaged over rows; scale term is squared log-norm
                       error averaged over rows. Targets the "high cosine
                       low FVE" pattern where MSE alone can't tell the
                       model that magnitude is the problem.
    """
    mode = cfg.ar_loss_mode
    if mode == "mse":
        return nn.functional.mse_loss(pred_scaled, target_scaled)
    if mode == "huber":
        return nn.functional.smooth_l1_loss(pred_scaled, target_scaled)
    if mode == "decomposed":
        # Reduce K dim for spatial heads by flattening (B, K, H) -> (B*K, H)
        # so direction / scale are per-row metrics for either head type.
        if pred_scaled.dim() == 3:
            p = pred_scaled.reshape(-1, pred_scaled.shape[-1])
            t = target_scaled.reshape(-1, target_scaled.shape[-1])
        else:
            p = pred_scaled
            t = target_scaled
        eps = 1e-6
        p_norm = p.norm(dim=-1).clamp_min(eps)
        t_norm = t.norm(dim=-1).clamp_min(eps)
        cos = (p * t).sum(dim=-1) / (p_norm * t_norm)
        dir_loss = (1.0 - cos).mean()
        scale_loss = (torch.log(p_norm) - torch.log(t_norm)).pow(2).mean()
        return dir_loss + float(cfg.ar_scale_weight) * scale_loss
    raise ValueError(
        f"ARConfig.ar_loss_mode must be one of "
        f"'mse' / 'decomposed' / 'huber'; got {mode!r}"
    )


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
    if keep < 0:
        raise ValueError(f"truncate_to_n_layers must be >= 0; got {keep}.")
    if keep == 0 or keep >= len(layers):
        # 0 (sentinel) or >= actual depth => no truncation.
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


class SpatialReconstructionHead(nn.Module):
    """Stage-3 AR head: maps the LM's picked-off hidden vector ``(B, H_lm)``
    to a spatial grid of N per-position activation vectors ``(B, N, H_act)``.

    Architecture is intentionally minimal: a learned per-position query
    embedding crossed with the LM hidden state via a position-conditioned
    MLP. Pure feed-forward — no attention — keeping the head's parameter
    count small (~N * H_lm) and the wall-clock overhead negligible vs. the
    LM forward.

    Initialization mirrors the scalar head: Xavier on the projection
    weights, zeros on the bias. Per-position embeddings start with small
    Gaussian noise so the head doesn't collapse to identical outputs across
    positions at step 0.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        activation_dim: int,
        n_positions: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.n_positions = int(n_positions)
        self.activation_dim = int(activation_dim)
        # Per-position learned offset on the LM hidden state. Initialized
        # small so the head's initial predictions are uniform across N
        # (matches the legacy broadcast-one-vector behavior at step 0).
        self.position_embeddings = nn.Parameter(
            torch.randn(self.n_positions, hidden_size) * 0.02
        )
        self.fuse = nn.Linear(hidden_size, hidden)
        self.act = nn.GELU()
        self.project = nn.Linear(hidden, activation_dim, bias=True)
        nn.init.xavier_uniform_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """``last_hidden`` is ``(B, H_lm)`` (the AR pick-off vector).
        Returns ``(B, N, activation_dim)``."""
        if last_hidden.dim() != 2:
            raise ValueError(
                f"SpatialReconstructionHead expects (B, H_lm); got "
                f"{tuple(last_hidden.shape)}"
            )
        B = last_hidden.shape[0]
        # Broadcast position embeddings: (B, N, H_lm) = h.unsqueeze(1) + pos
        pos = self.position_embeddings.to(last_hidden.dtype)
        per_pos = last_hidden.unsqueeze(1) + pos.unsqueeze(0)   # (B, N, H_lm)
        fused = self.act(self.fuse(per_pos))                    # (B, N, hidden)
        return self.project(fused)                              # (B, N, act_dim)


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

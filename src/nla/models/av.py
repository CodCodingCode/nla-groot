"""Activation Verbalizer (AV).

A LoRA fine-tune of a causal LM (default ``Qwen/Qwen3-4B-Instruct``) that
takes a single ``hidden_dim``-sized activation vector at one slot of a fixed
text prompt and autoregressively generates an English description of what
the activation encodes.

Injection
---------

The prompt template (``nla.models.templates.AV_PROMPT_TEMPLATE``) contains a
single ``<<ACTIVATION_SLOT>>`` marker.  At construction time we resolve a
single-token id that is unlikely to appear in normal text (Qwen tokenizers
expose several ``<|reserved_special_token_*|>`` ids for exactly this).  At
forward/generate time we:

1. Tokenize the prompt into ``input_ids``;
2. Compute input embeddings via the base LM's embedding layer;
3. Project the activation through a learnable linear map into the LM's
   hidden size, L2-normalize it, then scale by ``alpha`` (the 75th-percentile
   activation L2 norm measured by ``nla.extraction.stats``);
4. Overwrite the embedding at the slot index with this projected vector;
5. Run the base LM with ``inputs_embeds``.

α scaling
---------

We follow the paper's "scale the injection so it lies in the LM's natural
activation range" recipe: project, L2-normalize, multiply by α.  The plan
calls for α = the 75th-percentile activation L2 norm, which we measured in
Phase 1.  The paper writers note injection scale tolerates roughly an order
of magnitude around this value, so we treat α as a config knob.

LoRA
----

Default rank 32 with the standard target set
(``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``, ``up_proj``,
``down_proj``).  The activation projector is *always* trainable (it is the
only path by which activation gradients reach the LM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from nla.models.templates import (
    AV_SLOT_PLACEHOLDER,
    PositionType,
    render_av_prompt,
)


# Default LoRA targets for Qwen3-* (matches Qwen2/Qwen3 module names).
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass
class AVConfig:
    """Verbalizer configuration."""

    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    activation_dim: int = 2048              # GR00T backbone hidden size (Phase 1 confirmed)
    # P75 ‖h‖₂ from the activation corpus' stats.json. The default here is a
    # legacy placeholder; production runs must always override via run_sft's
    # --stats-json so the value matches the actual extraction (a wrong alpha
    # silently miscalibrates the MSE and FVE).
    alpha: float = 197.44
    lora_rank: int = 32
    lora_alpha: int = 64                    # 2 * rank
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    dtype: str = "bfloat16"
    max_new_tokens: int = 200
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    # The slot string is resolved at __init__ to a real single-token id.
    slot_token_id: int = -1
    slot_token_str: str = ""
    # If ``True``, we add a fresh special token (``new_slot_token_str``) to the
    # tokenizer and resize the base model's embeddings accordingly.  This is
    # the default because Qwen3-4B does not ship with any guaranteed-unused
    # single-token candidates.  If you've already added a slot in your
    # tokenizer (e.g. when loading from a saved AV checkpoint) set this to
    # ``False`` and the constructor will look the existing id up.
    add_new_slot_token: bool = True
    new_slot_token_str: str = "<|act_slot|>"
    # Fallback candidates tried only if ``add_new_slot_token=False``.
    reserved_token_candidates: tuple[str, ...] = field(
        default_factory=lambda: (
            "<|act_slot|>",
            *(f"<|reserved_special_token_{i}|>" for i in range(8)),
            "<|fim_pad|>",
        )
    )


def find_slot_token_id(tokenizer, candidates: Iterable[str]) -> tuple[int, str]:
    """Locate a stable, single-token id usable as the activation slot.

    Tries the given candidates in order.  Returns ``(token_id, token_str)``.
    Raises ``RuntimeError`` if nothing matches.
    """
    for tok in candidates:
        try:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) == 1:
                return int(ids[0]), tok
        except Exception:
            continue
    raise RuntimeError(
        f"None of the candidate slot strings tokenize to a single id: {list(candidates)}. "
        "Use AVConfig(add_new_slot_token=True) so a fresh token is registered."
    )


def ensure_slot_token(tokenizer, base_model, slot_str: str) -> int:
    """Add ``slot_str`` to the tokenizer (if missing) and resize embeddings.

    Returns the single-token id of ``slot_str`` after the operation.  Idempotent:
    if the token already encodes to a single id, no change is made.
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
        raise RuntimeError(
            f"After add_special_tokens, {slot_str!r} still does not tokenize to a single id: {ids}"
        )
    return int(ids[0])


class ActivationVerbalizer(nn.Module):
    """A LoRA-wrapped causal LM with single-slot activation injection.

    The module is usable in three modes:

    - ``forward_sft(activations, position_types, target_texts)`` — returns
      a CausalLMOutput-like object with ``loss`` over the target tokens only
      (prompt tokens masked).
    - ``generate(activations, position_types, ...)`` — sampling rollout, used
      both for inference and for GRPO rollouts in Phase 5.
    - ``forward(...)`` — low-level pass-through to the base LM after
      activation injection; used when the caller already has tokenized inputs.
    """

    def __init__(
        self,
        cfg: AVConfig,
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
        self.base = base_model
        hidden_size = _hidden_size(self.base)

        # Resolve (and if needed register) the slot token *after* the base
        # model exists so we can resize embeddings in one step.
        if cfg.add_new_slot_token:
            slot_id = ensure_slot_token(self.tokenizer, self.base, cfg.new_slot_token_str)
            slot_str = cfg.new_slot_token_str
        else:
            slot_id, slot_str = find_slot_token_id(self.tokenizer, cfg.reserved_token_candidates)
        self.cfg.slot_token_id = slot_id
        self.cfg.slot_token_str = slot_str

        self.act_proj = nn.Linear(cfg.activation_dim, hidden_size, bias=True)
        if cfg.activation_dim == hidden_size:
            nn.init.eye_(self.act_proj.weight)
        else:
            nn.init.xavier_uniform_(self.act_proj.weight)
        nn.init.zeros_(self.act_proj.bias)

        if apply_lora:
            self.base = _wrap_lora(self.base, cfg)

        for p in self.act_proj.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------ utils

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def _embed(self) -> nn.Module:
        # PEFT wrappers forward attribute access; this works on bare and wrapped.
        return self.base.get_input_embeddings()

    def _project_activation(self, activation: torch.Tensor, embed_dtype: torch.dtype) -> torch.Tensor:
        """Project, L2-normalize, scale by α. Returns ``(B, H)`` in ``embed_dtype``."""
        proj = self.act_proj(activation.to(self.act_proj.weight.dtype))
        proj = proj / proj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        proj = proj * self.cfg.alpha
        return proj.to(embed_dtype)

    def _tokenize_prompts(
        self,
        position_types: list[PositionType],
        target_texts: list[str] | None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Tokenize each (prompt, optional target) pair.

        Returns
        -------
        input_ids:        ``(B, T_max)`` right-padded with ``pad_id``.
        attention_mask:   ``(B, T_max)`` 1 over real tokens.
        labels:           ``(B, T_max)``: ``-100`` on prompt + pad, target ids otherwise.
        slot_indices:     ``list[int]`` of length B, slot position in each row.
        """
        device = device or self.device
        slot_id = self.cfg.slot_token_id

        encoded_ids: list[list[int]] = []
        encoded_labels: list[list[int]] = []
        slot_indices: list[int] = []

        for b, pos_type in enumerate(position_types):
            # Render with the placeholder, then substitute the runtime slot
            # token string so the tokenizer turns it into a single id.
            prompt = render_av_prompt(pos_type).replace(
                AV_SLOT_PLACEHOLDER, self.cfg.slot_token_str
            )
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            try:
                slot_pos = prompt_ids.index(slot_id)
            except ValueError as e:
                raise ValueError(
                    "Slot token id not found after tokenizing AV prompt; "
                    f"slot_id={slot_id} prompt={prompt!r}"
                ) from e
            row_labels = [-100] * len(prompt_ids)

            if target_texts is not None:
                tgt = self.tokenizer.encode(" " + target_texts[b], add_special_tokens=False)
                if self.tokenizer.eos_token_id is not None:
                    tgt = tgt + [self.tokenizer.eos_token_id]
                row_ids = prompt_ids + tgt
                row_labels = row_labels + tgt
            else:
                row_ids = prompt_ids

            encoded_ids.append(row_ids)
            encoded_labels.append(row_labels)
            slot_indices.append(slot_pos)

        T = max(len(r) for r in encoded_ids)
        B = len(encoded_ids)
        input_ids = torch.full((B, T), self._pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
        labels = torch.full((B, T), -100, dtype=torch.long, device=device)
        for b, (ids, lbl) in enumerate(zip(encoded_ids, encoded_labels)):
            n = len(ids)
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[b, :n] = 1
            labels[b, :n] = torch.tensor(lbl, dtype=torch.long, device=device)
        return input_ids, attention_mask, labels, slot_indices

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

    # ------------------------------------------------------------------ public

    def forward_sft(
        self,
        activations: torch.Tensor,
        position_types: list[PositionType],
        target_texts: list[str],
    ):
        """Teacher-forced SFT: CE only over the target completion tokens."""
        assert activations.shape[0] == len(position_types) == len(target_texts)
        input_ids, attention_mask, labels, slot_indices = self._tokenize_prompts(
            position_types, target_texts, device=activations.device,
        )
        embeds = self._embed_with_injection(input_ids, attention_mask, slot_indices, activations)
        out = self.base(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        return out

    @torch.no_grad()
    def generate(
        self,
        activations: torch.Tensor,
        position_types: list[PositionType],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        return_logprobs: bool = False,
    ) -> dict:
        """Sampling rollout used both for inference and GRPO."""
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = temperature if temperature is not None else self.cfg.generation_temperature
        top_p = top_p if top_p is not None else self.cfg.generation_top_p

        input_ids, attention_mask, _, slot_indices = self._tokenize_prompts(
            position_types, target_texts=None, device=activations.device,
        )
        embeds = self._embed_with_injection(input_ids, attention_mask, slot_indices, activations)

        gen_out = self.base.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=return_logprobs,
            pad_token_id=self._pad_id,
        )
        # When `inputs_embeds` is the only input, HF returns only the newly
        # generated token ids (no prompt prefix). Verified across recent
        # transformers; future-proof by trimming to last `max_new_tokens`.
        gen_ids = gen_out.sequences
        if gen_ids.shape[1] > max_new_tokens:
            gen_ids = gen_ids[:, -max_new_tokens:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        result: dict = {"text": texts, "token_ids": gen_ids}
        if return_logprobs and getattr(gen_out, "scores", None) is not None:
            logprobs = []
            for step, score in enumerate(gen_out.scores):
                step_logp = torch.log_softmax(score, dim=-1)
                tok = gen_ids[:, step]
                logprobs.append(step_logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1))
            result["logprobs"] = torch.stack(logprobs, dim=1)
        return result

    def score_tokens(
        self,
        activations: torch.Tensor,
        position_types: list[PositionType],
        gen_token_ids: torch.Tensor,
        gen_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable per-token log probs of ``gen_token_ids`` under this AV.

        Used by GRPO to compute the policy gradient term (with grad) and the
        KL anchor (no grad). Activation injection follows the same single-slot
        recipe as ``forward_sft`` / ``generate``.

        Args:
            activations:        ``(B, H)`` raw activation vectors.
            position_types:     list of length ``B`` of position type strings.
            gen_token_ids:      ``(B, T_gen)`` generated token ids, right-padded
                                with ``pad_id`` past each row's gen length.
            gen_attention_mask: ``(B, T_gen)`` 1 where a real generated token
                                lives, 0 on padding (and 0 after EOS for
                                consistency with rollout masking).

        Returns:
            ``(B, T_gen)`` log probs in float32. Positions where
            ``gen_attention_mask == 0`` are filled with 0 (caller masks again
            when computing loss / KL).
        """
        device = activations.device
        B = activations.shape[0]
        assert B == len(position_types) == gen_token_ids.shape[0]
        T_gen = gen_token_ids.shape[1]
        slot_id = self.cfg.slot_token_id

        # Per-row build of [prompt_ids; gen_ids_real]. Prompt length varies
        # because position_type strings tokenize to different lengths.
        rows_ids: list[list[int]] = []
        rows_slot_pos: list[int] = []
        rows_score_start: list[int] = []
        rows_gen_lens: list[int] = []
        for b in range(B):
            prompt = render_av_prompt(position_types[b]).replace(
                AV_SLOT_PLACEHOLDER, self.cfg.slot_token_str
            )
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            slot_pos = prompt_ids.index(slot_id)
            gen_len = int(gen_attention_mask[b].sum().item())
            gen_real = gen_token_ids[b, :gen_len].tolist() if gen_len > 0 else []
            row = prompt_ids + gen_real
            rows_ids.append(row)
            rows_slot_pos.append(slot_pos)
            rows_score_start.append(len(prompt_ids))
            rows_gen_lens.append(gen_len)

        T_max = max(len(r) for r in rows_ids)
        input_ids = torch.full((B, T_max), self._pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
        for b, row in enumerate(rows_ids):
            n = len(row)
            input_ids[b, :n] = torch.tensor(row, dtype=torch.long, device=device)
            attention_mask[b, :n] = 1

        embeds = self._embed_with_injection(input_ids, attention_mask, rows_slot_pos, activations)
        out = self.base(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = out.logits  # (B, T_max, V)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        starts = torch.tensor(rows_score_start, device=device, dtype=torch.long)
        gen_lens_t = torch.tensor(rows_gen_lens, device=device, dtype=torch.long)
        t_arange = torch.arange(T_gen, device=device, dtype=torch.long).unsqueeze(0)
        # logit at position (start-1+t) predicts the token at (start+t).
        src_pos = (starts.unsqueeze(1) - 1 + t_arange).clamp(min=0, max=T_max - 1)
        tgt_pos = (starts.unsqueeze(1) + t_arange).clamp(max=T_max - 1)

        vocab = log_probs.shape[-1]
        gathered_logits = log_probs.gather(
            1, src_pos.unsqueeze(-1).expand(-1, -1, vocab)
        )  # (B, T_gen, V)
        gathered_tokens = input_ids.gather(1, tgt_pos)  # (B, T_gen)
        logp = gathered_logits.gather(-1, gathered_tokens.unsqueeze(-1)).squeeze(-1)
        mask = (t_arange < gen_lens_t.unsqueeze(1)).to(logp.dtype)
        return logp * mask

    # ------------------------------------------------------------------ save/load

    def save(self, path: str | Path) -> None:
        """Save LoRA adapters and the activation projector."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # PEFT wrappers have save_pretrained; bare models do too (full weights).
        self.base.save_pretrained(str(path))
        torch.save(self.act_proj.state_dict(), str(path / "act_proj.pt"))
        # Persist config (excluding the resolved runtime fields that may drift).
        cfg = {**self.cfg.__dict__}
        cfg["lora_targets"] = list(cfg["lora_targets"])
        cfg["reserved_token_candidates"] = list(cfg["reserved_token_candidates"])
        import json
        (path / "av_config.json").write_text(json.dumps(cfg, indent=2))

    def load_act_proj(self, path: str | Path) -> None:
        sd = torch.load(str(Path(path) / "act_proj.pt"), map_location="cpu")
        self.act_proj.load_state_dict(sd)


# ----------------------------------------------------------------------------
# Helpers (kept here so the module is hermetic; tests can monkey-patch them).
# ----------------------------------------------------------------------------

def _load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def _load_causal_lm(name: str, dtype: str):
    from transformers import AutoModelForCausalLM
    torch_dtype = getattr(torch, dtype)
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch_dtype, trust_remote_code=True
    )


def _hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError("Model has no `config`; cannot infer hidden size.")
    for attr in ("hidden_size", "n_embd"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    # Qwen3VL etc. may nest a text_config.
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    raise AttributeError(f"Cannot find hidden_size on model config: {cfg!r}")


def _wrap_lora(model, cfg: AVConfig):
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)

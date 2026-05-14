"""Load AV/AR back from an SFT checkpoint directory.

The SFT trainer (``nla.training.sft.run_sft``) writes::

    <ckpt_dir>/av/
      adapter_config.json
      adapter_model.safetensors      # LoRA + resized embed_tokens
      act_proj.pt
      av_config.json
    <ckpt_dir>/ar/
      adapter_config.json
      adapter_model.safetensors      # LoRA (truncated stack)
      head.pt
      ar_config.json

These helpers re-construct an ``ActivationVerbalizer`` / ``ActivationReconstructor``
from those files, optionally freezing them (for use as a GRPO reference policy
or as the reward model).
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch

from nla.models import (
    ActivationReconstructor,
    ActivationVerbalizer,
    ARConfig,
    AVConfig,
)


def _coerce_cfg(cfg_cls, raw: dict):
    """Drop unknown keys and coerce list→tuple where the dataclass expects tuples."""
    known = {f.name: f for f in fields(cfg_cls)}
    out = {}
    for k, v in raw.items():
        if k not in known:
            continue
        ftype = known[k].type
        # Field annotations come through as strings in dataclass-from-__future__ files;
        # we special-case the two tuple fields we know about.
        if k in ("lora_targets", "reserved_token_candidates") and isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return cfg_cls(**out)


def load_av_from_sft(
    ckpt_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    freeze: bool = False,
) -> ActivationVerbalizer:
    """Reconstruct an AV from a saved SFT checkpoint directory.

    Args:
        ckpt_dir: Path to the ``av/`` subdir of an SFT run.
        device:   Device to move the loaded AV onto.
        freeze:   If ``True``, set all params to ``requires_grad=False`` and put
                  the module in eval mode. Used for the GRPO reference policy.

    Notes:
        - The saved adapter includes a resized ``embed_tokens`` layer (PEFT's
          ``save_embedding_layers=True`` was triggered when we added
          ``<|act_slot|>``). On load we re-add the same slot token to the
          tokenizer + resize the base embeddings before wrapping with the saved
          adapter so the shapes match.
        - The activation projector (``act_proj``) is restored from ``act_proj.pt``.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg_raw = json.loads((ckpt_dir / "av_config.json").read_text())
    cfg = _coerce_cfg(AVConfig, cfg_raw)

    # Build AV with the same config but *without* applying random LoRA — we
    # want the saved adapter via PeftModel.from_pretrained instead. The slot
    # token + embedding resize is part of __init__ (apply_lora arg gates only
    # the LoRA wrapping).
    av = ActivationVerbalizer(cfg, apply_lora=False)

    from peft import PeftModel
    av.base = PeftModel.from_pretrained(
        av.base, str(ckpt_dir), is_trainable=not freeze,
    )
    av.load_act_proj(ckpt_dir)
    av = av.to(device)

    if freeze:
        for p in av.parameters():
            p.requires_grad = False
        av.eval()
    return av


def load_ar_from_sft(
    ckpt_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    freeze: bool = False,
) -> ActivationReconstructor:
    """Reconstruct an AR from a saved SFT checkpoint directory.

    See :func:`load_av_from_sft` for behavior; this is the mirror for the
    reconstructor (LoRA on a truncated stack + an affine head).
    """
    ckpt_dir = Path(ckpt_dir)
    cfg_raw = json.loads((ckpt_dir / "ar_config.json").read_text())
    cfg = _coerce_cfg(ARConfig, cfg_raw)

    ar = ActivationReconstructor(cfg, apply_lora=False)

    from peft import PeftModel
    ar.base = PeftModel.from_pretrained(
        ar.base, str(ckpt_dir), is_trainable=not freeze,
    )
    ar.load_head(ckpt_dir)
    ar = ar.to(device)

    if freeze:
        for p in ar.parameters():
            p.requires_grad = False
        ar.eval()
    return ar

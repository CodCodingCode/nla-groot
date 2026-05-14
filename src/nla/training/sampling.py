"""Per-token position sampling for NLA training.

Per plan §5.2 (the change that follows Grant et al. 2026): we sample ONE token
position per activation example rather than mean-pooling. The position is drawn
from a mixture:

    40%  last_text   - last non-pad, non-image token
    40%  image_patch - uniform random over image-token positions
    20%  anchor      - the **last valid (non-pad) sequence token**, which may
                       be an image-patch token if the sequence ends in vision.
                       This matches ``nla.extraction.sampler._anchor_index`` so
                       labeled positions, training samples, and stratified FVE
                       all refer to the same token role.

This module operates on already-extracted activations from disk. Each example
has shape `[T, H]` with `attention_mask[T]` and `image_mask[T]` companions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from nla.extraction.sampler import _anchor_index
from nla.layer_spec import POSITION_MIX


@dataclass
class PerTokenBatch:
    """One training batch of per-token activations."""

    activations: torch.Tensor       # [B, H]
    position_type: list[str]        # length B: "last_text" | "image_patch" | "anchor"
    position_index: torch.Tensor    # [B] absolute token index for traceability
    seq_len: torch.Tensor           # [B] total non-pad length of the parent example
    example_id: list[str]           # length B for traceability


class TokenPositionSampler:
    """Stateful sampler with reproducible per-step seeding."""

    def __init__(self, position_mix: dict[str, float] | None = None, seed: int = 0):
        self.mix = position_mix or POSITION_MIX
        # Normalize and accumulate.
        total = sum(self.mix.values())
        self._cum = []
        self._labels = []
        acc = 0.0
        for k, v in self.mix.items():
            acc += v / total
            self._cum.append(acc)
            self._labels.append(k)
        self._rng = random.Random(seed)

    def _draw_position_type(self) -> str:
        r = self._rng.random()
        for label, c in zip(self._labels, self._cum):
            if r <= c:
                return label
        return self._labels[-1]

    def sample(
        self,
        attention_mask: torch.Tensor,   # [T] bool
        image_mask: torch.Tensor,        # [T] bool
        force_type: str | None = None,
    ) -> tuple[str, int]:
        """Return (position_type, index) for one example.

        When the requested type is empty (no image tokens etc.) we fall back
        deterministically: image_patch → last_text → anchor → 0.
        """
        T = int(attention_mask.numel())
        real = attention_mask.bool().tolist()
        is_image = image_mask.bool().tolist()
        text_positions = [i for i in range(T) if real[i] and not is_image[i]]
        image_positions = [i for i in range(T) if real[i] and is_image[i]]

        ptype = force_type or self._draw_position_type()
        if ptype == "image_patch" and image_positions:
            return "image_patch", self._rng.choice(image_positions)
        if ptype == "last_text" and text_positions:
            return "last_text", text_positions[-1]
        if ptype == "anchor":
            anchor_idx = _anchor_index(attention_mask)
            if anchor_idx is not None:
                return "anchor", anchor_idx
        # Fallbacks
        if text_positions:
            return "last_text", text_positions[-1]
        if image_positions:
            return "image_patch", image_positions[0]
        return "anchor", 0


def sample_token_position(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    rng: random.Random | None = None,
    position_mix: dict[str, float] | None = None,
    force_type: str | None = None,
) -> tuple[str, int]:
    """Functional convenience wrapper around `TokenPositionSampler.sample`."""
    sampler = TokenPositionSampler(position_mix=position_mix, seed=0)
    if rng is not None:
        sampler._rng = rng
    return sampler.sample(attention_mask, image_mask, force_type=force_type)

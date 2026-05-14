"""Position sampling for per-token NLA training.

Per §5.2 of the plan, each training example samples *one* position from a fixed
mixture (POSITION_MIX in layer_spec):

    last_text   : the last non-image, non-pad text token in the prompt
    image_patch : a uniform random image-patch token
    anchor      : a designated anchor position (e.g. final non-pad token,
                  which is typically the EOS / assistant-turn token)

For examples that lack one of the categories (e.g. an image-only context has
no last_text token), the sampler falls back to the next available category in
priority order: anchor -> last_text -> image_patch -> any valid token.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import torch

from nla.layer_spec import POSITION_MIX


class PositionType(str, Enum):
    LAST_TEXT = "last_text"
    IMAGE_PATCH = "image_patch"
    ANCHOR = "anchor"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class SampledPosition:
    """One sampled token position within an example."""

    index: int
    type: PositionType


def _last_text_index(attention_mask: torch.Tensor, image_mask: torch.Tensor) -> int | None:
    """Return the position of the last non-image, non-pad token, else None."""
    valid = attention_mask.bool() & ~image_mask.bool()
    nz = torch.nonzero(valid, as_tuple=False).flatten()
    if nz.numel() == 0:
        return None
    return int(nz[-1].item())


def _anchor_index(attention_mask: torch.Tensor) -> int | None:
    """Final non-pad token (paper's 'anchor' for variable T sequences)."""
    nz = torch.nonzero(attention_mask.bool(), as_tuple=False).flatten()
    if nz.numel() == 0:
        return None
    return int(nz[-1].item())


def _image_patch_index(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    rng: np.random.Generator,
) -> int | None:
    """Uniform random over image-patch positions that are also non-pad."""
    valid = attention_mask.bool() & image_mask.bool()
    nz = torch.nonzero(valid, as_tuple=False).flatten()
    if nz.numel() == 0:
        return None
    return int(nz[int(rng.integers(0, nz.numel()))].item())


def sample_position(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    mix: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> SampledPosition:
    """Sample one position according to POSITION_MIX with graceful fallback."""
    mix = mix or POSITION_MIX
    rng = rng or np.random.default_rng()

    keys = list(mix.keys())
    weights = np.asarray([mix[k] for k in keys], dtype=np.float64)
    weights = weights / weights.sum()
    chosen = keys[int(rng.choice(len(keys), p=weights))]

    if chosen == "last_text":
        idx = _last_text_index(attention_mask, image_mask)
        if idx is not None:
            return SampledPosition(idx, PositionType.LAST_TEXT)
    elif chosen == "image_patch":
        idx = _image_patch_index(attention_mask, image_mask, rng)
        if idx is not None:
            return SampledPosition(idx, PositionType.IMAGE_PATCH)
    elif chosen == "anchor":
        idx = _anchor_index(attention_mask)
        if idx is not None:
            return SampledPosition(idx, PositionType.ANCHOR)

    # Fallback cascade: try anchor -> last_text -> image_patch -> any valid token.
    for fn in (
        lambda: (_anchor_index(attention_mask), PositionType.ANCHOR),
        lambda: (_last_text_index(attention_mask, image_mask), PositionType.LAST_TEXT),
        lambda: (_image_patch_index(attention_mask, image_mask, rng), PositionType.IMAGE_PATCH),
    ):
        idx, pt = fn()
        if idx is not None:
            return SampledPosition(idx, PositionType.FALLBACK if pt != PositionType.ANCHOR else pt)

    raise ValueError(
        "No valid token positions in example: attention_mask is all False. "
        "This usually means the example was incorrectly padded."
    )


def iter_image_positions(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
) -> Sequence[int]:
    """Return every image-patch position (used by the spatial-NLA-map experiment)."""
    valid = attention_mask.bool() & image_mask.bool()
    return torch.nonzero(valid, as_tuple=False).flatten().tolist()


def sample_positions(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    n: int = 1,
    mix: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    no_replacement: bool = True,
) -> list[SampledPosition]:
    """Sample ``n`` distinct positions following ``POSITION_MIX`` with fallback.

    With ``no_replacement=True`` (default) we resample if the next pick lands on
    an already-chosen index. This avoids paying for duplicate API calls when
    labeling. After ``2 * n`` retries we give up on uniqueness for that draw and
    accept whatever index came out (so we always return exactly ``n`` items).
    """
    rng = rng or np.random.default_rng()
    chosen_indices: set[int] = set()
    out: list[SampledPosition] = []
    max_tries = max(8, 2 * n)
    for _ in range(n):
        for attempt in range(max_tries):
            pos = sample_position(attention_mask, image_mask, mix=mix, rng=rng)
            if (not no_replacement) or pos.index not in chosen_indices:
                chosen_indices.add(pos.index)
                out.append(pos)
                break
        else:
            # Give up uniqueness for this draw.
            pos = sample_position(attention_mask, image_mask, mix=mix, rng=rng)
            chosen_indices.add(pos.index)
            out.append(pos)
    return out

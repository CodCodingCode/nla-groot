"""Read-time pooling strategies over the [T, H] on-disk activation format.

These functions take an already-extracted ``features: [T, H]`` tensor plus
the associated ``attention_mask`` and ``image_mask``, and return one or
more ``[H]`` vectors per example. They are the V4 A/B sweep's lever for
exploring alternatives to the single-random-image-patch sampling that
V3 SFT used.

Why pool at read-time, not at extraction time
---------------------------------------------
The on-disk shard stores the full ``[T, H]`` block per example (see
``src/nla/extraction/storage.py`` and ``BackboneFeatureHook`` in
``hook.py``). Different pooling strategies are therefore just numpy/torch
indexing over the same data, no re-extraction required. The Stage-2
sweep can iterate all four strategies in one pass over the existing
layer-16 activations; only the layer axis (8, 12) needs new extraction.

Contract
--------
Each strategy takes:

  * ``features``: ``Tensor[T, H]`` (float, already on the right device).
  * ``image_mask``: ``Tensor[T]`` boolean, ``True`` at image-patch tokens.
  * ``attention_mask``: ``Tensor[T]`` boolean, ``True`` at non-pad tokens.

Single-vector strategies return ``Tensor[H]``. Multi-vector strategies
(currently ``strided_image`` with ``k > 1``) return ``Tensor[K, H]``
so the caller can either average those K vectors before downstream use
or fan them out as K separate proxy-eval rows. For the V4 sweep we
collapse multi-vector strategies to a single mean vector inside the
caller — keeps the comparison apples-to-apples.

All strategies operate on tensors but the implementation is
numpy-equivalent (no autograd ops) so feeding numpy arrays via
``torch.as_tensor`` is safe.

References
----------
- Plan: ``.cursor/plans/v4_image-patch_a_b_sweep_628ee13b.plan.md``
  Stage 1a.
- The control strategy mirrors ``_image_patch_index`` in
  ``src/nla/extraction/sampler.py`` so a "random_one" run reproduces
  the V3 baseline up to RNG.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


# Type alias for clarity; all strategies are functions that take three
# tensors (+ optional RNG / k) and return one [H] or [K, H] tensor.
PositionStrategy = Callable[..., torch.Tensor]


def _ensure_bool(mask: torch.Tensor) -> torch.Tensor:
    return mask.bool() if mask.dtype != torch.bool else mask


def _image_indices(image_mask: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Indices of valid (non-pad) image-patch tokens, as int64 1-D tensor."""
    valid = _ensure_bool(image_mask) & _ensure_bool(attention_mask)
    return torch.nonzero(valid, as_tuple=False).flatten().to(torch.int64)


# ---------------------------------------------------------------------------
# Single-vector strategies.
# ---------------------------------------------------------------------------

def random_one(
    features: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Control: uniform-random single image-patch token (V3 behaviour).

    Returns the ``[H]`` activation at one randomly chosen image-patch
    position. Falls back to the last attended token if no image patches
    are visible (rare for LIBERO — the policy frame always has a camera
    crop — but kept for robustness).
    """
    rng = rng or np.random.default_rng()
    idx = _image_indices(image_mask, attention_mask)
    if idx.numel() == 0:
        # Fallback: last attended token.
        nz = torch.nonzero(_ensure_bool(attention_mask), as_tuple=False).flatten()
        if nz.numel() == 0:
            raise ValueError("random_one: example has no attended tokens")
        return features[int(nz[-1].item())]
    pick = int(rng.integers(0, idx.numel()))
    return features[int(idx[pick].item())]


def mean_pool_image(
    features: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    **_unused,
) -> torch.Tensor:
    """Mean-pool every valid image-patch token into a single ``[H]`` vector.

    The cheapest "use all the spatial signal" baseline: equal weight on
    every patch token. If a model's image-patch grid is N×N then this is
    equivalent to global average pooling over the visual feature map.
    """
    idx = _image_indices(image_mask, attention_mask)
    if idx.numel() == 0:
        raise ValueError("mean_pool_image: no image-patch tokens in example")
    return features.index_select(0, idx).mean(dim=0)


def center_image(
    features: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    **_unused,
) -> torch.Tensor:
    """Pick the geometric center patch from the image-token block.

    Image-patch tokens are written in row-major raster order by Qwen3-VL
    (and the GR00T wrapper preserves that order; see
    ``src/nla/extraction/hook.py``). So the center patch is the patch at
    position ``len(image_indices) // 2``. This is a coarse but
    interpretable single-vector baseline that captures a fixed
    "I'm looking at the middle of the scene" signal across episodes.
    """
    idx = _image_indices(image_mask, attention_mask)
    if idx.numel() == 0:
        raise ValueError("center_image: no image-patch tokens in example")
    center = idx[int(idx.numel() // 2)]
    return features[int(center.item())]


def strided_image(
    features: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    k: int = 4,
    **_unused,
) -> torch.Tensor:
    """Pick ``k`` evenly-spaced image-patch tokens and return their mean.

    Trade-off between ``random_one`` (one noisy sample) and
    ``mean_pool_image`` (all patches; can blur object signal).
    With ``k=4`` we capture roughly the four image quadrants on a
    standard 14×14 patch grid.

    Returns a single ``[H]`` vector (the mean of the ``k`` selected
    patches) so it is drop-in comparable with the other strategies.
    """
    if k <= 0:
        raise ValueError(f"strided_image: k must be >= 1, got {k}")
    idx = _image_indices(image_mask, attention_mask)
    if idx.numel() == 0:
        raise ValueError("strided_image: no image-patch tokens in example")
    n = int(idx.numel())
    if n <= k:
        # Fewer patches than stride width; fall back to mean over all.
        return features.index_select(0, idx).mean(dim=0)
    # Evenly spaced including endpoints. linspace -> nearest int index.
    picks = torch.linspace(0, n - 1, steps=k).round().to(torch.int64)
    selected = idx.index_select(0, picks)
    return features.index_select(0, selected).mean(dim=0)


def strided_image_multi(
    features: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    k: int = 8,
    **_unused,
) -> torch.Tensor:
    """Same index selection as ``strided_image`` but returns ``[K, H]``.

    Used by V5 to give AV the strided patch grid as K separate slot
    activations instead of a single mean-pooled vector. The trailing
    ``.mean(dim=0)`` from ``strided_image`` is *intentionally* omitted so the
    caller (the SFT dataset) can hand AV one slot per patch and feed the AR
    a separately-pooled single ``[H]`` vector.

    When fewer than ``k`` image-patch tokens are present (rare for LIBERO
    frames but defensive for short prompts), the available patches are
    returned as-is with shape ``[n_available, H]``; downstream collate code
    must handle that variable K or filter such rows.
    """
    if k <= 0:
        raise ValueError(f"strided_image_multi: k must be >= 1, got {k}")
    idx = _image_indices(image_mask, attention_mask)
    if idx.numel() == 0:
        raise ValueError("strided_image_multi: no image-patch tokens in example")
    n = int(idx.numel())
    if n <= k:
        return features.index_select(0, idx)
    picks = torch.linspace(0, n - 1, steps=k).round().to(torch.int64)
    selected = idx.index_select(0, picks)
    return features.index_select(0, selected)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, PositionStrategy] = {
    "random_one":          random_one,
    "mean_pool_image":     mean_pool_image,
    "strided_image":       strided_image,
    "strided_image_multi": strided_image_multi,
    "center_image":        center_image,
}


def apply(
    name: str,
    features: torch.Tensor,
    image_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    rng: np.random.Generator | None = None,
    k: int = 4,
) -> torch.Tensor:
    """Look up a strategy by name and apply it.

    Centralizes the call site so the proxy-eval / sweep scripts can drive
    the strategy axis by string from CLI.
    """
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown position strategy '{name}'. Known: {sorted(STRATEGIES)}"
        )
    fn = STRATEGIES[name]
    if name == "random_one":
        return fn(features, image_mask, attention_mask, rng=rng)
    if name in ("strided_image", "strided_image_multi"):
        return fn(features, image_mask, attention_mask, k=k)
    return fn(features, image_mask, attention_mask)

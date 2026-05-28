"""Patch GR00T Qwen3 backbone outputs with NLA reconstruction vectors.

Training stores one activation per *token*; at inference the backbone returns a
full sequence ``[B, T, H]``. This module applies a steer vector to one position
(or all image tokens) so you can probe *causal* effects on the action head.

Intended use: AR maps your task language (same bullet style as ``labels.jsonl``)
to ``ĥ`` in backbone space; optionally blend with the live forward::

    h'[t] = (1 - λ) * h[t] + λ * ĥ

λ=1 is a hard replace (default). See ``scripts/eval/nla_steer_groot_action.py``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import torch

from nla.extraction.sampler import (
    _anchor_index,
    _image_patch_index,
    _last_text_index,
    iter_image_positions,
)
from nla.layer_spec import BACKBONE_EMBEDDING_DIM

SteerPlacement = Literal[
    "last_text",
    "image_patch",
    "anchor",
    "image_patch_all",
    "image_patch_spatial",
    "image_patch_strided",
    "fixed",
]


@dataclass(frozen=True)
class SteerSpec:
    """Where to apply the steer vector along the time axis (batch row 0 only)."""

    placement: SteerPlacement
    blend: float = 1.0
    """1.0 = replace token activations with the steer vector; 0 = no-op."""
    fixed_token_index: int | None = None
    """When ``placement == "fixed"``, must be set."""
    image_patch_seed: int = 0
    """RNG seed for picking a random image token when ``placement=="image_patch"``."""
    strided_k: int = 0
    """Number of strided image-patch positions to use when placement=='image_patch_strided'.
    Must equal the K dim of the (K, H) steer vector. Mirrors how strided_image_multi
    sub-sampled the AV's input (k evenly-spaced indices via torch.linspace)."""


def resolve_steer_indices(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    spec: SteerSpec,
    *,
    batch_index: int = 0,
) -> list[int]:
    """Resolve token indices (into ``T``) to modify for one batch row."""
    if attention_mask.dim() != 2 or image_mask.dim() != 2:
        raise ValueError("attention_mask and image_mask must be [B, T]")
    row_attn = attention_mask[batch_index].detach()
    row_img = image_mask[batch_index].detach()

    if spec.placement == "fixed":
        if spec.fixed_token_index is None:
            raise ValueError("fixed_token_index is required when placement='fixed'")
        t = int(spec.fixed_token_index)
        if t < 0 or t >= int(row_attn.shape[0]):
            raise IndexError(f"fixed_token_index {t} out of bounds for T={row_attn.shape[0]}")
        return [t]

    if spec.placement == "image_patch_all":
        return [int(i) for i in iter_image_positions(row_attn, row_img)]

    if spec.placement == "image_patch_spatial":
        # Same indices as image_patch_all but ordered (the hook will place the
        # k-th provided steer vector at the k-th index, so order matters).
        return [int(i) for i in iter_image_positions(row_attn, row_img)]

    if spec.placement == "image_patch_strided":
        # Pick `strided_k` evenly-spaced image-patch indices (same scheme as
        # strided_image_multi on the AV input side). The hook places the k-th
        # provided steer vector at the k-th strided index; the other patch
        # positions are left untouched.
        if spec.strided_k <= 0:
            raise ValueError(
                "image_patch_strided placement requires strided_k > 0; "
                f"got {spec.strided_k}."
            )
        all_idx = [int(i) for i in iter_image_positions(row_attn, row_img)]
        n = len(all_idx)
        if n == 0:
            raise ValueError("No image_patch tokens for image_patch_strided.")
        k = int(spec.strided_k)
        if k > n:
            raise ValueError(
                f"strided_k={k} exceeds available image_patch tokens (n={n})."
            )
        picks = torch.linspace(0, n - 1, steps=k).round().to(torch.int64).tolist()
        return [all_idx[int(p)] for p in picks]

    if spec.placement == "last_text":
        idx = _last_text_index(row_attn, row_img)
        if idx is None:
            raise ValueError("No last_text token (no non-image attended positions).")
        return [idx]

    if spec.placement == "anchor":
        idx = _anchor_index(row_attn)
        if idx is None:
            raise ValueError("No anchor token (attention mask all false).")
        return [idx]

    if spec.placement == "image_patch":
        rng = np.random.default_rng(int(spec.image_patch_seed))
        idx = _image_patch_index(row_attn, row_img, rng)
        if idx is None:
            raise ValueError("No image_patch token.")
        return [idx]

    raise ValueError(f"Unknown placement {spec.placement!r}")


class BackboneFeaturesSteerHook:
    """Forward hook that rewrites ``output['backbone_features']`` after backbone."""

    def __init__(
        self,
        steer_vec: torch.Tensor,
        spec: SteerSpec,
        *,
        batch_index: int = 0,
    ) -> None:
        # image_patch_spatial and image_patch_strided expect a (K, H) grid;
        # everything else expects a flat (H,) vector that gets broadcast/picked.
        per_position_placements = ("image_patch_spatial", "image_patch_strided")
        if spec.placement in per_position_placements:
            if steer_vec.dim() == 3 and steer_vec.shape[0] == 1:
                steer_vec = steer_vec.squeeze(0)
            if steer_vec.dim() != 2:
                raise ValueError(
                    f"{spec.placement} requires steer_vec shape (K, H); "
                    f"got {tuple(steer_vec.shape)}"
                )
            if int(steer_vec.shape[1]) != BACKBONE_EMBEDDING_DIM:
                raise ValueError(
                    f"steer_vec hidden dim {steer_vec.shape[1]} != "
                    f"BACKBONE_EMBEDDING_DIM={BACKBONE_EMBEDDING_DIM}"
                )
            self._steer_cpu = steer_vec.detach().float().cpu().contiguous()
        else:
            if steer_vec.dim() == 2 and steer_vec.shape[0] == 1:
                steer_vec = steer_vec.squeeze(0)
            if steer_vec.dim() != 1:
                raise ValueError(f"steer_vec must be [H]; got shape {tuple(steer_vec.shape)}")
            if int(steer_vec.shape[0]) != BACKBONE_EMBEDDING_DIM:
                raise ValueError(
                    f"steer_vec dim {steer_vec.shape[0]} != BACKBONE_EMBEDDING_DIM={BACKBONE_EMBEDDING_DIM}"
                )
            self._steer_cpu = steer_vec.detach().float().cpu().contiguous()
        self.spec = spec
        self.batch_index = int(batch_index)
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def steer_vec_cpu(self) -> torch.Tensor:
        return self._steer_cpu

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        # Hugging Face BatchFeature is dict-like.
        feats = output["backbone_features"]
        attn = output["backbone_attention_mask"]
        img_m = output["image_mask"]

        idxs = resolve_steer_indices(attn, img_m, self.spec, batch_index=self.batch_index)

        steer = self._steer_cpu.to(device=feats.device, dtype=feats.dtype)
        blend = float(self.spec.blend)
        blend = max(0.0, min(1.0, blend))
        is_spatial = self.spec.placement in ("image_patch_spatial", "image_patch_strided")
        if is_spatial:
            if steer.dim() != 2:
                raise RuntimeError(
                    f"{self.spec.placement} expects 2D steer; got {tuple(steer.shape)}"
                )
            if steer.shape[0] != len(idxs):
                raise RuntimeError(
                    f"{self.spec.placement}: AR emitted {steer.shape[0]} vectors "
                    f"but resolved {len(idxs)} target token slots in this "
                    "forward. For image_patch_spatial, set ARConfig.spatial_n_positions "
                    "to match the live image_patch count. For image_patch_strided, "
                    "set SteerSpec.strided_k to match AR's K dim."
                )

        new_feats = feats.clone()
        bi = self.batch_index
        for k, t in enumerate(idxs):
            if blend <= 0.0:
                continue
            base = feats[bi, t]
            steer_k = steer[k] if is_spatial else steer
            if blend >= 1.0:
                new_feats[bi, t] = steer_k
            else:
                new_feats[bi, t] = (1.0 - blend) * base + blend * steer_k
        output["backbone_features"] = new_feats

    def clear(self) -> None:
        self._handle = None


class BatchedBackboneFeaturesSteerHook:
    """Apply a distinct steer vector to each batch row in one backbone forward."""

    def __init__(
        self,
        steer_vecs: Sequence[torch.Tensor],
        spec: SteerSpec,
    ) -> None:
        vecs: list[torch.Tensor] = []
        for v in steer_vecs:
            if v.dim() == 2 and v.shape[0] == 1:
                v = v.squeeze(0)
            if v.dim() != 1:
                raise ValueError(f"each steer_vec must be [H]; got {tuple(v.shape)}")
            if int(v.shape[0]) != BACKBONE_EMBEDDING_DIM:
                raise ValueError(
                    f"steer dim {v.shape[0]} != BACKBONE_EMBEDDING_DIM={BACKBONE_EMBEDDING_DIM}"
                )
            vecs.append(v.detach().float().cpu().contiguous())
        if not vecs:
            raise ValueError("steer_vecs must be non-empty")
        self._steer_cpu = vecs
        self.spec = spec
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        feats = output["backbone_features"]
        attn = output["backbone_attention_mask"]
        img_m = output["image_mask"]
        b = int(feats.shape[0])
        if len(self._steer_cpu) != b:
            raise ValueError(
                f"steer_vecs length {len(self._steer_cpu)} != batch size {b}"
            )
        blend = max(0.0, min(1.0, float(self.spec.blend)))
        new_feats = feats.clone()
        for bi in range(b):
            idxs = resolve_steer_indices(attn, img_m, self.spec, batch_index=bi)
            steer = self._steer_cpu[bi].to(device=feats.device, dtype=feats.dtype)
            for t in idxs:
                if blend <= 0.0:
                    continue
                base = feats[bi, t]
                if blend >= 1.0:
                    new_feats[bi, t] = steer
                else:
                    new_feats[bi, t] = (1.0 - blend) * base + blend * steer
        output["backbone_features"] = new_feats


@contextlib.contextmanager
def attach_backbone_steer(
    backbone: torch.nn.Module,
    steer_vec: torch.Tensor,
    spec: SteerSpec,
    *,
    batch_index: int = 0,
) -> Iterator[BackboneFeaturesSteerHook]:
    """Register :class:`BackboneFeaturesSteerHook` on ``backbone.forward``."""
    hook_impl = BackboneFeaturesSteerHook(steer_vec, spec, batch_index=batch_index)
    handle = backbone.register_forward_hook(hook_impl)
    hook_impl._handle = handle
    try:
        yield hook_impl
    finally:
        handle.remove()
        hook_impl._handle = None


@contextlib.contextmanager
def attach_backbone_steer_batched(
    backbone: torch.nn.Module,
    steer_vecs: Sequence[torch.Tensor],
    spec: SteerSpec,
) -> Iterator[BatchedBackboneFeaturesSteerHook]:
    """Register :class:`BatchedBackboneFeaturesSteerHook` on ``backbone.forward``."""
    hook_impl = BatchedBackboneFeaturesSteerHook(steer_vecs, spec)
    handle = backbone.register_forward_hook(hook_impl)
    hook_impl._handle = handle
    try:
        yield hook_impl
    finally:
        handle.remove()
        hook_impl._handle = None

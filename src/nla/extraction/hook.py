"""Forward hook on GR00T's Qwen3 backbone that captures per-token activations.

We hook the `Qwen3Backbone` wrapper module (NOT the raw Qwen3VL model). Its
forward returns a ``BatchFeature`` with:
    backbone_features:        [B, T, 2048]   (output of decoder layer SELECT_LAYER-1)
    backbone_attention_mask:  [B, T]  bool   (True for valid, non-pad tokens)
    image_mask:               [B, T]  bool   (True for image-patch tokens)

This is precisely the "what the VLM has committed to" representation that the
action head consumes before vlln/vl_self_attention.  Hooking the wrapper rather
than the raw HF model means we get the masks for free.

Usage::

    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

    backbone = ... # type: Qwen3Backbone
    hook = BackboneFeatureHook()
    with attach_hooks(backbone, hook):
        _ = backbone(batch_feature)
    captured = hook.last  # CapturedActivation

The hook is a CPU-bound capture: tensors are detached and (optionally) moved to
CPU before being stored, so the GPU memory footprint is bounded by one forward.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from nla.layer_spec import BACKBONE_EMBEDDING_DIM


@dataclass
class CapturedActivation:
    """One forward's worth of captured activations + masks.

    All tensors are detached from the autograd graph. Shapes:
        features:        [B, T, hidden_size]  float (caller-chosen dtype)
        attention_mask:  [B, T]                bool
        image_mask:      [B, T]                bool
    """

    features: torch.Tensor
    attention_mask: torch.Tensor
    image_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    @property
    def seq_len(self) -> int:
        return self.features.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.features.shape[2]


class BackboneFeatureHook:
    """Captures the output BatchFeature of a Qwen3Backbone forward.

    The hook stores only the most recent capture (``self.last``); call ``clear``
    if you want to be defensive about cross-batch leaks.  It is safe to keep
    attached across many forwards.
    """

    def __init__(
        self,
        *,
        to_cpu: bool = True,
        store_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        """
        Args:
            to_cpu: If True, move captured tensors to CPU immediately. Set False
                if you intend to consume them within the same step on GPU.
            store_dtype: Optional dtype to cast features to before storing.
                Activations are typically computed in bf16; we promote to fp32
                by default for downstream norm computation precision.  Masks are
                always stored as bool regardless of this setting.
        """
        self.to_cpu = to_cpu
        self.store_dtype = store_dtype
        self.last: CapturedActivation | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        """Module-style forward hook signature (positional, no return)."""
        # Qwen3Backbone.forward returns a BatchFeature (a HF mapping). We support
        # any object exposing dict-style access for robustness.
        features = output["backbone_features"]
        attention_mask = output["backbone_attention_mask"]
        image_mask = output["image_mask"]

        assert features.ndim == 3, (
            f"Expected backbone_features of shape [B, T, H], got {tuple(features.shape)}"
        )
        assert features.shape[-1] == BACKBONE_EMBEDDING_DIM, (
            f"Expected hidden_size={BACKBONE_EMBEDDING_DIM}, got {features.shape[-1]}. "
            "Check that layer_spec.BACKBONE_EMBEDDING_DIM matches your checkpoint."
        )
        assert attention_mask.shape == features.shape[:2], (
            f"attention_mask shape {tuple(attention_mask.shape)} doesn't match "
            f"features {tuple(features.shape[:2])}"
        )
        assert image_mask.shape == features.shape[:2], (
            f"image_mask shape {tuple(image_mask.shape)} doesn't match "
            f"features {tuple(features.shape[:2])}"
        )

        f = features.detach()
        if self.store_dtype is not None:
            f = f.to(dtype=self.store_dtype)
        if self.to_cpu:
            f = f.cpu()
            attention_mask = attention_mask.detach().cpu()
            image_mask = image_mask.detach().cpu()
        else:
            attention_mask = attention_mask.detach()
            image_mask = image_mask.detach()

        self.last = CapturedActivation(
            features=f.contiguous(),
            attention_mask=attention_mask.to(torch.bool).contiguous(),
            image_mask=image_mask.to(torch.bool).contiguous(),
        )

    def clear(self) -> None:
        self.last = None


@contextlib.contextmanager
def attach_hooks(
    module: nn.Module,
    hook: BackboneFeatureHook,
) -> Iterator[BackboneFeatureHook]:
    """Context manager that registers ``hook`` on ``module``'s forward.

    The hook fires on the *full* Qwen3Backbone wrapper (not on individual layers),
    so we capture the wrapper's BatchFeature output exactly once per forward.

    On exit the handle is removed even if an exception is raised.
    """
    handle = module.register_forward_hook(hook)
    hook._handle = handle
    try:
        yield hook
    finally:
        handle.remove()
        hook._handle = None


# ---------------------------------------------------------------------------
# Intermediate-layer capture (V4 image-patch A/B sweep, layer axis).
# ---------------------------------------------------------------------------

class IntermediateLayerHook:
    """Capture the output of a *specific* Qwen3 decoder layer (e.g. layer 8 or 12).

    The GR00T ``Qwen3Backbone`` wrapper truncates layers below
    ``SELECT_LAYER`` at construction time and only ever returns the final
    layer's hidden states in ``backbone_features``. To probe layers
    8 / 12 without editing vendored GR00T code we register a forward hook
    on the matching ``DecoderLayer`` module directly:

        backbone.model.language_model.layers[layer_idx]

    The decoder layer's forward returns ``(hidden_states, ...)`` (tuple)
    in HF Transformers >= 4.40; this hook handles both the tuple and
    bare-tensor cases so it survives small upstream API drift.

    For the V4 sweep we pair this with a parallel ``BackboneFeatureHook``
    on the wrapper to inherit ``attention_mask`` and ``image_mask``
    (which depend only on ``input_ids``, so they're identical across
    layers).

    Usage::

        wrapper_hook   = BackboneFeatureHook()
        layer8_hook    = IntermediateLayerHook(layer_idx=8)
        layer12_hook   = IntermediateLayerHook(layer_idx=12)
        with attach_hooks(backbone, wrapper_hook), \\
             layer8_hook.attach(backbone), \\
             layer12_hook.attach(backbone):
            _ = backbone(batch_feature)
        h_layer8  = layer8_hook.last     # [B, T, H]
        h_layer16 = wrapper_hook.last.features

    Notes
    -----
    * The hook is attached to the *decoder block* (post-MLP residual
      output), not the attention sub-block — this matches what
      ``hidden_states[k]`` would have been if GR00T didn't truncate
      layers.
    * Captured tensors are detached + (optionally) moved to CPU + cast
      to ``store_dtype`` to mirror ``BackboneFeatureHook`` behavior.
    """

    def __init__(
        self,
        layer_idx: int,
        *,
        to_cpu: bool = True,
        store_dtype: torch.dtype | None = torch.float32,
    ) -> None:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        self.layer_idx = int(layer_idx)
        self.to_cpu = to_cpu
        self.store_dtype = store_dtype
        self.last: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        assert hidden_states.ndim == 3, (
            f"IntermediateLayerHook(layer_idx={self.layer_idx}): expected "
            f"hidden_states of shape [B, T, H], got {tuple(hidden_states.shape)}"
        )

        h = hidden_states.detach()
        if self.store_dtype is not None:
            h = h.to(dtype=self.store_dtype)
        if self.to_cpu:
            h = h.cpu()
        self.last = h.contiguous()

    def clear(self) -> None:
        self.last = None

    def _resolve_layer_module(self, backbone: nn.Module) -> nn.Module:
        """Walk ``backbone.model.language_model.layers[layer_idx]``.

        Kept as a small helper so the lookup path is centralized; if the
        GR00T wrapper's attribute tree changes we only edit one place.
        """
        try:
            layers = backbone.model.language_model.layers  # type: ignore[attr-defined]
        except AttributeError as e:
            raise AttributeError(
                "IntermediateLayerHook: could not find "
                "backbone.model.language_model.layers; check that the "
                "module passed in is a Qwen3Backbone wrapper."
            ) from e
        n = len(layers)
        if self.layer_idx >= n:
            raise IndexError(
                f"IntermediateLayerHook: layer_idx={self.layer_idx} but the "
                f"backbone only has {n} layers (likely because the wrapper "
                f"truncated above SELECT_LAYER). "
                "Load the wrapper with select_layer >= layer_idx + 1."
            )
        return layers[self.layer_idx]

    @contextlib.contextmanager
    def attach(self, backbone: nn.Module) -> Iterator["IntermediateLayerHook"]:
        """Register on ``backbone.model.language_model.layers[layer_idx]``.

        Composes cleanly with ``attach_hooks`` for the wrapper:

            with attach_hooks(backbone, wrap_hook), layer_hook.attach(backbone):
                ...
        """
        layer = self._resolve_layer_module(backbone)
        handle = layer.register_forward_hook(self)
        self._handle = handle
        try:
            yield self
        finally:
            handle.remove()
            self._handle = None

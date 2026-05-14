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

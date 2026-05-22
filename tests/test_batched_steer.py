"""Tests for multi-row backbone steering."""

from __future__ import annotations

import torch

from nla.layer_spec import BACKBONE_EMBEDDING_DIM
from nla.steering.backbone_steer import SteerSpec, attach_backbone_steer_batched


class _FakeBackbone(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, h = x.shape
        return {
            "backbone_features": x.clone(),
            "backbone_attention_mask": torch.ones(b, t, dtype=torch.bool),
            "image_mask": torch.zeros(b, t, dtype=torch.bool),
        }


def test_batched_steer_hook_modifies_each_row():
    backbone = _FakeBackbone()
    spec = SteerSpec(placement="fixed", fixed_token_index=0, blend=1.0)
    h = BACKBONE_EMBEDDING_DIM
    v0 = torch.randn(h)
    v1 = torch.randn(h)
    x = torch.randn(2, 4, h)
    with attach_backbone_steer_batched(backbone, [v0, v1], spec):
        out = backbone(x)
    assert torch.allclose(out["backbone_features"][0, 0], v0.to(out["backbone_features"].dtype))
    assert torch.allclose(out["backbone_features"][1, 0], v1.to(out["backbone_features"].dtype))
    assert not torch.allclose(out["backbone_features"][0, 1], v0.to(out["backbone_features"].dtype))

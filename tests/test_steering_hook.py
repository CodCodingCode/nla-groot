"""Unit tests for backbone feature steering (no GR00T dependency)."""

from __future__ import annotations

import torch
import torch.nn as nn

from nla.layer_spec import BACKBONE_EMBEDDING_DIM
from nla.steering.backbone_steer import SteerSpec, attach_backbone_steer, resolve_steer_indices


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: object = None) -> dict[str, torch.Tensor]:
        del x
        b, t, h = 1, 6, BACKBONE_EMBEDDING_DIM
        feats = torch.arange(float(t * h)).reshape(b, t, h) * 0.001 + self.bias
        attn = torch.ones(b, t, dtype=torch.bool)
        image = torch.zeros(b, t, dtype=torch.bool)
        image[0, 2] = True
        image[0, 3] = True
        return {
            "backbone_features": feats,
            "backbone_attention_mask": attn,
            "image_mask": image,
        }


def test_resolve_last_text_and_image_patch() -> None:
    attn = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    img = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    lt = resolve_steer_indices(attn, img, SteerSpec("last_text"))
    assert lt == [4]
    rng_idx = resolve_steer_indices(attn, img, SteerSpec("image_patch", image_patch_seed=42))
    assert rng_idx[0] in {2, 3}


def test_hook_replaces_anchor_token() -> None:
    backbone = ToyBackbone()
    out0 = backbone()
    feats0 = out0["backbone_features"].clone()

    steer = torch.full((BACKBONE_EMBEDDING_DIM,), 9.876)
    spec = SteerSpec("anchor", blend=1.0)
    with attach_backbone_steer(backbone, steer, spec):
        out1 = backbone()

    attn_row = out0["backbone_attention_mask"][0]
    anchor_idx = int(torch.nonzero(attn_row, as_tuple=False)[-1].item())

    patched = out1["backbone_features"][0, anchor_idx]
    assert torch.allclose(patched, steer.to(dtype=patched.dtype))

    out_base = backbone()
    assert torch.allclose(out_base["backbone_features"], feats0)


def test_blend_lerp() -> None:
    backbone = ToyBackbone()
    steer = torch.ones(BACKBONE_EMBEDDING_DIM) * 2.0
    spec = SteerSpec("anchor", blend=0.5)
    with attach_backbone_steer(backbone, steer, spec):
        out = backbone()
    attn = out["backbone_attention_mask"][0]
    idx = int(torch.nonzero(attn, as_tuple=False)[-1].item())
    base = ToyBackbone()()["backbone_features"][0, idx]
    got = out["backbone_features"][0, idx]
    expect = 0.5 * base + 0.5 * steer.to(dtype=base.dtype)
    assert torch.allclose(got, expect)

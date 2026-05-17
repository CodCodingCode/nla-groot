#!/usr/bin/env python
"""Smoke: AR(text) → backbone-space ĥ + ``attach_backbone_steer`` on a toy backbone.

Does **not** load GR00T or Cosmos — use after SFT ``ar/`` exists to sanity-check the
language→vector hook path without gated HF deps.

::

    PYTHONPATH=src .venv/bin/python scripts/eval/nla_steer_ar_smoke.py \\
        --ar-dir data/sft/libero_goal_pilot_v3/ar
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from nla.layer_spec import BACKBONE_EMBEDDING_DIM
from nla.steering import SteerSpec, attach_backbone_steer, ar_text_to_backbone_vec
from nla.training.checkpoint import load_ar_from_sft


def _default_text() -> str:
    return (
        "- scene: white round table with toys and a green bowl.\n"
        "- target: blue cube near the bowl rim.\n"
        "- gripper: open and approaching from the left.\n"
        "- spatial: upper workspace.\n"
        "- plan: grasp blue cube."
    )


class _ToyBackbone(nn.Module):
    def __init__(self, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self._device = device
        self._dtype = dtype

    def forward(self, x=None) -> dict[str, torch.Tensor]:
        del x
        b, t, h = 1, 6, BACKBONE_EMBEDDING_DIM
        feats = torch.randn(b, t, h, device=self._device, dtype=self._dtype)
        attn = torch.ones(b, t, dtype=torch.bool, device=self._device)
        img = torch.zeros(b, t, dtype=torch.bool, device=self._device)
        img[0, 2] = img[0, 3] = True
        return {
            "backbone_features": feats,
            "backbone_attention_mask": attn,
            "image_mask": img,
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ar-dir", required=True, help="SFT checkpoint subdirectory ``ar/``")
    p.add_argument("--device", default=None, help="Default: cuda:0 if available else cpu")
    p.add_argument("--text", default=None, help="Bullet-style caption; default is built-in")
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    text = args.text.strip() if args.text else _default_text()

    ar = load_ar_from_sft(args.ar_dir, device=device, freeze=True)
    h_hat = ar_text_to_backbone_vec(ar, text)
    print(f"device={device}  ĥ shape={tuple(h_hat.shape)}  L2={float(torch.linalg.norm(h_hat.float())):.4f}")

    toy = _ToyBackbone(device, dtype)
    spec = SteerSpec("anchor", blend=1.0)
    attn_len = 6
    anchor_idx = attn_len - 1

    with attach_backbone_steer(toy, h_hat.float(), spec):
        out = toy()
    feat = out["backbone_features"]
    d = feat[0, anchor_idx].float() - h_hat.to(device=feat.device, dtype=torch.float32)
    print(f"anchor replace max|Δ| (float32 vs ĥ): {d.abs().max().item():.6f}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

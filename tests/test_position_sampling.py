"""Tests for token position samplers.

In particular: ``nla.training.sampling.TokenPositionSampler`` and
``nla.extraction.sampler.sample_position`` MUST agree on what "anchor" means
(the last valid sequence token, which may be an image-patch token). If they
drift, labels and stratified metrics for the ``anchor`` bucket no longer refer
to the same token role.
"""

from __future__ import annotations

import numpy as np
import torch

from nla.extraction.sampler import _anchor_index, sample_position
from nla.training.sampling import TokenPositionSampler


def _build_image_tail_masks(T: int = 12, n_text: int = 3, n_image: int = 5):
    """Sequence of [text tokens][image tokens][pad], so anchor sits in image tail."""
    attn = torch.zeros(T, dtype=torch.bool)
    img = torch.zeros(T, dtype=torch.bool)
    attn[: n_text + n_image] = True
    img[n_text : n_text + n_image] = True
    return attn, img


def test_anchor_is_last_attention_token_when_image_tail():
    attn, img = _build_image_tail_masks(T=12, n_text=3, n_image=5)
    expected = int(_anchor_index(attn))
    assert expected == 7
    assert img[expected].item()
    sampler = TokenPositionSampler(seed=0)
    ptype, idx = sampler.sample(attn, img, force_type="anchor")
    assert ptype == "anchor"
    assert idx == expected


def test_anchor_agrees_between_training_and_extraction_samplers():
    attn, img = _build_image_tail_masks(T=10, n_text=2, n_image=6)
    train_sampler = TokenPositionSampler(seed=0)
    _, train_idx = train_sampler.sample(attn, img, force_type="anchor")
    sp = sample_position(attn, img, mix={"anchor": 1.0}, rng=np.random.default_rng(0))
    assert sp.type.value == "anchor"
    assert train_idx == sp.index


def test_anchor_with_text_tail_still_uses_last_attention_token():
    T = 8
    attn = torch.zeros(T, dtype=torch.bool)
    img = torch.zeros(T, dtype=torch.bool)
    attn[:6] = True
    img[1:3] = True
    sampler = TokenPositionSampler(seed=0)
    _, idx = sampler.sample(attn, img, force_type="anchor")
    assert idx == 5

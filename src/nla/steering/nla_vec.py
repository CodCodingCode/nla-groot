"""Map free-form task language to a GR00T backbone-space vector via AR."""

from __future__ import annotations

import torch

from nla.models import ActivationReconstructor


def ar_text_to_backbone_vec(ar: ActivationReconstructor, text: str) -> torch.Tensor:
    """Return ``ĥ`` with shape ``[H]`` (unscaled backbone space, same as extraction).

    ``text`` should follow the same bullet layout you used in ``labels.jsonl`` —
    the AR was trained on ``render_ar_prompt(description)`` from that corpus.
    """
    t = text.strip()
    if not t:
        raise ValueError("steer text is empty")
    out = ar.predict([t], unscale=True)
    return out.squeeze(0)

"""Map free-form task language to a GR00T backbone-space vector via AR."""

from __future__ import annotations

import torch

from nla.models import ActivationReconstructor
from nla.models.templates import PositionType


def ar_text_to_backbone_vec(
    ar: ActivationReconstructor,
    text: str,
    *,
    position_type: PositionType | None = None,
    step_index: int | None = None,
    instruction: str | None = None,
) -> torch.Tensor:
    """Return ``ĥ`` with shape ``[H]`` (unscaled backbone space, same as extraction).

    ``text`` should follow the same bullet layout you used in ``labels.jsonl`` —
    the AR was trained on ``render_ar_prompt(description)`` from that corpus.
    When the AR checkpoint uses ``ar_prompt_version='context_v5'``, pass
    ``position_type`` (and optionally ``step_index`` / ``instruction``) so the
    rendered prompt matches training.
    """
    t = text.strip()
    if not t:
        raise ValueError("steer text is empty")
    out = ar.predict(
        [t],
        unscale=True,
        position_types=[position_type] if position_type is not None else None,
        step_indices=[step_index],
        instructions=[instruction],
    )
    return out.squeeze(0)

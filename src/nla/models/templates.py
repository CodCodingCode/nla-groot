"""Prompt templates for the Verbalizer (AV) and Reconstructor (AR).

These templates are intentionally short and stable: AV/AR are downstream
consumers of an *embedding* (the activation), and the surrounding text is
mostly there to (a) signal task framing to the base LM and (b) reserve a slot
for activation injection (AV) or for the regression head pick-off (AR).

AR template is the NLA-paper-canonical "Summary" template, kept *exact* so the
affine head learns a stable pick-off position::

    Summary of the following text: <text>{explanation}</text> <summary>

AV template embeds the activation via a single reserved slot token whose
embedding we overwrite at forward time, and adds a short position-type hint
so the model can adapt its style across `last_text` / `image_patch` /
`anchor` activations.
"""

from __future__ import annotations

from typing import Literal


PositionType = Literal["last_text", "image_patch", "anchor", "fallback"]


# Single-slot AV template. The slot string is later resolved at runtime to a
# reserved special-token id (Qwen models reserve a handful of these).
AV_SLOT_PLACEHOLDER = "<<ACTIVATION_SLOT>>"

AV_PROMPT_TEMPLATE = (
    "You are interpretability tooling for the GR00T N1.7 vision-language-action "
    "robot model. You are shown a single internal activation vector taken from "
    "one token position inside the backbone, plus a short hint indicating where "
    "in the input the position sits.\n"
    "Position type: {position_type}.\n"
    "Activation: " + AV_SLOT_PLACEHOLDER + "\n"
    "Describe, in 4-5 bullet points (one per line, '- <category>: <content>.'), "
    "what features the model is internally tracking at this position to predict "
    "its next action. The last bullet should describe what this exact position "
    "encodes.\n"
    "Bullets:"
)


# AR template — keep verbatim so the head's pick-off position is stable.
AR_PROMPT_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def render_av_prompt(position_type: PositionType) -> str:
    """Substitute the position type into the AV template (slot stays as marker)."""
    return AV_PROMPT_TEMPLATE.format(position_type=position_type)


def render_ar_prompt(explanation: str) -> str:
    """Substitute the explanation into the AR template."""
    return AR_PROMPT_TEMPLATE.format(explanation=explanation.strip())

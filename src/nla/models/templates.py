"""Prompt templates for the Verbalizer (AV) and Reconstructor (AR).

These templates are intentionally short and stable: AV/AR are downstream
consumers of an *embedding* (the activation), and the surrounding text is
mostly there to (a) signal task framing to the base LM and (b) reserve a slot
for activation injection (AV) or for the regression head pick-off (AR).

AR template is the NLA-paper-canonical "Summary" template, kept *exact* so the
affine head learns a stable pick-off position::

    Summary of the following text: <text>{explanation}</text> <summary>

AV templates
------------

Two variants live here:

* ``AV_PROMPT_TEMPLATE_LEGACY`` — the V3/V4 single-slot template that gave AV
  only ``position_type`` plus the activation slot. Kept for ablations and so
  V4 checkpoints can still be loaded with byte-identical prompts.
* ``AV_PROMPT_TEMPLATE`` — the V5 "context" template. Same single-slot
  injection, but the prompt is enriched with the labeler-side fields the V3/V4
  AV never saw at SFT time: ``step_index`` (episode timestep) and the natural
  language ``instruction``. Both are sourced from the existing activation
  index / label JSONL — no relabeling required. Per
  ``.cursor/plans/v5_av_architecture_cd839da6.plan.md``.

For ``image_patch`` rows V5 also fans the activation out across **K** slots
(``<|act_slot_0|>`` ... ``<|act_slot_{K-1}|>``) so AV sees the strided patch
grid instead of a single mean-pooled vector. Single-slot rows
(``last_text``, ``anchor``, ``fallback``) keep the legacy
``<<ACTIVATION_SLOT>>`` placeholder so the existing
``AV_SLOT_PLACEHOLDER``-based injection path stays exact.
"""

from __future__ import annotations

from typing import Literal


PositionType = Literal["last_text", "image_patch", "anchor", "fallback"]


# Single-slot placeholder. ``AV_SLOT_PLACEHOLDER`` is the canonical name kept
# stable for V3/V4 callers; AV resolves it to a real reserved special-token id
# at __init__ time and overwrites the embedding at that index.
AV_SLOT_PLACEHOLDER = "<<ACTIVATION_SLOT>>"

# Multi-slot placeholder format. Numbered 0..K-1; each renders to a distinct
# reserved special-token id (``<|act_slot_0|>``, ...) so the injector can map
# slot k of K to its own (B, k, H) activation vector.
AV_MULTI_SLOT_PLACEHOLDER_FMT = "<<ACT_SLOT_{i}>>"


# Legacy V3/V4 single-slot template (preserved verbatim so old checkpoints and
# ablations stay byte-identical). New training defaults to ``AV_PROMPT_TEMPLATE``
# below.
AV_PROMPT_TEMPLATE_LEGACY = (
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


# V5 context-enriched template (single-slot). Adds the two labeler-side
# context lines (Timestep, Task instruction) that V3/V4 SFT never showed AV,
# even though they were available on every row of the activation index /
# labels JSONL.
AV_PROMPT_TEMPLATE = (
    "Position type: {position_type}.\n"
    "Timestep: {step_index}.\n"
    "Task instruction: \"{instruction}\"\n"
    "Activation: " + AV_SLOT_PLACEHOLDER + "\n"
    "Describe, in 4-5 bullet points (one per line, '- <category>: <content>.'), "
    "what features the model is internally tracking at this position to predict "
    "its next action. The last bullet should describe what this exact position "
    "encodes.\n"
    "Bullets:"
)


# Intent-conditioned variant used by sim-success GRPO. The model receives the
# usual scene activation AND a target task it should make the policy execute.
# The bullet structure mirrors the descriptive template so the AR (trained on
# bullet-style targets) sees a familiar surface form, but the *content* shifts
# toward "what would have to be in this activation for the policy to do the
# target task" rather than "what is in this activation now."
#
# V5: the same context lines (Timestep + Task instruction) are inserted above
# ``Target task:`` so GRPO/sim paths stay consistent when context is enabled.
# Both lines are still optional at render time -- omit either by passing
# ``None`` (renders an "(unknown)" / "(not provided)" placeholder).
AV_PROMPT_INTENT_CONDITIONED_TEMPLATE = (
    "You are interpretability tooling for the GR00T N1.7 vision-language-action "
    "robot model. You are shown one internal backbone activation, plus a target "
    "task you want the policy to execute next.\n"
    "Position type: {position_type}.\n"
    "Timestep: {step_index}.\n"
    "Task instruction: \"{instruction}\"\n"
    "Activation: " + AV_SLOT_PLACEHOLDER + "\n"
    "Target task: {target_intent}\n"
    "Write a 5-6 bullet description (one per line, '- <category>: <content>.'). "
    "Use these categories in order: scene, target, distractor, gripper, spatial, "
    "task. The last bullet ('- task:') must be the imperative for the target "
    "task above, phrased exactly as the model's instruction would say it. "
    "Write the bullets so that, if an activation reconstructor mapped this "
    "text back into backbone space, the resulting vector would make the policy "
    "execute the target task in this scene.\n"
    "Bullets:"
)


# AR template — keep verbatim so the head's pick-off position is stable.
AR_PROMPT_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"

# V5 context prefix prepended before the canonical Summary line (unchanged).
AR_PROMPT_TEMPLATE_CONTEXT_V5 = (
    "Position type: {position_type}.\n"
    "Timestep: {step_index}.\n"
    "Task instruction: \"{instruction}\"\n"
    + AR_PROMPT_TEMPLATE
)


# Sentinel rendering for missing context fields. We mirror the labeling
# pipeline's "(no instruction provided)" pattern (see
# ``nla.labeling.prompts.build_position_prompt``) so the AV prompt looks
# familiar to the base LM regardless of whether the row has metadata.
_TIMESTEP_UNKNOWN = "unknown"
_INSTRUCTION_NOT_PROVIDED = "(not provided)"


PromptVersion = Literal["legacy", "context_v5"]


def _format_step_index(step_index: int | None) -> str:
    return _TIMESTEP_UNKNOWN if step_index is None else str(int(step_index))


def _format_instruction(instruction: str | None) -> str:
    if instruction is None:
        return _INSTRUCTION_NOT_PROVIDED
    text = str(instruction).strip()
    return text or _INSTRUCTION_NOT_PROVIDED


def _build_multi_slot_av_prompt(
    *,
    position_type: PositionType,
    step_index: int | None,
    instruction: str | None,
    num_slots: int,
) -> str:
    """Render the multi-slot AV prompt for ``image_patch`` rows.

    Lays out ``num_slots`` lines of ``  patch i: <<ACT_SLOT_i>>`` after the
    context block. The AV injector finds each slot id, overwrites it with the
    matching projected activation, and runs the LM.
    """
    if num_slots < 1:
        raise ValueError(f"num_slots must be >= 1, got {num_slots}")
    slot_lines = "\n".join(
        f"  patch {i}: {AV_MULTI_SLOT_PLACEHOLDER_FMT.format(i=i)}"
        for i in range(num_slots)
    )
    return (
        f"Position type: {position_type}.\n"
        f"Timestep: {_format_step_index(step_index)}.\n"
        f"Task instruction: \"{_format_instruction(instruction)}\"\n"
        "Activation patches:\n"
        f"{slot_lines}\n"
        "Describe, in 4-5 bullet points (one per line, '- <category>: <content>.'), "
        "what features the model is internally tracking across these patches to "
        "predict its next action. The last bullet should describe what these "
        "patches collectively encode.\n"
        "Bullets:"
    )


def render_av_prompt(
    position_type: PositionType,
    *,
    target_intent: str | None = None,
    step_index: int | None = None,
    instruction: str | None = None,
    num_slots: int = 1,
    prompt_version: PromptVersion = "context_v5",
) -> str:
    """Render the AV prompt for one row.

    Parameters
    ----------
    position_type:
        Row's position-type literal (``last_text`` / ``image_patch`` /
        ``anchor`` / ``fallback``).
    target_intent:
        When provided, render the intent-conditioned variant (sim-GRPO). Mixes
        cleanly with ``step_index`` / ``instruction`` in V5.
    step_index, instruction:
        V5 context fields. ``None`` renders the canonical "(unknown)" /
        "(not provided)" sentinel so the prompt shape is constant across rows.
        Only consulted by the ``context_v5`` and intent-conditioned templates.
    num_slots:
        For ``context_v5`` + ``position_type == "image_patch"``: number of
        ``<<ACT_SLOT_i>>`` placeholders to emit (V5 default 8 patches). For
        every other (position_type, version) combination this must be 1.
    prompt_version:
        ``"context_v5"`` (default) renders the V5 context-enriched prompt;
        ``"legacy"`` renders the V3/V4 single-slot template byte-identical to
        the original code path. Ignored for the intent-conditioned variant
        (which is V5-shaped regardless of version, since it never shipped in
        V3/V4).
    """
    if target_intent is not None:
        if num_slots != 1:
            raise ValueError(
                "Intent-conditioned AV prompt only supports num_slots=1; "
                f"got num_slots={num_slots}."
            )
        return AV_PROMPT_INTENT_CONDITIONED_TEMPLATE.format(
            position_type=position_type,
            step_index=_format_step_index(step_index),
            instruction=_format_instruction(instruction),
            target_intent=str(target_intent).strip(),
        )

    if prompt_version == "legacy":
        if num_slots != 1:
            raise ValueError(
                "Legacy AV prompt only supports num_slots=1; "
                f"got num_slots={num_slots}."
            )
        return AV_PROMPT_TEMPLATE_LEGACY.format(position_type=position_type)

    if prompt_version != "context_v5":
        raise ValueError(
            f"Unknown prompt_version={prompt_version!r}; "
            "expected one of {'legacy', 'context_v5'}."
        )

    if num_slots > 1:
        return _build_multi_slot_av_prompt(
            position_type=position_type,
            step_index=step_index,
            instruction=instruction,
            num_slots=num_slots,
        )

    return AV_PROMPT_TEMPLATE.format(
        position_type=position_type,
        step_index=_format_step_index(step_index),
        instruction=_format_instruction(instruction),
    )


def render_ar_prompt(
    explanation: str,
    *,
    position_type: PositionType | None = None,
    step_index: int | None = None,
    instruction: str | None = None,
    prompt_version: PromptVersion = "legacy",
) -> str:
    """Substitute the explanation into the AR template.

    ``legacy`` (default) renders only the canonical Summary line so V3/V4
    checkpoints and tests stay byte-identical. ``context_v5`` prepends
    position type, timestep, and task instruction before that line.
    """
    text = explanation.strip()
    if prompt_version == "legacy":
        return AR_PROMPT_TEMPLATE.format(explanation=text)
    if prompt_version != "context_v5":
        raise ValueError(
            f"Unknown prompt_version={prompt_version!r}; "
            "expected one of {'legacy', 'context_v5'}."
        )
    if position_type is None:
        raise ValueError("position_type is required when prompt_version='context_v5'.")
    return AR_PROMPT_TEMPLATE_CONTEXT_V5.format(
        position_type=position_type,
        step_index=_format_step_index(step_index),
        instruction=_format_instruction(instruction),
        explanation=text,
    )

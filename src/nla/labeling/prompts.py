"""Warm-start labeling prompts for GR00T NLA.

Two prompt families:

1. ``build_position_prompt`` — *per-token-position* labeling, used as the SFT
   target for the AV.  This is the NLA paper's recipe adapted for a VLA: ask
   what features the model is tracking *at this token position* in order to
   choose the next action (or next token).  The labeler sees the camera
   frame(s), the text instruction, the decoded text portion of the context,
   robot state, and the position type (last_text / image_patch / anchor).

2. ``build_step_prompt`` — legacy *per-step* "scene description" labeling,
   useful as a fast/cheap baseline.  Kept for back-compat with earlier
   scaffolding; not used by the NLA pipeline.

Both prompts ask for the same fixed bullet style so the AV inherits a
predictable output format across SFT (per the NLA paper's observation that
"the AV inherits the format of warm-start data ... this style persists
through training").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Style spec (shared)
# ---------------------------------------------------------------------------

BULLET_CATEGORIES: tuple[str, ...] = (
    "scene",          # what's in the camera view, surfaces, lighting
    "target",         # the object/region the task is about
    "distractor",     # nearby objects that are NOT the target
    "spatial",        # relative positions, distances, frames of reference
    "plan",           # high-level sub-goal / next action class
    "motion",         # current and imminent end-effector motion direction
    "gripper",        # gripper state and grasp readiness
    "language",       # what the instruction tells the model to do
    "image_region",   # what region of the image the model attends to
)

_STYLE_CLAUSE = (
    "Output exactly 4-5 short bullets, each on its own line, in this fixed style:\n"
    '    "- <category>: <concrete content>."\n'
    f"  where <category> is drawn from: {', '.join(BULLET_CATEGORIES)}.\n"
    "- Be specific: reference concrete objects, colors, surfaces, relative positions.\n"
    "- Use declarative tone (no hedging like 'seems' or 'might').\n"
    "- No preamble, no conclusion, no non-bullet text."
)


# ---------------------------------------------------------------------------
# Per-position prompt (the NLA target)
# ---------------------------------------------------------------------------

PositionType = Literal["last_text", "image_patch", "anchor", "fallback"]


@dataclass
class PositionLabelInput:
    """A single per-token-position labeling request.

    Attributes
    ----------
    example_id:
        Stable id including the position, e.g. ``"traj000001_step000017@p042_image_patch"``.
    instruction:
        Natural-language task text (may be empty if the dataset lacked one).
    decoded_text_context:
        The text portion of the model's tokenized input, with image regions
        collapsed to ``<image: N patches>`` placeholders.  Truncated if very
        long.
    position_index:
        Token position the AV will see (0-indexed, in the original sequence).
    position_type:
        One of ``last_text``, ``image_patch``, ``anchor``, ``fallback``.
    sequence_length:
        T of the original sequence (used for "position 42 of 277").
    image_patch_meta:
        Optional ``(k_within_image, n_image_tokens)`` for image-patch positions
        so the labeler can be told "image patch 42 of 248".
    image_paths:
        Local paths to the camera frames at this step.  All attached to the
        OpenAI call; ordering preserved.
    state:
        Optional robot state vector (free-form list of floats, with a name
        hint in ``state_name`` for the prompt header).
    state_name:
        Human-readable name for the state vector, e.g.
        ``"x, y, z, roll, pitch, yaw, pad, gripper"``.
    episode_index, step_index:
        Provenance only; do not affect the prompt.
    """

    example_id: str
    instruction: str
    decoded_text_context: str
    position_index: int
    position_type: PositionType
    sequence_length: int
    image_paths: list[str]
    image_patch_meta: tuple[int, int] | None = None
    state: list[float] | None = None
    state_name: str | None = None
    episode_index: int | None = None
    step_index: int | None = None
    extra: dict = field(default_factory=dict)


_POSITION_SYSTEM = f"""You are an interpretability annotator for the GR00T N1.7 \
vision-language-action robot model.

You are shown one step of robot input: an instruction, robot state, the text \
tokens the model has processed, and one or more camera frames.  Your job is to \
predict 4-5 features that the model is internally tracking *at the highlighted \
token position*, in order to choose its next action (or generate its next token).

{_STYLE_CLAUSE}

Crucially, the LAST bullet must specifically describe what the highlighted \
position encodes — i.e., what the model is committing to *at that moment* — \
not just the overall scene.  Examples of last-bullet content by position type:
- last_text: "language: instruction has been read; goal is to grasp the blue cube."
- image_patch: "image_region: focusing on the bowl rim in the upper-right of the table."
- anchor: "plan: ready to begin reaching toward the target."
"""


def _format_state(state: list[float] | None, name: str | None) -> str:
    if not state:
        return ""
    body = ", ".join(f"{x:.3f}" for x in state)
    if name:
        return f"\nRobot state ({name}): [{body}]\n"
    return f"\nRobot state: [{body}]\n"


def _format_position_clause(inp: PositionLabelInput) -> str:
    pos_idx = inp.position_index
    seq_len = inp.sequence_length
    base = (
        f"\nThe model has processed the input up to and including token "
        f"position {pos_idx} (out of {seq_len}). "
    )
    if inp.position_type == "image_patch":
        if inp.image_patch_meta is not None:
            k, n = inp.image_patch_meta
            base += (
                f"This position is an IMAGE-PATCH token (image patch {k} of {n} "
                "across the attached camera frame(s))."
            )
        else:
            base += "This position is an IMAGE-PATCH token."
    elif inp.position_type == "last_text":
        base += "This position is the LAST TEXT token before the action head reads the context."
    elif inp.position_type == "anchor":
        base += "This position is the ANCHOR token at the very end of the prompt."
    else:
        base += "This position is a generic context token (fallback)."
    return base + "\n"


def build_position_prompt(inp: PositionLabelInput) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)``; images attached separately."""
    instruction = inp.instruction.strip() or "(no instruction provided)"
    user = (
        f'Task instruction: "{instruction}"\n'
        f"{_format_state(inp.state, inp.state_name)}"
        f"\nText portion of the model's tokenized input "
        "(image regions collapsed to placeholders):\n"
        f"<context>\n{inp.decoded_text_context}\n</context>\n"
        f"{_format_position_clause(inp)}"
        "\nProduce the 4-5 bullet description now."
    )
    return _POSITION_SYSTEM, user


_STRICT_EXTRA = (
    "\nAdditional rules (strict):\n"
    "- Both 'scene' and 'target' bullets are REQUIRED in every response. "
    "If no clear target object is visible at this position (especially for "
    "image_patch tokens that fall on background pixels), write exactly: "
    "'- target: none in this patch.'\n"
    "- Use ONLY these bullet categories: "
    f"{', '.join(BULLET_CATEGORIES)}. Do not invent new categories "
    "(no 'tool', 'object', 'secondary_target', etc.).\n"
)

_STRICT_POSITION_SYSTEM = _POSITION_SYSTEM + _STRICT_EXTRA


def build_strict_position_prompt(inp: PositionLabelInput) -> tuple[str, str]:
    """Like ``build_position_prompt`` but enforces required bullets and category set.

    Used for re-labeling rows that were missing ``scene:`` / ``target:`` bullets
    or that invented non-canonical categories. The user prompt is unchanged; the
    system prompt has two extra rules appended.
    """
    _, user = build_position_prompt(inp)
    return _STRICT_POSITION_SYSTEM, user


# ---------------------------------------------------------------------------
# Per-step prompt (legacy / scene-level baseline, kept for completeness)
# ---------------------------------------------------------------------------

@dataclass
class LabelInput:
    """Legacy per-episode-step labeling input (one label per step, not per token)."""

    example_id: str
    instruction: str
    image_path: str
    state: list[float] | None = None
    state_name: str | None = None
    episode_id: str | None = None
    timestep: int | None = None


_STEP_SYSTEM = f"""You are an interpretability annotator for the GR00T N1.7 \
vision-language-action robot model.  For each scene, produce a short \
structured description of what features the model is likely tracking in its \
internal representation at this step.

{_STYLE_CLAUSE}
"""


def build_step_prompt(inp: LabelInput) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for per-step labeling."""
    instruction = inp.instruction.strip() or "(no instruction provided)"
    user = (
        f'Task instruction: "{instruction}"\n'
        f"{_format_state(inp.state, inp.state_name)}"
        "\nProduce the 4-5 bullet description now."
    )
    return _STEP_SYSTEM, user


# ---------------------------------------------------------------------------
# Back-compat shim (some earlier scaffolding imported ``build_label_prompt``)
# ---------------------------------------------------------------------------

def build_label_prompt(inp) -> tuple[str, str]:
    """Dispatch by input type so old call sites keep working."""
    if isinstance(inp, PositionLabelInput):
        return build_position_prompt(inp)
    return build_step_prompt(inp)

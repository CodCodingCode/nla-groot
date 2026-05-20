"""Prompt variants for the A/B test.

Each variant exposes:

  variant(inp: PositionLabelInput) -> VariantOutput

where ``VariantOutput`` carries
  * ``system_prompt``: the system message
  * ``user_prompt``:   the user message text (images added by the caller)
  * ``response_format``: an OpenAI ``response_format`` dict or ``None``
  * ``post_process``:  optional callable(str) -> str applied to the model
    output before scoring (used by V2 / V6 to convert JSON -> bullet text).

Round-1 variants (V0..V6) per the plan:

  V0  baseline (current prompt, unchanged)
  V1  V0 + 3 hand-crafted few-shot exemplars (one per position type)
  V2  V0 + JSON-schema response_format (server-side structure); post-process to bullets
  V3  V0 + explicit anti-pattern paragraph
  V4  V0 + length cap clause (15-30 words/bullet, no paragraph bullets)
  V5  V0 + diversity-forcing clause (>=3 concrete distinguishing tokens)
  V6  V0 + V1 + V2 + V3 + V4 + V5 combined

Later rounds register additional variants via ``register_variant``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from nla.labeling.prompts import (
    BULLET_CATEGORIES,
    PositionLabelInput,
    _format_position_clause,
    _format_state,
    _STYLE_CLAUSE,
)


# ---------------------------------------------------------------------------
# Variant output type
# ---------------------------------------------------------------------------

@dataclass
class VariantOutput:
    system_prompt: str
    user_prompt: str
    response_format: dict | None = None
    post_process: Callable[[str], str] | None = None
    meta: dict = field(default_factory=dict)


VariantFn = Callable[[PositionLabelInput], VariantOutput]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, VariantFn] = {}


def register_variant(variant_id: str, fn: VariantFn) -> None:
    if variant_id in _REGISTRY:
        raise ValueError(f"variant {variant_id} already registered")
    _REGISTRY[variant_id] = fn


def get_variant(variant_id: str) -> VariantFn:
    if variant_id not in _REGISTRY:
        raise KeyError(
            f"unknown variant {variant_id}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[variant_id]


def list_variants() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_LAST_BULLET_GUIDANCE = (
    "Crucially, the LAST bullet must specifically describe what feature this "
    "token position carries forward into the action head -- the perceptual, "
    "spatial, or plan-level content tied to this slot -- not just the overall "
    "scene.  Examples of last-bullet content by position type:\n"
    "- last_text: \"plan: pick-and-place phase active; reach over the bowl, then place on the plate.\"\n"
    "- image_patch: \"target: bowl rim and gripper visible at this position; the specific patch within the frame is not localized.\"\n"
    "- anchor: \"plan: arm staged at the start of the reaching trajectory.\""
)


def _base_user_prompt(inp: PositionLabelInput) -> str:
    instruction = inp.instruction.strip() or "(no instruction provided)"
    return (
        f'Task instruction: "{instruction}"\n'
        f"{_format_state(inp.state, inp.state_name)}"
        f"\nText portion of the model's tokenized input "
        "(image regions collapsed to placeholders):\n"
        f"<context>\n{inp.decoded_text_context}\n</context>\n"
        f"{_format_position_clause(inp)}"
        "\nProduce the 4-5 bullet description now."
    )


# ---------------------------------------------------------------------------
# V0: current baseline (mirror of build_position_prompt)
# ---------------------------------------------------------------------------

_V0_SYSTEM = (
    "You are an interpretability annotator for the GR00T N1.7 "
    "vision-language-action robot model.\n\n"
    "You are shown one step of robot input: an instruction, robot state, the "
    "text tokens the model has processed, and one or more camera frames.  Your "
    "job is to predict 4-5 features that the model is internally tracking "
    "*at the highlighted token position*, in order to choose its next action "
    "(or generate its next token).\n\n"
    f"{_STYLE_CLAUSE}\n\n"
    f"{_LAST_BULLET_GUIDANCE}\n"
)


def v0_baseline(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V0_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=None,
        post_process=None,
        meta={"variant": "V0", "label": "baseline"},
    )


# ---------------------------------------------------------------------------
# V1: V0 + 3 hand-crafted few-shot exemplars
# ---------------------------------------------------------------------------

_FEWSHOT_LAST_TEXT = """Example -- position_type = last_text
Task instruction: "Put the blue block in the green bowl"
- scene: tabletop with a green ceramic bowl in the center-left and a wooden block tray on the right.
- target: small blue cube resting on the wooden tray near the bowl.
- spatial: blue cube is right of and slightly behind the green bowl; both within reach of the gripper.
- distractor: yellow banana toy and orange ring sit on the same tray, closer to the camera.
- plan: pick-and-place phase active; reach over the blue cube, then transport into the green bowl."""

_FEWSHOT_IMAGE_PATCH = """Example -- position_type = image_patch
Task instruction: "Wipe the spill near the cup"
- scene: bright kitchen counter with a white paper towel folded near a red ceramic cup.
- distractor: the red ceramic cup is right of the spill; it should not be displaced.
- spatial: spill is directly under the camera and slightly left of the cup; cup remains upright.
- motion: end-effector is mid-air above the towel, descending toward the wet patch.
- target: dark wet spill at this position; this slot tracks the spill boundary against the white counter."""

_FEWSHOT_ANCHOR = """Example -- position_type = anchor
Task instruction: "Stack the red block on top of the green block"
- scene: indoor robot table with two colored blocks and a flat foam mat under them.
- target: the red block currently held in the gripper, oriented flat-side down.
- spatial: green block is forward and to the right; landing pose is centered over its top face.
- plan: ready to begin the controlled lower-and-release stacking motion.
- gripper: gripper is closed on the red block with stable grasp; about to descend."""


_V1_SYSTEM = (
    _V0_SYSTEM
    + "\nFollow the style of these three exemplars exactly:\n\n"
    + _FEWSHOT_LAST_TEXT + "\n\n"
    + _FEWSHOT_IMAGE_PATCH + "\n\n"
    + _FEWSHOT_ANCHOR + "\n"
)


def v1_fewshot(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V1_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=None,
        post_process=None,
        meta={"variant": "V1", "label": "fewshot"},
    )


# ---------------------------------------------------------------------------
# V2: V0 + JSON-schema response_format
# ---------------------------------------------------------------------------

_V2_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nla_label",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["bullets"],
            "properties": {
                "bullets": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["category", "content"],
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": list(BULLET_CATEGORIES),
                            },
                            "content": {
                                "type": "string",
                                "minLength": 20,
                            },
                        },
                    },
                }
            },
        },
        "strict": True,
    },
}


_V2_SYSTEM = (
    _V0_SYSTEM
    + "\nOutput must conform to the response_format JSON schema. Each bullet is "
    "an object with a category (drawn from the allow-list) and a content string. "
    "Output ONLY the JSON object, no prose, no markdown."
)


def _v2_post_process(text: str) -> str:
    """Convert the JSON object the model returned into the canonical bullet
    string the scorers and downstream consumers expect.
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
    except Exception:
        return text  # let qa_metrics flag the failure
    bullets = obj.get("bullets")
    if not isinstance(bullets, list):
        return text
    lines: list[str] = []
    for b in bullets:
        if not isinstance(b, dict):
            continue
        cat = (b.get("category") or "").strip()
        cnt = (b.get("content") or "").strip().rstrip(".")
        if not cat or not cnt:
            continue
        lines.append(f"- {cat}: {cnt}.")
    return "\n".join(lines)


def v2_jsonschema(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V2_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=_V2_JSON_SCHEMA,
        post_process=_v2_post_process,
        meta={"variant": "V2", "label": "json_schema"},
    )


# ---------------------------------------------------------------------------
# V3: V0 + anti-pattern paragraph
# ---------------------------------------------------------------------------

_ANTI_PATTERN_CLAUSE = (
    "STRICT VOCABULARY CONSTRAINTS (any of these makes the label unusable):\n"
    "- Never include numeric measurements (e.g., \"90 degrees\", \"5 mm\", "
    "\"50% force\", \"3 N\"). Describe relative positions qualitatively.\n"
    "- Never ascribe affect or mental states (no \"feels\", \"wants\", "
    "\"thinks\", \"decides\", \"believes\", \"hopes\").\n"
    "- Never specify actuator-level commands (joint angles, torque values, "
    "force percentages, motor commands, PWM).\n"
    "- Never invent objects not visible in the camera frame(s) you were given.\n"
    "- Never mention pixel values, RGB tuples, or hex colors. Use natural "
    "color names (e.g., \"blue\", \"olive green\").\n"
)


_V3_SYSTEM = _V0_SYSTEM + "\n" + _ANTI_PATTERN_CLAUSE


def v3_antipattern(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V3_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=None,
        post_process=None,
        meta={"variant": "V3", "label": "antipattern"},
    )


# ---------------------------------------------------------------------------
# V4: V0 + length cap clause
# ---------------------------------------------------------------------------

_LENGTH_CAP_CLAUSE = (
    "LENGTH BUDGET (hard constraint):\n"
    "- Each bullet's content (after the category) must be 15-30 words.\n"
    "- No paragraph bullets, no multi-sentence bullets longer than 30 words.\n"
    "- One short declarative clause per bullet is ideal; semicolons OK but no "
    "more than two."
)


_V4_SYSTEM = _V0_SYSTEM + "\n" + _LENGTH_CAP_CLAUSE


def v4_length_cap(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V4_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=None,
        post_process=None,
        meta={"variant": "V4", "label": "length_cap"},
    )


# ---------------------------------------------------------------------------
# V5: V0 + diversity-forcing clause
# ---------------------------------------------------------------------------

_DIVERSITY_CLAUSE = (
    "SCENE-SPECIFIC GROUNDING (hard constraint):\n"
    "- The bullets together must reference at least three concrete tokens "
    "that distinguish THIS scene from any other manipulation scene: examples "
    "include named colors, named objects, specific surfaces or textures, "
    "or relative-position phrases (\"to the right of\", \"between\", \"in front of\").\n"
    "- Avoid template phrasings that would apply unchanged to any tabletop "
    "scene; if a sentence would be true of every demo, replace it with one "
    "naming what is visible here."
)


_V5_SYSTEM = _V0_SYSTEM + "\n" + _DIVERSITY_CLAUSE


def v5_diversity(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V5_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=None,
        post_process=None,
        meta={"variant": "V5", "label": "diversity"},
    )


# ---------------------------------------------------------------------------
# V6: V0 + V1 + V2 + V3 + V4 + V5 combined
# ---------------------------------------------------------------------------

_V6_SYSTEM = (
    _V0_SYSTEM
    + "\n" + _ANTI_PATTERN_CLAUSE
    + "\n" + _LENGTH_CAP_CLAUSE
    + "\n" + _DIVERSITY_CLAUSE
    + "\nFollow the style of these three exemplars exactly:\n\n"
    + _FEWSHOT_LAST_TEXT + "\n\n"
    + _FEWSHOT_IMAGE_PATCH + "\n\n"
    + _FEWSHOT_ANCHOR + "\n"
    + "\nOutput must conform to the response_format JSON schema. Output ONLY "
      "the JSON object, no prose, no markdown."
)


def v6_combined(inp: PositionLabelInput) -> VariantOutput:
    return VariantOutput(
        system_prompt=_V6_SYSTEM,
        user_prompt=_base_user_prompt(inp),
        response_format=_V2_JSON_SCHEMA,
        post_process=_v2_post_process,
        meta={"variant": "V6", "label": "combined"},
    )


# ---------------------------------------------------------------------------
# Register round-1 variants
# ---------------------------------------------------------------------------

register_variant("V0", v0_baseline)
register_variant("V1", v1_fewshot)
register_variant("V2", v2_jsonschema)
register_variant("V3", v3_antipattern)
register_variant("V4", v4_length_cap)
register_variant("V5", v5_diversity)
register_variant("V6", v6_combined)


# ---------------------------------------------------------------------------
# V5 nested JSON (per-timestep, three slots) — temperature A/B variants
# ---------------------------------------------------------------------------

def _position_to_step_label_input(inp: PositionLabelInput):
    from nla.labeling.prompts_v5 import position_input_to_step

    return position_input_to_step(inp)




def _v5_nested_post_process(raw: str) -> str:
    from nla.labeling.prompts_v5 import parse_v5_response
    from nla.labeling.schema_v5 import SLOT_NAMES, render_slot_bullets, validate_nested
    obj = parse_v5_response(raw)
    ok, errs, norm = validate_nested(obj)
    if not ok:
        raise ValueError("V5 schema: " + "; ".join(errs[:5]))
    parts = [f"[{slot.upper()}]\n" + render_slot_bullets(norm[slot]) for slot in SLOT_NAMES]
    return "\n\n".join(parts)

def _v5_nested_variant(temperature: float, variant_id: str) -> VariantFn:
    from nla.labeling.prompts_v5 import V5_NESTED_JSON_SCHEMA, build_v5_step_prompt

    def _fn(inp: PositionLabelInput) -> VariantOutput:
        step_inp = _position_to_step_label_input(inp)
        system, user = build_v5_step_prompt(step_inp)
        return VariantOutput(
            system_prompt=system,
            user_prompt=user,
            response_format=V5_NESTED_JSON_SCHEMA,
            post_process=_v5_nested_post_process,
            meta={
                "variant": variant_id,
                "label": "v5_nested",
                "temperature": temperature,
                "schema": "v5_nested",
            },
        )

    return _fn


def v5_nested_base(inp: PositionLabelInput) -> VariantOutput:
    """V5 nested JSON at temperature 0 (default nested builder)."""
    return _v5_nested_variant(0.0, "V5_nested_T0")(inp)


register_variant("V5_nested_T0", _v5_nested_variant(0.0, "V5_nested_T0"))
register_variant("V5_nested_T07", _v5_nested_variant(0.7, "V5_nested_T07"))
register_variant("V5_nested_T10", _v5_nested_variant(1.0, "V5_nested_T10"))


__all__ = [
    "VariantOutput",
    "VariantFn",
    "register_variant",
    "get_variant",
    "list_variants",
    "v0_baseline",
    "v1_fewshot",
    "v2_jsonschema",
    "v3_antipattern",
    "v4_length_cap",
    "v5_diversity",
    "v6_combined",
    "v5_nested_base",
]

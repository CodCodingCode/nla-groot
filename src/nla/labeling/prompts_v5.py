"""V5 per-timestep nested JSON labeling prompts.

One GPT call per robot timestep produces three slot objects (image_patch,
last_text, anchor), each with scene / target / plan / spatial fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from nla.labeling.prompts import infer_suite_from_example_id
from nla.labeling.schema_v5 import SLOT_NAMES

_V5_STEP_SYSTEM = """You are an interpretability annotator for the GR00T N1.7 vision-language-action robot.

You see ONE timestep: camera frame(s), the task instruction, and (optionally) decoded text context. Output ONE JSON object with exactly three top-level keys: image_patch, last_text, anchor.

Each slot is an object with string fields: scene, target, plan, spatial. Use the literal string NA when a field does not apply.

SLOT RULES (hard constraints):

image_patch — perceptual only
- scene: what is visible in the camera at this patch timestep (surfaces, objects, lighting).
- target: the object/region this visual token would attend to; patch-local, not whole-task summary.
- plan: MUST be NA (never describe motion or instruction phases here).
- spatial: relative layout visible in the frame, or NA.
- Do NOT paraphrase the instruction; do NOT name future trajectory steps.

last_text — language-bound, imminent plan
- scene: brief scene context as relevant to the language slot (may overlap objects with image_patch but wording should reflect text-side summary).
- target: the manipulation object or subgoal tied to the instruction.
- plan: exactly ONE imminent motion, formatted "phase: detail" where phase is one of approach, reach, grasp, pickup, lift, transport, place, release, retreat, idle, align, open, close.
- spatial: qualitative relations for the imminent action, or NA.
- Plan must describe what is happening NOW or immediately next — not the full episode.

anchor — instruction-bound goal, no future trajectory
- scene: task-relevant visible layout (objects needed for the instruction).
- target: the goal object or region named by the instruction and visible now.
- plan: ONE instruction-bound phase as "phase: detail" grounded in visible state (what subgoal this timestep serves).
- spatial: goal-relative layout, or NA.
- NEVER forecast remaining steps, "overall trajectory", "over the next N timesteps", or unseen future states.

GLOBAL
- Colors and object identities must match the attached frame(s), not the instruction text when they disagree.
- No scaffold jargon: action head, transformer, token position, this patch carries.
- Output JSON only; no markdown fences or prose outside the object."""


@dataclass
class StepLabelInput:
    example_id: str
    instruction: str
    image_paths: list[str]
    step_index: int | None = None
    decoded_text_context: str = ""
    suite: str | None = None
    episode_index: int | None = None
    extra: dict = field(default_factory=dict)


def position_input_to_step(inp) -> StepLabelInput:
    """Build :class:`StepLabelInput` from a :class:`PositionLabelInput`."""
    source_id = inp.extra.get("source_example_id")
    if source_id is None:
        source_id = inp.example_id.split("@", 1)[0]
    suite = getattr(inp, "suite", None) or inp.extra.get("suite")
    if suite is None:
        suite = infer_suite_from_example_id(inp.example_id, extra=inp.extra)
    return StepLabelInput(
        example_id=str(source_id),
        instruction=inp.instruction,
        image_paths=list(inp.image_paths),
        step_index=inp.step_index,
        decoded_text_context=inp.decoded_text_context,
        suite=suite,
        episode_index=inp.episode_index,
        extra=dict(inp.extra),
    )


V5_NESTED_JSON_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "v5_nested_slots",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                slot: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scene": {"type": "string"},
                        "target": {"type": "string"},
                        "plan": {"type": "string"},
                        "spatial": {"type": "string"},
                    },
                    "required": ["scene", "target", "plan", "spatial"],
                }
                for slot in SLOT_NAMES
            },
            "required": list(SLOT_NAMES),
        },
    },
}


def build_v5_step_prompt(inp: StepLabelInput) -> tuple[str, str]:
    instruction = inp.instruction.strip() or "(no instruction provided)"
    step = "unknown" if inp.step_index is None else str(int(inp.step_index))
    suite = inp.suite or infer_suite_from_example_id(inp.example_id, extra=inp.extra) or ""
    suite_line = f"Dataset suite: {suite}.\n" if suite else ""
    ctx = inp.decoded_text_context.strip()
    ctx_block = f"Decoded model text context (truncated):\n{ctx[:1200]}\n\n" if ctx else ""
    user = (
        f"{suite_line}"
        f'Task instruction: "{instruction}"\n'
        f"Timestep index: {step}\n"
        f"{ctx_block}"
        "Produce the nested JSON for image_patch, last_text, and anchor now."
    )
    return _V5_STEP_SYSTEM, user


def parse_v5_response(raw: str) -> dict:
    """Parse model output into a nested dict (raises on invalid JSON)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


__all__ = [
    "SLOT_NAMES",
    "StepLabelInput",
    "V5_NESTED_JSON_SCHEMA",
    "build_v5_step_prompt",
    "parse_v5_response",
    "position_input_to_step",
]

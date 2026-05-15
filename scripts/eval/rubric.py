"""Anchored rubric schema + validators for the LLM judge.

Why this exists
---------------

"LLM as judge" is only as structured as the schema you force the judge into.
This module is the single source of truth for:

1. The set of rubric dimensions we score (specificity, consistency,
   confabulation, overall_faithfulness, confidence).
2. The exact integer ranges allowed per dimension and the **anchored
   definition for each integer** (so a "2" means the same thing across runs
   and across judges).
3. A strict validator that:
   - rejects extra/missing fields,
   - clips out-of-range integers,
   - ensures evidence quotes come from a known source field
     (``baseline_text`` / ``edited_text`` / ``control_text``).

The judge prompt template imports ``RUBRIC_DEFINITIONS`` so the prompt and the
validator can never drift out of sync.

Schema
------
``judge_rows.jsonl`` rows (one per case)::

    {
      "case_id":                  "case_000003",
      "specificity_0_3":          int,         # 0..3
      "consistency_0_3":          int,         # 0..3
      "confabulation_0_3":        int,         # 0..3 (LOW is bad; see anchors)
      "overall_faithfulness_0_3": int,         # 0..3
      "confidence_0_1":           float,       # 0..1
      "evidence_spans": [
         {"source": "baseline_text"|"edited_text"|"control_text",
          "quote": "...verbatim substring of source..."},
         ...
      ],
      "rationale": "<=2 sentences",
      "judge_model": "gpt-4o-2024-08-06",      # filled by the runner
      "judge_seed":  0
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RubricDimension:
    """One scoring dimension with anchored bin definitions."""

    key: str
    min_val: int
    max_val: int
    description: str
    anchors: dict[int, str]  # int score -> short anchored definition


# ---------------------------------------------------------------------------
# RUBRIC DEFINITIONS — keep in lockstep with judge_prompt.md.
# ---------------------------------------------------------------------------
RUBRIC_DEFINITIONS: dict[str, RubricDimension] = {
    "specificity_0_3": RubricDimension(
        key="specificity_0_3",
        min_val=0,
        max_val=3,
        description=(
            "How concretely the baseline explanation describes what the "
            "activation encodes (objects, actions, spatial role)."
        ),
        anchors={
            0: "Generic boilerplate; no objects/actions/spatial role mentioned.",
            1: "One vague descriptor (e.g. 'tracks objects') without referents.",
            2: "Names at least one concrete object OR action OR position role.",
            3: "Names a specific object AND its role in the next action plan.",
        },
    ),
    "consistency_0_3": RubricDimension(
        key="consistency_0_3",
        min_val=0,
        max_val=3,
        description=(
            "Whether the edited explanation differs from the baseline in a way "
            "that aligns with the intervention's predicted direction."
        ),
        anchors={
            0: "Edited explanation is identical or contradicts the predicted edit.",
            1: "Tiny surface change unrelated to the predicted direction.",
            2: "Substantive change in the right semantic field but imprecise.",
            3: "Edited explanation reflects exactly the predicted direction.",
        },
    ),
    "confabulation_0_3": RubricDimension(
        key="confabulation_0_3",
        min_val=0,
        max_val=3,
        description=(
            "How much false content the explanations contain that contradicts "
            "what is actually shown by the activation/case context. "
            "HIGHER == BETTER (less confabulation)."
        ),
        anchors={
            0: "Most content is invented or contradicted by the case.",
            1: "Multiple invented claims mixed with real ones.",
            2: "One borderline claim; rest is grounded.",
            3: "All claims are plausibly grounded in the case context.",
        },
    ),
    "overall_faithfulness_0_3": RubricDimension(
        key="overall_faithfulness_0_3",
        min_val=0,
        max_val=3,
        description=(
            "Holistic judgement: do baseline + edited + control jointly behave "
            "as the hypothesis predicts?"
        ),
        anchors={
            0: "Edits and control are indistinguishable; explanation is hollow.",
            1: "Some directional change but control changes too (low specificity).",
            2: "Edit changes explanation; control mostly unchanged.",
            3: "Edit shifts explanation in the predicted direction; control unchanged.",
        },
    ),
}


CONFIDENCE_KEY = "confidence_0_1"
EVIDENCE_KEY = "evidence_spans"
RATIONALE_KEY = "rationale"

ALLOWED_EVIDENCE_SOURCES = {"baseline_text", "edited_text", "control_text"}


REQUIRED_FIELDS = (
    "case_id",
    *RUBRIC_DEFINITIONS.keys(),
    CONFIDENCE_KEY,
    EVIDENCE_KEY,
    RATIONALE_KEY,
)


# ---------------------------------------------------------------------------
# Schema doc generators (so judge_prompt.md and runtime always agree).
# ---------------------------------------------------------------------------

def render_rubric_for_prompt() -> str:
    """Render the anchored rubric as plain text for inclusion in the LLM prompt."""
    lines: list[str] = []
    for dim in RUBRIC_DEFINITIONS.values():
        lines.append(f"### {dim.key}  (range {dim.min_val}..{dim.max_val})")
        lines.append(dim.description)
        for v in range(dim.min_val, dim.max_val + 1):
            anchor = dim.anchors.get(v, "(no anchor)")
            lines.append(f"  - {v}: {anchor}")
        lines.append("")
    lines.append(
        f"### {CONFIDENCE_KEY}  (range 0.0..1.0)\n"
        "Self-reported judge confidence. 0.0 = pure guess, 1.0 = clear-cut."
    )
    lines.append("")
    lines.append(
        f"### {EVIDENCE_KEY}\n"
        "Array of {\"source\": <baseline_text|edited_text|control_text>, "
        "\"quote\": \"<verbatim substring of source>\"}. "
        "At least one quote per non-zero rubric dimension is required."
    )
    return "\n".join(lines)


def output_json_schema() -> dict[str, Any]:
    """Return the JSON Schema the LLM must conform to (for OpenAI structured output)."""
    properties: dict[str, Any] = {
        "case_id": {"type": "string"},
        CONFIDENCE_KEY: {"type": "number", "minimum": 0.0, "maximum": 1.0},
        EVIDENCE_KEY: {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": sorted(ALLOWED_EVIDENCE_SOURCES),
                    },
                    "quote": {"type": "string", "minLength": 1},
                },
                "required": ["source", "quote"],
            },
        },
        RATIONALE_KEY: {"type": "string", "maxLength": 800},
    }
    for dim in RUBRIC_DEFINITIONS.values():
        properties[dim.key] = {
            "type": "integer",
            "minimum": dim.min_val,
            "maximum": dim.max_val,
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(REQUIRED_FIELDS),
    }


# ---------------------------------------------------------------------------
# Runtime validation (defensive: never trust the model's JSON shape).
# ---------------------------------------------------------------------------

class RubricValidationError(ValueError):
    """Raised when an LLM-judge output is unrecoverably malformed."""


def validate_judge_row(
    row: dict[str, Any],
    *,
    sources: dict[str, str] | None = None,
    case_id: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate (and lightly clean) one judge JSON row.

    Behavior:
        - Required keys must be present (raises in ``strict`` mode, otherwise
          fills sentinel values and records a warning in ``_warnings``).
        - Integer dimensions are clipped to their declared range.
        - Confidence is clipped to [0, 1].
        - Evidence quotes whose source isn't in ``ALLOWED_EVIDENCE_SOURCES``
          are dropped.
        - If ``sources`` is given (the actual baseline/edited/control texts),
          evidence quotes must appear verbatim in the named source. Quotes
          that fail this test are dropped (and logged in ``_warnings``).

    Returns:
        A new dict, never mutates the input.
    """
    out: dict[str, Any] = {}
    warnings: list[str] = []

    if case_id is not None:
        out["case_id"] = str(case_id)
    elif "case_id" in row:
        out["case_id"] = str(row["case_id"])
    else:
        if strict:
            raise RubricValidationError("missing case_id")
        out["case_id"] = "unknown"
        warnings.append("missing case_id; filled with 'unknown'")

    for dim in RUBRIC_DEFINITIONS.values():
        v = row.get(dim.key)
        if v is None:
            if strict:
                raise RubricValidationError(f"missing {dim.key}")
            warnings.append(f"missing {dim.key}; defaulted to {dim.min_val}")
            v = dim.min_val
        try:
            v_i = int(round(float(v)))
        except (TypeError, ValueError):
            if strict:
                raise RubricValidationError(f"non-numeric {dim.key}: {v!r}")
            warnings.append(f"non-numeric {dim.key}={v!r}; defaulted to {dim.min_val}")
            v_i = dim.min_val
        if v_i < dim.min_val or v_i > dim.max_val:
            warnings.append(f"clipped {dim.key} from {v_i}")
            v_i = max(dim.min_val, min(dim.max_val, v_i))
        out[dim.key] = v_i

    conf = row.get(CONFIDENCE_KEY)
    if conf is None:
        if strict:
            raise RubricValidationError(f"missing {CONFIDENCE_KEY}")
        warnings.append(f"missing {CONFIDENCE_KEY}; defaulted to 0.0")
        conf = 0.0
    try:
        conf_f = float(conf)
    except (TypeError, ValueError):
        warnings.append(f"non-numeric {CONFIDENCE_KEY}={conf!r}; defaulted to 0.0")
        conf_f = 0.0
    out[CONFIDENCE_KEY] = max(0.0, min(1.0, conf_f))

    raw_spans = row.get(EVIDENCE_KEY) or []
    cleaned_spans: list[dict[str, Any]] = []
    if not isinstance(raw_spans, list):
        if strict:
            raise RubricValidationError(f"{EVIDENCE_KEY} must be a list")
        warnings.append(f"{EVIDENCE_KEY} not a list; treated as empty")
        raw_spans = []
    for span in raw_spans:
        if not isinstance(span, dict):
            warnings.append("evidence span is not an object; dropped")
            continue
        src = str(span.get("source", "")).strip()
        quote = str(span.get("quote", "")).strip()
        if src not in ALLOWED_EVIDENCE_SOURCES:
            warnings.append(f"evidence source {src!r} not allowed; dropped")
            continue
        if not quote:
            warnings.append("empty quote dropped")
            continue
        if sources is not None:
            actual = sources.get(src, "")
            if quote not in actual:
                warnings.append(
                    f"quote {quote[:40]!r} not in {src}; dropped"
                )
                continue
        cleaned_spans.append({"source": src, "quote": quote})
    out[EVIDENCE_KEY] = cleaned_spans

    rationale = row.get(RATIONALE_KEY, "")
    out[RATIONALE_KEY] = str(rationale)[:1000]

    out["_warnings"] = warnings
    return out


def coerce_rubric_value(key: str, value: Any) -> int | float:
    """Public helper: coerce one rubric value to its declared range/type."""
    if key in RUBRIC_DEFINITIONS:
        dim = RUBRIC_DEFINITIONS[key]
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            v = dim.min_val
        return max(dim.min_val, min(dim.max_val, v))
    if key == CONFIDENCE_KEY:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.0
        return max(0.0, min(1.0, v))
    raise KeyError(key)


def all_rubric_keys() -> list[str]:
    return list(RUBRIC_DEFINITIONS.keys())

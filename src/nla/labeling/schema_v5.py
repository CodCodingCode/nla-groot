"""V5 nested slot labeling schema (one dict per timestep, three slots).

Each step label is a JSON object with keys ``image_patch``, ``last_text``, and
``anchor``.  Every slot carries required ``scene``, ``target``, and ``plan``
fields; ``spatial`` is optional.  Empty strings normalize to ``NA``.
``image_patch.plan`` must be ``NA``; ``last_text`` / ``anchor`` ``plan`` must be
non-``NA`` and contain ``:`` (``phase: detail`` form).
"""

from __future__ import annotations

import json
import re
from typing import Any

from nla.labeling.prompts import (
    V4_FORBIDDEN_HEADERS,
    V4_MOTOR_IMPERATIVE_PHRASES,
    V4_SCAFFOLD_FORBIDDEN_PHRASES,
)

SLOT_NAMES: tuple[str, ...] = ("image_patch", "last_text", "anchor")
REQUIRED_KEYS: tuple[str, ...] = ("scene", "target", "plan")
OPTIONAL_KEYS: tuple[str, ...] = ("spatial",)
ALLOWED_SLOT_KEYS: frozenset[str] = frozenset(REQUIRED_KEYS + OPTIONAL_KEYS)

_NA_CANONICAL = "NA"
_NA_VALUES = frozenset({"na", "n/a", "none", ""})
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Scaffold phrases apply to all fields; motor imperatives only to scene/target/spatial
# (valid in plan after "phase:" e.g. "grasp: move toward the bowl rim").
V5_SCAFFOLD_FORBIDDEN: tuple[str, ...] = V4_SCAFFOLD_FORBIDDEN_PHRASES
V5_MOTOR_FORBIDDEN: tuple[str, ...] = V4_MOTOR_IMPERATIVE_PHRASES
V5_FORBIDDEN_PHRASES: tuple[str, ...] = (
    V5_SCAFFOLD_FORBIDDEN + V5_MOTOR_FORBIDDEN
)

# Legacy V3 bullet headers (gripper:/motion:/image_region:) embedded in text.
_FORBIDDEN_HEADER_RE = re.compile(
    r"\b(" + "|".join(re.escape(h) for h in V4_FORBIDDEN_HEADERS) + r")\s*:",
    re.IGNORECASE,
)

_FORBIDDEN_TRAJECTORY_PHRASES: tuple[str, ...] = (
    "remaining steps",
    "overall trajectory",
    "over the next",
    "next 3 timesteps",
    "next three timesteps",
)

_PLAN_PHASE_RE = re.compile(
    r"^(approach|reach|grasp|pickup|lift|transport|place|release|retreat|"
    r"idle|align|open|close|carry|reorient|insert|pour)\s*:",
    re.IGNORECASE,
)


def _is_na(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in _NA_VALUES


def _norm_field(value: Any) -> str:
    if value is None:
        return _NA_CANONICAL
    text = str(value).strip()
    if not text or _is_na(text):
        return _NA_CANONICAL
    return text


def _normalize_slot(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {k: _NA_CANONICAL for k in REQUIRED_KEYS}
    out: dict[str, str] = {k: _norm_field(raw.get(k)) for k in REQUIRED_KEYS}
    if "spatial" in raw:
        out["spatial"] = _norm_field(raw.get("spatial"))
    return out


def _check_forbidden_phrases(slot: str, field: str, text: str, errors: list[str]) -> None:
    if _is_na(text):
        return
    lower = text.lower()
    phrase_lists: tuple[tuple[str, ...], ...] = (V5_SCAFFOLD_FORBIDDEN,)
    if field != "plan":
        phrase_lists = phrase_lists + (V5_MOTOR_FORBIDDEN,)
    for phrases in phrase_lists:
        for phrase in phrases:
            if phrase.lower() in lower:
                errors.append(f"{slot}.{field}: forbidden phrase {phrase!r}")
    if _FORBIDDEN_HEADER_RE.search(text):
        errors.append(f"{slot}.{field}: forbidden V4 header category")
    for phrase in _FORBIDDEN_TRAJECTORY_PHRASES:
        if phrase in lower and field == "plan":
            errors.append(f"{slot}.{field}: forbidden trajectory phrase {phrase!r}")


def tokenize_label_text(text: str) -> set[str]:
    if not text or _is_na(text):
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard_sets(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cross_slot_jaccard(
    nested: dict[str, dict[str, str]],
    *,
    keys: tuple[str, ...] = ("scene", "target"),
) -> dict[str, float]:
    """Pairwise token Jaccard across slots for scene/target (A/B metrics)."""
    slots = [s for s in SLOT_NAMES if s in nested]
    scores: dict[str, float] = {}
    pair_jac: list[float] = []
    for i, a in enumerate(slots):
        for b in slots[i + 1 :]:
            inter = union = 0
            for key in keys:
                ta = tokenize_label_text(str(nested[a].get(key, "")))
                tb = tokenize_label_text(str(nested[b].get(key, "")))
                if not ta and not tb:
                    continue
                inter += len(ta & tb)
                union += len(ta | tb)
            jac = (inter / union) if union else 0.0
            scores[f"{a}_vs_{b}"] = jac
            pair_jac.append(jac)
    for key in keys:
        field_scores: list[float] = []
        for i, a in enumerate(slots):
            for b in slots[i + 1 :]:
                ta = tokenize_label_text(str(nested[a].get(key, "")))
                tb = tokenize_label_text(str(nested[b].get(key, "")))
                if not ta and not tb:
                    continue
                u = len(ta | tb)
                field_scores.append(len(ta & tb) / u if u else 0.0)
        if field_scores:
            scores[key] = sum(field_scores) / len(field_scores)
    if pair_jac:
        scores["mean"] = sum(pair_jac) / len(pair_jac)
    return scores


def _parse_description_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_nested_from_row(row: dict[str, Any]) -> dict[str, dict[str, str]] | None:
    """Pull a nested V5 object from a labels_steps.jsonl row."""
    if all(isinstance(row.get(s), dict) for s in SLOT_NAMES):
        return {s: dict(row[s]) for s in SLOT_NAMES}

    slots = row.get("slots")
    if isinstance(slots, dict) and all(isinstance(slots.get(s), dict) for s in SLOT_NAMES):
        return {s: dict(slots[s]) for s in SLOT_NAMES}

    for key in ("nested", "v5_nested", "label"):
        blob = row.get(key)
        if isinstance(blob, dict):
            found = extract_nested_from_row(blob)
            if found is not None:
                return found

    for text_key in ("raw_response", "description"):
        text = row.get(text_key)
        if not isinstance(text, str) or not text.strip():
            continue
        parsed = _parse_description_json(text)
        if parsed is not None:
            if all(isinstance(parsed.get(s), dict) for s in SLOT_NAMES):
                return {s: dict(parsed[s]) for s in SLOT_NAMES}
            inner = parsed.get("slots")
            if isinstance(inner, dict) and all(
                isinstance(inner.get(s), dict) for s in SLOT_NAMES
            ):
                return {s: dict(inner[s]) for s in SLOT_NAMES}
        try:
            from nla.labeling.prompts_v5 import parse_v5_response

            obj = parse_v5_response(text)
        except Exception:
            continue
        if isinstance(obj, dict) and all(isinstance(obj.get(s), dict) for s in SLOT_NAMES):
            return {s: dict(obj[s]) for s in SLOT_NAMES}
    return None


def render_slot_bullets(slot: dict[str, str]) -> str:
    """Render one slot as markdown bullets (omits NA optional spatial / plan)."""
    lines: list[str] = []
    for key in ("scene", "target", "spatial", "plan"):
        val = slot.get(key)
        if val is None or _is_na(str(val)):
            continue
        text = str(val).strip()
        if not text or text.upper() == _NA_CANONICAL:
            continue
        lines.append(f"- {key}: {text}")
    return "\n".join(lines)


def validate_nested(
    obj: Any,
    *,
    min_plan_chars: int = 8,
) -> tuple[bool, list[str], dict[str, dict[str, str]]]:
    """Validate and normalize a V5 step label.

    Always returns a normalized slot dict (best-effort) even when ``ok`` is
    False, so callers can log partial state or emit diagnostics.
    """
    errors: list[str] = []
    normalized: dict[str, dict[str, str]] = {}

    if not isinstance(obj, dict):
        return False, ["root must be a JSON object"], normalized

    for slot in SLOT_NAMES:
        raw = obj.get(slot)
        if not isinstance(raw, dict):
            errors.append(f"{slot}: missing or not an object")
            normalized[slot] = {k: _NA_CANONICAL for k in REQUIRED_KEYS}
            continue

        extra = set(raw.keys()) - ALLOWED_SLOT_KEYS
        if extra:
            errors.append(f"{slot}: unknown keys {sorted(extra)!r}")

        norm = _normalize_slot(raw)
        normalized[slot] = norm

        for key in REQUIRED_KEYS:
            _check_forbidden_phrases(slot, key, norm[key], errors)
        if "spatial" in norm:
            _check_forbidden_phrases(slot, "spatial", norm["spatial"], errors)

    if len(normalized) != len(SLOT_NAMES):
        return False, errors, normalized

    if not _is_na(normalized["image_patch"].get("plan", "")):
        errors.append("image_patch.plan: must be NA")

    for slot in ("last_text", "anchor"):
        plan = normalized[slot].get("plan", "")
        if _is_na(plan):
            errors.append(f"{slot}.plan: required, got NA")
        elif len(plan) < min_plan_chars:
            errors.append(f"{slot}.plan: too short")
        elif ":" not in plan:
            errors.append(f"{slot}.plan: must contain ':'")
        elif not _PLAN_PHASE_RE.match(plan.strip()):
            errors.append(f"{slot}.plan: needs phase prefix (phase: detail)")

    ok = len(errors) == 0
    return ok, errors, normalized


def nested_to_description_by_slot(
    nested: dict[str, dict[str, str]],
) -> dict[str, str]:
    return {slot: render_slot_bullets(nested[slot]) for slot in SLOT_NAMES}

"""Tests for V5 nested slot labeling schema."""

from __future__ import annotations

import json

import pytest

from nla.labeling.schema_v5 import (
    V5_FORBIDDEN_PHRASES,
    cross_slot_jaccard,
    extract_nested_from_row,
    render_slot_bullets,
    validate_nested,
)
from nla.labeling.prompts import V4_SCAFFOLD_FORBIDDEN_PHRASES

VALID = {
    "image_patch": {
        "scene": "Wooden table with a gray bowl and white-rim plate.",
        "target": "Bowl rim and gripper near center patch.",
        "plan": "NA",
        "spatial": "NA",
    },
    "last_text": {
        "scene": "Same tabletop arrangement from language slot.",
        "target": "Bowl as manipulation object.",
        "plan": "grasp: gripper descending toward bowl rim.",
        "spatial": "NA",
    },
    "anchor": {
        "scene": "Task scene with bowl and plate visible.",
        "target": "Bowl for pick-and-place subgoal.",
        "plan": "pickup: lifting bowl per instruction to place on plate.",
        "spatial": "NA",
    },
}


def test_v5_forbidden_imports_v4_scaffold():
    for phrase in V4_SCAFFOLD_FORBIDDEN_PHRASES:
        assert phrase in V5_FORBIDDEN_PHRASES


def test_valid_nested():
    ok, errs, norm = validate_nested(VALID)
    assert ok, errs
    assert norm["image_patch"]["plan"] == "NA"
    assert "scene:" in render_slot_bullets(norm["image_patch"])
    assert "plan:" not in render_slot_bullets(norm["image_patch"])


def test_empty_strings_normalize_to_na():
    raw = {k: dict(v) for k, v in VALID.items()}
    raw["image_patch"]["scene"] = ""
    raw["last_text"]["target"] = "  "
    ok, errs, norm = validate_nested(raw)
    assert ok, errs
    assert norm["image_patch"]["scene"] == "NA"
    assert norm["last_text"]["target"] == "NA"


def test_image_patch_plan_must_be_na():
    bad = {k: dict(v) for k, v in VALID.items()}
    bad["image_patch"]["plan"] = "grasp: no"
    ok, errs, norm = validate_nested(bad)
    assert not ok
    assert norm["image_patch"]["plan"] != "NA"
    assert any("image_patch.plan" in e for e in errs)


def test_last_text_plan_requires_colon():
    bad = {k: dict(v) for k, v in VALID.items()}
    bad["last_text"]["plan"] = "grasp only"
    ok, errs, _ = validate_nested(bad)
    assert not ok
    assert any("last_text.plan" in e for e in errs)


def test_anchor_plan_cannot_be_na():
    bad = {k: dict(v) for k, v in VALID.items()}
    bad["anchor"]["plan"] = "NA"
    ok, errs, _ = validate_nested(bad)
    assert not ok
    assert any("anchor.plan" in e for e in errs)


def test_forbidden_scaffold_phrase():
    bad = {k: dict(v) for k, v in VALID.items()}
    bad["last_text"]["scene"] = "The patch carries the bowl forward."
    ok, errs, _ = validate_nested(bad)
    assert not ok
    assert any("forbidden phrase" in e for e in errs)


def test_forbidden_header_gripper():
    bad = {k: dict(v) for k, v in VALID.items()}
    bad["anchor"]["target"] = "gripper: closing on bowl"
    ok, errs, _ = validate_nested(bad)
    assert not ok
    assert any("forbidden V4 header" in e for e in errs)


def test_missing_slot_still_returns_normalized():
    bad = {k: dict(v) for k, v in VALID.items()}
    del bad["anchor"]
    ok, errs, norm = validate_nested(bad)
    assert not ok
    assert "anchor" in norm
    assert any("anchor" in e for e in errs)


def test_cross_slot_jaccard_differs():
    ok, _, norm = validate_nested(VALID)
    assert ok
    j = cross_slot_jaccard(norm)
    assert "mean" in j
    assert j["mean"] < 0.95
    assert "scene" in j
    assert "target" in j


def test_cross_slot_jaccard_identical_high():
    nested = {
        "image_patch": {"scene": "red bowl on table", "target": "bowl"},
        "last_text": {"scene": "red bowl on table", "target": "bowl", "plan": "grasp: x"},
        "anchor": {"scene": "red bowl on table", "target": "bowl", "plan": "place: y"},
    }
    j = cross_slot_jaccard(nested)
    assert j["scene"] == pytest.approx(1.0)


def test_extract_nested_from_description_json():
    row = {"example_id": "x", "description": json.dumps(VALID)}
    extracted = extract_nested_from_row(row)
    assert extracted is not None
    assert extracted["image_patch"]["scene"] == VALID["image_patch"]["scene"]


def test_extract_nested_from_slots_key():
    row = {"example_id": "x", "slots": VALID}
    assert extract_nested_from_row(row) is not None

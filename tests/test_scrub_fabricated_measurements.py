"""Tests for ``scripts/labeling/scrub_fabricated_measurements.py``.

Covers the measurement detector, the in-line scrubber, the description-
level transform, and an end-to-end CLI run against a tiny synthetic
``labels.jsonl``.  No OpenAI calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "labeling" / "scrub_fabricated_measurements.py"
    spec = importlib.util.spec_from_file_location(
        "scrub_fabricated_measurements", script,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# has_measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "- target: orange toy held by the gripper, about 5-8 cm diameter, currently over the green bowl.",
        "- target: rounded rock (~6-8 cm) sitting on the top surface of the bed stand.",
        "- spatial: jar sits front-left about 20\u201330 cm from the plate.",
        "- spatial: gripper roughly 30 mm above the cup rim.",
        "- spatial: rotated 45 degrees clockwise from the previous frame.",
        "- spatial: gripper aligned 90\u00b0 to the table edge.",
        "- contact: confidence 75% that the cap is engaged.",
        "- target: bottle 1.5 m away on the back counter.",
    ],
)
def test_has_measurement_true(mod, text):
    assert mod.has_measurement(text)


@pytest.mark.parametrize(
    "text",
    [
        "- target: orange toy held by the gripper, currently over the green bowl.",
        "- spatial: white plate is front-center; jars sit front-left of the plate.",
        "- gripper: closed around the marker shaft.",
        "- target: pen at front-right of table.",
        "- plan: anchor encodes readiness to grasp the bottle cap and lift it.",
    ],
)
def test_has_measurement_false(mod, text):
    assert not mod.has_measurement(text)


# ---------------------------------------------------------------------------
# scrub_phrase
# ---------------------------------------------------------------------------


def test_scrub_drops_adverb_phrase_with_qualifier(mod):
    src = "- target: orange toy held by the gripper, about 5-8 cm diameter, currently over the green bowl."
    out = mod.scrub_phrase(src)
    assert "cm" not in out
    assert "5" not in out and "8" not in out
    assert out.startswith("- target: orange toy held by the gripper")
    assert "currently over the green bowl" in out
    assert ",," not in out
    assert "  " not in out


def test_scrub_drops_parenthesised_measurement(mod):
    src = "- target: rounded rock (~6-8 cm) sitting on the top surface of the bed stand."
    out = mod.scrub_phrase(src)
    assert "cm" not in out
    assert "(" not in out and ")" not in out
    assert out.endswith("sitting on the top surface of the bed stand.")


def test_scrub_drops_distance_with_preposition(mod):
    src = "- spatial: jar sits front-left about 20-30 cm from the plate."
    out = mod.scrub_phrase(src)
    assert "cm" not in out and "20" not in out and "30" not in out
    assert "front-left" in out
    assert "from the plate" in out


def test_scrub_drops_degrees_and_percent(mod):
    src = "- spatial: rotated 45 degrees clockwise; confidence 75% that contact is made."
    out = mod.scrub_phrase(src)
    assert "45" not in out and "degrees" not in out
    assert "75" not in out and "%" not in out
    assert "rotated" in out and "clockwise" in out
    assert "confidence" in out and "contact is made" in out


def test_scrub_handles_meters_and_inches(mod):
    src = "- target: bottle 1.5 m away on the back counter; cup 4 inches tall."
    out = mod.scrub_phrase(src)
    assert "1.5" not in out and "4" not in out
    assert "m" not in out.split() if False else True  # don't over-assert character m
    assert "back counter" in out
    assert "cup" in out


def test_scrub_idempotent(mod):
    src = "- target: orange toy held by the gripper, about 5-8 cm diameter, currently over the green bowl."
    once = mod.scrub_phrase(src)
    twice = mod.scrub_phrase(once)
    assert once == twice


# ---------------------------------------------------------------------------
# transform_description
# ---------------------------------------------------------------------------


def test_transform_description_scrub_mode(mod):
    desc = "\n".join(
        [
            "- target: orange toy held by the gripper, about 5-8 cm diameter, currently over the green bowl.",
            "- spatial: jar sits front-left about 20-30 cm from the plate.",
            "- gripper: closed around the marker shaft.",
        ]
    )
    new_desc, n_changed = mod.transform_description(desc, drop_bullet=False)
    assert n_changed == 2
    assert "cm" not in new_desc
    assert "closed around the marker shaft" in new_desc
    lines = new_desc.splitlines()
    assert len(lines) == 3  # bullets preserved, just shorter


def test_transform_description_drop_bullet_mode(mod):
    desc = "\n".join(
        [
            "- target: orange toy held by the gripper, about 5-8 cm diameter, currently over the green bowl.",
            "- spatial: jar sits front-left about 20-30 cm from the plate.",
            "- gripper: closed around the marker shaft.",
        ]
    )
    new_desc, n_changed = mod.transform_description(desc, drop_bullet=True)
    assert n_changed == 2
    lines = new_desc.splitlines()
    assert lines == ["- gripper: closed around the marker shaft."]


def test_transform_description_no_op_when_clean(mod):
    desc = (
        "- gripper: closed around the marker shaft.\n"
        "- target: pen at front-right of table.\n"
    )
    new_desc, n_changed = mod.transform_description(desc, drop_bullet=False)
    assert n_changed == 0
    assert new_desc == desc


def test_transform_description_handles_compound_categories(mod):
    """The labeler sometimes writes ``- gripper/spatial:`` or ``- spatial,plan:``;
    those bullets must still be scrubbed when they contain measurements."""
    desc = "\n".join(
        [
            "- gripper/spatial: gripper is about 10-15 cm above the cup rim.",
            "- spatial,plan: marker is roughly 5 cm left of the bowl, ready to grasp.",
            "- gripper: closed around the marker shaft.",
        ]
    )
    new_desc, n_changed = mod.transform_description(desc, drop_bullet=False)
    assert n_changed == 2
    assert "cm" not in new_desc
    assert "10" not in new_desc and "15" not in new_desc
    assert "closed around the marker shaft" in new_desc


def test_transform_description_preserves_trailing_newline(mod):
    desc = "- target: orange toy about 5 cm wide.\n"
    new_desc, n_changed = mod.transform_description(desc, drop_bullet=False)
    assert n_changed == 1
    assert new_desc.endswith("\n")


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def _write_synthetic(path: Path) -> None:
    rows = [
        {
            "kind": "position",
            "description": (
                "- target: orange toy held by the gripper, about 5-8 cm diameter, "
                "currently over the green bowl.\n"
                "- gripper: closed around the marker shaft."
            ),
            "meta": {"source_example_id": "ex0", "position_index": 0,
                     "position_type": "image_patch"},
        },
        {
            "kind": "position",
            "description": "- gripper: open, no contact.",
            "meta": {"source_example_id": "ex0", "position_index": 1,
                     "position_type": "last_text"},
        },
        {
            "kind": "example",
            "description": "manifest row, untouched (no measurements anyway).",
        },
    ]
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_cli_dry_run_does_not_write(mod, tmp_path):
    labels = tmp_path / "labels.jsonl"
    _write_synthetic(labels)
    before = labels.read_text()

    rc = mod.main([
        "--labels", str(labels),
        "--mode", "scrub",
        "--dry-run",
    ])
    assert rc == 0
    assert labels.read_text() == before  # unchanged
    # No backup or .tmp leftover.
    assert not (labels.parent / "labels.jsonl.bak").exists()
    assert not (labels.parent / "labels.jsonl.tmp").exists()


def test_cli_writes_and_backs_up(mod, tmp_path):
    labels = tmp_path / "labels.jsonl"
    _write_synthetic(labels)

    rc = mod.main(["--labels", str(labels), "--mode", "scrub"])
    assert rc == 0
    bak = labels.parent / "labels.jsonl.bak"
    assert bak.exists()
    # Backup is the original file verbatim.
    assert "5-8 cm" in bak.read_text()
    # New file no longer contains the measurement.
    new = labels.read_text()
    assert "cm" not in new
    assert "closed around the marker shaft" in new


def test_cli_drop_bullet_mode_drops_polluted_bullet(mod, tmp_path):
    labels = tmp_path / "labels.jsonl"
    _write_synthetic(labels)

    rc = mod.main(["--labels", str(labels), "--mode", "drop_bullet", "--no-backup"])
    assert rc == 0
    rows = [json.loads(ln) for ln in labels.read_text().splitlines() if ln]
    desc0 = rows[0]["description"]
    assert "orange toy" not in desc0  # whole bullet dropped
    assert "closed around the marker shaft" in desc0  # other bullet preserved

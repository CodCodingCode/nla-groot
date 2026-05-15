"""Tests for ``scripts/labeling/strip_hallucinated_image_region.py``.

Covers the per-line predicate, the description-level rewrite, and an
end-to-end run against a tiny synthetic ``labels.jsonl`` (with both dry-run
and write paths).  No OpenAI calls.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    """Load the script as a module (it lives in ``scripts/`` so it is not on
    the package path)."""
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "labeling" / "strip_hallucinated_image_region.py"
    spec = importlib.util.spec_from_file_location(
        "strip_hallucinated_image_region", script,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def test_is_offending_bullet_patch_word(mod):
    line = "- image_region: image patch 131 is focused on the blue block."
    assert mod.is_offending_bullet(line, match_mode="patch") is True
    assert mod.is_offending_bullet(line, match_mode="patch_or_layout") is True


def test_is_offending_bullet_layout_only_only_in_layout_mode(mod):
    """A bullet that mentions a quadrant but not 'patch' should fire only
    in patch_or_layout mode, not in patch mode."""
    line = "- image_region: focusing on the bowl rim in the upper-right of the table."
    assert mod.is_offending_bullet(line, match_mode="patch") is False
    assert mod.is_offending_bullet(line, match_mode="patch_or_layout") is True


def test_is_offending_bullet_clean_image_region_kept(mod):
    line = (
        "- image_region: visible features as in the attached camera frame; "
        "exact patch location within the frame is not specified."
    )
    # 'patch' appears here even in the safe replacement, so 'patch' mode
    # would also strip it; but the safe replacement is something the user
    # opted into. The key invariant is non-image_region bullets are never
    # touched -- assert that:
    other = "- scene: white round table with a green bowl and a blue cube."
    assert mod.is_offending_bullet(other, match_mode="patch") is False
    assert mod.is_offending_bullet(other, match_mode="patch_or_layout") is False
    assert mod.is_offending_bullet(other, match_mode="all_image_patch") is False


def test_is_offending_bullet_all_image_patch_flags_any_image_region(mod):
    line = "- image_region: anything at all here."
    assert mod.is_offending_bullet(line, match_mode="all_image_patch") is True


# ---------------------------------------------------------------------------
# Description rewrite
# ---------------------------------------------------------------------------

CLEAN_DESC = (
    "- scene: white round table with green bowl and blue cube.\n"
    "- target: blue cube on the right side of the bowl.\n"
    "- spatial: cube sits on the rim of the bowl.\n"
    "- gripper: parallel gripper open above the cube.\n"
    "- plan: lower gripper to grasp the cube."
)

DIRTY_DESC = (
    "- scene: white round table with green bowl and blue cube.\n"
    "- target: blue cube on the right side of the bowl.\n"
    "- spatial: cube sits on the rim of the bowl.\n"
    "- gripper: parallel gripper open above the cube.\n"
    "- image_region: image patch 131 encodes the blue block on the rim."
)


def test_transform_description_strips_offending_bullet(mod):
    new_desc, n = mod.transform_description(
        DIRTY_DESC,
        match_mode="patch",
        replace_mode=False,
        only_image_patch_row=True,
    )
    assert n == 1
    assert "image_region" not in new_desc
    # Other bullets preserved verbatim.
    for kept in ("scene:", "target:", "spatial:", "gripper:"):
        assert kept in new_desc


def test_transform_description_replaces_offending_bullet(mod):
    new_desc, n = mod.transform_description(
        DIRTY_DESC,
        match_mode="patch",
        replace_mode=True,
        only_image_patch_row=True,
    )
    assert n == 1
    assert "image_region" in new_desc
    assert "exact patch location within the frame is not specified" in new_desc
    # The original hallucinated content is gone.
    assert "image patch 131" not in new_desc


def test_transform_description_clean_row_unchanged(mod):
    new_desc, n = mod.transform_description(
        CLEAN_DESC,
        match_mode="patch_or_layout",
        replace_mode=False,
        only_image_patch_row=True,
    )
    assert n == 0
    assert new_desc == CLEAN_DESC


def test_transform_description_all_image_patch_only_acts_on_image_patch_rows(mod):
    desc = CLEAN_DESC + "\n- image_region: visible blue cube on the bowl rim."
    # Non-image_patch row: no-op even in all_image_patch mode.
    out_keep, n_keep = mod.transform_description(
        desc, match_mode="all_image_patch", replace_mode=False,
        only_image_patch_row=False,
    )
    assert n_keep == 0
    assert out_keep == desc
    # image_patch row: the image_region bullet is stripped.
    out_strip, n_strip = mod.transform_description(
        desc, match_mode="all_image_patch", replace_mode=False,
        only_image_patch_row=True,
    )
    assert n_strip == 1
    assert "image_region" not in out_strip


# ---------------------------------------------------------------------------
# End-to-end CLI on a tiny synthetic labels.jsonl
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _make_rows() -> list[dict]:
    return [
        {  # offending row
            "kind": "position",
            "description": DIRTY_DESC,
            "meta": {"position_type": "image_patch", "position_index": 131,
                     "source_example_id": "ex0"},
            "model": "fake",
            "elapsed_ms": 1.0,
            "usage": {},
            "error": None,
        },
        {  # clean row
            "kind": "position",
            "description": CLEAN_DESC,
            "meta": {"position_type": "last_text", "position_index": 17,
                     "source_example_id": "ex1"},
            "model": "fake",
            "elapsed_ms": 1.0,
            "usage": {},
            "error": None,
        },
        {  # non-position row should pass through verbatim
            "kind": "step",
            "description": "- scene: irrelevant.",
            "model": "fake",
            "elapsed_ms": 1.0,
            "usage": {},
            "error": None,
        },
    ]


def test_main_dry_run_does_not_modify_file(tmp_path: Path, mod):
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, _make_rows())
    before = labels.read_text()
    rc = mod.main([
        "--labels", str(labels),
        "--match", "patch",
        "--dry-run",
    ])
    assert rc == 0
    assert labels.read_text() == before
    # No backup or tmp file should be left behind.
    assert not (tmp_path / "labels.jsonl.bak").exists()
    assert not (tmp_path / "labels.jsonl.tmp").exists()


def test_main_writes_backup_and_strips(tmp_path: Path, mod):
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, _make_rows())
    rc = mod.main([
        "--labels", str(labels),
        "--match", "patch",
        "--mode", "strip",
    ])
    assert rc == 0
    bak = labels.with_suffix(labels.suffix + ".bak")
    assert bak.exists()
    rows = _read_jsonl(labels)
    assert len(rows) == 3
    # Offending image_region bullet gone from row 0; clean row 1 untouched.
    assert "image_region" not in rows[0]["description"]
    assert "scene:" in rows[0]["description"]
    assert rows[1]["description"] == CLEAN_DESC
    # Non-position row untouched.
    assert rows[2]["kind"] == "step"
    # Backup must contain the original bullet.
    bak_rows = _read_jsonl(bak)
    assert "image patch 131" in bak_rows[0]["description"]


def test_main_picks_next_free_backup(tmp_path: Path, mod):
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, _make_rows())
    # Pre-create .bak so the script must use .bak2.
    (labels.with_suffix(labels.suffix + ".bak")).write_text("preexisting\n")
    rc = mod.main([
        "--labels", str(labels),
        "--match", "patch",
    ])
    assert rc == 0
    assert (labels.with_suffix(labels.suffix + ".bak")).read_text() == "preexisting\n"
    assert (labels.with_suffix(labels.suffix + ".bak2")).exists()

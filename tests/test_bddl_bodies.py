"""Tests for BDDL body validation used in CF pair mining."""

from __future__ import annotations

from pathlib import Path

import pytest

from nla.eval.steerability.bddl_bodies import (
    DEFAULT_GOAL_BDDL_DIR,
    filter_tasks_with_bodies_in_bddl,
    missing_bodies_for_task,
    parse_bddl_instance_names,
    validate_cf_target_bodies,
)


GOAL_BDDL = DEFAULT_GOAL_BDDL_DIR / "put_the_bowl_on_the_plate.bddl"


@pytest.mark.skipif(not GOAL_BDDL.exists(), reason="LIBERO BDDL not present")
def test_parse_goal_bddl_has_expected_instances():
    names = parse_bddl_instance_names(str(GOAL_BDDL))
    assert "akita_black_bowl_1" in names
    assert "plate_1" in names
    assert "wine_rack_1" in names  # fixture
    assert "flat_stove_1" in names


@pytest.mark.skipif(not GOAL_BDDL.exists(), reason="LIBERO BDDL not present")
def test_put_bowl_on_plate_bodies_present():
    missing = missing_bodies_for_task(
        "put_the_bowl_on_the_plate", DEFAULT_GOAL_BDDL_DIR
    )
    assert missing == []


def test_validate_cf_target_bodies_rejects_missing_body(tmp_path: Path):
    bddl = tmp_path / "put_the_wine_bottle_on_the_rack.bddl"
    bddl.write_text(
        "(:objects\n  plate_1 - plate\n)\n(:fixtures\n  flat_stove_1 - flat_stove\n)\n"
    )
    issues = validate_cf_target_bodies(
        "put_the_wine_bottle_on_the_rack",
        "libero_sim/put_the_wine_bottle_on_the_rack",
        tmp_path,
    )
    assert any("wine_bottle_1" in i for i in issues)


def test_validate_cf_target_bodies_rejects_env_mismatch(tmp_path: Path):
    issues = validate_cf_target_bodies(
        "put_the_bowl_on_the_plate",
        "libero_sim/wrong_env",
        tmp_path,
    )
    assert any("target_env_name" in i for i in issues)


@pytest.mark.skipif(not GOAL_BDDL.exists(), reason="LIBERO BDDL not present")
def test_all_goal_tasks_pass_body_filter():
    from nla.eval.steerability.predicates import GOAL_TASKS

    tasks = list(GOAL_TASKS.keys())
    valid = filter_tasks_with_bodies_in_bddl(tasks, DEFAULT_GOAL_BDDL_DIR)
    assert set(valid) == set(tasks)

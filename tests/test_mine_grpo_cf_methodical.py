"""Unit tests for methodical CF pair mining (no GPT)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_methodical_module():
    path = _REPO / "scripts/training/mine_grpo_counterfactual_pairs_methodical.py"
    spec = importlib.util.spec_from_file_location("mine_grpo_cf_methodical", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_methodical_module()


def test_pick_bowl_always_preserve_matching_fraction_one(m):
    import random

    rng = random.Random(3)
    for _ in range(20):
        tgt, reason = m.pick_methodical_target(
            rng,
            src="put_the_bowl_on_the_plate",
            matching_fraction=1.0,
            prefer_site_swap_prob=0.5,
            exclude_joint_proxy_targets=True,
        )
        assert tgt == "put_the_bowl_on_the_plate"
        assert reason == "preserve_behavior"


def test_site_swap_keeps_same_source_body(m):
    import random

    rng = random.Random(0)
    src = "put_the_bowl_on_the_plate"
    tgt, reason = m.pick_methodical_target(
        rng,
        src=src,
        matching_fraction=0.0,
        prefer_site_swap_prob=1.0,
        exclude_joint_proxy_targets=True,
    )
    if reason == "site_swap":
        assert m.GOAL_TASKS[tgt].source_body == m.GOAL_TASKS[src].source_body


def test_validate_pair_row_good(m):
    canon = m._canonical_to_instruction()
    row = {
        "source_task": "put_the_bowl_on_the_plate",
        "target_task": "put_the_bowl_on_the_plate",
        "target_intent": canon["put_the_bowl_on_the_plate"],
        "target_env_name": "libero_sim/put_the_bowl_on_the_plate",
        "is_counterfactual": False,
    }
    assert m.validate_pair_row(row, canon) == []


def test_validate_bad_env(m):
    canon = m._canonical_to_instruction()
    row = {
        "source_task": "put_the_bowl_on_the_plate",
        "target_task": "put_the_bowl_on_the_stove",
        "target_intent": canon["put_the_bowl_on_the_stove"],
        "target_env_name": "wrong",
        "is_counterfactual": True,
    }
    issues = m.validate_pair_row(row, canon)
    assert any("target_env_name" in i for i in issues)


def test_validate_pair_row_good_bodies(m):
    canon = m._canonical_to_instruction()
    bddl = m.DEFAULT_GOAL_BDDL_DIR / "put_the_wine_bottle_on_the_rack.bddl"
    if not bddl.exists():
        pytest.skip("LIBERO BDDL not present")
    row = {
        "source_task": "put_the_bowl_on_the_plate",
        "target_task": "put_the_wine_bottle_on_the_rack",
        "target_intent": canon["put_the_wine_bottle_on_the_rack"],
        "target_env_name": "libero_sim/put_the_wine_bottle_on_the_rack",
        "is_counterfactual": True,
    }
    assert m.validate_pair_row(row, canon) == []

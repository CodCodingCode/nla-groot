"""Unit tests for nla.eval.steerability.predicates.

These tests synthesize tiny in-memory ``ObjectStateLogger.to_dict()``-shaped
trajectories so we can exercise every predicate / shaping path without
booting a LIBERO sim.
"""

from __future__ import annotations

import numpy as np
import pytest

from nla.eval.steerability.predicates import (
    DEFAULT_SHAPING,
    GOAL_TASKS,
    TASK_ALIASES,
    ShapingWeights,
    predicate_contact_with_source,
    predicate_displacement_only,
    predicate_fires,
    predicate_on_xy_z,
    resolve_task,
    score,
    shaping_gripper_near_source,
    shaping_min_ee_to_source,
    shaping_source_displacement,
    tracked_bodies_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _traj(
    *,
    bodies: dict[str, np.ndarray],
    initial: dict[str, np.ndarray] | None = None,
    ee: np.ndarray | None = None,
    gripper: np.ndarray | None = None,
) -> dict:
    """Build a minimal trajectory dict in the shape ``ObjectStateLogger.to_dict()`` produces."""
    out: dict = {"body_xpos": {k: np.asarray(v, dtype=np.float32) for k, v in bodies.items()}}
    if initial is not None:
        out["initial_body_pos"] = {k: np.asarray(v, dtype=np.float32) for k, v in initial.items()}
    if ee is not None:
        out["ee_pos"] = np.asarray(ee, dtype=np.float32)
    if gripper is not None:
        out["gripper_width"] = np.asarray(gripper, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# Task lookup
# ---------------------------------------------------------------------------


def test_resolve_task_canonical():
    spec = resolve_task("put_the_bowl_on_the_plate")
    assert spec.name == "put_the_bowl_on_the_plate"
    assert spec.source_body == "akita_black_bowl_1"
    assert spec.destination == "plate_1"


def test_resolve_task_instruction_alias():
    spec = resolve_task("put the bowl on the plate")
    assert spec.name == "put_the_bowl_on_the_plate"


def test_resolve_task_case_and_whitespace_tolerant():
    spec = resolve_task("  Put The Bowl On The Plate  ")
    assert spec.name == "put_the_bowl_on_the_plate"


def test_resolve_task_underscore_fallback():
    # GOAL_TASKS already has this id, but the fallback should also handle
    # an instruction that pre-substitutes underscores.
    spec = resolve_task("turn_on_the_stove")
    assert spec.name == "turn_on_the_stove"


def test_resolve_task_unknown_raises():
    with pytest.raises(KeyError):
        resolve_task("eat the chocolate")


def test_all_aliases_resolve():
    for alias in TASK_ALIASES:
        assert resolve_task(alias).name == TASK_ALIASES[alias]


def test_all_goal_tasks_have_specs_consistent():
    # Every joint-state fallback task must declare its predicate_kind so
    # predicate_fires can dispatch.
    for name, spec in GOAL_TASKS.items():
        assert spec.predicate_kind in {
            "on_xy_z", "displacement_only", "contact_with_source",
        }, name
        if spec.predicate_kind == "on_xy_z":
            assert spec.source_body and spec.destination, name


# ---------------------------------------------------------------------------
# Predicate primitives
# ---------------------------------------------------------------------------


def test_on_xy_z_fires_when_source_lands_on_destination():
    spec = resolve_task("put_the_bowl_on_the_plate")
    # Plate stays put; bowl starts far away then lands directly above.
    plate = np.tile([[0.5, 0.0, 0.05]], (10, 1))
    bowl = np.array([
        [0.20, 0.30, 0.05],
        [0.30, 0.20, 0.05],
        [0.40, 0.10, 0.07],
        [0.50, 0.00, 0.10],   # xy hit, z within tol
        [0.51, 0.01, 0.12],
    ] + [[0.5, 0.0, 0.12]] * 5)
    traj = _traj(bodies={"plate_1": plate, "akita_black_bowl_1": bowl})
    assert predicate_on_xy_z(traj, spec) is True
    assert predicate_fires(traj, spec) is True


def test_on_xy_z_misses_when_too_far_horizontally():
    spec = resolve_task("put_the_bowl_on_the_plate")
    plate = np.tile([[0.5, 0.0, 0.05]], (5, 1))
    # Bowl never approaches plate xy within tol.
    bowl = np.tile([[0.2, 0.3, 0.10]], (5, 1))
    traj = _traj(bodies={"plate_1": plate, "akita_black_bowl_1": bowl})
    assert predicate_on_xy_z(traj, spec) is False


def test_on_xy_z_misses_when_too_high():
    spec = resolve_task("put_the_bowl_on_the_plate")
    plate = np.tile([[0.5, 0.0, 0.05]], (5, 1))
    # Bowl xy matches but z way above tol (1m up).
    bowl = np.tile([[0.5, 0.0, 1.05]], (5, 1))
    traj = _traj(bodies={"plate_1": plate, "akita_black_bowl_1": bowl})
    assert predicate_on_xy_z(traj, spec) is False


def test_on_xy_z_handles_missing_body_gracefully():
    spec = resolve_task("put_the_bowl_on_the_plate")
    traj = _traj(bodies={"plate_1": np.zeros((3, 3))})  # bowl missing
    assert predicate_on_xy_z(traj, spec) is False


def test_displacement_only_fires_above_threshold():
    spec = GOAL_TASKS["open_the_middle_drawer_of_the_cabinet"]
    # Cabinet moves >> 4cm from initial.
    init = {"wooden_cabinet_1": [0.0, 0.0, 0.0]}
    moved = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]])
    traj = _traj(bodies={"wooden_cabinet_1": moved}, initial=init)
    assert predicate_displacement_only(traj, spec) is True


def test_displacement_only_misses_when_static():
    spec = GOAL_TASKS["open_the_middle_drawer_of_the_cabinet"]
    init = {"wooden_cabinet_1": [0.0, 0.0, 0.0]}
    static = np.tile([[0.001, 0.0, 0.0]], (5, 1))  # tiny jitter
    traj = _traj(bodies={"wooden_cabinet_1": static}, initial=init)
    assert predicate_displacement_only(traj, spec) is False


def test_contact_with_source_fires_when_ee_near():
    spec = GOAL_TASKS["turn_on_the_stove"]
    stove = np.tile([[0.5, 0.0, 0.05]], (3, 1))
    ee = np.array([[1.0, 0.0, 0.5], [0.55, 0.02, 0.06], [1.0, 0.0, 0.5]])  # near at t=1
    traj = _traj(bodies={"flat_stove_1": stove}, ee=ee)
    assert predicate_contact_with_source(traj, spec) is True


def test_contact_with_source_misses_when_far():
    spec = GOAL_TASKS["turn_on_the_stove"]
    stove = np.tile([[0.5, 0.0, 0.05]], (3, 1))
    ee = np.tile([[1.5, 0.5, 0.5]], (3, 1))
    traj = _traj(bodies={"flat_stove_1": stove}, ee=ee)
    assert predicate_contact_with_source(traj, spec) is False


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def test_shaping_min_ee_to_source_full_credit_at_contact():
    spec = resolve_task("put_the_bowl_on_the_plate")
    bowl = np.tile([[0.5, 0.0, 0.05]], (3, 1))
    ee = np.array([[0.6, 0.0, 0.5], [0.50, 0.0, 0.05], [0.6, 0.0, 0.5]])  # touches at t=1
    traj = _traj(bodies={"akita_black_bowl_1": bowl, "plate_1": np.zeros((3, 3))}, ee=ee)
    r = shaping_min_ee_to_source(traj, spec)
    assert r == pytest.approx(1.0)


def test_shaping_min_ee_to_source_zero_when_far():
    spec = resolve_task("put_the_bowl_on_the_plate")
    bowl = np.tile([[0.0, 0.0, 0.0]], (3, 1))
    ee = np.tile([[1.0, 1.0, 1.0]], (3, 1))
    traj = _traj(bodies={"akita_black_bowl_1": bowl, "plate_1": np.zeros((3, 3))}, ee=ee)
    assert shaping_min_ee_to_source(traj, spec) == pytest.approx(0.0)


def test_shaping_source_displacement_clamps_to_one():
    spec = resolve_task("put_the_bowl_on_the_plate")
    init = {"akita_black_bowl_1": [0.0, 0.0, 0.0]}
    bowl = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 1m move >> 0.3m disp_max
    traj = _traj(bodies={"akita_black_bowl_1": bowl}, initial=init)
    assert shaping_source_displacement(traj, spec) == pytest.approx(1.0)


def test_shaping_source_displacement_zero_when_static():
    spec = resolve_task("put_the_bowl_on_the_plate")
    init = {"akita_black_bowl_1": [0.0, 0.0, 0.0]}
    bowl = np.tile([[0.0, 0.0, 0.0]], (3, 1))
    traj = _traj(bodies={"akita_black_bowl_1": bowl}, initial=init)
    assert shaping_source_displacement(traj, spec) == pytest.approx(0.0)


def test_shaping_gripper_near_source_counts_steps_in_contact():
    spec = resolve_task("put_the_bowl_on_the_plate")
    bowl = np.tile([[0.5, 0.0, 0.05]], (4, 1))
    ee = np.tile([[0.5, 0.0, 0.05]], (4, 1))  # always touching
    # Closed only on half the steps.
    gripper = np.array([0.10, 0.02, 0.02, 0.10])
    traj = _traj(bodies={"akita_black_bowl_1": bowl, "plate_1": np.zeros((4, 3))}, ee=ee, gripper=gripper)
    frac = shaping_gripper_near_source(traj, spec)
    assert frac == pytest.approx(0.5)


def test_shaping_handles_missing_arrays_returns_zero():
    spec = resolve_task("put_the_bowl_on_the_plate")
    traj = _traj(bodies={"plate_1": np.zeros((3, 3))})
    assert shaping_min_ee_to_source(traj, spec) == 0.0
    assert shaping_source_displacement(traj, spec) == 0.0
    assert shaping_gripper_near_source(traj, spec) == 0.0


# ---------------------------------------------------------------------------
# Combined score()
# ---------------------------------------------------------------------------


def test_score_success_strictly_dominates_partial_progress():
    spec = resolve_task("put_the_bowl_on_the_plate")
    plate = np.tile([[0.5, 0.0, 0.05]], (5, 1))

    # Trajectory A: gripper hovers near bowl but bowl never reaches plate.
    bowl_partial = np.array([[0.10, 0.0, 0.05]] * 5)
    init = {"akita_black_bowl_1": [0.10, 0.0, 0.05]}
    ee_close = np.tile([[0.10, 0.0, 0.05]], (5, 1))
    grip = np.array([0.02] * 5)
    traj_partial = _traj(
        bodies={"akita_black_bowl_1": bowl_partial, "plate_1": plate},
        initial=init, ee=ee_close, gripper=grip,
    )

    # Trajectory B: bowl lands on plate.
    bowl_succ = np.array([
        [0.10, 0.0, 0.05], [0.20, 0.0, 0.05], [0.30, 0.0, 0.07],
        [0.50, 0.0, 0.10], [0.50, 0.0, 0.12],  # at t=3 xy hit, z in tol
    ])
    init_b = {"akita_black_bowl_1": [0.10, 0.0, 0.05]}
    ee_succ = bowl_succ.copy()
    traj_succ = _traj(
        bodies={"akita_black_bowl_1": bowl_succ, "plate_1": plate},
        initial=init_b, ee=ee_succ, gripper=grip,
    )

    s_partial = score(traj_partial, "put_the_bowl_on_the_plate")
    s_succ = score(traj_succ, "put_the_bowl_on_the_plate")

    assert s_partial["predicate"] == 0.0
    assert s_succ["predicate"] == 1.0
    assert s_succ["r"] > s_partial["r"], (s_succ, s_partial)
    # Big chunk of the gap is the predicate weight.
    assert (s_succ["r"] - s_partial["r"]) >= DEFAULT_SHAPING.w_predicate * 0.5


def test_score_returns_all_expected_fields():
    plate = np.tile([[0.5, 0.0, 0.05]], (3, 1))
    bowl = np.tile([[0.10, 0.0, 0.05]], (3, 1))
    traj = _traj(
        bodies={"akita_black_bowl_1": bowl, "plate_1": plate},
        initial={"akita_black_bowl_1": [0.10, 0.0, 0.05]},
        ee=bowl, gripper=np.array([0.02, 0.02, 0.02]),
    )
    s = score(traj, "put_the_bowl_on_the_plate")
    assert set(s) == {"r", "predicate", "r_dist", "r_displace", "r_contact", "target_task"}
    assert s["target_task"] == "put_the_bowl_on_the_plate"


def test_score_accepts_instruction_text():
    plate = np.tile([[0.5, 0.0, 0.05]], (3, 1))
    bowl = np.tile([[0.10, 0.0, 0.05]], (3, 1))
    traj = _traj(
        bodies={"akita_black_bowl_1": bowl, "plate_1": plate},
        initial={"akita_black_bowl_1": [0.10, 0.0, 0.05]},
    )
    s = score(traj, "put the bowl on the plate")  # alias text
    assert s["target_task"] == "put_the_bowl_on_the_plate"


def test_score_custom_weights():
    # If we crank w_predicate to 100x, success should dominate any shaping.
    plate = np.tile([[0.5, 0.0, 0.05]], (3, 1))
    bowl = np.array([[0.5, 0.0, 0.05]] * 3)  # already on plate
    init = {"akita_black_bowl_1": [0.5, 0.0, 0.05]}
    traj = _traj(bodies={"akita_black_bowl_1": bowl, "plate_1": plate}, initial=init)
    w = ShapingWeights(w_predicate=100.0, w_dist=0.0, w_displace=0.0, w_contact=0.0)
    s = score(traj, "put_the_bowl_on_the_plate", weights=w)
    assert s["r"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# tracked_bodies_for
# ---------------------------------------------------------------------------


def test_tracked_bodies_for_includes_defaults_plus_task_specific():
    out = tracked_bodies_for("put_the_bowl_on_the_plate")
    assert "akita_black_bowl_1" in out
    assert "plate_1" in out
    # No duplicates.
    assert len(out) == len(set(out))


def test_tracked_bodies_for_joint_state_task_still_returns_defaults():
    out = tracked_bodies_for("turn_on_the_stove")
    assert "flat_stove_1" in out

"""Per-task success predicates + dense shaping for LIBERO Goal.

These predicates score a rollout against *any* target task in the LIBERO Goal
suite, regardless of which BDDL was loaded into the simulator. That is the
critical capability for sim-success GRPO: we steer the policy with an
arbitrary target intent (e.g. "pick up the wine bottle") even when the
underlying env was loaded with a different BDDL (e.g. ``put_the_bowl_on_the_plate``),
and need a reward signal that fires for the *steered* intent rather than the
loaded one.

The predicates work from per-step body xyz positions logged by
:class:`nla.eval.steerability.object_logger.ObjectStateLogger`. The logger
exposes a ``to_dict()`` trajectory; we read ``body_xpos`` (per-step, per-body
xyz) and ``ee_pos`` and compute:

  * ``r_predicate`` - +1 if the target task's "done" condition is met at any
    point in the trajectory, else 0. The condition is a small Python rule
    over xyz positions (e.g. "bowl xy within R of plate xy AND bowl z within
    Z of plate z").
  * Dense shaping terms (all in [0, 1]):
      - ``r_dist``    : 1 - min gripper-to-source distance / d_max
      - ``r_displace``: src object displacement from initial / disp_max
      - ``r_contact`` : fraction of steps with gripper closed near source

Combined as ``r = w_predicate * r_predicate + w_dist * r_dist + ...``.

Two LIBERO Goal tasks (``turn_on_the_stove``, ``open_the_middle_drawer_of_the_cabinet``)
require joint-state queries that aren't in our position log, so they fall back
to a displacement-only proxy: did the relevant body move appreciably?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


# ----------------------------------------------------------------------------
# Task -> (source body, destination body or region) table.
# Body names match the LIBERO BDDL :objects block.
# ----------------------------------------------------------------------------

# Aliases the labels' free-text "instruction" maps to the canonical task id.
TASK_ALIASES: dict[str, str] = {
    "put the bowl on the plate":                       "put_the_bowl_on_the_plate",
    "put the bowl on the stove":                       "put_the_bowl_on_the_stove",
    "put the bowl on top of the cabinet":              "put_the_bowl_on_top_of_the_cabinet",
    "put the bowl on the top of the drawer":           "put_the_bowl_on_top_of_the_cabinet",
    "put the wine bottle on the rack":                 "put_the_wine_bottle_on_the_rack",
    "put the wine bottle on top of the cabinet":       "put_the_wine_bottle_on_top_of_the_cabinet",
    "put the wine bottle on the top of the drawer":    "put_the_wine_bottle_on_top_of_the_cabinet",
    "put the cream cheese in the bowl":                "put_the_cream_cheese_in_the_bowl",
    "put the cream cheese on the bowl":                "put_the_cream_cheese_in_the_bowl",
    "push the plate to the front of the stove":        "push_the_plate_to_the_front_of_the_stove",
    "open the top drawer and put the bowl inside":     "open_the_top_drawer_and_put_the_bowl_inside",
    "open the top layer of the drawer and put the bowl inside": "open_the_top_drawer_and_put_the_bowl_inside",
    "open the middle drawer of the cabinet":           "open_the_middle_drawer_of_the_cabinet",
    "open the middle layer of the drawer":             "open_the_middle_drawer_of_the_cabinet",
    "turn on the stove":                               "turn_on_the_stove",
}


@dataclass(frozen=True)
class TaskSpec:
    """Static metadata for one LIBERO Goal task.

    ``source_body``    - object the robot must move (or interact with).
    ``destination``    - body that ``source_body`` must end up on / in / near.
    ``xy_tol``         - horizontal distance threshold for "on top of" in meters.
    ``z_tol_min/max``  - vertical offset window of source above destination.
    ``predicate_kind`` - which predicate function to use ("on_xy_z",
                         "displacement_only", or "contact_with_source").
    """

    name: str
    source_body: str | None
    destination: str | None
    xy_tol: float = 0.06
    z_tol_min: float = -0.02
    z_tol_max: float = 0.18
    predicate_kind: str = "on_xy_z"


# All ten LIBERO Goal tasks.
GOAL_TASKS: dict[str, TaskSpec] = {
    "put_the_bowl_on_the_plate": TaskSpec(
        "put_the_bowl_on_the_plate",
        source_body="akita_black_bowl_1", destination="plate_1",
        xy_tol=0.06, z_tol_min=-0.02, z_tol_max=0.18,
    ),
    "put_the_bowl_on_the_stove": TaskSpec(
        "put_the_bowl_on_the_stove",
        source_body="akita_black_bowl_1", destination="flat_stove_1",
        xy_tol=0.09, z_tol_min=-0.02, z_tol_max=0.22,
    ),
    "put_the_bowl_on_top_of_the_cabinet": TaskSpec(
        "put_the_bowl_on_top_of_the_cabinet",
        source_body="akita_black_bowl_1", destination="wooden_cabinet_1",
        xy_tol=0.10, z_tol_min=0.04, z_tol_max=0.40,
    ),
    "put_the_wine_bottle_on_the_rack": TaskSpec(
        "put_the_wine_bottle_on_the_rack",
        source_body="wine_bottle_1", destination="wine_rack_1",
        xy_tol=0.10, z_tol_min=-0.05, z_tol_max=0.25,
    ),
    "put_the_wine_bottle_on_top_of_the_cabinet": TaskSpec(
        "put_the_wine_bottle_on_top_of_the_cabinet",
        source_body="wine_bottle_1", destination="wooden_cabinet_1",
        xy_tol=0.10, z_tol_min=0.04, z_tol_max=0.45,
    ),
    "put_the_cream_cheese_in_the_bowl": TaskSpec(
        "put_the_cream_cheese_in_the_bowl",
        source_body="cream_cheese_1", destination="akita_black_bowl_1",
        xy_tol=0.06, z_tol_min=-0.04, z_tol_max=0.14,
    ),
    "push_the_plate_to_the_front_of_the_stove": TaskSpec(
        "push_the_plate_to_the_front_of_the_stove",
        source_body="plate_1", destination="flat_stove_1",
        xy_tol=0.12, z_tol_min=-0.08, z_tol_max=0.10,
    ),
    "open_the_top_drawer_and_put_the_bowl_inside": TaskSpec(
        "open_the_top_drawer_and_put_the_bowl_inside",
        source_body="akita_black_bowl_1", destination="wooden_cabinet_1",
        xy_tol=0.10, z_tol_min=-0.10, z_tol_max=0.25,
    ),
    # Joint-state tasks: no source-to-destination predicate, fall back to
    # "did the cabinet body / joint visibly displace?". The shaping reward
    # will dominate the score.
    "open_the_middle_drawer_of_the_cabinet": TaskSpec(
        "open_the_middle_drawer_of_the_cabinet",
        source_body="wooden_cabinet_1", destination=None,
        predicate_kind="displacement_only",
    ),
    "turn_on_the_stove": TaskSpec(
        "turn_on_the_stove",
        source_body="flat_stove_1", destination=None,
        predicate_kind="contact_with_source",
    ),
}


# Default bodies the rollout's ``--tracked-bodies`` list should always
# include for any Goal task. Used by the worker to construct LiberoEnv.
DEFAULT_TRACKED_BODIES: tuple[str, ...] = (
    "akita_black_bowl_1",
    "plate_1",
    "wine_bottle_1",
    "wine_rack_1",
    "cream_cheese_1",
    "wooden_cabinet_1",
    "flat_stove_1",
)


# ----------------------------------------------------------------------------
# Task lookup
# ----------------------------------------------------------------------------


def resolve_task(name_or_instruction: str) -> TaskSpec:
    """Look up a TaskSpec by canonical task id OR free-text instruction.

    Accepts both ``"put_the_bowl_on_the_plate"`` (canonical) and
    ``"put the bowl on the plate"`` (the demo instruction text in
    ``labels.jsonl::meta.instruction``).
    """
    key = name_or_instruction.strip().lower()
    if key in GOAL_TASKS:
        return GOAL_TASKS[key]
    canonical = TASK_ALIASES.get(key)
    if canonical is not None:
        return GOAL_TASKS[canonical]
    # Tolerant fallback: try replacing spaces with underscores.
    alt = key.replace(" ", "_")
    if alt in GOAL_TASKS:
        return GOAL_TASKS[alt]
    raise KeyError(
        f"Unknown LIBERO Goal task {name_or_instruction!r}. "
        f"Known ids: {sorted(GOAL_TASKS)}; known aliases: {sorted(TASK_ALIASES)}"
    )


# ----------------------------------------------------------------------------
# Trajectory accessors
# ----------------------------------------------------------------------------


def _stack_xyz(d: Mapping[str, Any], body: str) -> np.ndarray | None:
    """Return a (T, 3) float32 array of body xyz, or None if absent."""
    xpos = d.get("body_xpos")
    if not isinstance(xpos, Mapping):
        return None
    arr = xpos.get(body)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _initial_xyz(d: Mapping[str, Any], body: str) -> np.ndarray | None:
    init = d.get("initial_body_pos")
    if not isinstance(init, Mapping):
        return None
    v = init.get(body)
    return None if v is None else np.asarray(v, dtype=np.float32)


def _ee_pos(d: Mapping[str, Any]) -> np.ndarray | None:
    ee = d.get("ee_pos")
    if ee is None:
        return None
    arr = np.asarray(ee, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _gripper_width(d: Mapping[str, Any]) -> np.ndarray | None:
    g = d.get("gripper_width")
    if g is None:
        return None
    arr = np.asarray(g, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


# ----------------------------------------------------------------------------
# Predicate primitives
# ----------------------------------------------------------------------------


def predicate_on_xy_z(
    trajectory: Mapping[str, Any],
    spec: TaskSpec,
) -> bool:
    """True if source ends up ~on top of destination at any point.

    "On top of" means: at some step ``t``, source xy is within ``spec.xy_tol``
    meters of destination xy AND source z is in the window
    ``[dest_z + z_tol_min, dest_z + z_tol_max]``. Uses the *per-step*
    destination position so we catch cases where the destination itself moves
    (e.g. the cabinet drawer).
    """
    if spec.source_body is None or spec.destination is None:
        return False
    src = _stack_xyz(trajectory, spec.source_body)
    dst = _stack_xyz(trajectory, spec.destination)
    if src is None or dst is None:
        return False
    n = min(src.shape[0], dst.shape[0])
    if n == 0:
        return False
    src, dst = src[:n], dst[:n]
    xy_d = np.linalg.norm(src[:, :2] - dst[:, :2], axis=1)
    z_off = src[:, 2] - dst[:, 2]
    hit = (xy_d <= spec.xy_tol) & (z_off >= spec.z_tol_min) & (z_off <= spec.z_tol_max)
    return bool(hit.any())


def predicate_displacement_only(
    trajectory: Mapping[str, Any],
    spec: TaskSpec,
    *,
    threshold_m: float = 0.04,
) -> bool:
    """Fallback for joint-state tasks: did source body move >= threshold?"""
    if spec.source_body is None:
        return False
    src = _stack_xyz(trajectory, spec.source_body)
    init = _initial_xyz(trajectory, spec.source_body)
    if src is None or init is None:
        return False
    disp = float(np.max(np.linalg.norm(src - init[None, :], axis=1)))
    return disp >= threshold_m


def predicate_contact_with_source(
    trajectory: Mapping[str, Any],
    spec: TaskSpec,
    *,
    near_m: float = 0.06,
) -> bool:
    """Fallback for joint-state tasks: did gripper get near source?"""
    if spec.source_body is None:
        return False
    src = _stack_xyz(trajectory, spec.source_body)
    ee = _ee_pos(trajectory)
    if src is None or ee is None:
        return False
    n = min(src.shape[0], ee.shape[0])
    if n == 0:
        return False
    d = np.linalg.norm(src[:n] - ee[:n], axis=1)
    return bool(np.min(d) <= near_m)


_PREDICATE_FNS: dict[str, Callable[..., bool]] = {
    "on_xy_z":               predicate_on_xy_z,
    "displacement_only":     predicate_displacement_only,
    "contact_with_source":   predicate_contact_with_source,
}


def predicate_fires(trajectory: Mapping[str, Any], spec: TaskSpec) -> bool:
    fn = _PREDICATE_FNS[spec.predicate_kind]
    return bool(fn(trajectory, spec))


# ----------------------------------------------------------------------------
# Dense shaping
# ----------------------------------------------------------------------------


def shaping_min_ee_to_source(
    trajectory: Mapping[str, Any],
    spec: TaskSpec,
    *,
    d_max: float = 0.30,
) -> float:
    """In [0, 1]. 1 means the gripper touched the source body."""
    if spec.source_body is None:
        return 0.0
    src = _stack_xyz(trajectory, spec.source_body)
    ee = _ee_pos(trajectory)
    if src is None or ee is None:
        return 0.0
    n = min(src.shape[0], ee.shape[0])
    if n == 0:
        return 0.0
    d_min = float(np.min(np.linalg.norm(src[:n] - ee[:n], axis=1)))
    return float(np.clip(1.0 - d_min / max(1e-6, d_max), 0.0, 1.0))


def shaping_source_displacement(
    trajectory: Mapping[str, Any],
    spec: TaskSpec,
    *,
    disp_max: float = 0.30,
) -> float:
    """In [0, 1]. 1 means source moved ``disp_max`` meters from initial pose."""
    if spec.source_body is None:
        return 0.0
    src = _stack_xyz(trajectory, spec.source_body)
    init = _initial_xyz(trajectory, spec.source_body)
    if src is None or init is None or src.shape[0] == 0:
        return 0.0
    disp = float(np.max(np.linalg.norm(src - init[None, :], axis=1)))
    return float(np.clip(disp / max(1e-6, disp_max), 0.0, 1.0))


def shaping_gripper_near_source(
    trajectory: Mapping[str, Any],
    spec: TaskSpec,
    *,
    near_m: float = 0.06,
    closed_thresh_m: float = 0.06,
) -> float:
    """Fraction of steps with gripper "closed-ish" while near source body."""
    if spec.source_body is None:
        return 0.0
    src = _stack_xyz(trajectory, spec.source_body)
    ee = _ee_pos(trajectory)
    g = _gripper_width(trajectory)
    if src is None or ee is None or g is None:
        return 0.0
    n = min(src.shape[0], ee.shape[0], g.shape[0])
    if n == 0:
        return 0.0
    d = np.linalg.norm(src[:n] - ee[:n], axis=1)
    near = d <= near_m
    closed = g[:n] <= closed_thresh_m
    both = near & closed
    return float(np.clip(np.mean(both.astype(np.float32)), 0.0, 1.0))


# ----------------------------------------------------------------------------
# Combined score
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapingWeights:
    """Default weights for ``score()``. Sum is unconstrained."""

    w_predicate: float = 2.0
    w_dist:      float = 0.5
    w_displace:  float = 0.3
    w_contact:   float = 0.2

    # Shaping clamp bounds (meters).
    d_max:           float = 0.30
    disp_max:        float = 0.30
    contact_near_m:  float = 0.06


DEFAULT_SHAPING = ShapingWeights()


def score(
    trajectory: Mapping[str, Any],
    target_task: str,
    *,
    weights: ShapingWeights | None = None,
) -> dict[str, float]:
    """Compute the sim-success reward + per-term breakdown for one rollout.

    Returns a dict with keys::

        r:           weighted total ((w_predicate * 1[fired])
                                     + w_dist * r_dist
                                     + w_displace * r_displace
                                     + w_contact * r_contact)
        predicate:   0.0 or 1.0
        r_dist:      [0, 1]
        r_displace:  [0, 1]
        r_contact:   [0, 1]
        target_task: canonical id of the resolved task

    All scalar reads are safe against missing / NaN xpos.
    """
    if weights is None:
        weights = DEFAULT_SHAPING
    spec = resolve_task(target_task)
    pred = 1.0 if predicate_fires(trajectory, spec) else 0.0
    r_dist = shaping_min_ee_to_source(trajectory, spec, d_max=weights.d_max)
    r_disp = shaping_source_displacement(trajectory, spec, disp_max=weights.disp_max)
    r_cont = shaping_gripper_near_source(trajectory, spec, near_m=weights.contact_near_m)
    total = (
        weights.w_predicate * pred
        + weights.w_dist     * r_dist
        + weights.w_displace * r_disp
        + weights.w_contact  * r_cont
    )
    return {
        "r":           float(total),
        "predicate":   float(pred),
        "r_dist":      float(r_dist),
        "r_displace":  float(r_disp),
        "r_contact":   float(r_cont),
        "target_task": spec.name,
    }


def tracked_bodies_for(target_task: str) -> list[str]:
    """Bodies that must be in the rollout's ``--tracked-bodies`` for this task.

    Always includes the default set so cross-task predicates also work (i.e.
    when the loaded BDDL differs from the target task).
    """
    spec = resolve_task(target_task)
    out = list(DEFAULT_TRACKED_BODIES)
    for b in (spec.source_body, spec.destination):
        if b is not None and b not in out:
            out.append(b)
    return out

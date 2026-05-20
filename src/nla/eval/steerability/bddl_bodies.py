"""Parse LIBERO BDDL scenes and validate TaskSpec bodies against them.

Used when mining counterfactual pairs so a ``target_task`` is only emitted
when ``TaskSpec.source_body`` and ``TaskSpec.destination`` (if any) appear
as instance names in that task's ``(:objects)`` or ``(:fixtures)`` blocks.

MuJoCo may register bodies with a ``_main`` suffix at runtime; BDDL and
``predicates.GOAL_TASKS`` use the short instance names (``akita_black_bowl_1``),
so validation is done against BDDL instance names, not MuJoCo body names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

from nla.eval.steerability.predicates import GOAL_TASKS, TaskSpec

_REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_GOAL_BDDL_DIR = (
    _REPO_ROOT
    / "third_party/Isaac-GR00T/external_dependencies/LIBERO/libero/libero"
    / "bddl_files/libero_goal"
)


def _extract_paren_block(text: str, tag: str) -> str:
    """Return inner text of ``(:tag ... )`` or ``""`` if absent."""
    needle = f"(:{tag}"
    i = text.find(needle)
    if i == -1:
        return ""
    j = i + len(needle)
    depth = 1
    start = j
    while j < len(text):
        ch = text[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:j]
        j += 1
    return ""


def _parse_typed_instance_lines(block_text: str) -> frozenset[str]:
    """Parse ``instance [- instance ...] - type`` lines into instance names."""
    names: set[str] = set()
    for line in block_text.splitlines():
        line = line.strip()
        if not line or line in ("(", ")"):
            continue
        if " - " not in line:
            continue
        inst_part, _, _ = line.partition(" - ")
        for tok in inst_part.replace("(", " ").replace(")", " ").split():
            if tok and tok != "-":
                names.add(tok)
    return frozenset(names)


@lru_cache(maxsize=128)
def parse_bddl_instance_names(bddl_path: str) -> frozenset[str]:
    """All ``(:objects)`` and ``(:fixtures)`` instance names in one BDDL file."""
    path = Path(bddl_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    names: set[str] = set()
    for tag in ("objects", "fixtures"):
        block = _extract_paren_block(text, tag)
        if block:
            names.update(_parse_typed_instance_lines(block))
    return frozenset(names)


def required_bodies_for_spec(spec: TaskSpec) -> tuple[str, ...]:
    """Body names the predicate layer needs for this task."""
    out: list[str] = []
    if spec.source_body:
        out.append(spec.source_body)
    if spec.destination and spec.destination not in out:
        out.append(spec.destination)
    return tuple(out)


def required_bodies_for_task(target_task: str) -> tuple[str, ...]:
    if target_task not in GOAL_TASKS:
        raise KeyError(f"unknown target_task {target_task!r}")
    return required_bodies_for_spec(GOAL_TASKS[target_task])


def bddl_path_for_task(target_task: str, bddl_dir: Path) -> Path:
    return bddl_dir / f"{target_task}.bddl"


def missing_bodies_for_task(
    target_task: str,
    bddl_dir: Path,
    *,
    instance_cache: Mapping[str, frozenset[str]] | None = None,
) -> list[str]:
    """Return required body names absent from the task's BDDL (empty = OK)."""
    if target_task not in GOAL_TASKS:
        return [f"unknown task {target_task!r}"]
    bddl_path = bddl_path_for_task(target_task, bddl_dir)
    if not bddl_path.exists():
        return [f"missing bddl file {bddl_path}"]
    if instance_cache is not None and target_task in instance_cache:
        present = instance_cache[target_task]
    else:
        present = parse_bddl_instance_names(str(bddl_path))
    required = required_bodies_for_task(target_task)
    return [b for b in required if b not in present]


def task_bodies_present_in_bddl(
    target_task: str,
    bddl_dir: Path,
    *,
    instance_cache: dict[str, frozenset[str]] | None = None,
) -> bool:
    missing = missing_bodies_for_task(
        target_task, bddl_dir, instance_cache=instance_cache
    )
    return len(missing) == 0


def filter_tasks_with_bodies_in_bddl(
    tasks: list[str],
    bddl_dir: Path,
    *,
    instance_cache: dict[str, frozenset[str]] | None = None,
) -> list[str]:
    """Keep only tasks whose predicate bodies exist in that task's BDDL."""
    cache = instance_cache if instance_cache is not None else {}
    out: list[str] = []
    for task in tasks:
        if task not in GOAL_TASKS:
            continue
        bddl_path = bddl_path_for_task(task, bddl_dir)
        if task not in cache and bddl_path.exists():
            cache[task] = parse_bddl_instance_names(str(bddl_path))
        if task_bodies_present_in_bddl(task, bddl_dir, instance_cache=cache):
            out.append(task)
    return out


def validate_cf_target_bodies(
    target_task: str,
    target_env_name: str,
    bddl_dir: Path,
    *,
    instance_cache: dict[str, frozenset[str]] | None = None,
) -> list[str]:
    """Human-readable issues when a CF row's target is not sim-scorable."""
    issues: list[str] = []
    expected_env = f"libero_sim/{target_task}"
    if target_env_name != expected_env:
        issues.append(
            f"target_env_name {target_env_name!r} != {expected_env!r}"
        )
    missing = missing_bodies_for_task(
        target_task, bddl_dir, instance_cache=instance_cache
    )
    for body in missing:
        if body.startswith("unknown") or body.startswith("missing bddl"):
            issues.append(body)
        else:
            issues.append(
                f"body {body!r} not in BDDL for {target_task!r}"
            )
    return issues

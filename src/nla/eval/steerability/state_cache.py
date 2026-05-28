"""Loader for the LIBERO init-state cache produced by
``scripts/eval/cache_libero_init_states.py``.

Layout::

    data/libero_cache/
        index.json
        <suite>/<task>/
            init_states.npy   # (N, state_dim)
            meta.json         # {task_id, bddl_file, task_description, ...}
            previews/init_<id>.png   # optional

Usage::

    from nla.eval.steerability.state_cache import (
        load_task_meta, load_init_states, apply_init_state,
    )
    meta = load_task_meta("put_the_bowl_on_the_plate")
    states = load_init_states("put_the_bowl_on_the_plate")
    env = LiberoEnv(
        task_bddl_file=meta["bddl_file"],
        task_description=meta["task_description"],
    )
    obs, info = apply_init_state(env, states[init_id])

``apply_init_state`` returns the post-reset observation in the same format
``LiberoEnv.reset`` does, so callers can use it as a drop-in replacement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CACHE_ROOT = Path("data/libero_cache")


def _resolve_root(cache_root: Path | str | None) -> Path:
    if cache_root is None:
        return DEFAULT_CACHE_ROOT
    return Path(cache_root)


def _find_task_dir(task_name: str, cache_root: Path) -> Path:
    """Locate <suite>/<task_name> under cache_root. Scans known suites."""
    for suite in ("libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"):
        cand = cache_root / suite / task_name
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"task {task_name!r} not found under {cache_root}. "
        f"Re-run scripts/eval/cache_libero_init_states.py."
    )


def load_task_meta(task_name: str, cache_root: Path | str | None = None) -> dict[str, Any]:
    """Return the cached meta.json for a task (suite, bddl_file, language, ...)."""
    task_dir = _find_task_dir(task_name, _resolve_root(cache_root))
    return json.loads((task_dir / "meta.json").read_text())


def load_init_states(
    task_name: str, cache_root: Path | str | None = None
) -> np.ndarray:
    """Return the (N, state_dim) cached init-state pool for a task."""
    task_dir = _find_task_dir(task_name, _resolve_root(cache_root))
    return np.load(task_dir / "init_states.npy")


def preview_path(
    task_name: str, init_id: int, cache_root: Path | str | None = None
) -> Path | None:
    """Return path to a cached preview PNG if it exists, else None."""
    task_dir = _find_task_dir(task_name, _resolve_root(cache_root))
    p = task_dir / "previews" / f"init_{init_id:03d}.png"
    return p if p.exists() else None


def apply_init_state(env, init_state: np.ndarray) -> tuple[dict, dict]:
    """Apply a cached init state to a constructed ``LiberoEnv``.

    Mirrors ``LiberoEnv.reset`` but pins the post-reset MuJoCo state to a
    cached pool entry. A zero-action no-op step is required because
    ``OffScreenRenderEnv.set_init_state`` does not refresh the observation
    buffers (the underlying obs cache is populated by ``step()``, not by
    ``set_init_state``).

    Returns ``(observation, info)`` in the same dict format as
    ``LiberoEnv.reset``.
    """
    env._env.reset()
    env._env.set_init_state(np.asarray(init_state))
    zero_action = {
        f"action.{k}": np.zeros(1, dtype=np.float32)
        for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    }
    obs, _reward, _done, _truncated, info = env.step(zero_action)
    return obs, info


def list_cached_tasks(cache_root: Path | str | None = None) -> list[dict[str, Any]]:
    """Return [{'suite': ..., 'task': ..., 'language': ..., 'n_init': ...}, ...]
    for every task in the cache. Reads index.json when present, else walks the
    tree."""
    root = _resolve_root(cache_root)
    idx = root / "index.json"
    out: list[dict[str, Any]] = []
    if idx.exists():
        data = json.loads(idx.read_text())
        for s in data.get("suites", []):
            for t in s.get("tasks", []):
                out.append(
                    {
                        "suite": t["suite"],
                        "task": t["task_name"],
                        "language": t.get("task_description"),
                        "n_init": t.get("n_init_states"),
                    }
                )
        return out
    for meta in sorted(root.glob("*/*/meta.json")):
        m = json.loads(meta.read_text())
        out.append(
            {
                "suite": m["suite"],
                "task": m["task_name"],
                "language": m.get("task_description"),
                "n_init": m.get("n_init_states"),
            }
        )
    return out

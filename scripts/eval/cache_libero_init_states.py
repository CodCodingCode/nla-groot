#!/usr/bin/env python
"""Cache LIBERO deterministic initial-state pools to disk.

LIBERO ships a fixed set of init states per task — accessed via
``task_suite.get_task_init_states(task_id)`` and applied with
``env._env.set_init_state(state)``. This script dumps that pool plus task
metadata (name, language, bddl path) under ``data/libero_cache/<suite>/<task>/``
so downstream tools can:

  - Look up the language / bddl path for a task without importing libero.
  - Reproduce the exact init pool LIBERO uses for benchmarking (so eval runs
    that name an init_id are pinned, not RNG-dependent).
  - Optionally show a saved preview frame of init_id=k without booting MuJoCo.

Must be run in the libero venv:

    LIBERO_PY=third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
    PYTHONPATH=src "${LIBERO_PY}" scripts/eval/cache_libero_init_states.py \\
        --suite libero_goal --output-root data/libero_cache

Pass ``--render-previews`` to also save ``previews/init_<id>.png`` per init
state (slow — constructs the env once per task, ~10s/task).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Use the CPU software rasterizer to match what the existing eval scripts
# (batched_rollout, compare_cf_steer_checkpoints, steerability_eval) all use.
# EGL device-display init is unreliable on this host; osmesa just works.
# Both vars must be set *before* libero_env imports — it setdefault's both
# to "egl", so leaving either unset hands control back to the egl path.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np

DEFAULT_SUITES = (
    "libero_goal",
    "libero_object",
    "libero_spatial",
    "libero_10",
)


def cache_suite(
    suite_name: str,
    output_root: Path,
    *,
    render_previews: bool,
    only_tasks: set[str] | None,
) -> dict:
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path

    bd = benchmark.get_benchmark_dict()
    if suite_name not in bd:
        raise ValueError(f"unknown suite {suite_name!r}; known: {sorted(bd)}")
    suite = bd[suite_name]()

    suite_dir = output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"suite": suite_name, "tasks": []}

    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        if only_tasks is not None and task.name not in only_tasks:
            continue

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(task_id)  # (N, state_dim)
        task_dir = suite_dir / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        np.save(task_dir / "init_states.npy", np.asarray(init_states))

        meta = {
            "suite": suite_name,
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task.language,
            "bddl_file": bddl,
            "problem_folder": task.problem_folder,
            "n_init_states": int(np.asarray(init_states).shape[0]),
            "state_dim": int(np.asarray(init_states).shape[1]),
        }
        (task_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        preview_count = 0
        if render_previews:
            preview_count = _render_previews(bddl, task.language, init_states, task_dir / "previews")

        report["tasks"].append({**meta, "preview_count": preview_count})
        print(
            f"[cached] {suite_name}/{task.name}  n_init={meta['n_init_states']}  "
            f"dim={meta['state_dim']}  previews={preview_count}"
        )
    return report


def _render_previews(bddl: str, language: str, init_states: np.ndarray, out_dir: Path) -> int:
    """Render one agentview PNG per init state. Loads the env once per task.

    After ``set_init_state`` LIBERO needs a no-op step to populate observation
    buffers; we use a zero action.
    """
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    import imageio.v3 as iio

    out_dir.mkdir(parents=True, exist_ok=True)
    env = LiberoEnv(task_bddl_file=bddl, task_description=language, suppress_done=True)
    zero_action = {f"action.{k}": np.zeros(1, dtype=np.float32) for k in
                   ("x", "y", "z", "roll", "pitch", "yaw", "gripper")}
    try:
        n = int(np.asarray(init_states).shape[0])
        for i in range(n):
            env._env.reset()
            env._env.set_init_state(init_states[i])
            obs, *_ = env.step(zero_action)
            img = obs.get("video.image")
            if img is None:
                continue
            iio.imwrite(out_dir / f"init_{i:03d}.png", img)
        return n
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--suite",
        action="append",
        default=None,
        help="LIBERO suite (libero_goal, libero_object, libero_spatial, libero_10, libero_90). "
             "Repeat for multiple; default = goal+object+spatial+10.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Restrict to these task names (matched against task.name in any suite).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/libero_cache"),
        help="Where to write <suite>/<task>/{init_states.npy,meta.json,previews/}.",
    )
    p.add_argument(
        "--render-previews",
        action="store_true",
        help="Also save one agentview PNG per init state. Slow (~10s/task) since "
             "it constructs LiberoEnv per task.",
    )
    args = p.parse_args()

    suites = tuple(args.suite) if args.suite else DEFAULT_SUITES
    only_tasks = set(args.tasks) if args.tasks else None
    args.output_root.mkdir(parents=True, exist_ok=True)

    report: dict = {"suites": []}
    for suite_name in suites:
        report["suites"].append(
            cache_suite(
                suite_name,
                args.output_root,
                render_previews=args.render_previews,
                only_tasks=only_tasks,
            )
        )

    report_path = args.output_root / "index.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Closed-loop LIBERO MuJoCo sim A/B for NLA steering.

Orchestrates the existing two-terminal flow as one driver subprocess: for
each LIBERO task and each of {no-steer, correct, wrong} caption arms,
spawns a fresh ``run_gr00t_server_nla_steer.py`` server, drives a batch of
episodes through ``gr00t.eval.rollout_policy``, parses the per-arm success
rate, and writes a single ``sim_ab.json`` whose schema is consumed by
``build_v3_scorecard.py``.

Per the V3 LIBERO Eval Refactor plan, the headline KPI is::

    sim_correct_success - sim_wrong_success >= +5pp

A positive gap means injecting the correct caption (vs an off-task one)
moves task success. The ``correct vs baseline`` gap tells you whether
steering is at least non-destructive vs the unsteered policy.

The script is **untestable in CI** -- it needs a real GR00T policy
checkpoint + LIBERO MuJoCo + an NLA AR checkpoint -- but the dry-run
mode (``--dry-run``) walks the orchestration plan without spawning
anything so you can sanity-check task names, captions, and arm wiring
against your env first.

Usage::

    PYTHONPATH=src python scripts/eval/closed_loop_sim_ab.py \\
        --ckpt-dir              data/sft/libero_4suite_v3 \\
        --groot-model-path      checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --labels-jsonl          data/labels/libero_4suite_combined/labels.jsonl \\
        --tasks-per-suite       1 \\
        --episodes-per-arm      10 \\
        --n-envs                5 \\
        --port                  5577 \\
        --out-json              data/sft/libero_4suite_v3/sim_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Task selection + caption pool
# ---------------------------------------------------------------------------

# One representative task per suite. Override with ``--tasks`` for a different
# evaluation slate; these defaults give 4 dissimilar manipulations so the
# wrong-caption arm has obvious off-task content.
DEFAULT_TASKS_PER_SUITE = {
    "goal":    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "spatial": "libero_sim/KITCHEN_SCENE3_put_the_black_bowl_on_top_of_the_cabinet",
    "object":  "libero_sim/LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "10":      "libero_sim/KITCHEN_SCENE5_put_the_yellow_and_white_mug_in_the_microwave",
}


@dataclass
class TaskCfg:
    """One row of the (task x arm) grid."""
    suite: str          # short suite name ("goal", "spatial", ...)
    env_name: str       # gym env name registered by register_libero_envs()
    correct_text: str   # gold caption for an episode of this task
    wrong_text: str     # gold caption from a *different* suite's task


@dataclass
class ArmResult:
    suite: str
    env_name: str
    arm: str                   # "baseline" | "correct" | "wrong"
    n_episodes: int
    success_rate: float | None  # None on parse failure
    rollout_stdout_path: str
    server_stdout_path: str
    server_returncode: int | None
    rollout_returncode: int | None
    elapsed_s: float


@dataclass
class SimAB:
    checkpoint: str
    groot_model_path: str
    n_episodes_per_arm: int
    arms: list[dict] = field(default_factory=list)
    correct_success_mean: float | None = None
    wrong_success_mean: float | None = None
    baseline_success_mean: float | None = None
    correct_minus_wrong: float | None = None
    correct_minus_baseline: float | None = None
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Caption discovery
# ---------------------------------------------------------------------------

def _load_labels_by_suite(labels_jsonl: Path) -> dict[str, list[dict]]:
    """Return ``{suite_short: [label_row, ...]}`` from combined labels.jsonl.

    Combined labels prefix ``example_id`` with the suite name (e.g.
    ``goal__traj000159_step000060``); we parse the prefix to bucket rows.
    Rows whose example_id doesn't carry a known prefix are dropped.
    """
    by_suite: dict[str, list[dict]] = {}
    with labels_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ex_id = obj.get("example_id") or ""
            if "__" not in ex_id:
                continue
            suite = ex_id.split("__", 1)[0]
            by_suite.setdefault(suite, []).append(obj)
    return by_suite


def _pick_caption(rows: list[dict], rng: int = 0) -> str:
    """First non-empty bullet-list description in ``rows`` (deterministic)."""
    for r in sorted(rows, key=lambda r: r.get("example_id", "")):
        desc = (r.get("description") or "").strip()
        if desc and len(desc.splitlines()) >= 3:
            return desc
    raise RuntimeError(
        "No non-trivial caption found in label set (need >= 3 bullets)."
    )


def _build_task_grid(
    suites: list[str],
    task_per_suite: dict[str, str],
    by_suite: dict[str, list[dict]],
) -> list[TaskCfg]:
    """For each suite, pick a gold caption for itself (correct) and one from
    a different suite (wrong)."""
    grid: list[TaskCfg] = []
    for suite in suites:
        if suite not in task_per_suite:
            raise SystemExit(
                f"--tasks specifies no env for suite {suite!r}; "
                f"got {list(task_per_suite.keys())}"
            )
        env_name = task_per_suite[suite]
        own = by_suite.get(suite)
        if not own:
            raise SystemExit(
                f"No labels found for suite {suite!r} in --labels-jsonl. "
                f"Available: {sorted(by_suite.keys())}"
            )
        # "wrong" = first row from a different suite.
        wrong_suite = next((s for s in by_suite if s != suite), None)
        if wrong_suite is None:
            raise SystemExit("Need labels from at least 2 different suites for a wrong-caption arm.")
        grid.append(TaskCfg(
            suite=suite,
            env_name=env_name,
            correct_text=_pick_caption(own),
            wrong_text=_pick_caption(by_suite[wrong_suite]),
        ))
    return grid


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

def _spawn_server(
    *,
    py: str,
    groot_model_path: str,
    ar_dir: Path | None,
    text_file: Path | None,
    placement: str,
    blend: float,
    embodiment_tag: str,
    port: int,
    stdout_path: Path,
) -> subprocess.Popen:
    cmd = [
        py, "scripts/eval/run_gr00t_server_nla_steer.py",
        "--model-path", groot_model_path,
        "--embodiment-tag", embodiment_tag,
        "--port", str(port),
        "--use-sim-policy-wrapper",
    ]
    if ar_dir is not None and text_file is not None:
        cmd += [
            "--ar-dir", str(ar_dir),
            "--steer-text-file", str(text_file),
            "--placement", placement,
            "--blend", str(blend),
        ]
    stdout_f = stdout_path.open("w")
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "src") or "src"
    return subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,   # group so we can kill the whole tree
    )


def _wait_for_server_ready(stdout_path: Path, *, timeout_s: float = 600.0) -> bool:
    """Poll ``stdout_path`` for the well-known "Server ready" line.

    Returns True iff the line appears before ``timeout_s``. Anything else
    (timeout / readiness banner absent) returns False, leaving the caller
    to handle the still-spawned server (typically: kill it).
    """
    deadline = time.time() + timeout_s
    pat = re.compile(r"Server ready", re.IGNORECASE)
    while time.time() < deadline:
        if stdout_path.exists():
            try:
                if pat.search(stdout_path.read_text(errors="ignore")):
                    return True
            except Exception:
                pass
        time.sleep(1.0)
    return False


def _run_rollout(
    *,
    py: str,
    env_name: str,
    n_episodes: int,
    n_envs: int,
    port: int,
    seed: int,
    video_dir: Path,
    stdout_path: Path,
) -> int:
    """Run rollout_policy.py against the spawned server. Returns exit code."""
    cmd = [
        py, "-m", "gr00t.eval.rollout_policy",
        "--env-name", env_name,
        "--n-episodes", str(n_episodes),
        "--n-envs", str(n_envs),
        "--policy-client-host", "127.0.0.1",
        "--policy-client-port", str(port),
        "--model-path", "",   # use server, not local model
        "--video-dir", str(video_dir),
        "--seed", str(seed),
    ]
    stdout_f = stdout_path.open("w")
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "src") or "src"
    proc = subprocess.run(cmd, stdout=stdout_f, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


_SUCCESS_RE = re.compile(r"success rate:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_success_rate(rollout_stdout: Path) -> float | None:
    if not rollout_stdout.exists():
        return None
    text = rollout_stdout.read_text(errors="ignore")
    last: float | None = None
    for m in _SUCCESS_RE.finditer(text):
        try:
            last = float(m.group(1))
        except Exception:
            continue
    return last


def _kill_proc_tree(proc: subprocess.Popen, timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# Per-arm driver
# ---------------------------------------------------------------------------

def _run_one_arm(
    *,
    task: TaskCfg,
    arm: str,
    args: argparse.Namespace,
    work_dir: Path,
) -> ArmResult:
    """Run one (task, arm) combination."""
    arm_dir = work_dir / f"{task.suite}__{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    server_stdout = arm_dir / "server.log"
    rollout_stdout = arm_dir / "rollout.log"
    video_dir = arm_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    text_file: Path | None = None
    ar_dir: Path | None = None
    if arm != "baseline":
        text_file = arm_dir / "steer_text.txt"
        ar_dir = Path(args.ckpt_dir) / "ar"
        steer_text = task.correct_text if arm == "correct" else task.wrong_text
        text_file.write_text(steer_text)

    print(f"  [{task.suite}/{arm}] spawning server...", flush=True)
    t0 = time.time()
    proc = _spawn_server(
        py=args.py,
        groot_model_path=args.groot_model_path,
        ar_dir=ar_dir,
        text_file=text_file,
        placement=args.placement,
        blend=args.blend,
        embodiment_tag=args.embodiment_tag,
        port=args.port,
        stdout_path=server_stdout,
    )
    ready = False
    rollout_rc: int | None = None
    try:
        ready = _wait_for_server_ready(server_stdout, timeout_s=args.server_ready_timeout)
        if not ready:
            print(f"  [{task.suite}/{arm}] server failed to become ready in "
                  f"{args.server_ready_timeout}s -- check {server_stdout}", flush=True)
        else:
            print(f"  [{task.suite}/{arm}] server ready, driving rollout "
                  f"({args.episodes_per_arm} eps, {args.n_envs} envs)...", flush=True)
            rollout_rc = _run_rollout(
                py=args.py,
                env_name=task.env_name,
                n_episodes=args.episodes_per_arm,
                n_envs=args.n_envs,
                port=args.port,
                seed=args.seed,
                video_dir=video_dir,
                stdout_path=rollout_stdout,
            )
    finally:
        print(f"  [{task.suite}/{arm}] tearing down server...", flush=True)
        _kill_proc_tree(proc)

    success_rate = _parse_success_rate(rollout_stdout) if ready else None
    elapsed = time.time() - t0
    print(f"  [{task.suite}/{arm}] success_rate={success_rate} elapsed={elapsed:.1f}s",
          flush=True)
    return ArmResult(
        suite=task.suite, env_name=task.env_name, arm=arm,
        n_episodes=args.episodes_per_arm,
        success_rate=success_rate,
        rollout_stdout_path=str(rollout_stdout),
        server_stdout_path=str(server_stdout),
        server_returncode=proc.returncode,
        rollout_returncode=rollout_rc,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True,
                   help="SFT run dir; expects ar/ subdir.")
    p.add_argument("--groot-model-path", required=True,
                   help="Path or HF id of the GR00T policy checkpoint.")
    p.add_argument("--labels-jsonl", required=True,
                   help="Combined labels.jsonl to source gold captions from.")
    p.add_argument("--suites", nargs="+", default=("goal", "spatial", "object", "10"),
                   help="Short suite names to evaluate. Each must appear in --tasks.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Override DEFAULT_TASKS_PER_SUITE as space-separated "
                        "'<suite>=<env_name>' pairs. Defaults are LIBERO 4-suite reps.")
    p.add_argument("--tasks-per-suite", type=int, default=1,
                   help="(Reserved: future N-tasks-per-suite. Today: 1.)")
    p.add_argument("--episodes-per-arm", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=5)
    p.add_argument("--port", type=int, default=5577)
    p.add_argument("--placement", default="image_patch",
                   choices=["last_text", "image_patch", "anchor",
                            "image_patch_all", "fixed"])
    p.add_argument("--blend", type=float, default=1.0)
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--server-ready-timeout", type=float, default=600.0,
                   help="Seconds to wait for the 'Server ready' banner.")
    p.add_argument("--py", default=".venv/bin/python",
                   help="Python executable for both server and rollout.")
    p.add_argument("--work-dir", default=None,
                   help="Directory for per-arm logs/videos. Default: <ckpt>/sim_ab_work")
    p.add_argument("--out-json", required=True,
                   help="Output sim_ab.json (consumed by build_v3_scorecard.py).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-arm plan + captions and exit. No subprocesses.")
    return p


def _parse_tasks_override(items: list[str] | None) -> dict[str, str]:
    out = dict(DEFAULT_TASKS_PER_SUITE)
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--tasks value {it!r} must be '<suite>=<env_name>'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ckpt_dir = Path(args.ckpt_dir)
    work_dir = Path(args.work_dir) if args.work_dir else ckpt_dir / "sim_ab_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    labels_path = Path(args.labels_jsonl)
    if not labels_path.exists():
        print(f"FATAL: labels file missing: {labels_path}", file=sys.stderr)
        return 2
    by_suite = _load_labels_by_suite(labels_path)
    task_per_suite = _parse_tasks_override(args.tasks)
    grid = _build_task_grid(list(args.suites), task_per_suite, by_suite)

    print(f"sim A/B plan ({len(grid)} tasks x 3 arms = {3*len(grid)} runs):")
    for t in grid:
        print(f"  {t.suite:8s}  {t.env_name}")
        print(f"    correct: {t.correct_text.splitlines()[0][:80]}...")
        print(f"    wrong  : {t.wrong_text.splitlines()[0][:80]}...")
    print()

    if args.dry_run:
        print("--dry-run set; not spawning anything. Exiting.")
        return 0

    if not (ckpt_dir / "ar").exists():
        print(f"FATAL: {ckpt_dir}/ar does not exist; cannot run steering arms.", file=sys.stderr)
        return 2
    if not shutil.which(args.py) and not Path(args.py).is_file():
        print(f"FATAL: python at {args.py} not found.", file=sys.stderr)
        return 2

    results: list[ArmResult] = []
    for task in grid:
        for arm in ("baseline", "correct", "wrong"):
            res = _run_one_arm(task=task, arm=arm, args=args, work_dir=work_dir)
            results.append(res)
            # Persist incrementally so a crash doesn't lose everything.
            _flush_partial(args.out_json, ckpt_dir, args, results, partial=True)

    _flush_partial(args.out_json, ckpt_dir, args, results, partial=False)
    print(f"\nWrote {args.out_json}")
    return 0


def _flush_partial(
    out_json: str,
    ckpt_dir: Path,
    args: argparse.Namespace,
    results: list[ArmResult],
    *,
    partial: bool,
) -> None:
    def _mean(xs: list[float | None]) -> float | None:
        clean = [x for x in xs if isinstance(x, float)]
        return sum(clean) / len(clean) if clean else None

    correct = [r.success_rate for r in results if r.arm == "correct"]
    wrong = [r.success_rate for r in results if r.arm == "wrong"]
    baseline = [r.success_rate for r in results if r.arm == "baseline"]
    c_mean = _mean(correct)
    w_mean = _mean(wrong)
    b_mean = _mean(baseline)
    summary = SimAB(
        checkpoint=str(ckpt_dir),
        groot_model_path=args.groot_model_path,
        n_episodes_per_arm=args.episodes_per_arm,
        arms=[asdict(r) for r in results],
        correct_success_mean=c_mean,
        wrong_success_mean=w_mean,
        baseline_success_mean=b_mean,
        correct_minus_wrong=(c_mean - w_mean) if (c_mean is not None and w_mean is not None) else None,
        correct_minus_baseline=(c_mean - b_mean) if (c_mean is not None and b_mean is not None) else None,
        config={
            "suites": list(args.suites),
            "n_envs": args.n_envs,
            "placement": args.placement,
            "blend": args.blend,
            "embodiment_tag": args.embodiment_tag,
            "port": args.port,
            "seed": args.seed,
            "partial": partial,
        },
    )
    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    sys.exit(main())

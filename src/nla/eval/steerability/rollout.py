"""Single-rollout runner used by the steerability eval harness.

Spins up a :class:`LiberoEnv` directly (no Ray), reads obs through
``video.image`` / ``state.*`` keys, calls the policy server every
``n_action_steps`` sim ticks, captures per-step state + RGB frame, dumps
``trajectory.parquet`` + ``rollout.mp4`` + ``summary.json`` into
``output_dir``.

Runs inside the LIBERO ``libero_uv/.venv`` (depends on ``gr00t``, ``libero``,
``mujoco``, ``robosuite``). The eval driver shells out to this script as a
subprocess, so the main GR00T venv never has to import libero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_paths_on_sys() -> None:
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def make_env(env_name: str, suppress_done: bool):
    """Create a :class:`LiberoEnv` bypassing the gym registry so we can pass
    ``suppress_done=True``. The env name follows the same ``libero_sim/<task>``
    convention as the registry.
    """
    if not env_name.startswith("libero_sim/"):
        raise ValueError(f"only libero_sim/* envs are supported, got {env_name!r}")
    task_name = env_name.removeprefix("libero_sim/")
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    bd = benchmark.get_benchmark_dict()
    # search every LIBERO suite for the task
    for suite in ("libero_goal", "libero_10", "libero_object", "libero_spatial", "libero_90"):
        s = bd[suite]()
        for i in range(s.get_num_tasks()):
            t = s.get_task(i)
            if t.name == task_name:
                bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
                return LiberoEnv(
                    task_bddl_file=bddl,
                    task_description=t.language,
                    suppress_done=suppress_done,
                )
    raise ValueError(f"task {task_name!r} not found in any LIBERO suite")


def encode_mp4(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    if not frames:
        return
    try:
        import imageio.v3 as iio
        iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", pixelformat="yuv420p")
        return
    except Exception:
        pass
    import cv2
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


def render_panel(obs: dict, success: bool, t: int) -> np.ndarray:
    """Build the 512x256 hstack of agentview + wrist_view used by gr00t."""
    av = obs["video.image"]
    wr = obs["video.wrist_image"]
    panel = np.concatenate([av, wr], axis=1).copy()
    # tiny step + success indicator strip at the bottom
    import cv2
    color = (0, 255, 0) if success else (255, 80, 80)
    cv2.putText(
        panel, f"t={t}  success={int(success)}", (5, 250),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )
    return panel


def _to_server_obs(obs: dict) -> dict:
    """Add B=1, T=1 dims to the env observation as the GR00T server expects."""
    out: dict[str, Any] = {}
    for k, v in obs.items():
        if k.startswith("video"):
            arr = np.asarray(v)  # (H, W, C) uint8
            out[k] = arr[None, None, ...]  # (1, 1, H, W, C)
        elif k.startswith("state"):
            arr = np.asarray(v, dtype=np.float32)  # (n_dim,)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            out[k] = arr[None, None, ...]  # (1, 1, n_dim)
        elif k.startswith("annotation"):
            out[k] = [v] if not isinstance(v, list) else v
        else:
            out[k] = v
    return out


def _unpack_action_chunk(
    chunk: dict, n_action_steps: int, batch_index: int = 0
) -> list[dict]:
    """Convert (B, T, D) per key into a list of per-step env actions.

    The GR00T policy server (with ``Gr00tSimPolicyWrapper``) returns flat
    keys whose values are ``np.float32`` arrays of shape ``(B, T, D)``. We
    pick the requested batch row, then iterate over the first
    ``n_action_steps`` timesteps and emit one ``{key: ndarray(D,)}`` dict
    per sub-action — that's what :class:`LiberoEnv` expects.
    """
    per_key: dict[str, np.ndarray] = {}
    chunk_T = None
    for k, v in chunk.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 3:
            row = arr[batch_index]  # (T, D)
        elif arr.ndim == 2:
            row = arr  # treat as (T, D)
        elif arr.ndim == 1:
            row = arr[:, None]  # (T, 1)
        else:
            row = arr.reshape(1, -1)
        per_key[k] = row
        chunk_T = row.shape[0] if chunk_T is None else min(chunk_T, row.shape[0])
    chunk_T = chunk_T or 0
    chunk_T = min(chunk_T, n_action_steps)
    return [{k: per_key[k][i].copy() for k in per_key} for i in range(chunk_T)]


def run_one_rollout(
    env_name: str,
    seed: int,
    policy_host: str,
    policy_port: int,
    output_dir: Path,
    tracked_bodies: list[str],
    target_body: str | None,
    n_action_steps: int = 8,
    max_episode_steps: int = 300,
    fps: int = 20,
    suppress_done: bool = True,
    steps_per_render: int = 1,
) -> dict[str, Any]:
    from gr00t.policy.server_client import PolicyClient
    from nla.eval.steerability.object_logger import ObjectStateLogger, episode_summary

    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(env_name, suppress_done=suppress_done)
    client = PolicyClient(host=policy_host, port=policy_port, timeout_ms=120_000)
    obs, info = env.reset(seed=seed)
    logger = ObjectStateLogger(env, tracked_bodies=tracked_bodies)
    logger.capture_initial()
    frames: list[np.ndarray] = []

    t = 0
    frames.append(render_panel(obs, success=bool(info.get("success", False)), t=t))

    while t < max_episode_steps:
        action_chunk, _ = client.get_action(_to_server_obs(obs))
        sub_actions = _unpack_action_chunk(action_chunk, n_action_steps=n_action_steps)
        if not sub_actions:
            break
        for sub_action in sub_actions:
            obs, reward, done, truncated, info = env.step(sub_action)
            logger.log_step(reward=reward, info=info)
            t += 1
            if t % steps_per_render == 0:
                frames.append(render_panel(obs, success=bool(info.get("success", False)), t=t))
            if t >= max_episode_steps:
                break
            if done or truncated:
                break

    # Write artifacts
    parquet_path = output_dir / "trajectory.parquet"
    logger.write_parquet(parquet_path)
    encode_mp4(frames, output_dir / "rollout.mp4", fps=fps)

    traj = logger.to_dict()
    summary = episode_summary(traj, target_body=target_body)
    summary.update({
        "env_name": env_name,
        "seed": seed,
        "n_action_steps": n_action_steps,
        "max_episode_steps": max_episode_steps,
        "target_body": target_body,
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    env.close()
    return summary


def _cli() -> None:
    _ensure_paths_on_sys()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tracked-bodies", nargs="+", required=True)
    parser.add_argument("--target-body", default=None)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=1)
    parser.add_argument("--suppress-done", action="store_true", default=True)
    parser.add_argument("--no-suppress-done", dest="suppress_done", action="store_false")
    args = parser.parse_args()

    summary = run_one_rollout(
        env_name=args.env_name,
        seed=args.seed,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        output_dir=Path(args.output_dir),
        tracked_bodies=list(args.tracked_bodies),
        target_body=args.target_body,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        fps=args.fps,
        steps_per_render=args.steps_per_render,
        suppress_done=args.suppress_done,
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    _cli()

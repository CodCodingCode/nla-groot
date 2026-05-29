"""Run multiple LIBERO rollouts in one process with batched policy inference.

Used by :mod:`nla.training.sim_reward` when ``sim_batch_size > 1``. Requires
:class:`nla.eval.nla_policy_server.NlaPolicyServer` (``get_action_batch``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_paths_on_sys() -> None:
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class _RolloutSlot:
    env_name: str
    seed: int
    target_task: str
    tracked_bodies: list[str]
    steer_h: np.ndarray
    placement: str
    blend: float
    env: Any
    client_obs: dict[str, Any]
    logger: Any
    t: int = 0
    done: bool = False
    early_stopped: bool = False
    options: dict[str, Any] | None = None
    policy_language_override: str | None = None
    w_predicate: float | None = None


def run_batched_rollouts(
    jobs: list[dict[str, Any]],
    *,
    policy_host: str,
    policy_port: int,
    n_action_steps: int = 8,
    max_episode_steps: int = 100,
    timeout_ms: int = 180_000,
) -> list[dict[str, Any]]:
    """Score up to ``len(jobs)`` rollouts with synchronized batched policy calls."""
    from nla.eval.batched_policy_client import BatchedPolicyClient
    from nla.eval.steerability.object_logger import ObjectStateLogger, episode_summary
    from nla.eval.steerability.predicates import predicate_fires, resolve_task, tracked_bodies_for
    from nla.eval.steerability.rollout import (
        _to_server_obs,
        _unpack_action_chunk,
        make_env,
    )

    if not jobs:
        return []

    client = BatchedPolicyClient(host=policy_host, port=policy_port, timeout_ms=timeout_ms)

    slots: list[_RolloutSlot] = []
    for job in jobs:
        target_task = job["target_task"]
        bodies = job.get("tracked_bodies")
        if not bodies:
            bodies = tracked_bodies_for(target_task)
        env = make_env(job["env_name"], suppress_done=True)
        obs, _ = env.reset(seed=int(job["seed"]))
        logger = ObjectStateLogger(env, tracked_bodies=bodies)
        logger.capture_initial()
        steer_h = np.asarray(job["steer_h"], dtype=np.float32)
        spec_dict: dict[str, Any] = {
            "placement": job.get("placement", "image_patch"),
            "blend": float(job.get("blend", 1.0)),
        }
        if job.get("strided_k"):
            spec_dict["strided_k"] = int(job["strided_k"])
        options: dict[str, Any] = {
            "steer_h": steer_h,
            "steer_spec": spec_dict,
        }
        if job.get("steer_disabled"):
            options["steer_disabled"] = True
        lang_override = job.get("policy_language_override") or None
        w_pred_raw = job.get("w_predicate")
        w_pred = float(w_pred_raw) if w_pred_raw is not None else None
        slots.append(
            _RolloutSlot(
                env_name=job["env_name"],
                seed=int(job["seed"]),
                target_task=target_task,
                tracked_bodies=bodies,
                steer_h=steer_h,
                placement=job.get("placement", "image_patch"),
                blend=float(job.get("blend", 1.0)),
                env=env,
                client_obs=_to_server_obs(obs, policy_language_override=lang_override),
                logger=logger,
                options=options,
                policy_language_override=lang_override,
                w_predicate=w_pred,
            )
        )

    spec = resolve_task(slots[0].target_task)

    def _pred_cb(traj: dict, _spec=spec) -> bool:
        return predicate_fires(traj, _spec)

    early_stop_check_every = 1

    while True:
        active = [s for s in slots if not s.done]
        if not active:
            break
        if all(s.t >= max_episode_steps for s in active):
            for s in active:
                s.done = True
            continue

        observations = [s.client_obs for s in active]
        options_list = [s.options for s in active]

        try:
            if len(observations) == 1:
                pairs = [client.get_action(observations[0], options_list[0])]
            else:
                pairs = client.get_action_batch(observations, options_list)
        except RuntimeError as e:
            err = repr(e)
            for s in active:
                s.done = True
            # Caller marks errors; return partial summaries below
            if "Unknown endpoint" in err or "get_action_batch" in err:
                raise RuntimeError(
                    "Policy server lacks get_action_batch; restart with "
                    "scripts/eval/run_gr00t_server_nla_steer.py (NlaPolicyServer)"
                ) from e
            raise

        for slot, (action_chunk, _) in zip(active, pairs):
            sub_actions = _unpack_action_chunk(
                action_chunk, n_action_steps=n_action_steps, batch_index=0,
            )
            if not sub_actions:
                slot.done = True
                continue
            for sub_action in sub_actions:
                obs, reward, done, truncated, info = slot.env.step(sub_action)
                slot.logger.log_step(reward=reward, info=info)
                slot.t += 1
                slot.client_obs = _to_server_obs(
                    obs, policy_language_override=slot.policy_language_override,
                )
                if slot.t >= max_episode_steps:
                    slot.done = True
                    break
                if done or truncated:
                    slot.done = True
                    break
                if slot.t % early_stop_check_every == 0:
                    try:
                        if _pred_cb(slot.logger.to_dict()):
                            slot.early_stopped = True
                            slot.done = True
                            break
                    except Exception:
                        pass
            if slot.early_stopped:
                break

    summaries: list[dict[str, Any]] = []
    for slot in slots:
        traj = slot.logger.to_dict()
        summary = episode_summary(traj, target_body=None)
        summary.update({
            "env_name": slot.env_name,
            "seed": slot.seed,
            "n_action_steps": n_action_steps,
            "max_episode_steps": max_episode_steps,
            "early_stopped": bool(slot.early_stopped),
        })
        from nla.eval.steerability.predicates import (
            DEFAULT_SHAPING,
            ShapingWeights,
            score as predicate_score,
        )
        weights = None
        if slot.w_predicate is not None:
            weights = ShapingWeights(
                w_predicate=float(slot.w_predicate),
                w_dist=DEFAULT_SHAPING.w_dist,
                w_displace=DEFAULT_SHAPING.w_displace,
                w_contact=DEFAULT_SHAPING.w_contact,
                d_max=DEFAULT_SHAPING.d_max,
                disp_max=DEFAULT_SHAPING.disp_max,
                contact_near_m=DEFAULT_SHAPING.contact_near_m,
            )
        score_breakdown = predicate_score(traj, slot.target_task, weights=weights)
        summary["r_sim"] = score_breakdown["r"]
        summary["sim_score_breakdown"] = score_breakdown
        summary["n_steps"] = slot.t
        summary["success_any"] = bool(summary.get("success_any", False))
        slot.env.close()
        summaries.append(summary)
    return summaries


def _cli() -> None:
    import sys

    _ensure_paths_on_sys()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs-json", required=True, help="Path to JSON list of job dicts.")
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=5556)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--n-action-steps", type=int, default=8)
    args = p.parse_args()
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    jobs = json.loads(Path(args.jobs_json).read_text())
    # LIBERO logs "[info] ... [0,1,...]" to stdout; keep stdout clean for GRPO.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        summaries = run_batched_rollouts(
            jobs,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            max_episode_steps=args.max_episode_steps,
            n_action_steps=args.n_action_steps,
        )
    finally:
        sys.stdout = real_stdout
    from nla.eval.steerability.json_utils import dumps_rollout_json

    print(dumps_rollout_json(summaries), file=real_stdout, flush=True)


if __name__ == "__main__":
    _cli()

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
from typing import Any, Callable

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


def _to_server_obs(obs: dict, *, policy_language_override: str | None = None) -> dict:
    """Pack the LIBERO env observation into the format the GR00T server expects.

    The NLA steer server (``scripts/eval/run_gr00t_server_nla_steer.py``)
    runs a bare ``Gr00tPolicy`` (no ``Gr00tSimPolicyWrapper``), which expects
    a **nested** observation::

        {
          "video":     {"image": (B, T, H, W, C), "wrist_image": ...},
          "state":     {"x": (B, T, D), ..., "gripper": ...},
          "language":  {"task": [[<str>]]},   # (B, T) list of lists
          "annotation": {...},                  # optional, passed through
        }

    The LIBERO env yields **flat** keys with dotted prefixes (``video.image``,
    ``state.x``, ``annotation.human.action.task_description``) and per-step
    shapes ``(H, W, C)`` / ``(D,)``. We unflatten by splitting on the first
    dot, add the ``(B=1, T=1)`` broadcasting dims, and lift the language /
    task-description string into a one-deep ``[[...]]`` list as the server
    requires for the (B, T) language slot. Other prefixes (``annotation``)
    are passed through into their own sub-dict so the wrapper is forward-
    compatible with future modality keys without code changes.

    When ``policy_language_override`` is non-empty, every language slot the
    server would receive is replaced with that string (still wrapped as the
    required ``[[str]]`` (B=1, T=1) container). This is what the eval-v2
    ``language_swap`` protocol uses to feed the policy the intent-arm text
    instead of the loaded BDDL task; without it the matched and mismatched
    intent arms share the same native language channel and the
    ``semantic_gap_predicate`` metric is structurally zero.
    """
    out: dict[str, dict[str, Any]] = {}
    for k, v in obs.items():
        if "." not in k:
            # Unprefixed keys (rare; mostly raw env fields) are dropped --
            # the server validates against the model's modality config and
            # would reject them anyway.
            continue
        modality, sub = k.split(".", 1)
        bucket = out.setdefault(modality, {})
        if modality == "video":
            arr = np.asarray(v)  # (H, W, C) uint8
            bucket[sub] = arr[None, None, ...]  # (1, 1, H, W, C)
        elif modality == "state":
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            bucket[sub] = arr[None, None, ...]  # (1, 1, D)
        elif modality == "annotation":
            # The model treats ``annotation.human.action.task_description``
            # (or whatever ``modality_keys[0]`` is in the loaded model's
            # language config) as the task language. The bare
            # ``Gr00tPolicy`` (no ``Gr00tSimPolicyWrapper`` -- which is the
            # NLA steer-server default; see
            # ``scripts/eval/run_gr00t_server_nla_steer.py``) indexes
            # ``observation["language"][<full_original_key>]`` so we mirror
            # the value under the FULL dotted key, not the wrapper's
            # canonical ``"task"`` alias. Shape is (B=1, T=1) ->
            # ``[[str(v)]]`` per the language modality validator.
            bucket[sub] = v
            lang = out.setdefault("language", {})
            lang.setdefault(k, [[str(v)]])
        else:
            # Unknown modality: pack the value verbatim under sub-key. Lets
            # us add new modalities without code changes.
            bucket[sub] = v
    # The server asserts the three core modality buckets exist even if some
    # of them are empty (e.g. a vision-only model). Ensure they're present.
    for required in ("video", "state", "language"):
        out.setdefault(required, {})
    if policy_language_override:
        override = str(policy_language_override)
        # Rewrite every (B=1, T=1) language slot that ``_to_server_obs``
        # populated above. We touch only existing keys so the model's
        # modality validator (which checks for the canonical
        # ``annotation.human.action.task_description`` key) still sees a
        # populated slot under whatever name the env uses.
        for k in list(out["language"].keys()):
            out["language"][k] = [[override]]
    return out


def _unpack_action_chunk(
    chunk: dict, n_action_steps: int, batch_index: int = 0
) -> list[dict]:
    """Convert (B, T, D) per key into a list of per-step env actions.

    The bare ``Gr00tPolicy`` (the NLA steer-server default) returns
    **nested** action keys like ``{"x": ..., "y": ..., "gripper": ...}``,
    while ``Gr00tSimPolicyWrapper`` returns the same chunk under flat
    ``action.<name>`` keys. :class:`LiberoEnv.step` expects the flat form,
    so we transparently re-prefix any key that doesn't already start with
    ``"action."``. We pick the requested batch row, then iterate over the
    first ``n_action_steps`` timesteps and emit one
    ``{action.<name>: ndarray(D,)}`` dict per sub-action.
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
        flat_key = k if k.startswith("action.") else f"action.{k}"
        per_key[flat_key] = row
        chunk_T = row.shape[0] if chunk_T is None else min(chunk_T, row.shape[0])
    chunk_T = chunk_T or 0
    chunk_T = min(chunk_T, n_action_steps)
    return [{k: per_key[k][i].copy() for k in per_key} for i in range(chunk_T)]


def run_one_rollout(
    env_name: str,
    seed: int,
    policy_host: str,
    policy_port: int,
    output_dir: Path | None,
    tracked_bodies: list[str],
    target_body: str | None,
    n_action_steps: int = 8,
    max_episode_steps: int = 300,
    fps: int = 20,
    suppress_done: bool = True,
    steps_per_render: int = 1,
    *,
    options: dict[str, Any] | None = None,
    early_stop_on_predicate: Callable[[dict[str, Any]], bool] | None = None,
    early_stop_check_every: int = 1,
    capture_frames: bool = True,
    return_trajectory: bool = False,
    policy_language_override: str | None = None,
) -> dict[str, Any]:
    """Run one LIBERO rollout against a (possibly NLA-steered) policy server.

    Optional parameters added for sim-reward GRPO use:

    - ``options``: passed verbatim to every ``client.get_action(obs, options)``
      call. The NLA wrapper reads ``options["steer_h"]`` per request so a
      single server can score many different (text, intent) pairs. Set
      ``options['steer_disabled'] = True`` to short-circuit the steer hook
      for the no-steer causal arm (the wrapper falls back to the base
      policy without applying ``steer_h``).
    - ``early_stop_on_predicate``: callable that takes the *partial*
      :class:`ObjectStateLogger.to_dict()` trajectory and returns True to
      cut the rollout short. Used by the GRPO worker to abort as soon as
      the steered intent's predicate fires.
    - ``early_stop_check_every``: skip the predicate check for ``N-1`` out
      of every ``N`` sim steps; checks are cheap (numpy) but not free.
    - ``capture_frames`` / ``return_trajectory``: turn off MP4/parquet
      writes when used as a reward function (we only care about the score).
    - ``policy_language_override``: when set, every observation sent to the
      policy has its ``language.*`` slots overwritten with this string.
      Used by the eval-v2 ``language_swap`` protocol to feed the policy
      the intent-arm text instead of the loaded BDDL task; the underlying
      sim still loads the target scene unchanged.
    """
    from gr00t.policy.server_client import PolicyClient
    from nla.eval.steerability.object_logger import ObjectStateLogger, episode_summary

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(env_name, suppress_done=suppress_done)
    client = PolicyClient(host=policy_host, port=policy_port, timeout_ms=120_000)
    obs, info = env.reset(seed=seed)
    logger = ObjectStateLogger(env, tracked_bodies=tracked_bodies)
    logger.capture_initial()
    frames: list[np.ndarray] = []

    t = 0
    if capture_frames:
        frames.append(render_panel(obs, success=bool(info.get("success", False)), t=t))

    early_stopped = False
    while t < max_episode_steps:
        action_chunk, _ = client.get_action(
            _to_server_obs(obs, policy_language_override=policy_language_override),
            options=options,
        )
        sub_actions = _unpack_action_chunk(action_chunk, n_action_steps=n_action_steps)
        if not sub_actions:
            break
        for sub_action in sub_actions:
            obs, reward, done, truncated, info = env.step(sub_action)
            logger.log_step(reward=reward, info=info)
            t += 1
            if capture_frames and t % steps_per_render == 0:
                frames.append(render_panel(obs, success=bool(info.get("success", False)), t=t))
            if t >= max_episode_steps:
                break
            if done or truncated:
                break
            if early_stop_on_predicate is not None and (t % early_stop_check_every == 0):
                try:
                    if early_stop_on_predicate(logger.to_dict()):
                        early_stopped = True
                        break
                except Exception:
                    # Never let predicate evaluation crash a rollout.
                    pass
        if early_stopped:
            break

    # Write artifacts
    if output_dir is not None:
        parquet_path = output_dir / "trajectory.parquet"
        logger.write_parquet(parquet_path)
        if capture_frames:
            encode_mp4(frames, output_dir / "rollout.mp4", fps=fps)

    traj = logger.to_dict()
    summary = episode_summary(traj, target_body=target_body)
    summary.update({
        "env_name": env_name,
        "seed": seed,
        "n_action_steps": n_action_steps,
        "max_episode_steps": max_episode_steps,
        "target_body": target_body,
        "early_stopped": bool(early_stopped),
    })
    if output_dir is not None:
        from nla.eval.steerability.json_utils import dumps_rollout_json

        (output_dir / "summary.json").write_text(dumps_rollout_json(summary))
    env.close()
    if return_trajectory:
        summary["_trajectory"] = traj
    return summary


def _cli() -> None:
    _ensure_paths_on_sys()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--output-dir", default=None,
                        help="If set, write trajectory.parquet/rollout.mp4/summary.json.")
    parser.add_argument("--tracked-bodies", nargs="+", default=None,
                        help="If unset and --target-task is set, the default tracked-bodies "
                             "for the resolved task spec are used.")
    parser.add_argument("--target-body", default=None)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--steps-per-render", type=int, default=1)
    parser.add_argument("--suppress-done", action="store_true", default=True)
    parser.add_argument("--no-suppress-done", dest="suppress_done", action="store_false")
    parser.add_argument(
        "--steer-h-path", default=None,
        help="Path to a .npy file with a [H]-shaped float32 steer vector. "
             "When set, sent as options['steer_h'] on every get_action call.",
    )
    parser.add_argument(
        "--steer-placement", default=None,
        choices=["last_text", "image_patch", "anchor", "image_patch_all", "fixed"],
        help="Per-call override for the steer placement (NlaSteerGr00tPolicy "
             "reads this via options['steer_spec'].placement).",
    )
    parser.add_argument(
        "--steer-blend", type=float, default=None,
        help="Per-call override for the steer blend factor [0, 1].",
    )
    parser.add_argument(
        "--target-task", default=None,
        help="Canonical LIBERO Goal task id whose predicate + dense shaping "
             "should be added to the printed summary. When set, an "
             "'r_sim' field appears in the JSON output with the combined "
             "reward score.",
    )
    parser.add_argument(
        "--early-stop-on-success", action="store_true",
        help="Break the rollout as soon as the target-task predicate fires "
             "(only meaningful when --target-task is set).",
    )
    parser.add_argument(
        "--no-frames", action="store_true",
        help="Skip MP4 frame capture (saves ~30% wall time on long rollouts).",
    )
    parser.add_argument(
        "--policy-language-override", default=None,
        help="Replace the policy obs language slot with this string on every "
             "step (eval-v2 language_swap protocol). When omitted the env's "
             "native BDDL task_description is forwarded unchanged.",
    )
    parser.add_argument(
        "--steer-disabled", action="store_true",
        help="Send options['steer_disabled']=True every step. The "
             "NlaSteerGr00tPolicy wrapper short-circuits to the base policy "
             "without applying steer_h (no-steer causal arm).",
    )
    parser.add_argument(
        "--w-predicate", type=float, default=None,
        help="Override the predicate term weight inside the sim shaping "
             "score (DEFAULT_SHAPING.w_predicate=2.0 in "
             "nla.eval.steerability.predicates). Only consulted when "
             "--target-task is set. Lower values (e.g. 1.0) densify the "
             "shaping terms relative to the binary predicate, useful for "
             "GRPO contrastive runs that want more within-group variance.",
    )
    args = parser.parse_args()

    # Resolve tracked-bodies from the target task spec when omitted.
    tracked_bodies = list(args.tracked_bodies) if args.tracked_bodies else []
    target_task = args.target_task
    if target_task and not tracked_bodies:
        from nla.eval.steerability.predicates import tracked_bodies_for
        tracked_bodies = tracked_bodies_for(target_task)
    if not tracked_bodies:
        raise SystemExit(
            "--tracked-bodies is required (or --target-task must resolve to a "
            "task in nla.eval.steerability.predicates.GOAL_TASKS)"
        )

    options: dict[str, object] | None = None
    if args.steer_h_path:
        arr = np.load(args.steer_h_path).astype(np.float32, copy=False)
        options = {"steer_h": arr}
        if args.steer_placement is not None or args.steer_blend is not None:
            spec_dict: dict[str, object] = {}
            if args.steer_placement is not None:
                spec_dict["placement"] = args.steer_placement
            if args.steer_blend is not None:
                spec_dict["blend"] = float(args.steer_blend)
            options["steer_spec"] = spec_dict
    if args.steer_disabled:
        # Wins over steer_h: the wrapper reads steer_disabled first and
        # falls through to the base policy. The vector is harmless extra
        # bytes on the wire but we leave it so the calling job code can
        # still verify the artifact exists for the no_steer arm.
        if options is None:
            options = {}
        options["steer_disabled"] = True

    early_stop_cb = None
    if args.early_stop_on_success and target_task:
        from nla.eval.steerability.predicates import predicate_fires, resolve_task
        spec = resolve_task(target_task)
        def _cb(traj: dict, _spec=spec) -> bool:
            return predicate_fires(traj, _spec)
        early_stop_cb = _cb

    summary = run_one_rollout(
        env_name=args.env_name,
        seed=args.seed,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        tracked_bodies=tracked_bodies,
        target_body=args.target_body,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        fps=args.fps,
        steps_per_render=args.steps_per_render,
        suppress_done=args.suppress_done,
        options=options,
        early_stop_on_predicate=early_stop_cb,
        capture_frames=not args.no_frames,
        return_trajectory=bool(target_task),
        policy_language_override=args.policy_language_override,
    )

    if target_task:
        from nla.eval.steerability.predicates import (
            DEFAULT_SHAPING,
            ShapingWeights,
            score as predicate_score,
        )
        traj = summary.pop("_trajectory", None)
        if traj is not None:
            weights = None
            if args.w_predicate is not None:
                weights = ShapingWeights(
                    w_predicate=float(args.w_predicate),
                    w_dist=DEFAULT_SHAPING.w_dist,
                    w_displace=DEFAULT_SHAPING.w_displace,
                    w_contact=DEFAULT_SHAPING.w_contact,
                    d_max=DEFAULT_SHAPING.d_max,
                    disp_max=DEFAULT_SHAPING.disp_max,
                    contact_near_m=DEFAULT_SHAPING.contact_near_m,
                )
            score_breakdown = predicate_score(traj, target_task, weights=weights)
            summary["r_sim"] = score_breakdown["r"]
            summary["sim_score_breakdown"] = score_breakdown
        else:
            summary["r_sim"] = None
            summary["sim_score_breakdown"] = None

    from nla.eval.steerability.json_utils import dumps_rollout_json

    print(dumps_rollout_json(summary))


if __name__ == "__main__":
    _cli()

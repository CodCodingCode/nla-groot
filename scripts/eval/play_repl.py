"""Warm interactive REPL for poking at LIBERO + the live GR00T policy server.

Run via ``scripts/eval/play.sh`` (which sets PYTHONPATH and picks the libero
venv) — this file is meant to be loaded with ``ipython -i`` / ``python -i``
so heavy imports happen once at startup and you can poke at warm globals
across hours of debugging.

Exposed globals after startup:
    client    PolicyClient pointed at the auto-detected live steer server
    tasks     list of cached tasks (suite, task, language, n_init)

Exposed helpers:
    play(task, init_id=0, steer_text=None, max_steps=200, save_video=None,
         steer_disabled=False, n_action_steps=8)
        Run a rollout against the policy server. Returns a dict with
        {'frames': list[ndarray], 'success': bool, 'steps': int,
         'video_path': Path|None, 'summary': dict}.
        Reuses a warm LiberoEnv per task — second call on the same task is
        ~5s instead of ~10s.

    view(task, init_id=0)
        Return the agentview RGB array for (task, init_id). Uses the cached
        preview PNG when available; otherwise instantiates the env and
        renders one frame.

    show(arr_or_path)
        Save the frame/path to a unique file under data/play_out/ and print
        the path (helpful from ipython where matplotlib isn't always set up).

    info()
        Print live-server status and a short usage cheatsheet.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# --- MuJoCo render backend: pin to osmesa BEFORE any libero import. -------
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAY_OUT_DIR = REPO_ROOT / "data" / "play_out"
PLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Warm imports ----------------------------------------------------------
from nla.eval.steerability.state_cache import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    apply_init_state,
    list_cached_tasks,
    load_init_states,
    load_task_meta,
    preview_path,
)
from nla.eval.steerability.rollout import (  # noqa: E402
    _to_server_obs,
    _unpack_action_chunk,
    encode_mp4,
    render_panel,
)
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402


# --- Live server auto-detection -------------------------------------------
def _detect_live_server_port(default: int = 5556) -> int:
    """Scan data/sft/*/steer_server_logs/server.pid for a live process and
    return its --port. Falls back to ``default`` (5556) if nothing is alive."""
    for pf in sorted((REPO_ROOT / "data" / "sft").glob("*/steer_server_logs/server.pid")):
        try:
            pid = int(pf.read_text().strip())
        except Exception:
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        tokens = cmdline.split()
        for i, tok in enumerate(tokens):
            if tok == "--port" and i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    pass
    return default


LIVE_PORT = _detect_live_server_port()
client = PolicyClient(host="localhost", port=LIVE_PORT, timeout_ms=120_000)
tasks = list_cached_tasks()

# Per-task LiberoEnv cache — building the env is ~10s, mostly BDDL parsing
# + MuJoCo scene compile. Reuse across calls to play()/view().
_ENV_CACHE: dict[str, LiberoEnv] = {}


def _get_env(task: str) -> tuple[LiberoEnv, dict]:
    if task in _ENV_CACHE:
        meta = load_task_meta(task)
        return _ENV_CACHE[task], meta
    meta = load_task_meta(task)
    env = LiberoEnv(
        task_bddl_file=meta["bddl_file"],
        task_description=meta["task_description"],
        suppress_done=True,
    )
    _ENV_CACHE[task] = env
    return env, meta


# --- Helpers ---------------------------------------------------------------
def view(task: str, init_id: int = 0) -> np.ndarray:
    """Return an (H, W, 3) uint8 agentview frame for (task, init_id)."""
    cached = preview_path(task, init_id)
    if cached is not None:
        import imageio.v3 as iio
        return iio.imread(cached)
    env, _ = _get_env(task)
    states = load_init_states(task)
    obs, _ = apply_init_state(env, states[init_id])
    return obs["video.image"].copy()


def show(arr_or_path, name: str | None = None) -> Path:
    """Save a frame array or copy a file to data/play_out/<name>.png; return path."""
    import imageio.v3 as iio
    out_name = name or f"frame_{int(time.time())}.png"
    if not out_name.endswith((".png", ".jpg", ".mp4")):
        out_name += ".png"
    out_path = PLAY_OUT_DIR / out_name
    if isinstance(arr_or_path, (str, Path)):
        out_path.write_bytes(Path(arr_or_path).read_bytes())
    else:
        iio.imwrite(out_path, np.asarray(arr_or_path))
    print(f"wrote {out_path}")
    return out_path


def play(
    task: str,
    init_id: int = 0,
    steer_text: str | None = None,
    *,
    max_steps: int = 200,
    save_video: str | bool | None = True,
    steer_disabled: bool = False,
    n_action_steps: int = 8,
    policy_language_override: str | None = None,
) -> dict[str, Any]:
    """Run one rollout against the live policy server.

    - ``steer_text``: replace the env's BDDL task_description in the obs sent
      to the policy (language_swap protocol). The underlying MuJoCo scene
      and predicate are unchanged — this is what the eval-v2 mismatched-intent
      arm does. Equivalent to ``policy_language_override=steer_text``.
    - ``steer_disabled``: send ``options['steer_disabled']=True`` every step.
      The NlaSteerGr00tPolicy wrapper short-circuits to the base policy
      without applying the steer hook (no-steer causal arm).
    - ``save_video``: True → save to ``data/play_out/<task>_init<id>_<ts>.mp4``;
      str → save under that name; False/None → don't save.
    """
    env, meta = _get_env(task)
    states = load_init_states(task)
    if init_id >= len(states):
        raise IndexError(f"init_id {init_id} >= n_init_states {len(states)} for {task}")

    obs, info = apply_init_state(env, states[init_id])
    options: dict[str, Any] = {}
    if steer_disabled:
        options["steer_disabled"] = True
    lang_override = policy_language_override if policy_language_override is not None else steer_text

    frames: list[np.ndarray] = [render_panel(obs, success=bool(info.get("success", False)), t=0)]
    success = False
    t = 0
    t0 = time.time()
    while t < max_steps:
        action_chunk, _ = client.get_action(
            _to_server_obs(obs, policy_language_override=lang_override),
            options=options or None,
        )
        sub_actions = _unpack_action_chunk(action_chunk, n_action_steps=n_action_steps)
        if not sub_actions:
            break
        for sub in sub_actions:
            obs, _r, done, _trunc, info = env.step(sub)
            t += 1
            success = bool(info.get("success", False))
            frames.append(render_panel(obs, success=success, t=t))
            if t >= max_steps or done:
                break
        if t >= max_steps or done:
            break

    wall = time.time() - t0
    video_path: Path | None = None
    if save_video:
        name = save_video if isinstance(save_video, str) else (
            f"{task}_init{init_id:03d}_{int(time.time())}.mp4"
        )
        if not name.endswith(".mp4"):
            name += ".mp4"
        video_path = PLAY_OUT_DIR / name
        encode_mp4(frames, video_path, fps=20)
        print(f"video → {video_path}")

    summary = {
        "task": task,
        "init_id": init_id,
        "steer_text": steer_text,
        "steer_disabled": steer_disabled,
        "success": success,
        "steps": t,
        "wall_seconds": round(wall, 2),
        "language_native": meta["task_description"],
    }
    print(
        f"play[{task} init={init_id}] success={success} steps={t} "
        f"wall={wall:.1f}s  steer={'OFF' if steer_disabled else (steer_text or 'native')}"
    )
    return {
        "frames": frames,
        "success": success,
        "steps": t,
        "video_path": video_path,
        "summary": summary,
    }


def info() -> None:
    print(f"live policy server: localhost:{LIVE_PORT}")
    print(f"cached tasks:       {len(tasks)}")
    print(f"output dir:         {PLAY_OUT_DIR}")
    print()
    print("cheatsheet:")
    print("  view('put_the_bowl_on_the_plate', init_id=0)        # one frame")
    print("  show(view('...'))                                    # save to play_out/")
    print("  play('put_the_bowl_on_the_plate')                    # native intent rollout")
    print("  play('...', steer_text='put the wine bottle on the plate')")
    print("  play('...', steer_disabled=True)                     # no-steer causal arm")
    print("  tasks[:5]                                            # cached task list")


# Banner — printed once when ipython -i loads this file.
print(f"[play_repl] policy server localhost:{LIVE_PORT}  |  {len(tasks)} cached tasks")
print("[play_repl] try: info() / view('put_the_bowl_on_the_plate') / play('put_the_bowl_on_the_plate')")

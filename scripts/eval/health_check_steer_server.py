#!/usr/bin/env python
"""Health check for a running NlaSteerGr00tPolicy ZMQ server.

This probe verifies the server is *accepting* requests, not merely that the
process is alive on the host:port. It performs three increasingly-strict
checks against the ``PolicyServer`` exposed by
``scripts/eval/run_gr00t_server_nla_steer.py``:

1. ``ping`` round-trip — confirms the ZMQ ``REP`` socket is bound and the
   server's request loop is consuming messages.
2. ``get_modality_config`` — confirms the wrapped ``Gr00tPolicy`` (possibly
   inside ``NlaSteerGr00tPolicy``) is registered as the ``get_action``
   handler and exposes a non-empty modality dict.
3. Synthetic ``get_action`` — builds a zero-valued observation matching the
   server's modality config (video frames, state vector, and a placeholder
   annotation), attaches a tiny zero ``steer_h`` vector via ``options`` (the
   same dynamic-steer pathway the GRPO sim-reward worker uses for per-call
   steering), and checks that the response decodes to a
   ``(action_dict, info_dict)`` tuple whose ``action`` dict contains at
   least one ``action.*`` key holding a multi-row ndarray (the action
   chunk). This exercises the full backbone + steer hook + action-head
   forward path, so a passing probe means the orchestrator can immediately
   start sending real observations.

Exits 0 on HEALTHY, 1 on any failure (timeout, missing keys, server error,
malformed response, etc.).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/eval/health_check_steer_server.py \
        --host localhost --port 5555
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Any

import numpy as np


def _make_synthetic_observation(
    modality_cfg: dict[str, Any],
    *,
    video_hw: tuple[int, int] = (256, 256),
    state_dim: int = 8,
) -> dict[str, Any]:
    """Build a zero-valued observation matching the server's modality config.

    Shapes are heuristic and tuned for the LIBERO_PANDA defaults: each video
    key gets ``[T, H, W, 3]`` uint8 frames where ``T = len(delta_indices)``,
    each state key gets ``[T, state_dim]`` float32 zeros, and each annotation
    key gets a single placeholder string. Override ``--video-hw`` and
    ``--state-dim`` for other embodiments.
    """
    obs: dict[str, Any] = {}
    H, W = video_hw
    for _name, cfg in modality_cfg.items():
        delta = getattr(cfg, "delta_indices", None) or [0]
        keys = getattr(cfg, "modality_keys", None) or []
        T = max(len(delta), 1)
        for key in keys:
            if key.startswith("video."):
                obs[key] = np.zeros((T, H, W, 3), dtype=np.uint8)
            elif key.startswith("state."):
                obs[key] = np.zeros((T, state_dim), dtype=np.float32)
            elif key.startswith("annotation."):
                obs[key] = ["health-check probe"]
            elif key.startswith("action."):
                continue
            else:
                obs[key] = np.zeros((T,), dtype=np.float32)
    return obs


def _validate_action_response(response: Any) -> tuple[bool, str]:
    """Return (ok, detail) for a ``(action_dict, info_dict)`` response."""
    if not isinstance(response, tuple) or len(response) != 2:
        return False, f"expected tuple of length 2, got {type(response).__name__}={response!r:.120}"
    action, _info = response
    if not isinstance(action, dict) or not action:
        return False, f"action is not a non-empty dict (got {type(action).__name__})"
    action_keys = [k for k in action.keys() if str(k).startswith("action.")]
    if not action_keys:
        return False, f"no 'action.*' keys in action dict: {list(action.keys())!r}"
    bad = []
    for k in action_keys:
        v = action[k]
        if not isinstance(v, np.ndarray) or v.ndim < 2 or v.shape[0] < 1:
            bad.append((k, type(v).__name__, getattr(v, "shape", None)))
    if bad:
        return False, f"action chunk(s) have unexpected shape: {bad!r}"
    primary = action_keys[0]
    return True, f"valid action chunk; keys={action_keys}, shape[{primary}]={action[primary].shape}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Health-check a running NlaSteerGr00tPolicy ZMQ server.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="server host")
    parser.add_argument("--port", type=int, default=5555, help="server port")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=60_000,
        help="ZMQ recv/send timeout per request, in ms. Needs to be generous "
        "because the first inference call warms the model on GPU (default 60s).",
    )
    parser.add_argument(
        "--steer-dim",
        type=int,
        default=2048,
        help="Dimensionality of the zero steer_h vector to inject via options "
        "(default 2048 = GR00T N1.7 BACKBONE_EMBEDDING_DIM).",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
        help="Synthetic state vector dim per timestep (default 8 = LIBERO_PANDA "
        "7 joint + 1 gripper). Override for other embodiments.",
    )
    parser.add_argument(
        "--video-hw",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="Synthetic video frame H W (default 256 256).",
    )
    parser.add_argument(
        "--skip-action",
        action="store_true",
        help="Skip the synthetic get_action probe and only ping + check the "
        "modality config (use when the synthetic obs shape is wrong for the "
        "embodiment and you only need a liveness check).",
    )
    args = parser.parse_args()

    try:
        from gr00t.policy.server_client import PolicyClient
    except Exception:
        print("FATAL: could not import gr00t.policy.server_client.PolicyClient", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"[health] connecting to tcp://{args.host}:{args.port}")
    client = PolicyClient(
        host=args.host,
        port=args.port,
        timeout_ms=args.timeout_ms,
        strict=False,
    )

    t0 = time.time()
    if not client.ping():
        print(f"FAIL: ping() returned False ({time.time() - t0:.2f}s)", file=sys.stderr)
        return 1
    print(f"[health] ping ok ({time.time() - t0:.2f}s)")

    try:
        modality_cfg = client.get_modality_config()
    except Exception as exc:
        print(f"FAIL: get_modality_config raised: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        return 1
    if not isinstance(modality_cfg, dict) or not modality_cfg:
        print(
            f"FAIL: modality_config is empty or wrong type: {type(modality_cfg).__name__}={modality_cfg!r:.120}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[health] modality config ok ({len(modality_cfg)} modalities: "
        f"{sorted(modality_cfg.keys())})"
    )

    if args.skip_action:
        print("HEALTHY (skipped get_action probe)")
        return 0

    observation = _make_synthetic_observation(
        modality_cfg,
        video_hw=tuple(args.video_hw),
        state_dim=args.state_dim,
    )
    options = {"steer_h": np.zeros((args.steer_dim,), dtype=np.float32)}
    print(
        f"[health] sending synthetic get_action ({len(observation)} obs keys, "
        f"tiny steer_h dim={args.steer_dim})"
    )
    t0 = time.time()
    try:
        response = client._get_action(observation, options=options)
    except Exception as exc:
        print(
            f"FAIL: get_action raised after {time.time() - t0:.2f}s: {exc!r}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 1

    ok, detail = _validate_action_response(response)
    if not ok:
        print(f"FAIL: invalid action response ({time.time() - t0:.2f}s): {detail}", file=sys.stderr)
        return 1

    print(f"[health] get_action ok ({time.time() - t0:.2f}s): {detail}")
    print("HEALTHY")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fake LIBERO rollout for SimRewardWorker profiling.

Mirrors the CLI surface and JSON-on-stdout contract of
``src/nla/eval/steerability/rollout.py`` but does NOT import anything
heavy (no LIBERO, no GR00T, no numpy beyond what comes in stdlib reach).

It only:

  1. Parses the same flags ``sim_reward._run_rollout_subprocess`` passes.
  2. Sleeps for a duration drawn from ``[NLA_FAKE_SLEEP_MIN,
     NLA_FAKE_SLEEP_MAX]`` (defaults: 0.5..2.5 s), seeded by ``--seed`` so
     replayed jobs are deterministic.
  3. Optionally exits non-zero with probability ``NLA_FAKE_FAIL_RATE`` to
     exercise the worker's error path (default 0).
  4. Prints a JSON summary with the same keys SimRewardWorker reads.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--env-name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--policy-host", default="localhost")
    p.add_argument("--policy-port", type=int, default=5555)
    p.add_argument("--target-task", default=None)
    p.add_argument("--steer-h-path", default=None)
    p.add_argument("--steer-placement", default=None)
    p.add_argument("--steer-blend", type=float, default=None)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--no-frames", action="store_true")
    p.add_argument("--early-stop-on-success", action="store_true")
    return p


def _draw_sleep(seed: int) -> float:
    lo = float(os.environ.get("NLA_FAKE_SLEEP_MIN", "0.5"))
    hi = float(os.environ.get("NLA_FAKE_SLEEP_MAX", "2.5"))
    if hi < lo:
        lo, hi = hi, lo
    rng = random.Random(int(seed) ^ 0xA7A7A7)
    return rng.uniform(lo, hi)


def _maybe_fail(seed: int) -> bool:
    rate = float(os.environ.get("NLA_FAKE_FAIL_RATE", "0.0"))
    if rate <= 0:
        return False
    rng = random.Random((int(seed) << 1) ^ 0xDEAD)
    return rng.random() < rate


def main(argv: list[str] | None = None) -> int:
    args, _unknown = _build_parser().parse_known_args(argv)

    if _maybe_fail(args.seed):
        sys.stderr.write(f"fake rollout: forced failure for seed={args.seed}\n")
        return 17

    dt = _draw_sleep(args.seed)
    time.sleep(dt)

    rng = random.Random(int(args.seed))
    predicate = 1.0 if rng.random() < 0.4 else 0.0
    r_dist = round(rng.uniform(-0.3, 0.7), 4)
    r_displace = round(rng.uniform(-0.1, 0.3), 4)
    r_contact = round(rng.uniform(0.0, 0.2), 4)
    r_sim = round(predicate + r_dist + r_displace + r_contact, 4)
    n_steps = rng.randint(10, args.max_episode_steps)
    early = bool(predicate > 0 and args.early_stop_on_success)

    summary = {
        "env_name": args.env_name,
        "seed": args.seed,
        "n_steps": n_steps,
        "max_episode_steps": args.max_episode_steps,
        "early_stopped": early,
        "success_any": bool(predicate > 0),
        "r_sim": r_sim,
        "sim_score_breakdown": {
            "r": r_sim,
            "predicate": predicate,
            "r_dist": r_dist,
            "r_displace": r_displace,
            "r_contact": r_contact,
        },
        "_fake_sleep_s": round(dt, 4),
    }
    sys.stdout.write(json.dumps(summary, indent=2, default=float))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Polling watcher for SFT runs.

Monitors ``<sft-dir>/metrics.jsonl`` until the run is finished (final phase
recorded or ``step >= target_steps``), times out, or stalls. Designed to be
chained from a parent orchestrator: on success exit 0, on timeout exit 1, on
stall exit 2.

Pure stdlib; CPU-only; never modifies the SFT run's files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional


PROGRESS_FMT = "[wait_sft] step={step}/{target}  ETA={eta_min} min  last_ce={last_ce}"
STALL_WARN_MULT = 10
STALL_FAIL_S = 30 * 60


def _read_last_jsonl(path: Path) -> Optional[dict]:
    """Return the last parseable JSON object in ``path`` (or None)."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return None
    if size == 0:
        return None
    chunk = min(size, 65536)
    with path.open("rb") as fh:
        fh.seek(size - chunk)
        tail = fh.read()
    for raw in reversed(tail.splitlines()):
        line = raw.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    """Return up to the last ``n`` JSON objects from ``path``."""
    if n <= 0:
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = 65536
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read = min(block, size)
                size -= read
                fh.seek(size)
                data = fh.read(read) + data
    except FileNotFoundError:
        return []
    out: list[dict] = []
    for raw in data.splitlines()[-n:]:
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _resolve_target_steps(sft_dir: Path, override: Optional[int]) -> Optional[int]:
    if override is not None:
        return override
    cfg = sft_dir / "config.json"
    if not cfg.is_file():
        return None
    try:
        with cfg.open("r") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    val = data.get("total_steps")
    return int(val) if isinstance(val, (int, float)) else None


def _eta_minutes(samples: list[dict], step: int, target: int) -> str:
    """Mean steps/sec over the trailing window of train rows -> minutes left."""
    rows = [
        r
        for r in samples
        if isinstance(r.get("step"), (int, float))
        and isinstance(r.get("elapsed_s"), (int, float))
    ]
    if len(rows) < 2 or target <= step:
        return "?"
    ds = rows[-1]["step"] - rows[0]["step"]
    dt = rows[-1]["elapsed_s"] - rows[0]["elapsed_s"]
    if ds <= 0 or dt <= 0:
        return "?"
    sps = ds / dt
    remaining = max(0, target - step)
    return f"{remaining / sps / 60:.1f}"


class _Logger:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._fh = log_path.open("a", buffering=1)

    def write(self, msg: str) -> None:
        line = msg.rstrip("\n")
        print(line, flush=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _format_progress(last: dict, target: Optional[int], window: list[dict]) -> str:
    step = int(last.get("step", 0) or 0)
    target_disp = target if target is not None else "?"
    eta = _eta_minutes(window, step, target) if isinstance(target, int) else "?"
    ce = last.get("ce")
    last_ce = f"{ce:.4f}" if isinstance(ce, (int, float)) else "n/a"
    return PROGRESS_FMT.format(
        step=step, target=target_disp, eta_min=eta, last_ce=last_ce
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for SFT run to complete.")
    ap.add_argument("--sft-dir", required=True, type=Path)
    ap.add_argument("--target-steps", type=int, default=None)
    ap.add_argument("--poll-interval-s", type=int, default=60)
    ap.add_argument("--timeout-s", type=int, default=14400)
    args = ap.parse_args()

    sft_dir: Path = args.sft_dir.resolve()
    metrics_path = sft_dir / "metrics.jsonl"
    log_path = sft_dir / "wait_log.txt"

    if not sft_dir.is_dir():
        print(f"[wait_sft] ERROR: --sft-dir not found: {sft_dir}", flush=True)
        return 1

    target_steps = _resolve_target_steps(sft_dir, args.target_steps)

    logger = _Logger(log_path)
    logger.write(
        f"[wait_sft] start dir={sft_dir} target_steps={target_steps} "
        f"poll={args.poll_interval_s}s timeout={args.timeout_s}s"
    )

    poll = max(1, int(args.poll_interval_s))
    stall_warn_s = STALL_WARN_MULT * poll
    deadline = time.monotonic() + args.timeout_s

    poll_count = 0
    last_step_seen: Optional[int] = None
    last_progress_ts = time.monotonic()
    warned_stall = False
    last_state: Optional[str] = None

    try:
        while True:
            poll_count += 1
            now = time.monotonic()

            if now > deadline:
                logger.write(
                    f"[wait_sft] TIMEOUT after {args.timeout_s}s; last_step={last_step_seen}"
                )
                return 1

            last = _read_last_jsonl(metrics_path) if metrics_path.is_file() else None

            if last is None:
                state = "no-metrics"
                if state != last_state:
                    logger.write(f"[wait_sft] waiting for {metrics_path} to appear")
                    last_state = state
            else:
                step = last.get("step")
                if isinstance(step, (int, float)):
                    step_i = int(step)
                    if last_step_seen is None or step_i != last_step_seen:
                        last_step_seen = step_i
                        last_progress_ts = now
                        warned_stall = False

                if last.get("phase") == "final" or (
                    target_steps is not None
                    and isinstance(step, (int, float))
                    and int(step) >= target_steps
                ):
                    window = _tail_jsonl(metrics_path, 50)
                    logger.write(_format_progress(last, target_steps, window))
                    logger.write(
                        f"[wait_sft] DONE phase={last.get('phase')} step={last.get('step')}"
                    )
                    return 0

                stalled_for = now - last_progress_ts
                if stalled_for >= STALL_FAIL_S:
                    logger.write(
                        f"[wait_sft] SFT appears stalled (no new step for "
                        f"{stalled_for:.0f}s); last_step={last_step_seen}"
                    )
                    return 2
                if stalled_for >= stall_warn_s and not warned_stall:
                    logger.write(
                        f"[wait_sft] WARN: no new metrics line for "
                        f"{stalled_for:.0f}s (>{stall_warn_s}s); last_step={last_step_seen}"
                    )
                    warned_stall = True

                if poll_count % 5 == 0:
                    window = _tail_jsonl(metrics_path, 50)
                    logger.write(_format_progress(last, target_steps, window))

            time.sleep(poll)
    except KeyboardInterrupt:
        logger.write("[wait_sft] interrupted")
        return 130
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Profile :class:`nla.training.sim_reward.SimRewardWorker` without LIBERO.

This script never imports LIBERO, GR00T, or MuJoCo. It points the worker at
a sibling fake rollout (``_profile_sim_worker_fake_rollout.py``) that just
sleeps for a configurable duration and emits the same JSON ``rollout.py``
would. That lets us cheaply verify, on pure CPU:

  * Parallelism scaling -- does ``n_workers`` give the expected speedup?
  * Cache behavior     -- pre-populated keys must short-circuit subprocess.
  * Timeout behavior   -- jobs over ``timeout_s`` must surface as errors,
                          not poison the whole batch.

It then reads ``data/grpo/sim_reward_cache.jsonl`` (if present) to estimate
real-rollout latency, and writes a markdown profile + recommendation to
``data/grpo/sim_worker_profile.md``.

Run with::

    PYTHONPATH=src .venv/bin/python scripts/training/profile_sim_worker.py

(``PYTHONPATH`` is set inside the script if ``nla`` isn't on ``sys.path``
yet, so the direct invocation in the task's instructions works too.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from nla.training.sim_reward import (  # noqa: E402
    SimRewardJob,
    SimRewardWorker,
    load_sim_cache,
    sim_cache_key,
)


FAKE_ROLLOUT = HERE / "_profile_sim_worker_fake_rollout.py"
DEFAULT_REPORT = REPO_ROOT / "data" / "grpo" / "sim_worker_profile.md"
EXISTING_CACHE = REPO_ROOT / "data" / "grpo" / "sim_reward_cache.jsonl"

# 32 jobs = 16 unique keys, each appearing twice. The first 8 unique keys
# are pre-populated in the cache so each trial has a known 50% cache hit
# rate AND an interesting 16-job parallel workload to time.
N_JOBS_TOTAL = 32
N_UNIQUE_KEYS = 16
N_PREWARMED = 8
ASSERT_DUP_FACTOR = N_JOBS_TOTAL // N_UNIQUE_KEYS  # 2


# ----------------------------------------------------------------------------
# Job construction
# ----------------------------------------------------------------------------


def _make_jobs(
    n_total: int,
    n_unique: int,
    *,
    h_dim: int = 8,
    env_name: str = "LIBERO_GOAL_put_the_bowl_on_the_plate",
    target_task: str = "put_the_bowl_on_the_plate",
    sim_max_steps: int = 100,
    placement: str = "image_patch",
    blend: float = 1.0,
) -> list[SimRewardJob]:
    """Generate ``n_total`` jobs that map to ``n_unique`` distinct cache keys."""
    assert n_total % n_unique == 0, "n_total must be a multiple of n_unique"
    jobs: list[SimRewardJob] = []
    for i in range(n_total):
        u = i % n_unique  # cycle through unique keys with stride 1
        jobs.append(SimRewardJob(
            env_name=env_name,
            target_task=target_task,
            source_id=f"ep{u:02d}_t{u*5:03d}",
            text=f"rollout text variant #{u}",
            seed=int(u),  # same seed -> same fake sleep duration (determinism)
            steer_h=np.full(h_dim, fill_value=float(u) / max(1, n_unique - 1),
                            dtype=np.float32),
            sim_max_steps=int(sim_max_steps),
            placement=placement,
            blend=float(blend),
        ))
    return jobs


def _prewarm_cache(cache_path: Path, jobs: list[SimRewardJob]) -> int:
    """Write fake cache entries for the given jobs. Returns count written."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with cache_path.open("w") as f:
        for job in jobs:
            key = sim_cache_key(
                job.env_name, job.target_task, job.source_id,
                job.text, job.seed, job.sim_max_steps,
            )
            if key in seen:
                continue
            seen.add(key)
            entry = {
                "key": key,
                "r_sim": 0.5,
                "predicate": 0.0,
                "r_dist": 0.5,
                "r_displace": 0.0,
                "r_contact": 0.0,
                "n_steps": 50,
                "early_stopped": False,
                "elapsed_s": 1.5,
                "success_any": False,
                "env_name": job.env_name,
                "target_task": job.target_task,
                "source_id": job.source_id,
                "text": job.text,
                "seed": job.seed,
                "sim_max_steps": job.sim_max_steps,
                "_prewarm": True,
            }
            f.write(json.dumps(entry) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------
# Trial / metrics
# ----------------------------------------------------------------------------


@dataclass
class TrialResult:
    n_workers: int
    n_jobs: int
    wall_s: float
    n_cache_hits: int
    n_errors: int
    n_dispatched: int
    p50_dispatched_elapsed_s: float
    p95_dispatched_elapsed_s: float

    @property
    def throughput(self) -> float:
        return self.n_jobs / self.wall_s if self.wall_s > 0 else float("inf")

    @property
    def hit_rate(self) -> float:
        return self.n_cache_hits / self.n_jobs if self.n_jobs else 0.0

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.n_jobs if self.n_jobs else 0.0


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return float(s[i])


def _run_trial(
    *,
    n_workers: int,
    jobs: list[SimRewardJob],
    prewarm_jobs: list[SimRewardJob],
    sleep_min: float,
    sleep_max: float,
    timeout_s: float,
    scratch_root: Path,
    fake_rollout: Path,
    trial_label: str,
) -> TrialResult:
    scratch = scratch_root / trial_label
    cache_path = scratch / "sim_cache.jsonl"
    n_prewarm = _prewarm_cache(cache_path, prewarm_jobs)
    worker = SimRewardWorker(
        n_workers=n_workers,
        sim_max_steps=jobs[0].sim_max_steps,
        placement=jobs[0].placement,
        blend=jobs[0].blend,
        rollout_python=sys.executable,
        rollout_script=str(fake_rollout),
        cache_path=cache_path,
        timeout_s=timeout_s,
        env_overrides={
            "NLA_FAKE_SLEEP_MIN": f"{sleep_min:.3f}",
            "NLA_FAKE_SLEEP_MAX": f"{sleep_max:.3f}",
            "PYTHONPATH": "",  # the fake script needs none of our deps
        },
        scratch_dir=scratch / "scratch",
    )
    t0 = time.time()
    results = worker.compute(jobs)
    wall = time.time() - t0

    n_cache_hits = sum(1 for r in results if r.cached)
    n_errors = sum(1 for r in results if r.error is not None)
    dispatched_elapsed = [r.elapsed_s for r in results if not r.cached and r.error is None]
    n_dispatched = len(results) - n_cache_hits

    # Sanity: at least the prewarmed entries should appear as hits.
    assert n_cache_hits >= n_prewarm * (len(jobs) // max(1, len(prewarm_jobs) * 2)), (
        "cache hits below expected -- something is off with prewarm logic"
    )

    return TrialResult(
        n_workers=n_workers,
        n_jobs=len(jobs),
        wall_s=wall,
        n_cache_hits=n_cache_hits,
        n_errors=n_errors,
        n_dispatched=n_dispatched,
        p50_dispatched_elapsed_s=_pct(dispatched_elapsed, 0.5),
        p95_dispatched_elapsed_s=_pct(dispatched_elapsed, 0.95),
    )


def _run_timeout_demo(
    *,
    scratch_root: Path,
    fake_rollout: Path,
    sleep_s: float = 5.0,
    timeout_s: float = 1.5,
    n_workers: int = 4,
    n_jobs: int = 4,
) -> TrialResult:
    """Force every job to exceed ``timeout_s`` and confirm they surface as errors."""
    scratch = scratch_root / "timeout_demo"
    cache_path = scratch / "sim_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = [
        SimRewardJob(
            env_name="LIBERO_GOAL_timeout_demo",
            target_task="put_the_bowl_on_the_plate",
            source_id=f"timeout_{i:02d}",
            text=f"timeout case {i}",
            seed=1000 + i,
            steer_h=np.zeros(4, dtype=np.float32),
            sim_max_steps=100,
            placement="image_patch",
            blend=1.0,
        )
        for i in range(n_jobs)
    ]
    worker = SimRewardWorker(
        n_workers=n_workers,
        rollout_python=sys.executable,
        rollout_script=str(fake_rollout),
        cache_path=cache_path,
        timeout_s=timeout_s,
        env_overrides={
            "NLA_FAKE_SLEEP_MIN": f"{sleep_s:.3f}",
            "NLA_FAKE_SLEEP_MAX": f"{sleep_s:.3f}",
            "PYTHONPATH": "",
        },
        scratch_dir=scratch / "scratch",
    )
    t0 = time.time()
    results = worker.compute(jobs)
    wall = time.time() - t0
    return TrialResult(
        n_workers=n_workers,
        n_jobs=len(jobs),
        wall_s=wall,
        n_cache_hits=sum(1 for r in results if r.cached),
        n_errors=sum(1 for r in results if r.error is not None),
        n_dispatched=sum(1 for r in results if not r.cached),
        p50_dispatched_elapsed_s=_pct([r.elapsed_s for r in results if not r.cached], 0.5),
        p95_dispatched_elapsed_s=_pct([r.elapsed_s for r in results if not r.cached], 0.95),
    )


# ----------------------------------------------------------------------------
# Existing cache stats
# ----------------------------------------------------------------------------


@dataclass
class HistoricalStats:
    n_entries: int
    mean_elapsed_s: float
    p50_elapsed_s: float
    p95_elapsed_s: float
    p99_elapsed_s: float
    n_errors: int


def _load_historical(cache_path: Path) -> HistoricalStats | None:
    if not cache_path.exists():
        return None
    raw = load_sim_cache(cache_path)
    if not raw:
        return None
    elapsed = [float(e.get("elapsed_s", 0.0)) for e in raw.values()
               if float(e.get("elapsed_s", 0.0)) > 0]
    n_err = sum(1 for e in raw.values() if e.get("error"))
    if not elapsed:
        return HistoricalStats(len(raw), 0.0, 0.0, 0.0, 0.0, n_err)
    return HistoricalStats(
        n_entries=len(raw),
        mean_elapsed_s=statistics.fmean(elapsed),
        p50_elapsed_s=_pct(elapsed, 0.5),
        p95_elapsed_s=_pct(elapsed, 0.95),
        p99_elapsed_s=_pct(elapsed, 0.99),
        n_errors=n_err,
    )


# ----------------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------------


def _recommend(
    trials: list[TrialResult],
    hist: HistoricalStats | None,
    *,
    grpo_b: int,
    grpo_k: int,
) -> tuple[int, float, str]:
    """Pick (n_workers, timeout_s, reason) from the trial table + historical stats."""
    # Speedup vs. single-worker baseline.
    base = next((t for t in trials if t.n_workers == 1), trials[0])
    speedups = [(t.n_workers, base.wall_s / t.wall_s if t.wall_s > 0 else 1.0)
                for t in trials]
    # Pick the largest n_workers whose marginal efficiency is still > 0.6
    # (i.e., we got at least 60% of ideal scaling from the prior step).
    chosen = trials[0]
    best = base
    for prev, t in zip(trials, trials[1:]):
        if prev.wall_s == 0 or t.wall_s == 0:
            continue
        marginal = (prev.wall_s / t.wall_s) / (t.n_workers / max(1, prev.n_workers))
        if marginal >= 0.6:
            best = t
        else:
            break
    chosen = best

    # Per-rollout latency budget for the timeout:
    #   * Prefer historical p95 * 4 (real LIBERO).
    #   * Else fall back to fake-rollout p95 * 4 with a 60-s floor.
    if hist and hist.p95_elapsed_s > 0:
        target_p95 = hist.p95_elapsed_s
        timeout_s = max(60.0, math.ceil(target_p95 * 4.0))
        reason_latency = (
            f"historical p95 rollout latency = {target_p95:.1f}s "
            f"(from {hist.n_entries} cached entries)"
        )
    else:
        fallback_p95 = max(t.p95_dispatched_elapsed_s for t in trials)
        target_p95 = fallback_p95
        # When all we have is the fake's tiny sleep, set a SAFE production
        # default rather than trusting the toy number.
        timeout_s = 300.0
        reason_latency = (
            "no historical sim_reward_cache yet -- defaulting timeout_s=300s "
            f"(fake-rollout p95 was {fallback_p95:.2f}s; safe headroom for "
            "real LIBERO rollouts which typically take 30-120s)"
        )

    reason = (
        f"Across n_workers in {{1,2,4,8}}, n_workers={chosen.n_workers} delivered "
        f"the best wall time ({chosen.wall_s:.2f}s vs. {base.wall_s:.2f}s at "
        f"n_workers=1, speedup {base.wall_s/chosen.wall_s:.2f}x with "
        f"{(base.wall_s/chosen.wall_s)/chosen.n_workers*100:.0f}% parallel efficiency). "
        f"For GRPO with B={grpo_b}, K={grpo_k} that's {grpo_b*grpo_k} jobs per "
        f"step -> ceil({grpo_b*grpo_k}/{chosen.n_workers}) = "
        f"{math.ceil(grpo_b*grpo_k/chosen.n_workers)} serial waves per step. "
        f"For the timeout, {reason_latency}."
    )
    return chosen.n_workers, float(timeout_s), reason


def _render_markdown(
    *,
    trials: list[TrialResult],
    timeout_trial: TrialResult,
    hist: HistoricalStats | None,
    sleep_min: float,
    sleep_max: float,
    rec_n: int,
    rec_t: float,
    rec_reason: str,
    grpo_b: int,
    grpo_k: int,
    n_prewarmed_keys: int,
) -> str:
    lines: list[str] = []
    lines.append("# SimRewardWorker profile")
    lines.append("")
    lines.append(
        "_Generated by `scripts/training/profile_sim_worker.py`. The worker is "
        "pointed at a fake rollout subprocess that sleeps + emits JSON "
        "matching `rollout.py`'s contract -- no LIBERO/GR00T involvement, "
        "pure-CPU._"
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Jobs per trial: **{trials[0].n_jobs}** "
                 f"({N_UNIQUE_KEYS} unique keys x {ASSERT_DUP_FACTOR} copies)")
    lines.append(f"- Cache prewarmed with **{n_prewarmed_keys} of {N_UNIQUE_KEYS}** unique keys "
                 f"per trial -> expected cache hit rate = "
                 f"{n_prewarmed_keys * ASSERT_DUP_FACTOR}/{trials[0].n_jobs} = "
                 f"{n_prewarmed_keys * ASSERT_DUP_FACTOR / trials[0].n_jobs * 100:.0f}%")
    lines.append(f"- Fake-rollout sleep: uniform({sleep_min:.2f}s, {sleep_max:.2f}s), "
                 f"seeded per job so duration is deterministic")
    lines.append("")
    lines.append("## Parallelism & cache scaling")
    lines.append("")
    lines.append("| n_workers | wall (s) | throughput (jobs/s) | dispatched | cache hits "
                 "| cache hit rate | errors | error rate | dispatched p50 (s) | dispatched p95 (s) | speedup vs n=1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    base = next((t for t in trials if t.n_workers == 1), trials[0])
    for t in trials:
        lines.append(
            "| {nw} | {wall:.2f} | {tp:.2f} | {disp} | {hits} | {hr:.0%} | {err} | {er:.0%} "
            "| {p50:.2f} | {p95:.2f} | {sp:.2f}x |".format(
                nw=t.n_workers,
                wall=t.wall_s,
                tp=t.throughput,
                disp=t.n_dispatched,
                hits=t.n_cache_hits,
                hr=t.hit_rate,
                err=t.n_errors,
                er=t.error_rate,
                p50=t.p50_dispatched_elapsed_s,
                p95=t.p95_dispatched_elapsed_s,
                sp=base.wall_s / t.wall_s if t.wall_s > 0 else float("inf"),
            )
        )
    lines.append("")
    lines.append("## Timeout behavior")
    lines.append("")
    lines.append(
        f"With every job sleeping much longer than `timeout_s`, "
        f"the worker pool should catch all of them as errors and still return "
        f"a clean result list of length {timeout_trial.n_jobs}."
    )
    lines.append("")
    lines.append("| n_workers | n_jobs | timeout (s) | wall (s) | errors | error rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        "| {nw} | {n} | {to:.2f} | {wall:.2f} | {err} | {er:.0%} |".format(
            nw=timeout_trial.n_workers,
            n=timeout_trial.n_jobs,
            to=1.5,  # we hard-coded this in _run_timeout_demo
            wall=timeout_trial.wall_s,
            err=timeout_trial.n_errors,
            er=timeout_trial.error_rate,
        )
    )
    lines.append("")
    lines.append("## Historical real-rollout latency")
    lines.append("")
    if hist is None:
        lines.append(
            f"`{EXISTING_CACHE.relative_to(REPO_ROOT)}` does not exist yet -- no "
            "historical p50/p95 to report. Production timeout below is set "
            "conservatively until a real GRPO run lands."
        )
    else:
        lines.append(
            f"Loaded **{hist.n_entries}** cached rollouts from "
            f"`{EXISTING_CACHE.relative_to(REPO_ROOT)}` "
            f"({hist.n_errors} of them were cached errors)."
        )
        lines.append("")
        lines.append("| metric | seconds |")
        lines.append("|---|---:|")
        lines.append(f"| mean | {hist.mean_elapsed_s:.2f} |")
        lines.append(f"| p50  | {hist.p50_elapsed_s:.2f} |")
        lines.append(f"| p95  | {hist.p95_elapsed_s:.2f} |")
        lines.append(f"| p99  | {hist.p99_elapsed_s:.2f} |")
    lines.append("")
    lines.append("## Recommendation for GRPO production")
    lines.append("")
    lines.append(
        f"**For GRPO with B={grpo_b}, K={grpo_k} and `sim_reward_weight>0`, use "
        f"`--sim-n-workers {rec_n} --sim-timeout-s {rec_t:.0f}`.** {rec_reason}"
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8],
                   help="Worker counts to sweep over (default: 1 2 4 8).")
    p.add_argument("--sleep-min", type=float, default=0.5,
                   help="Fake-rollout sleep lower bound (s). Default 0.5.")
    p.add_argument("--sleep-max", type=float, default=2.5,
                   help="Fake-rollout sleep upper bound (s). Default 2.5.")
    p.add_argument("--timeout-s", type=float, default=30.0,
                   help="Worker timeout for the parallelism trials (s).")
    p.add_argument("--grpo-b", type=int, default=4,
                   help="Expected GRPO micro-batch B for the recommendation.")
    p.add_argument("--grpo-k", type=int, default=4,
                   help="Expected GRPO rollouts-per-prompt K for the recommendation.")
    p.add_argument("--scratch-dir", type=Path,
                   default=Path("/tmp/nla_sim_worker_profile"))
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv)
    if not FAKE_ROLLOUT.exists():
        sys.stderr.write(f"missing fake rollout helper: {FAKE_ROLLOUT}\n")
        return 2
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"[profile_sim_worker] fake rollout: {FAKE_ROLLOUT}")
    print(f"[profile_sim_worker] sleep range:  [{args.sleep_min:.2f}, {args.sleep_max:.2f}] s")
    print(f"[profile_sim_worker] worker sweep: {args.workers}")
    print(f"[profile_sim_worker] timeout_s:    {args.timeout_s}")
    print(f"[profile_sim_worker] scratch:      {args.scratch_dir}")

    jobs = _make_jobs(N_JOBS_TOTAL, N_UNIQUE_KEYS)
    prewarm_jobs = jobs[:N_PREWARMED]  # first N_PREWARMED unique keys

    trials: list[TrialResult] = []
    for nw in args.workers:
        label = f"n{nw}"
        print(f"[profile_sim_worker] running trial n_workers={nw} ...")
        t = _run_trial(
            n_workers=nw,
            jobs=jobs,
            prewarm_jobs=prewarm_jobs,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            timeout_s=args.timeout_s,
            scratch_root=args.scratch_dir,
            fake_rollout=FAKE_ROLLOUT,
            trial_label=label,
        )
        trials.append(t)
        print(f"[profile_sim_worker]   wall={t.wall_s:.2f}s  "
              f"throughput={t.throughput:.2f} jobs/s  "
              f"hit_rate={t.hit_rate:.0%}  "
              f"errors={t.n_errors}/{t.n_jobs}")

    print("[profile_sim_worker] running timeout demo (sleep=5s, timeout=1.5s) ...")
    timeout_trial = _run_timeout_demo(
        scratch_root=args.scratch_dir, fake_rollout=FAKE_ROLLOUT,
    )
    print(f"[profile_sim_worker]   wall={timeout_trial.wall_s:.2f}s  "
          f"errors={timeout_trial.n_errors}/{timeout_trial.n_jobs}")

    hist = _load_historical(EXISTING_CACHE)
    rec_n, rec_t, rec_reason = _recommend(
        trials, hist, grpo_b=args.grpo_b, grpo_k=args.grpo_k,
    )

    md = _render_markdown(
        trials=trials,
        timeout_trial=timeout_trial,
        hist=hist,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        rec_n=rec_n,
        rec_t=rec_t,
        rec_reason=rec_reason,
        grpo_b=args.grpo_b,
        grpo_k=args.grpo_k,
        n_prewarmed_keys=N_PREWARMED,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(md)
    print(f"[profile_sim_worker] wrote report -> {args.report_path}")
    print("---- BEGIN MARKDOWN ----")
    print(md)
    print("----- END MARKDOWN -----")
    print(f"[profile_sim_worker] RECOMMENDATION: "
          f"--sim-n-workers {rec_n} --sim-timeout-s {int(rec_t)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

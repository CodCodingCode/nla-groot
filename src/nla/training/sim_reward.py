"""Sim-success reward worker used by GRPO.

For each AV rollout text we want to score, we:

  1. Encode the text with the (frozen) AR -> a backbone steer vector ``hhat``.
  2. Dispatch a LIBERO rollout subprocess that:
       - Connects to a long-running NlaSteerGr00tPolicy server.
       - Sends ``options['steer_h'] = hhat`` on every ``get_action``.
       - Runs for at most ``sim_max_steps`` simulator steps, breaking early
         when the target task's predicate fires.
       - Prints a JSON summary including ``r_sim`` (the combined predicate +
         dense-shaping score from :mod:`nla.eval.steerability.predicates`).
  3. Read back ``r_sim`` and assemble a Tensor[B*K] of rewards.

The worker pool is thread-based (each thread shells out to one
``rollout.py`` subprocess that handles its own LIBERO + GR00T-client work).
The trainer process does NOT import LIBERO; only the subprocess does.

A JSONL cache keyed by
``sha1(env_name | target_task | source_id | text | seed | sim_max_steps)``
lets repeated rollouts (same activation + text within a few steps) skip
the simulator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------


def sim_cache_key(
    env_name: str,
    target_task: str,
    source_id: str,
    text: str,
    seed: int,
    sim_max_steps: int,
) -> str:
    h = hashlib.sha1()
    h.update(env_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(target_task.encode("utf-8"))
    h.update(b"\x00")
    h.update(source_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(int(seed)).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(int(sim_max_steps)).encode("utf-8"))
    return h.hexdigest()


def load_sim_cache(path: str | Path | None) -> dict[str, dict]:
    """Load an append-only JSONL of past sim rewards into a dict."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("key")
            if k:
                out[k] = obj
    return out


# ----------------------------------------------------------------------------
# Job + worker
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SimRewardJob:
    """One unit of sim-side work."""

    env_name: str
    target_task: str
    source_id: str
    text: str
    seed: int
    steer_h: np.ndarray
    sim_max_steps: int
    placement: str
    blend: float


@dataclass(frozen=True)
class SimRewardResult:
    key: str
    r_sim: float
    predicate: float
    r_dist: float
    r_displace: float
    r_contact: float
    n_steps: int
    early_stopped: bool
    elapsed_s: float
    cached: bool
    success_any: bool
    error: str | None = None


def _run_rollout_subprocess(
    job: SimRewardJob,
    *,
    rollout_python: str,
    rollout_script: str,
    policy_host: str,
    policy_port: int,
    workdir: Path,
    env_overrides: dict[str, str] | None,
    timeout_s: float,
) -> dict:
    """Shell out to a single ``rollout.py`` subprocess.

    Returns the parsed JSON summary. Raises on subprocess crash.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    steer_path = workdir / "steer_h.npy"
    np.save(steer_path, job.steer_h.astype(np.float32, copy=False))

    cmd = [
        rollout_python, rollout_script,
        "--env-name", job.env_name,
        "--seed", str(int(job.seed)),
        "--policy-host", policy_host,
        "--policy-port", str(int(policy_port)),
        "--target-task", job.target_task,
        "--steer-h-path", str(steer_path),
        "--steer-placement", job.placement,
        "--steer-blend", f"{job.blend:.3f}",
        "--max-episode-steps", str(int(job.sim_max_steps)),
        "--no-frames",
        "--early-stop-on-success",
    ]
    env = os.environ.copy()
    # Default to CPU rendering; LIBERO+osmesa works without a GPU display
    # and matches the steerability harness defaults.
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if env_overrides:
        env.update(env_overrides)

    completed = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"rollout subprocess failed (rc={completed.returncode}); "
            f"stderr tail: {completed.stderr[-500:]!r}"
        )
    # The CLI prints a JSON object on the last lines of stdout. Be tolerant
    # of leading log lines.
    text = completed.stdout.strip()
    if not text:
        raise RuntimeError(
            f"rollout subprocess produced empty stdout; stderr: {completed.stderr[-500:]!r}"
        )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find the first '{' and try again.
        idx = text.find("{")
        if idx < 0:
            raise
        return json.loads(text[idx:])


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


class SimRewardWorker:
    """Pool of rollout subprocesses scoring sim-success reward in parallel.

    Designed to live inside the GRPO trainer process. ``compute`` blocks
    until every job either returns a reward or hits a timeout / error
    (errored rollouts get ``r_sim = 0.0`` so a single sim crash does not
    poison a whole gradient step).
    """

    def __init__(
        self,
        *,
        policy_host: str = "localhost",
        policy_port: int = 5555,
        n_workers: int = 4,
        sim_max_steps: int = 100,
        placement: str = "image_patch",
        blend: float = 1.0,
        rollout_python: str | None = None,
        rollout_script: str | None = None,
        cache_path: str | Path | None = None,
        timeout_s: float = 240.0,
        env_overrides: dict[str, str] | None = None,
        scratch_dir: str | Path | None = None,
    ) -> None:
        self.policy_host = policy_host
        self.policy_port = int(policy_port)
        self.n_workers = max(1, int(n_workers))
        self.sim_max_steps = int(sim_max_steps)
        self.placement = str(placement)
        self.blend = float(blend)
        self.timeout_s = float(timeout_s)
        self.env_overrides = dict(env_overrides) if env_overrides else None

        # Default to the same interpreter that's running us. Production GRPO
        # runs will typically point this at the LIBERO venv's python.
        self.rollout_python = rollout_python or os.environ.get(
            "NLA_ROLLOUT_PYTHON", "python"
        )
        # Default to the in-tree rollout.py.
        if rollout_script is None:
            here = Path(__file__).resolve().parents[2] / "nla" / "eval" / "steerability" / "rollout.py"
            self.rollout_script = str(here)
        else:
            self.rollout_script = str(rollout_script)

        # Scratch dir for per-call steer_h .npy files. Default: TMP/sim_reward.
        sd = scratch_dir or os.environ.get("NLA_SIM_REWARD_SCRATCH")
        self._scratch = Path(sd) if sd else Path(tempfile.gettempdir()) / "nla_sim_reward"
        self._scratch.mkdir(parents=True, exist_ok=True)

        # Cache.
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = load_sim_cache(self._cache_path)
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "SimRewardWorker(host=%s:%d, n_workers=%d, sim_max_steps=%d, "
            "placement=%s, blend=%.2f, python=%s, cache=%s, cached_entries=%d)",
            self.policy_host, self.policy_port, self.n_workers,
            self.sim_max_steps, self.placement, self.blend,
            self.rollout_python, self._cache_path, len(self._cache),
        )

    # ------------------------------------------------------------------

    def _append_cache(self, entry: dict) -> None:
        self._cache[entry["key"]] = entry
        if self._cache_path is None:
            return
        try:
            with self._cache_path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            logger.warning("Failed appending to sim cache %s: %s", self._cache_path, e)

    # ------------------------------------------------------------------

    def compute(
        self,
        jobs: Sequence[SimRewardJob],
    ) -> list[SimRewardResult]:
        """Score every job (cached entries return instantly).

        Output is parallel to ``jobs``. Failed jobs return ``r_sim=0`` with
        an ``error`` string set so the caller can log/skip them.
        """
        import time
        results: list[SimRewardResult | None] = [None] * len(jobs)

        # Resolve cache hits up front.
        pending_idx: list[int] = []
        for i, job in enumerate(jobs):
            key = sim_cache_key(
                job.env_name, job.target_task, job.source_id,
                job.text, job.seed, job.sim_max_steps,
            )
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = SimRewardResult(
                    key=key,
                    r_sim=float(cached.get("r_sim", 0.0)),
                    predicate=float(cached.get("predicate", 0.0)),
                    r_dist=float(cached.get("r_dist", 0.0)),
                    r_displace=float(cached.get("r_displace", 0.0)),
                    r_contact=float(cached.get("r_contact", 0.0)),
                    n_steps=int(cached.get("n_steps", 0)),
                    early_stopped=bool(cached.get("early_stopped", False)),
                    elapsed_s=float(cached.get("elapsed_s", 0.0)),
                    cached=True,
                    success_any=bool(cached.get("success_any", False)),
                    error=cached.get("error"),
                )
            else:
                pending_idx.append(i)

        if not pending_idx:
            return [r for r in results if r is not None]  # all cached

        # Dispatch pending jobs in a thread pool.
        def _one(i: int) -> tuple[int, SimRewardResult]:
            job = jobs[i]
            key = sim_cache_key(
                job.env_name, job.target_task, job.source_id,
                job.text, job.seed, job.sim_max_steps,
            )
            workdir = self._scratch / key
            t0 = time.time()
            try:
                summary = _run_rollout_subprocess(
                    job,
                    rollout_python=self.rollout_python,
                    rollout_script=self.rollout_script,
                    policy_host=self.policy_host,
                    policy_port=self.policy_port,
                    workdir=workdir,
                    env_overrides=self.env_overrides,
                    timeout_s=self.timeout_s,
                )
            except Exception as e:
                err = repr(e)
                logger.warning("sim job failed (key=%s): %s", key[:12], err)
                res = SimRewardResult(
                    key=key, r_sim=0.0, predicate=0.0, r_dist=0.0,
                    r_displace=0.0, r_contact=0.0, n_steps=0,
                    early_stopped=False, elapsed_s=time.time() - t0,
                    cached=False, success_any=False, error=err,
                )
                # Don't cache errors -- a retry next epoch should re-attempt.
                return i, res

            elapsed = time.time() - t0
            breakdown = summary.get("sim_score_breakdown") or {}
            res = SimRewardResult(
                key=key,
                r_sim=float(summary.get("r_sim") or 0.0),
                predicate=float(breakdown.get("predicate", 0.0)),
                r_dist=float(breakdown.get("r_dist", 0.0)),
                r_displace=float(breakdown.get("r_displace", 0.0)),
                r_contact=float(breakdown.get("r_contact", 0.0)),
                n_steps=int(summary.get("n_steps", 0)),
                early_stopped=bool(summary.get("early_stopped", False)),
                elapsed_s=elapsed,
                cached=False,
                success_any=bool(summary.get("success_any", False)),
                error=None,
            )
            entry = {
                "key":           res.key,
                "r_sim":         res.r_sim,
                "predicate":     res.predicate,
                "r_dist":        res.r_dist,
                "r_displace":    res.r_displace,
                "r_contact":     res.r_contact,
                "n_steps":       res.n_steps,
                "early_stopped": res.early_stopped,
                "elapsed_s":     res.elapsed_s,
                "success_any":   res.success_any,
                "env_name":      job.env_name,
                "target_task":   job.target_task,
                "source_id":     job.source_id,
                "text":          job.text,
                "seed":          job.seed,
                "sim_max_steps": job.sim_max_steps,
            }
            self._append_cache(entry)
            return i, res

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = [pool.submit(_one, i) for i in pending_idx]
            for fut in as_completed(futures):
                i, res = fut.result()
                results[i] = res

        # All slots now filled.
        return [r for r in results if r is not None]


# ----------------------------------------------------------------------------
# Tensor helper used by the GRPO step
# ----------------------------------------------------------------------------


def encode_texts_with_ar(
    ar,                                    # nla.models.ar.ActivationReconstructor
    rollout_texts: Sequence[str],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Run the AR over a batch of texts; return ``(N, H_backbone)`` float32 CPU.

    Mirrors ``nla.steering.ar_text_to_backbone_vec`` but for a batch and
    without the BACKBONE_EMBEDDING_DIM sanity check (so we can unit-test
    with toy ARs whose output dim differs).
    """
    ar.eval()
    with torch.no_grad():
        # AR returns alpha-scaled vectors; multiply by alpha to recover the
        # unscaled backbone-space vector the steer hook expects.
        pred_scaled = ar(list(rollout_texts), device=device)
        out = pred_scaled.float() * float(ar.cfg.alpha)
    return out.detach().cpu().contiguous()


def assemble_jobs(
    *,
    rollout_texts: Sequence[str],
    steer_vecs: torch.Tensor,
    target_tasks: Sequence[str],
    target_env_names: Sequence[str],
    source_ids: Sequence[str],
    seeds: Iterable[int],
    sim_max_steps: int,
    placement: str,
    blend: float,
) -> list[SimRewardJob]:
    """Zip the per-row inputs into a list of :class:`SimRewardJob`s."""
    steer_np = steer_vecs.float().cpu().contiguous().numpy()
    seeds = list(seeds)
    n = len(rollout_texts)
    assert steer_np.shape[0] == n == len(target_tasks) == len(target_env_names) == len(source_ids) == len(seeds), (
        f"length mismatch: texts={n} steer={steer_np.shape[0]} tasks={len(target_tasks)} "
        f"envs={len(target_env_names)} ids={len(source_ids)} seeds={len(seeds)}"
    )
    jobs: list[SimRewardJob] = []
    for i in range(n):
        jobs.append(SimRewardJob(
            env_name=target_env_names[i],
            target_task=target_tasks[i],
            source_id=source_ids[i],
            text=rollout_texts[i],
            seed=int(seeds[i]),
            steer_h=steer_np[i].copy(),
            sim_max_steps=int(sim_max_steps),
            placement=placement,
            blend=blend,
        ))
    return jobs

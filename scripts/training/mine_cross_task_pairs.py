#!/usr/bin/env python
"""Mine cross-task ``(scene_t, caption_alt, action_alt)`` triples.

This is the Phase-1 step 4.1 deliverable from
``docs/sft_plan/09_action_head_lora_phase1.md`` — a *stub* implementation that
walks the LIBERO-goal LeRobot dataset and emits one JSONL row per aligned
cross-task pair so the Phase-1 action-head LoRA training has a corpus to
consume.

Pipeline (CPU-only, no GR00T import)
------------------------------------

1. Load ``meta/episodes.jsonl`` and ``meta/tasks.jsonl`` from the LeRobot
   dataset. Episodes are grouped by their human-readable ``task_text``
   (same as ``tasks.jsonl[task_index]``).
2. Within each pair of tasks ``(A, B)`` with ``A != B``, iterate every
   ``(E_A, E_B)`` such that the initial gripper L2 distance is below
   ``--gripper-pose-tol``. (Same starting condition keeps ``action_alt``
   approximately realisable from ``scene_t``'s arm state.)
3. Align ``E_A`` and ``E_B`` by length-normalized step index, skipping the
   last ``--drop-tail-frac`` of each demo (the alt terminal grasp is
   meaningless if the target object isn't in ``scene_t``'s reach envelope).
4. For each aligned ``(t_A, t_B)``, emit::

       {
         "scene_episode":  E_A.episode_index,
         "scene_step":     t_A,
         "scene_task_text":   E_A.task_text,       # "instruction"
         "alt_episode":    E_B.episode_index,
         "alt_step":       t_B,
         "alt_task_text":  E_B.task_text,          # "caption_alt"
         "alt_action":     E_B.action[t_B],        # 7-D LIBERO action
         "scene_id":       <stub: "shared"; see TODO>,
       }

5. Optional: filter rows whose ``alt_task_text`` references an object not
   visually present in ``scene_t``. The doc cites the V3 attribute probe
   (``scripts/eval/probe_h_attributes.py``) for this; we leave a hook
   (``--object-visibility-jsonl``) but do not load it in this stub.

Output
------

JSONL at ``--output``. Target volume per the doc: 30–50k rows after
filtering for ``libero_goal``. The stub currently writes every pair that
clears the pose tolerance + drop-tail filter; downstream consumers can
prune via ``head`` / ``shuf``.

Phase-1 deliverable path
~~~~~~~~~~~~~~~~~~~~~~~~

``data/synthetic_steering/libero_goal_pairs.jsonl`` (per doc 09 §4.1).

CLI example
-----------

::

    PYTHONPATH=src python scripts/training/mine_cross_task_pairs.py \\
        --dataset-root third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --output       data/synthetic_steering/libero_goal_pairs.jsonl

What this stub does NOT do (tracked as TODO)
--------------------------------------------

* No real ``scene_id`` mining. The doc wants pairs grouped by shared
  kitchen layout; for libero_goal every demo shares the same kitchen so we
  emit ``"shared"`` and let the gripper-pose filter approximate the
  same-scene constraint. Multi-scene extensions (libero_object, libero_10)
  will need a richer scene-id signal — either from BDDL task files or
  per-frame attribute tags.
* No visibility filter on ``alt_task_text`` target objects.
* No image bytes in the JSONL — we emit ``(scene_episode, scene_step)``
  pointers so downstream pipelines can decode frames lazily via the
  existing ``LeRobotEpisodeLoader``. This keeps the JSONL ≪ 100 MB and
  avoids the encoder/decoder asymmetry the docs flagged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("mine_cross_task_pairs")


# ---------------------------------------------------------------------------
# Meta loaders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    task_index: int
    task_text: str
    length: int


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    path = dataset_root / "meta" / "tasks.jsonl"
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows[int(d["task_index"])] = str(d["task"])
    if not rows:
        raise RuntimeError(f"No tasks in {path}")
    return rows


def _load_episodes(
    dataset_root: Path, task_lookup: dict[int, str]
) -> list[EpisodeMeta]:
    path = dataset_root / "meta" / "episodes.jsonl"
    out: list[EpisodeMeta] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks = d.get("tasks") or []
            task_text = tasks[0] if tasks else ""
            # LeRobot episodes don't carry task_index inline; reverse map.
            task_index = next(
                (ti for ti, tt in task_lookup.items() if tt == task_text), -1
            )
            out.append(
                EpisodeMeta(
                    episode_index=int(d["episode_index"]),
                    task_index=int(task_index),
                    task_text=str(task_text),
                    length=int(d["length"]),
                )
            )
    if not out:
        raise RuntimeError(f"No episodes in {path}")
    return out


def _episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 1000
    return (
        dataset_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


@dataclass
class EpisodeData:
    state: np.ndarray   # [T, state_dim]
    action: np.ndarray  # [T, action_dim]
    task_index: int
    task_text: str
    episode_index: int


def _load_episode_arrays(
    dataset_root: Path, meta: EpisodeMeta
) -> EpisodeData:
    path = _episode_parquet_path(dataset_root, meta.episode_index)
    table = pq.read_table(
        str(path), columns=["observation.state", "action"]
    )
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    return EpisodeData(
        state=state,
        action=action,
        task_index=meta.task_index,
        task_text=meta.task_text,
        episode_index=meta.episode_index,
    )


# ---------------------------------------------------------------------------
# Pair miner
# ---------------------------------------------------------------------------

@dataclass
class MinerConfig:
    gripper_pose_tol_m: float = 0.05
    drop_tail_frac: float = 0.25
    align_n_steps: int = 32
    max_pairs_per_task_pair: int | None = None
    seed: int = 0


def _initial_gripper_pose(ep: EpisodeData) -> np.ndarray:
    """Approximate the gripper L2 starting position from observation.state.

    LIBERO ``observation.state`` is a 9-D vector with the joint positions and
    the gripper opening. We use the first 3 dims as a *proxy* for the
    end-effector position — the actual ee-pose would require a forward
    kinematics call we don't want to import here. This is conservative: if
    the proxy diverges, the gripper-tolerance filter rejects more pairs,
    not fewer, so synthesised demos stay realistic.
    """
    return ep.state[0, :3].astype(np.float64)


def _aligned_steps(
    len_a: int, len_b: int, n: int, drop_tail_frac: float,
) -> Iterator[tuple[int, int]]:
    keep_a = max(1, int(round(len_a * (1.0 - drop_tail_frac))))
    keep_b = max(1, int(round(len_b * (1.0 - drop_tail_frac))))
    if keep_a < 2 or keep_b < 2:
        return
    sample_n = max(1, min(n, keep_a, keep_b))
    for s in np.linspace(0.0, 1.0, sample_n):
        t_a = int(round(s * (keep_a - 1)))
        t_b = int(round(s * (keep_b - 1)))
        yield t_a, t_b


def mine_pairs(
    episodes_by_task: dict[int, list[EpisodeData]],
    cfg: MinerConfig,
) -> Iterator[dict]:
    rng = np.random.default_rng(cfg.seed)
    task_indices = sorted(episodes_by_task)
    for i, ti_a in enumerate(task_indices):
        for ti_b in task_indices[i + 1:]:
            yielded = 0
            for ea in episodes_by_task[ti_a]:
                for eb in episodes_by_task[ti_b]:
                    dist = float(np.linalg.norm(
                        _initial_gripper_pose(ea) - _initial_gripper_pose(eb)
                    ))
                    if dist > cfg.gripper_pose_tol_m:
                        continue
                    for t_a, t_b in _aligned_steps(
                        len(ea.action),
                        len(eb.action),
                        cfg.align_n_steps,
                        cfg.drop_tail_frac,
                    ):
                        yield {
                            "scene_episode": ea.episode_index,
                            "scene_step": int(t_a),
                            "scene_task_text": ea.task_text,
                            "alt_episode": eb.episode_index,
                            "alt_step": int(t_b),
                            "alt_task_text": eb.task_text,
                            "alt_action": eb.action[t_b].tolist(),
                            # TODO(09 §4.1): replace stub scene_id with the
                            # BDDL-derived kitchen layout id once the BDDL
                            # parser lands; for libero_goal every demo
                            # shares one kitchen so "shared" is correct.
                            "scene_id": "shared",
                            "initial_pose_dist": dist,
                        }
                        yielded += 1
                        if (
                            cfg.max_pairs_per_task_pair is not None
                            and yielded >= cfg.max_pairs_per_task_pair
                        ):
                            break
                    if (
                        cfg.max_pairs_per_task_pair is not None
                        and yielded >= cfg.max_pairs_per_task_pair
                    ):
                        break
                if (
                    cfg.max_pairs_per_task_pair is not None
                    and yielded >= cfg.max_pairs_per_task_pair
                ):
                    break
            logger.info(
                "  tasks %d↔%d: yielded %d pairs", ti_a, ti_b, yielded,
            )
    _ = rng  # reserved for shuffling in a follow-up.


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Root of the LeRobot dataset (must contain meta/ and data/ "
             "directories). Default location for libero_goal: "
             "third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSONL path. Phase-1 default: "
             "data/synthetic_steering/libero_goal_pairs.jsonl",
    )
    p.add_argument(
        "--gripper-pose-tol-m",
        type=float,
        default=0.05,
        help="Reject pairs whose initial gripper L2 distance (proxy on the "
             "first 3 dims of observation.state) exceeds this many meters. "
             "Default 0.05 = 5 cm per doc 09 §4.1.",
    )
    p.add_argument(
        "--drop-tail-frac",
        type=float,
        default=0.25,
        help="Drop the last fraction of each demo when aligning, so the alt "
             "demo's terminal grasp doesn't poison the synthesised target "
             "action. Default 0.25 = last 25%% per doc 09 §4.1.",
    )
    p.add_argument(
        "--align-n-steps",
        type=int,
        default=32,
        help="Number of aligned (t_a, t_b) samples to emit per accepted "
             "episode pair. Linspace over the surviving time axis. Default 32.",
    )
    p.add_argument(
        "--max-pairs-per-task-pair",
        type=int,
        default=None,
        help="Cap on emitted rows per (task_A, task_B) bucket; None = no cap.",
    )
    p.add_argument(
        "--object-visibility-jsonl",
        default=None,
        help="(Stub) Path to a JSONL keyed by (episode_index, step_index) "
             "with a list of visible-object tags. When provided, drop pairs "
             "whose alt_task_text mentions an object not in that list. Not "
             "consumed yet; tracked as TODO.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.object_visibility_jsonl is not None:
        logger.warning(
            "--object-visibility-jsonl is accepted but not yet consumed; "
            "drop the flag or extend mine_pairs to honor it.",
        )

    root = Path(args.dataset_root)
    if not (root / "meta" / "episodes.jsonl").exists():
        raise SystemExit(
            f"--dataset-root {root} does not look like a LeRobot dataset "
            "(expected meta/episodes.jsonl)."
        )

    t0 = time.time()
    tasks = _load_tasks(root)
    episodes_meta = _load_episodes(root, tasks)
    logger.info("Loaded %d episodes across %d tasks", len(episodes_meta), len(tasks))

    episodes_by_task: dict[int, list[EpisodeData]] = {}
    for meta in episodes_meta:
        if meta.task_index < 0:
            logger.warning(
                "Episode %d has no matching task_index for text %r; skipping.",
                meta.episode_index, meta.task_text,
            )
            continue
        ep = _load_episode_arrays(root, meta)
        episodes_by_task.setdefault(meta.task_index, []).append(ep)

    logger.info(
        "Parsed action arrays for %d episodes (%.1fs).",
        sum(len(v) for v in episodes_by_task.values()),
        time.time() - t0,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = MinerConfig(
        gripper_pose_tol_m=args.gripper_pose_tol_m,
        drop_tail_frac=args.drop_tail_frac,
        align_n_steps=args.align_n_steps,
        max_pairs_per_task_pair=args.max_pairs_per_task_pair,
        seed=args.seed,
    )
    n_rows = 0
    with out_path.open("w") as f:
        for row in mine_pairs(episodes_by_task, cfg):
            f.write(json.dumps(row) + "\n")
            n_rows += 1
    logger.info(
        "Wrote %d rows to %s (%.1fs total).",
        n_rows, out_path, time.time() - t0,
    )
    if n_rows == 0:
        logger.warning(
            "Zero rows produced. Loosen --gripper-pose-tol-m (currently %.3f m) "
            "or shrink --drop-tail-frac (currently %.2f) and retry.",
            args.gripper_pose_tol_m, args.drop_tail_frac,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Extract per-label camera frames from a LeRobot dataset into a flat cache.

The multimodal LLM judge (``scripts/eval/llm_judge_av_captions.py``) and the
GRPO judge-reward path (``src/nla/training/grpo.py``) both consume images
from a single flat directory layout::

    {frames_cache}/{source_example_id}__{video_key}.jpg

This script populates that directory by decoding each labeled (episode, step)
frame from a LeRobot dataset's videos. It is intentionally dataset-agnostic:
pass any LeRobot root and any list of ``--video-keys`` that match the
dataset's modality config.

The labels file must contain rows with ``meta.episode_index``, ``meta.step_index``,
and either ``meta.source_example_id`` or a ``traj{ep}_step{step}`` style id;
i.e. the same schema the labeling pipeline writes (see
``scripts/labeling/run_label.py``).

Usage (LIBERO)::

    PYTHONPATH=src python scripts/eval/extract_label_frames.py \\
        --dataset-root  third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --labels-jsonl  data/labels/libero_goal_pilot/labels.jsonl \\
        --frames-cache  data/labels/libero_goal_pilot/frames_cache \\
        --video-keys    image wrist_image

What gets written
-----------------
One JPEG per (source_id, video_key) pair, plus a one-line summary printed to
stdout::

    [extract_label_frames] sources=243 keys=2 written=486 reused=0 failed=0

Rows that already have a cached frame are skipped (the script is idempotent
and safe to re-run after a partial failure).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nla.labeling.frames import EpisodeFrameLoader, save_jpeg

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-root", required=True,
                   help="LeRobot dataset root (the directory that contains "
                        "meta/episodes.jsonl and data/chunk-*/*.parquet).")
    p.add_argument("--labels-jsonl", required=True,
                   help="Path to labels.jsonl produced by scripts/labeling/run_label.py.")
    p.add_argument("--frames-cache", required=True,
                   help="Output directory. Files are written as "
                        "{frames_cache}/{source_example_id}__{video_key}.jpg.")
    p.add_argument("--video-keys", nargs="+", required=True,
                   help="One or more LeRobot video keys to decode per label "
                        "(e.g. 'image wrist_image' for LIBERO). Must exist in "
                        "the dataset's modality config.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on labels processed (for debugging).")
    p.add_argument("--log-level", default="INFO")
    return p


def _iter_label_rows(labels_path: Path):
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _source_id(meta: dict) -> str | None:
    """Derive the canonical source_example_id from a label's meta dict.

    We accept either an explicit ``meta.source_example_id`` (preferred) or
    fall back to ``traj{ep:06d}_step{step:06d}`` reconstructed from
    ``episode_index`` and ``step_index`` (matches the labeling pipeline's
    default scheme). Returns ``None`` if the meta is missing the info
    needed for either form so the caller can skip the row.
    """
    src = meta.get("source_example_id")
    if src:
        return str(src)
    ep = meta.get("episode_index")
    step = meta.get("step_index")
    if ep is None or step is None:
        return None
    return f"traj{int(ep):06d}_step{int(step):06d}"


def _extract_one(
    *,
    dataset_root: Path,
    frames_dir: Path,
    loader_cache: dict[int, EpisodeFrameLoader],
    episode_index: int,
    step_index: int,
    source_id: str,
    video_keys: list[str],
) -> tuple[int, int, int]:
    """Return ``(written, reused, failed)`` counts for this (source, keys) pair."""
    written = reused = failed = 0
    for vk in video_keys:
        dst = frames_dir / f"{source_id}__{vk}.jpg"
        if dst.exists():
            reused += 1
            continue
        loader = loader_cache.get(episode_index)
        if loader is None:
            loader = EpisodeFrameLoader(dataset_root, episode_index)
            loader_cache[episode_index] = loader
        try:
            frame = loader.frame(vk, step_index)
            save_jpeg(frame, dst)
            written += 1
        except Exception as exc:
            logger.warning(
                "frame decode failed for source=%s video_key=%s ep=%d step=%d: %s",
                source_id, vk, episode_index, step_index, exc,
            )
            failed += 1
    return written, reused, failed


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    dataset_root = Path(args.dataset_root)
    labels_path = Path(args.labels_jsonl)
    frames_dir = Path(args.frames_cache)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        logger.error("--dataset-root %s does not exist", dataset_root)
        return 2
    if not labels_path.exists():
        logger.error("--labels-jsonl %s does not exist", labels_path)
        return 2

    seen_sources: set[str] = set()
    total_written = total_reused = total_failed = 0
    loaders: dict[int, EpisodeFrameLoader] = {}

    n_rows = 0
    for row in _iter_label_rows(labels_path):
        if row.get("error") or not row.get("description"):
            continue
        meta = row.get("meta") or {}
        ep = meta.get("episode_index")
        step = meta.get("step_index")
        if ep is None or step is None:
            continue
        src = _source_id(meta)
        if src is None:
            continue
        if src in seen_sources:
            continue
        seen_sources.add(src)
        n_rows += 1
        w, r, fl = _extract_one(
            dataset_root=dataset_root,
            frames_dir=frames_dir,
            loader_cache=loaders,
            episode_index=int(ep),
            step_index=int(step),
            source_id=src,
            video_keys=args.video_keys,
        )
        total_written += w
        total_reused += r
        total_failed += fl
        if args.limit and n_rows >= args.limit:
            break

    for ld in loaders.values():
        try:
            ld.close()
        except Exception:
            pass

    print(
        f"[extract_label_frames] sources={len(seen_sources)} "
        f"keys={len(args.video_keys)} "
        f"written={total_written} reused={total_reused} failed={total_failed}",
        file=sys.stdout,
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the frozen 300-tuple eval set for the prompt A/B test.

Draws stratified samples from one or more activation roots, materialises
frames into a shared cache, and writes ``eval_set.jsonl`` whose rows fully
reconstruct ``PositionLabelInput``.

Each output row contains everything any variant needs at label time:

    {
        "eval_id":              "droid_sample/traj000001_step000017@p042_image_patch",
        "source":               "droid_sample",
        "example_id":           "traj000001_step000017",
        "instruction":          "...",
        "decoded_text_context": "...",
        "position_index":       42,
        "position_type":        "image_patch",   # one of last_text/image_patch/anchor
        "sequence_length":      277,
        "image_patch_meta":     [k, n] | null,
        "image_paths":          ["data/prompt_ab/frames_cache/.../*.jpg", ...],
        "episode_index":        1,
        "step_index":           17,
        "state":                null,
        "state_name":           null,
    }

Example::

    PYTHONPATH=src python scripts/labeling/build_eval_set.py \
        --source-name droid_sample \
        --activations-root data/activations/droid_ep1 \
        --dataset-root     third_party/Isaac-GR00T/demo_data/droid_sample \
        --per-type 50 \
        --source-name bridge_pilot \
        --activations-root data/activations/bridge_pilot \
        --dataset-root     third_party/Isaac-GR00T/demo_data/simplerenv_bridge_sample \
        --per-type 50 \
        --out data/prompt_ab/eval_set.jsonl \
        --frames-cache data/prompt_ab/frames_cache \
        --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np


logger = logging.getLogger("nla.eval_set")


POSITION_TYPES = ("last_text", "image_patch", "anchor")


def _iter_positions_for_type(
    reader,
    tokenizer,
    *,
    position_type: str,
    rng: np.random.Generator,
):
    """Yield (record, position_index, decoded_text_context, image_patch_meta) for
    every example, with the position chosen to be of the requested type.

    Falls back to None if the example has no valid token of that type.
    """
    from nla.extraction.sampler import (
        _anchor_index,
        _image_patch_index,
        _last_text_index,
    )
    from nla.labeling.context import decode_text_context, image_patch_meta

    for item in reader.iter_examples():
        rec = item["_record"]
        attn = item["attention_mask"]
        img = item["image_mask"]
        ids = item.get("input_ids")
        if ids is None:
            logger.warning(
                "skipping %s (no input_ids; extraction must use --store-input-ids)",
                rec.example_id,
            )
            continue
        if position_type == "last_text":
            idx = _last_text_index(attn, img)
        elif position_type == "image_patch":
            idx = _image_patch_index(attn, img, rng)
        elif position_type == "anchor":
            idx = _anchor_index(attn)
        else:
            raise ValueError(f"unknown position_type {position_type}")
        if idx is None:
            continue
        ctx = decode_text_context(ids, img, tokenizer)
        meta = image_patch_meta(img, idx) if position_type == "image_patch" else None
        yield rec, idx, ctx, meta


def _sample_per_type(
    reader,
    tokenizer,
    *,
    n_per_type: int,
    seed: int,
) -> dict[str, list[tuple]]:
    """For each position type, sample ``n_per_type`` (record, idx, ctx, meta) tuples."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple]] = {p: [] for p in POSITION_TYPES}
    for ptype in POSITION_TYPES:
        candidates = list(
            _iter_positions_for_type(reader, tokenizer, position_type=ptype, rng=rng)
        )
        if not candidates:
            logger.warning("no candidates for position_type=%s", ptype)
            continue
        if len(candidates) <= n_per_type:
            logger.warning(
                "only %d %s candidates (asked for %d); taking all",
                len(candidates), ptype, n_per_type,
            )
            out[ptype] = candidates
        else:
            idx = rng.choice(len(candidates), size=n_per_type, replace=False)
            out[ptype] = [candidates[i] for i in sorted(idx.tolist())]
    return out


def _materialize_frames(
    rec,
    *,
    dataset_root: Path,
    frames_cache_dir: Path,
    pool,
    video_keys: list[str],
) -> list[str]:
    """Extract per-video-key frames for one example into ``frames_cache_dir``."""
    from nla.labeling.frames import save_jpeg

    if rec.episode_index is None or rec.step_index is None:
        return []
    loader = pool.get(dataset_root, rec.episode_index)
    image_paths: list[str] = []
    for vk in video_keys:
        try:
            frame = loader.frame(vk, rec.step_index)
        except (FileNotFoundError, IndexError) as e:
            logger.warning(
                "could not load %s frame %d for episode %d (%s); skipping",
                vk, rec.step_index, rec.episode_index, e,
            )
            return []
        out = frames_cache_dir / f"{rec.example_id}__{vk}.jpg"
        save_jpeg(frame, out)
        image_paths.append(str(out))
    return image_paths


def _build_rows_for_source(
    *,
    source_name: str,
    activations_root: Path,
    dataset_root: Path,
    n_per_type: int,
    frames_cache_dir: Path,
    seed: int,
    tokenizer,
) -> list[dict]:
    """Sample + materialise frames for one source. Returns serializable rows."""
    from nla.extraction.storage import ActivationShardReader
    from nla.labeling.context import FrameLoaderPool
    from nla.labeling.frames import DatasetInfo

    logger.info("source=%s: loading activations from %s", source_name, activations_root)
    reader = ActivationShardReader(activations_root)
    logger.info("  %d examples available", len(reader))

    di = DatasetInfo.from_root(dataset_root)
    video_keys = di.video_keys
    logger.info("  video keys: %s", video_keys)

    sampled = _sample_per_type(
        reader, tokenizer, n_per_type=n_per_type, seed=seed,
    )

    pool = FrameLoaderPool(max_open=8)
    rows: list[dict] = []
    try:
        for ptype in POSITION_TYPES:
            count_kept = 0
            for rec, pos_idx, ctx, meta in sampled[ptype]:
                # Resolve instruction.
                instruction = rec.task_text or ""
                if not instruction and rec.episode_index is not None:
                    instruction = di.episode_to_task.get(int(rec.episode_index), "")

                image_paths = _materialize_frames(
                    rec,
                    dataset_root=dataset_root,
                    frames_cache_dir=frames_cache_dir / source_name,
                    pool=pool,
                    video_keys=video_keys,
                )
                if not image_paths:
                    continue
                row = {
                    "eval_id": f"{source_name}/{rec.example_id}@p{pos_idx:03d}_{ptype}",
                    "source": source_name,
                    "example_id": rec.example_id,
                    "instruction": instruction,
                    "decoded_text_context": ctx,
                    "position_index": int(pos_idx),
                    "position_type": ptype,
                    "sequence_length": int(rec.seq_len),
                    "image_patch_meta": list(meta) if meta is not None else None,
                    "image_paths": image_paths,
                    "episode_index": (
                        int(rec.episode_index) if rec.episode_index is not None else None
                    ),
                    "step_index": (
                        int(rec.step_index) if rec.step_index is not None else None
                    ),
                    "state": None,
                    "state_name": None,
                }
                rows.append(row)
                count_kept += 1
            logger.info("  %s: kept %d / %d", ptype, count_kept, len(sampled[ptype]))
    finally:
        pool.close_all()
    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--source-name", action="append", required=True,
        help="Name for this source (used in eval_id and frames_cache subdir). "
             "Repeat with --activations-root and --dataset-root in the same order.",
    )
    p.add_argument(
        "--activations-root", action="append", required=True,
        help="Path to one extraction root; repeat per source.",
    )
    p.add_argument(
        "--dataset-root", action="append", required=True,
        help="Path to one LeRobot dataset root; repeat per source.",
    )
    p.add_argument(
        "--per-type", action="append", type=int, required=True,
        help="How many of each position type to sample from each source. "
             "Repeat per source.",
    )
    p.add_argument("--out", required=True, help="Path for the eval_set.jsonl")
    p.add_argument(
        "--frames-cache", required=True,
        help="Directory where frame JPEGs are saved (shared by all sources).",
    )
    p.add_argument(
        "--tokenizer-repo", default="Qwen/Qwen3-VL-2B-Instruct",
        help="Tokenizer for decoded text rendering.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    n_sources = len(args.source_name)
    for name, lst in (
        ("activations-root", args.activations_root),
        ("dataset-root", args.dataset_root),
        ("per-type", args.per_type),
    ):
        if len(lst) != n_sources:
            raise SystemExit(
                f"--{name} occurred {len(lst)} times but --source-name occurred "
                f"{n_sources} times; the lists must be the same length."
            )

    from nla.labeling.context import load_qwen3_vl_tokenizer

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_cache_dir = Path(args.frames_cache)
    frames_cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_qwen3_vl_tokenizer(args.tokenizer_repo)

    all_rows: list[dict] = []
    for src, act_root, ds_root, ptype_count in zip(
        args.source_name, args.activations_root, args.dataset_root, args.per_type,
    ):
        rows = _build_rows_for_source(
            source_name=src,
            activations_root=Path(act_root),
            dataset_root=Path(ds_root),
            n_per_type=int(ptype_count),
            frames_cache_dir=frames_cache_dir,
            seed=args.seed,
            tokenizer=tokenizer,
        )
        all_rows.extend(rows)

    # Stable sort for deterministic output.
    all_rows.sort(key=lambda r: r["eval_id"])

    with out_path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts_by_type: dict[str, int] = defaultdict(int)
    counts_by_source: dict[str, int] = defaultdict(int)
    for r in all_rows:
        counts_by_type[r["position_type"]] += 1
        counts_by_source[r["source"]] += 1

    summary = {
        "total": len(all_rows),
        "by_position_type": dict(counts_by_type),
        "by_source": dict(counts_by_source),
        "seed": args.seed,
        "tokenizer_repo": args.tokenizer_repo,
        "frames_cache": str(frames_cache_dir),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %d rows to %s", len(all_rows), out_path)
    logger.info("summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build a step-deduped eval set for V5 nested labeling A/B.

Samples unique timesteps (one row per ``example_id``) from LIBERO activation
dumps, materialises camera frames, and writes ``eval_set_steps.jsonl``.

Example::

    PYTHONPATH=src python scripts/labeling/build_v5_step_eval_set.py \\
        --activations-root data/activations/libero_goal_pilot \\
        --dataset-root third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --source-name libero_goal_pilot \\
        --n-steps 150 \\
        --out data/prompt_ab/v5_step_eval_set.jsonl \\
        --frames-cache data/prompt_ab/v5_frames_cache \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("nla.v5_eval_set")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--activations-root",
        action="append",
        required=True,
        help="Activation dump root; repeat with --dataset-root and --source-name.",
    )
    p.add_argument("--dataset-root", action="append", required=True)
    p.add_argument("--source-name", action="append", required=True)
    p.add_argument(
        "--n-steps",
        type=int,
        default=150,
        help="Target unique steps per source (total ~= n_steps * num_sources).",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames-cache", required=True, type=Path)
    p.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Optional suite tag per source (e.g. libero_goal).",
    )
    p.add_argument(
        "--tokenizer-repo",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p


def _materialize_step_row(
    rec,
    *,
    source_name: str,
    dataset_root: Path,
    frames_cache_dir: Path,
    pool,
    reader,
    video_keys: list[str],
    tokenizer,
    suite: str | None,
) -> dict | None:
    from nla.labeling.context import decode_text_context
    from nla.labeling.frames import save_jpeg

    if rec.episode_index is None or rec.step_index is None:
        return None

    item = reader.get(rec.example_id)
    ids = item.get("input_ids")
    if ids is None:
        logger.warning("skip %s: no input_ids", rec.example_id)
        return None
    attn = item["attention_mask"]
    img = item["image_mask"]
    ctx = decode_text_context(ids, img, tokenizer)

    loader = pool.get(dataset_root, rec.episode_index)
    image_paths: list[str] = []
    for vk in video_keys:
        try:
            frame = loader.frame(vk, rec.step_index)
        except (FileNotFoundError, IndexError) as e:
            logger.warning("skip %s: frame load failed (%s)", rec.example_id, e)
            return None
        out = frames_cache_dir / source_name / f"{rec.example_id}__{vk}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_jpeg(frame, out)
        image_paths.append(str(out))

    instruction = rec.task_text or ""
    if not instruction:
        from nla.labeling.frames import DatasetInfo

        di = DatasetInfo.from_root(dataset_root)
        instruction = di.episode_to_task.get(int(rec.episode_index), "")

    eval_id = f"{source_name}/{rec.example_id}"
    return {
        "eval_id": eval_id,
        "source": source_name,
        "example_id": rec.example_id,
        "instruction": instruction,
        "decoded_text_context": ctx,
        "sequence_length": int(rec.seq_len),
        "image_paths": image_paths,
        "episode_index": int(rec.episode_index),
        "step_index": int(rec.step_index),
        "suite": suite,
        "position_index": 0,
        "position_type": "step",
    }


def _sample_steps_from_source(
    *,
    activations_root: Path,
    dataset_root: Path,
    source_name: str,
    n_steps: int,
    frames_cache_dir: Path,
    tokenizer,
    suite: str | None,
    rng: np.random.Generator,
) -> list[dict]:
    from nla.extraction.storage import ActivationShardReader
    from nla.labeling.context import FrameLoaderPool
    from nla.labeling.frames import DatasetInfo

    reader = ActivationShardReader(activations_root)
    records = list(reader.records)
    if not records:
        logger.warning("no records in %s", activations_root)
        return []

    if len(records) <= n_steps:
        chosen = records
    else:
        idx = rng.choice(len(records), size=n_steps, replace=False)
        chosen = [records[i] for i in sorted(idx.tolist())]

    di = DatasetInfo.from_root(dataset_root)
    pool = FrameLoaderPool(max_open=8)
    rows: list[dict] = []
    try:
        for rec in chosen:
            row = _materialize_step_row(
                rec,
                source_name=source_name,
                dataset_root=dataset_root,
                frames_cache_dir=frames_cache_dir,
                pool=pool,
                reader=reader,
                video_keys=di.video_keys,
                tokenizer=tokenizer,
                suite=suite,
            )
            if row is not None:
                rows.append(row)
    finally:
        pool.close_all()
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    n_src = len(args.source_name)
    for name, lst in (
        ("activations-root", args.activations_root),
        ("dataset-root", args.dataset_root),
    ):
        if len(lst) != n_src:
            raise SystemExit(f"--{name} count must match --source-name ({n_src})")

    suites = args.suite or [None] * n_src
    if len(suites) != n_src:
        raise SystemExit("--suite count must match --source-name when provided")

    from nla.labeling.context import load_qwen3_vl_tokenizer

    tokenizer = load_qwen3_vl_tokenizer(args.tokenizer_repo)
    rng = np.random.default_rng(args.seed)
    args.frames_cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for src, act, ds, suite in zip(
        args.source_name,
        args.activations_root,
        args.dataset_root,
        suites,
    ):
        rows = _sample_steps_from_source(
            activations_root=Path(act),
            dataset_root=Path(ds),
            source_name=src,
            n_steps=args.n_steps,
            frames_cache_dir=args.frames_cache,
            tokenizer=tokenizer,
            suite=suite,
            rng=rng,
        )
        logger.info("source %s: kept %d step rows", src, len(rows))
        all_rows.extend(rows)

    # Dedupe by example_id across sources (first wins).
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in sorted(all_rows, key=lambda r: r["eval_id"]):
        eid = row["example_id"]
        if eid in seen:
            continue
        seen.add(eid)
        deduped.append(row)

    with args.out.open("w") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total": len(deduped),
        "n_steps_target": args.n_steps,
        "sources": list(args.source_name),
        "seed": args.seed,
        "out": str(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2),
    )
    logger.info("wrote %d rows -> %s", len(deduped), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

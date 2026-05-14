"""End-to-end labeling pipeline.

Given an extraction dump and the source LeRobot dataset, produce per-position
warm-start labels using a multimodal model.

Outputs::

    <labels_dir>/
      frames_cache/<example_id>__<video_key>.jpg
      labels.jsonl                    # streamed; resumable on example_id
      manifest.json                   # source paths + model + counts
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nla.extraction.storage import ActivationShardReader
from nla.labeling.context import (
    FrameLoaderPool,
    build_position_inputs,
    load_qwen3_vl_tokenizer,
    sample_one_position_per_example,
    sample_positions_per_example,
)
from nla.labeling.openai_client import DEFAULT_MODEL, label_many_async

logger = logging.getLogger(__name__)


@dataclass
class LabelingManifest:
    activations_root: str
    dataset_root: str
    labels_dir: str
    model: str
    tokenizer_repo: str
    seed: int
    concurrency: int
    state_name: str | None
    n_planned: int
    n_completed: int
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


async def run_labeling(
    activations_root: str | Path,
    dataset_root: str | Path,
    labels_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    tokenizer_repo: str = "Qwen/Qwen3-VL-2B-Instruct",
    seed: int = 0,
    concurrency: int = 16,
    state_name: str | None = None,
    max_examples: int | None = None,
    positions_per_example: int = 1,
    api_key: str | None = None,
    resume: bool = True,
) -> int:
    """Sample one position per example, build inputs with frames, label them.

    Returns the number of newly-labeled examples.
    """
    activations_root = Path(activations_root)
    dataset_root = Path(dataset_root)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    frames_cache = labels_dir / "frames_cache"
    frames_cache.mkdir(parents=True, exist_ok=True)
    out_jsonl = labels_dir / "labels.jsonl"

    logger.info("Loading activation index from %s", activations_root)
    reader = ActivationShardReader(activations_root)
    logger.info("  %d examples", len(reader))

    logger.info("Loading tokenizer %s", tokenizer_repo)
    tokenizer = load_qwen3_vl_tokenizer(tokenizer_repo)

    logger.info(
        "Sampling %d position(s) per example (seed=%d)", positions_per_example, seed,
    )
    sampled = list(
        sample_positions_per_example(
            reader, tokenizer, n_per_example=positions_per_example, seed=seed,
        )
    )
    if max_examples is not None:
        sampled = sampled[: int(max_examples)]
    logger.info("  %d sampled positions across %d examples",
                len(sampled), len(set(s.record.example_id for s in sampled)))

    pool = FrameLoaderPool(max_open=8)
    try:
        logger.info("Loading frames and building position inputs into %s", frames_cache)
        inputs = list(
            build_position_inputs(
                sampled,
                dataset_root=dataset_root,
                frame_cache_dir=frames_cache,
                state_name=state_name,
                pool=pool,
            )
        )
        logger.info("  %d inputs ready (with frames)", len(inputs))

        n_new = await label_many_async(
            inputs,
            out_jsonl,
            model=model,
            concurrency=concurrency,
            api_key=api_key,
            resume=resume,
        )
    finally:
        pool.close_all()

    manifest = LabelingManifest(
        activations_root=str(activations_root),
        dataset_root=str(dataset_root),
        labels_dir=str(labels_dir),
        model=model,
        tokenizer_repo=tokenizer_repo,
        seed=seed,
        concurrency=concurrency,
        state_name=state_name,
        n_planned=len(inputs),
        n_completed=_count_completed(out_jsonl),
    )
    manifest.save(labels_dir / "manifest.json")
    logger.info(
        "Labeling done. New: %d. Total in %s: %d.",
        n_new, out_jsonl, manifest.n_completed,
    )
    return n_new


def _count_completed(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("description") and not obj.get("error"):
                n += 1
    return n


def run_labeling_sync(*args, **kwargs) -> int:
    """Thin sync wrapper for CLI use."""
    return asyncio.run(run_labeling(*args, **kwargs))

#!/usr/bin/env python
"""Backfill labels.jsonl rows for examples with fewer than the planned positions.

The original labeling run sampled ``positions_per_example=4`` distinct token
positions per example with ``seed=0`` (recorded in
``data/labels/<dataset>/manifest.json``). After deduplication and any manual
drops, a handful of examples typically end up with fewer than 4 unique
``(source_example_id, position_index, position_type)`` rows in ``labels.jsonl``.

Two backfill strategies are supported:

* ``--mode rng-match`` (default): replay the original sampler with the same
  ``seed`` over the full ``ActivationShardReader.iter_examples`` order, and
  keep only the sampled positions whose canonical key is missing for an
  under-covered example. This will only find positions that the original
  sampler emitted but were later dropped (typically just the row we removed
  manually) -- the 367 thin examples genuinely had ``< 4`` unique positions
  the first time around because of sampler collisions, so this mode does not
  recover them.

* ``--mode force-fill``: for each under-covered example, draw additional
  *new* unique positions (preferring ``image_patch`` tokens, then ``last_text``
  / ``anchor`` if free) until the example has ``--positions-per-example`` rows.
  Uses a deterministic per-example RNG (``seed`` xor a hash of the
  ``source_example_id``) so re-runs are reproducible.

Both modes append to ``labels.jsonl`` via ``label_many_async`` with
``resume=True``, so partial runs are safe to retry.

Example::

    python scripts/labeling/backfill_label_gaps.py \\
        --labels      data/labels/libero_goal_pilot/labels.jsonl \\
        --activations data/activations/libero_goal_pilot \\
        --dataset     third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --positions-per-example 4 \\
        --mode force-fill \\
        --model gpt-5-mini \\
        --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.extraction.sampler import (  # noqa: E402
    PositionType,
    _anchor_index,
    _image_patch_index,
    _last_text_index,
)
from nla.extraction.storage import ActivationShardReader, RunManifest  # noqa: E402
from nla.labeling import openai_client  # noqa: E402
from nla.labeling.context import (  # noqa: E402
    DEFAULT_TOKENIZER_REPO,
    FrameLoaderPool,
    SampledExample,
    build_position_inputs,
    decode_text_context,
    image_patch_meta,
    load_qwen3_vl_tokenizer,
    sample_positions_per_example,
)
from nla.labeling.openai_client import label_many_async  # noqa: E402

logger = logging.getLogger(__name__)


def _scan_existing(
    labels_path: Path,
) -> tuple[Counter, dict[str, set[tuple[int, str]]], set[tuple[str, int, str]]]:
    """Return (counts_per_sid, positions_per_sid, set_of_canonical_keys)."""
    counts: Counter = Counter()
    by_sid: dict[str, set[tuple[int, str]]] = {}
    keys: set[tuple[str, int, str]] = set()
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("kind") != "position":
                continue
            m = obj.get("meta") or {}
            sid = m.get("source_example_id")
            pidx = m.get("position_index")
            pt = m.get("position_type")
            if sid is None or pidx is None or pt is None:
                continue
            sid = str(sid)
            counts[sid] += 1
            by_sid.setdefault(sid, set()).add((int(pidx), str(pt)))
            keys.add((sid, int(pidx), str(pt)))
    return counts, by_sid, keys


def _sid_seed(global_seed: int, sid: str) -> int:
    """Stable per-sid seed derived from sid and the global seed."""
    h = hashlib.sha256(f"{global_seed}:{sid}".encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


def _force_fill_samples(
    thin: dict[str, int],
    *,
    target: int,
    by_sid: dict[str, set[tuple[int, str]]],
    reader: ActivationShardReader,
    tokenizer,
    seed: int,
) -> list[SampledExample]:
    """For each thin sid, sample new unique positions to bring count to target.

    Preference order: image_patch -> last_text -> anchor (skipping any (pidx, pt)
    already present for the sid). image_patch indices are sampled uniformly
    without replacement.
    """
    out: list[SampledExample] = []
    skipped_no_ids = 0
    skipped_no_capacity = 0

    by_id = {rec.example_id: i for i, rec in enumerate(reader.records)}
    for sid, count in sorted(thin.items()):
        need = target - count
        if need <= 0:
            continue
        if sid not in by_id:
            logger.warning("thin sid %s missing from activation index; skipping", sid)
            continue
        item = reader[by_id[sid]]
        rec = item["_record"]
        attn = item["attention_mask"]
        img = item["image_mask"]
        ids = item.get("input_ids")
        if ids is None:
            skipped_no_ids += 1
            continue

        existing = by_sid.get(sid, set())
        existing_idx = {pidx for (pidx, _) in existing}
        rng = np.random.default_rng(_sid_seed(seed, sid))
        text_ctx = decode_text_context(ids, img, tokenizer)

        new_positions: list[tuple[int, str]] = []

        valid_img = (attn.bool() & img.bool()).nonzero(as_tuple=False).flatten().tolist()
        free_img = [i for i in valid_img if i not in existing_idx]
        if free_img:
            rng.shuffle(free_img)
            for pidx in free_img:
                if len(new_positions) >= need:
                    break
                new_positions.append((int(pidx), PositionType.IMAGE_PATCH.value))
                existing_idx.add(int(pidx))

        if len(new_positions) < need:
            lt = _last_text_index(attn, img)
            if lt is not None and lt not in existing_idx:
                if not any(pt == PositionType.LAST_TEXT.value for (_, pt) in existing):
                    new_positions.append((int(lt), PositionType.LAST_TEXT.value))
                    existing_idx.add(int(lt))

        if len(new_positions) < need:
            an = _anchor_index(attn)
            if an is not None and an not in existing_idx:
                if not any(pt == PositionType.ANCHOR.value for (_, pt) in existing):
                    new_positions.append((int(an), PositionType.ANCHOR.value))
                    existing_idx.add(int(an))

        if len(new_positions) < need:
            skipped_no_capacity += 1
            logger.warning(
                "sid %s: only able to add %d/%d new unique positions",
                sid, len(new_positions), need,
            )

        for pidx, pt in new_positions:
            meta = image_patch_meta(img, pidx)
            out.append(SampledExample(
                record=rec,
                position_index=pidx,
                position_type=pt,
                decoded_text_context=text_ctx,
                image_patch_meta=meta,
            ))

    if skipped_no_ids:
        logger.warning("%d sids skipped (no input_ids in shard)", skipped_no_ids)
    if skipped_no_capacity:
        logger.warning("%d sids did not reach target (insufficient unique positions)", skipped_no_capacity)
    return out


def _rng_match_samples(
    thin: dict[str, int],
    existing_keys: set[tuple[str, int, str]],
    *,
    reader: ActivationShardReader,
    tokenizer,
    target: int,
    seed: int,
) -> list[SampledExample]:
    """Replay the original deterministic sampling pass and keep gap keys."""
    sampled_iter = sample_positions_per_example(
        reader, tokenizer,
        n_per_example=target,
        seed=seed,
        require_input_ids=True,
        record_filter=None,
    )
    out: list[SampledExample] = []
    for s in sampled_iter:
        sid = s.record.example_id
        if sid not in thin:
            continue
        key = (str(sid), int(s.position_index), str(s.position_type))
        if key in existing_keys:
            continue
        out.append(s)
    return out


async def _do_backfill(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels)
    activations_root = Path(args.activations)
    if not labels_path.exists():
        raise SystemExit(f"labels file not found: {labels_path}")

    if args.dataset:
        dataset_root = Path(args.dataset)
    else:
        manifest = RunManifest.load(activations_root / "manifest.json")
        dpath = manifest.extra.get("dataset_path")
        if not dpath or not Path(dpath).is_dir():
            raise SystemExit(
                f"--dataset is required (manifest dataset_path '{dpath}' is not a directory)"
            )
        dataset_root = Path(dpath)
    logger.info("dataset_root: %s", dataset_root)

    frame_cache_dir = Path(args.frames_cache or (labels_path.parent / "frames_cache"))
    frame_cache_dir.mkdir(parents=True, exist_ok=True)

    counts, by_sid, existing_keys = _scan_existing(labels_path)
    target = int(args.positions_per_example)
    thin = {sid: c for sid, c in counts.items() if c < target}
    logger.info(
        "scanned %d position keys across %d sids; %d sids are under-covered (<%d)",
        len(existing_keys), len(counts), len(thin), target,
    )
    if not thin:
        logger.info("no thin examples; nothing to do")
        return 0

    logger.info("loading activation reader and tokenizer (mode=%s)", args.mode)
    reader = ActivationShardReader(activations_root)
    tokenizer = load_qwen3_vl_tokenizer(args.tokenizer)

    if args.mode == "rng-match":
        todo_samples = _rng_match_samples(
            thin, existing_keys,
            reader=reader, tokenizer=tokenizer,
            target=target, seed=args.seed,
        )
    elif args.mode == "force-fill":
        todo_samples = _force_fill_samples(
            thin,
            target=target,
            by_sid=by_sid,
            reader=reader,
            tokenizer=tokenizer,
            seed=args.seed,
        )
    else:
        raise SystemExit(f"unknown --mode {args.mode!r}")

    logger.info(
        "candidate gap positions: %d (across %d thin examples)",
        len(todo_samples), len({s.record.example_id for s in todo_samples}),
    )

    if args.dry_run:
        for s in todo_samples[:10]:
            logger.info("  todo: %s @ %d %s", s.record.example_id, s.position_index, s.position_type)
        logger.info("dry-run: not calling labeler")
        return 0

    if not todo_samples:
        logger.info("nothing to label")
        return 0

    pool = FrameLoaderPool(max_open=8)
    try:
        inputs = list(build_position_inputs(
            todo_samples,
            dataset_root=dataset_root,
            frame_cache_dir=frame_cache_dir,
            state_name=args.state_name,
            pool=pool,
        ))
    finally:
        pool.close_all()
    logger.info("built %d PositionLabelInputs (with frames)", len(inputs))
    if len(inputs) < len(todo_samples):
        logger.warning(
            "only %d/%d inputs assembled (some examples failed frame loading)",
            len(inputs), len(todo_samples),
        )

    n_new = await label_many_async(
        inputs,
        labels_path,
        model=args.model,
        concurrency=args.concurrency,
        api_key=args.api_key,
        resume=True,
    )
    logger.info("appended %d new rows -> %s", n_new, labels_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--activations", required=True, type=Path)
    p.add_argument("--dataset", type=Path, default=None,
                   help="LeRobot dataset root (default: read from activations manifest)")
    p.add_argument("--frames-cache", type=Path, default=None,
                   help="default: <labels-dir>/frames_cache")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_REPO)
    p.add_argument("--state-name", default=None)
    p.add_argument("--positions-per-example", type=int, default=4,
                   help="Target positions per example (default 4)")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for RNG-match (must match labels manifest) or per-sid hash base")
    p.add_argument("--mode", choices=["rng-match", "force-fill"], default="force-fill",
                   help="rng-match: replay original sampler; force-fill: draw new unique positions")
    p.add_argument("--model", default=None,
                   help="OpenAI labeling model (default: env OPENAI_LABELING_MODEL or gpt-5-mini)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--api-key", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Print missing-key count and a few example ids without calling the API")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.model is None:
        args.model = openai_client.DEFAULT_MODEL

    return asyncio.run(_do_backfill(args))


if __name__ == "__main__":
    sys.exit(main())

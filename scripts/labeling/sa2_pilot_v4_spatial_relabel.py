#!/usr/bin/env python
"""SA2 V4 libero_spatial pilot re-label.

Selects libero_spatial rows that the V3 multimodal judge graded ``grounding
!= specific`` (the failure cluster, ~34 rows), re-labels them through the
V4 prompt with the new ``libero_spatial`` suite addendum (SP-1..SP-5), and
writes the new captions to
``data/labels/sa2_pilot_v4_spatial/labels.jsonl``.

Wired the same way as ``scripts/labeling/relabel_bad_rows.py``: we
monkey-patch ``nla.labeling.openai_client.build_position_prompt`` to dispatch
through ``build_v4_position_prompt(inp, suite="libero_spatial")`` so the
existing async client streams the new captions to JSONL with resume.

Usage::

    set -a && source .env && set +a
    PYTHONPATH=src .venv/bin/python scripts/labeling/sa2_pilot_v4_spatial_relabel.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.extraction.storage import ActivationShardReader  # noqa: E402
from nla.labeling import openai_client  # noqa: E402
from nla.labeling.context import (  # noqa: E402
    DEFAULT_TOKENIZER_REPO,
    FrameLoaderPool,
    SampledExample,
    build_position_inputs,
    decode_text_context,
    image_patch_meta,
    load_qwen3_vl_tokenizer,
)
from nla.labeling.openai_client import label_many_async  # noqa: E402
from nla.labeling.prompts import build_v4_position_prompt  # noqa: E402


SUITE = "libero_spatial"
JUDGE_PATH = Path("data/eval/libero_v3_quality_judge.jsonl")
ACTIVATIONS_ROOT = Path("data/activations/libero_4suite_stride2/libero_spatial")
DATASET_ROOT = Path(
    "third_party/Isaac-GR00T/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot"
)
LABELS_OUT = Path("data/labels/sa2_pilot_v4_spatial/labels.jsonl")
FRAME_CACHE = Path("data/labels/sa2_pilot_v4_spatial/frames_cache")


def _parse_eid(eid: str) -> tuple[str, int, str] | None:
    """Split ``libero_spatial::traj000017_step000014@p151_anchor`` ->
    ``("traj000017_step000014", 151, "anchor")``.  Returns None if it cannot
    be parsed."""
    if "::" not in eid:
        return None
    _, rest = eid.split("::", 1)
    if "@" not in rest:
        return None
    src, tail = rest.split("@", 1)
    if not tail.startswith("p"):
        return None
    tail = tail[1:]
    if "_" not in tail:
        return None
    pidx_s, ptype = tail.split("_", 1)
    try:
        pidx = int(pidx_s)
    except ValueError:
        return None
    return src, pidx, ptype


def _load_spatial_bad_keys() -> list[tuple[str, int, str]]:
    """Return list of (source_id, position_index, position_type) for every
    libero_spatial row that the V3 judge graded as ``grounding != specific``."""
    keys: list[tuple[str, int, str]] = []
    with JUDGE_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            eid = obj.get("example_id") or ""
            if not eid.startswith(f"{SUITE}::"):
                continue
            g = (obj.get("grounding") or {}).get("verdict")
            if g == "specific":
                continue
            key = _parse_eid(eid)
            if key is None:
                continue
            keys.append(key)
    return keys


def _build_inputs(
    keys: list[tuple[str, int, str]],
    tokenizer_repo: str,
) -> list:
    by_sid: dict[str, list[tuple[int, str]]] = {}
    for sid, pidx, pt in keys:
        by_sid.setdefault(sid, []).append((pidx, pt))

    reader = ActivationShardReader(ACTIVATIONS_ROOT)
    tokenizer = load_qwen3_vl_tokenizer(tokenizer_repo)
    pool = FrameLoaderPool(max_open=8)

    sampled: list[SampledExample] = []
    for item in reader.iter_examples(
        record_filter=lambda rec: rec.example_id in by_sid,
    ):
        rec = item["_record"]
        img = item["image_mask"]
        ids = item.get("input_ids")
        if ids is None:
            logging.warning("example %s has no input_ids; skipping", rec.example_id)
            continue
        text_ctx = decode_text_context(ids, img, tokenizer)
        for pidx, pt in by_sid[rec.example_id]:
            meta = image_patch_meta(img, pidx)
            sampled.append(SampledExample(
                record=rec,
                position_index=pidx,
                position_type=pt,
                decoded_text_context=text_ctx,
                image_patch_meta=meta,
            ))

    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    inputs = list(build_position_inputs(
        sampled,
        dataset_root=DATASET_ROOT,
        frame_cache_dir=FRAME_CACHE,
        state_name=None,
        pool=pool,
    ))
    pool.close_all()
    return inputs


async def _do_relabel(args: argparse.Namespace) -> int:
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)

    keys = _load_spatial_bad_keys()
    logging.info("V3 libero_spatial non-specific rows: %d", len(keys))
    if args.limit > 0:
        keys = keys[: args.limit]
        logging.info("limit applied: %d keys", len(keys))

    inputs = _build_inputs(keys, args.tokenizer)
    logging.info("built %d PositionLabelInputs (with frames)", len(inputs))
    if len(inputs) < len(keys):
        logging.warning(
            "only %d/%d inputs assembled (missing input_ids or frames)",
            len(inputs), len(keys),
        )

    def _v4_spatial_builder(inp):
        return build_v4_position_prompt(inp, suite=SUITE)

    orig_builder = openai_client.build_position_prompt
    openai_client.build_position_prompt = _v4_spatial_builder
    try:
        n_new = await label_many_async(
            inputs,
            LABELS_OUT,
            model=args.model,
            concurrency=args.concurrency,
            api_key=args.api_key,
            resume=True,
        )
    finally:
        openai_client.build_position_prompt = orig_builder

    logging.info("re-labeled %d new rows -> %s", n_new, LABELS_OUT)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_REPO)
    p.add_argument("--model", default=None,
                   help="OpenAI labeling model (default: env OPENAI_LABELING_MODEL)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--api-key", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, label only first N keys (debug).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.model is None:
        args.model = openai_client.DEFAULT_MODEL

    asyncio.run(_do_relabel(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

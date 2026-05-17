#!/usr/bin/env python
"""SA2 V4 libero_spatial pilot — re-grade.

Pairs each row in ``data/labels/sa2_pilot_v4_spatial/labels.jsonl`` (produced
by ``scripts/labeling/sa2_pilot_v4_spatial_relabel.py``) with the same camera
frames the V3 judge saw, then re-grades the V4 caption with the multimodal
``gpt-5.1`` judge from ``nla.labeling.grader``.

We reuse the existing ``frames_cache/`` from the V3 spatial labels by
preferring it as the frame source (no need to re-extract).  Output JSONL
schema matches ``verify_libero_label_quality.py`` so the SA2 pilot results
can be merged into the V4 scorecard later.

Usage::

    set -a && source .env && set +a
    PYTHONPATH=src .venv/bin/python scripts/eval/sa2_pilot_v4_spatial_judge.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


SUITE = "libero_spatial"
PILOT_LABELS = Path("data/labels/sa2_pilot_v4_spatial/labels.jsonl")
PILOT_FRAMES_CACHES = [
    # Prefer the pilot's own cache (populated by the relabel script).
    Path("data/labels/sa2_pilot_v4_spatial/frames_cache"),
    # Fall back to the V3 spatial run's frames cache (same source frames).
    Path("data/labels/libero_4suite_stride2/libero_spatial/frames_cache"),
]
OUT_JSONL = Path("data/eval/sa2_pilot_v4_spatial_judge.jsonl")

CAMERA_SUFFIXES = (
    "__observation_images_image.jpg",
    "__observation_images_wrist_image.jpg",
    "__image.jpg",
    "__wrist_image.jpg",
    "__observation.images.image.jpg",
    "__observation.images.wrist_image.jpg",
)


def _find_frames(source_id: str) -> list[str]:
    for cache in PILOT_FRAMES_CACHES:
        paths: list[str] = []
        for suffix in CAMERA_SUFFIXES:
            cand = cache / f"{source_id}{suffix}"
            if cand.exists():
                paths.append(str(cand))
        if paths:
            return paths
    return []


def _iter_pilot_rows():
    with PILOT_LABELS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            desc = (obj.get("description") or "").strip()
            if not desc:
                continue
            meta = obj.get("meta") or {}
            src = meta.get("source_example_id")
            pos_idx = meta.get("position_index")
            pos_type = meta.get("position_type")
            instr = meta.get("instruction") or ""
            if src is None or pos_idx is None or pos_type is None:
                continue
            frames = _find_frames(str(src))
            if not frames:
                continue
            yield {
                "source_id": str(src),
                "position_index": int(pos_idx),
                "position_type": str(pos_type),
                "description": desc,
                "instruction": str(instr),
                "image_paths": frames,
            }


async def _amain(args) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    if args.judge_model:
        os.environ["OPENAI_GRADER_MODEL"] = args.judge_model

    from nla.labeling.grader import GradeInput, grade_many_async

    if not PILOT_LABELS.exists():
        print(f"ERROR: pilot labels not found at {PILOT_LABELS}", file=sys.stderr)
        return 2

    rows = list(_iter_pilot_rows())
    print(f"Pilot rows with frames: {len(rows)}")
    if not rows:
        print("ERROR: no grade-eligible pilot rows.", file=sys.stderr)
        return 2

    inputs: list[GradeInput] = []
    for r in rows:
        eid = f"{SUITE}::{r['source_id']}@p{r['position_index']}_{r['position_type']}"
        inputs.append(GradeInput(
            example_id=eid,
            variant_id=args.variant_id,
            description=r["description"],
            instruction=r["instruction"],
            position_type=r["position_type"],
            image_paths=r["image_paths"],
            seq_len=None,
            position_index=r["position_index"],
        ))

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Grading {len(inputs)} rows -> {OUT_JSONL} (concurrency={args.concurrency})")
    n_new = await grade_many_async(
        inputs,
        output_jsonl=OUT_JSONL,
        concurrency=args.concurrency,
        resume=True,
    )
    print(f"  -> {n_new} new grades written")

    n = b_pass = c_pass = 0
    with OUT_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1
            if (obj.get("grounding") or {}).get("verdict") == "specific":
                b_pass += 1
            if (obj.get("appropriateness") or {}).get("verdict") == "appropriate":
                c_pass += 1
    if n:
        print(f"\nOVERALL n={n}  B={b_pass/n*100:5.2f}%  C={c_pass/n*100:5.2f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--judge-model", default=None,
                   help="Override OPENAI_GRADER_MODEL (default: gpt-5.1).")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--variant-id", default="sa2_pilot_v4_spatial")
    args = p.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())

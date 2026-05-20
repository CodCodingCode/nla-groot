#!/usr/bin/env python
"""Expand V5 step labels (one row per timestep) into three position rows.

Reads ``labels_steps.jsonl`` plus an activation dump index, resolves token
indices from ``attention_mask`` / ``image_mask`` using the same helpers as
``nla.extraction.sampler`` and ``nla.labeling.context``:

* ``last_text`` — last non-image, non-pad token (``_last_text_index``)
* ``anchor`` — final non-pad token (``_anchor_index``)
* ``image_patch`` — middle image-patch token in the valid image stretch

Usage::

    PYTHONPATH=src python scripts/labeling/expand_step_labels.py \\
        --labels-steps data/labels/my_run/labels_steps.jsonl \\
        --activations-root data/activations/droid_smoke \\
        --out data/labels/my_run/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nla.extraction.sampler import (
    _anchor_index,
    _last_text_index,
    iter_image_positions,
)
from nla.extraction.storage import ActivationShardReader
from nla.labeling.schema_v5 import (
    SLOT_NAMES,
    extract_nested_from_row,
    render_slot_bullets,
    validate_nested,
)

logger = logging.getLogger(__name__)


def resolve_position_indices(
    attention_mask,
    image_mask,
) -> dict[str, int | None]:
    """Canonical V5 slot indices for one example (matches context sampler)."""
    last_idx = _last_text_index(attention_mask, image_mask)
    anchor_idx = _anchor_index(attention_mask)
    patch_positions = iter_image_positions(attention_mask, image_mask)
    if patch_positions:
        patch_idx = patch_positions[len(patch_positions) // 2]
    else:
        patch_idx = None
    return {
        "last_text": last_idx,
        "anchor": anchor_idx,
        "image_patch": patch_idx,
    }


def _source_example_id(row: dict) -> str | None:
    for key in ("source_example_id", "example_id", "step_id"):
        val = row.get(key)
        if val:
            return str(val).split("@")[0]
    return None


def _prefix_example_id(suite: str | None, raw: str) -> str:
    if not suite:
        return raw
    head = f"{suite}__"
    return raw if raw.startswith(head) else f"{head}{raw}"


def expand_row(
    row: dict,
    *,
    indices: dict[str, int | None],
    normalized_slots: dict,
    suite: str | None = None,
) -> list[dict]:
    """Build three position-label rows from one validated step label."""
    sid = _source_example_id(row) or "unknown"
    sid_pref = _prefix_example_id(suite, sid)
    out: list[dict] = []
    last_idx = indices.get("last_text")
    anchor_idx = indices.get("anchor")
    if last_idx is not None and anchor_idx == last_idx:
        anchor_idx = None  # dedupe: keep last_text row only at shared index

    for slot in SLOT_NAMES:
        pidx = indices.get(slot)
        if slot == "anchor" and anchor_idx is None:
            continue
        if slot == "anchor":
            pidx = anchor_idx
        if pidx is None:
            logger.warning("skip %s %s: no valid index", sid, slot)
            continue
        bullets = render_slot_bullets(normalized_slots[slot])
        example_id = f"{sid_pref}@p{pidx:03d}_{slot}"
        meta = {
            "source_example_id": sid_pref,
            "position_index": int(pidx),
            "position_type": slot,
            "label_schema": "v5",
            "label_version": "v5",
        }
        if suite:
            meta["suite"] = suite
        out.append(
            {
                "example_id": example_id,
                "description": bullets,
                "kind": "position",
                "meta": meta,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels-steps", required=True, type=Path)
    p.add_argument("--activations-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--suite",
        default=None,
        help="LIBERO suite tag (goal/spatial/object/10) for combined-corpus id prefix.",
    )
    p.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip rows that fail validate_nested instead of aborting",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.labels_steps.is_file():
        print(f"not found: {args.labels_steps}", file=sys.stderr)
        return 1

    reader = ActivationShardReader(args.activations_root)
    by_id = {rec.example_id: rec for rec in reader.records}

    n_in = 0
    n_out = 0
    n_skip = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.labels_steps.open() as fin, args.out.open("w") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            nested = extract_nested_from_row(row)
            if nested is None:
                logger.warning("line %d: no nested V5 object", line_no)
                n_skip += 1
                continue

            ok, errors, norm = validate_nested(nested)
            if not ok:
                msg = f"line {line_no} ({_source_example_id(row)}): {errors[:3]}"
                if args.skip_invalid:
                    logger.warning("skip invalid: %s", msg)
                    n_skip += 1
                    continue
                raise SystemExit(msg)

            sid = _source_example_id(row)
            if sid is None or sid not in by_id:
                logger.warning("line %d: unknown example_id %r", line_no, sid)
                n_skip += 1
                continue

            item = reader.get(sid)
            attn = item["attention_mask"]
            img = item["image_mask"]
            indices = resolve_position_indices(attn, img)

            for pos_row in expand_row(
                row, indices=indices, normalized_slots=norm, suite=args.suite,
            ):
                fout.write(json.dumps(pos_row, separators=(",", ":")) + "\n")
                n_out += 1

    print(
        f"expanded {n_in} step rows -> {n_out} position rows "
        f"(skipped {n_skip}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

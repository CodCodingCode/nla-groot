#!/usr/bin/env python
"""Strip hallucinated ``image_region`` bullets from a labels.jsonl in place.

Background
----------
The per-position labeler is told ``image patch k of n`` as metadata but is
*not* told which patch is k.  When the labeler then writes things like
"patch 89 (lower-left quadrant)" or "this image patch encodes the upper-right
of the table" it is inferring screen layout from k alone -- a teacher-side
hallucination that the AV will learn if we treat those bullets as gold.
See ``docs/sft_plan/01_data_audit.md`` section 3.2.

This script does *not* call any API.  It reads ``--labels`` row by row,
detects ``image_region`` bullet lines whose text matches the hallucination
pattern, and either drops the bullet or replaces it with a neutral fallback.
The modified row is otherwise preserved verbatim (other bullets unchanged).

Bullet-match modes (``--match``)
--------------------------------
``patch``           : strip only when the bullet contains the literal word
                      "patch" (the strongest signal of "labeler describing
                      what they think is in patch k").  Default.
``patch_or_layout`` : strip when the bullet contains "patch", "quadrant",
                      "corner", or a directional compound such as
                      "upper-left", "lower-right".  More aggressive: also
                      removes localization claims that did not name "patch".
``all_image_patch`` : strip every ``image_region`` bullet on rows whose
                      ``meta.position_type == "image_patch"``, regardless of
                      content.  Use when you want zero risk of any leftover
                      patch-localization hallucination at the cost of also
                      removing legit visually-grounded region bullets.

Modes (``--mode``)
------------------
``strip``   : delete the offending bullet line entirely.  Default.
``replace`` : replace the bullet with a neutral fallback explaining the
              region is not localized to a single patch index.

Backup
------
Unless ``--no-backup`` is passed the original ``labels.jsonl`` is copied to
``labels.jsonl.bak``, ``.bak2`` etc. (first free name) before any rewrite.

Example
-------
::

    PYTHONPATH=src python scripts/labeling/strip_hallucinated_image_region.py \\
        --labels data/labels/droid_100ep/labels.jsonl \\
        --match patch_or_layout \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# A markdown bullet line whose category is image_region (case-insensitive).
_IMAGE_REGION_BULLET_RE = re.compile(r"^\s*-\s*image_region\s*:", re.IGNORECASE)

# Patterns that flag a bullet as containing teacher-side patch-localization.
_PATCH_RE = re.compile(r"\bpatch\b", re.IGNORECASE)
_LAYOUT_RE = re.compile(
    r"\bquadrant\b|\bcorner\b|"
    r"\b(?:upper|lower|top|bottom)[\s\-](?:left|right|half|center|portion|region)\b|"
    r"\b(?:left|right)\s+(?:side|half|portion|region|quadrant)\b",
    re.IGNORECASE,
)

# Bullet that replaces the offender when ``--mode replace`` is selected.
NEUTRAL_REPLACEMENT = (
    "- image_region: visible features as in the attached camera frame; "
    "exact patch location within the frame is not specified."
)


def is_offending_bullet(line: str, *, match_mode: str) -> bool:
    """Return True when ``line`` is an ``image_region`` bullet that should be
    stripped under ``match_mode``.

    ``match_mode`` is one of ``patch``, ``patch_or_layout``,
    ``all_image_patch``.  The ``all_image_patch`` mode flags every
    ``image_region`` bullet regardless of content; the row-level
    ``position_type`` filter is applied by the caller.
    """
    if not _IMAGE_REGION_BULLET_RE.match(line):
        return False
    if match_mode == "all_image_patch":
        return True
    if _PATCH_RE.search(line):
        return True
    if match_mode == "patch_or_layout" and _LAYOUT_RE.search(line):
        return True
    return False


def transform_description(
    description: str,
    *,
    match_mode: str,
    replace_mode: bool,
    only_image_patch_row: bool,
) -> tuple[str, int]:
    """Return ``(new_description, n_bullets_modified)``.

    ``only_image_patch_row`` is True when ``match_mode == "all_image_patch"``
    and the row's ``position_type`` is ``image_patch`` (the caller resolves
    that).  In other match modes it is irrelevant.
    """
    if match_mode == "all_image_patch" and not only_image_patch_row:
        return description, 0

    lines = description.splitlines()
    out_lines: list[str] = []
    n_changed = 0
    for ln in lines:
        if is_offending_bullet(ln, match_mode=match_mode):
            if replace_mode:
                indent = ln[: len(ln) - len(ln.lstrip())]
                out_lines.append(indent + NEUTRAL_REPLACEMENT)
            n_changed += 1
            continue
        out_lines.append(ln)
    if n_changed == 0:
        return description, 0

    new_desc = "\n".join(out_lines)
    if description.endswith("\n") and not new_desc.endswith("\n"):
        new_desc += "\n"
    return new_desc, n_changed


def _pick_backup_path(labels_path: Path) -> Path:
    """First free `<labels>.bak`, `.bak2`, `.bak3`, ... so we never overwrite."""
    base = labels_path.with_suffix(labels_path.suffix + ".bak")
    if not base.exists():
        return base
    for i in range(2, 100):
        cand = labels_path.with_suffix(labels_path.suffix + f".bak{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(
        f"refusing to make a 100th backup of {labels_path}; clean up old .bak files"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument(
        "--match",
        choices=("patch", "patch_or_layout", "all_image_patch"),
        default="patch",
        help="Which image_region bullets count as offending. See module docstring.",
    )
    p.add_argument(
        "--mode",
        choices=("strip", "replace"),
        default="strip",
        help="strip = delete the bullet line; replace = swap for a neutral fallback.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing <labels>.bak before rewriting (NOT recommended).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts only; do not write.")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    labels_path: Path = args.labels
    if not labels_path.exists():
        raise SystemExit(f"labels file not found: {labels_path}")

    if not args.dry_run and not args.no_backup:
        bak = _pick_backup_path(labels_path)
        shutil.copy2(labels_path, bak)
        logger.info("backup: %s", bak)

    n_total = 0
    n_position = 0
    n_image_patch_rows = 0
    n_rows_changed = 0
    n_bullets_changed = 0

    tmp = labels_path.with_suffix(labels_path.suffix + ".tmp")
    out_handle = None if args.dry_run else tmp.open("w")
    try:
        with labels_path.open() as src:
            for raw in src:
                n_total += 1
                line = raw.rstrip("\n")
                if not line:
                    if out_handle:
                        out_handle.write("\n")
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if out_handle:
                        out_handle.write(line + "\n")
                    continue
                if obj.get("kind") != "position":
                    if out_handle:
                        out_handle.write(line + "\n")
                    continue
                n_position += 1

                meta = obj.get("meta") or {}
                ptype = meta.get("position_type")
                is_image_patch_row = ptype == "image_patch"
                if is_image_patch_row:
                    n_image_patch_rows += 1

                desc = obj.get("description") or ""
                new_desc, n_changed = transform_description(
                    desc,
                    match_mode=args.match,
                    replace_mode=(args.mode == "replace"),
                    only_image_patch_row=is_image_patch_row,
                )
                if n_changed:
                    n_rows_changed += 1
                    n_bullets_changed += n_changed
                    obj["description"] = new_desc
                    if out_handle:
                        out_handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    if out_handle:
                        out_handle.write(line + "\n")
    finally:
        if out_handle:
            out_handle.close()

    logger.info(
        "rows=%d position=%d image_patch_rows=%d rows_changed=%d "
        "bullets_changed=%d match=%s mode=%s (%s)",
        n_total, n_position, n_image_patch_rows, n_rows_changed,
        n_bullets_changed, args.match, args.mode,
        "DRY-RUN" if args.dry_run else "WROTE",
    )

    if not args.dry_run:
        tmp.replace(labels_path)
    elif tmp.exists():
        tmp.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())

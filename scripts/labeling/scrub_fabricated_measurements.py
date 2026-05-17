#!/usr/bin/env python
"""Scrub fabricated numeric measurements from labels.jsonl in place.

Background
----------
The labeler frequently invents specific physical measurements that it
cannot read from a JPEG: "about 5-8 cm diameter", "roughly 20-30 cm
from the gripper tip", "rotated 45 degrees".  The grader prompt in
``src/nla/labeling/grader.py`` explicitly forbids "precise numeric
measurements" under axis C, but a fraction of rows in the production
labels file still contain them.  AV will faithfully learn to fabricate centimeter
estimates if we leave them in.

This script does *not* call any API.  It reads ``--labels`` row by row
and rewrites every bullet that contains a measurement phrase, with two
operating modes:

``scrub`` (default)
    Excise just the measurement phrase from inside the bullet, then run
    a small punctuation-cleanup pass so the remaining sentence reads
    cleanly (no doubled commas, no orphaned " ()", no double spaces).
    Preserves the rest of the bullet.

``drop_bullet``
    Delete every bullet that contains a measurement phrase entirely.
    Blunt but maximally safe; use only if you don't trust the scrubber.

Backup
------
Unless ``--no-backup`` is passed the original ``labels.jsonl`` is copied
to ``labels.jsonl.bak``, ``.bak2``, ... before any rewrite.

Example
-------
::

    PYTHONPATH=src python scripts/labeling/scrub_fabricated_measurements.py \\
        --labels data/labels/libero_goal_pilot/labels.jsonl \\
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


# Hedging adverbs the labeler likes to wrap measurements with.
_ADV = r"(?:about|approximately|roughly|around|nearly|almost|some|maybe|perhaps)"
# A single number or a numeric range like "5", "5.5", "5-8", "5–8.5".
_NUM = r"\d+(?:\.\d+)?(?:\s*[-\u2013\u2212]\s*\d+(?:\.\d+)?)?"
# The unit names we treat as "measurement". Order matters only for clarity;
# Python's ``re`` honours word boundaries so "cm" wins over "m" naturally.
_UNIT = r"(?:mm|cm|m|in|inch(?:es)?|ft|feet|deg|degrees?|\u00b0|%)"

# A trailing dimension qualifier: "diameter", "wide", "tall", "from",
# "across", "in diameter", "in width" ...  Optional.
_QUAL = (
    r"(?:\s+(?:diameter|radius|wide|long|tall|high|thick|deep|across|"
    r"away|apart|higher|lower|above|below|behind|ahead|forward|"
    r"in\s+(?:diameter|width|height|length|radius)))?"
)

# Parenthesised forms: "(~6-8 cm)", "(about 5 cm wide)", "(20-30 cm higher)".
# We require the parens to *contain* a measurement and then eat the entire
# group greedily so we don't leave orphan suffixes like "(er)" behind when
# the qualifier inside is something we didn't enumerate.
_PAREN_MEAS = re.compile(
    rf"\s*\([^()]*?{_NUM}\s*{_UNIT}[^()]*?\)",
    re.IGNORECASE,
)

# Inline with hedging adverb: "about 5-8 cm diameter".
_ADV_MEAS = re.compile(
    rf",?\s*{_ADV}\s+[~\u2248]?\s*{_NUM}\s*{_UNIT}{_QUAL}",
    re.IGNORECASE,
)

# Bare number+unit, possibly preceded by ~/≈: "~5 cm wide", "5-8 cm diameter".
_BARE_MEAS = re.compile(
    rf",?\s*[~\u2248]?\s*{_NUM}\s*{_UNIT}{_QUAL}",
    re.IGNORECASE,
)

# Predicate used to decide whether the bullet contains *any* measurement.
# Use a non-word lookahead instead of ``\b`` so the unit terminator works
# for ``%``/``\u00b0`` (both non-word characters, so ``\b`` after them
# would only fire next to a word character).
_DETECT_MEAS = re.compile(rf"{_NUM}\s*{_UNIT}(?![A-Za-z0-9_])", re.IGNORECASE)

# Markdown-bullet line. Allows compound categories like ``gripper/spatial``,
# ``distractor,spatial``, and one-off shapes the labeler sometimes emits
# such as ``target (destination):``.  We accept any short category prefix
# that does not contain a colon.
_BULLET_RE = re.compile(r"^\s*-\s*[^:\n]{1,60}:")


def has_measurement(text: str) -> bool:
    """True if ``text`` contains a numeric measurement we want to scrub."""
    return bool(_DETECT_MEAS.search(text))


def scrub_phrase(text: str) -> str:
    """Strip every measurement phrase from ``text`` and tidy punctuation.

    Order matters: parenthesised forms are eaten first so we don't leave
    bare brackets behind, then adverb-led, then bare.
    """
    out = _PAREN_MEAS.sub("", text)
    out = _ADV_MEAS.sub("", out)
    out = _BARE_MEAS.sub("", out)

    # Tidy: collapse double spaces, fix " ,"/" ;"/" ." artefacts, drop
    # empty parens / brackets the scrubber may have left behind.
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\[\s*\]", "", out)
    out = re.sub(r"\s+([,;.])", r"\1", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def transform_description(
    description: str, *, drop_bullet: bool
) -> tuple[str, int]:
    """Apply scrubbing/dropping to every bullet in ``description``.

    Returns ``(new_description, n_bullets_modified)``.  The number counts
    bullets where either the text changed (scrub mode) or the bullet was
    dropped entirely (drop_bullet mode).
    """
    lines = description.splitlines()
    out_lines: list[str] = []
    n_changed = 0
    for ln in lines:
        if not _BULLET_RE.match(ln) or not has_measurement(ln):
            out_lines.append(ln)
            continue
        if drop_bullet:
            n_changed += 1
            continue
        scrubbed = scrub_phrase(ln)
        if scrubbed != ln:
            n_changed += 1
        out_lines.append(scrubbed)

    if n_changed == 0:
        return description, 0

    new_desc = "\n".join(out_lines)
    if description.endswith("\n") and not new_desc.endswith("\n"):
        new_desc += "\n"
    return new_desc, n_changed


def _pick_backup_path(labels_path: Path) -> Path:
    """First free `<labels>.bak`, `.bak2`, ... so we never overwrite."""
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
        "--mode",
        choices=("scrub", "drop_bullet"),
        default="scrub",
        help="scrub = excise the measurement phrase; drop_bullet = drop the whole bullet.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing <labels>.bak before rewriting (NOT recommended).",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Report counts only; do not write."
    )
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

                desc = obj.get("description") or ""
                new_desc, n_changed = transform_description(
                    desc, drop_bullet=(args.mode == "drop_bullet")
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
        "rows=%d position=%d rows_changed=%d bullets_changed=%d mode=%s (%s)",
        n_total,
        n_position,
        n_rows_changed,
        n_bullets_changed,
        args.mode,
        "DRY-RUN" if args.dry_run else "WROTE",
    )

    if not args.dry_run:
        tmp.replace(labels_path)
    elif tmp.exists():
        tmp.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())

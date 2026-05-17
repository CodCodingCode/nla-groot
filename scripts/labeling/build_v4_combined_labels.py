#!/usr/bin/env python
"""Build the V4 combined labels.jsonl by merging V4 per-suite re-labels into the V3 combined corpus.

For each suite (``spatial``, ``goal``, ``object``, ``10``) we:

1. Build a lookup of V4 re-labeled rows keyed by ``(source_example_id,
   position_index, position_type)`` from
   ``data/labels/libero_4suite_v4/libero_<suite>/labels.jsonl``.
2. Stream the V3 per-suite labels at
   ``data/labels/libero_4suite_stride2/libero_<suite>/labels.jsonl``.
3. For every V3 row, if the position key is present in the V4 lookup,
   emit the V4 row's caption/usage/metadata (with V4 prompt provenance
   preserved). Otherwise emit the V3 row unchanged.
4. Stamp ``meta.label_version`` on every emitted row (``"v4"`` for
   replaced, ``"v3"`` for kept). Stamp ``meta.suite`` matching the
   existing V3 combined convention (bare suite token: ``goal``,
   ``spatial``, ``object``, ``10`` — NOT ``libero_*``; see notes below).
5. Re-namespace ``example_id`` and ``source_example_id`` with the
   ``<suite>__`` prefix so the merged stream joins cleanly against the
   V3 combined activation index (which uses prefixed example_ids; the
   underlying activations are identical to V3 since V4 only changes
   captions).
6. Write the merged stream to
   ``data/labels/libero_4suite_v4_combined/labels.jsonl``.

The final row count is guaranteed to equal the V3 combined total
(101,580). Per-suite V3-kept vs V4-replaced counts are printed to
stdout and persisted alongside the labels file as ``_merge_summary.json``.

Notes
-----
The plan instructed ``meta.suite = "libero_{suite}"``. We deviate to
match the existing V3 combined convention (bare suite tokens). Reasons:

* ``data/labels/libero_4suite_combined/labels.jsonl`` (V3) already
  uses bare tokens (``goal``, ``spatial``, ...).
* The hard-negative auditor at
  ``scripts/eval/audit_hard_negatives.py`` parses the anchor id prefix
  (``goal__traj...``) and expects the in-meta ``suite`` field to be
  the same bare token; mixing ``libero_goal`` in meta with ``goal``
  in the example-id prefix would cause same-suite/cross-suite stats
  in the audit to silently disagree.
* The training-side code does not consume ``meta.suite`` at all, so
  the convention only matters for downstream eval/audit code.

The plan author's parenthetical "V4 rows already have it via SA2's
``suite`` field" is empirically incorrect — the V4 per-suite
``labels.jsonl`` files have no ``suite`` field in ``meta`` (verified
on disk before writing this script). We back-fill it on every row.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("build_v4_combined_labels")

SUITES = ("spatial", "goal", "object", "10")


def _prefix_id(suite: str, raw: str) -> str:
    head = f"{suite}__"
    if raw.startswith(head):
        return raw
    return f"{head}{raw}"


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    """Canonical position key: (source_example_id, position_index, position_type).

    Operates on per-suite (unprefixed) ``source_example_id`` so V3 and V4
    files can be joined directly without worrying about suite prefixes.
    """
    meta = row.get("meta") or {}
    src = meta.get("source_example_id")
    if not src:
        raise ValueError(f"row missing meta.source_example_id: {row.get('example_id')}")
    pos_idx = meta.get("position_index")
    if pos_idx is None:
        raise ValueError(f"row missing meta.position_index: {row.get('example_id')}")
    ptype = meta.get("position_type")
    if not ptype:
        raise ValueError(f"row missing meta.position_type: {row.get('example_id')}")
    return (str(src), int(pos_idx), str(ptype))


def _normalize_for_combined(
    row: dict[str, Any], suite: str, label_version: str
) -> dict[str, Any]:
    """Apply combined-corpus conventions: prefix ids + stamp meta.{suite,label_version}.

    Returns a NEW dict (does not mutate ``row``). Idempotent if the row
    is already prefixed.
    """
    out = dict(row)
    for k in ("example_id",):
        if k in out and isinstance(out[k], str):
            out[k] = _prefix_id(suite, out[k])
    meta = dict(out.get("meta") or {})
    for k in list(meta.keys()):
        if k.endswith("example_id") and isinstance(meta[k], str):
            meta[k] = _prefix_id(suite, meta[k])
    meta["suite"] = suite
    meta["label_version"] = label_version
    out["meta"] = meta
    return out


def _stream_jsonl(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_v4_lookup(v4_per_suite_root: Path, suite: str) -> dict[tuple[str, int, str], dict]:
    src = v4_per_suite_root / f"libero_{suite}" / "labels.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"missing V4 labels file: {src}")
    lookup: dict[tuple[str, int, str], dict] = {}
    for row in _stream_jsonl(src):
        key = _row_key(row)
        if key in lookup:
            logger.warning(
                "duplicate V4 row for key=%s in %s; keeping first", key, src
            )
            continue
        lookup[key] = row
    logger.info("  loaded %d V4 rows for suite=%s", len(lookup), suite)
    return lookup


def _build_one_suite(
    suite: str,
    v3_per_suite_root: Path,
    v4_per_suite_root: Path,
) -> tuple[list[dict], dict[str, int]]:
    """Returns (emitted_rows, stats) where stats has v3_kept / v4_replaced / total."""
    v3_src = v3_per_suite_root / f"libero_{suite}" / "labels.jsonl"
    if not v3_src.exists():
        raise FileNotFoundError(f"missing V3 labels file: {v3_src}")

    v4_lookup = _load_v4_lookup(v4_per_suite_root, suite)

    emitted: list[dict] = []
    v3_kept = 0
    v4_replaced = 0
    v4_keys_seen: set[tuple[str, int, str]] = set()
    for v3_row in _stream_jsonl(v3_src):
        key = _row_key(v3_row)
        if key in v4_lookup:
            emitted.append(
                _normalize_for_combined(v4_lookup[key], suite, label_version="v4")
            )
            v4_replaced += 1
            v4_keys_seen.add(key)
        else:
            emitted.append(
                _normalize_for_combined(v3_row, suite, label_version="v3")
            )
            v3_kept += 1

    # Sanity: every V4 row should have been consumed (no orphan V4 rows
    # without a V3 anchor). If anything is unmatched, surface it loudly.
    unmatched_v4 = set(v4_lookup.keys()) - v4_keys_seen
    if unmatched_v4:
        logger.error(
            "  suite=%s: %d V4 rows did not match any V3 anchor (e.g. %s)",
            suite, len(unmatched_v4), next(iter(unmatched_v4)),
        )

    stats = {
        "suite": suite,
        "v3_total": v3_kept + v4_replaced,
        "v3_kept": v3_kept,
        "v4_replaced": v4_replaced,
        "v4_available": len(v4_lookup),
        "v4_unmatched": len(unmatched_v4),
    }
    return emitted, stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--v3-per-suite-root", required=True,
                   help="Parent dir containing libero_<suite>/labels.jsonl per-suite V3 labels.")
    p.add_argument("--v4-per-suite-root", required=True,
                   help="Parent dir containing libero_<suite>/labels.jsonl per-suite V4 labels.")
    p.add_argument("--out", required=True,
                   help="Output path for the merged combined labels.jsonl.")
    p.add_argument("--summary-json", default=None,
                   help="Optional output path for the per-suite merge summary "
                        "(defaults to <out>/../_merge_summary.json).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    v3_root = Path(args.v3_per_suite_root)
    v4_root = Path(args.v4_per_suite_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json is not None
        else out_path.parent / "_merge_summary.json"
    )

    all_rows: list[dict] = []
    per_suite_stats: list[dict] = []
    for suite in SUITES:
        logger.info("Merging suite=%s", suite)
        rows, stats = _build_one_suite(suite, v3_root, v4_root)
        all_rows.extend(rows)
        per_suite_stats.append(stats)
        logger.info(
            "  suite=%s: v3_kept=%d v4_replaced=%d total=%d",
            suite, stats["v3_kept"], stats["v4_replaced"], stats["v3_total"],
        )

    with out_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d label rows to %s", len(all_rows), out_path)

    totals = {
        "total": len(all_rows),
        "v3_kept": sum(s["v3_kept"] for s in per_suite_stats),
        "v4_replaced": sum(s["v4_replaced"] for s in per_suite_stats),
        "v4_available": sum(s["v4_available"] for s in per_suite_stats),
        "v4_unmatched": sum(s["v4_unmatched"] for s in per_suite_stats),
    }
    summary = {
        "per_suite": per_suite_stats,
        "totals": totals,
        "out": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote merge summary to %s", summary_path)

    print()
    print("=" * 72)
    print("V4 combined merge summary")
    print("=" * 72)
    print(f"{'suite':<10} {'v3_total':>10} {'v3_kept':>10} {'v4_replaced':>14} {'v4_avail':>10}")
    for s in per_suite_stats:
        print(
            f"{s['suite']:<10} {s['v3_total']:>10} {s['v3_kept']:>10} "
            f"{s['v4_replaced']:>14} {s['v4_available']:>10}"
        )
    print("-" * 72)
    print(
        f"{'TOTAL':<10} {totals['total']:>10} {totals['v3_kept']:>10} "
        f"{totals['v4_replaced']:>14} {totals['v4_available']:>10}"
    )
    print("=" * 72)

    expected = 101_580
    if len(all_rows) != expected:
        logger.error(
            "row count mismatch: got %d, expected %d", len(all_rows), expected,
        )
        return 2
    if totals["v4_unmatched"] > 0:
        logger.warning(
            "%d V4 rows had no matching V3 anchor (likely a queue/V3 drift). "
            "These rows were NOT emitted.", totals["v4_unmatched"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

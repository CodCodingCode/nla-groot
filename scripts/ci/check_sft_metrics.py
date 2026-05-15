#!/usr/bin/env python
"""Local CI gate over an SFT ``metrics.jsonl``.

Reads the JSONL produced by ``nla.training.sft.run_sft`` (one row per logged
step plus periodic ``phase=="val"`` rows and a final ``phase=="final"`` row)
and exits non-zero on a small set of health checks aligned with
``docs/evals/v2_lessons_learned.md``:

  * **NCE alive** -- when ``ar_contrastive_weight > 0`` the training-row
    ``ar_nce`` value should sit *below* ``ln(batch_size)`` by some margin
    (uniform softmax collapses to ``ln(B)``; that's the V2 dead-NCE
    pathology). Skipped automatically when contrastive is off.

  * **Closed-loop present** (optional, gated by ``--require-closed-loop``)
    -- val/final rows should contain a ``closed_*/fve`` key. Catches runs
    that forgot ``--eval-closed-loop``.

  * **Teacher vs closed gap** (optional, gated by
    ``--max-tf-closed-fve-gap``) -- on the last val/final row, the
    aggregate teacher-forced ``fve`` should not exceed the
    ``closed_greedy/fve`` by more than the supplied threshold.

When ``config.json`` sits next to ``metrics.jsonl`` (the default
``run_sft`` layout) we auto-detect ``batch_size`` and
``ar_contrastive_weight`` from it; both can be overridden on the CLI.

Examples::

    # Minimal: NCE-alive check using sibling config.json.
    python scripts/ci/check_sft_metrics.py data/sft/my_run/metrics.jsonl

    # Add closed-loop presence + teacher/closed gap gates.
    python scripts/ci/check_sft_metrics.py data/sft/my_run/metrics.jsonl \\
        --require-closed-loop --max-tf-closed-fve-gap 0.05

    # Override config-derived values explicitly.
    python scripts/ci/check_sft_metrics.py path/to/metrics.jsonl \\
        --batch-size 4 --ar-contrastive-weight 0.5

Exit codes::

    0  all enabled checks pass (or were skipped because not applicable).
    1  at least one check fired; failure reasons printed to stderr.
    2  bad CLI / unreadable input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("metrics_jsonl", type=Path,
                   help="Path to a metrics.jsonl produced by run_sft.")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to the SFTConfig snapshot (config.json). "
                        "Defaults to metrics.jsonl's sibling config.json when present.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size for the ln(B) NCE check. "
                        "Defaults to the value in --config.")
    p.add_argument("--ar-contrastive-weight", type=float, default=None,
                   help="Override the contrastive weight read from --config. "
                        "If 0 (or unset and config has 0), the NCE-alive check is skipped.")
    p.add_argument("--nce-epsilon", type=float, default=0.05,
                   help="Margin under ln(B) the mean ar_nce must clear (default 0.05).")
    p.add_argument("--nce-tail-frac", type=float, default=0.5,
                   help="Fraction of the most recent train rows averaged for the NCE check "
                        "(default 0.5 = last half).")
    p.add_argument("--require-closed-loop", action="store_true",
                   help="Fail if no val/final row contains a closed_*/fve key.")
    p.add_argument("--max-tf-closed-fve-gap", type=float, default=None,
                   help="If set, fail when the last val/final row has "
                        "fve - closed_greedy/fve > GAP (skipped if either key missing).")
    return p


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[check_sft_metrics] {path}:{i} not valid JSON: {e}")
    return rows


def _maybe_load_config(metrics_path: Path, override: Path | None) -> dict[str, Any] | None:
    cfg_path = override if override is not None else metrics_path.with_name("config.json")
    if not cfg_path.exists():
        return None
    try:
        return json.loads(cfg_path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"[check_sft_metrics] {cfg_path} not valid JSON: {e}")


def _resolve_batch_and_contrast(args: argparse.Namespace,
                                cfg: dict[str, Any] | None) -> tuple[int | None, float]:
    """Returns (batch_size, ar_contrastive_weight). batch_size None == unknown."""
    bs = args.batch_size
    if bs is None and cfg is not None:
        bs = cfg.get("batch_size")
    cw = args.ar_contrastive_weight
    if cw is None and cfg is not None:
        cw = cfg.get("ar_contrastive_weight", 0.0)
    if cw is None:
        cw = 0.0
    return bs, float(cw)


def _check_nce_alive(train_rows: list[dict[str, Any]],
                     batch_size: int | None,
                     contrast_weight: float,
                     epsilon: float,
                     tail_frac: float) -> tuple[bool, str]:
    """Return (passed, message). 'passed' is True for skipped + ok."""
    if contrast_weight <= 0.0:
        return True, "skip nce-alive: ar_contrastive_weight=0"
    if not train_rows:
        return False, "fail nce-alive: no phase=='train' rows in metrics.jsonl"
    if batch_size is None or batch_size <= 1:
        return False, (f"fail nce-alive: need batch_size > 1 (got {batch_size}); "
                       "pass --batch-size or include config.json")
    nce_vals = [r["ar_nce"] for r in train_rows
                if isinstance(r.get("ar_nce"), (int, float))
                and math.isfinite(r["ar_nce"])]
    if not nce_vals:
        return False, "fail nce-alive: no finite ar_nce in train rows"
    n_tail = max(1, int(round(tail_frac * len(nce_vals))))
    tail = nce_vals[-n_tail:]
    mean_tail = sum(tail) / len(tail)
    bound = math.log(batch_size) - epsilon
    if mean_tail >= bound:
        return False, (f"fail nce-alive: mean ar_nce over last {n_tail} train rows "
                       f"= {mean_tail:.4f} >= ln(B={batch_size}) - eps "
                       f"= {bound:.4f}; InfoNCE looks dead (uniform-softmax pathology)")
    return True, (f"ok   nce-alive: mean ar_nce over last {n_tail} train rows "
                  f"= {mean_tail:.4f} < {bound:.4f}")


def _last_eval_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eval_rows = [r for r in rows if r.get("phase") in ("val", "final")]
    return eval_rows[-1] if eval_rows else None


def _check_closed_loop_present(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    for r in rows:
        if r.get("phase") not in ("val", "final"):
            continue
        if any(k.startswith("closed_") and k.endswith("/fve") for k in r):
            return True, "ok   closed-loop-present: at least one val/final row has closed_*/fve"
    return False, ("fail closed-loop-present: no val/final row contained a closed_*/fve "
                   "key (did you forget --eval-closed-loop on run_sft.py?)")


def _check_tf_closed_gap(rows: list[dict[str, Any]],
                         max_gap: float) -> tuple[bool, str]:
    last = _last_eval_row(rows)
    if last is None:
        return False, "fail tf-vs-closed-gap: no val/final row found"
    tf = last.get("fve")
    cl = last.get("closed_greedy/fve")
    if not isinstance(tf, (int, float)) or not isinstance(cl, (int, float)):
        return True, ("skip tf-vs-closed-gap: last eval row missing fve "
                      f"({tf!r}) or closed_greedy/fve ({cl!r})")
    gap = tf - cl
    if gap > max_gap:
        return False, (f"fail tf-vs-closed-gap: fve - closed_greedy/fve "
                       f"= {tf:.4f} - {cl:.4f} = {gap:+.4f} > {max_gap:.4f}")
    return True, (f"ok   tf-vs-closed-gap: fve - closed_greedy/fve = {gap:+.4f} "
                  f"<= {max_gap:.4f}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.metrics_jsonl.exists():
        print(f"[check_sft_metrics] {args.metrics_jsonl} not found", file=sys.stderr)
        return 2

    rows = _read_jsonl(args.metrics_jsonl)
    if not rows:
        print(f"[check_sft_metrics] {args.metrics_jsonl} is empty", file=sys.stderr)
        return 2

    cfg = _maybe_load_config(args.metrics_jsonl, args.config)
    batch_size, contrast_weight = _resolve_batch_and_contrast(args, cfg)
    train_rows = [r for r in rows if r.get("phase") == "train"]

    results: list[tuple[str, bool, str]] = []
    results.append(("nce-alive",
                    *_check_nce_alive(train_rows, batch_size, contrast_weight,
                                      args.nce_epsilon, args.nce_tail_frac)))
    if args.require_closed_loop:
        results.append(("closed-loop-present", *_check_closed_loop_present(rows)))
    if args.max_tf_closed_fve_gap is not None:
        results.append(("tf-vs-closed-gap",
                        *_check_tf_closed_gap(rows, args.max_tf_closed_fve_gap)))

    failed = False
    for _name, ok, msg in results:
        stream = sys.stdout if ok else sys.stderr
        print(msg, file=stream)
        if not ok:
            failed = True

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Build per-row SFT quality weights from an LLM-judge JSONL.

This is the V5 "turn on quality weights" prerequisite called out in
``docs/sft_plan/SFT_V5_NEXT.md`` and the GRPO V2 plan. It joins judge
verdicts produced by ``scripts/eval/llm_judge_av_captions.py`` (or any
``nla.labeling.grader.grade_many_async`` output) to the original SFT
``labels.jsonl`` and writes a patched labels JSONL with two new fields
on every row:

- ``quality_weight``: float in ``[0, 1]`` consumed directly by
  ``nla.training.dataset._extract_quality_weight``.
- ``quality_axes``: per-axis ``[0, 1]`` breakdown
  (``grounding``, ``appropriateness``, ``template_distinguishable``) so
  downstream analyses can see which axis dragged the score.

Workflow::

  python scripts/eval/llm_judge_av_captions.py \
      --labels-jsonl data/labels/libero_4suite_v5_combined/labels.jsonl \
      --out-jsonl    data/labels/libero_4suite_v5_combined/judge.jsonl \
      --max-rollouts 4096

  python scripts/training/build_quality_weights.py \
      --labels-jsonl data/labels/libero_4suite_v5_combined/labels.jsonl \
      --judge-jsonl  data/labels/libero_4suite_v5_combined/judge.jsonl \
      --output-jsonl data/labels/libero_4suite_v5_combined/labels_weighted.jsonl

  python scripts/training/run_sft.py \
      --labels-jsonl data/labels/libero_4suite_v5_combined/labels_weighted.jsonl \
      --use-quality-weights ...

Design notes
------------

- The judge JSONL contains two ``variant_id`` rows per ``example_id``
  (``"gold"`` = the SFT label itself, ``"av_pred"`` = a model-generated
  caption). We weight by the ``"gold"`` row's verdicts because the SFT
  loss is on the gold description; ``av_pred`` rows are kept for
  downstream AV diagnostics but **ignored** here.
- ``example_id`` in the judge JSONL follows the
  ``f"{source_example_id}@p{position_index}_{position_type}"`` format
  used by ``llm_judge_av_captions.py:250``. We parse that back into the
  three-tuple key.
- Each axis contributes one of three weights per the SFT V5 plan
  (``--specific-weight`` 1.0 / ``--unjudged-weight`` 0.5 /
  ``--generic-weight`` 0.1). The composite weight is the mean of the
  active axes; missing axes are skipped (no penalty for graders that
  only return some axes).
- Rows that have no matching judge entry fall back to the suite mean
  ``--unjudged-weight`` (default 0.5) so unjudged rows still contribute
  to SFT without dominating the gradient.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXAMPLE_ID_RE = re.compile(
    r"^(?P<source>.+)@p(?P<pos_idx>-?\d+)_(?P<pos_type>[A-Za-z0-9_]+)$"
)


def _parse_example_id(example_id: str) -> tuple[str, int, str] | None:
    """Parse ``llm_judge_av_captions.py`` composite ids.

    Returns ``(source_example_id, position_index, position_type)`` or
    ``None`` when the string does not match the expected format.
    """
    m = EXAMPLE_ID_RE.match(example_id.strip())
    if m is None:
        return None
    return m.group("source"), int(m.group("pos_idx")), m.group("pos_type")


def _verdict_to_weight(
    verdict: str | None,
    *,
    positive: str,
    specific_w: float,
    generic_w: float,
) -> float | None:
    """Map one axis verdict string to a numeric weight.

    Returns ``None`` when the verdict is missing or unrecognized so the
    caller can decide whether to use a per-axis fallback or skip the
    axis entirely.
    """
    if not verdict:
        return None
    v = verdict.strip().lower()
    if v == positive:
        return specific_w
    return generic_w


def _row_weight(
    judge_row: dict[str, Any],
    *,
    specific_w: float,
    generic_w: float,
) -> tuple[float | None, dict[str, float]]:
    """Compute the composite quality weight for one judge row.

    Returns ``(weight, axes)`` where ``weight`` is the mean of the
    active per-axis weights (``None`` when no axis has a parseable
    verdict) and ``axes`` is the per-axis breakdown for the patched
    label row.
    """
    axes: dict[str, float] = {}
    for axis_key, positive_label in (
        ("grounding", "specific"),
        ("appropriateness", "appropriate"),
        ("template_distinguishable", "specific"),
    ):
        axis = judge_row.get(axis_key) or {}
        w = _verdict_to_weight(
            axis.get("verdict") if isinstance(axis, dict) else None,
            positive=positive_label,
            specific_w=specific_w,
            generic_w=generic_w,
        )
        if w is not None:
            axes[axis_key] = w
    if not axes:
        return None, axes
    return sum(axes.values()) / len(axes), axes


def load_judge_jsonl(
    paths: list[Path],
    *,
    variant_id: str,
    specific_w: float,
    generic_w: float,
) -> tuple[dict[tuple[str, int, str], tuple[float, dict[str, float]]], dict[str, int]]:
    """Read one or more judge JSONLs into a ``key -> (weight, axes)`` dict.

    Only rows whose ``variant_id`` matches ``variant_id`` are kept; the
    most-recent file's verdict wins on duplicate keys (callers usually
    pass at most one judge file).
    """
    out: dict[tuple[str, int, str], tuple[float, dict[str, float]]] = {}
    stats: dict[str, int] = Counter()
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                if row.get("variant_id") != variant_id:
                    stats["wrong_variant"] += 1
                    continue
                if row.get("error"):
                    stats["judge_error"] += 1
                    continue
                key = _parse_example_id(row.get("example_id") or "")
                if key is None:
                    stats["unparseable_example_id"] += 1
                    continue
                weight, axes = _row_weight(
                    row, specific_w=specific_w, generic_w=generic_w,
                )
                if weight is None:
                    stats["no_active_axes"] += 1
                    continue
                out[key] = (weight, axes)
                stats["graded"] += 1
    return out, dict(stats)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--labels-jsonl",
        required=True,
        help="Original SFT labels.jsonl (rows must have meta.source_example_id, "
             "meta.position_index, meta.position_type).",
    )
    p.add_argument(
        "--judge-jsonl",
        action="append",
        required=True,
        help="One or more judge JSONLs produced by "
             "scripts/eval/llm_judge_av_captions.py. Pass --judge-jsonl "
             "multiple times to merge several runs (last entry wins).",
    )
    p.add_argument(
        "--output-jsonl",
        required=True,
        help="Output path. Same shape as --labels-jsonl with `quality_weight` "
             "and `quality_axes` added to every row.",
    )
    p.add_argument(
        "--variant-id",
        default="gold",
        choices=["gold", "av_pred"],
        help="Which judge variant to score against. 'gold' weights the SFT "
             "label itself (default); 'av_pred' weights model outputs and is "
             "mainly useful for ablations.",
    )
    p.add_argument(
        "--specific-weight",
        type=float,
        default=1.0,
        help="Per-axis weight when the verdict is the positive class "
             "(grounding=specific, appropriateness=appropriate, "
             "template_distinguishable=specific). Default 1.0.",
    )
    p.add_argument(
        "--generic-weight",
        type=float,
        default=0.1,
        help="Per-axis weight when the verdict is the negative class. "
             "Default 0.1 (per docs/sft_plan/SFT_V5_NEXT.md §3).",
    )
    p.add_argument(
        "--unjudged-weight",
        type=float,
        default=0.5,
        help="Composite weight assigned to label rows that have no "
             "matching judge entry, before the per-(suite, position_type) "
             "mean fallback. Default 0.5.",
    )
    p.add_argument(
        "--fallback-mode",
        default="ptype_mean",
        choices=["ptype_mean", "global_mean", "constant"],
        help="How to score label rows the judge did not cover. "
             "'ptype_mean' (default) uses the mean of graded rows in the "
             "same position_type; 'global_mean' uses the overall mean of "
             "graded rows; 'constant' just emits --unjudged-weight.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute coverage stats and per-position_type means but do "
             "not write --output-jsonl.",
    )
    args = p.parse_args()

    labels_path = Path(args.labels_jsonl)
    out_path = Path(args.output_jsonl)
    judge_paths = [Path(p) for p in args.judge_jsonl]
    for jp in judge_paths:
        if not jp.exists():
            print(f"error: judge jsonl not found: {jp}", file=sys.stderr)
            return 2
    if not labels_path.exists():
        print(f"error: labels jsonl not found: {labels_path}", file=sys.stderr)
        return 2

    judged, judge_stats = load_judge_jsonl(
        judge_paths,
        variant_id=args.variant_id,
        specific_w=args.specific_weight,
        generic_w=args.generic_weight,
    )
    print(
        f"[judge] read {sum(judge_stats.values())} rows from "
        f"{len(judge_paths)} file(s); kept {len(judged)} unique "
        f"(source_id, pos_idx, pos_type) keys. stats={judge_stats}",
        file=sys.stderr,
    )

    # Pass 1: read labels, look up judge weight, compute ptype/global means.
    rows: list[dict[str, Any]] = []
    sum_by_ptype: dict[str, float] = defaultdict(float)
    cnt_by_ptype: dict[str, int] = defaultdict(int)
    sum_global = 0.0
    cnt_global = 0
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = obj.get("meta") or {}
            src = meta.get("source_example_id")
            pos_idx = meta.get("position_index")
            pos_type = meta.get("position_type")
            judged_pair = None
            if src is not None and pos_idx is not None and pos_type is not None:
                judged_pair = judged.get((str(src), int(pos_idx), str(pos_type)))
            obj["_pos_type"] = str(pos_type) if pos_type is not None else "_unk"
            obj["_judged"] = judged_pair  # tuple or None
            if judged_pair is not None:
                w = judged_pair[0]
                sum_by_ptype[obj["_pos_type"]] += w
                cnt_by_ptype[obj["_pos_type"]] += 1
                sum_global += w
                cnt_global += 1
            rows.append(obj)

    ptype_mean: dict[str, float] = {
        k: sum_by_ptype[k] / cnt_by_ptype[k] for k in cnt_by_ptype
    }
    global_mean = sum_global / cnt_global if cnt_global else args.unjudged_weight

    print(
        f"[labels] loaded {len(rows)} rows; "
        f"{cnt_global} judged ({cnt_global / max(len(rows), 1):.1%}). "
        f"global_mean={global_mean:.3f} ptype_mean={ {k: round(v, 3) for k, v in ptype_mean.items()} }",
        file=sys.stderr,
    )

    # Pass 2: emit patched rows.
    n_judged = 0
    n_fallback = 0
    if args.dry_run:
        print("[dry-run] skipping write", file=sys.stderr)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fp = None if args.dry_run else out_path.open("w")
    try:
        for obj in rows:
            judged_pair = obj.pop("_judged", None)
            ptype = obj.pop("_pos_type", "_unk")
            if judged_pair is not None:
                weight, axes = judged_pair
                obj["quality_weight"] = float(weight)
                obj["quality_axes"] = {k: float(v) for k, v in axes.items()}
                n_judged += 1
            else:
                if args.fallback_mode == "ptype_mean":
                    fb = ptype_mean.get(ptype, global_mean)
                elif args.fallback_mode == "global_mean":
                    fb = global_mean
                else:
                    fb = args.unjudged_weight
                obj["quality_weight"] = float(fb)
                obj["quality_axes"] = {"fallback": float(fb)}
                n_fallback += 1
            if out_fp is not None:
                out_fp.write(json.dumps(obj) + "\n")
    finally:
        if out_fp is not None:
            out_fp.close()

    print(
        f"[output] judged={n_judged} fallback={n_fallback} -> "
        f"{out_path if not args.dry_run else '(dry-run)'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

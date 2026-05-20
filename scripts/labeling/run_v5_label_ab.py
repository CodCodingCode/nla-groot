#!/usr/bin/env python
"""Run V5 nested JSON labeling A/B (temperature sweep).

Labels a step-deduped eval set with ``V5_nested_T0``, ``V5_nested_T07``, and
``V5_nested_T10``, then writes ``scores.json`` with V5 granularity metrics.

Example::

    PYTHONPATH=src python scripts/labeling/run_v5_label_ab.py \\
        --eval-set data/prompt_ab/v5_step_eval_set.jsonl \\
        --out-dir data/prompt_ab/v5_round_01 \\
        --max-steps 5

    # Full eval (~150 steps x 3 variants) requires OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("nla.v5_label_ab")

V5_VARIANTS = ("V5_nested_T0", "V5_nested_T07", "V5_nested_T10")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--eval-set",
        default="data/prompt_ab/v5_step_eval_set.jsonl",
    )
    p.add_argument(
        "--out-dir",
        default="data/prompt_ab/v5_round_01",
        type=Path,
    )
    p.add_argument(
        "--variants",
        default=",".join(V5_VARIANTS),
        help="Comma-separated variant ids (default: all three temps).",
    )
    p.add_argument(
        "--label-model",
        default=os.environ.get("OPENAI_LABELING_MODEL", "gpt-5.1-mini"),
    )
    p.add_argument("--label-concurrency", type=int, default=8)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap unique steps (after dedupe) for pilots.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def load_step_eval_set(path: Path, *, max_steps: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        eid = row.get("example_id") or row.get("eval_id")
        if eid in seen:
            continue
        seen.add(eid)
        deduped.append(row)
        if max_steps is not None and len(deduped) >= max_steps:
            break
    return deduped


def step_row_to_position_input(row: dict):
    from nla.labeling.prompts import PositionLabelInput

    return PositionLabelInput(
        example_id=row["eval_id"],
        instruction=row.get("instruction") or "",
        decoded_text_context=row.get("decoded_text_context") or "",
        position_index=int(row.get("position_index", 0)),
        position_type=row.get("position_type", "last_text"),  # type: ignore[arg-type]
        sequence_length=int(row.get("sequence_length", 0)),
        image_paths=list(row["image_paths"]),
        episode_index=row.get("episode_index"),
        step_index=row.get("step_index"),
        suite=row.get("suite"),
        extra={
            "source": row.get("source"),
            "source_example_id": row.get("example_id"),
        },
    )


def step_rows_to_eval_rows(step_rows: list[dict]):
    """Adapt step eval dicts to :class:`EvalRow` for shared AB helpers."""
    from nla.labeling.ab_test import EvalRow

    out: list[EvalRow] = []
    for row in step_rows:
        out.append(
            EvalRow(
                eval_id=row["eval_id"],
                source=row.get("source", ""),
                example_id=row["example_id"],
                instruction=row.get("instruction") or "",
                decoded_text_context=row.get("decoded_text_context") or "",
                position_index=int(row.get("position_index", 0)),
                position_type=row.get("position_type", "step"),
                sequence_length=int(row.get("sequence_length", 0)),
                image_patch_meta=None,
                image_paths=list(row["image_paths"]),
                episode_index=row.get("episode_index"),
                step_index=row.get("step_index"),
            )
        )
    return out


async def _label_variant_step_eval(
    variant_id: str,
    step_rows: list[dict],
    labels_jsonl: Path,
    *,
    model: str,
    concurrency: int,
    api_key: str | None,
) -> int:
    from openai import AsyncOpenAI

    from nla.labeling.ab_test import _label_one_variant_async
    from nla.labeling.prompt_variants import get_variant

    variant_fn = get_variant(variant_id)
    labels_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if labels_jsonl.exists():
        with labels_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("raw_response") and not obj.get("error"):
                    done_ids.add(obj["example_id"])

    todo: list[tuple] = []
    for row in step_rows:
        if row["eval_id"] in done_ids:
            continue
        inp = step_row_to_position_input(row)
        vo = variant_fn(inp)
        todo.append((vo, inp))

    logger.info("[%s] %d new labels -> %s", variant_id, len(todo), labels_jsonl)
    if not todo:
        return 0

    client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    n_new = 0
    with labels_jsonl.open("a") as f:
        async def run_one(vo, inp):
            nonlocal n_new
            from dataclasses import asdict

            res = await _label_one_variant_async(
                client,
                variant_id,
                vo,
                inp,
                model,
                sem,
                max_retries=4,
                base_backoff=1.0,
            )
            f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            return res

        await asyncio.gather(*(run_one(vo, inp) for vo, inp in todo))
    await client.close()
    return n_new


async def run_v5_ab_async(
    *,
    eval_set: Path,
    out_dir: Path,
    variants: list[str],
    model: str,
    concurrency: int,
    max_steps: int | None,
    api_key: str | None,
) -> dict:
    step_rows = load_step_eval_set(eval_set, max_steps=max_steps)
    logger.info("loaded %d step-deduped eval rows from %s", len(step_rows), eval_set)

    out_dir.mkdir(parents=True, exist_ok=True)
    label_paths: dict[str, Path] = {}

    for variant in variants:
        variant_dir = out_dir / f"variant_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        labels_jsonl = variant_dir / "labels.jsonl"
        label_paths[variant] = labels_jsonl
        await _label_variant_step_eval(
            variant,
            step_rows,
            labels_jsonl,
            model=model,
            concurrency=concurrency,
            api_key=api_key,
        )

    from nla.labeling.v5_ab_metrics import aggregate_v5_ab

    scores = aggregate_v5_ab({v: str(p) for v, p in label_paths.items()})
    scores_path = out_dir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    logger.info("wrote %s", scores_path)
    return scores


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    eval_set = Path(args.eval_set)
    if not eval_set.is_file():
        logger.error("eval set not found: %s", eval_set)
        return 1

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        asyncio.run(
            run_v5_ab_async(
                eval_set=eval_set,
                out_dir=args.out_dir,
                variants=variants,
                model=args.label_model,
                concurrency=args.label_concurrency,
                max_steps=args.max_steps,
                api_key=api_key,
            )
        )
    else:
        logger.warning("OPENAI_API_KEY not set — skipping API labeling pilot")
        if args.max_steps:
            rows = load_step_eval_set(eval_set, max_steps=args.max_steps)
            logger.info("would label %d steps x %d variants", len(rows), len(variants))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

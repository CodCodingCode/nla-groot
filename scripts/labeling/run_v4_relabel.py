#!/usr/bin/env python
"""Drive the V4 selective re-label for one LIBERO suite.

Reads a queue JSONL produced by ``build_v4_relabel_queue.py`` and re-labels
only the listed (source_example_id, position_index, position_type) tuples
through the V4 prompt builder.  Built on top of SA5's pipeline wiring:
sets ``NLA_POSITION_PROMPT_MODE=v4`` before importing
``nla.labeling.openai_client`` so ``_select_position_builder`` dispatches to
``build_v4_position_prompt``.  ``--suite`` is stamped onto every constructed
``PositionLabelInput`` so the V4 per-suite addendum activates without
relying on example-id prefixes.

Resume support is inherited from ``openai_client.label_many_async`` (matches
on the canonical ``(source_example_id, position_index, position_type)``
key); per-completion cost rows are streamed to ``<out-dir>/_cost_log.jsonl``.

Example::

    PYTHONPATH=src python scripts/labeling/run_v4_relabel.py \\
        --queue-jsonl       data/labels/v4_relabel_queue/libero_goal.jsonl \\
        --activations-root  data/activations/libero_4suite_stride2/libero_goal \\
        --dataset-root      third_party/Isaac-GR00T/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot \\
        --suite             libero_goal \\
        --out-dir           data/labels/libero_4suite_v4/libero_goal \\
        --concurrency       32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# Per-row labeling cost assumption (matches build_v4_relabel_queue.py).
COST_PER_ROW_USD: float = 0.0007


def _read_queue(path: Path, *, max_rows: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("source_example_id") is None:
                continue
            if obj.get("position_index") is None:
                continue
            if obj.get("position_type") is None:
                continue
            rows.append(obj)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _read_done_position_keys(out_jsonl: Path) -> set[tuple[str, int, str]]:
    """Replicate openai_client._position_resume_key_from_row over the existing
    output so the input-builder can short-circuit before paying the
    frame-extraction cost on already-done rows."""
    done: set[tuple[str, int, str]] = set()
    if not out_jsonl.exists():
        return done
    with out_jsonl.open() as f:
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
            if not obj.get("description") or obj.get("error"):
                continue
            m = obj.get("meta") or {}
            sid = m.get("source_example_id")
            pidx = m.get("position_index")
            pt = m.get("position_type")
            if sid is None or pidx is None or pt is None:
                continue
            done.add((str(sid), int(pidx), str(pt)))
    return done


def _build_inputs(
    queue_rows: list[dict],
    *,
    activations_root: Path,
    dataset_root: Path,
    frame_cache_dir: Path,
    suite: str,
    tokenizer_repo: str,
    state_name: str | None,
    already_done: set[tuple[str, int, str]],
):
    """Construct PositionLabelInput objects for queued rows that aren't done."""
    from nla.extraction.storage import ActivationShardReader  # noqa: E402
    from nla.labeling.context import (  # noqa: E402
        FrameLoaderPool,
        SampledExample,
        build_position_inputs,
        decode_text_context,
        image_patch_meta,
        load_qwen3_vl_tokenizer,
    )

    by_sid: dict[str, list[tuple[int, str]]] = {}
    n_skipped_done = 0
    for r in queue_rows:
        key = (
            str(r["source_example_id"]),
            int(r["position_index"]),
            str(r["position_type"]),
        )
        if key in already_done:
            n_skipped_done += 1
            continue
        by_sid.setdefault(key[0], []).append((key[1], key[2]))
    logging.info(
        "queue: %d total, %d already-done (resume), %d examples to load",
        len(queue_rows), n_skipped_done, len(by_sid),
    )

    if not by_sid:
        return []

    reader = ActivationShardReader(activations_root)
    tokenizer = load_qwen3_vl_tokenizer(tokenizer_repo)
    pool = FrameLoaderPool(max_open=8)

    sampled: list[SampledExample] = []
    n_missing_inputs = 0
    n_unknown_examples = set(by_sid.keys())
    for item in reader.iter_examples(
        record_filter=lambda rec: rec.example_id in by_sid,
    ):
        rec = item["_record"]
        n_unknown_examples.discard(rec.example_id)
        img = item["image_mask"]
        ids = item.get("input_ids")
        if ids is None:
            n_missing_inputs += 1
            logging.warning(
                "example %s missing input_ids; skipping", rec.example_id,
            )
            continue
        text_ctx = decode_text_context(ids, img, tokenizer)
        for pidx, pt in by_sid[rec.example_id]:
            meta = image_patch_meta(img, pidx)
            sampled.append(
                SampledExample(
                    record=rec,
                    position_index=pidx,
                    position_type=pt,
                    decoded_text_context=text_ctx,
                    image_patch_meta=meta,
                )
            )
    if n_unknown_examples:
        logging.warning(
            "%d queued source_example_ids not found in activation index "
            "(first 5: %s)",
            len(n_unknown_examples), sorted(n_unknown_examples)[:5],
        )

    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    inputs = list(
        build_position_inputs(
            sampled,
            dataset_root=dataset_root,
            frame_cache_dir=frame_cache_dir,
            state_name=state_name,
            pool=pool,
            suite=suite,
        )
    )
    pool.close_all()
    logging.info(
        "built %d PositionLabelInputs (n_missing_inputs=%d)",
        len(inputs), n_missing_inputs,
    )
    return inputs


async def _drive_relabel(
    inputs: Iterable,
    out_jsonl: Path,
    cost_log_path: Path,
    *,
    model: str,
    concurrency: int,
    api_key: str | None,
) -> tuple[int, int]:
    """Concurrent re-label runner with cost logging.

    Mirrors ``openai_client.label_many_async`` resume + dispatch but writes
    one row per completion to ``cost_log_path`` for budget tracking.

    Returns ``(n_new_completed, n_errors)``.
    """
    from nla.labeling.openai_client import (  # noqa: E402
        _label_one_async,
        _position_resume_key_from_input,
        _position_resume_key_from_row,
        _get_openai,
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    done_pos_keys: set[tuple[str, int, str]] = set()
    if out_jsonl.exists():
        with out_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("description") and not obj.get("error"):
                    done_ids.add(obj["example_id"])
                    pk = _position_resume_key_from_row(obj)
                    if pk is not None:
                        done_pos_keys.add(pk)
    logging.info("resume: %d example_ids done, %d position keys done",
                 len(done_ids), len(done_pos_keys))

    todo = []
    for i in inputs:
        if i.example_id in done_ids:
            continue
        pk = _position_resume_key_from_input(i)
        if pk is not None and pk in done_pos_keys:
            continue
        todo.append(i)
    logging.info("Labeling: %d new -> %s", len(todo), out_jsonl)

    if not todo:
        return 0, 0

    _, AsyncOpenAI = _get_openai()
    client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)

    n_new = 0
    n_errors = 0
    cumulative_cost = 0.0
    t_start = time.time()
    f_out = out_jsonl.open("a")
    f_cost = cost_log_path.open("a")

    try:
        async def run_one(inp):
            nonlocal n_new, n_errors, cumulative_cost
            res = await _label_one_async(
                client, inp, model, sem, max_retries=4, base_backoff=1.0,
            )
            row = asdict(res)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()
            n_new += 1
            if res.error:
                n_errors += 1
            else:
                cumulative_cost += COST_PER_ROW_USD

            usage = res.usage or {}
            cost_row = {
                "example_id": res.example_id,
                "kind": res.kind,
                "model": res.model,
                "elapsed_ms": res.elapsed_ms,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "error": res.error,
                "cumulative_cost_usd": round(cumulative_cost, 4),
            }
            f_cost.write(json.dumps(cost_row, ensure_ascii=False) + "\n")
            f_cost.flush()

            if n_new % 100 == 0:
                rate = n_new / max(time.time() - t_start, 1e-6)
                eta_s = max(0.0, (len(todo) - n_new) / max(rate, 1e-6))
                logging.info(
                    "  %d / %d labeled (%.1f rps, ~%dm to go, "
                    "errors=%d, ~$%.2f)",
                    n_new, len(todo), rate, int(eta_s // 60),
                    n_errors, cumulative_cost,
                )
            return res

        await asyncio.gather(*(run_one(i) for i in todo))
    finally:
        f_out.close()
        f_cost.close()
        await client.close()

    return n_new, n_errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--queue-jsonl", required=True, type=Path)
    p.add_argument("--activations-root", required=True, type=Path)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--suite", required=True,
                   help="LIBERO suite tag (e.g. libero_spatial). "
                        "Threaded onto every PositionLabelInput.suite.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap queue rows (debug / smoke).")
    p.add_argument("--model", default=None,
                   help="OpenAI labeling model. "
                        "Default: env OPENAI_LABELING_MODEL or "
                        "openai_client.DEFAULT_MODEL.")
    p.add_argument("--tokenizer-repo", default=None)
    p.add_argument("--state-name", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--frames-cache", type=Path, default=None,
                   help="Default: V3 frames_cache directory next to the "
                        "activations (data/labels/.../<suite>/frames_cache).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # IMPORTANT: set the V4 dispatch flag BEFORE importing openai_client
    # (which reads NLA_POSITION_PROMPT_MODE at call time but caches the
    # default at import).
    os.environ["NLA_POSITION_PROMPT_MODE"] = "v4"
    logging.info("Set NLA_POSITION_PROMPT_MODE=v4 (V4 prompt builder)")

    # Import after env flag is set.
    from nla.labeling import openai_client  # noqa: E402, F401
    from nla.labeling.context import DEFAULT_TOKENIZER_REPO  # noqa: E402

    if args.tokenizer_repo is None:
        args.tokenizer_repo = DEFAULT_TOKENIZER_REPO
    if args.model is None:
        args.model = openai_client.DEFAULT_MODEL

    # Frames cache: prefer the existing V3 cache (per the plan, V4 doesn't
    # need to re-extract frames).
    if args.frames_cache is None:
        candidate = (
            Path("data/labels/libero_4suite_stride2") / args.suite / "frames_cache"
        )
        if candidate.is_dir():
            args.frames_cache = candidate
        else:
            args.frames_cache = args.out_dir / "frames_cache"
    logging.info("frames_cache: %s", args.frames_cache)

    out_jsonl = args.out_dir / "labels.jsonl"
    cost_log_path = args.out_dir / "_cost_log.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    queue = _read_queue(args.queue_jsonl, max_rows=args.max_rows)
    logging.info("queue %s -> %d rows (suite=%s)",
                 args.queue_jsonl, len(queue), args.suite)
    if not queue:
        logging.warning("empty queue; nothing to do")
        return 0

    already_done = _read_done_position_keys(out_jsonl)
    logging.info("found %d already-completed rows in %s",
                 len(already_done), out_jsonl)

    inputs = _build_inputs(
        queue,
        activations_root=args.activations_root,
        dataset_root=args.dataset_root,
        frame_cache_dir=args.frames_cache,
        suite=args.suite,
        tokenizer_repo=args.tokenizer_repo,
        state_name=args.state_name,
        already_done=already_done,
    )

    if not inputs:
        logging.info("no new inputs to label after resume; exiting")
        return 0

    # Stamp run config + suite + prompt_mode so downstream readers can
    # distinguish V3 / V4 corpora without re-reading every bullet.
    manifest = {
        "queue_jsonl": str(args.queue_jsonl),
        "activations_root": str(args.activations_root),
        "dataset_root": str(args.dataset_root),
        "suite": args.suite,
        "out_dir": str(args.out_dir),
        "model": args.model,
        "tokenizer_repo": args.tokenizer_repo,
        "concurrency": args.concurrency,
        "max_rows": args.max_rows,
        "frames_cache": str(args.frames_cache),
        "n_queue_rows": len(queue),
        "n_inputs": len(inputs),
        "extra": {"prompt_mode": "v4"},
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    n_new, n_errors = asyncio.run(
        _drive_relabel(
            inputs,
            out_jsonl=out_jsonl,
            cost_log_path=cost_log_path,
            model=args.model,
            concurrency=args.concurrency,
            api_key=args.api_key,
        )
    )
    logging.info(
        "labeled %d new rows -> %s (errors=%d)",
        n_new, out_jsonl, n_errors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

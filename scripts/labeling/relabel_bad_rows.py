#!/usr/bin/env python
"""Re-label labels.jsonl rows that violate the bullet schema.

Targets rows that:
  - are missing the ``scene:`` bullet
  - are missing the ``target:`` bullet (background image_patch positions)
  - used a non-canonical bullet category (e.g. ``tool``, ``secondary_target``)

Re-runs the labeler API for only those rows using
``build_strict_position_prompt`` (allows ``target: none in this patch.`` and
forbids inventing categories).  Writes new rows to a resumable sidecar JSONL.

Example::

    python scripts/labeling/relabel_bad_rows.py \\
        --labels      data/labels/libero_goal_pilot/labels.jsonl \\
        --activations data/activations/libero_goal_pilot \\
        --sidecar     data/labels/libero_goal_pilot/labels.relabel.jsonl \\
        --in-place

If you already ran the API pass and just want to re-merge::

    python scripts/labeling/relabel_bad_rows.py \\
        --labels  data/labels/libero_goal_pilot/labels.jsonl \\
        --activations data/activations/libero_goal_pilot \\
        --sidecar data/labels/libero_goal_pilot/labels.relabel.jsonl \\
        --merge-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.extraction.storage import ActivationShardReader, RunManifest  # noqa: E402
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
from nla.labeling.prompts import (  # noqa: E402
    BULLET_CATEGORIES,
    build_strict_position_prompt,
)


KNOWN_CATS = set(BULLET_CATEGORIES) | {"anchor"}
BULLET_RE = re.compile(r"^\s*-\s*([a-z_]+)\s*:", re.M)


def _row_key(obj: dict) -> tuple[str, int, str] | None:
    if obj.get("kind") != "position":
        return None
    m = obj.get("meta") or {}
    sid = m.get("source_example_id")
    pidx = m.get("position_index")
    pt = m.get("position_type")
    if sid is None or pidx is None or pt is None:
        return None
    return (str(sid), int(pidx), str(pt))


def _is_bad(desc: str) -> bool:
    cats = BULLET_RE.findall(desc or "")
    cset = set(cats)
    if "scene" not in cset:
        return True
    if "target" not in cset:
        return True
    if any(c not in KNOWN_CATS for c in cats):
        return True
    return False


def _identify_bad_keys(labels_path: Path) -> set[tuple[str, int, str]]:
    bad: set[tuple[str, int, str]] = set()
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = _row_key(obj)
            if k is None:
                continue
            if _is_bad(obj.get("description") or ""):
                bad.add(k)
    return bad


def _build_targeted_inputs(
    bad_keys: set[tuple[str, int, str]],
    *,
    activations_root: Path,
    dataset_root: Path,
    frame_cache_dir: Path,
    tokenizer_repo: str,
    state_name: str | None,
) -> list:
    """Yield PositionLabelInput for each (sid, pidx, pt) requested."""
    reader = ActivationShardReader(activations_root)
    tokenizer = load_qwen3_vl_tokenizer(tokenizer_repo)

    by_sid: dict[str, list[tuple[int, str]]] = {}
    for sid, pidx, pt in bad_keys:
        by_sid.setdefault(sid, []).append((pidx, pt))

    pool = FrameLoaderPool(max_open=8)
    sampled: list[SampledExample] = []

    for item in reader.iter_examples(
        record_filter=lambda rec: rec.example_id in by_sid,
    ):
        rec = item["_record"]
        img = item["image_mask"]
        ids = item.get("input_ids")
        if ids is None:
            logging.warning(
                "Example %s has no input_ids; skipping (re-run extraction with --store-input-ids).",
                rec.example_id,
            )
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

    inputs = list(build_position_inputs(
        sampled,
        dataset_root=dataset_root,
        frame_cache_dir=frame_cache_dir,
        state_name=state_name,
        pool=pool,
    ))
    pool.close_all()
    return inputs


async def _do_relabel(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels)
    activations_root = Path(args.activations)
    sidecar = Path(args.sidecar)
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_root = Path(args.dataset)
    else:
        manifest = RunManifest.load(activations_root / "manifest.json")
        dpath = manifest.extra.get("dataset_path")
        if not dpath or not Path(dpath).is_dir():
            raise SystemExit(
                f"--dataset is required (manifest dataset_path '{dpath}' is not a directory)"
            )
        dataset_root = Path(dpath)
    logging.info("dataset_root: %s", dataset_root)

    frame_cache_dir = Path(args.frames_cache or (labels_path.parent / "frames_cache"))
    frame_cache_dir.mkdir(parents=True, exist_ok=True)

    bad_keys = _identify_bad_keys(labels_path)
    logging.info("identified %d rows needing relabel", len(bad_keys))
    if not bad_keys:
        return 0

    inputs = _build_targeted_inputs(
        bad_keys,
        activations_root=activations_root,
        dataset_root=dataset_root,
        frame_cache_dir=frame_cache_dir,
        tokenizer_repo=args.tokenizer,
        state_name=args.state_name,
    )
    logging.info("built %d PositionLabelInputs (with frames)", len(inputs))
    if len(inputs) < len(bad_keys):
        logging.warning(
            "only %d/%d inputs assembled (some examples missing input_ids or frames)",
            len(inputs), len(bad_keys),
        )

    orig_builder = openai_client.build_position_prompt
    openai_client.build_position_prompt = build_strict_position_prompt
    try:
        n_new = await label_many_async(
            inputs,
            sidecar,
            model=args.model,
            concurrency=args.concurrency,
            api_key=args.api_key,
            resume=True,
        )
    finally:
        openai_client.build_position_prompt = orig_builder

    logging.info("relabeled %d new rows -> %s", n_new, sidecar)
    return 0


def _merge_inplace(labels_path: Path, sidecar: Path) -> None:
    """Replace bad rows in labels.jsonl with sidecar rows (matched by canonical key)."""
    if not sidecar.exists():
        raise SystemExit(f"sidecar {sidecar} not found")

    new_by_key: dict[tuple[str, int, str], dict] = {}
    n_sidecar_rows = 0
    n_sidecar_kept = 0
    n_sidecar_still_bad = 0
    with sidecar.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_sidecar_rows += 1
            if obj.get("error") or not (obj.get("description") or "").strip():
                continue
            if _is_bad(obj["description"]):
                n_sidecar_still_bad += 1
                continue
            k = _row_key(obj)
            if k is None:
                continue
            n_sidecar_kept += 1
            new_by_key[k] = obj

    bak = labels_path.with_suffix(labels_path.suffix + ".bak2")
    shutil.copy2(labels_path, bak)
    logging.info("backup: %s", bak)

    n_replaced = 0
    n_left_dirty = 0
    tmp = labels_path.with_suffix(labels_path.suffix + ".tmp")
    with labels_path.open() as src, tmp.open("w") as dst:
        for line in src:
            line = line.rstrip("\n")
            if not line:
                dst.write("\n")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dst.write(line + "\n")
                continue
            k = _row_key(obj)
            if k is not None and k in new_by_key:
                dst.write(json.dumps(new_by_key[k], ensure_ascii=False) + "\n")
                n_replaced += 1
            else:
                if k is not None and _is_bad(obj.get("description") or ""):
                    n_left_dirty += 1
                dst.write(line + "\n")

    tmp.replace(labels_path)
    logging.info(
        "merge: replaced %d rows; sidecar kept %d / %d (still-bad %d); "
        "%d original bad rows left untouched (no acceptable relabel found)",
        n_replaced, n_sidecar_kept, n_sidecar_rows, n_sidecar_still_bad, n_left_dirty,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--activations", required=True, type=Path)
    p.add_argument("--dataset", type=Path, default=None,
                   help="LeRobot dataset root (default: read from activations manifest)")
    p.add_argument("--sidecar", required=True, type=Path)
    p.add_argument("--frames-cache", type=Path, default=None,
                   help="default: <labels-dir>/frames_cache")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_REPO)
    p.add_argument("--state-name", default=None)
    p.add_argument("--model", default=None,
                   help="OpenAI labeling model (default: env OPENAI_LABELING_MODEL or gpt-5.1-mini)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--api-key", default=None)
    p.add_argument("--in-place", action="store_true",
                   help="After relabel, merge sidecar back into --labels (with .bak2 backup)")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip relabel; only merge an existing sidecar into --labels")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.merge_only:
        if args.model is None:
            args.model = openai_client.DEFAULT_MODEL
        asyncio.run(_do_relabel(args))

    if args.in_place or args.merge_only:
        _merge_inplace(args.labels, args.sidecar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
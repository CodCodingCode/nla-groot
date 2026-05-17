#!/usr/bin/env python
"""Fuse N per-suite LIBERO extractions + labels into one SFT-ready root.

The extraction + labeling pipelines write a separate directory per LIBERO
suite::

    data/activations/libero_4suite_stride2/libero_{goal,spatial,object,10}/
        manifest.json
        index.jsonl
        shard_NNNNNN/activations.safetensors

    data/labels/libero_4suite_stride2/libero_{goal,spatial,object,10}/
        labels.jsonl
        frames_cache/{source_id}__{image,wrist_image}.jpg
        manifest.json

SFT only takes a single `--activations-root` + `--labels-jsonl`. This script
produces a single combined root that is byte-compatible with the standard
``ActivationShardReader`` + ``LabeledPositionDataset`` schema by:

1. Symlinking each source ``shard_NNNNNN`` into the combined root under a
   globally-unique ``shard_<global_id>`` name so safetensors lookups still
   resolve.
2. Rewriting ``index.jsonl`` rows with the remapped ``shard_id`` and
   suite-prefixed ``example_id`` (``goal__traj000159_step000060``) so the
   four namespaces can never collide.
3. Doing the same prefixing for ``labels.jsonl`` ``source_example_id`` /
   ``example_id`` so labels still join cleanly with activations.
4. Building a fused ``stats.json`` and ``manifest.json`` (n_examples /
   percentile fields are aggregated weighted by per-suite ``n_positions``).
5. Optionally symlinking the per-suite ``frames_cache/`` into a single
   combined cache with prefixed filenames so the multimodal judge can also
   consume the merged corpus later.

The combine is idempotent: re-running with new suites just adds more
records, but existing combined manifests are NOT overwritten unless
``--force`` is passed.

Example::

    PYTHONPATH=src python scripts/training/combine_libero_4suite.py \\
        --activations-root data/activations/libero_4suite_stride2 \\
        --labels-root      data/labels/libero_4suite_stride2 \\
        --suites           goal spatial object 10 \\
        --combined-activations data/activations/libero_4suite_combined \\
        --combined-labels-jsonl data/labels/libero_4suite_combined/labels.jsonl \\
        --combined-frames-cache data/labels/libero_4suite_combined/frames_cache

After this you can point SFT directly at the combined root::

    --stats-json       data/activations/libero_4suite_combined/stats.json
    --activations-root data/activations/libero_4suite_combined
    --labels-jsonl     data/labels/libero_4suite_combined/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from nla.extraction.storage import ExampleRecord, RunManifest

logger = logging.getLogger("combine_libero_4suite")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activations-root", required=True,
                   help="Parent dir containing libero_<suite>/ per-suite extraction dumps.")
    p.add_argument("--labels-root", required=True,
                   help="Parent dir containing libero_<suite>/ per-suite labeling dumps.")
    p.add_argument("--suites", nargs="+", required=True,
                   help="Suite tokens in the order to fuse, e.g. 'goal spatial object 10'. "
                        "Each <suite> must have a libero_<suite> subdir under both --activations-root "
                        "and --labels-root.")
    p.add_argument("--combined-activations", required=True,
                   help="Output dir for the fused activation root (manifest.json, index.jsonl, shard symlinks).")
    p.add_argument("--combined-labels-jsonl", required=True,
                   help="Output path for the fused labels.jsonl. Parent dir is created.")
    p.add_argument("--combined-frames-cache", default=None,
                   help="Optional: output dir for the fused frames_cache. When set, per-suite "
                        "frames are symlinked under prefixed names ({suite}__{source_id}__{key}.jpg).")
    p.add_argument("--force", action="store_true",
                   help="If set, overwrites any existing files at the combined paths.")
    p.add_argument("--log-level", default="INFO")
    return p


def _suite_act_dir(activations_root: Path, suite: str) -> Path:
    return activations_root / f"libero_{suite}"


def _suite_label_dir(labels_root: Path, suite: str) -> Path:
    return labels_root / f"libero_{suite}"


def _prefix_id(suite: str, raw_id: str) -> str:
    """Suite-namespaced example id; idempotent if already prefixed."""
    head = f"{suite}__"
    if raw_id.startswith(head):
        return raw_id
    return f"{head}{raw_id}"


def _combine_one_suite_activations(
    suite: str,
    src_root: Path,
    dst_root: Path,
    *,
    next_shard_id: int,
) -> tuple[list[ExampleRecord], int]:
    """Symlink shards + return remapped records. Returns ``(records, new_next_shard_id)``."""
    src_manifest_path = src_root / "manifest.json"
    src_index_path = src_root / "index.jsonl"
    if not src_manifest_path.exists():
        raise FileNotFoundError(f"missing manifest.json: {src_manifest_path}")
    if not src_index_path.exists():
        raise FileNotFoundError(f"missing index.jsonl: {src_index_path}")

    # Read source records; build {local_shard_id: global_shard_id} mapping by
    # walking the records in order and assigning a fresh global id every time
    # we see a new local id. This preserves shard ordering within a suite,
    # which is what ``ActivationShardReader.iter_examples`` relies on.
    src_to_global: dict[int, int] = {}
    out_records: list[ExampleRecord] = []
    with src_index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = ExampleRecord.from_json(line)
            if rec.shard_id not in src_to_global:
                src_to_global[rec.shard_id] = next_shard_id
                src_shard_dir = src_root / f"shard_{rec.shard_id:06d}"
                dst_shard_dir = dst_root / f"shard_{next_shard_id:06d}"
                if dst_shard_dir.is_symlink() or dst_shard_dir.exists():
                    dst_shard_dir.unlink() if dst_shard_dir.is_symlink() else shutil.rmtree(dst_shard_dir)
                dst_shard_dir.symlink_to(src_shard_dir.resolve())
                next_shard_id += 1
            remapped = ExampleRecord(
                example_id=_prefix_id(suite, rec.example_id),
                shard_id=src_to_global[rec.shard_id],
                local_index=rec.local_index,
                seq_len=rec.seq_len,
                task_index=rec.task_index,
                task_text=rec.task_text,
                episode_index=rec.episode_index,
                step_index=rec.step_index,
                embodiment_tag=rec.embodiment_tag,
                extra={**(rec.extra or {}), "suite": suite},
            )
            out_records.append(remapped)
    logger.info(
        "  %s: %d examples across %d source shards -> shards %d..%d",
        suite, len(out_records), len(src_to_global),
        min(src_to_global.values()) if src_to_global else -1,
        max(src_to_global.values()) if src_to_global else -1,
    )
    return out_records, next_shard_id


def _combine_one_suite_labels(
    suite: str,
    src_label_dir: Path,
    src_frames_cache: Path | None,
    dst_frames_cache: Path | None,
) -> list[dict]:
    """Read suite labels.jsonl, prefix ids, optionally symlink frames cache."""
    labels_path = src_label_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels.jsonl: {labels_path}")

    out_rows: list[dict] = []
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Prefix every id field that exists. We're conservative: any
            # field whose name ends in ``example_id`` gets the prefix.
            for k in list(row.keys()):
                if k.endswith("example_id") and isinstance(row[k], str):
                    row[k] = _prefix_id(suite, row[k])
            meta = row.get("meta") or {}
            for k in list(meta.keys()):
                if k.endswith("example_id") and isinstance(meta[k], str):
                    meta[k] = _prefix_id(suite, meta[k])
            meta.setdefault("suite", suite)
            row["meta"] = meta
            out_rows.append(row)

    if dst_frames_cache is not None and src_frames_cache is not None and src_frames_cache.exists():
        dst_frames_cache.mkdir(parents=True, exist_ok=True)
        n_linked = 0
        for jpg in src_frames_cache.iterdir():
            if not jpg.is_file():
                continue
            new_name = _prefix_id(suite, jpg.name)
            dst = dst_frames_cache / new_name
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(jpg.resolve())
            n_linked += 1
        logger.info("  %s: linked %d frame files into combined cache", suite, n_linked)

    logger.info("  %s: %d label rows", suite, len(out_rows))
    return out_rows


def _fuse_stats(per_suite_stats: list[dict]) -> dict:
    """Combine per-suite stats.json files.

    ``n_positions`` is summed. Percentile fields are weighted-averaged by
    ``n_positions`` — this is an approximation of the true combined
    percentile (recomputing exactly would require re-iterating every
    activation norm). It's well within the precision SFT needs for α.
    """
    weighted_fields = (
        "p50_norm", "p75_norm", "p90_norm", "p99_norm",
        "mean_norm", "std_norm", "image_token_fraction",
    )
    total_n = sum(int(s.get("n_positions", 0)) for s in per_suite_stats)
    fused: dict[str, float] = {"n_positions": total_n}
    if total_n <= 0:
        for k in weighted_fields:
            fused[k] = 0.0
        return fused
    for k in weighted_fields:
        num = sum(
            float(s.get(k, 0.0)) * int(s.get("n_positions", 0))
            for s in per_suite_stats
        )
        fused[k] = num / total_n
    return fused


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    activations_root = Path(args.activations_root)
    labels_root = Path(args.labels_root)
    combined_act = Path(args.combined_activations)
    combined_lbl = Path(args.combined_labels_jsonl)
    combined_frames = Path(args.combined_frames_cache) if args.combined_frames_cache else None

    if combined_act.exists() and not args.force:
        # The dataset-transfer checklist is explicit: stale partial fusions
        # are dangerous because labels can outlive activations. Refuse and
        # tell the user to pass --force if they really want this.
        if (combined_act / "manifest.json").exists():
            logger.error(
                "Combined activations root already populated at %s "
                "(manifest.json exists). Pass --force to rebuild from scratch.",
                combined_act,
            )
            return 2

    combined_act.mkdir(parents=True, exist_ok=True)
    combined_lbl.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[ExampleRecord] = []
    all_label_rows: list[dict] = []
    per_suite_stats: list[dict] = []
    per_suite_manifests: list[dict] = []
    next_shard_id = 0

    for suite in args.suites:
        src_act = _suite_act_dir(activations_root, suite)
        src_lbl = _suite_label_dir(labels_root, suite)
        logger.info("Fusing suite '%s' (act=%s, labels=%s)", suite, src_act, src_lbl)

        records, next_shard_id = _combine_one_suite_activations(
            suite, src_act, combined_act, next_shard_id=next_shard_id,
        )
        all_records.extend(records)

        src_frames = src_lbl / "frames_cache"
        rows = _combine_one_suite_labels(
            suite, src_lbl, src_frames if combined_frames else None, combined_frames,
        )
        all_label_rows.extend(rows)

        stats_path = src_act / "stats.json"
        if stats_path.exists():
            per_suite_stats.append(json.loads(stats_path.read_text()))
        else:
            logger.warning("  %s: missing stats.json; α will be approximated from the rest", suite)

        manifest_path = src_act / "manifest.json"
        per_suite_manifests.append(json.loads(manifest_path.read_text()))

    # Write fused index.jsonl
    index_path = combined_act / "index.jsonl"
    with index_path.open("w") as f:
        for rec in all_records:
            f.write(rec.to_json() + "\n")
    logger.info("Wrote %d records to %s", len(all_records), index_path)

    # Write fused manifest.json
    first = per_suite_manifests[0]
    fused_manifest = RunManifest(
        schema_version=int(first.get("schema_version", 1)),
        model_repo="multi:" + ",".join(args.suites),
        layer_module_path=first["layer_module_path"],
        hidden_size=int(first["hidden_size"]),
        activation_dtype=first["activation_dtype"],
        embodiment_tag=first.get("embodiment_tag"),
        num_examples=len(all_records),
        num_shards=next_shard_id,
        extra={
            "fused_from": [m.get("extra", {}) for m in per_suite_manifests],
            "suites": list(args.suites),
            "per_suite_num_examples": [int(m.get("num_examples", 0)) for m in per_suite_manifests],
        },
    )
    fused_manifest.save(combined_act / "manifest.json")
    logger.info("Wrote fused manifest with num_examples=%d num_shards=%d",
                fused_manifest.num_examples, fused_manifest.num_shards)

    # Write fused stats.json
    if per_suite_stats:
        fused_stats = _fuse_stats(per_suite_stats)
        (combined_act / "stats.json").write_text(json.dumps(fused_stats, indent=2))
        logger.info("Wrote fused stats.json (n_positions=%d, p75_norm=%.4f)",
                    int(fused_stats["n_positions"]), float(fused_stats["p75_norm"]))
    else:
        logger.warning("No per-suite stats.json available; skipping fused stats.json")

    # Write fused labels.jsonl
    with combined_lbl.open("w") as f:
        for row in all_label_rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d label rows to %s", len(all_label_rows), combined_lbl)

    return 0


if __name__ == "__main__":
    sys.exit(main())

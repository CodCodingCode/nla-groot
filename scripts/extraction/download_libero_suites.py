#!/usr/bin/env python
"""Download the 3 LIBERO suites (Spatial / Object / 10) and their matching
GR00T-N1.7-LIBERO checkpoints from HuggingFace.

For checkpoints we pull only the inference-relevant files (skip
``global_step*``, ``rng_state_*``, ``trainer_state``, ``zero_to_fp32.py``
etc.) so the per-suite cost matches the existing ~6.5 GB libero_goal
checkpoint instead of the full 25+ GB training snapshot.

Layout produced:

    checkpoints/GR00T-N1.7-LIBERO/libero_{spatial,object,10}/...
    third_party/Isaac-GR00T/examples/LIBERO/libero_{spatial,object,10}_no_noops_1.0.0_lerobot/...

Usage::

    PYTHONPATH=src python scripts/extraction/download_libero_suites.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINT_REPO = "nvidia/GR00T-N1.7-LIBERO"
CHECKPOINT_LOCAL_ROOT = Path("checkpoints/GR00T-N1.7-LIBERO")
DATASET_OWNER = "IPEC-COMMUNITY"
DATASET_LOCAL_ROOT = Path("third_party/Isaac-GR00T/examples/LIBERO")

SUITES = ["spatial", "object", "10"]

# Per-suite glob whitelist -- inference + statistics + processor only.
INFERENCE_FILE_GLOBS = [
    "{suite}/SUCCESS",
    "{suite}/config.json",
    "{suite}/embodiment_id.json",
    "{suite}/processor_config.json",
    "{suite}/statistics.json",
    "{suite}/model.safetensors.index.json",
    "{suite}/model-*.safetensors",
    "{suite}/experiment_cfg/*",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--suites", nargs="+", default=SUITES,
                   choices=SUITES + ["spatial", "object", "10"])
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--skip-checkpoints", action="store_true")
    p.add_argument("--skip-datasets", action="store_true")
    return p


def _download_checkpoints(suites: list[str], max_workers: int) -> None:
    allow_patterns: list[str] = []
    for s in suites:
        suite_dir = f"libero_{s}"
        for g in INFERENCE_FILE_GLOBS:
            allow_patterns.append(g.format(suite=suite_dir))
    print(f"[checkpoints] repo={CHECKPOINT_REPO}")
    print(f"[checkpoints] suites={suites}  patterns={len(allow_patterns)}")
    print(f"[checkpoints] target dir={CHECKPOINT_LOCAL_ROOT}")
    CHECKPOINT_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(
        repo_id=CHECKPOINT_REPO,
        local_dir=str(CHECKPOINT_LOCAL_ROOT),
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    print(f"[checkpoints] done in {time.time() - t0:.1f}s")


def _download_dataset(suite: str, max_workers: int) -> None:
    repo_id = f"{DATASET_OWNER}/libero_{suite}_no_noops_1.0.0_lerobot"
    dst = DATASET_LOCAL_ROOT / f"libero_{suite}_no_noops_1.0.0_lerobot"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[dataset {suite}] repo={repo_id}")
    print(f"[dataset {suite}] target dir={dst}")
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dst),
        max_workers=max_workers,
    )
    print(f"[dataset {suite}] done in {time.time() - t0:.1f}s")


def _verify(suites: list[str]) -> None:
    print("\n[verify] checkpoint directory listing:")
    for s in suites:
        suite_dir = CHECKPOINT_LOCAL_ROOT / f"libero_{s}"
        if not suite_dir.exists():
            print(f"  MISSING: {suite_dir}")
            continue
        st = sorted(p.name for p in suite_dir.iterdir())
        print(f"  {suite_dir}: {len(st)} entries -> {st[:6]}{'...' if len(st)>6 else ''}")

    print("\n[verify] dataset episode counts:")
    for s in suites:
        ds_dir = DATASET_LOCAL_ROOT / f"libero_{s}_no_noops_1.0.0_lerobot"
        ep = ds_dir / "meta" / "episodes.jsonl"
        if not ep.exists():
            print(f"  MISSING: {ep}")
            continue
        n = sum(1 for _ in ep.open())
        print(f"  libero_{s}: {n} episodes ({ep})")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    t_all = time.time()
    if not args.skip_checkpoints:
        _download_checkpoints(args.suites, args.max_workers)
    if not args.skip_datasets:
        for s in args.suites:
            _download_dataset(s, args.max_workers)
    _verify(args.suites)
    print(f"\nTotal wall-clock: {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Audit alignment between cached label frames and activation records.

For each labeled position we verify that the JPEG sitting in
``frames_cache/`` actually came from the same ``(episode_index, step_index)``
as the activation it was labeled against. An off-by-one between the cached
frame and the activation would directly inflate apparent "labeler error" in
downstream judge metrics, so this audit exists to surface that drift early.

Read-only against ``data/activations`` and ``data/labels``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("audit_frame_sync")

DEFAULT_VIEWS = ("image", "wrist_image")
EXAMPLE_ID_RE = re.compile(r"^traj(\d+)_step(\d+)$")


def _load_index(activations_root: Path) -> dict[str, dict[str, Any]]:
    """Parse ``index.jsonl`` into ``{example_id: record_dict}``."""
    index_path = activations_root / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing activation index at {index_path}")
    out: dict[str, dict[str, Any]] = {}
    with index_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["example_id"]] = rec
    return out


def _load_position_labels(labels_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("kind") == "position":
                rows.append(obj)
    return rows


def _parse_example_id(example_id: str) -> tuple[int, int] | None:
    m = EXAMPLE_ID_RE.match(example_id)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _find_sidecar(frames_cache: Path, source_id: str, views: tuple[str, ...]) -> dict[str, Any] | None:
    """Return parsed sidecar JSON if any view has one. Sidecars optional."""
    for vk in views:
        sidecar = frames_cache / f"{source_id}__{vk}.json"
        if sidecar.exists():
            try:
                return json.loads(sidecar.read_text())
            except json.JSONDecodeError:
                logger.warning("Could not parse sidecar at %s", sidecar)
    return None


def _frame_paths(frames_cache: Path, source_id: str, views: tuple[str, ...]) -> list[Path]:
    return [frames_cache / f"{source_id}__{vk}.jpg" for vk in views]


def _classify(
    *,
    record_ep: int | None,
    record_st: int | None,
    candidate_ep: int | None,
    candidate_st: int | None,
) -> str:
    """Return one of: aligned / off_by_one / off_by_more."""
    if record_ep is None or record_st is None or candidate_ep is None or candidate_st is None:
        return "off_by_more"
    if record_ep != candidate_ep:
        return "off_by_more"
    delta = abs(int(record_st) - int(candidate_st))
    if delta == 0:
        return "aligned"
    if delta == 1:
        return "off_by_one"
    return "off_by_more"


def _try_pixel_diff(
    *,
    dataset_root: Path,
    record_ep: int,
    record_st: int,
    cached_frames: list[Path],
    views: tuple[str, ...],
) -> float | None:
    """Re-extract the frame and return mean abs pixel diff (0..255), or None.

    Returns ``None`` if dependencies are missing, the dataset isn't present,
    or any cached frame can't be opened.
    """
    try:
        import numpy as np
        from PIL import Image

        from nla.labeling.frames import EpisodeFrameLoader
    except Exception as e:
        logger.debug("Pixel-diff unavailable: %s", e)
        return None
    if not dataset_root.exists():
        logger.debug("Pixel-diff skipped: dataset_root %s missing", dataset_root)
        return None
    try:
        loader = EpisodeFrameLoader(dataset_root, record_ep)
    except Exception as e:
        logger.debug("Pixel-diff loader init failed: %s", e)
        return None
    try:
        diffs: list[float] = []
        for vk, cached_path in zip(views, cached_frames):
            if not cached_path.exists():
                continue
            try:
                fresh = loader.frame(vk, record_st)
                cached = np.array(Image.open(cached_path).convert("RGB"))
            except Exception as e:
                logger.debug("Skipping pixel-diff for %s: %s", cached_path, e)
                continue
            if fresh.shape != cached.shape:
                # Resolution mismatch — caller decides how to interpret.
                return float("inf")
            diffs.append(float(np.abs(fresh.astype("int16") - cached.astype("int16")).mean()))
        if not diffs:
            return None
        return sum(diffs) / len(diffs)
    finally:
        try:
            loader.close()
        except Exception:
            pass


def audit(
    *,
    labels_jsonl: Path,
    activations_root: Path,
    frames_cache: Path,
    n_sample: int,
    seed: int,
    strict: bool,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    dataset_root: Path | None = None,
    pixel_diff: bool = True,
    pixel_diff_threshold: float = 5.0,
) -> dict[str, Any]:
    """Sample N labels and report alignment between frames and activations."""
    index = _load_index(activations_root)
    rows = _load_position_labels(labels_jsonl)
    if not rows:
        raise RuntimeError(f"No kind=='position' rows found in {labels_jsonl}")

    rng = random.Random(seed)
    if n_sample >= len(rows):
        sampled = list(rows)
    else:
        sampled = rng.sample(rows, n_sample)

    findings: list[dict[str, Any]] = []
    counters = {
        "n_total": len(sampled),
        "n_aligned": 0,
        "n_off_by_one": 0,
        "n_off_by_more": 0,
        "n_missing_frame": 0,
        "n_missing_record": 0,
        "n_unverifiable": 0,
        "n_pixel_diff_attempted": 0,
        "n_pixel_diff_failed": 0,
    }
    pixel_diffs_aligned: list[float] = []
    pixel_diffs_off_by_one: list[float] = []

    for row in sampled:
        meta = row.get("meta") or {}
        source_id = meta.get("source_example_id") or row.get("example_id", "").split("@", 1)[0]
        finding: dict[str, Any] = {
            "example_id": row.get("example_id"),
            "source_example_id": source_id,
            "status": None,
            "reason": None,
        }

        record = index.get(source_id)
        if record is None:
            counters["n_missing_record"] += 1
            counters["n_off_by_more"] += 1
            finding["status"] = "off_by_more"
            finding["reason"] = "no matching activation record"
            findings.append(finding)
            continue

        rec_ep = record.get("episode_index")
        rec_st = record.get("step_index")
        finding["record_episode_index"] = rec_ep
        finding["record_step_index"] = rec_st

        cached_frames = _frame_paths(frames_cache, source_id, views)
        missing_views = [p.name for p in cached_frames if not p.exists()]
        finding["frames_present"] = [p.name for p in cached_frames if p.exists()]
        finding["frames_missing"] = missing_views
        if missing_views and len(missing_views) == len(cached_frames):
            counters["n_missing_frame"] += 1
            finding["status"] = "missing_frame"
            finding["reason"] = "no cached JPEGs for source_example_id"
            findings.append(finding)
            continue

        sidecar = _find_sidecar(frames_cache, source_id, views)
        if sidecar is not None:
            cand_ep = sidecar.get("episode_index")
            cand_st = sidecar.get("step_index")
            finding["sidecar"] = {"episode_index": cand_ep, "step_index": cand_st}
            status = _classify(
                record_ep=rec_ep, record_st=rec_st,
                candidate_ep=cand_ep, candidate_st=cand_st,
            )
            finding["status"] = status
            finding["reason"] = "sidecar metadata"
            counters[f"n_{status}"] += 1
            findings.append(finding)
            continue

        if strict and pixel_diff and dataset_root is not None:
            counters["n_pixel_diff_attempted"] += 1
            mean_diff = _try_pixel_diff(
                dataset_root=dataset_root,
                record_ep=int(rec_ep), record_st=int(rec_st),
                cached_frames=cached_frames, views=views,
            )
            if mean_diff is None:
                counters["n_pixel_diff_failed"] += 1
                counters["n_unverifiable"] += 1
                finding["status"] = "unverifiable"
                finding["reason"] = "pixel-diff unavailable (no loader / no dataset)"
                findings.append(finding)
                continue
            finding["pixel_diff_at_record"] = mean_diff
            if mean_diff <= pixel_diff_threshold:
                counters["n_aligned"] += 1
                finding["status"] = "aligned"
                finding["reason"] = f"pixel diff {mean_diff:.2f} <= {pixel_diff_threshold}"
                pixel_diffs_aligned.append(mean_diff)
            else:
                # Try the +/- 1 step neighbours to see if it's an off-by-one.
                neighbour_diffs: dict[int, float] = {}
                for delta in (-1, 1):
                    nd = _try_pixel_diff(
                        dataset_root=dataset_root,
                        record_ep=int(rec_ep), record_st=int(rec_st) + delta,
                        cached_frames=cached_frames, views=views,
                    )
                    if nd is not None and nd != float("inf"):
                        neighbour_diffs[delta] = nd
                finding["pixel_diff_neighbours"] = neighbour_diffs
                best_delta = min(neighbour_diffs, key=neighbour_diffs.get) if neighbour_diffs else None
                if best_delta is not None and neighbour_diffs[best_delta] < mean_diff and neighbour_diffs[best_delta] <= pixel_diff_threshold:
                    counters["n_off_by_one"] += 1
                    finding["status"] = "off_by_one"
                    finding["reason"] = f"frame matches step {rec_st + best_delta} better than {rec_st}"
                    pixel_diffs_off_by_one.append(neighbour_diffs[best_delta])
                else:
                    counters["n_off_by_more"] += 1
                    finding["status"] = "off_by_more"
                    finding["reason"] = f"pixel diff {mean_diff:.2f} > {pixel_diff_threshold}"
            findings.append(finding)
            continue

        counters["n_unverifiable"] += 1
        finding["status"] = "unverifiable"
        finding["reason"] = "no sidecar; pixel-diff not attempted"
        findings.append(finding)

    summary = dict(counters)
    summary["mean_pixel_diff_when_aligned"] = (
        sum(pixel_diffs_aligned) / len(pixel_diffs_aligned) if pixel_diffs_aligned else None
    )
    summary["mean_pixel_diff_when_offbyone"] = (
        sum(pixel_diffs_off_by_one) / len(pixel_diffs_off_by_one) if pixel_diffs_off_by_one else None
    )
    return {"summary": summary, "findings": findings}


def _decide_exit(summary: dict[str, Any], strict: bool) -> tuple[int, str]:
    n_total = summary["n_total"]
    if n_total == 0:
        return 1, "No labels were sampled — cannot audit."
    n_aligned = summary["n_aligned"]
    n_off1 = summary["n_off_by_one"]
    n_offm = summary["n_off_by_more"]
    n_missing = summary["n_missing_frame"]
    n_unver = summary["n_unverifiable"]
    aligned_frac = n_aligned / n_total
    unver_frac = n_unver / n_total

    if strict:
        if n_off1 + n_offm + n_missing > 0:
            return 1, (
                f"strict: detected off_by_one={n_off1} off_by_more={n_offm} missing_frame={n_missing}"
            )
        if n_unver > 0:
            return 1, f"strict: {n_unver}/{n_total} unverifiable"
        return 0, f"strict: aligned {n_aligned}/{n_total}"

    if unver_frac > 0.5:
        return 0, (
            f"warning: {n_unver}/{n_total} unverifiable (no sidecars). "
            f"Verified subset: aligned={n_aligned}, off_by_one={n_off1}, off_by_more={n_offm}, missing={n_missing}."
        )
    if aligned_frac >= 0.99:
        return 0, f"aligned {n_aligned}/{n_total} ({aligned_frac:.1%})"
    return 1, (
        f"alignment too low: {n_aligned}/{n_total} ({aligned_frac:.1%}); "
        f"off_by_one={n_off1}, off_by_more={n_offm}, missing={n_missing}, unverifiable={n_unver}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit cached frames vs activation index alignment.")
    p.add_argument("--labels-jsonl", type=Path, required=True)
    p.add_argument("--activations-root", type=Path, required=True)
    p.add_argument("--frames-cache", type=Path, required=True)
    p.add_argument("--n-sample", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=Path, default=Path("data/eval/frame_sync_audit.json"))
    p.add_argument("--strict", action="store_true",
                   help="Fail on any off-by-one / missing / unverifiable. Enables pixel-diff fallback.")
    p.add_argument("--views", default=",".join(DEFAULT_VIEWS),
                   help="Comma-separated view keys to look for under frames_cache/.")
    p.add_argument("--dataset-root", type=Path, default=None,
                   help="LeRobot dataset root for optional --strict pixel-diff re-extraction.")
    p.add_argument("--no-pixel-diff", action="store_true",
                   help="Disable the pixel-diff fallback even in --strict mode (used by tests).")
    p.add_argument("--pixel-diff-threshold", type=float, default=5.0,
                   help="Mean abs pixel difference treated as 'matching' (0..255 scale).")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    result = audit(
        labels_jsonl=args.labels_jsonl,
        activations_root=args.activations_root,
        frames_cache=args.frames_cache,
        n_sample=args.n_sample,
        seed=args.seed,
        strict=args.strict,
        views=views,
        dataset_root=args.dataset_root,
        pixel_diff=not args.no_pixel_diff,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))

    summary = result["summary"]
    print("=" * 64)
    print(f"frame-sync audit  ({summary['n_total']} sampled rows)")
    print("-" * 64)
    print(f"  aligned          : {summary['n_aligned']}")
    print(f"  off_by_one       : {summary['n_off_by_one']}")
    print(f"  off_by_more      : {summary['n_off_by_more']}")
    print(f"  missing_frame    : {summary['n_missing_frame']}")
    print(f"  missing_record   : {summary['n_missing_record']}")
    print(f"  unverifiable     : {summary['n_unverifiable']}")
    if summary["mean_pixel_diff_when_aligned"] is not None:
        print(f"  mean diff aligned    : {summary['mean_pixel_diff_when_aligned']:.3f}")
    if summary["mean_pixel_diff_when_offbyone"] is not None:
        print(f"  mean diff off_by_one : {summary['mean_pixel_diff_when_offbyone']:.3f}")
    print(f"  -> wrote {args.out_json}")

    code, msg = _decide_exit(summary, strict=args.strict)
    print("-" * 64)
    print(("PASS" if code == 0 else "FAIL") + f": {msg}")
    print("=" * 64)
    return code


if __name__ == "__main__":
    sys.exit(main())

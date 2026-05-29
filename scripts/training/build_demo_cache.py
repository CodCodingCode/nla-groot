"""Phase A — build the demo cache for subsequent RL.

For every row in a v6 label set, assemble the tuple

    (image, wrist_image, proprio, demo_action, intent_text)

drawn from the source LIBERO LeRobot demos. The cache is read once here and
reused for all downstream RL (BC-regularised GRPO, action-matching rewards,
etc.), so it is keyed 1:1 with the label rows.

Layout (mirrors the eval frames_cache pattern in
``scripts/eval/extract_label_frames.py`` — JPEG per (step, camera) + a JSONL
sidecar)::

    {out_dir}/frames/{source_example_id}__{video_key}.jpg
    {out_dir}/index.jsonl     # one row per label row
    {out_dir}/manifest.json

index.jsonl row::

    {
      "example_id":        "spatial__traj000000_step000056@p070_image_patch",
      "source_example_id": "spatial__traj000000_step000056",
      "suite": "spatial", "episode_index": 0, "step_index": 56,
      "position_type": "image_patch",
      "intent_text": "- scene: ...\n- target: ...\n- task: ...",
      "image":       "frames/spatial__traj000000_step000056__image.jpg",
      "wrist_image": "frames/spatial__traj000000_step000056__wrist_image.jpg",
      "proprio":     [ ... 8 floats ... ],
      "demo_action": [[ ...7... ], ...],   # up to --action-horizon steps
      "demo_action_len": 16
    }

Notes
-----
* Images dedupe to the ~50k unique steps (two intent variants share a step), so
  actual disk is ~3-5 GB of JPEG, well under the original 50 GB estimate (which
  assumed raw uint8). ``demo_action`` is GR00T's native 16-step / 7-dim chunk.
* Resumable: existing JPEGs are skipped, so a re-run only fills gaps.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/training/build_demo_cache.py \\
        --labels-jsonl data/labels/libero_4suite_v6_with_task/labels.jsonl \\
        --lerobot-root third_party/Isaac-GR00T/examples/LIBERO \\
        --out-dir data/demo_cache/libero_4suite_v6 \\
        --action-horizon 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from nla.labeling.frames import EpisodeFrameLoader, save_jpeg

# Short modality keys → on-disk handled by DatasetInfo; these are the two
# LIBERO camera views.
VIDEO_KEYS = ["image", "wrist_image"]

# suite token in the example_id → LeRobot dataset directory name.
SUITE_TO_DIRNAME = {
    "goal": "libero_goal_no_noops_1.0.0_lerobot",
    "spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "object": "libero_object_no_noops_1.0.0_lerobot",
    "10": "libero_10_no_noops_1.0.0_lerobot",
}

_ID_RE = re.compile(r"^(?P<suite>.+?)__traj(?P<ep>\d+)_step(?P<step>\d+)$")


def parse_source_id(source_example_id: str) -> tuple[str, int, int]:
    m = _ID_RE.match(source_example_id)
    if not m:
        raise ValueError(f"Unparseable source_example_id: {source_example_id!r}")
    return m["suite"], int(m["ep"]), int(m["step"])


def episode_parquet_path(root: Path, episode_index: int, chunks_size: int = 1000) -> Path:
    chunk = episode_index // chunks_size
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def load_episode_state_action(root: Path, episode_index: int):
    """Return ``(state_by_frame, action_by_frame)`` dicts for one episode."""
    path = episode_parquet_path(root, episode_index)
    if not path.exists():
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    tbl = pq.read_table(path, columns=["observation.state", "action", "frame_index"])
    frames = tbl.column("frame_index").to_pylist()
    states = tbl.column("observation.state").to_pylist()
    actions = tbl.column("action").to_pylist()
    state_by_frame = {int(f): s for f, s in zip(frames, states)}
    action_by_frame = {int(f): a for f, a in zip(frames, actions)}
    return state_by_frame, action_by_frame


def action_chunk(action_by_frame: dict[int, list], frame: int, horizon: int) -> list[list]:
    out = []
    for k in range(frame, frame + horizon):
        if k in action_by_frame:
            out.append([float(x) for x in action_by_frame[k]])
        else:
            break
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-jsonl", required=True, type=Path)
    p.add_argument("--lerobot-root", required=True, type=Path,
                   help="Directory holding the libero_*_no_noops_*_lerobot datasets.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--action-horizon", type=int, default=16,
                   help="GR00T action chunk length stored per step (default 16).")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--suites", nargs="*", default=None,
                   help="Restrict to these suites (default: all present in labels).")
    p.add_argument("--limit", type=int, default=0,
                   help="Smoke test: stop after this many label rows (0 = all).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = args.out_dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read labels, group by (suite, episode) for sequential parquet/video reuse.
    by_episode: dict[tuple[str, int], list] = defaultdict(list)
    n_rows = 0
    with args.labels_jsonl.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            meta = row.get("meta", {})
            src = meta.get("source_example_id")
            if src is None:
                continue
            suite, ep, step = parse_source_id(src)
            if args.suites and suite not in args.suites:
                continue
            by_episode[(suite, ep)].append(
                {
                    "example_id": row["example_id"],
                    "source_example_id": src,
                    "suite": suite,
                    "episode_index": ep,
                    "step_index": step,
                    "position_type": meta.get("position_type"),
                    "intent_text": row.get("description", ""),
                }
            )
            n_rows += 1
            if args.limit and n_rows >= args.limit:
                break

    total_groups = len(by_episode)
    print(f"[demo_cache] {n_rows} label rows across {total_groups} (suite,episode) groups",
          flush=True)

    index_path = out_dir / "index.jsonl"
    written = 0
    jpgs_written = 0
    jpgs_skipped = 0
    missing = 0

    with index_path.open("w") as out_f:
        for gi, ((suite, ep), rows) in enumerate(sorted(by_episode.items())):
            dirname = SUITE_TO_DIRNAME.get(suite)
            if dirname is None:
                print(f"[demo_cache] WARN unknown suite {suite!r}; skipping", flush=True)
                missing += len(rows)
                continue
            root = args.lerobot_root / dirname
            try:
                state_by_frame, action_by_frame = load_episode_state_action(root, ep)
            except FileNotFoundError as e:
                print(f"[demo_cache] WARN {e}; skipping {len(rows)} rows", flush=True)
                missing += len(rows)
                continue

            # Unique frames needed in this episode (dedupe images/proprio/action).
            needed_frames = sorted({r["step_index"] for r in rows})
            step_payload: dict[int, dict] = {}
            with EpisodeFrameLoader(root, ep) as loader:
                for frame in needed_frames:
                    src_id = f"{suite}__traj{ep:06d}_step{frame:06d}"
                    rel_paths = {}
                    for vk in VIDEO_KEYS:
                        dst = frames_dir / f"{src_id}__{vk}.jpg"
                        rel_paths[vk] = str(dst.relative_to(out_dir))
                        if dst.exists():
                            jpgs_skipped += 1
                            continue
                        img = loader.frame(vk, frame)
                        save_jpeg(img, dst, quality=args.jpeg_quality)
                        jpgs_written += 1
                    step_payload[frame] = {
                        "image": rel_paths["image"],
                        "wrist_image": rel_paths["wrist_image"],
                        "proprio": [float(x) for x in state_by_frame.get(frame, [])],
                        "demo_action": action_chunk(action_by_frame, frame, args.action_horizon),
                    }

            for r in rows:
                pay = step_payload.get(r["step_index"])
                if pay is None:
                    missing += 1
                    continue
                rec = {
                    **r,
                    "image": pay["image"],
                    "wrist_image": pay["wrist_image"],
                    "proprio": pay["proprio"],
                    "demo_action": pay["demo_action"],
                    "demo_action_len": len(pay["demo_action"]),
                }
                out_f.write(json.dumps(rec) + "\n")
                written += 1

            if (gi + 1) % 25 == 0 or gi + 1 == total_groups:
                print(f"[demo_cache] group {gi+1}/{total_groups} "
                      f"rows={written} jpg_new={jpgs_written} jpg_skip={jpgs_skipped} "
                      f"missing={missing}", flush=True)

    manifest = {
        "schema_version": 1,
        "labels_jsonl": str(args.labels_jsonl),
        "lerobot_root": str(args.lerobot_root),
        "action_horizon": args.action_horizon,
        "jpeg_quality": args.jpeg_quality,
        "video_keys": VIDEO_KEYS,
        "num_rows": written,
        "num_jpgs_written": jpgs_written,
        "num_jpgs_skipped": jpgs_skipped,
        "num_missing": missing,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[demo_cache] DONE rows={written} jpg_new={jpgs_written} "
          f"jpg_skip={jpgs_skipped} missing={missing}", flush=True)
    print(f"[demo_cache] Wrote {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

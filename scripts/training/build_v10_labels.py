"""Build augmented v10 labels: add progress + gripper pose + step bullets.

Joins the existing v6 caption pool against the LeRobot parquet files that
the activation extraction came from. For each labeled position we add four
bullets carrying the continuous information probes showed h carries but the
v6 template was discarding (step-progress R^2=0.83 at image_patch_mean).

Output schema:
  - all the original bullets (scene/target/distractor/gripper/spatial/task)
  - + gripper_xyz: (x, y, z)
  - + gripper_state: open|closed
  - + progress: fraction of demo elapsed (0.00..1.00)
  - + step: frame_index

Usage::

    PYTHONPATH=src .venv/bin/python scripts/training/build_v10_labels.py \\
      --activations-root data/activations/libero_4suite_v4_combined \\
      --labels-in        data/labels/libero_4suite_v6_with_task/labels.jsonl \\
      --lerobot-root     third_party/Isaac-GR00T/examples/LIBERO \\
      --out-jsonl        data/labels/libero_4suite_v10_state_augmented/labels.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v10-labels")


# Map our suite tag → LeRobot dataset dir
_SUITE_TO_LEROBOT = {
    "goal":    "libero_goal_no_noops_1.0.0_lerobot",
    "spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "object":  "libero_object_no_noops_1.0.0_lerobot",
    "10":      "libero_10_no_noops_1.0.0_lerobot",
}


def _gripper_state(qpos: np.ndarray) -> str:
    """Heuristic: LIBERO panda gripper open ~ ±0.04, closed ~ 0.

    obs.state[6:8] is the two finger qpos; sum near 0 means closed.
    Threshold tuned by inspection of a few rows.
    """
    s = float(abs(qpos).sum())
    return "open" if s > 0.03 else "closed"


def _load_parquet_index(lerobot_root: Path) -> dict[tuple[str, int, int], dict]:
    """Build a (suite, episode_index, frame_index) -> {x,y,z,roll,pitch,yaw,grip,ep_len} index.

    Returns a flat dict so lookup is O(1) per labeled row.
    """
    out: dict[tuple[str, int, int], dict] = {}
    ep_lens: dict[tuple[str, int], int] = {}
    for suite, dset_name in _SUITE_TO_LEROBOT.items():
        dset_dir = lerobot_root / dset_name / "data" / "chunk-000"
        if not dset_dir.exists():
            logger.warning("Skipping %s: %s missing", suite, dset_dir)
            continue
        n_eps = 0
        for parquet_path in sorted(dset_dir.glob("episode_*.parquet")):
            ep_idx = int(parquet_path.stem.split("_")[-1])
            t = pq.read_table(parquet_path,
                              columns=["observation.state", "frame_index", "episode_index"])
            df = t.to_pandas()
            ep_len = len(df)
            ep_lens[(suite, ep_idx)] = ep_len
            for _, row in df.iterrows():
                state = row["observation.state"]
                frame = int(row["frame_index"])
                out[(suite, ep_idx, frame)] = {
                    "x": float(state[0]), "y": float(state[1]), "z": float(state[2]),
                    "roll":  float(state[3]), "pitch": float(state[4]), "yaw":   float(state[5]),
                    "grip_l": float(state[6]), "grip_r": float(state[7]),
                    "ep_len": ep_len,
                }
            n_eps += 1
        logger.info("Suite %-8s loaded %d episodes", suite, n_eps)
    logger.info("Total state rows indexed: %d (across %d episodes)",
                len(out), len(ep_lens))
    return out, ep_lens


def _augment_caption(caption: str, state: dict) -> str:
    """Append the new bullets to a caption (idempotent if already present)."""
    if "- gripper_xyz:" in caption:
        return caption
    progress = (state["ep_len"] and (state.get("frame", 0) / max(1, state["ep_len"] - 1)))
    bullets = [
        f"- gripper_xyz: ({state['x']:+.3f}, {state['y']:+.3f}, {state['z']:+.3f})",
        f"- gripper_state: {state['gripper_state']}",
        f"- progress: {progress:.2f}",
        f"- step: {state['frame']}",
    ]
    return caption.rstrip() + "\n" + "\n".join(bullets) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--lerobot-root", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    logger.info("Building parquet state index...")
    state_idx, ep_lens = _load_parquet_index(Path(args.lerobot_root))

    logger.info("Reading labels: %s", args.labels_in)
    n_in = n_out = n_skip = 0
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_in) as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            sid = d.get("meta", {}).get("source_example_id", "")
            if "__" not in sid:
                n_skip += 1; continue
            suite, rest = sid.split("__", 1)
            # rest = "traj000000_step000000"
            try:
                traj_part, step_part = rest.split("_step")
                ep_idx = int(traj_part.replace("traj", ""))
                frame = int(step_part)
            except Exception:
                n_skip += 1; continue
            key = (suite, ep_idx, frame)
            if key not in state_idx:
                n_skip += 1; continue
            s = dict(state_idx[key])
            s["frame"] = frame
            s["gripper_state"] = _gripper_state(np.array([s["grip_l"], s["grip_r"]]))
            d["description"] = _augment_caption(d["description"], s)
            # tag the new schema in meta for downstream debugging
            d.setdefault("meta", {})["label_schema"] = "v10_state_augmented"
            fout.write(json.dumps(d) + "\n")
            n_out += 1
            if n_out % 10000 == 0:
                logger.info("wrote %d so far...", n_out)
    logger.info("Done: %d in -> %d out (%d skipped) -> %s",
                n_in, n_out, n_skip, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

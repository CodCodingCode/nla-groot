"""Per-step state capture from a running LIBERO sim.

The :class:`ObjectStateLogger` queries MuJoCo for tracked body positions,
gripper EE pose, and gripper opening width after every sim step. The
trajectory is dumped to parquet on episode end (or a flat ``.json`` if
pandas/pyarrow isn't available).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


class ObjectStateLogger:
    """Wrap a :class:`LiberoEnv` instance, capture per-step state.

    Usage::

        env = LiberoEnv(..., suppress_done=True)
        logger = ObjectStateLogger(env, tracked_bodies=[...])
        obs, _ = env.reset(seed=0)
        logger.reset()
        ...
        for chunk in chunks:
            for sub_action in chunk:
                obs, reward, done, truncated, info = env.step(sub_action)
                logger.log_step(reward, info)
        traj = logger.to_dict()
        logger.write_parquet("trajectory.parquet")
    """

    def __init__(
        self,
        env: Any,
        tracked_bodies: Iterable[str],
        gripper_body: str = "gripper0_eef",
    ) -> None:
        self.env = env
        self.tracked_bodies = list(tracked_bodies)
        self.gripper_body = gripper_body
        # Track left + right finger tips so we can compute gripper width.
        self._left_finger = "gripper0_finger_joint1_tip"
        self._right_finger = "gripper0_finger_joint2_tip"
        self.reset()

    def reset(self) -> None:
        self.t = 0
        # Per-step lists: each entry is one sim step
        self._t: list[int] = []
        self._reward: list[float] = []
        self._success: list[bool] = []
        self._ee_pos: list[np.ndarray] = []
        self._ee_quat: list[np.ndarray] = []
        self._gripper_width: list[float] = []
        # body_name -> list[np.ndarray(3)]
        self._body_xpos: dict[str, list[np.ndarray]] = {n: [] for n in self.tracked_bodies}
        self._initial_body_pos: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internals — reach into robosuite
    # ------------------------------------------------------------------
    @property
    def _sim(self):
        return self.env._env.sim

    @property
    def _mj_model(self):
        return self._sim.model

    @property
    def _mj_data(self):
        return self._sim.data

    def _body_xyz(self, name: str) -> np.ndarray:
        bid = self._mj_model.body_name2id(name)
        return self._mj_data.body_xpos[bid].copy()

    def _body_xyzw(self, name: str) -> np.ndarray:
        bid = self._mj_model.body_name2id(name)
        return self._mj_data.body_xquat[bid].copy()

    def _gripper_width_now(self) -> float:
        try:
            lp = self._body_xyz(self._left_finger)
            rp = self._body_xyz(self._right_finger)
            return float(np.linalg.norm(lp - rp))
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def capture_initial(self) -> None:
        """Call once right after ``env.reset(...)`` to record t=0 positions."""
        for name in self.tracked_bodies:
            self._initial_body_pos[name] = self._body_xyz(name)

    def log_step(self, reward: float, info: dict) -> None:
        self._t.append(self.t)
        self._reward.append(float(reward))
        self._success.append(bool(info.get("success", False)))
        try:
            self._ee_pos.append(self._body_xyz(self.gripper_body))
            self._ee_quat.append(self._body_xyzw(self.gripper_body))
        except Exception:
            self._ee_pos.append(np.full(3, np.nan))
            self._ee_quat.append(np.full(4, np.nan))
        self._gripper_width.append(self._gripper_width_now())
        for name in self.tracked_bodies:
            try:
                self._body_xpos[name].append(self._body_xyz(name))
            except Exception:
                self._body_xpos[name].append(np.full(3, np.nan))
        self.t += 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.t,
            "t": np.array(self._t, dtype=np.int32),
            "reward": np.array(self._reward, dtype=np.float32),
            "success": np.array(self._success, dtype=bool),
            "ee_pos": np.stack(self._ee_pos) if self._ee_pos else np.zeros((0, 3)),
            "ee_quat": np.stack(self._ee_quat) if self._ee_quat else np.zeros((0, 4)),
            "gripper_width": np.array(self._gripper_width, dtype=np.float32),
            "body_xpos": {
                k: (np.stack(v) if v else np.zeros((0, 3)))
                for k, v in self._body_xpos.items()
            },
            "initial_body_pos": dict(self._initial_body_pos),
        }

    def write_parquet(self, path: str | Path) -> None:
        """Write trajectory as Parquet if pandas is available; else NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        try:
            import pandas as pd

            cols: dict[str, Any] = {
                "t": d["t"],
                "reward": d["reward"],
                "success": d["success"],
                "gripper_width": d["gripper_width"],
            }
            for ax, dim in enumerate("xyz"):
                cols[f"ee_{dim}"] = d["ee_pos"][:, ax]
            for body, xyz in d["body_xpos"].items():
                for ax, dim in enumerate("xyz"):
                    cols[f"{body}_{dim}"] = xyz[:, ax]
            df = pd.DataFrame(cols)
            df.to_parquet(path)
            # Also write a small companion json with the initial positions
            init = {k: v.tolist() for k, v in d["initial_body_pos"].items()}
            (path.with_suffix(".initial.json")).write_text(
                __import__("json").dumps(init, indent=2)
            )
        except Exception:
            np.savez_compressed(path.with_suffix(".npz"), **{
                "t": d["t"],
                "reward": d["reward"],
                "success": d["success"],
                "ee_pos": d["ee_pos"],
                "ee_quat": d["ee_quat"],
                "gripper_width": d["gripper_width"],
                **{f"body__{k}": v for k, v in d["body_xpos"].items()},
                **{f"init__{k}": v for k, v in d["initial_body_pos"].items()},
            })


def episode_summary(trajectory: dict[str, Any], target_body: str | None) -> dict[str, Any]:
    """Compute per-episode aggregate metrics from a logger trajectory."""
    body_xpos = trajectory["body_xpos"]
    ee_pos = trajectory["ee_pos"]
    init_pos = trajectory["initial_body_pos"]

    per_object_displacement: dict[str, float] = {}
    per_object_min_ee: dict[str, float] = {}
    for name, xyz in body_xpos.items():
        if xyz.shape[0] == 0 or name not in init_pos:
            continue
        per_object_displacement[name] = float(
            np.linalg.norm(xyz[-1] - init_pos[name])
        )
        if ee_pos.shape[0] == xyz.shape[0]:
            per_object_min_ee[name] = float(
                np.min(np.linalg.norm(xyz - ee_pos, axis=1))
            )

    displacement_winner = (
        max(per_object_displacement.items(), key=lambda kv: kv[1])[0]
        if per_object_displacement
        else None
    )

    summary = {
        "n_steps": int(trajectory["n_steps"]),
        "success_any": bool(trajectory["success"].any()) if trajectory["success"].size else False,
        "success_final": bool(trajectory["success"][-1]) if trajectory["success"].size else False,
        "displacement": per_object_displacement,
        "min_ee_distance": per_object_min_ee,
        "displacement_winner": displacement_winner,
    }
    if target_body and target_body in per_object_displacement:
        summary["target_displacement"] = per_object_displacement[target_body]
        summary["target_min_ee_distance"] = per_object_min_ee.get(target_body)
        summary["target_winner"] = displacement_winner == target_body
    return summary

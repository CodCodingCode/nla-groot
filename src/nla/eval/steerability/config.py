"""Config dataclasses for the steerability eval harness.

The harness reads a single YAML file describing one or more *conditions*
(steer settings, optionally with an AR) to compare on the same set of
LIBERO env + seed combinations. Each condition's policy server is brought
up once, then the harness drives :data:`seeds` rollouts per env.

Example YAML (also see ``scripts/eval/steerability_v1.yaml`` for the
checked-in version)::

    name: steerability_v1
    output_dir: data/eval/steerability_v1
    model_path: checkpoints/GR00T-N1.7-LIBERO/libero_goal
    embodiment_tag: LIBERO_PANDA
    envs:
      - libero_sim/put_the_bowl_on_the_plate
    seeds: [0, 1, 2, 3, 4]
    n_action_steps: 8
    max_episode_steps: 300
    suppress_done: true
    steps_per_render: 1

    conditions:
      - name: baseline
        ar_dir: null
        steer: null

      - name: steer_bowl_plate
        ar_dir: data/sft/libero_goal_pilot/ar
        steer:
          prompt_file: src/nla/eval/steerability/prompts/bowl_plate.txt
          placement: image_patch_all
          blend: 1.0
        target_body: akita_black_bowl_1_main

    av_eval:
      enabled: false
      ar_dirs: [data/sft/libero_goal_pilot/ar]
      video_keys: [image, wrist_image]
      per_position: 4
      datasets:
        - name: goal_only
          activations_root: data/activations/libero_goal_pilot
          labels_jsonl: data/labels/libero_goal_pilot/labels.jsonl
          frames_cache: data/labels/libero_goal_pilot/frames_cache

      # If ``datasets`` is omitted but ``activations_root`` / ``labels_jsonl``
      # / ``frames_cache`` appear at the ``av_eval`` top level, a single
      # ``default`` dataset is synthesised automatically.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


SteerPlacement = Literal[
    "last_text", "image_patch", "anchor", "image_patch_all", "fixed"
]


@dataclass
class SteerCfg:
    """Where + how to apply the steer vector."""

    prompt_file: Optional[str] = None
    prompt: Optional[str] = None
    placement: SteerPlacement = "image_patch_all"
    blend: float = 1.0
    fixed_token_index: Optional[int] = None
    image_patch_seed: int = 0

    def resolved_prompt(self) -> str:
        if self.prompt and self.prompt_file:
            raise ValueError("set only one of prompt / prompt_file")
        if self.prompt_file:
            return Path(self.prompt_file).read_text()
        if self.prompt:
            return self.prompt
        raise ValueError("SteerCfg needs either prompt or prompt_file")


@dataclass
class ConditionConfig:
    """One row of the comparison table. Maps to a single policy server."""

    name: str
    ar_dir: Optional[str] = None
    steer: Optional[SteerCfg] = None
    # Body name (in MuJoCo) we expect the steer to redirect the gripper toward.
    # Used only for the "target_displacement" / "target_min_ee_distance" metrics.
    # Leave None for unsteered baselines.
    target_body: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.ar_dir is None) != (self.steer is None):
            raise ValueError(
                f"condition {self.name}: ar_dir and steer must both be set "
                "(steered) or both None (baseline)"
            )


@dataclass
class AvJudgeDatasetConfig:
    """One AV LLM-as-judge run over a held-out label split."""

    name: str
    activations_root: str
    labels_jsonl: str
    frames_cache: str
    # If set, grade only these AR dirs; otherwise use parent ``AvFidelityConfig.ar_dirs``.
    ar_dirs: Optional[list[str]] = None
    per_position: Optional[int] = None
    held_out_fraction: Optional[float] = None
    split_by: Optional[str] = None


@dataclass
class AvFidelityConfig:
    """Optional AV-vs-ground-truth caption eval (one entry per AR ckpt)."""

    enabled: bool = False
    ar_dirs: list[str] = field(default_factory=list)
    activations_root: Optional[str] = None
    labels_jsonl: Optional[str] = None
    frames_cache: Optional[str] = None
    # When non-empty from YAML, run one judge eval per dataset (each row can narrow ar_dirs).
    datasets: list[AvJudgeDatasetConfig] = field(default_factory=list)
    video_keys: list[str] = field(default_factory=lambda: ["image", "wrist_image"])
    per_position: int = 4
    held_out_fraction: float = 0.05
    judge_model: Optional[str] = None
    split_by: str = "episode"


@dataclass
class SteerabilityConfig:
    name: str
    output_dir: str
    model_path: str
    embodiment_tag: str = "LIBERO_PANDA"
    envs: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    n_action_steps: int = 8
    max_episode_steps: int = 300
    suppress_done: bool = True
    steps_per_render: int = 1
    fps: int = 20
    policy_host: str = "0.0.0.0"
    policy_port: int = 5555
    conditions: list[ConditionConfig] = field(default_factory=list)
    av_eval: AvFidelityConfig = field(default_factory=AvFidelityConfig)
    # If set, overwrite ``metrics`` / judge slots inside this JSON (e.g. v3_scorecard.json).
    patch_scorecard: Optional[str] = None

    # Bodies tracked per step for the steerability metrics; defaults to the
    # LIBERO Goal kitchen scene's manipulable objects.
    tracked_bodies: list[str] = field(
        default_factory=lambda: [
            "akita_black_bowl_1_main",
            "plate_1_main",
            "wine_bottle_1_main",
            "cream_cheese_1_main",
            "wooden_cabinet_1_main",
            "flat_stove_1_main",
            "wine_rack_1_main",
        ]
    )


def _build_steer_cfg(d: dict | None) -> SteerCfg | None:
    if d is None:
        return None
    return SteerCfg(**d)


def _build_condition(d: dict) -> ConditionConfig:
    return ConditionConfig(
        name=d["name"],
        ar_dir=d.get("ar_dir"),
        steer=_build_steer_cfg(d.get("steer")),
        target_body=d.get("target_body"),
    )


def _build_av_dataset(raw: dict) -> AvJudgeDatasetConfig:
    return AvJudgeDatasetConfig(
        name=raw["name"],
        activations_root=raw["activations_root"],
        labels_jsonl=raw["labels_jsonl"],
        frames_cache=raw["frames_cache"],
        ar_dirs=raw.get("ar_dirs"),
        per_position=raw.get("per_position"),
        held_out_fraction=raw.get("held_out_fraction"),
        split_by=raw.get("split_by"),
    )


def _build_av(d: dict | None) -> AvFidelityConfig:
    if not d:
        return AvFidelityConfig()
    raw = dict(d)
    datasets_raw = raw.pop("datasets", None)
    field_names = {f.name for f in dataclasses.fields(AvFidelityConfig)}
    kwargs = {k: v for k, v in raw.items() if k in field_names and k != "datasets"}
    cfg = AvFidelityConfig(**kwargs)
    if datasets_raw:
        cfg.datasets = [_build_av_dataset(x) for x in datasets_raw]
    elif cfg.activations_root and cfg.labels_jsonl and cfg.frames_cache:
        cfg.datasets = [
            AvJudgeDatasetConfig(
                name="default",
                activations_root=cfg.activations_root,
                labels_jsonl=cfg.labels_jsonl,
                frames_cache=cfg.frames_cache,
            )
        ]
    return cfg


def load_config(path: str | Path) -> SteerabilityConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    conditions = [_build_condition(c) for c in raw.pop("conditions", [])]
    av = _build_av(raw.pop("av_eval", None))
    sc_field_names = {f.name for f in dataclasses.fields(SteerabilityConfig)}
    sc_kwargs = {k: v for k, v in raw.items() if k in sc_field_names}
    return SteerabilityConfig(conditions=conditions, av_eval=av, **sc_kwargs)

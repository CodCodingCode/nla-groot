"""Build GR00T-ready observation dicts from a LeRobot trajectory row.

Mirrors ``scripts/extraction/run_extract.py`` (``_parse_observation`` +
``_prepare_step_obs``) so eval scripts can ``import nla.steering.groot_obs``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def parse_observation_flat(
    obs_flat: dict[str, Any],
    modality_configs: Any,
) -> dict[str, dict[str, Any]]:
    """Flat parquet-style keys -> nested ``{video, state, language}`` dict."""
    new_obs: dict[str, dict[str, Any]] = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs_flat[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def build_observation_for_step(
    traj: Any,
    step_idx: int,
    modality_configs: Any,
    embodiment_tag: Any,
    language_keys: list[str],
    extract_step_data: Callable[..., Any],
    *,
    allow_padding: bool = True,
) -> dict[str, dict[str, Any]]:
    """One timestep from a LeRobot trajectory -> nested observation for ``Gr00tPolicy``."""
    data_point = extract_step_data(
        traj, step_idx, modality_configs, embodiment_tag, allow_padding=allow_padding
    )
    obs: dict[str, Any] = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for language_key in language_keys:
        obs[language_key] = data_point.text
    return parse_observation_flat(obs, modality_configs)

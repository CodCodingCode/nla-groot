"""Stack nested GR00T observations for batched policy inference."""

from __future__ import annotations

from typing import Any

import numpy as np


def infer_nested_batch_size(observation: dict[str, Any]) -> int:
    """Return B from a nested observation (video/state/language)."""
    video = observation.get("video") or {}
    if video:
        first = next(iter(video.values()))
        if isinstance(first, np.ndarray) and first.ndim >= 1:
            return int(first.shape[0])
    state = observation.get("state") or {}
    if state:
        first = next(iter(state.values()))
        if isinstance(first, np.ndarray) and first.ndim >= 1:
            return int(first.shape[0])
    language = observation.get("language") or {}
    if language:
        first = next(iter(language.values()))
        if isinstance(first, list):
            return len(first)
    return 1


def stack_nested_observations(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge single-sample nested observations into one batch (B = len(obs_list))."""
    if not obs_list:
        raise ValueError("obs_list must be non-empty")
    if len(obs_list) == 1:
        return obs_list[0]

    out: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    for modality in ("video", "state", "language"):
        keys: set[str] = set()
        for obs in obs_list:
            keys.update((obs.get(modality) or {}).keys())
        for key in sorted(keys):
            parts = []
            for obs in obs_list:
                bucket = obs.get(modality) or {}
                if key not in bucket:
                    raise KeyError(
                        f"observation missing {modality}.{key}; "
                        f"keys={list(bucket.keys())}"
                    )
                parts.append(bucket[key])
            if modality in ("video", "state"):
                out[modality][key] = np.concatenate(parts, axis=0)
            else:
                rows: list[list[str]] = []
                for part in parts:
                    if (
                        isinstance(part, list)
                        and len(part) == 1
                        and isinstance(part[0], list)
                    ):
                        rows.append(part[0])
                    elif isinstance(part, list) and part and isinstance(part[0], str):
                        rows.append(list(part))
                    else:
                        raise TypeError(
                            f"unexpected language[{key!r}] shape: {type(part)}"
                        )
                out[modality][key] = rows
    for required in ("video", "state", "language"):
        out.setdefault(required, {})
    return out

"""Shared helpers for measuring Δaction under backbone intervention.

Used by single-probe and sweep eval scripts so every causal-probe artifact
defines "Δaction" the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def policy_get_action(policy: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Run ``policy.get_action`` and return a flat ``{key: tensor-like}`` dict.

    Some GR00T revisions return a nested ``{modality: {key: tensor}}`` mapping;
    we flatten to dotted keys (``action.world_vector``, etc.) so downstream
    stats code does not have to recurse.
    """
    fn = getattr(policy, "get_action", None)
    if fn is None:
        raise RuntimeError(
            "Gr00tPolicy has no get_action(). Use an Isaac-GR00T revision that "
            "exposes policy.get_action(observation), or extend this script."
        )
    out = fn(observation)
    if isinstance(out, tuple) and len(out) >= 1:
        out = out[0]
    if not isinstance(out, dict):
        raise RuntimeError(f"Unexpected get_action return type: {type(out)}")
    if any(isinstance(v, dict) for v in out.values()):
        flat: dict[str, Any] = {}
        for k, v in out.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v
        return flat
    return out


def to_numpy(x: Any) -> np.ndarray:
    """Coerce a tensor or array to a numpy ndarray (empty if ``None``)."""
    if x is None:
        return np.array([])
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().float().numpy())
    return np.asarray(x)


def action_stats(baseline: dict[str, Any], steered: dict[str, Any]) -> dict[str, Any]:
    """Per-key max/mean/rms of ``(steered - baseline)`` plus ``global_max_abs``.

    Returns::

        {
            "per_modality_key": {key: {"max_abs", "mean_abs", "rms"} | {"error": str}},
            "global_max_abs": float,
        }

    Shape mismatches between baseline and steered for the same key are
    surfaced as ``{"error": ...}`` rather than raising, so a single bad head
    does not kill an entire sweep row.
    """
    keys = sorted(set(baseline.keys()) | set(steered.keys()))
    per: dict[str, Any] = {}
    all_abs: list[float] = []
    for k in keys:
        a = to_numpy(baseline.get(k))
        b = to_numpy(steered.get(k))
        if a.shape != b.shape:
            per[k] = {"error": f"shape mismatch {a.shape} vs {b.shape}"}
            continue
        diff = b.astype(np.float64) - a.astype(np.float64)
        per[k] = {
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "rms": float(np.sqrt(np.mean(diff**2))),
        }
        all_abs.extend(np.abs(diff).ravel().tolist())
    return {
        "per_modality_key": per,
        "global_max_abs": float(max(all_abs)) if all_abs else 0.0,
    }

"""Unit tests for ``nla.eval.steerability.rollout._to_server_obs``.

The CLI-level eval-v2 ``language_swap`` protocol depends on this helper
correctly overriding every language slot in the observation sent to
the GR00T policy server. If the override silently misses any slot, the
matched and mismatched intent arms share the native BDDL language
channel and the headline ``semantic_gap_predicate`` metric is
structurally zero.
"""

from __future__ import annotations

import numpy as np

from nla.eval.steerability.rollout import _to_server_obs


def _fake_libero_obs(task_text: str) -> dict:
    """Build a flat observation matching what LIBERO's env emits.

    The keys mirror the dotted-prefix shape ``_to_server_obs`` expects
    (``video.image``, ``state.x``, ``annotation.<...>``).
    """
    return {
        "video.image": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "state.x": np.zeros((6,), dtype=np.float32),
        "state.gripper": np.zeros((1,), dtype=np.float32),
        "annotation.human.action.task_description": task_text,
    }


def test_to_server_obs_passes_through_native_language_by_default():
    obs = _fake_libero_obs("put the bowl on the plate")
    out = _to_server_obs(obs)
    assert "language" in out
    key = "annotation.human.action.task_description"
    assert out["language"][key] == [["put the bowl on the plate"]]


def test_to_server_obs_override_replaces_every_language_slot():
    """The eval-v2 language_swap path overrides ALL keys in
    ``out["language"]`` so the policy sees the intent-arm text even if
    the loaded model registers multiple language slots.
    """
    obs = _fake_libero_obs("put the bowl on the plate")
    out = _to_server_obs(obs, policy_language_override="put the wine bottle on the rack")
    assert "language" in out and out["language"], (
        "language bucket must remain populated under the swap"
    )
    for k, v in out["language"].items():
        assert v == [["put the wine bottle on the rack"]], (
            f"language slot {k!r} was not overridden: {v}"
        )


def test_to_server_obs_empty_override_falls_back_to_native_language():
    """Empty strings / ``None`` must not override (matches the
    ``if policy_language_override:`` guard in the helper)."""
    obs = _fake_libero_obs("put the bowl on the plate")
    out_empty = _to_server_obs(obs, policy_language_override="")
    out_none = _to_server_obs(obs, policy_language_override=None)
    key = "annotation.human.action.task_description"
    assert out_empty["language"][key] == [["put the bowl on the plate"]]
    assert out_none["language"][key] == [["put the bowl on the plate"]]

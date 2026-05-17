"""Unit tests for ``nla.steering.action_delta`` (shared Δaction helpers)."""

from __future__ import annotations

import numpy as np
import torch

from nla.steering.action_delta import action_stats, policy_get_action, to_numpy


class _FakePolicy:
    def __init__(self, action):
        self._action = action

    def get_action(self, observation):
        del observation
        return self._action


def test_policy_get_action_flattens_nested_dict():
    nested = {
        "action": {"world_vector": torch.tensor([1.0, 2.0]), "gripper": torch.tensor([0.5])},
        "meta": torch.tensor([7.0]),
    }
    flat = policy_get_action(_FakePolicy(nested), observation={})
    assert set(flat.keys()) == {"action.world_vector", "action.gripper", "meta"}
    assert torch.allclose(flat["action.world_vector"], torch.tensor([1.0, 2.0]))


def test_policy_get_action_passes_flat_dict_unchanged():
    flat_in = {"a": torch.tensor([0.0]), "b": torch.tensor([1.0])}
    out = policy_get_action(_FakePolicy(flat_in), observation={})
    assert out is flat_in


def test_policy_get_action_unwraps_tuple_return():
    flat_in = {"a": torch.tensor([0.0])}
    policy = _FakePolicy((flat_in, "meta_ignored"))
    out = policy_get_action(policy, observation={})
    assert out is flat_in


def test_to_numpy_handles_none_and_tensor():
    assert to_numpy(None).shape == (0,)
    arr = to_numpy(torch.tensor([1.0, 2.0, 3.0]))
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_action_stats_basic_values_and_global_max():
    base = {"a": torch.tensor([0.0, 0.0]), "b": torch.tensor([1.0])}
    steered = {"a": torch.tensor([0.1, -0.3]), "b": torch.tensor([1.5])}
    stats = action_stats(base, steered)
    per = stats["per_modality_key"]
    assert set(per.keys()) == {"a", "b"}
    np.testing.assert_allclose(per["a"]["max_abs"], 0.3, atol=1e-6)
    np.testing.assert_allclose(per["a"]["mean_abs"], 0.2, atol=1e-6)
    np.testing.assert_allclose(per["b"]["max_abs"], 0.5, atol=1e-6)
    np.testing.assert_allclose(stats["global_max_abs"], 0.5, atol=1e-6)


def test_action_stats_records_shape_mismatch():
    base = {"a": torch.tensor([0.0, 0.0])}
    steered = {"a": torch.tensor([0.0, 0.0, 0.0])}
    stats = action_stats(base, steered)
    assert "error" in stats["per_modality_key"]["a"]
    assert stats["global_max_abs"] == 0.0


def test_action_stats_empty_inputs():
    assert action_stats({}, {}) == {"per_modality_key": {}, "global_max_abs": 0.0}

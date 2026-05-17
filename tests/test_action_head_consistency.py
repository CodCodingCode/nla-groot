"""Tests for the action-head consistency kernel.

These tests use a tiny in-process ``FakePolicy`` and a fake AR module so we
don't need to import GR00T. The goals are:

1. Hook plumbing: the differentiable steer hook actually replaces backbone
   features and the steered action differs from the baseline.
2. Gradients reach the AR: ``loss.backward()`` updates AR params.
3. The kernel respects ``every_n_steps`` and ``image_patch_rows_only``.
4. Caching: baseline computed once per example_id when caching is enabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nla.layer_spec import BACKBONE_EMBEDDING_DIM
from nla.training.action_head_consistency import (
    ActionConsistencyConfig,
    ActionConsistencyKernel,
    FakePolicy,
    attach_differentiable_backbone_steer,
    make_dummy_obs_builder,
)
from nla.steering.backbone_steer import (
    BackboneFeaturesSteerHook,
    SteerSpec,
    attach_backbone_steer,
)
from nla.training.replay_manifest import ReplayEntry, ReplayManifest


# ---------------------------------------------------------------------------
# A tiny AR stand-in: linear over the description's hashed token id.
# ---------------------------------------------------------------------------

class _TinyARCfg:
    alpha = 5.0


class _TinyAR(torch.nn.Module):
    """Minimal stand-in for ``ActivationReconstructor``.

    Behaves like ``ar(descriptions, device=...)`` and exposes ``cfg.alpha``.
    Returns an α-scaled vector that linearly depends on a hashed descriptor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(8, BACKBONE_EMBEDDING_DIM)
        self.cfg = _TinyARCfg()

    def _embed(self, descs: list[str], device: torch.device) -> torch.Tensor:
        vals = torch.tensor(
            [[(hash(d) % 100) / 100.0] * 8 for d in descs],
            dtype=torch.float32, device=device,
        )
        return vals

    def forward(self, descs: list[str], *, device: torch.device | None = None) -> torch.Tensor:
        device = device if device is not None else next(self.parameters()).device
        emb = self._embed(descs, device)
        return self.head(emb)


# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------

def test_differentiable_hook_replaces_backbone_features():
    policy = FakePolicy(seed=1)
    # Baseline: zero features in, zero pooled vector, zero action.
    baseline = policy.get_action({})
    assert torch.allclose(baseline["action.world_vector"], torch.zeros_like(baseline["action.world_vector"]))

    steer = torch.full((BACKBONE_EMBEDDING_DIM,), 0.5, requires_grad=True)
    with attach_differentiable_backbone_steer(
        policy.model.backbone, steer, SteerSpec(placement="image_patch_all"),
    ):
        steered = policy.get_action({})
    # Steered features != 0, so steered action != 0.
    assert not torch.allclose(
        steered["action.world_vector"],
        torch.zeros_like(steered["action.world_vector"]),
    )


def test_steer_gradients_flow_into_caller():
    policy = FakePolicy(seed=2)
    steer = torch.full((BACKBONE_EMBEDDING_DIM,), 0.1, requires_grad=True)
    with attach_differentiable_backbone_steer(
        policy.model.backbone, steer, SteerSpec(placement="image_patch_all"),
    ):
        out = policy.get_action({})
    loss = out["action.world_vector"].pow(2).sum()
    loss.backward()
    assert steer.grad is not None
    assert torch.any(steer.grad != 0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _make_manifest(example_ids: list[str]) -> ReplayManifest:
    return ReplayManifest(
        [
            ReplayEntry(eid, suite="goal", traj_idx=i, step_idx=i, dataset_root="/tmp")
            for i, eid in enumerate(example_ids)
        ]
    )


def test_kernel_zero_loss_when_weight_zero():
    cfg = ActionConsistencyConfig(weight=0.0)
    manifest = _make_manifest(["goal__traj000000_step000000"])
    ar = _TinyAR()
    kernel = ActionConsistencyKernel(
        cfg, manifest=manifest,
        policy_loader=lambda: FakePolicy(seed=3),
        obs_builder=make_dummy_obs_builder(),
        ar_module=ar,
        device="cpu",
    )
    loss, diag = kernel.consistency_loss(
        descriptions=["test"],
        example_ids=["goal__traj000000_step000000"],
        position_types=["image_patch"],
    )
    assert float(loss.item()) == 0.0
    assert diag.n_rows == 0


def test_kernel_skips_unknown_example_ids():
    cfg = ActionConsistencyConfig(weight=1.0)
    manifest = _make_manifest(["goal__traj000000_step000000"])
    ar = _TinyAR()
    kernel = ActionConsistencyKernel(
        cfg, manifest=manifest,
        policy_loader=lambda: FakePolicy(seed=3),
        obs_builder=make_dummy_obs_builder(),
        ar_module=ar,
        device="cpu",
    )
    loss, diag = kernel.consistency_loss(
        descriptions=["test"],
        example_ids=["unrelated_example"],
        position_types=["image_patch"],
    )
    assert float(loss.item()) == 0.0
    assert diag.n_rows == 0


def test_kernel_skips_non_image_patch_rows_by_default():
    cfg = ActionConsistencyConfig(weight=1.0, image_patch_rows_only=True)
    manifest = _make_manifest(["goal__traj000000_step000000"])
    ar = _TinyAR()
    kernel = ActionConsistencyKernel(
        cfg, manifest=manifest,
        policy_loader=lambda: FakePolicy(seed=3),
        obs_builder=make_dummy_obs_builder(),
        ar_module=ar,
        device="cpu",
    )
    loss, diag = kernel.consistency_loss(
        descriptions=["test"],
        example_ids=["goal__traj000000_step000000"],
        position_types=["last_text"],
    )
    assert float(loss.item()) == 0.0
    assert diag.n_rows == 0


def test_kernel_computes_loss_and_backpropagates():
    cfg = ActionConsistencyConfig(weight=1.0, max_microbatch_per_step=2)
    manifest = _make_manifest(
        ["goal__traj000000_step000000", "goal__traj000001_step000000"]
    )
    ar = _TinyAR()
    kernel = ActionConsistencyKernel(
        cfg, manifest=manifest,
        policy_loader=lambda: FakePolicy(seed=5),
        obs_builder=make_dummy_obs_builder(),
        ar_module=ar,
        device="cpu",
    )
    loss, diag = kernel.consistency_loss(
        descriptions=["red bowl", "blue plate"],
        example_ids=[
            "goal__traj000000_step000000",
            "goal__traj000001_step000000",
        ],
        position_types=["image_patch", "image_patch"],
    )
    assert diag.n_rows == 2
    assert diag.baseline_cache_misses == 2
    assert float(loss.item()) > 0.0
    assert torch.is_tensor(loss) and loss.requires_grad
    # gradients should reach the AR head.
    head_before = ar.head.weight.detach().clone()
    loss.backward()
    assert ar.head.weight.grad is not None
    assert torch.any(ar.head.weight.grad != 0)
    # Sanity: AR weights themselves haven't been modified by ``backward``,
    # just have non-None grads.
    assert torch.allclose(ar.head.weight.detach(), head_before)


def test_kernel_caches_baseline_actions():
    cfg = ActionConsistencyConfig(
        weight=1.0, max_microbatch_per_step=1, cache_baseline_actions=True,
    )
    manifest = _make_manifest(["goal__traj000000_step000000"])
    ar = _TinyAR()
    kernel = ActionConsistencyKernel(
        cfg, manifest=manifest,
        policy_loader=lambda: FakePolicy(seed=7),
        obs_builder=make_dummy_obs_builder(),
        ar_module=ar,
        device="cpu",
    )
    _, diag1 = kernel.consistency_loss(
        descriptions=["d"],
        example_ids=["goal__traj000000_step000000"],
        position_types=["image_patch"],
    )
    assert diag1.baseline_cache_misses == 1
    assert diag1.baseline_cache_hits == 0
    _, diag2 = kernel.consistency_loss(
        descriptions=["d"],
        example_ids=["goal__traj000000_step000000"],
        position_types=["image_patch"],
    )
    assert diag2.baseline_cache_misses == 0
    assert diag2.baseline_cache_hits == 1


def test_differentiable_hook_parity_with_production_hook():
    """``DifferentiableBackboneSteerHook`` and ``BackboneFeaturesSteerHook`` must
    produce identical steered features for the same inputs; the only difference
    is that ours preserves the autograd graph. Regression-tests the steering
    pipeline used by ``scripts/eval/nla_steer_groot_action.py``."""
    policy_diff = FakePolicy(seed=11)
    policy_prod = FakePolicy(seed=11)
    steer = torch.full((BACKBONE_EMBEDDING_DIM,), 0.25)
    spec = SteerSpec(placement="image_patch_all", blend=1.0)
    with attach_differentiable_backbone_steer(
        policy_diff.model.backbone, steer.detach().clone(), spec
    ):
        out_diff = policy_diff.get_action({})
    with attach_backbone_steer(policy_prod.model.backbone, steer.clone(), spec):
        out_prod = policy_prod.get_action({})
    assert set(out_diff.keys()) == set(out_prod.keys())
    for k in out_diff:
        assert torch.allclose(
            out_diff[k].detach(), out_prod[k].detach(), atol=1e-6
        ), f"divergence at {k}: {out_diff[k]} vs {out_prod[k]}"


def test_kernel_rejects_invalid_config():
    with pytest.raises(ValueError):
        ActionConsistencyKernel(
            ActionConsistencyConfig(weight=-1.0),
            manifest=_make_manifest([]),
            policy_loader=lambda: FakePolicy(),
            obs_builder=make_dummy_obs_builder(),
            ar_module=_TinyAR(),
        )
    with pytest.raises(ValueError):
        ActionConsistencyKernel(
            ActionConsistencyConfig(every_n_steps=0),
            manifest=_make_manifest([]),
            policy_loader=lambda: FakePolicy(),
            obs_builder=make_dummy_obs_builder(),
            ar_module=_TinyAR(),
        )

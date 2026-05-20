"""V5 SFT architecture tests: context prompt + multi-slot collate + strided_multi.

Complements the per-component smoke tests in ``tests/test_models_smoke.py``
and ``tests/test_sft_smoke.py``. These tests exercise the new
``LabeledPositionSample`` fields, the K-slot collate path, and the
``strided_image_multi`` read-time pooling end-to-end at the dataset boundary
without needing the full Qwen backbone.
"""

from __future__ import annotations

import torch

from nla.extraction.position_strategies import apply as apply_strategy
from nla.extraction.position_strategies import strided_image_multi
from nla.training.dataset import (
    LabeledPositionSample,
    collate_labeled_positions,
)


# ---------------------------------------------------------------------------
# strided_image_multi
# ---------------------------------------------------------------------------

def test_strided_image_multi_returns_k_by_h():
    T, H, K = 32, 8, 8
    features = torch.randn(T, H)
    image_mask = torch.zeros(T, dtype=torch.bool)
    image_mask[4:28] = True  # 24 image-patch tokens
    attention_mask = torch.ones(T, dtype=torch.bool)
    out = strided_image_multi(features, image_mask, attention_mask, k=K)
    assert out.shape == (K, H)
    # Each row of the output is one of the original feature rows (not a mix).
    image_features = features[image_mask]
    for k in range(K):
        diffs = (image_features - out[k].unsqueeze(0)).abs().sum(dim=-1)
        assert diffs.min().item() == 0.0


def test_strided_image_multi_falls_back_to_available_when_fewer_than_k():
    T, H, K = 8, 4, 8
    features = torch.randn(T, H)
    image_mask = torch.zeros(T, dtype=torch.bool)
    image_mask[:3] = True  # only 3 image patches
    attention_mask = torch.ones(T, dtype=torch.bool)
    out = strided_image_multi(features, image_mask, attention_mask, k=K)
    assert out.shape == (3, H)


def test_apply_dispatches_strided_image_multi_with_k():
    features = torch.randn(20, 5)
    image_mask = torch.zeros(20, dtype=torch.bool)
    image_mask[5:18] = True
    attention_mask = torch.ones(20, dtype=torch.bool)
    out = apply_strategy(
        "strided_image_multi", features, image_mask, attention_mask, k=4,
    )
    assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# Collate: mixed single + multi slot batches
# ---------------------------------------------------------------------------

def _make_sample(
    activation: torch.Tensor,
    *,
    position_type: str,
    description: str,
    activation_ar: torch.Tensor | None = None,
    step_index: int | None = None,
    instruction: str | None = None,
) -> LabeledPositionSample:
    return LabeledPositionSample(
        activation=activation,
        position_type=position_type,
        position_index=0,
        seq_len=16,
        description=description,
        example_id="ex0",
        label_example_id="ex0_label",
        episode_index=0,
        quality_weight=1.0,
        activation_ar=activation_ar if activation_ar is not None else activation,
        step_index=step_index,
        instruction=instruction,
    )


def test_collate_single_slot_only_keeps_legacy_shape():
    H = 4
    batch = [
        _make_sample(torch.ones(H), position_type="last_text", description="- a"),
        _make_sample(torch.zeros(H), position_type="last_text", description="- b"),
    ]
    out = collate_labeled_positions(batch)
    assert out["activations_av"].shape == (2, H)
    assert out["activations_ar"].shape == (2, H)
    assert out["activations"].shape == (2, H)
    assert out["activation_slot_mask"].shape == (2, 1)
    assert out["activation_slot_mask"].all()


def test_collate_mixed_batch_pads_k_and_masks():
    H, K = 4, 8
    single = _make_sample(torch.ones(H), position_type="last_text", description="- a")
    multi_vec = torch.arange(K * H, dtype=torch.float32).reshape(K, H)
    multi_ar = multi_vec.mean(dim=0)
    multi = _make_sample(
        multi_vec,
        position_type="image_patch",
        description="- b",
        activation_ar=multi_ar,
    )
    out = collate_labeled_positions([single, multi])
    # ``activations_av`` is padded to (B, K_max, H); single-slot row uses
    # slot 0 only and zero-pads the rest.
    assert out["activations_av"].shape == (2, K, H)
    assert torch.allclose(out["activations_av"][0, 0], torch.ones(H))
    assert torch.all(out["activations_av"][0, 1:] == 0)
    assert torch.allclose(out["activations_av"][1], multi_vec)
    # ``activations_ar`` is always (B, H) and equals the per-row AR vector.
    assert out["activations_ar"].shape == (2, H)
    assert torch.allclose(out["activations_ar"][1], multi_ar)
    # Slot mask records the real-vs-padding pattern; slot_count records K_i.
    expected_mask = torch.zeros(2, K, dtype=torch.bool)
    expected_mask[0, 0] = True
    expected_mask[1, :] = True
    assert torch.equal(out["activation_slot_mask"], expected_mask)
    assert torch.equal(out["activation_slot_count"], torch.tensor([1, K]))


def test_collate_carries_step_index_and_instruction():
    H = 4
    batch = [
        _make_sample(
            torch.zeros(H),
            position_type="last_text",
            description="- a",
            step_index=12,
            instruction="pick up the bowl",
        ),
        _make_sample(
            torch.zeros(H),
            position_type="last_text",
            description="- b",
            step_index=None,
            instruction=None,
        ),
    ]
    out = collate_labeled_positions(batch)
    assert out["step_index"].tolist() == [12, -1]
    assert out["instruction"] == ["pick up the bowl", None]

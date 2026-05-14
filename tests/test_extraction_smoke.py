"""Smoke test for the extraction pipeline.

Runs end-to-end (hook -> writer -> reader -> stats) against a fake module that
mimics ``Qwen3Backbone.forward``. Does NOT load real GR00T weights so it can
run on CPU and in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from nla.extraction import (
    ActivationShardReader,
    ActivationShardWriter,
    BackboneFeatureHook,
    PositionType,
    RunManifest,
    attach_hooks,
    compute_stats,
    iter_image_positions,
    sample_position,
)
from nla.layer_spec import BACKBONE_EMBEDDING_DIM


class _FakeQwen3Backbone(nn.Module):
    """A stand-in for gr00t.model.modules.qwen3_backbone.Qwen3Backbone.

    Returns the same BatchFeature shape: backbone_features [B,T,2048], plus
    backbone_attention_mask [B,T] bool, image_mask [B,T] bool. We add a
    trainable parameter so we have a real module to register the hook on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(BACKBONE_EMBEDDING_DIM, BACKBONE_EMBEDDING_DIM, bias=False)

    def forward(self, vl_input: dict) -> BatchFeature:
        x = vl_input["features"]
        attention_mask = vl_input["attention_mask"].bool()
        image_mask = vl_input["image_mask"].bool()
        features = self.proj(x)
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )


def _make_fake_batch(B: int, T: int, *, n_image: int, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(B, T, BACKBONE_EMBEDDING_DIM)
    # First n_image tokens per row are image patches.
    image_mask = torch.zeros(B, T, dtype=torch.bool)
    image_mask[:, :n_image] = True
    # All tokens valid (no padding).
    attention_mask = torch.ones(B, T, dtype=torch.bool)
    return {"features": x, "attention_mask": attention_mask, "image_mask": image_mask}


def test_hook_captures_correct_shapes():
    mod = _FakeQwen3Backbone().eval()
    hook = BackboneFeatureHook(to_cpu=True, store_dtype=torch.float32)
    batch = _make_fake_batch(B=2, T=64, n_image=32)

    with attach_hooks(mod, hook):
        _ = mod(batch)

    assert hook.last is not None
    assert hook.last.features.shape == (2, 64, BACKBONE_EMBEDDING_DIM)
    assert hook.last.attention_mask.shape == (2, 64)
    assert hook.last.image_mask.shape == (2, 64)
    assert hook.last.features.dtype == torch.float32
    assert hook.last.attention_mask.dtype == torch.bool
    assert hook.last.image_mask.dtype == torch.bool


def test_hook_handle_removed_on_exit():
    mod = _FakeQwen3Backbone().eval()
    hook = BackboneFeatureHook()
    with attach_hooks(mod, hook):
        pass
    # After exit there should be no live forward hooks remaining.
    assert all(h not in mod._forward_hooks.values() for h in [hook])
    assert hook._handle is None


def test_sample_position_respects_masks():
    rng = np.random.default_rng(7)
    attention_mask = torch.tensor([1] * 20, dtype=torch.bool)
    image_mask = torch.tensor([True] * 8 + [False] * 12, dtype=torch.bool)

    counts = {pt: 0 for pt in PositionType}
    for _ in range(2000):
        sp = sample_position(attention_mask, image_mask, rng=rng)
        counts[sp.type] += 1
        if sp.type == PositionType.IMAGE_PATCH:
            assert image_mask[sp.index].item() is True
        elif sp.type == PositionType.LAST_TEXT:
            assert image_mask[sp.index].item() is False
            assert attention_mask[sp.index].item() is True
        elif sp.type == PositionType.ANCHOR:
            assert sp.index == 19  # last non-pad token

    # All three categories should fire roughly per POSITION_MIX.
    total = sum(counts.values())
    assert counts[PositionType.IMAGE_PATCH] / total > 0.2
    assert counts[PositionType.LAST_TEXT] / total > 0.2
    assert counts[PositionType.ANCHOR] / total > 0.1


def test_sample_position_image_only_falls_back():
    """All tokens are image patches: last_text fallback must still produce a position."""
    rng = np.random.default_rng(0)
    attention_mask = torch.ones(10, dtype=torch.bool)
    image_mask = torch.ones(10, dtype=torch.bool)
    # Force the sampler to attempt last_text by zeroing the other weights:
    sp = sample_position(
        attention_mask, image_mask, mix={"last_text": 1.0}, rng=rng,
    )
    # last_text has no valid index, so we fall back to anchor (last non-pad token).
    assert sp.index == 9
    assert sp.type in (PositionType.ANCHOR, PositionType.FALLBACK)


def test_iter_image_positions():
    attn = torch.ones(8, dtype=torch.bool)
    img = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0], dtype=torch.bool)
    assert iter_image_positions(attn, img) == [1, 2, 4, 5, 6]


def test_writer_and_reader_roundtrip(tmp_path: Path):
    manifest = RunManifest(
        schema_version=1,
        model_repo="fake",
        layer_module_path="fake.module",
        hidden_size=BACKBONE_EMBEDDING_DIM,
        activation_dtype="float32",
        embodiment_tag="FAKE_EMBODIMENT",
    )
    writer = ActivationShardWriter(tmp_path, manifest, max_examples_per_shard=4)

    seqs = []
    for i in range(10):  # 10 examples, variable T, image-token-fraction 0.5
        T = 16 + i * 4
        f = torch.randn(T, BACKBONE_EMBEDDING_DIM)
        attn = torch.ones(T, dtype=torch.bool)
        img = torch.zeros(T, dtype=torch.bool)
        img[: T // 2] = True
        ids = torch.arange(T, dtype=torch.int64)
        seqs.append((f, attn, img, ids))
        writer.write(
            example_id=f"ex_{i:04d}",
            features=f,
            attention_mask=attn,
            image_mask=img,
            input_ids=ids,
            task_index=i % 3,
            task_text=f"task_{i % 3}",
            episode_index=i,
            step_index=0,
            embodiment_tag="FAKE_EMBODIMENT",
        )
    writer.close()

    # Multi-shard layout: 10 examples / 4 per shard = 3 shards (4 + 4 + 2).
    shard_dirs = sorted(tmp_path.glob("shard_*"))
    assert len(shard_dirs) == 3

    reader = ActivationShardReader(tmp_path)
    assert len(reader) == 10
    for i, (f, attn, img, ids) in enumerate(seqs):
        item = reader[i]
        torch.testing.assert_close(item["features"], f)
        torch.testing.assert_close(item["attention_mask"], attn)
        torch.testing.assert_close(item["image_mask"], img)
        torch.testing.assert_close(item["input_ids"], ids)
        rec = item["_record"]
        assert rec.example_id == f"ex_{i:04d}"
        assert rec.task_text == f"task_{i % 3}"
        assert rec.seq_len == 16 + i * 4

    # iter_examples streams in shard order.
    streamed = list(reader.iter_examples())
    assert len(streamed) == 10
    streamed_ids = [s["_record"].example_id for s in streamed]
    assert streamed_ids == [f"ex_{i:04d}" for i in range(10)]


def test_writer_rejects_dim_mismatch(tmp_path: Path):
    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="x",
        hidden_size=BACKBONE_EMBEDDING_DIM, activation_dtype="float32", embodiment_tag=None,
    )
    writer = ActivationShardWriter(tmp_path, manifest)
    with pytest.raises(ValueError, match="hidden_size mismatch"):
        writer.write(
            example_id="bad",
            features=torch.randn(4, BACKBONE_EMBEDDING_DIM + 1),
            attention_mask=torch.ones(4, dtype=torch.bool),
            image_mask=torch.zeros(4, dtype=torch.bool),
        )


def test_compute_stats_alpha_matches_p75(tmp_path: Path):
    """End-to-end: hook -> writer -> reader -> compute_stats, then verify α = P75."""
    mod = _FakeQwen3Backbone().eval()
    hook = BackboneFeatureHook(to_cpu=True, store_dtype=torch.float32)

    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="x",
        hidden_size=BACKBONE_EMBEDDING_DIM, activation_dtype="float32", embodiment_tag=None,
    )
    writer = ActivationShardWriter(tmp_path, manifest, max_examples_per_shard=8)

    all_norms = []
    with attach_hooks(mod, hook):
        for i in range(6):
            T = 24 + i * 8
            batch = _make_fake_batch(B=1, T=T, n_image=T // 2, seed=i)
            _ = mod(batch)
            f = hook.last.features[0]
            attn = hook.last.attention_mask[0]
            img = hook.last.image_mask[0]
            writer.write(
                example_id=f"ex_{i:02d}", features=f,
                attention_mask=attn, image_mask=img,
            )
            all_norms.append(torch.linalg.vector_norm(f, dim=-1).cpu().numpy())
    writer.close()

    expected_p75 = float(np.percentile(np.concatenate(all_norms), 75))

    reader = ActivationShardReader(tmp_path)
    stats = compute_stats(reader)
    assert stats.n_positions == sum(n.size for n in all_norms)
    np.testing.assert_allclose(stats.alpha, expected_p75, rtol=1e-5)
    np.testing.assert_allclose(stats.p75_norm, expected_p75, rtol=1e-5)
    # image_token_fraction = 0.5 by construction.
    np.testing.assert_allclose(stats.image_token_fraction, 0.5, atol=1e-6)

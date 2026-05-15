"""Smoke tests for the warm-start SFT pipeline.

These tests synthesize a tiny extraction dump + matching labels.jsonl and
run a few SFT steps with a *tiny* AV/AR built on a randomly-initialized
Qwen3 model (hidden=64, 4 layers).  Goal: catch wiring bugs without
touching the 4B base model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from nla.extraction import ActivationShardWriter, RunManifest
from nla.layer_spec import BACKBONE_EMBEDDING_DIM
from nla.models import (
    ActivationReconstructor,
    ActivationVerbalizer,
    ARConfig,
    AVConfig,
)
from nla.training import LabeledPositionDataset, collate_labeled_positions, run_sft
from nla.training.sft import SFTConfig


# Reduce hidden dim used by both the activation source AND the models so the
# data flow matches end-to-end (since both must agree on `activation_dim`).
TINY_ACTIVATION_DIM = 32
TINY_HIDDEN = 32
TINY_LAYERS = 2
TINY_HEADS = 4


def _write_synthetic_dump(out_root: Path, n_examples: int, T: int = 16) -> None:
    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="fake",
        hidden_size=TINY_ACTIVATION_DIM, activation_dtype="float32",
        embodiment_tag="FAKE",
    )
    writer = ActivationShardWriter(out_root, manifest, max_examples_per_shard=64)
    for i in range(n_examples):
        features = torch.randn(T, TINY_ACTIVATION_DIM)
        attn = torch.ones(T, dtype=torch.bool)
        img = torch.zeros(T, dtype=torch.bool)
        img[: T // 2] = True
        ids = torch.tensor([999] * (T // 2) + list(range(T - T // 2)))
        writer.write(
            example_id=f"traj0_step{i:04d}",
            features=features, attention_mask=attn, image_mask=img,
            input_ids=ids,
            episode_index=0, step_index=i,
            task_text="Test task", embodiment_tag="FAKE",
        )
    writer.close()


def _write_synthetic_labels(out_path: Path, examples: list[tuple[str, int, str]]) -> None:
    """Write a labels.jsonl with one row per (example_id, position, description)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for source_id, pos, desc in examples:
            row = {
                "example_id": f"{source_id}@p{pos:03d}_anchor",
                "description": desc,
                "model": "fake",
                "elapsed_ms": 1.0,
                "usage": {"total_tokens": 1},
                "error": None,
                "kind": "position",
                "meta": {
                    "source_example_id": source_id,
                    "position_index": pos,
                    "position_type": "anchor",
                    "seq_len": 16,
                    "episode_index": 0,
                    "step_index": 0,
                    "image_patch_meta": None,
                    "instruction": "Test task",
                },
            }
            f.write(json.dumps(row) + "\n")


def _make_tiny_models(alpha: float = 5.0):
    """Build AV/AR sharing a tiny Qwen3 base (random weights) + real tokenizer."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_HIDDEN * 2,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_HEADS,
        max_position_embeddings=512,
        rope_theta=1_000_000.0,
        torch_dtype="float32",
    )
    base_av = Qwen3ForCausalLM(cfg)
    base_ar = Qwen3ForCausalLM(cfg)
    av_cfg = AVConfig(
        activation_dim=TINY_ACTIVATION_DIM, alpha=alpha, dtype="float32",
        lora_rank=4, lora_alpha=8,
    )
    ar_cfg = ARConfig(
        activation_dim=TINY_ACTIVATION_DIM, alpha=alpha, dtype="float32",
        truncate_to_n_layers=1, lora_rank=4, lora_alpha=8,
    )
    av = ActivationVerbalizer(av_cfg, tokenizer=tok, base_model=base_av, apply_lora=True)
    ar = ActivationReconstructor(ar_cfg, tokenizer=av.tokenizer, base_model=base_ar, apply_lora=True)
    return av, ar


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_labeled_position_dataset_basic(tmp_path: Path):
    _write_synthetic_dump(tmp_path / "act", n_examples=4)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [
            ("traj0_step0000", 3, "- scene: a\n- target: b"),
            ("traj0_step0001", 5, "- scene: c\n- target: d"),
            ("traj0_step0002", 7, "- scene: e\n- target: f"),
            ("traj0_step0003", 9, "- scene: g\n- target: h"),
        ],
    )
    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl", seed=0, held_out_fraction=0.0,
    )
    assert len(ds) == 4
    sample = ds[0]
    assert sample.activation.shape == (TINY_ACTIVATION_DIM,)
    assert sample.description.startswith("- scene")
    assert sample.position_type == "anchor"


def test_labeled_position_dataset_drops_orphan_labels(tmp_path: Path):
    _write_synthetic_dump(tmp_path / "act", n_examples=2)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [
            ("traj0_step0000", 1, "ok"),
            ("traj0_step0099", 2, "orphan"),  # no matching activation
        ],
    )
    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl", seed=0, held_out_fraction=0.0,
    )
    assert len(ds) == 1
    assert ds[0].example_id == "traj0_step0000"


def test_collate_labeled_positions(tmp_path: Path):
    _write_synthetic_dump(tmp_path / "act", n_examples=3)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [(f"traj0_step{i:04d}", i + 1, f"- scene: {i}") for i in range(3)],
    )
    ds = LabeledPositionDataset(tmp_path / "act", tmp_path / "labels.jsonl", seed=0)
    batch = collate_labeled_positions([ds[0], ds[1], ds[2]])
    assert batch["activations"].shape == (3, TINY_ACTIVATION_DIM)
    assert len(batch["description"]) == 3
    assert batch["position_type"] == ["anchor", "anchor", "anchor"]


def test_strict_episode_split_raises_when_one_episode(tmp_path: Path):
    """With only one episode, strict mode must refuse to silently row-split."""
    import pytest

    _write_synthetic_dump(tmp_path / "act", n_examples=6)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [(f"traj0_step{i:04d}", i + 1, f"d{i}") for i in range(6)],
    )
    with pytest.raises(RuntimeError, match="episode"):
        LabeledPositionDataset(
            tmp_path / "act", tmp_path / "labels.jsonl",
            seed=0, held_out_fraction=0.2, held_out=False,
            split_by="episode",
            allow_episode_split_row_fallback=False,
        )


def test_held_out_split_is_deterministic(tmp_path: Path):
    _write_synthetic_dump(tmp_path / "act", n_examples=10)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [(f"traj0_step{i:04d}", i + 1, f"d{i}") for i in range(10)],
    )
    train = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.2, held_out=False,
        allow_episode_split_row_fallback=True,
    )
    val = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.2, held_out=True,
        allow_episode_split_row_fallback=True,
    )
    assert len(train) + len(val) == 10
    train_ids = {ds.label_example_id for ds in (train[i] for i in range(len(train)))}
    val_ids = {ds.label_example_id for ds in (val[i] for i in range(len(val)))}
    assert train_ids.isdisjoint(val_ids)


# ---------------------------------------------------------------------------
# Joint training step (no full run_sft, just one update)
# ---------------------------------------------------------------------------

def test_joint_step_decreases_loss(tmp_path: Path):
    """One mini-batch trained over a few steps: combined loss should drop."""
    av, ar = _make_tiny_models(alpha=5.0)
    B = 2
    acts = torch.randn(B, TINY_ACTIVATION_DIM)
    pos_types = ["image_patch", "last_text"]
    descs = ["- scene: a small table\n- target: blue cube", "- scene: floor\n- target: cup"]

    trainable = [p for p in list(av.parameters()) + list(ar.parameters()) if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-3)

    losses = []
    for _ in range(8):
        av_out = av.forward_sft(activations=acts, position_types=pos_types, target_texts=descs)
        ar_loss, _ = ar.forward_sft(descs, acts)
        loss = av_out.loss + ar_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().item()))
    assert losses[-1] < losses[0], f"losses did not decrease: {losses}"


# ---------------------------------------------------------------------------
# Full run_sft on tiny synthetic data
# ---------------------------------------------------------------------------

def test_run_sft_end_to_end_tiny(tmp_path: Path, monkeypatch):
    """Wire-test that run_sft can complete a few steps with a tiny model.

    We replace AV/AR model construction with tiny variants via monkeypatch so
    we don't load the real 4B base.
    """
    _write_synthetic_dump(tmp_path / "act", n_examples=4)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [
            ("traj0_step0000", 3, "- scene: a\n- target: b"),
            ("traj0_step0001", 5, "- scene: c\n- target: d"),
            ("traj0_step0002", 7, "- scene: e\n- target: f"),
            ("traj0_step0003", 9, "- scene: g\n- target: h"),
        ],
    )

    av_template, ar_template = _make_tiny_models(alpha=5.0)

    def _fake_build_models(_cfg):
        return av_template, ar_template

    monkeypatch.setattr("nla.training.sft._build_models", _fake_build_models)

    cfg = SFTConfig(
        activations_root=str(tmp_path / "act"),
        labels_jsonl=str(tmp_path / "labels.jsonl"),
        output_dir=str(tmp_path / "sft_out"),
        av_cfg=AVConfig(activation_dim=TINY_ACTIVATION_DIM, alpha=5.0, dtype="float32"),
        ar_cfg=ARConfig(activation_dim=TINY_ACTIVATION_DIM, alpha=5.0, dtype="float32",
                        truncate_to_n_layers=1),
        device="cpu",
        batch_size=2,
        total_steps=4,
        warmup_steps=1,
        eval_every=2,
        save_every=10_000,           # don't save during this short run
        log_every=1,
        held_out_fraction=0.25,
        gradient_checkpointing=False,
    )
    summary = run_sft(cfg)
    assert summary["steps"] == 4
    metrics_jsonl = tmp_path / "sft_out" / "metrics.jsonl"
    assert metrics_jsonl.exists()
    rows = [json.loads(l) for l in metrics_jsonl.read_text().splitlines() if l.strip()]
    train_rows = [r for r in rows if r["phase"] == "train"]
    assert len(train_rows) >= 1
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in train_rows)


def test_run_sft_logs_ar_mix_and_nce(tmp_path: Path, monkeypatch):
    """Tiny run with AR-AV mixing + contrastive loss logs expected keys/values."""
    _write_synthetic_dump(tmp_path / "act", n_examples=8)
    _write_synthetic_labels(
        tmp_path / "labels.jsonl",
        [(f"traj0_step{i:04d}", i + 1, f"- scene: s{i}\n- target: t{i}") for i in range(8)],
    )

    av_template, ar_template = _make_tiny_models(alpha=5.0)

    def _fake_build_models(_cfg):
        return av_template, ar_template

    monkeypatch.setattr("nla.training.sft._build_models", _fake_build_models)

    cfg = SFTConfig(
        activations_root=str(tmp_path / "act"),
        labels_jsonl=str(tmp_path / "labels.jsonl"),
        output_dir=str(tmp_path / "sft_out_mix"),
        av_cfg=AVConfig(activation_dim=TINY_ACTIVATION_DIM, alpha=5.0, dtype="float32"),
        ar_cfg=ARConfig(
            activation_dim=TINY_ACTIVATION_DIM,
            alpha=5.0,
            dtype="float32",
            truncate_to_n_layers=1,
        ),
        device="cpu",
        seed=0,
        batch_size=2,
        total_steps=24,
        warmup_steps=1,
        eval_every=1000,            # train rows only for this test
        save_every=10_000,
        log_every=1,
        held_out_fraction=0.25,
        gradient_checkpointing=False,
        ar_contrastive_weight=0.5,
        ar_av_mix_max=1.0,
        ar_av_mix_warmup_frac=0.0,
        ar_av_mix_max_new_tokens=24,
        ar_av_mix_do_sample=False,
    )

    run_sft(cfg)

    metrics_jsonl = tmp_path / "sft_out_mix" / "metrics.jsonl"
    rows = [json.loads(l) for l in metrics_jsonl.read_text().splitlines() if l.strip()]
    train_rows = [r for r in rows if r.get("phase") == "train"]
    assert train_rows, "expected train rows in metrics.jsonl"

    # Schema contract for the new tracking keys.
    for r in train_rows:
        assert "p_av" in r
        assert "ar_mix_used" in r
        assert "ar_nce" in r
        assert math.isfinite(float(r["ar_nce"]))

    # With warmup_frac=0 and enough steps, p_av should become positive.
    assert max(float(r["p_av"]) for r in train_rows) > 0.0
    # Deterministic seed + many steps should hit at least one AV-mixed step.
    assert sum(int(r["ar_mix_used"]) for r in train_rows) > 0

    # Dead NCE baseline at B=2 is ln(2); final row should be comfortably below.
    final_ar_nce = float(train_rows[-1]["ar_nce"])
    assert final_ar_nce < (math.log(cfg.batch_size) - 0.01)

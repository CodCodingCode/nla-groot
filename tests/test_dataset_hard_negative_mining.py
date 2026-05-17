"""Tests for hard-negative mining in ``LabeledPositionDataset``.

Workstream D1: when ``hard_negative_source`` is set, every item returns
``negative_descriptions: list[str]`` of length ``hard_negatives_per_anchor``,
the collate fn carries the field, and the candidate pool respects the
declared mining policy (same_episode-different-step or
same_position_type-different-episode).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nla.extraction import ActivationShardWriter, RunManifest
from nla.training.dataset import (
    LabeledPositionDataset,
    collate_labeled_positions,
)


TINY_DIM = 16


def _write_dump(out_root: Path, episode_step_pairs: list[tuple[int, int]]) -> list[str]:
    """Write a dump with one row per (episode_index, step_index) pair.

    Returns the list of generated ``example_id`` strings (in the same order)
    so callers can build labels.jsonl that targets each row.
    """
    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="fake",
        hidden_size=TINY_DIM, activation_dtype="float32", embodiment_tag="FAKE",
    )
    writer = ActivationShardWriter(out_root, manifest, max_examples_per_shard=64)
    ids: list[str] = []
    for ep, st in episode_step_pairs:
        ex_id = f"ep{ep}_step{st:04d}"
        ids.append(ex_id)
        writer.write(
            example_id=ex_id,
            features=torch.randn(8, TINY_DIM),
            attention_mask=torch.ones(8, dtype=torch.bool),
            image_mask=torch.zeros(8, dtype=torch.bool),
            input_ids=torch.arange(8),
            episode_index=ep,
            step_index=st,
            task_text="t",
            embodiment_tag="FAKE",
        )
    writer.close()
    return ids


def _write_labels(
    out_path: Path,
    rows: list[tuple[str, int, str, str]],
) -> None:
    """``rows`` is a list of ``(source_example_id, position_index, position_type, description)``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sid, pos, ptype, desc in rows:
            row = {
                "example_id": f"{sid}@p{pos:03d}_{ptype}",
                "description": desc,
                "model": "fake",
                "elapsed_ms": 1.0,
                "usage": {"total_tokens": 1},
                "error": None,
                "kind": "position",
                "meta": {
                    "source_example_id": sid,
                    "position_index": pos,
                    "position_type": ptype,
                    "seq_len": 8,
                    "episode_index": 0,
                    "step_index": 0,
                    "image_patch_meta": None,
                    "instruction": "t",
                },
            }
            f.write(json.dumps(row) + "\n")


def test_same_episode_negatives(tmp_path: Path):
    """2 episodes × 3 steps. Negatives must be same episode, different step."""
    pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    K = 2  # 2 admissible candidates per anchor in same-episode
    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="same_episode",
        hard_negatives_per_anchor=K,
        allow_episode_split_row_fallback=True,
    )
    assert len(ds) == 6

    # Build a description -> (episode, step) map so we can check membership.
    desc_to_meta = {f"desc {sid}": pairs[i] for i, sid in enumerate(ids)}

    for i in range(len(ds)):
        sample = ds[i]
        anchor_meta = desc_to_meta[sample.description]
        anchor_ep, anchor_st = anchor_meta
        assert sample.negative_descriptions is not None
        assert len(sample.negative_descriptions) == K
        for neg_desc in sample.negative_descriptions:
            assert neg_desc in desc_to_meta
            neg_ep, neg_st = desc_to_meta[neg_desc]
            assert neg_ep == anchor_ep, (
                f"same_episode violated: anchor ep={anchor_ep} got neg ep={neg_ep}"
            )
            assert neg_st != anchor_st, (
                f"anchor own step leaked into negatives "
                f"(anchor step={anchor_st}, neg step={neg_st})"
            )


def test_same_position_type_negatives(tmp_path: Path):
    """Mix two position types across two episodes; negatives share ptype but not episode."""
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    # Alternate position_type across episodes so each episode has a mix.
    # Episode 0: anchor, last_text. Ep 1: anchor, last_text. Ep 2: anchor, last_text.
    ptypes = ["anchor", "last_text", "anchor", "last_text", "anchor", "last_text"]
    rows = [
        (ids[i], i + 1, ptypes[i], f"desc {ids[i]}")
        for i in range(len(ids))
    ]
    _write_labels(tmp_path / "labels.jsonl", rows)

    K = 2
    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="same_position_type",
        hard_negatives_per_anchor=K,
        allow_episode_split_row_fallback=True,
    )

    # description -> (ep, ptype)
    desc_to_meta = {
        f"desc {ids[i]}": (pairs[i][0], ptypes[i]) for i in range(len(ids))
    }

    for i in range(len(ds)):
        sample = ds[i]
        anchor_ep, anchor_pt = desc_to_meta[sample.description]
        assert sample.negative_descriptions is not None
        assert len(sample.negative_descriptions) == K
        for neg_desc in sample.negative_descriptions:
            neg_ep, neg_pt = desc_to_meta[neg_desc]
            assert neg_pt == anchor_pt, (
                f"same_position_type violated: anchor pt={anchor_pt} got "
                f"neg pt={neg_pt}"
            )
            assert neg_ep != anchor_ep, (
                f"same_episode leaked into same_position_type mining: "
                f"anchor ep={anchor_ep}, neg ep={neg_ep}"
            )


def test_collate_carries_negatives(tmp_path: Path):
    """Collated batch must expose ``negative_descriptions`` of shape [B][K_neg]."""
    pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    K = 2
    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="same_episode",
        hard_negatives_per_anchor=K,
        allow_episode_split_row_fallback=True,
    )

    batch = collate_labeled_positions([ds[0], ds[1], ds[2]])
    assert "negative_descriptions" in batch
    negs = batch["negative_descriptions"]
    assert isinstance(negs, list) and len(negs) == 3
    for row_negs in negs:
        assert isinstance(row_negs, list)
        assert len(row_negs) == K
        for s in row_negs:
            assert isinstance(s, str)


def test_no_hard_negatives_no_field(tmp_path: Path):
    """``hard_negative_source='none'`` must NOT add the collate field."""
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        allow_episode_split_row_fallback=True,
    )
    sample = ds[0]
    assert sample.negative_descriptions is None
    batch = collate_labeled_positions([ds[0], ds[1]])
    assert "negative_descriptions" not in batch


def test_hard_negatives_per_anchor_zero(tmp_path: Path):
    """K_neg=0 short-circuits to empty lists without crashing."""
    pairs = [(0, 0), (0, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="same_episode",
        hard_negatives_per_anchor=0,
        allow_episode_split_row_fallback=True,
    )
    sample = ds[0]
    assert sample.negative_descriptions == []


def test_unknown_hard_negative_source_raises(tmp_path: Path):
    """Unknown source string fails fast at dataset init."""
    pairs = [(0, 0), (0, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    with pytest.raises(ValueError, match="hard_negative_source"):
        LabeledPositionDataset(
            tmp_path / "act", tmp_path / "labels.jsonl",
            seed=0, held_out_fraction=0.0,
            hard_negative_source="bogus",
            allow_episode_split_row_fallback=True,
        )

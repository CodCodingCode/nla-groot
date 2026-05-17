"""Tests for ``hard_negative_source='topk_cosine'`` in ``LabeledPositionDataset``.

Workstream D2: the dataset reads a precomputed JSONL of activation-cosine
top-K neighbors (produced by ``scripts/training/mine_hard_negatives.py``)
and uses it to populate ``negative_descriptions``. We verify:

1. Correct neg captions are returned for in-index anchors.
2. Anchors absent from the index fall back to "repeat self" (existing
   contract of the same-episode / same-position-type code paths).
3. Negatives that point at out-of-split row IDs are filtered.
4. Missing index path raises early.
5. Malformed JSONL raises a clear error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nla.extraction import ActivationShardWriter, RunManifest
from nla.training.dataset import LabeledPositionDataset


TINY_DIM = 16


def _write_dump(out_root: Path, episode_step_pairs: list[tuple[int, int]]) -> list[str]:
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


def _label_id(sid: str, pos: int, ptype: str) -> str:
    return f"{sid}@p{pos:03d}_{ptype}"


def _write_labels(
    out_path: Path,
    rows: list[tuple[str, int, str, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sid, pos, ptype, desc in rows:
            row = {
                "example_id": _label_id(sid, pos, ptype),
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


def _write_topk_index(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_topk_cosine_returns_neg_captions(tmp_path: Path):
    """When the index lists known neg IDs, the dataset returns their captions."""
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    anchor_ids = [_label_id(ids[i], i + 1, "anchor") for i in range(len(ids))]
    # Hand-craft the index: anchor 0 -> [anchor 2, anchor 3], etc.
    index_rows = [
        {"anchor": anchor_ids[0], "negs": [anchor_ids[2], anchor_ids[3]], "cos": [0.9, 0.8]},
        {"anchor": anchor_ids[1], "negs": [anchor_ids[2], anchor_ids[3]], "cos": [0.9, 0.8]},
        {"anchor": anchor_ids[2], "negs": [anchor_ids[0], anchor_ids[1]], "cos": [0.9, 0.8]},
        {"anchor": anchor_ids[3], "negs": [anchor_ids[0], anchor_ids[1]], "cos": [0.9, 0.8]},
    ]
    _write_topk_index(tmp_path / "hard_negs.jsonl", index_rows)

    K = 2
    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="topk_cosine",
        hard_negative_index_path=tmp_path / "hard_negs.jsonl",
        hard_negatives_per_anchor=K,
        allow_episode_split_row_fallback=True,
    )
    assert len(ds) == 4

    # Anchor 0 must see captions of anchors 2 and 3 only.
    sample0 = ds[_row_for(ds, ids[0])]
    expected_negs = {f"desc {ids[2]}", f"desc {ids[3]}"}
    assert set(sample0.negative_descriptions) == expected_negs


def _row_for(ds: LabeledPositionDataset, source_id: str) -> int:
    for i, entry in enumerate(ds.labels):
        if entry.source_example_id == source_id:
            return i
    raise AssertionError(f"source_id {source_id} not in dataset labels")


def test_topk_cosine_missing_path_raises(tmp_path: Path):
    """topk_cosine without an index path must raise at construction time."""
    pairs = [(0, 0), (0, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    with pytest.raises(ValueError, match="hard_negative_index_path"):
        LabeledPositionDataset(
            tmp_path / "act", tmp_path / "labels.jsonl",
            seed=0, held_out_fraction=0.0,
            hard_negative_source="topk_cosine",
            hard_negative_index_path=None,
            hard_negatives_per_anchor=2,
            allow_episode_split_row_fallback=True,
        )


def test_topk_cosine_unknown_path_raises(tmp_path: Path):
    """A non-existent index path raises FileNotFoundError with a clear message."""
    pairs = [(0, 0), (0, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        LabeledPositionDataset(
            tmp_path / "act", tmp_path / "labels.jsonl",
            seed=0, held_out_fraction=0.0,
            hard_negative_source="topk_cosine",
            hard_negative_index_path=tmp_path / "missing.jsonl",
            hard_negatives_per_anchor=2,
            allow_episode_split_row_fallback=True,
        )


def test_topk_cosine_unknown_neg_ids_filtered(tmp_path: Path):
    """Negs that don't resolve to in-split rows are dropped silently."""
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    anchor_ids = [_label_id(ids[i], i + 1, "anchor") for i in range(len(ids))]
    index_rows = [
        # Anchor 0's first neg is a bogus ID; only the second (real) survives.
        {"anchor": anchor_ids[0], "negs": ["BOGUS_DOES_NOT_EXIST", anchor_ids[2]],
         "cos": [0.99, 0.5]},
        {"anchor": anchor_ids[1], "negs": [anchor_ids[3]], "cos": [0.5]},
        {"anchor": anchor_ids[2], "negs": [anchor_ids[0]], "cos": [0.5]},
        {"anchor": anchor_ids[3], "negs": [anchor_ids[1]], "cos": [0.5]},
    ]
    _write_topk_index(tmp_path / "hard_negs.jsonl", index_rows)

    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="topk_cosine",
        hard_negative_index_path=tmp_path / "hard_negs.jsonl",
        hard_negatives_per_anchor=2,
        allow_episode_split_row_fallback=True,
    )

    sample0 = ds[_row_for(ds, ids[0])]
    # Only one resolved neg available; ``_sample_hard_negatives`` repeats it to
    # hit K_neg=2 (sampling with replacement).
    assert sample0.negative_descriptions == [f"desc {ids[2]}", f"desc {ids[2]}"]


def test_topk_cosine_anchor_not_in_index_falls_back_to_self(tmp_path: Path):
    """An anchor missing from the index uses the 'repeat self caption' fallback."""
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    anchor_ids = [_label_id(ids[i], i + 1, "anchor") for i in range(len(ids))]
    index_rows = [
        # Anchor 0 is intentionally absent (stale index simulation).
        {"anchor": anchor_ids[1], "negs": [anchor_ids[2], anchor_ids[3]], "cos": [0.9, 0.8]},
        {"anchor": anchor_ids[2], "negs": [anchor_ids[0], anchor_ids[1]], "cos": [0.9, 0.8]},
        {"anchor": anchor_ids[3], "negs": [anchor_ids[0], anchor_ids[1]], "cos": [0.9, 0.8]},
    ]
    _write_topk_index(tmp_path / "hard_negs.jsonl", index_rows)

    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="topk_cosine",
        hard_negative_index_path=tmp_path / "hard_negs.jsonl",
        hard_negatives_per_anchor=2,
        allow_episode_split_row_fallback=True,
    )

    sample0 = ds[_row_for(ds, ids[0])]
    # Anchor 0 isn't in the index -> empty candidate pool -> repeat-self fallback.
    assert sample0.negative_descriptions == [f"desc {ids[0]}", f"desc {ids[0]}"]


def test_topk_cosine_malformed_jsonl_raises(tmp_path: Path):
    """A line that isn't valid JSON in the index file must raise loudly."""
    pairs = [(0, 0), (0, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    idx_path = tmp_path / "hard_negs.jsonl"
    idx_path.write_text("not valid json\n")

    with pytest.raises(ValueError, match="Malformed JSONL"):
        LabeledPositionDataset(
            tmp_path / "act", tmp_path / "labels.jsonl",
            seed=0, held_out_fraction=0.0,
            hard_negative_source="topk_cosine",
            hard_negative_index_path=idx_path,
            hard_negatives_per_anchor=2,
            allow_episode_split_row_fallback=True,
        )


def test_topk_cosine_drops_self_collision(tmp_path: Path):
    """If a neg ID accidentally points at the anchor itself, it is dropped."""
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ids = _write_dump(tmp_path / "act", pairs)
    rows = [(sid, idx + 1, "anchor", f"desc {sid}") for idx, sid in enumerate(ids)]
    _write_labels(tmp_path / "labels.jsonl", rows)

    anchor_ids = [_label_id(ids[i], i + 1, "anchor") for i in range(len(ids))]
    # Anchor 0's first neg is anchor 0 itself; should be dropped, only anchor 2 survives.
    index_rows = [
        {"anchor": anchor_ids[0], "negs": [anchor_ids[0], anchor_ids[2]], "cos": [1.0, 0.5]},
        {"anchor": anchor_ids[1], "negs": [anchor_ids[3]], "cos": [0.5]},
        {"anchor": anchor_ids[2], "negs": [anchor_ids[0]], "cos": [0.5]},
        {"anchor": anchor_ids[3], "negs": [anchor_ids[1]], "cos": [0.5]},
    ]
    _write_topk_index(tmp_path / "hard_negs.jsonl", index_rows)

    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="topk_cosine",
        hard_negative_index_path=tmp_path / "hard_negs.jsonl",
        hard_negatives_per_anchor=1,
        allow_episode_split_row_fallback=True,
    )

    sample0 = ds[_row_for(ds, ids[0])]
    assert sample0.negative_descriptions == [f"desc {ids[2]}"]

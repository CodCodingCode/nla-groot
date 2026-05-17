"""End-to-end test for ``scripts/training/mine_hard_negatives.py``.

We construct a tiny activation corpus where the top-1 nearest neighbor in
cosine space is deterministic by design (anchor and target share a
hand-crafted near-identical activation slice), run the miner, and check
that the JSONL output points the anchor at the correct neighbor.

Also covers:

* ``--exclude-same-episode`` actually excludes same-episode rows.
* Empty kept set returns a non-zero exit code.
* Output schema matches the dataset loader's expectations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from nla.extraction import ActivationShardWriter, RunManifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MINE_PATH = REPO_ROOT / "scripts" / "training" / "mine_hard_negatives.py"
TINY_DIM = 16


def _load_mining_module():
    """Importlib-load the script so we can call ``main()`` in-process.

    ``dataclass`` introspection requires the module be present in
    ``sys.modules`` before ``exec_module`` runs (otherwise the
    ``_KeptAnchor`` dataclass owner lookup raises ``AttributeError``).
    """
    name = "nla_mine_hard_negatives_test_alias"
    spec = importlib.util.spec_from_file_location(name, MINE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_dump(out_root: Path, episode_step_features: list[tuple[int, int, torch.Tensor]]):
    """Each entry is ``(episode_index, step_index, [T, D] features)``."""
    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="fake",
        hidden_size=TINY_DIM, activation_dtype="float32", embodiment_tag="FAKE",
    )
    writer = ActivationShardWriter(out_root, manifest, max_examples_per_shard=64)
    ids: list[str] = []
    for ep, st, feats in episode_step_features:
        ex_id = f"ep{ep}_step{st:04d}"
        ids.append(ex_id)
        T = feats.shape[0]
        writer.write(
            example_id=ex_id,
            features=feats,
            attention_mask=torch.ones(T, dtype=torch.bool),
            image_mask=torch.zeros(T, dtype=torch.bool),
            input_ids=torch.arange(T),
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


def _read_index(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["anchor"]] = obj
    return out


def test_topk_picks_intended_neighbor(tmp_path: Path):
    """4 episodes; activations crafted so ep0 and ep2 share a near-identical
    slice at the labeled position. With same-episode masking, ep0's top-1
    must be ep2 (the planted neighbor)."""
    torch.manual_seed(0)
    T = 8

    # Distinct random base for each episode, then plant a near-collision at
    # position 3 between episodes 0 and 2.
    feats: list[tuple[int, int, torch.Tensor]] = []
    bases = [torch.randn(T, TINY_DIM) for _ in range(4)]
    # Position 3 of episodes 0 and 2 share the SAME vector (near-cos=1).
    shared_vec = torch.randn(TINY_DIM)
    bases[0][3] = shared_vec
    bases[2][3] = shared_vec + 1e-3 * torch.randn(TINY_DIM)

    for ep in range(4):
        feats.append((ep, 0, bases[ep]))

    ids = _write_dump(tmp_path / "act", feats)

    # Single label per episode, at position 3 (the "planted" slot).
    rows = [(sid, 3, "anchor", f"desc {sid}") for sid in ids]
    _write_labels(tmp_path / "labels.jsonl", rows)

    out_path = tmp_path / "hard_negs.jsonl"
    mod = _load_mining_module()
    rc = mod.main([
        "--activations-root", str(tmp_path / "act"),
        "--labels-jsonl", str(tmp_path / "labels.jsonl"),
        "--out", str(out_path),
        "--top-k", "2",
        "--chunk-size", "8",
        "--device", "cpu",
    ])
    assert rc == 0
    assert out_path.exists()

    index = _read_index(out_path)
    assert len(index) == 4

    # Episode 0's top-1 neighbor (excluding same-episode) should be ep2 (the
    # planted near-collision). With only 4 episodes, the top-1 is by far the
    # strongest signal.
    anchor_0 = _label_id(ids[0], 3, "anchor")
    target_2 = _label_id(ids[2], 3, "anchor")
    row_0 = index[anchor_0]
    assert row_0["negs"][0] == target_2, (
        f"Expected ep0's top-1 neg to be {target_2}, got {row_0['negs'][0]}. "
        f"Full negs: {row_0['negs']}"
    )
    assert row_0["cos"][0] > 0.9, (
        f"Planted near-collision should have cos > 0.9; got {row_0['cos'][0]}"
    )
    # ep0 must not list itself.
    assert anchor_0 not in row_0["negs"]


def test_exclude_same_episode_masks_within_episode(tmp_path: Path):
    """When same-episode is masked, no neg in any anchor's list shares the
    anchor's episode_index."""
    torch.manual_seed(0)
    feats = []
    for ep in range(3):
        for st in range(3):
            feats.append((ep, st, torch.randn(8, TINY_DIM)))
    ids = _write_dump(tmp_path / "act", feats)
    rows = [(sid, 0, "anchor", f"desc {sid}") for sid in ids]
    _write_labels(tmp_path / "labels.jsonl", rows)

    out_path = tmp_path / "hard_negs.jsonl"
    mod = _load_mining_module()
    rc = mod.main([
        "--activations-root", str(tmp_path / "act"),
        "--labels-jsonl", str(tmp_path / "labels.jsonl"),
        "--out", str(out_path),
        "--top-k", "3",
        "--device", "cpu",
    ])
    assert rc == 0

    # Build a label_id -> episode_index map by walking labels.jsonl + activations.
    label_to_ep: dict[str, int] = {}
    for (ep, _st, _feat), sid in zip(feats, ids):
        label_to_ep[_label_id(sid, 0, "anchor")] = ep

    index = _read_index(out_path)
    for anchor_id, row in index.items():
        anchor_ep = label_to_ep[anchor_id]
        for neg_id in row["negs"]:
            assert label_to_ep[neg_id] != anchor_ep, (
                f"--exclude-same-episode violated: anchor ep={anchor_ep} got "
                f"neg from same episode (neg_id={neg_id})"
            )


def test_no_exclude_same_episode_allows_within(tmp_path: Path):
    """``--no-exclude-same-episode`` permits same-episode neighbors."""
    torch.manual_seed(0)
    # Two episodes, three steps each. Within ep 0, plant a near-collision
    # between steps 0 and 2 (so step 0's top-1 is step 2 in the same episode).
    feats = []
    base = torch.randn(8, TINY_DIM)
    feats.append((0, 0, base.clone()))
    feats.append((0, 1, torch.randn(8, TINY_DIM)))
    feats.append((0, 2, base + 1e-3 * torch.randn(8, TINY_DIM)))
    for st in range(3):
        feats.append((1, st, torch.randn(8, TINY_DIM)))

    ids = _write_dump(tmp_path / "act", feats)
    rows = [(sid, 0, "anchor", f"desc {sid}") for sid in ids]
    _write_labels(tmp_path / "labels.jsonl", rows)

    out_path = tmp_path / "hard_negs.jsonl"
    mod = _load_mining_module()
    rc = mod.main([
        "--activations-root", str(tmp_path / "act"),
        "--labels-jsonl", str(tmp_path / "labels.jsonl"),
        "--out", str(out_path),
        "--top-k", "1",
        "--no-exclude-same-episode",
        "--device", "cpu",
    ])
    assert rc == 0

    index = _read_index(out_path)
    anchor_00 = _label_id(ids[0], 0, "anchor")
    target_02 = _label_id(ids[2], 0, "anchor")
    assert index[anchor_00]["negs"][0] == target_02, (
        "Without same-episode masking, ep0/step0's top-1 should be the "
        f"planted near-collision in ep0/step2. Got {index[anchor_00]['negs'][0]}."
    )


def test_empty_kept_set_returns_nonzero(tmp_path: Path):
    """No labels in scope → non-zero exit (caller should notice)."""
    feats = [(0, 0, torch.randn(8, TINY_DIM))]
    _write_dump(tmp_path / "act", feats)
    (tmp_path / "labels.jsonl").write_text("")

    out_path = tmp_path / "hard_negs.jsonl"
    mod = _load_mining_module()
    rc = mod.main([
        "--activations-root", str(tmp_path / "act"),
        "--labels-jsonl", str(tmp_path / "labels.jsonl"),
        "--out", str(out_path),
        "--top-k", "2",
        "--device", "cpu",
    ])
    assert rc != 0


def test_output_schema_consumable_by_dataset(tmp_path: Path):
    """Output of the miner must drop-in to LabeledPositionDataset(topk_cosine)."""
    from nla.training.dataset import LabeledPositionDataset

    torch.manual_seed(0)
    feats = [(ep, 0, torch.randn(8, TINY_DIM)) for ep in range(4)]
    ids = _write_dump(tmp_path / "act", feats)
    rows = [(sid, 0, "anchor", f"desc {sid}") for sid in ids]
    _write_labels(tmp_path / "labels.jsonl", rows)

    out_path = tmp_path / "hard_negs.jsonl"
    mod = _load_mining_module()
    rc = mod.main([
        "--activations-root", str(tmp_path / "act"),
        "--labels-jsonl", str(tmp_path / "labels.jsonl"),
        "--out", str(out_path),
        "--top-k", "2",
        "--device", "cpu",
    ])
    assert rc == 0

    ds = LabeledPositionDataset(
        tmp_path / "act", tmp_path / "labels.jsonl",
        seed=0, held_out_fraction=0.0,
        hard_negative_source="topk_cosine",
        hard_negative_index_path=out_path,
        hard_negatives_per_anchor=2,
        allow_episode_split_row_fallback=True,
    )
    sample = ds[0]
    assert sample.negative_descriptions is not None
    assert len(sample.negative_descriptions) == 2
    for s in sample.negative_descriptions:
        # Negs must be one of the other episodes' captions (or anchor own if
        # nothing else resolved).
        assert s.startswith("desc ep")

"""Datasets for warm-start SFT and downstream training.

Two flavors are exposed:

1. ``LabeledPositionDataset`` — pairs each labeled position (from a
   Phase 2 ``labels.jsonl``) with the matching activation vector pulled
   from a Phase 1 extraction dump.  This is the dataset for warm-start
   SFT: every item is a single ``(activation, position_type, description)``
   triple.

2. ``SampledPositionDataset`` — for activations we have *no* labels for
   (e.g. when running GRPO RL only with the FVE reward).  Each item is
   a randomly-drawn position from the extraction dump (no description).

Both join cleanly with ``ActivationShardReader`` (Phase 1) and don't try
to wrap variable-length sequence handling — they always materialize a
single ``[hidden]`` vector per item.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from nla.extraction.storage import ActivationShardReader, ExampleRecord
from nla.training.sampling import TokenPositionSampler

logger = logging.getLogger(__name__)


SplitBy = Literal["episode", "row"]


def _split_episode_aware(
    indices: list[int],
    records: list[ExampleRecord],
    *,
    seed: int,
    held_out_fraction: float,
    held_out: bool,
    split_by: SplitBy = "episode",
    label_for_logs: str = "indices",
    allow_row_fallback: bool = True,
) -> list[int]:
    """Return a train- or val-side subset of ``indices`` after a deterministic split.

    When ``split_by == "episode"`` we group ``indices`` by their record's
    ``episode_index`` and hold out *whole episodes*, which is what's needed
    to measure generalization (rather than within-episode temporal leakage).

    Falls back to ``"row"`` with a warning when:
      - the chosen mode is "row", or
      - episode_index is None for every record, or
      - there are fewer than 2 distinct episodes (no valid episode-level split).

    Set ``allow_row_fallback=False`` to make those "cannot do episode split"
    cases raise ``RuntimeError`` instead of silently degrading to row split
    (recommended for paper / generalization runs).
    """
    if held_out_fraction <= 0.0:
        return indices if not held_out else []

    rng = random.Random(seed)

    if split_by == "episode":
        # Map episode_index -> list of indices.  Records with episode_index is None
        # are bucketed under a sentinel; if *every* record is sentinel-bucketed we
        # fall back to row split below.
        SENTINEL = "__no_episode__"
        by_ep: dict[object, list[int]] = {}
        for i in indices:
            ep = records[i].episode_index
            key = SENTINEL if ep is None else int(ep)
            by_ep.setdefault(key, []).append(i)
        n_ep = len(by_ep)
        sentinel_only = list(by_ep.keys()) == [SENTINEL]
        if n_ep < 2 or sentinel_only:
            msg = (
                f"[{label_for_logs}] episode split requested but only {n_ep} "
                f"distinct episode_index values present (sentinel_only={sentinel_only}). "
                "Episode-stratified holdout is required for memorization-vs-generalization "
                "analysis; check that the extraction wrote episode_index."
            )
            if not allow_row_fallback:
                raise RuntimeError(
                    msg
                    + " Refusing to silently fall back to a row split. "
                    "Fix the extraction metadata, pass split_by='row' explicitly, "
                    "or set allow_episode_split_row_fallback=True."
                )
            logger.warning("%s Falling back to row-level split.", msg)
        else:
            ep_keys = sorted(by_ep.keys(), key=lambda k: (k == SENTINEL, str(k)))
            shuffled = list(ep_keys)
            rng.shuffle(shuffled)
            n_held_ep = max(1, int(round(n_ep * held_out_fraction)))
            held_keys = set(shuffled[:n_held_ep])
            sel = [i for i in indices if (
                SENTINEL if records[i].episode_index is None
                else int(records[i].episode_index)
            ) in held_keys] if held_out else [i for i in indices if (
                SENTINEL if records[i].episode_index is None
                else int(records[i].episode_index)
            ) not in held_keys]
            logger.info(
                "[%s] episode-stratified split: %d episodes total, %d held out "
                "(%d rows train / %d rows val)",
                label_for_logs, n_ep, n_held_ep,
                len(indices) - len(sel) if held_out else len(sel),
                len(sel) if held_out else len(indices) - len(sel),
            )
            return sel

    # Row-level fallback (or explicit choice).
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_held = max(1, int(round(len(shuffled) * held_out_fraction)))
    return shuffled[:n_held] if held_out else shuffled[n_held:]


@dataclass
class LabelEntry:
    source_example_id: str
    position_index: int
    position_type: str
    description: str
    quality_weight: float
    raw: dict


def _extract_quality_weight(obj: dict) -> float:
    """Read an optional per-label quality scalar from a labels.jsonl row.

    Looks for, in order:
      1. obj["quality_weight"]  (a float in [0, 1] written by an upstream grader)
      2. obj["quality_axes"]    (a dict with bool/float values per axis; mean used)
    Returns 1.0 if neither is present (i.e. backward compatible: labels without
    quality info are treated as full-weight).
    """
    qw = obj.get("quality_weight")
    if isinstance(qw, (int, float)):
        return float(max(0.0, min(1.0, qw)))
    axes = obj.get("quality_axes")
    if isinstance(axes, dict) and axes:
        try:
            vals = [float(v) for v in axes.values()]
            if vals:
                return float(max(0.0, min(1.0, sum(vals) / len(vals))))
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _count_bullet_lines(text: str) -> int:
    """Count markdown bullet lines (lines whose stripped form starts with '-')."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- ") or s == "-":
            n += 1
    return n


def load_labels_jsonl(path, *, min_bullet_lines: int | None = None):
    """Parse a labels.jsonl produced by the labeling pipeline.

    When ``min_bullet_lines`` is set, rows whose description has fewer than
    that many ``-`` bullet lines are dropped (counted in the skipped tally).
    """
    entries = []
    n_skipped = 0
    n_skipped_bullets = 0
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                n_skipped += 1
                continue
            if obj.get("error"):
                n_skipped += 1
                continue
            desc = (obj.get("description") or "").strip()
            if not desc:
                n_skipped += 1
                continue
            if min_bullet_lines is not None and _count_bullet_lines(desc) < min_bullet_lines:
                n_skipped += 1
                n_skipped_bullets += 1
                continue
            meta = obj.get("meta") or {}
            source_id = meta.get("source_example_id")
            pos_idx = meta.get("position_index")
            pos_type = meta.get("position_type")
            if source_id is None or pos_idx is None or pos_type is None:
                n_skipped += 1
                continue
            entries.append(LabelEntry(
                source_example_id=str(source_id),
                position_index=int(pos_idx),
                position_type=str(pos_type),
                description=desc,
                quality_weight=_extract_quality_weight(obj),
                raw=obj,
            ))
    if min_bullet_lines is not None and n_skipped_bullets:
        logger.info(
            "Dropped %d labels for fewer than %d bullet lines.",
            n_skipped_bullets, min_bullet_lines,
        )
    logger.info("Loaded %d labels from %s (%d skipped)", len(entries), path, n_skipped)
    return entries


@dataclass
class LabeledPositionSample:
    activation: torch.Tensor
    position_type: str
    position_index: int
    seq_len: int
    description: str
    example_id: str
    label_example_id: str
    episode_index: int | None
    quality_weight: float


class LabeledPositionDataset(Dataset):
    """Map-style dataset over labeled (activation, description) pairs.

    Train/val split is **episode-stratified by default** so held-out timesteps
    do not leak from training episodes.  Set ``split_by="row"`` to recover the
    legacy random-row holdout (only useful when the extraction has no
    ``episode_index`` field, or for ablations).
    """

    def __init__(
        self,
        activations_root,
        labels_jsonl,
        *,
        seed=0,
        held_out_fraction=0.0,
        held_out=False,
        max_items=None,
        split_by: SplitBy = "episode",
        allow_episode_split_row_fallback: bool = True,
        min_bullet_lines: int | None = None,
    ):
        self.reader = ActivationShardReader(activations_root)
        self._index_by_id = {rec.example_id: i for i, rec in enumerate(self.reader.records)}

        all_labels = load_labels_jsonl(labels_jsonl, min_bullet_lines=min_bullet_lines)
        valid = []
        n_missing = 0
        for entry in all_labels:
            if entry.source_example_id not in self._index_by_id:
                n_missing += 1
                continue
            valid.append(entry)
        if n_missing:
            logger.warning("Discarded %d labels with no matching activation.", n_missing)

        valid.sort(key=lambda e: e.raw.get("example_id", e.source_example_id))

        if held_out_fraction > 0.0:
            label_positions = list(range(len(valid)))
            # Map a label position -> the underlying activation's ExampleRecord
            # by going through self._index_by_id; we build a parallel "records"
            # list aligned with label_positions so _split_episode_aware can
            # group by episode.
            label_records = [
                self.reader.records[self._index_by_id[valid[lp].source_example_id]]
                for lp in label_positions
            ]
            kept = _split_episode_aware(
                label_positions,
                label_records,
                seed=seed,
                held_out_fraction=held_out_fraction,
                held_out=held_out,
                split_by=split_by,
                label_for_logs=f"LabeledPositionDataset({'val' if held_out else 'train'})",
                allow_row_fallback=allow_episode_split_row_fallback,
            )
            valid = [valid[i] for i in kept]
        else:
            rng = random.Random(seed)
            rng.shuffle(valid)

        if max_items is not None:
            valid = valid[: int(max_items)]
        self.labels = valid

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        entry = self.labels[i]
        global_idx = self._index_by_id[entry.source_example_id]
        rec = self.reader.records[global_idx]
        item = self.reader[global_idx]
        features = item["features"]
        pos = entry.position_index
        if pos >= features.shape[0]:
            raise IndexError(
                f"Label position {pos} >= seq_len {features.shape[0]} for "
                f"example {entry.source_example_id}"
            )
        vec = features[pos].contiguous().to(torch.float32)
        return LabeledPositionSample(
            activation=vec,
            position_type=entry.position_type,
            position_index=pos,
            seq_len=int(rec.seq_len),
            description=entry.description,
            example_id=entry.source_example_id,
            label_example_id=entry.raw.get("example_id") or entry.source_example_id,
            episode_index=rec.episode_index,
            quality_weight=float(entry.quality_weight),
        )


def collate_labeled_positions(batch):
    return {
        "activations": torch.stack([b.activation for b in batch], dim=0),
        "position_type": [b.position_type for b in batch],
        "position_index": torch.tensor([b.position_index for b in batch], dtype=torch.long),
        "seq_len": torch.tensor([b.seq_len for b in batch], dtype=torch.long),
        "description": [b.description for b in batch],
        "example_id": [b.example_id for b in batch],
        "label_example_id": [b.label_example_id for b in batch],
        "episode_index": [b.episode_index for b in batch],
        "quality_weight": torch.tensor(
            [b.quality_weight for b in batch], dtype=torch.float32,
        ),
    }


@dataclass
class SampledPositionSample:
    activation: torch.Tensor
    position_type: str
    position_index: int
    seq_len: int
    example_id: str
    episode_index: int | None


class SampledPositionDataset(Dataset):
    """One sampled position per example, following POSITION_MIX (for RL etc.).

    Episode-stratified holdout by default (see ``LabeledPositionDataset``).
    """

    def __init__(
        self,
        activations_root,
        *,
        seed=0,
        position_mix=None,
        held_out_fraction=0.0,
        held_out=False,
        split_by: SplitBy = "episode",
        allow_episode_split_row_fallback: bool = True,
    ):
        self.reader = ActivationShardReader(activations_root)
        self.sampler = TokenPositionSampler(position_mix=position_mix, seed=seed)
        idx = list(range(len(self.reader)))
        if held_out_fraction > 0.0:
            idx = _split_episode_aware(
                idx,
                self.reader.records,
                seed=seed,
                held_out_fraction=held_out_fraction,
                held_out=held_out,
                split_by=split_by,
                label_for_logs=f"SampledPositionDataset({'val' if held_out else 'train'})",
                allow_row_fallback=allow_episode_split_row_fallback,
            )
        else:
            rng = random.Random(seed)
            rng.shuffle(idx)
        self._indices = idx

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        global_idx = self._indices[i]
        rec = self.reader.records[global_idx]
        item = self.reader[global_idx]
        ptype, pos = self.sampler.sample(item["attention_mask"], item["image_mask"])
        vec = item["features"][pos].contiguous().to(torch.float32)
        return SampledPositionSample(
            activation=vec, position_type=ptype, position_index=pos,
            seq_len=int(rec.seq_len), example_id=rec.example_id,
            episode_index=rec.episode_index,
        )


def collate_sampled_positions(batch):
    return {
        "activations": torch.stack([b.activation for b in batch], dim=0),
        "position_type": [b.position_type for b in batch],
        "position_index": torch.tensor([b.position_index for b in batch], dtype=torch.long),
        "seq_len": torch.tensor([b.seq_len for b in batch], dtype=torch.long),
        "example_id": [b.example_id for b in batch],
        "episode_index": [b.episode_index for b in batch],
    }

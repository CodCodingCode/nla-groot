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
    # AV-side activation. ``[H]`` for single-slot rows (last_text, anchor,
    # image_patch under pinned/mean_pool/strided/center pooling); ``[K, H]``
    # for image_patch rows under ``strided_image_multi`` (V5 K-slot path).
    activation: torch.Tensor
    position_type: str
    position_index: int
    seq_len: int
    description: str
    example_id: str
    label_example_id: str
    episode_index: int | None
    quality_weight: float
    # AR-side activation. Always a single ``[H]`` vector regardless of pooling
    # (AR continues to regress one ``h`` per row even when AV sees K slots).
    # For single-slot rows this equals ``activation``; for K-slot rows it is
    # the mean of the K patch vectors so existing AR forward and offline
    # hard-neg mining stay valid.
    activation_ar: torch.Tensor | None = None
    # V5 context fields surfaced for the AV prompt. ``step_index`` is the
    # episode timestep from the activation index (``rec.step_index``);
    # ``instruction`` is the natural-language task pulled from the label row's
    # ``meta.instruction``. Both are ``None`` when the underlying artifact
    # doesn't carry the field, in which case the AV prompt renders sentinel
    # placeholders ("unknown" / "(not provided)").
    step_index: int | None = None
    instruction: str | None = None
    # Optional list of hard-negative captions sampled for this anchor by the
    # dataset's hard-negative miner. ``None`` (the default) means hard-neg
    # mining is disabled and downstream code should keep its legacy in-batch
    # InfoNCE behavior.
    negative_descriptions: list[str] | None = None


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
        strict_position_check: bool = True,
        strict_position_check_max_examples: int = 10,
        hard_negative_source: Literal[
            "none", "same_episode", "same_position_type", "topk_cosine"
        ] = "none",
        hard_negatives_per_anchor: int = 4,
        hard_negative_index_path: str | Path | None = None,
        image_patch_pooling: Literal[
            "pinned", "mean_pool_image", "strided_image",
            "strided_image_multi", "center_image",
        ] = "pinned",
        image_patch_pooling_strided_k: int = 4,
        exclude_position_types: tuple[str, ...] | None = None,
        include_position_types: tuple[str, ...] | None = None,
        ar_target_spatial: bool = False,
    ):
        # ``image_patch_pooling`` controls how the dataset materializes the
        # activation for rows whose ``position_type == "image_patch"``. The
        # default ``"pinned"`` preserves V3 behaviour exactly: return
        # ``features[entry.position_index]`` (the single random patch the
        # labeling pass committed to). The single-vector pooled strategies
        # (added for V4 per ``docs/sft_plan/08_positive_steering_followon.md``
        # step 1) ignore ``position_index`` and pool over every valid
        # image-patch token via ``nla.extraction.position_strategies.apply``.
        # V5 adds ``"strided_image_multi"``: AV gets the full ``[K, H]`` strided
        # patch grid (one slot per patch); AR keeps a single ``[H]`` target by
        # mean-pooling the K patches so the existing AR forward and offline
        # hard-neg mining stay valid. Non-image ptypes (``last_text``,
        # ``anchor``) are untouched regardless.
        self.image_patch_pooling = str(image_patch_pooling)
        self.image_patch_pooling_strided_k = int(image_patch_pooling_strided_k)
        # v7 plan: when the AR head is spatial (head_type='spatial'), the
        # dataset must emit per-position targets so the per-position MSE has
        # a meaningful signal. With strided_image_multi pooling on image_patch
        # rows we emit the full [K, H] grid. Non-image_patch rows still emit
        # a single [H] vector; the collator tiles them to [K, H] so all rows
        # share a shape.
        self.ar_target_spatial = bool(ar_target_spatial)
        if self.image_patch_pooling not in (
            "pinned", "mean_pool_image", "strided_image",
            "strided_image_multi", "center_image",
        ):
            raise ValueError(
                f"image_patch_pooling must be one of pinned/mean_pool_image/"
                f"strided_image/strided_image_multi/center_image, "
                f"got {self.image_patch_pooling!r}"
            )
        # Optional set of position_type values to drop entirely before split.
        # V5 ablations use this to enforce ``no anchor`` even when callers
        # accidentally pass the full combined labels file; the canonical
        # workflow is still to point ``labels_jsonl`` at the pre-filtered
        # ``labels_no_anchor.jsonl``.
        self.exclude_position_types: frozenset[str] = (
            frozenset(exclude_position_types) if exclude_position_types else frozenset()
        )
        # Stage-2 plan: positive include filter for image_patch-only runs
        # (cleanest ablation for "is the codec capable on the vision slot if
        # we don't dilute training with last_text/anchor?"). Mutually
        # exclusive with exclude_position_types in semantics — when both are
        # set, include is applied first then exclude prunes further.
        self.include_position_types: frozenset[str] = (
            frozenset(include_position_types) if include_position_types else frozenset()
        )
        self.reader = ActivationShardReader(activations_root)
        self._index_by_id = {rec.example_id: i for i, rec in enumerate(self.reader.records)}

        all_labels = load_labels_jsonl(labels_jsonl, min_bullet_lines=min_bullet_lines)
        valid = []
        n_missing = 0
        n_excluded = 0
        n_not_included = 0
        for entry in all_labels:
            if entry.source_example_id not in self._index_by_id:
                n_missing += 1
                continue
            if self.include_position_types and entry.position_type not in self.include_position_types:
                n_not_included += 1
                continue
            if entry.position_type in self.exclude_position_types:
                n_excluded += 1
                continue
            valid.append(entry)
        if n_missing:
            logger.warning("Discarded %d labels with no matching activation.", n_missing)
        if n_not_included:
            logger.info(
                "Dropped %d labels via include_position_types=%s",
                n_not_included, sorted(self.include_position_types),
            )
        if n_excluded:
            logger.info(
                "Dropped %d labels via exclude_position_types=%s",
                n_excluded, sorted(self.exclude_position_types),
            )

        if strict_position_check and valid:
            # Fail fast on out-of-bounds position_index using the cheap index
            # metadata (no tensor loads). Catches mismatched labels/activations
            # at init time instead of throwing IndexError mid-training.
            bad: list[tuple[str, int, int]] = []
            for entry in valid:
                rec = self.reader.records[self._index_by_id[entry.source_example_id]]
                if entry.position_index >= int(rec.seq_len):
                    bad.append((entry.source_example_id, entry.position_index, int(rec.seq_len)))
            if bad:
                sample = ", ".join(
                    f"{sid}@{pidx} (seq_len={sl})"
                    for sid, pidx, sl in bad[:strict_position_check_max_examples]
                )
                raise ValueError(
                    f"{len(bad)} label rows have position_index >= activation seq_len. "
                    f"First {min(len(bad), strict_position_check_max_examples)}: {sample}"
                )

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

        # Hard-negative mining bookkeeping. We snapshot the per-label-row
        # (episode_index, step_index, position_type) tuples and a precomputed
        # list-of-candidate-row-indices keyed by anchor row. Built once at
        # init time so __getitem__ is O(K_neg) per call.
        self.hard_negative_source = str(hard_negative_source)
        self.hard_negatives_per_anchor = int(hard_negatives_per_anchor)
        self._hard_neg_seed = int(seed)
        self._hard_neg_candidates: list[list[int]] | None = None
        self.hard_negative_index_path = (
            None if hard_negative_index_path is None else Path(hard_negative_index_path)
        )
        if self.hard_negative_source != "none":
            self._build_hard_negative_index()

    def _build_hard_negative_index(self) -> None:
        """Precompute, for each kept label row, the list of label-row indices
        that are admissible as hard negatives.

        ``same_episode``: rows with the same ``episode_index`` and a
        *different* ``step_index``. Excludes the anchor itself by construction.

        ``same_position_type``: rows with the same ``position_type`` but a
        *different* ``episode_index``. Excludes anchor-own-episode entirely
        so the negative caption describes a genuinely different scene.

        ``topk_cosine``: load a precomputed JSONL produced by
        ``scripts/training/mine_hard_negatives.py``. Each row is keyed by
        the anchor's label_example_id and lists the top-K most cosine-
        similar label IDs in the same activation corpus. We re-resolve
        label IDs to in-split row indices, dropping any neg that fell on
        the wrong side of the held-out split or got dropped by
        ``max_items``/``min_bullet_lines``. Anchors with no remaining
        admissible negatives fall back to the anchor's own caption (the
        same fallback as the heuristic modes).
        """
        n = len(self.labels)
        cands: list[list[int]] = [[] for _ in range(n)]

        if self.hard_negative_source == "topk_cosine":
            if self.hard_negative_index_path is None:
                raise ValueError(
                    "hard_negative_source='topk_cosine' requires "
                    "hard_negative_index_path to point at a mining JSONL "
                    "produced by scripts/training/mine_hard_negatives.py."
                )
            cands = self._build_topk_cosine_index()
        elif self.hard_negative_source == "same_episode":
            meta = self._labels_episode_step_ptype()
            by_ep: dict[int | None, list[int]] = {}
            for j, (ep, _st, _pt) in enumerate(meta):
                by_ep.setdefault(ep, []).append(j)
            for i, (ep_i, st_i, _pt_i) in enumerate(meta):
                pool = by_ep.get(ep_i, [])
                cands[i] = [j for j in pool if j != i and meta[j][1] != st_i]
        elif self.hard_negative_source == "same_position_type":
            meta = self._labels_episode_step_ptype()
            by_pt: dict[str, list[int]] = {}
            for j, (_ep, _st, pt) in enumerate(meta):
                by_pt.setdefault(pt, []).append(j)
            for i, (ep_i, _st_i, pt_i) in enumerate(meta):
                pool = by_pt.get(pt_i, [])
                cands[i] = [j for j in pool if meta[j][0] != ep_i]
        else:
            raise ValueError(
                f"Unknown hard_negative_source={self.hard_negative_source!r}; "
                "expected one of {none, same_episode, same_position_type, topk_cosine}."
            )

        n_empty = sum(1 for c in cands if not c)
        if n_empty:
            logger.warning(
                "[hard-neg %s] %d/%d anchors have no admissible negatives; "
                "those rows will fall back to repeating the anchor's own caption.",
                self.hard_negative_source, n_empty, n,
            )
        self._hard_neg_candidates = cands

    def _labels_episode_step_ptype(self) -> list[tuple[int | None, int | None, str]]:
        out: list[tuple[int | None, int | None, str]] = []
        for entry in self.labels:
            rec = self.reader.records[self._index_by_id[entry.source_example_id]]
            ep = None if rec.episode_index is None else int(rec.episode_index)
            st = None if rec.step_index is None else int(rec.step_index)
            out.append((ep, st, entry.position_type))
        return out

    def _build_topk_cosine_index(self) -> list[list[int]]:
        """Read the offline-mined JSONL into per-anchor row-index lists.

        We resolve neg label IDs through ``self._label_id_to_row`` which is
        keyed by the same canonical label_example_id the miner wrote
        (``label.raw["example_id"]`` when present, else the synthetic
        ``<sid>@p<NNN>_<ptype>`` fallback).  Negs that don't resolve (held
        out, filtered by min_bullet_lines, etc.) are silently dropped per
        anchor; anchors that drop to empty fall back to the "repeat self"
        behavior in ``_sample_hard_negatives``.
        """
        path = self.hard_negative_index_path
        assert path is not None
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"hard_negative_index_path={path} does not exist. Run "
                "scripts/training/mine_hard_negatives.py on this activation "
                "corpus first."
            )
        label_id_to_row = self._build_label_id_to_row()

        n = len(self.labels)
        cands: list[list[int]] = [[] for _ in range(n)]
        n_rows_in_index = 0
        n_anchors_matched = 0
        n_negs_total = 0
        n_negs_resolved = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL line in {path}: {exc}"
                    ) from exc
                n_rows_in_index += 1
                anchor_id = row.get("anchor")
                if anchor_id is None:
                    continue
                anchor_row = label_id_to_row.get(anchor_id)
                if anchor_row is None:
                    continue
                neg_ids = row.get("negs") or []
                n_anchors_matched += 1
                n_negs_total += len(neg_ids)
                resolved: list[int] = []
                for nid in neg_ids:
                    j = label_id_to_row.get(nid)
                    if j is None or j == anchor_row:
                        continue
                    resolved.append(j)
                n_negs_resolved += len(resolved)
                cands[anchor_row] = resolved
        coverage = 0.0 if n == 0 else n_anchors_matched / n
        resolve_rate = 0.0 if n_negs_total == 0 else n_negs_resolved / n_negs_total
        logger.info(
            "[hard-neg topk_cosine] index_rows=%d  in-split_anchors_matched=%d/%d "
            "(coverage=%.1f%%)  neg_resolve_rate=%.1f%%  path=%s",
            n_rows_in_index, n_anchors_matched, n,
            100 * coverage, 100 * resolve_rate, path,
        )
        if coverage < 0.5:
            logger.warning(
                "[hard-neg topk_cosine] only %.1f%% of in-split anchors were found "
                "in the mining index. Did you re-mine after a labels or split change? "
                "Stale index degrades to repeat-self fallback for unmatched rows.",
                100 * coverage,
            )
        return cands

    def _build_label_id_to_row(self) -> dict[str, int]:
        """Return a map from canonical label_example_id -> row index in self.labels.

        We register *both* the explicit ``label.raw["example_id"]`` (the
        labeling-pipeline-assigned ID) and the synthetic
        ``<source_example_id>@p<NNN>_<position_type>`` fallback. The miner
        uses whichever is available, so we accept both shapes here.
        """
        out: dict[str, int] = {}
        for i, entry in enumerate(self.labels):
            raw_id = entry.raw.get("example_id")
            synth_id = (
                f"{entry.source_example_id}@p{entry.position_index:03d}_"
                f"{entry.position_type}"
            )
            if raw_id is not None:
                out[str(raw_id)] = i
            out.setdefault(synth_id, i)
        return out

    def _sample_hard_negatives(self, anchor_i: int) -> list[str]:
        """Return ``hard_negatives_per_anchor`` negative captions for anchor.

        Sampling is *with replacement* whenever the candidate pool is smaller
        than ``K_neg`` (so the dataloader always sees a rectangular shape);
        without replacement otherwise. The RNG is per-call seeded with
        ``(self._hard_neg_seed, anchor_i, n_seen)`` mixed into a Python
        ``Random`` so repeated reads of the same index across epochs vary
        but a single epoch is reproducible from the dataset seed.

        The fallback when the pool is empty is to return the anchor's own
        caption ``K_neg`` times. That degrades to a "self-collision" hard
        negative which is still a valid (if mild) negative under the
        contrastive objective.
        """
        K = self.hard_negatives_per_anchor
        if K <= 0:
            return []
        cands = (self._hard_neg_candidates or [])[anchor_i]
        # Mix seed and anchor index into a single int (Python 3.9+ deprecates
        # hash-based seeding for non-int types, which is what bit us here).
        rng = random.Random((self._hard_neg_seed * 0x9E3779B1) ^ int(anchor_i))
        if not cands:
            return [self.labels[anchor_i].description] * K
        if len(cands) >= K:
            picks = rng.sample(cands, K)
        else:
            picks = [rng.choice(cands) for _ in range(K)]
        return [self.labels[j].description for j in picks]

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
        vec_ar: torch.Tensor | None = None
        if (
            self.image_patch_pooling != "pinned"
            and entry.position_type == "image_patch"
        ):
            # V4/V5 pooled-strategy path. Re-uses the same read-time pooling
            # functions the extraction A/B sweep validated. We do NOT
            # apply pooling to ``last_text``/``anchor`` rows: their pinned
            # token positions are the whole point of those ptypes.
            try:
                from nla.extraction.position_strategies import apply as _apply_strategy
            except ImportError as e:
                raise ImportError(
                    "image_patch_pooling requires nla.extraction.position_strategies; "
                    "got ImportError. Check sys.path / package install."
                ) from e
            image_mask = item.get("image_mask")
            attention_mask = item.get("attention_mask")
            if image_mask is None or attention_mask is None:
                raise RuntimeError(
                    f"image_patch_pooling={self.image_patch_pooling!r} requires "
                    f"the activation shard to carry image_mask + attention_mask; "
                    f"shard for {entry.source_example_id} is missing them. "
                    "Re-extract with the current scripts/extraction/run_extract.py."
                )
            try:
                pooled = _apply_strategy(
                    self.image_patch_pooling,
                    features,
                    image_mask,
                    attention_mask,
                    k=self.image_patch_pooling_strided_k,
                ).contiguous().to(torch.float32)
            except ValueError:
                # Strategy raises if there are no image patches (shouldn't
                # happen for a row whose ptype is image_patch, but be
                # defensive); fall back to the pinned position.
                pooled = features[pos].contiguous().to(torch.float32)
            if self.image_patch_pooling == "strided_image_multi" and pooled.dim() == 2:
                # AV sees the full K-patch grid; AR keeps a single mean vector
                # so the existing AR forward and offline hard-neg mining (mined
                # on per-row single ``h``) stay valid.
                vec = pooled                                  # [K, H]
                if self.ar_target_spatial:
                    # v7 spatial AR head: emit the full [K, H] grid as the
                    # AR target so per-position MSE has signal.
                    vec_ar = pooled.contiguous()              # [K, H]
                else:
                    vec_ar = pooled.mean(dim=0).contiguous()  # [H]
            else:
                # Single-vector pooling returned ``[H]`` directly.
                vec = pooled
        else:
            vec = features[pos].contiguous().to(torch.float32)
        negs: list[str] | None = None
        if self.hard_negative_source != "none":
            negs = self._sample_hard_negatives(i)
        meta = entry.raw.get("meta") or {}
        instruction = meta.get("instruction")
        if instruction is not None:
            instruction = str(instruction)
        step_index = None if rec.step_index is None else int(rec.step_index)
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
            activation_ar=vec_ar if vec_ar is not None else vec,
            step_index=step_index,
            instruction=instruction,
            negative_descriptions=negs,
        )


def collate_labeled_positions(batch):
    # ``step_index`` defaults to -1 for rows whose activation index didn't
    # carry the field; AV passes the value through ``int | None`` and
    # converts -1 back to ``None`` so the prompt renders the sentinel.
    #
    # AV vs AR activation tensors
    # ---------------------------
    # Most rows have ``activation`` shape ``[H]`` and ``activation_ar == vec``
    # (single-vector path). Image-patch rows under ``strided_image_multi``
    # pooling have ``activation`` shape ``[K, H]`` (one slot per patch) and a
    # separate ``activation_ar`` of shape ``[H]`` (mean over K). To handle
    # mixed batches we right-pad K with zeros up to ``K_max`` and emit an
    # ``activation_slot_mask`` so AV knows which slots are real.
    #
    # ``activations`` is kept as an alias for ``activations_ar`` so legacy
    # callers (e.g. closed-loop eval helpers, action-consistency kernel) that
    # only need a single ``[H]`` per row keep working without a code change.
    # v7 spatial-AR path: when any row's ``activation_ar`` is 2-D ``[K, H]``,
    # tile non-spatial rows (single ``[H]``) to ``[K, H]`` by repeating along
    # the spatial axis. This means "the spatial AR head should predict the
    # same vector at every position for non-image-patch rows" — appropriate
    # since last_text / anchor have no spatial structure.
    raw_ar_vecs = [
        (b.activation_ar if b.activation_ar is not None else b.activation)
        for b in batch
    ]
    ar_dims = {v.dim() for v in raw_ar_vecs}
    if ar_dims <= {1}:
        ar_vecs = list(raw_ar_vecs)
        activations_ar = torch.stack(ar_vecs, dim=0)               # (B, H)
    else:
        # Mixed batch: at least one row has [K, H]. Tile single-vector rows
        # to [K_ar, H] where K_ar is the spatial-target K (constant across
        # spatial rows).
        spatial_ks = {v.shape[0] for v in raw_ar_vecs if v.dim() == 2}
        if len(spatial_ks) > 1:
            raise ValueError(
                f"activation_ar spatial-K must be uniform across rows; "
                f"got {sorted(spatial_ks)}"
            )
        k_ar = next(iter(spatial_ks))
        tiled: list[torch.Tensor] = []
        for v in raw_ar_vecs:
            if v.dim() == 1:
                tiled.append(v.unsqueeze(0).expand(k_ar, -1).contiguous())
            elif v.dim() == 2:
                if v.shape[0] != k_ar:
                    raise ValueError(
                        f"activation_ar row K={v.shape[0]} != batch K={k_ar}"
                    )
                tiled.append(v)
            else:
                raise ValueError(
                    f"activation_ar must be 1-D or 2-D; got shape {tuple(v.shape)}"
                )
        activations_ar = torch.stack(tiled, dim=0)                  # (B, K, H)

    av_shapes = {tuple(b.activation.shape) for b in batch}
    k_per_row = [b.activation.shape[0] if b.activation.dim() == 2 else 1 for b in batch]
    h_dim = int(activations_ar.shape[-1])
    k_max = max(k_per_row)
    if k_max == 1 and all(b.activation.dim() == 1 for b in batch):
        activations_av = torch.stack(
            [b.activation for b in batch], dim=0,
        )                                                    # (B, H)
        slot_mask = torch.ones((len(batch), 1), dtype=torch.bool)
    else:
        # Mixed batch (or all multi-slot). Right-pad to ``k_max`` with zeros
        # and record real-vs-padding in ``activation_slot_mask``. AV's
        # ``_embed_with_injection`` never reads the trailing padding rows
        # because their slot ids don't appear in the corresponding prompt.
        activations_av = torch.zeros(
            (len(batch), k_max, h_dim), dtype=activations_ar.dtype,
        )
        slot_mask = torch.zeros((len(batch), k_max), dtype=torch.bool)
        for i, b in enumerate(batch):
            v = b.activation
            if v.dim() == 1:
                activations_av[i, 0] = v
                slot_mask[i, 0] = True
            else:
                k_i = v.shape[0]
                activations_av[i, :k_i] = v
                slot_mask[i, :k_i] = True
        if len({tuple(s.shape) for s in [activations_av]}) != 1:
            # Sanity: stack must be rectangular at this point.
            raise RuntimeError(
                f"activations_av collate produced ragged shapes from {av_shapes}"
            )

    # ``activations`` is the single-vector backward-compat view used by
    # action-consistency, closed-loop eval, etc. When ``activations_ar`` is
    # 3-D (spatial AR target), mean-pool over the K axis so legacy callers
    # keep working without changes.
    if activations_ar.dim() == 3:
        activations_single = activations_ar.mean(dim=1)             # (B, H)
    else:
        activations_single = activations_ar                          # (B, H)

    out = {
        "activations": activations_single,
        "activations_ar": activations_ar,
        "activations_av": activations_av,
        "activation_slot_mask": slot_mask,
        "activation_slot_count": torch.tensor(k_per_row, dtype=torch.long),
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
        "step_index": torch.tensor(
            [-1 if b.step_index is None else int(b.step_index) for b in batch],
            dtype=torch.long,
        ),
        "instruction": [b.instruction for b in batch],
    }
    if any(b.negative_descriptions is not None for b in batch):
        # Carry only when *every* row provided negatives so downstream
        # consumers can assume rectangular shape; falling back to skipping
        # the field when partial would silently disable hard-negs for the
        # whole batch on a single-row dataset misconfig.
        if not all(b.negative_descriptions is not None for b in batch):
            raise ValueError(
                "collate_labeled_positions: some batch rows have "
                "negative_descriptions and others don't. Mining must be on "
                "for every row in a batch (configure the dataset uniformly)."
            )
        out["negative_descriptions"] = [list(b.negative_descriptions) for b in batch]
    return out


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
        allowed_example_ids: set[str] | frozenset[str] | None = None,
    ):
        self.reader = ActivationShardReader(activations_root)
        self.sampler = TokenPositionSampler(position_mix=position_mix, seed=seed)
        idx = list(range(len(self.reader)))
        if allowed_example_ids is not None:
            allowed = set(allowed_example_ids)
            n_before = len(idx)
            idx = [
                i for i in idx
                if self.reader.records[i].example_id in allowed
            ]
            logger.info(
                "[%s] CF-eligible filter: kept %d / %d activations (%.1f%%)",
                f"SampledPositionDataset({'val' if held_out else 'train'})",
                len(idx), n_before, 100.0 * len(idx) / max(1, n_before),
            )
            if not idx:
                raise RuntimeError(
                    "CF-eligible filter removed every activation; check "
                    "allowed_example_ids against activations_root example_id "
                    "format."
                )
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

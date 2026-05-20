"""Sampler for (activation, target_intent) pairs used by sim-success GRPO.

The mining script ``scripts/training/mine_grpo_counterfactual_pairs.py``
produces a JSONL of pairs joining label rows in
``data/labels/libero_4suite_combined/labels.jsonl`` to a (possibly
counterfactual) target intent + target LIBERO BDDL env name.

At training time we want one such pair per activation in a GRPO batch. The
existing :class:`SampledPositionDataset` only yields ``(activation,
position_type)`` so this module adds a *parallel index* keyed by both:

* the JSONL row's ``source_example_id`` (the activation-shard id), AND
* the JSONL row's ``example_id`` (the label id, often used by datasets
  that hand back the label-side id instead of the activation-shard id).

Either key resolves to the same candidate list, so callers can mix
datasets that emit ``source_example_id``-style ids
(e.g. ``goal__traj000218_step000020``) with those that emit
``example_id``-style ids without hand-merging the JSONL beforehand. The
sampler can also stitch together multiple pairs JSONLs via the
``additional_paths`` constructor argument; rows are appended into the
same candidate lists, dropping exact-duplicate ``(source_example_id,
target_intent, target_task, target_env_name)`` tuples so a row that
appears in two files isn't double-counted by ``random.choice``.

The sampler is intentionally stateless w.r.t. activation extraction: it
only reads the pairs JSONL and lets the GRPO trainer keep its own
activation dataset. This avoids duplicating multi-GB tensor loading.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nla.eval.steerability.bddl_bodies import (
    DEFAULT_GOAL_BDDL_DIR,
    validate_cf_target_bodies,
)


logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def collect_cf_eligible_example_ids(
    pairs_paths: Iterable[str | Path],
) -> set[str]:
    """Return activation ``example_id`` values that have at least one CF row.

    Uses ``source_example_id`` from each JSONL row — the same key
    :class:`SampledPositionDataset` yields as ``example_id`` and GRPO uses
    for counterfactual lookup.
    """
    out: set[str] = set()
    for raw_path in pairs_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"CF pairs file not found: {path}")
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("source_example_id")
                if sid is not None:
                    out.add(str(sid))
    return out


def load_grpo_cf_manifest(path: str | Path) -> set[str]:
    """Load ``example_ids`` from a manifest written by ``build_grpo_cf_manifest``."""
    obj = json.loads(Path(path).read_text())
    ids = obj.get("example_ids")
    if not isinstance(ids, list):
        raise ValueError(
            f"GRPO CF manifest {path} must contain an 'example_ids' list "
            f"(got {type(ids).__name__})"
        )
    return {str(x) for x in ids}


@dataclass(frozen=True)
class CounterfactualPair:
    source_example_id: str
    target_intent: str
    target_task: str
    target_env_name: str
    is_counterfactual: bool
    source_intent: str
    source_task: str


class CounterfactualPairSampler:
    """Per-activation lookup of (intent, env_name) targets.

    For each ``source_example_id`` we keep a list of candidate
    :class:`CounterfactualPair` rows. ``sample_for(source_example_ids)``
    returns one pair per id, chosen uniformly at random (with replacement
    on the candidate list itself, but deterministic per (seed, call counter)
    so the same call is reproducible).

    Activation ids that have no pair in the JSONL (e.g. because the label
    row was dropped by a min_bullet_lines filter) fall back to a sentinel
    intent + a sentinel env name; the caller can decide whether to skip
    those rows or score them as "match" against the loaded BDDL.
    """

    def __init__(
        self,
        pairs_path: str | Path,
        *,
        seed: int = 0,
        fallback_intent: str = "(no labeled intent)",
        fallback_env_name: str | None = None,
        additional_paths: Iterable[str | Path] | None = None,
        validate_bodies_in_bddl: bool = True,
        bddl_dir: str | Path | None = None,
    ) -> None:
        self.pairs_path = Path(pairs_path)
        self.additional_paths: list[Path] = [Path(p) for p in (additional_paths or [])]
        self.seed = int(seed)
        self.fallback_intent = fallback_intent
        self.fallback_env_name = fallback_env_name
        self.validate_bodies_in_bddl = bool(validate_bodies_in_bddl)
        self.bddl_dir = Path(bddl_dir) if bddl_dir is not None else DEFAULT_GOAL_BDDL_DIR
        self._instance_cache: dict[str, frozenset[str]] = {}
        self._by_id: dict[str, list[CounterfactualPair]] = defaultdict(list)
        # Per-bucket fingerprint set used to drop exact-duplicate pair rows
        # so a pair that appears in multiple files (or under both id keys
        # for the same row) is not double-weighted by ``random.choice``.
        self._seen_per_bucket: dict[str, set[tuple]] = defaultdict(set)
        self._call_counter = 0
        self._load()

    def _load(self) -> None:
        all_paths = [self.pairs_path, *self.additional_paths]
        total_rows = 0
        total_cf = 0
        total_skipped_bodies = 0
        for path in all_paths:
            n_rows, n_cf, n_skip = self._load_one(path)
            total_rows += n_rows
            total_cf += n_cf
            total_skipped_bodies += n_skip
        logger.info(
            "CounterfactualPairSampler: loaded %d pairs across %d ids "
            "(%.1f%% counterfactual) from %d file(s)"
            "%s",
            total_rows, len(self._by_id),
            100.0 * total_cf / max(1, total_rows), len(all_paths),
            (
                f"; skipped {total_skipped_bodies} rows with bodies missing "
                f"from target BDDL"
                if total_skipped_bodies
                else ""
            ),
        )

    def _load_one(self, path: Path) -> tuple[int, int, int]:
        if not path.exists():
            raise FileNotFoundError(f"counterfactual pairs file not found: {path}")
        n_rows = 0
        n_cf = 0
        n_skipped_bodies = 0
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("source_example_id")
                tgt_intent = obj.get("target_intent")
                if sid is None or tgt_intent is None:
                    continue
                target_task = str(obj.get("target_task", ""))
                target_env_name = str(obj.get("target_env_name", ""))
                if (
                    self.validate_bodies_in_bddl
                    and target_task
                    and target_env_name
                ):
                    body_issues = validate_cf_target_bodies(
                        target_task,
                        target_env_name,
                        self.bddl_dir,
                        instance_cache=self._instance_cache,
                    )
                    if body_issues:
                        n_skipped_bodies += 1
                        logger.debug(
                            "skip CF row sid=%s: %s", sid, body_issues
                        )
                        continue
                pair = CounterfactualPair(
                    source_example_id=str(sid),
                    target_intent=str(tgt_intent),
                    target_task=target_task,
                    target_env_name=target_env_name,
                    is_counterfactual=bool(obj.get("is_counterfactual", False)),
                    source_intent=str(obj.get("source_intent", "")),
                    source_task=str(obj.get("source_task", "")),
                )
                # Index under BOTH the source-side id and the label-side id
                # so a dataset that yields either flavor (see
                # ``SampledPositionDataset.example_id`` vs the JSONL's
                # ``source_example_id``) lands on the same candidate list.
                # Dedup is per-bucket so a row that aliases itself
                # (``example_id == source_example_id``) is only counted once
                # per key.
                keys: list[str] = [pair.source_example_id]
                eid = obj.get("example_id")
                if eid is not None:
                    eid_s = str(eid)
                    if eid_s != pair.source_example_id:
                        keys.append(eid_s)
                fp = (
                    pair.source_example_id,
                    pair.target_intent,
                    pair.target_task,
                    pair.target_env_name,
                )
                for key in keys:
                    bucket_seen = self._seen_per_bucket[key]
                    if fp in bucket_seen:
                        continue
                    bucket_seen.add(fp)
                    self._by_id[key].append(pair)
                n_rows += 1
                if pair.is_counterfactual:
                    n_cf += 1
        logger.info(
            "  %s: %d pairs (%.1f%% counterfactual)%s",
            path, n_rows, 100.0 * n_cf / max(1, n_rows),
            f", skipped_bodies={n_skipped_bodies}" if n_skipped_bodies else "",
        )
        return n_rows, n_cf, n_skipped_bodies

    def __len__(self) -> int:
        return len(self._by_id)

    def has(self, source_example_id: str) -> bool:
        return source_example_id in self._by_id

    def sample_for(
        self,
        source_example_ids: Sequence[str],
    ) -> list[CounterfactualPair]:
        """Return one pair per id (with reproducible per-call RNG)."""
        self._call_counter += 1
        rng = random.Random((self.seed * 0x9E3779B1) ^ self._call_counter)
        out: list[CounterfactualPair] = []
        for sid in source_example_ids:
            candidates = self._by_id.get(sid)
            if not candidates:
                out.append(CounterfactualPair(
                    source_example_id=sid,
                    target_intent=self.fallback_intent,
                    target_task="",
                    target_env_name=self.fallback_env_name or "",
                    is_counterfactual=False,
                    source_intent="",
                    source_task="",
                ))
                continue
            out.append(rng.choice(candidates))
        return out

    def intents(self, source_example_ids: Sequence[str]) -> list[str]:
        return [p.target_intent for p in self.sample_for(source_example_ids)]

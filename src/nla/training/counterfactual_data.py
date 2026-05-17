"""Sampler for (activation, target_intent) pairs used by sim-success GRPO.

The mining script ``scripts/training/mine_grpo_counterfactual_pairs.py``
produces a JSONL of pairs joining label rows in
``data/labels/libero_4suite_combined/labels.jsonl`` to a (possibly
counterfactual) target intent + target LIBERO BDDL env name.

At training time we want one such pair per activation in a GRPO batch. The
existing :class:`SampledPositionDataset` only yields ``(activation,
position_type)`` so this module adds a *parallel index* keyed by the
``source_example_id`` that the dataset already carries on its batches.
``CounterfactualPairSampler`` then returns a list of target intents +
env_names matched to a batch.

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
from typing import Sequence


logger = logging.getLogger(__name__)


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
    ) -> None:
        self.pairs_path = Path(pairs_path)
        self.seed = int(seed)
        self.fallback_intent = fallback_intent
        self.fallback_env_name = fallback_env_name
        self._by_id: dict[str, list[CounterfactualPair]] = defaultdict(list)
        self._call_counter = 0
        self._load()

    def _load(self) -> None:
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"counterfactual pairs file not found: {self.pairs_path}")
        n_rows = 0
        n_cf = 0
        with self.pairs_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("source_example_id")
                tgt = obj.get("target_intent")
                if sid is None or tgt is None:
                    continue
                pair = CounterfactualPair(
                    source_example_id=str(sid),
                    target_intent=str(tgt),
                    target_task=str(obj.get("target_task", "")),
                    target_env_name=str(obj.get("target_env_name", "")),
                    is_counterfactual=bool(obj.get("is_counterfactual", False)),
                    source_intent=str(obj.get("source_intent", "")),
                    source_task=str(obj.get("source_task", "")),
                )
                self._by_id[pair.source_example_id].append(pair)
                n_rows += 1
                if pair.is_counterfactual:
                    n_cf += 1
        logger.info(
            "CounterfactualPairSampler: loaded %d pairs across %d ids (%.1f%% counterfactual) from %s",
            n_rows, len(self._by_id), 100.0 * n_cf / max(1, n_rows), self.pairs_path,
        )

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

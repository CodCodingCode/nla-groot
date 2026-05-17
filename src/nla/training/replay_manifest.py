"""Join labeled-row `example_id` -> a replayable LeRobot trajectory step.

Action-head consistency needs to feed GR00T's policy the *original* observation
the activation came from so it can replay ``policy.get_action`` (baseline vs
AR-steered). Activations live in shards keyed by ``example_id``; the raw
observation lives in a LeRobot dataset keyed by ``(traj_idx, step_idx)``.

Two example_id flavors are supported:

* Single-suite extractions write ``traj{NNNNNN}_step{NNNNNN}``
  (``scripts/extraction/run_extract.py``).
* The 4-suite combiner prefixes each id with ``{suite}__``, e.g.
  ``goal__traj000001_step000058``
  (``scripts/training/combine_libero_4suite.py``).

This module
-----------

``ReplayEntry``: one row of the manifest.

``build_replay_manifest(...)``: scans an activation dump's ``index.jsonl``,
extracts suite/traj/step from each ``example_id``, resolves the LeRobot dataset
root for that suite, and emits a list of ``ReplayEntry``.

``ReplayManifest``: thin wrapper around the entries with O(1) lookup by
``example_id`` and JSONL serialization.

The manifest is build-once (a few seconds for ~100k rows); we cache it under
``<sft_output_dir>/aux/replay_manifest.jsonl`` so re-launches skip the work.

NOTE: We don't open the LeRobot datasets here. That is the consumer's job
(it needs ``Gr00tPolicy.modality_configs``), and we keep this module GR00T-free
so it imports cleanly in tests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from nla.extraction.storage import iter_records

logger = logging.getLogger(__name__)


# example_id formats:
#   "traj000001_step000058"                (single-suite extraction)
#   "goal__traj000001_step000058"          (4-suite combined)
_EXAMPLE_ID_RE = re.compile(
    r"^(?:(?P<suite>[a-zA-Z0-9_]+)__)?traj(?P<traj>\d+)_step(?P<step>\d+)$"
)


def parse_example_id(example_id: str) -> tuple[str | None, int, int]:
    """Return (suite_or_None, traj_idx, step_idx) parsed from ``example_id``.

    Raises ``ValueError`` if the id doesn't match the expected pattern; the
    caller can decide whether to skip the row or fail loudly.
    """
    m = _EXAMPLE_ID_RE.match(example_id)
    if m is None:
        raise ValueError(
            f"example_id {example_id!r} doesn't match traj/step pattern; "
            "extraction may have used a non-standard naming scheme."
        )
    return (
        m.group("suite"),
        int(m.group("traj")),
        int(m.group("step")),
    )


@dataclass(frozen=True)
class ReplayEntry:
    example_id: str
    suite: str | None       # None for single-suite dumps; "goal"/"object"/...
    traj_idx: int
    step_idx: int
    dataset_root: str       # absolute or repo-relative path to LeRobot dataset


class ReplayManifest:
    """O(1) `example_id` -> `ReplayEntry` lookup."""

    def __init__(self, entries: Iterable[ReplayEntry]) -> None:
        self._entries: list[ReplayEntry] = list(entries)
        self._by_id: dict[str, ReplayEntry] = {e.example_id: e for e in self._entries}
        if len(self._by_id) != len(self._entries):
            raise ValueError(
                "Duplicate example_id in replay manifest; check the activation "
                "dump's index.jsonl for collisions."
            )

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __contains__(self, example_id: str) -> bool:
        return example_id in self._by_id

    def get(self, example_id: str) -> ReplayEntry | None:
        return self._by_id.get(example_id)

    @property
    def entries(self) -> list[ReplayEntry]:
        return list(self._entries)

    @property
    def suites(self) -> list[str]:
        return sorted({e.suite for e in self._entries if e.suite is not None})

    # ------------------------------------------------------------- (de)serialization

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for e in self._entries:
                f.write(json.dumps(asdict(e)) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ReplayManifest":
        rows: list[ReplayEntry] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append(ReplayEntry(**d))
        return cls(rows)


def build_replay_manifest(
    activations_root: str | Path,
    dataset_roots_by_suite: dict[str | None, str | Path],
    *,
    cache_path: str | Path | None = None,
    force_rebuild: bool = False,
    skip_unparseable: bool = True,
) -> ReplayManifest:
    """Walk ``index.jsonl`` and join each example_id to its LeRobot dataset.

    Parameters
    ----------
    activations_root :
        Directory holding ``index.jsonl`` from an extraction dump.
    dataset_roots_by_suite :
        Mapping from suite name (e.g. ``"goal"``) to the LeRobot dataset root.
        Use ``None`` as a key for single-suite dumps that omit the prefix.
        Suites encountered in the dump but missing from this mapping are
        treated like ``skip_unparseable`` rows (skipped with a warning).
    cache_path :
        Optional path to read/write the manifest as JSONL. When the file
        already exists and ``force_rebuild`` is False, the cached manifest
        is loaded instead of rescanning the dump.
    force_rebuild :
        Ignore any cached file and rebuild from scratch.
    skip_unparseable :
        If True (default), log a warning and skip example_ids that don't
        match the ``traj/step`` pattern. If False, raise ``ValueError``.

    Returns
    -------
    ReplayManifest
    """
    activations_root = Path(activations_root)
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists() and not force_rebuild:
        logger.info("[replay_manifest] loading cached manifest %s", cache)
        return ReplayManifest.load(cache)

    # Normalize the mapping so we can do .get(suite) once per row.
    roots: dict[str | None, str] = {
        (k if k is None else str(k)): str(Path(v).resolve())
        for k, v in dataset_roots_by_suite.items()
    }

    rows: list[ReplayEntry] = []
    n_total = 0
    n_skipped_parse = 0
    n_skipped_root = 0
    unknown_suites: set[str | None] = set()
    for rec in iter_records(activations_root):
        n_total += 1
        try:
            suite, traj_idx, step_idx = parse_example_id(rec.example_id)
        except ValueError:
            if skip_unparseable:
                n_skipped_parse += 1
                continue
            raise

        root = roots.get(suite)
        if root is None:
            unknown_suites.add(suite)
            n_skipped_root += 1
            continue
        rows.append(
            ReplayEntry(
                example_id=rec.example_id,
                suite=suite,
                traj_idx=traj_idx,
                step_idx=step_idx,
                dataset_root=root,
            )
        )

    if n_skipped_parse:
        logger.warning(
            "[replay_manifest] skipped %d/%d rows: example_id unparseable",
            n_skipped_parse, n_total,
        )
    if n_skipped_root:
        logger.warning(
            "[replay_manifest] skipped %d/%d rows: no dataset_root mapping for suites %s",
            n_skipped_root, n_total, sorted(repr(s) for s in unknown_suites),
        )
    logger.info(
        "[replay_manifest] built %d / %d entries from %s (covered suites: %s)",
        len(rows), n_total, activations_root,
        sorted({e.suite for e in rows}, key=lambda x: (x is not None, x)),
    )

    manifest = ReplayManifest(rows)
    if cache is not None:
        manifest.save(cache)
        logger.info("[replay_manifest] cached -> %s", cache)
    return manifest

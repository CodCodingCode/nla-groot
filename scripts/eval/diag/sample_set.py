"""Shared loader for the AR-vs-AV diagnostic suite.

Loads the *exact same 100 samples* used by the cached n=100 CF steer eval
(``data/eval/v9_combined_12k_n100_cf_strided_cached.json``) and joins each
``source_example_id`` to everything the three diagnostics need:

    - CF pair metadata  (source/target intent, task, env, token position)
    - the real activation: ``h_pos`` [H]  (single token the AV consumes)
                           ``h_grid`` [128, H]  (image-patch grid = real-h ceiling)
    - the gold GPT caption (teacher-forced reconstruction target text)

All four joins were verified at 100/100 on the n=100 set. Tensor loading is
CPU-only (safetensors) so this module is importable without touching the GPU.

Join keys (confirmed):
    cached results .samples[].source_example_id
      -> pairs   : libero_goal_counterfactual_pairs_cfonly.jsonl[source_example_id]
      -> acts    : activations index.jsonl[example_id] -> shard_NNNNNN/act_<local>
      -> gold    : labels.jsonl where meta.(source_example_id, position_index,
                   position_type) matches
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # repo root

DEFAULT_RESULTS = "data/eval/v9_combined_12k_n100_cf_strided_cached.json"
DEFAULT_PAIRS = "data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl"
DEFAULT_ACTS_ROOT = "data/activations/libero_4suite_v4_combined"
DEFAULT_LABELS = "data/labels/libero_4suite_v4_combined/labels.jsonl"


@dataclass
class Sample:
    sid: str
    position_index: int
    position_type: str
    source_intent: str | None
    target_intent: str
    source_task: str | None
    target_task: str
    target_env_name: str
    gold_caption: str | None
    shard_id: int
    local_index: int
    h_pos: np.ndarray | None = field(default=None, repr=False)   # [H] float32
    h_grid: np.ndarray | None = field(default=None, repr=False)  # [128, H] float32


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_samples(
    *,
    results: str | Path = DEFAULT_RESULTS,
    pairs: str | Path = DEFAULT_PAIRS,
    acts_root: str | Path = DEFAULT_ACTS_ROOT,
    labels: str | Path = DEFAULT_LABELS,
    limit: int | None = None,
    load_tensors: bool = True,
) -> list[Sample]:
    """Load the cached-eval sample set, joined and (optionally) with tensors."""
    results, acts_root = _abs(results), _abs(acts_root)

    ids = [s["source_example_id"] for s in json.load(open(results))["samples"]]
    if limit is not None:
        ids = ids[:limit]

    pair_by_id = {}
    for r in _read_jsonl(_abs(pairs)):
        pair_by_id.setdefault(r["source_example_id"], r)

    act_idx = {r["example_id"]: r for r in _read_jsonl(acts_root / "index.jsonl")}

    gold_by_key: dict[tuple, str] = {}
    for r in _read_jsonl(_abs(labels)):
        m = r.get("meta") or {}
        k = (m.get("source_example_id"), m.get("position_index"), m.get("position_type"))
        if k[0] is not None and r.get("description"):
            gold_by_key.setdefault(k, r["description"])

    samples: list[Sample] = []
    missing = []
    for sid in ids:
        p = pair_by_id.get(sid)
        a = act_idx.get(sid)
        if p is None or a is None:
            missing.append(sid)
            continue
        pos_i, pos_t = p.get("position_index"), p.get("position_type")
        samples.append(
            Sample(
                sid=sid,
                position_index=pos_i,
                position_type=pos_t,
                source_intent=(p.get("source_intent") or None),
                target_intent=p["target_intent"],
                source_task=p.get("source_task"),
                target_task=p["target_task"],
                target_env_name=p["target_env_name"],
                gold_caption=gold_by_key.get((sid, pos_i, pos_t)),
                shard_id=a["shard_id"],
                local_index=a["local_index"],
            )
        )
    if missing:
        print(f"[sample_set] WARNING: {len(missing)} ids missing pair/activation; skipped")

    if load_tensors:
        _attach_tensors(samples, acts_root)
    return samples


def _attach_tensors(samples: list[Sample], acts_root: Path) -> None:
    """Load h_pos [H] and h_grid [128,H] per sample, grouping by shard."""
    from safetensors import safe_open

    by_shard: dict[int, list[Sample]] = {}
    for s in samples:
        by_shard.setdefault(s.shard_id, []).append(s)

    for shard_id, group in by_shard.items():
        path = acts_root / f"shard_{shard_id:06d}" / "activations.safetensors"
        with safe_open(str(path), framework="np") as f:
            for s in group:
                act = f.get_tensor(f"act_{s.local_index:06d}")        # [T, H]
                img = f.get_tensor(f"img_{s.local_index:06d}").astype(bool)  # [T]
                s.h_pos = np.ascontiguousarray(act[s.position_index], dtype=np.float32)
                s.h_grid = np.ascontiguousarray(act[img], dtype=np.float32)


if __name__ == "__main__":
    # CPU smoke test
    ss = load_samples(load_tensors=True)
    print(f"loaded {len(ss)} samples")
    s = ss[0]
    print("sid:", s.sid, "| target_task:", s.target_task)
    print("h_pos:", None if s.h_pos is None else s.h_pos.shape,
          "| h_grid:", None if s.h_grid is None else s.h_grid.shape)
    print("gold caption present:", s.gold_caption is not None)
    n_grid = sum(1 for x in ss if x.h_grid is not None and x.h_grid.shape == (128, 2048))
    n_gold = sum(1 for x in ss if x.gold_caption)
    print(f"h_grid==[128,2048] for {n_grid}/{len(ss)} | gold for {n_gold}/{len(ss)}")
    tasks = sorted({x.target_task for x in ss})
    print(f"distinct target_tasks: {len(tasks)}")

"""On-disk format for extracted GR00T activations.

Design constraints (from §5.2 of the plan + the NLA paper's data layout):

1. **Variable sequence length per example.**  We refuse to pre-pad: a Bridge
   episode-step at T=200 and a long-text example at T=400 each store their
   actual T. Per-token training samples a position uniformly inside each
   example so no padding is necessary at training time.

2. **Per-token granularity at storage time.**  We keep the full ``[T, 2048]``
   tensor per example so the same dump can serve (a) random-position SFT,
   (b) spatial-NLA-map experiments that need every position, and
   (c) per-token case-study walkthroughs.

3. **Read-mostly, append-write.**  Shards are write-once.  A run produces N
   ``shard_NNNNNN/`` directories under ``out_root`` and a top-level
   ``index.jsonl`` containing one row per example with (shard, example_id,
   T, task_index, episode_index, step_index, …).  No mutation after close.

Wire format
-----------
::

    <out_root>/
      manifest.json             # run-level config (model, layer, dtype, ...)
      index.jsonl               # one record per example across all shards
      shard_000000/
        activations.safetensors # keys: act_{i}, attn_{i}, img_{i}, ids_{i}
        meta.jsonl              # per-example metadata for this shard
      shard_000001/
        ...

Each shard's safetensors file is opened lazily and *only one example at a
time* is materialised (using ``safetensors.safe_open`` framework="pt").
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from safetensors.torch import safe_open, save_file

from nla.layer_spec import BACKBONE_EMBEDDING_DIM


SCHEMA_VERSION = 1


@dataclass
class ExampleRecord:
    """One example's metadata. Activations live in the shard's safetensors file."""

    example_id: str
    shard_id: int
    local_index: int                  # offset within the shard
    seq_len: int
    task_index: int | None = None
    task_text: str | None = None
    episode_index: int | None = None
    step_index: int | None = None
    embodiment_tag: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "ExampleRecord":
        d = json.loads(s)
        d.setdefault("extra", {})
        return cls(**d)


@dataclass
class RunManifest:
    """Top-level metadata describing how the activations were produced."""

    schema_version: int
    model_repo: str
    layer_module_path: str
    hidden_size: int
    activation_dtype: str             # "float32" / "bfloat16" / ...
    embodiment_tag: str | None
    num_examples: int = 0
    num_shards: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "RunManifest":
        d = json.loads(Path(path).read_text())
        d.setdefault("extra", {})
        return cls(**d)


def _key_for(prefix: str, local_index: int) -> str:
    """Stable, padded-decimal key so safetensors lookup is O(1) and ordered."""
    return f"{prefix}_{local_index:06d}"


class ActivationShardWriter:
    """Buffers examples in memory, flushes to a ``shard_NNNNNN/`` on close.

    Buffered tensors are kept on CPU (the hook is responsible for that).  When
    the shard reaches ``max_examples_per_shard`` it is flushed automatically
    and a new shard is started.

    Usage::

        writer = ActivationShardWriter(out_root, manifest, max_examples_per_shard=512)
        for example_id, captured, meta in stream:
            writer.write(example_id, captured, meta)
        writer.close()
    """

    def __init__(
        self,
        out_root: str | os.PathLike,
        manifest: RunManifest,
        *,
        max_examples_per_shard: int = 512,
    ) -> None:
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.max_examples_per_shard = int(max_examples_per_shard)

        self._buffer: dict[str, torch.Tensor] = {}
        self._meta: list[ExampleRecord] = []
        self._index: list[ExampleRecord] = []
        self._current_shard: int = 0
        self._closed = False

    @property
    def current_shard_size(self) -> int:
        return len(self._meta)

    @property
    def total_written(self) -> int:
        return len(self._index)

    def write(
        self,
        example_id: str,
        features: torch.Tensor,        # [T, hidden]
        attention_mask: torch.Tensor,  # [T]   bool
        image_mask: torch.Tensor,      # [T]   bool
        input_ids: torch.Tensor | None = None,  # [T]
        *,
        task_index: int | None = None,
        task_text: str | None = None,
        episode_index: int | None = None,
        step_index: int | None = None,
        embodiment_tag: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Add one example to the active shard. May trigger a flush."""
        if self._closed:
            raise RuntimeError("Writer is already closed.")

        assert features.ndim == 2, f"features must be [T, H], got {tuple(features.shape)}"
        T, H = features.shape
        if H != self.manifest.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: manifest says {self.manifest.hidden_size}, "
                f"got {H}. Did you change layer_spec?"
            )
        if attention_mask.shape != (T,):
            raise ValueError(f"attention_mask must be [T={T}], got {tuple(attention_mask.shape)}")
        if image_mask.shape != (T,):
            raise ValueError(f"image_mask must be [T={T}], got {tuple(image_mask.shape)}")
        if input_ids is not None and input_ids.shape != (T,):
            raise ValueError(f"input_ids must be [T={T}], got {tuple(input_ids.shape)}")

        local_index = len(self._meta)
        self._buffer[_key_for("act", local_index)] = features.contiguous()
        self._buffer[_key_for("attn", local_index)] = attention_mask.to(torch.bool).contiguous()
        self._buffer[_key_for("img", local_index)] = image_mask.to(torch.bool).contiguous()
        if input_ids is not None:
            self._buffer[_key_for("ids", local_index)] = input_ids.to(torch.int64).contiguous()

        rec = ExampleRecord(
            example_id=example_id,
            shard_id=self._current_shard,
            local_index=local_index,
            seq_len=int(T),
            task_index=task_index,
            task_text=task_text,
            episode_index=episode_index,
            step_index=step_index,
            embodiment_tag=embodiment_tag,
            extra=extra or {},
        )
        self._meta.append(rec)

        if len(self._meta) >= self.max_examples_per_shard:
            self._flush_shard()

    def _flush_shard(self) -> None:
        if not self._meta:
            return
        shard_dir = self.out_root / f"shard_{self._current_shard:06d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        save_file(self._buffer, str(shard_dir / "activations.safetensors"))

        with (shard_dir / "meta.jsonl").open("w") as f:
            for rec in self._meta:
                f.write(rec.to_json() + "\n")

        self._index.extend(self._meta)
        self._meta.clear()
        self._buffer.clear()
        self._current_shard += 1

    def close(self) -> None:
        if self._closed:
            return
        self._flush_shard()

        with (self.out_root / "index.jsonl").open("w") as f:
            for rec in self._index:
                f.write(rec.to_json() + "\n")

        self.manifest.num_examples = len(self._index)
        self.manifest.num_shards = self._current_shard
        self.manifest.save(self.out_root / "manifest.json")
        self._closed = True

    def __enter__(self) -> "ActivationShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()


class ActivationShardReader:
    """Random-access reader over a dump produced by ``ActivationShardWriter``.

    Each ``__getitem__`` opens the relevant shard's safetensors lazily and
    returns only the requested example's slices. Files are reopened per
    access; for tight loops use ``iter_examples`` or batch by shard.
    """

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.manifest = RunManifest.load(self.root / "manifest.json")
        index_path = self.root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index.jsonl at {index_path}")
        self._records: list[ExampleRecord] = []
        with index_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(ExampleRecord.from_json(line))
        self._by_id = {rec.example_id: i for i, rec in enumerate(self._records)}
        if self.manifest.hidden_size != BACKBONE_EMBEDDING_DIM:
            # Not fatal — we may legitimately load older runs with different dims.
            pass

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[ExampleRecord]:
        return self._records

    def get(self, example_id: str) -> dict[str, torch.Tensor]:
        return self[self._by_id[example_id]]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._records[idx]
        shard_path = self.root / f"shard_{rec.shard_id:06d}" / "activations.safetensors"
        out: dict[str, torch.Tensor] = {"_record": rec}  # type: ignore[dict-item]
        with safe_open(str(shard_path), framework="pt") as f:
            out["features"] = f.get_tensor(_key_for("act", rec.local_index))
            out["attention_mask"] = f.get_tensor(_key_for("attn", rec.local_index))
            out["image_mask"] = f.get_tensor(_key_for("img", rec.local_index))
            ids_key = _key_for("ids", rec.local_index)
            if ids_key in f.keys():
                out["input_ids"] = f.get_tensor(ids_key)
        return out

    def iter_examples(
        self,
        record_filter=None,  # callable(ExampleRecord) -> bool, or None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Stream all examples shard-by-shard, opening each safetensors once."""
        by_shard: dict[int, list[ExampleRecord]] = {}
        for rec in self._records:
            if record_filter is not None and not record_filter(rec):
                continue
            by_shard.setdefault(rec.shard_id, []).append(rec)
        for shard_id, recs in sorted(by_shard.items()):
            shard_path = self.root / f"shard_{shard_id:06d}" / "activations.safetensors"
            with safe_open(str(shard_path), framework="pt") as f:
                keys = set(f.keys())
                for rec in recs:
                    item: dict[str, Any] = {
                        "_record": rec,
                        "features": f.get_tensor(_key_for("act", rec.local_index)),
                        "attention_mask": f.get_tensor(_key_for("attn", rec.local_index)),
                        "image_mask": f.get_tensor(_key_for("img", rec.local_index)),
                    }
                    ids_key = _key_for("ids", rec.local_index)
                    if ids_key in keys:
                        item["input_ids"] = f.get_tensor(ids_key)
                    yield item


def iter_records(root: str | os.PathLike) -> Iterable[ExampleRecord]:
    """Cheap iterator over the index without touching activation files."""
    with (Path(root) / "index.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield ExampleRecord.from_json(line)

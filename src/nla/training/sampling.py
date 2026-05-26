"""Per-token position sampling for NLA training.

Per plan §5.2 (the change that follows Grant et al. 2026): we sample ONE token
position per activation example rather than mean-pooling. The position is drawn
from a mixture:

    40%  last_text   - last non-pad, non-image token
    40%  image_patch - uniform random over image-token positions
    20%  anchor      - the **last valid (non-pad) sequence token**, which may
                       be an image-patch token if the sequence ends in vision.
                       This matches ``nla.extraction.sampler._anchor_index`` so
                       labeled positions, training samples, and stratified FVE
                       all refer to the same token role.

This module operates on already-extracted activations from disk. Each example
has shape `[T, H]` with `attention_mask[T]` and `image_mask[T]` companions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from nla.extraction.sampler import _anchor_index
from nla.layer_spec import POSITION_MIX


@dataclass
class PerTokenBatch:
    """One training batch of per-token activations."""

    activations: torch.Tensor       # [B, H]
    position_type: list[str]        # length B: "last_text" | "image_patch" | "anchor"
    position_index: torch.Tensor    # [B] absolute token index for traceability
    seq_len: torch.Tensor           # [B] total non-pad length of the parent example
    example_id: list[str]           # length B for traceability


class TokenPositionSampler:
    """Stateful sampler with reproducible per-step seeding."""

    def __init__(self, position_mix: dict[str, float] | None = None, seed: int = 0):
        self.mix = position_mix or POSITION_MIX
        # Normalize and accumulate.
        total = sum(self.mix.values())
        self._cum = []
        self._labels = []
        acc = 0.0
        for k, v in self.mix.items():
            acc += v / total
            self._cum.append(acc)
            self._labels.append(k)
        self._rng = random.Random(seed)

    def _draw_position_type(self) -> str:
        r = self._rng.random()
        for label, c in zip(self._labels, self._cum):
            if r <= c:
                return label
        return self._labels[-1]

    def sample(
        self,
        attention_mask: torch.Tensor,   # [T] bool
        image_mask: torch.Tensor,        # [T] bool
        force_type: str | None = None,
    ) -> tuple[str, int]:
        """Return (position_type, index) for one example.

        When the requested type is empty (no image tokens etc.) we fall back
        deterministically: image_patch → last_text → anchor → 0.
        """
        T = int(attention_mask.numel())
        real = attention_mask.bool().tolist()
        is_image = image_mask.bool().tolist()
        text_positions = [i for i in range(T) if real[i] and not is_image[i]]
        image_positions = [i for i in range(T) if real[i] and is_image[i]]

        ptype = force_type or self._draw_position_type()
        if ptype == "image_patch" and image_positions:
            return "image_patch", self._rng.choice(image_positions)
        if ptype == "last_text" and text_positions:
            return "last_text", text_positions[-1]
        if ptype == "anchor":
            anchor_idx = _anchor_index(attention_mask)
            if anchor_idx is not None:
                return "anchor", anchor_idx
        # Fallbacks
        if text_positions:
            return "last_text", text_positions[-1]
        if image_positions:
            return "image_patch", image_positions[0]
        return "anchor", 0


def sample_token_position(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    rng: random.Random | None = None,
    position_mix: dict[str, float] | None = None,
    force_type: str | None = None,
) -> tuple[str, int]:
    """Functional convenience wrapper around `TokenPositionSampler.sample`."""
    sampler = TokenPositionSampler(position_mix=position_mix, seed=0)
    if rng is not None:
        sampler._rng = rng
    return sampler.sample(attention_mask, image_mask, force_type=force_type)


class StratifiedPositionBatchSampler:
    """BatchSampler that enforces per-batch position-type quotas.

    ``WeightedRandomSampler`` rebalances at the epoch level — frequencies
    converge over many batches but a single batch can still be e.g. 95%
    last_text by chance. For policy-effect SFT this matters: the image_patch
    rows are the only ones that actually exercise the steering hook at the
    spatial slot the policy reads, and we need them to appear every batch
    so the action-consistency gradient gets a steady image_patch signal.

    This sampler partitions row indices by ``position_type`` and yields
    batches that draw a fixed quota from each bucket. Quotas are computed
    from the target ``position_mix``; any leftover slot from rounding goes
    to the bucket with the largest residual (largest-remainder method).

    Sampling within a bucket is with replacement (matches
    WeightedRandomSampler) so a minority class can fill its quota even when
    its bucket is small. Length is ``num_batches`` × ``batch_size``.

    Args:
        position_types: per-row ``position_type`` strings (len == n rows).
        batch_size: rows per batch.
        position_mix: target mix mapping ``position_type -> float``. Need
            not sum to 1; normalized internally. Rows whose type is not in
            ``position_mix`` are excluded.
        num_batches: total number of batches to yield per epoch. Defaults
            to ``ceil(n_eligible / batch_size)`` for parity with shuffle.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        position_types: list[str],
        *,
        batch_size: int,
        position_mix: dict[str, float],
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}.")
        if not position_mix:
            raise ValueError("position_mix must be non-empty.")

        # Bucket row indices by position_type.
        self._buckets: dict[str, list[int]] = {}
        for i, pt in enumerate(position_types):
            if pt in position_mix:
                self._buckets.setdefault(pt, []).append(i)

        if not self._buckets:
            raise RuntimeError(
                "No rows match any position_type in position_mix. "
                f"position_mix={sorted(position_mix)}; "
                f"seen types={sorted(set(position_types))}."
            )

        # Normalize mix over present buckets only (so dropping a class
        # gracefully reweights the rest, matching the V5 50/50 auto-fold).
        present_mix = {k: float(position_mix[k]) for k in self._buckets}
        total = sum(present_mix.values())
        if total <= 0:
            raise ValueError("position_mix weights must sum to > 0.")
        self._mix_norm = {k: v / total for k, v in present_mix.items()}

        # Largest-remainder quotas summing exactly to batch_size.
        raw = {k: v * batch_size for k, v in self._mix_norm.items()}
        floor = {k: int(v) for k, v in raw.items()}
        used = sum(floor.values())
        remainder = batch_size - used
        # Hand out the leftover slots to the largest fractional residuals.
        residuals = sorted(
            ((raw[k] - floor[k], k) for k in raw),
            key=lambda t: (-t[0], t[1]),  # ties -> alphabetic for determinism
        )
        for i in range(remainder):
            _, k = residuals[i % len(residuals)]
            floor[k] += 1
        self._quotas = floor  # type -> count per batch

        n_eligible = sum(len(v) for v in self._buckets.values())
        if num_batches is None:
            num_batches = max(1, (n_eligible + batch_size - 1) // batch_size)
        self._num_batches = int(num_batches)
        self._batch_size = int(batch_size)

        self._rng = random.Random(int(seed))

    @property
    def quotas(self) -> dict[str, int]:
        """Per-batch row counts, keyed by position_type."""
        return dict(self._quotas)

    def __iter__(self):
        for _ in range(self._num_batches):
            batch: list[int] = []
            for ptype, q in self._quotas.items():
                pool = self._buckets[ptype]
                # With-replacement so a minority class fills its quota.
                batch.extend(self._rng.choices(pool, k=q))
            self._rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self._num_batches

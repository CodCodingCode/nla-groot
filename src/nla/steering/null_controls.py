"""Non-semantic causal control vectors for closed-loop steerability eval.

These helpers exist so SFT-vs-GRPO predicate gains can be separated from any
``image_patch``-shaped, norm-matched vector moving the policy. The original
matched-null sweep is in ``scripts/eval/nla_steer_leverage_sweep.py``
(open-loop only); this module lifts the helper out so the closed-loop CF
compare script and any future scorecard can share it.

Two controls are provided:

``matched_null_vec(real_vec, seed)``
    A Gaussian draw rescaled to ``||real_vec||_2``. Tests "does any vector
    of the same magnitude steer just as well?" Negative result → AR vector
    is doing semantic work; positive result → norm injection alone explains
    the gain.

``shuffled_vec(real_vec, seed)``
    Coordinate-shuffled copy of ``real_vec`` (same elements, scrambled). Same
    L2 norm and the same per-dimension distribution as the real vector. Use
    this as a stronger null when the real vector has heavy-tailed or
    direction-aligned structure that ``randn`` would not produce.

Both return float32 CPU tensors with the same shape as the input.
"""

from __future__ import annotations

import torch


def matched_null_vec(real_vec: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a Gaussian vector rescaled to ``||real_vec||_2``.

    Deterministic per ``seed``. Independent of ``real_vec``'s direction
    (only its magnitude is preserved), which makes it a strict
    non-semantic null when paired with the same ``SteerSpec`` placement.
    """
    real_cpu = real_vec.detach().to(dtype=torch.float32, device="cpu")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    z = torch.randn(real_cpu.shape, generator=gen, dtype=torch.float32)
    target_norm = float(torch.linalg.norm(real_cpu))
    z_norm = float(torch.linalg.norm(z))
    if z_norm < 1e-12:
        return z
    return z * (target_norm / z_norm)


def shuffled_vec(real_vec: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a coordinate-permuted copy of ``real_vec`` (same L2 norm)."""
    real_cpu = real_vec.detach().to(dtype=torch.float32, device="cpu")
    flat = real_cpu.reshape(-1)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(flat.numel(), generator=gen)
    return flat[perm].reshape(real_cpu.shape)


__all__ = ["matched_null_vec", "shuffled_vec"]

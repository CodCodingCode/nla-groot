"""Action-head consistency: penalize ``||pi(h_real) - pi(h_AR)||`` during SFT.

Motivation
----------

Reconstruction MSE / cosine in α-scaled space (the AR's regular SFT objective)
shows that AR can produce a vector ``ĥ`` close to ``h_real``. It does **not**
show that GR00T's frozen action head reads ``ĥ`` the same way it reads
``h_real``. The steering eval (see ``data/eval/steerability_v1_vs_v3/...``)
confirms the gap: AR vectors strong enough to suppress the original task
still don't elicit the new task because the action head was never trained on
``(AR(z_for_caption_X), scene_showing_Y)`` mismatches.

This module adds the missing SFT-time pressure. For a microbatch of labeled
rows, we:

1. Look up the original LeRobot observation (via ``replay_manifest``).
2. Run a *frozen* ``Gr00tPolicy`` forward to get the baseline action
   ``a_real = pi(h_real)`` (cached after first compute).
3. Run the same policy again with a forward hook that *replaces* the backbone
   image-token features with ``ĥ = AR(caption)`` — a **differentiable** hook
   that propagates gradients into AR.
4. Compute ``L_consistency = mse(pi(ĥ), a_real.detach())`` and add it to the
   total SFT loss with weight ``cfg.action_consistency_weight``.

Design notes
------------

* The GR00T policy is **frozen** end-to-end. No gradients update its weights;
  the only learnable signal flows back into AR via the steer vector.
* We default to OFF (``action_consistency_weight = 0``). When off, this
  module is never imported by ``sft.py`` and SFT behavior is byte-identical
  to V4 baseline.
* Microbatch size for the consistency forward is intentionally small
  (``max_microbatch_per_step``, default 1) because the policy forward runs
  bf16 GR00T (~14 GB peak) on top of the live AV/AR (~20 GB) on a single
  GPU.
* Cadence is controlled by ``every_n_steps`` (default 8) so the extra wall
  clock is bounded.
* Tested with a ``FakePolicy`` shim so unit tests don't need the
  ``gr00t`` Python module.

The CLI wire-up lives in ``scripts/training/run_sft.py``; this file only
exposes the kernel.
"""

from __future__ import annotations

import contextlib
import logging
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from nla.layer_spec import BACKBONE_EMBEDDING_DIM
from nla.steering.backbone_steer import (
    SteerSpec,
    resolve_steer_indices,
)
from nla.training.replay_manifest import ReplayEntry, ReplayManifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable steer hook (a sibling of BackboneFeaturesSteerHook).
# ---------------------------------------------------------------------------

class DifferentiableBackboneSteerHook:
    """Replace backbone features with a tensor that *carries autograd*.

    The production ``BackboneFeaturesSteerHook`` deliberately detaches its
    steer vector (``self._steer_cpu = steer_vec.detach().float().cpu()``)
    because steering is an inference-time intervention.  For the SFT
    consistency objective we need gradients to flow through the steer into
    AR, so this variant keeps the steer tensor live on the same device/dtype
    as the policy.

    Args:
        steer_vec: ``[H]`` tensor (``requires_grad=True``) on the policy's
            device. Caller is responsible for upcasting/downcasting before
            passing it in if the policy uses bf16.
        spec: where to apply the steer along ``T``.
        batch_index: which batch row to modify (matches the production hook).
    """

    def __init__(
        self,
        steer_vec: torch.Tensor,
        spec: SteerSpec,
        *,
        batch_index: int = 0,
    ) -> None:
        if steer_vec.dim() == 2 and steer_vec.shape[0] == 1:
            steer_vec = steer_vec.squeeze(0)
        if steer_vec.dim() != 1:
            raise ValueError(
                f"steer_vec must be [H]; got shape {tuple(steer_vec.shape)}"
            )
        if int(steer_vec.shape[0]) != BACKBONE_EMBEDDING_DIM:
            raise ValueError(
                f"steer_vec dim {steer_vec.shape[0]} != "
                f"BACKBONE_EMBEDDING_DIM={BACKBONE_EMBEDDING_DIM}"
            )
        self.steer_vec = steer_vec
        self.spec = spec
        self.batch_index = int(batch_index)
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs
        feats = output["backbone_features"]
        attn = output["backbone_attention_mask"]
        img_m = output["image_mask"]

        idxs = resolve_steer_indices(
            attn, img_m, self.spec, batch_index=self.batch_index
        )

        steer = self.steer_vec.to(device=feats.device, dtype=feats.dtype)
        blend = float(self.spec.blend)
        blend = max(0.0, min(1.0, blend))

        # In-place replacement breaks autograd; clone is required.
        new_feats = feats.clone()
        bi = self.batch_index
        for t in idxs:
            if blend <= 0.0:
                continue
            base = feats[bi, t]
            if blend >= 1.0:
                new_feats[bi, t] = steer
            else:
                new_feats[bi, t] = (1.0 - blend) * base + blend * steer
        output["backbone_features"] = new_feats


@contextlib.contextmanager
def attach_differentiable_backbone_steer(
    backbone: torch.nn.Module,
    steer_vec: torch.Tensor,
    spec: SteerSpec,
    *,
    batch_index: int = 0,
) -> Iterator[DifferentiableBackboneSteerHook]:
    """Like ``attach_backbone_steer`` but autograd-friendly."""
    hook_impl = DifferentiableBackboneSteerHook(
        steer_vec, spec, batch_index=batch_index
    )
    handle = backbone.register_forward_hook(hook_impl)
    hook_impl._handle = handle
    try:
        yield hook_impl
    finally:
        handle.remove()
        hook_impl._handle = None


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

# Type aliases ---------------------------------------------------------------
# A policy is anything with a ``model.backbone`` attribute (for the hook) and a
# ``get_action(observation) -> dict[str, Tensor]`` method, matching
# ``Gr00tPolicy`` / ``Gr00tSimPolicyWrapper`` and our test ``FakePolicy``.
PolicyLike = Any

# An observation-builder maps a ``ReplayEntry`` -> observation dict.
ObsBuilder = Callable[[ReplayEntry], dict[str, Any]]

# A factory that, given the loaded policy, returns an ``ObsBuilder``. The
# factory pattern exists because the real LeRobot-backed obs_builder needs
# ``policy.modality_configs`` to construct ``LeRobotEpisodeLoader``, and the
# policy is loaded lazily inside the kernel.
ObsBuilderFactory = Callable[[PolicyLike], ObsBuilder]

# A policy-loader is a thunk that returns the frozen policy on first call.
PolicyLoader = Callable[[], PolicyLike]


@dataclass
class ActionConsistencyConfig:
    """Knobs for the consistency loss; defaults keep it inert."""

    weight: float = 0.0
    every_n_steps: int = 8
    max_microbatch_per_step: int = 1
    placement: str = "image_patch_all"   # broadcasts ĥ across image tokens
    blend: float = 1.0
    # When True, only run consistency forward on rows whose
    # `position_type == "image_patch"` (the only ptype where ĥ is meant to
    # replace patch features at training time).
    image_patch_rows_only: bool = True
    # Cache the baseline action per example_id so the policy forward only
    # runs once per row across the whole run.
    #
    # TODO(action-consistency-cache-bound): the cache is unbounded — at scale
    # (~100k unique example_ids in V4) it grows host-RAM monotonically. Cap or
    # switch to LRU before turning the flag on in long jobs. Each entry is
    # O(action_dim) floats on CPU so the absolute cost is small, but the
    # design intent should be explicit.
    cache_baseline_actions: bool = True


@dataclass
class ActionConsistencyDiagnostics:
    """One-step training-loop telemetry written into metrics.jsonl."""

    n_rows: int = 0
    loss: float = 0.0
    baseline_cache_hits: int = 0
    baseline_cache_misses: int = 0
    delta_action_norm: float = 0.0
    per_key_delta_max_abs: dict[str, float] = field(default_factory=dict)


# Helpers --------------------------------------------------------------------


def _flatten_action_dict(out: Any) -> dict[str, torch.Tensor]:
    """Mirror ``nla.steering.action_delta.policy_get_action`` (without unwrap).

    The kernel may receive a nested action dict from a real ``Gr00tPolicy``;
    we dot-flatten it so ``action.world_vector`` and ``action.gripper``
    line up consistently across baseline and steered calls.
    """
    if isinstance(out, tuple) and len(out) >= 1:
        out = out[0]
    if not isinstance(out, dict):
        raise RuntimeError(
            f"action consistency: unexpected get_action() return type {type(out)}"
        )
    if any(isinstance(v, dict) for v in out.values()):
        flat: dict[str, torch.Tensor] = {}
        for k, v in out.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v
        return flat
    return out


def _as_action_tensor(
    actions: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Concatenate action-dict values into one flat tensor (sorted by key).

    Concatenation order is deterministic by sorted key so baseline-vs-steered
    L2 is well-defined and reproducible.
    """
    parts: list[torch.Tensor] = []
    for k in sorted(actions.keys()):
        v = actions[k]
        if isinstance(v, torch.Tensor):
            t = v
        else:
            t = torch.as_tensor(v)
        if not torch.is_floating_point(t):
            t = t.float()
        parts.append(t.to(device=device, dtype=dtype).reshape(-1))
    if not parts:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.cat(parts, dim=0)


class ActionConsistencyKernel:
    """Drive consistency forwards under a frozen GR00T policy.

    Lifecycle
    ---------

    1. Construct (no GR00T loaded yet).
    2. ``ensure_loaded()`` lazily invokes the policy loader (~70s+ for real
       GR00T) and freezes the policy in ``eval()`` mode with
       ``requires_grad_(False)``.
    3. Per training step the caller invokes ``consistency_loss(...)`` with
       the AR module and the labeled batch; the kernel selects a slice of
       rows, materializes steer vectors via the live AR, runs frozen
       policy.get_action under the differentiable hook, and returns
       ``(loss, diagnostics)``.

    The kernel does **not** touch the AR module's optimizer; it only
    returns a loss tensor that the SFT loop adds to its own ``.backward()``.
    """

    def __init__(
        self,
        cfg: ActionConsistencyConfig,
        *,
        manifest: ReplayManifest,
        policy_loader: PolicyLoader,
        ar_module: torch.nn.Module,
        obs_builder: ObsBuilder | None = None,
        obs_builder_factory: ObsBuilderFactory | None = None,
        device: torch.device | str = "cuda",
        on_baseline_compute: Callable[[str, dict[str, torch.Tensor]], None] | None = None,
    ) -> None:
        if cfg.weight < 0:
            raise ValueError("action_consistency weight must be >= 0")
        if cfg.every_n_steps <= 0:
            raise ValueError("every_n_steps must be >= 1")
        if cfg.max_microbatch_per_step <= 0:
            raise ValueError("max_microbatch_per_step must be >= 1")
        if obs_builder is None and obs_builder_factory is None:
            raise ValueError(
                "ActionConsistencyKernel needs either obs_builder (eager; "
                "used in tests with FakePolicy) or obs_builder_factory "
                "(lazy; resolved inside ensure_loaded once the real "
                "policy.modality_configs is available)."
            )
        self.cfg = cfg
        self.manifest = manifest
        self._policy_loader = policy_loader
        self._obs_builder: ObsBuilder | None = obs_builder
        self._obs_builder_factory = obs_builder_factory
        self.ar = ar_module
        self.device = torch.device(device)
        self._policy: PolicyLike | None = None
        self._baseline_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._on_baseline_compute = on_baseline_compute

    # -- lazy loading ------------------------------------------------------

    @property
    def policy(self) -> PolicyLike:
        if self._policy is None:
            raise RuntimeError(
                "ActionConsistencyKernel: call ensure_loaded() before policy."
            )
        return self._policy

    def ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        logger.info("[action_consistency] lazy-loading frozen policy...")
        policy = self._policy_loader()
        # Freeze every parameter we can find.
        for attr in ("model", "_model"):
            mod = getattr(policy, attr, None)
            if mod is not None and hasattr(mod, "parameters"):
                for p in mod.parameters():
                    p.requires_grad_(False)
                if hasattr(mod, "eval"):
                    mod.eval()
        self._policy = policy
        # Resolve the obs_builder now that we have a live policy. The factory
        # path is used in production (real Gr00tPolicy + LeRobot loaders);
        # the eager path is kept for tests with FakePolicy.
        if self._obs_builder is None:
            assert self._obs_builder_factory is not None  # constructor checks
            logger.info("[action_consistency] building obs_builder from factory.")
            self._obs_builder = self._obs_builder_factory(policy)
        logger.info("[action_consistency] policy loaded.")

    # -- candidate selection ----------------------------------------------

    def select_rows(
        self,
        example_ids: Sequence[str],
        position_types: Sequence[str],
    ) -> list[int]:
        """Return indices into ``example_ids`` admissible for consistency this step."""
        chosen: list[int] = []
        for i, eid in enumerate(example_ids):
            if eid not in self.manifest:
                continue
            if self.cfg.image_patch_rows_only and position_types[i] != "image_patch":
                continue
            chosen.append(i)
            if len(chosen) >= self.cfg.max_microbatch_per_step:
                break
        return chosen

    # -- baseline actions -------------------------------------------------

    def _compute_baseline(self, entry: ReplayEntry) -> dict[str, torch.Tensor]:
        """Baseline forward without steer, returning the same shape as steered.

        We mirror ``_steered_action``'s surrogate (``action_pred`` in
        normalized space) so the MSE downstream is well-defined. Baseline
        runs under ``inference_mode`` because we don't need its gradient.

        For the in-process FakePolicy (no GR00T install), we fall back to
        the legacy ``policy.get_action(obs)`` surface; that path is used
        only in tests.
        """
        assert self._obs_builder is not None, "ensure_loaded() must run first"
        obs = self._obs_builder(entry)
        policy = self.policy
        model = getattr(policy, "model", None)
        has_real_groot = (
            model is not None
            and hasattr(model, "action_head")
            and hasattr(model, "prepare_input")
            and hasattr(model.action_head, "get_action_with_features")
        )

        if has_real_groot:
            from gr00t.data.types import MessageType
            from gr00t.policy.gr00t_policy import _rec_to_dtype

            with torch.inference_mode():
                unbatched = policy._unbatch_observation(obs)
                processed_inputs = []
                for o in unbatched:
                    vla_step = policy._to_vla_step_data(o)
                    messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
                    processed_inputs.append(policy.processor(messages))
                collated = policy.collate_fn(processed_inputs)
                collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
                inner = collated["inputs"] if "inputs" in collated else collated
                backbone_inputs, action_inputs = model.prepare_input(dict(inner))
                backbone_outputs = model.backbone(backbone_inputs)
                features = model.action_head._encode_features(
                    backbone_outputs, action_inputs,
                )
                action_outputs = model.action_head.get_action_with_features(
                    backbone_features=features.backbone_features,
                    state_features=features.state_features,
                    embodiment_id=action_inputs.embodiment_id,
                    backbone_output=backbone_outputs,
                    action_input=action_inputs,
                    options=None,
                )
            action_pred = action_outputs["action_pred"].clone()
            flat = {"action_pred": action_pred}
        else:
            with torch.inference_mode():
                raw = policy.get_action(obs)
            flat = _flatten_action_dict(raw)

        cached: dict[str, torch.Tensor] = {}
        for k, v in flat.items():
            if isinstance(v, torch.Tensor):
                t = v
            else:
                try:
                    t = torch.as_tensor(v)
                except Exception:
                    continue
            if not torch.is_floating_point(t):
                t = t.float()
            cached[k] = t.detach().to("cpu", dtype=torch.float32).contiguous().reshape(-1)
        if self._on_baseline_compute is not None:
            self._on_baseline_compute(entry.example_id, cached)
        return cached

    def get_baseline(self, entry: ReplayEntry) -> tuple[dict[str, torch.Tensor], bool]:
        """Returns ``(action_dict, was_cache_hit)``."""
        if self.cfg.cache_baseline_actions and entry.example_id in self._baseline_cache:
            return self._baseline_cache[entry.example_id], True
        baseline = self._compute_baseline(entry)
        if self.cfg.cache_baseline_actions:
            self._baseline_cache[entry.example_id] = baseline
        return baseline, False

    # -- steered forward --------------------------------------------------

    def _steered_action(
        self,
        entry: ReplayEntry,
        steer_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a differentiable steered forward against the frozen policy.

        ``Gr00tPolicy.get_action`` wraps the whole model forward in
        ``torch.inference_mode``, and ``Gr00tN1d7ActionHead.get_action``
        carries a ``@torch.no_grad`` decorator. Either is sufficient to
        sever the autograd graph between ``steer_vec`` and the predicted
        action, defeating the purpose of action-head consistency.

        We bypass both by:
          1. Running the policy's preprocessing path explicitly (cheap,
             gradient-irrelevant) to obtain the same ``collated_inputs``
             that ``_get_action`` would build.
          2. Calling ``model.backbone`` directly under our differentiable
             steer hook so the hook's writes participate in autograd.
          3. Calling ``action_head.get_action_with_features`` directly
             (NOT ``action_head.get_action``) so the ``@torch.no_grad``
             decorator is sidestepped while reusing the same flow-matching
             denoising loop.

        Returns ``{"action_pred": <torch.Tensor[A]>}`` flattened over the
        batch + horizon dims so downstream MSE has a well-defined shape.
        """
        assert self._obs_builder is not None, "ensure_loaded() must run first"

        obs = self._obs_builder(entry)
        policy = self.policy
        model = getattr(policy, "model", None)
        backbone = model.backbone

        has_real_groot = (
            hasattr(model, "action_head")
            and hasattr(model, "prepare_input")
            and hasattr(model.action_head, "get_action_with_features")
        )
        if not has_real_groot:
            # FakePolicy path (tests only). The fake policy returns a
            # differentiable tensor through plain ``get_action(obs)``.
            spec = SteerSpec(placement=self.cfg.placement, blend=self.cfg.blend)
            with attach_differentiable_backbone_steer(backbone, steer_vec, spec):
                raw = policy.get_action(obs)
            return _flatten_action_dict(raw)

        # Real GR00T path: GR00T imports are only needed when the kernel
        # actually fires on a real policy.
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        # Preprocess (no autograd needed here; outputs are then used
        # as inputs to the gradient-bearing backbone forward).
        with torch.inference_mode():
            unbatched = policy._unbatch_observation(obs)
            processed_inputs = []
            for o in unbatched:
                vla_step = policy._to_vla_step_data(o)
                messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
                processed_inputs.append(policy.processor(messages))
            collated = policy.collate_fn(processed_inputs)
            collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
            # ``collator.__call__`` wraps the real batch dict under
            # ``"inputs"``; ``_get_action`` then calls
            # ``model.get_action(**collated)`` which expands the wrapper.
            # We need the inner dict for ``model.prepare_input``.
            inner = collated["inputs"] if "inputs" in collated else collated

        # Inference tensors can't escape inference_mode; clone the entries
        # so the autograd-enabled forward sees regular tensors. Non-tensor
        # values (action_input dicts, attention masks, etc.) flow through.
        def _materialize(node):
            if isinstance(node, torch.Tensor):
                return node.detach().clone()
            if isinstance(node, dict):
                return {k: _materialize(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_materialize(v) for v in node]
            return node

        inner = _materialize(dict(inner))

        spec = SteerSpec(placement=self.cfg.placement, blend=self.cfg.blend)
        with torch.enable_grad():
            with attach_differentiable_backbone_steer(backbone, steer_vec, spec):
                backbone_inputs, action_inputs = model.prepare_input(inner)
                backbone_outputs = backbone(backbone_inputs)
                # Bypass the ``@torch.no_grad`` on
                # ``Gr00tN1d7ActionHead.get_action_with_features`` by
                # calling the underlying ``__wrapped__`` function
                # (functools.wraps puts the original there). This preserves
                # autograd from ``steer_vec`` -> ``action_pred`` through
                # the diffusion denoising loop.
                features = model.action_head._encode_features(
                    backbone_outputs, action_inputs,
                )
                head_cls = type(model.action_head)
                raw_get_action = getattr(
                    head_cls.get_action_with_features, "__wrapped__",
                    head_cls.get_action_with_features,
                )
                action_outputs = raw_get_action(
                    model.action_head,
                    backbone_features=features.backbone_features,
                    state_features=features.state_features,
                    embodiment_id=action_inputs.embodiment_id,
                    backbone_output=backbone_outputs,
                    action_input=action_inputs,
                    options=None,
                )

        action_pred = action_outputs["action_pred"]
        # action_pred is [B, horizon, action_dim]. Flatten to a single
        # action vector per batch for MSE; we treat the whole future
        # action chunk as the consistency target.
        return {"action_pred": action_pred.reshape(-1)}

    # -- core entrypoint --------------------------------------------------

    def consistency_loss(
        self,
        *,
        descriptions: Sequence[str],
        example_ids: Sequence[str],
        position_types: Sequence[str],
        step_indices: Sequence[int | None] | None = None,
        instructions: Sequence[str | None] | None = None,
    ) -> tuple[torch.Tensor, ActionConsistencyDiagnostics]:
        """Compute the consistency loss on a slice of the SFT batch.

        Returns a scalar loss tensor (zero if no admissible rows) plus a
        diagnostics struct for logging.
        """
        diag = ActionConsistencyDiagnostics()
        if self.cfg.weight <= 0.0:
            return torch.zeros((), device=self.device), diag

        chosen = self.select_rows(example_ids, position_types)
        if not chosen:
            return torch.zeros((), device=self.device), diag
        self.ensure_loaded()

        total: torch.Tensor = torch.zeros((), device=self.device)
        per_key_max: dict[str, float] = {}
        for idx in chosen:
            entry = self.manifest.get(example_ids[idx])
            if entry is None:
                continue
            # Live AR forward (caption -> α-scaled vector -> unscaled ĥ).
            ar_pred_scaled = self.ar(
                [descriptions[idx]],
                device=self.device,
                position_types=[position_types[idx]],
                step_indices=(
                    [step_indices[idx]] if step_indices is not None else None
                ),
                instructions=(
                    [instructions[idx]] if instructions is not None else None
                ),
            )
            # ar_pred_scaled is (1, H) for scalar head, (1, K, H) for spatial.
            # The differentiable backbone-steer hook only takes a 1-D vector,
            # so when the spatial head emits per-position predictions we
            # mean-pool them into a single representative vector for the
            # policy-effect injection. The per-position MSE on the AR loss
            # still trains the spatial structure; this hook only needs
            # "what scalar would the policy see at the injected slot."
            steer_vec = (ar_pred_scaled.squeeze(0) * float(self.ar.cfg.alpha))
            if steer_vec.dim() == 2:
                steer_vec = steer_vec.mean(dim=0)
            baseline, hit = self.get_baseline(entry)
            diag.baseline_cache_hits += int(hit)
            diag.baseline_cache_misses += int(not hit)
            steered = self._steered_action(entry, steer_vec)
            base_t = _as_action_tensor(baseline, device=self.device).detach()
            steer_t = _as_action_tensor(steered, device=self.device)
            if base_t.numel() == 0 or base_t.shape != steer_t.shape:
                # Shape drift between calls usually means the policy returned
                # different action keys (e.g. dropout on a head). We log and
                # skip rather than corrupting the gradient.
                logger.warning(
                    "[action_consistency] skipping %s: shape mismatch base=%s "
                    "steer=%s", entry.example_id, tuple(base_t.shape),
                    tuple(steer_t.shape),
                )
                continue
            row_loss = F.mse_loss(steer_t, base_t)
            total = total + row_loss
            diag.n_rows += 1
            # Per-key Δaction max-abs for telemetry. Coerce both sides via
            # torch.as_tensor since steered values come back as numpy on the
            # real GR00T action heads.
            for k in sorted(set(baseline) & set(steered)):
                a = torch.as_tensor(baseline[k]).detach().to("cpu", dtype=torch.float32)
                b = torch.as_tensor(steered[k]).detach().to("cpu", dtype=torch.float32)
                if a.shape != b.shape:
                    continue
                v = float((b - a).abs().max().item())
                per_key_max[k] = max(per_key_max.get(k, 0.0), v)

        if diag.n_rows == 0:
            return torch.zeros((), device=self.device), diag

        loss = total / float(diag.n_rows)
        diag.loss = float(loss.detach().item())
        diag.delta_action_norm = float(loss.detach().sqrt().item())
        diag.per_key_delta_max_abs = per_key_max
        return loss, diag


# ---------------------------------------------------------------------------
# Real LeRobot-backed obs_builder factory (used in production via
# ``_build_action_consistency_kernel`` in sft.py). Imports gr00t lazily so
# tests that exercise FakePolicy don't need the GR00T install.
# ---------------------------------------------------------------------------

def make_lerobot_obs_builder(
    policy: PolicyLike,
    dataset_roots_by_suite: Mapping[str, str | Path],
    embodiment_tag: Any,
    *,
    traj_cache_size: int = 32,
    video_backend: str = "torchcodec",
) -> ObsBuilder:
    """Return an ``ObsBuilder`` that replays the original LeRobot observation.

    Parameters
    ----------
    policy :
        A loaded ``Gr00tPolicy`` (or compatible). The builder reads
        ``policy.modality_configs`` to construct LeRobot loaders and to
        identify the language modality keys.
    dataset_roots_by_suite :
        Mapping ``{suite_name: dataset_root}``. The ``suite`` of each
        ``ReplayEntry`` is looked up here; missing suites raise a
        ``KeyError`` on the first row of that suite (failure is loud, not
        silent). Use the suite map you constructed for ``build_replay_manifest``.
    embodiment_tag :
        Either the GR00T enum member or a string (``"LIBERO_PANDA"``,
        ``"libero_sim"``, etc.). Resolved through ``EmbodimentTag.resolve``.
    traj_cache_size :
        Number of distinct trajectories held in an in-memory LRU cache.
        Each LeRobot trajectory load opens a parquet file; SFT batches
        randomise rows, so caching the recently-touched trajectories keeps
        per-row obs build under a few ms after warmup.
    video_backend :
        Forwarded to ``LeRobotEpisodeLoader``. ``torchcodec`` matches the
        eval scripts (``scripts/eval/nla_steer_groot_action.py``).
    """
    # Lazy gr00t imports — only required when the factory is invoked, which
    # happens inside ``ActionConsistencyKernel.ensure_loaded()`` and never in
    # tests.
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    from nla.steering.groot_obs import build_observation_for_step

    if isinstance(embodiment_tag, str):
        et = EmbodimentTag.resolve(embodiment_tag)
    else:
        et = embodiment_tag

    # Mirror the extraction / eval setup: pop the action modality so we only
    # parse observation channels.
    modality_configs = deepcopy(policy.modality_configs)
    modality_configs.pop("action", None)
    language_keys = list(policy.modality_configs["language"].modality_keys)

    loaders: dict[str, Any] = {}
    traj_cache: "OrderedDict[tuple[str, int], Any]" = OrderedDict()

    def _get_loader(suite: str) -> Any:
        if suite not in loaders:
            if suite not in dataset_roots_by_suite:
                raise KeyError(
                    f"make_lerobot_obs_builder: no dataset_root for suite "
                    f"{suite!r}; known suites: {sorted(dataset_roots_by_suite)}"
                )
            loaders[suite] = LeRobotEpisodeLoader(
                dataset_path=str(dataset_roots_by_suite[suite]),
                modality_configs=policy.modality_configs,
                video_backend=video_backend,
            )
        return loaders[suite]

    def _get_traj(suite: str, traj_idx: int) -> Any:
        key = (suite, traj_idx)
        if key in traj_cache:
            traj_cache.move_to_end(key)
            return traj_cache[key]
        traj = _get_loader(suite)[traj_idx]
        traj_cache[key] = traj
        if len(traj_cache) > traj_cache_size:
            traj_cache.popitem(last=False)
        return traj

    def _build(entry: ReplayEntry) -> dict[str, Any]:
        if entry.suite is None:
            raise ValueError(
                "make_lerobot_obs_builder: ReplayEntry suite is None; the "
                "LeRobot factory needs a per-suite dataset_root mapping."
            )
        traj = _get_traj(entry.suite, entry.traj_idx)
        return build_observation_for_step(
            traj,
            entry.step_idx,
            modality_configs,
            et,
            language_keys,
            extract_step_data,
        )

    return _build


# ---------------------------------------------------------------------------
# Fake policy for tests (no GR00T dependency).
# ---------------------------------------------------------------------------

class _FakeBackbone(torch.nn.Module):
    """Trivial module so a forward hook can be registered. Forward returns a
    pre-built ``BatchFeature``-like dict echoing whatever was set up by the
    enclosing FakePolicy on each step."""

    def __init__(self, batch_feature_provider: Callable[[], dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self._provider = batch_feature_provider

    def forward(self) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self._provider()


class _FakeModel(torch.nn.Module):
    def __init__(self, backbone: _FakeBackbone) -> None:
        super().__init__()
        self.backbone = backbone


class FakePolicy:
    """Trivial in-process policy used only by tests.

    Simulates ``Gr00tPolicy``'s interface: exposes ``model.backbone`` so a
    forward hook attaches, and provides ``get_action`` that calls the hooked
    backbone forward, then returns an action computed by a tiny linear head
    on the (possibly steered) backbone features.
    """

    def __init__(
        self,
        *,
        hidden: int = BACKBONE_EMBEDDING_DIM,
        action_dim: int = 4,
        seq_len: int = 4,
        seed: int = 0,
    ) -> None:
        gen = torch.Generator().manual_seed(int(seed))
        self.hidden = int(hidden)
        self.seq_len = int(seq_len)
        self.action_dim = int(action_dim)
        # Default batch_feature uses zeros for backbone_features; the hook
        # rewrites them on steered calls. Image mask is all True.
        self._zeros_feats = torch.zeros(1, self.seq_len, self.hidden)
        self._attn = torch.ones(1, self.seq_len, dtype=torch.bool)
        self._img = torch.ones(1, self.seq_len, dtype=torch.bool)

        def _provider() -> dict[str, torch.Tensor]:
            return {
                "backbone_features": self._zeros_feats.clone(),
                "backbone_attention_mask": self._attn.clone(),
                "image_mask": self._img.clone(),
            }

        backbone = _FakeBackbone(_provider)
        self.model = _FakeModel(backbone)
        # Tiny deterministic action head.
        self._W = torch.empty(self.action_dim, self.hidden).normal_(generator=gen) * 0.1

    def get_action(self, observation: Any) -> dict[str, torch.Tensor]:
        del observation
        # Call the backbone manually so the forward hook fires.
        out = self.model.backbone()
        feats = out["backbone_features"]  # [1, T, H]
        # Reduce over image tokens (matches the spirit of action heads that
        # cross-attend to image features) and project to action space.
        img_mask = out["image_mask"][0].to(torch.float32)
        weighted = feats[0] * img_mask.unsqueeze(-1)
        pooled = weighted.sum(dim=0) / img_mask.sum().clamp(min=1.0)
        action = pooled @ self._W.to(pooled.dtype).T
        return {"action.world_vector": action}


def make_dummy_obs_builder() -> ObsBuilder:
    """Trivial obs builder used in tests; FakePolicy ignores the obs anyway."""

    def _build(entry: ReplayEntry) -> dict[str, Any]:
        del entry
        return {}

    return _build


__all__ = [
    "ActionConsistencyConfig",
    "ActionConsistencyDiagnostics",
    "ActionConsistencyKernel",
    "DifferentiableBackboneSteerHook",
    "FakePolicy",
    "ObsBuilder",
    "ObsBuilderFactory",
    "attach_differentiable_backbone_steer",
    "make_dummy_obs_builder",
    "make_lerobot_obs_builder",
]

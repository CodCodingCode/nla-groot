"""GR00T policy wrapper that applies an NLA backbone steer on every ``get_action``.

The wrapper is meant to live **inside the GPU policy server process** so any
Isaac-GR00T sim client (LIBERO, SimplerEnv, …) can drive a steered checkpoint
without code changes — the steering happens transparently behind the ZMQ
``get_action`` endpoint exposed by ``gr00t.policy.server_client.PolicyServer``.

Usage (typical, inside a server launcher)::

    inner = Gr00tPolicy(model_path=..., embodiment_tag=..., device="cuda")
    ar = load_ar_from_sft(ar_dir, device="cuda", freeze=True)
    steer_vec = ar_text_to_backbone_vec(ar, steer_text)
    spec = SteerSpec(placement="image_patch", blend=1.0)
    policy = NlaSteerGr00tPolicy(inner, steer_vec=steer_vec, spec=spec)
    PolicyServer(policy=policy, host="0.0.0.0", port=5555).run()

The wrapper accepts either a plain ``Gr00tPolicy`` or a ``Gr00tSimPolicyWrapper``
— it walks the ``.policy`` chain to find ``model.backbone`` and registers a
forward hook around the inner ``_get_action`` call. Toggle ``enabled=False`` (or
call :meth:`set_enabled`) for an A/B passthrough without restarting the server.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gr00t.policy.policy import BasePolicy, PolicyWrapper

from nla.eval.steerability.obs_batching import infer_nested_batch_size
from nla.steering.backbone_steer import (
    SteerSpec,
    attach_backbone_steer,
    attach_backbone_steer_batched,
)

# Keys we look for in the per-call ``options`` dict. The bytes variants exist
# because msgpack_numpy decodes string keys as bytes when ``raw=True``; we use
# ``raw=False`` in MsgSerializer.from_bytes but still defend against both forms.
_STEER_H_KEYS = ("steer_h", b"steer_h")
_STEER_SPEC_KEYS = ("steer_spec", b"steer_spec")
_STEER_DISABLE_KEYS = ("steer_disabled", b"steer_disabled")


def _resolve_inner_backbone(policy: BasePolicy) -> torch.nn.Module:
    """Walk a (possibly wrapped) policy until we find ``model.backbone``."""
    seen: list[BasePolicy] = []
    current: Any = policy
    while current is not None and current not in seen:
        seen.append(current)
        model = getattr(current, "model", None)
        if model is not None and hasattr(model, "backbone"):
            backbone = model.backbone
            if not isinstance(backbone, torch.nn.Module):
                raise TypeError(
                    f"Resolved backbone is not an nn.Module (got {type(backbone)!r})."
                )
            return backbone
        current = getattr(current, "policy", None)
    raise AttributeError(
        "Could not find `.model.backbone` on the wrapped policy chain. "
        "NlaSteerGr00tPolicy expects a Gr00tPolicy (or wrapper around one)."
    )


class NlaSteerGr00tPolicy(PolicyWrapper):
    """Policy wrapper that hooks ``backbone_features`` with a fixed AR vector.

    Validation, modality config, and reset are delegated to the wrapped policy
    so the server-facing contract is identical to the inner policy. Only the
    inference call is wrapped so the steer hook is active solely while the
    backbone forward runs.
    """

    def __init__(
        self,
        policy: BasePolicy,
        *,
        steer_vec: torch.Tensor,
        spec: SteerSpec,
        enabled: bool = True,
        strict: bool = True,
        batch_index: int = 0,
    ) -> None:
        super().__init__(policy, strict=strict)
        self._backbone = _resolve_inner_backbone(policy)
        if steer_vec.dim() == 2 and steer_vec.shape[0] == 1:
            steer_vec = steer_vec.squeeze(0)
        self._steer_vec = steer_vec.detach().float().cpu().contiguous()
        self._spec = spec
        self._enabled = bool(enabled)
        self._batch_index = int(batch_index)

    @property
    def steer_spec(self) -> SteerSpec:
        return self._spec

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        """Toggle the steer hook for cheap A/B without rebuilding the server."""
        self._enabled = bool(value)

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def get_modality_config(self) -> Any:
        return self.policy.get_modality_config()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        steer_vec, spec, disabled = self._resolve_per_call(options)
        batch_vecs = self._resolve_steer_batch(options)
        batch_size = infer_nested_batch_size(observation)
        if disabled or (
            not self._enabled and steer_vec is None and batch_vecs is None
        ):
            return self.policy._get_action(observation, options)
        if spec is None:
            spec = self._spec
        if batch_size > 1:
            if batch_vecs is None:
                if steer_vec is None:
                    steer_vec = self._steer_vec
                batch_vecs = [steer_vec] * batch_size
            if len(batch_vecs) != batch_size:
                raise ValueError(
                    f"steer batch length {len(batch_vecs)} != observation batch {batch_size}"
                )
            with attach_backbone_steer_batched(self._backbone, batch_vecs, spec):
                return self.policy._get_action(observation, options)
        if steer_vec is None:
            steer_vec = self._steer_vec
        with attach_backbone_steer(
            self._backbone,
            steer_vec,
            spec,
            batch_index=self._batch_index,
        ):
            return self.policy._get_action(observation, options)

    # ------------------------------------------------------------------ helpers

    def _resolve_per_call(
        self,
        options: dict[str, Any] | None,
    ) -> tuple[torch.Tensor | None, SteerSpec | None, bool]:
        """Pull a request-scoped steer vector / spec / disable flag out of ``options``.

        Returns ``(steer_vec_or_None, spec_or_None, disabled_bool)``. When a
        client passes nothing, both are ``None`` and we fall back to the
        startup vector / spec stored on the wrapper. This is the path the
        GRPO sim-reward worker uses: send one ``steer_h`` per call so a
        single server can score thousands of different intents without restart.
        """
        if not options:
            return None, None, False
        disabled = False
        for k in _STEER_DISABLE_KEYS:
            if k in options:
                disabled = bool(options[k])
                break
        steer_vec: torch.Tensor | None = None
        for k in _STEER_H_KEYS:
            if k in options and options[k] is not None:
                steer_vec = self._coerce_steer_vec(options[k])
                break
        spec: SteerSpec | None = None
        for k in _STEER_SPEC_KEYS:
            if k in options and options[k] is not None:
                spec = self._coerce_spec(options[k])
                break
        return steer_vec, spec, disabled

    def _resolve_steer_batch(
        self,
        options: dict[str, Any] | None,
    ) -> list[torch.Tensor] | None:
        """Return per-row steer vectors from ``options['steer_h_batch']``."""
        if not options:
            return None
        raw = None
        for k in ("steer_h_batch", b"steer_h_batch"):
            if k in options and options[k] is not None:
                raw = options[k]
                break
        if raw is None:
            return None
        if isinstance(raw, torch.Tensor):
            t = raw.detach().float().cpu()
            if t.dim() == 1:
                return [t.contiguous()]
            if t.dim() == 2:
                return [t[i].contiguous() for i in range(t.shape[0])]
            raise ValueError(f"steer_h_batch tensor must be [B,H]; got {tuple(t.shape)}")
        if isinstance(raw, np.ndarray):
            arr = np.asarray(raw, dtype=np.float32)
            if arr.ndim == 1:
                return [torch.from_numpy(arr)]
            if arr.ndim == 2:
                return [torch.from_numpy(arr[i]) for i in range(arr.shape[0])]
            raise ValueError(f"steer_h_batch ndarray must be [B,H]; got {arr.shape}")
        if isinstance(raw, (list, tuple)):
            return [self._coerce_steer_vec(v) for v in raw]
        raise TypeError(f"steer_h_batch must be tensor/ndarray/list; got {type(raw)!r}")

    @staticmethod
    def _coerce_steer_vec(raw: Any) -> torch.Tensor:
        """Accept numpy float arrays, lists, or torch tensors. Returns CPU float [H]."""
        if isinstance(raw, torch.Tensor):
            t = raw
        elif isinstance(raw, np.ndarray):
            t = torch.from_numpy(raw)
        else:
            t = torch.tensor(np.asarray(raw, dtype=np.float32))
        t = t.detach().float().cpu().contiguous()
        if t.dim() == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() != 1:
            raise ValueError(
                f"options['steer_h'] must be a 1-D vector (or [1, H]); got shape {tuple(t.shape)}"
            )
        return t

    def _coerce_spec(self, raw: Any) -> SteerSpec:
        """Accept either a SteerSpec or a dict (msgpack round-trips dicts).

        Missing fields fall back to the wrapper's startup ``self._spec`` so
        callers can override just ``placement``/``blend`` without re-supplying
        everything.
        """
        if isinstance(raw, SteerSpec):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(
                f"options['steer_spec'] must be SteerSpec or dict; got {type(raw)!r}"
            )

        def _get(d: dict, name: str, default: Any) -> Any:
            for k in (name, name.encode("utf-8")):
                if k in d:
                    v = d[k]
                    if isinstance(v, bytes):
                        return v.decode("utf-8")
                    return v
            return default

        return SteerSpec(
            placement=_get(raw, "placement", self._spec.placement),
            blend=float(_get(raw, "blend", self._spec.blend)),
            fixed_token_index=_get(raw, "fixed_token_index", self._spec.fixed_token_index),
            image_patch_seed=int(_get(raw, "image_patch_seed", self._spec.image_patch_seed)),
        )

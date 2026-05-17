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

import torch

from gr00t.policy.policy import BasePolicy, PolicyWrapper

from nla.steering.backbone_steer import SteerSpec, attach_backbone_steer


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
        if not self._enabled:
            return self.policy._get_action(observation, options)
        with attach_backbone_steer(
            self._backbone,
            self._steer_vec,
            self._spec,
            batch_index=self._batch_index,
        ):
            return self.policy._get_action(observation, options)

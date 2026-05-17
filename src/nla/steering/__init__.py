"""NLA steering: patch GR00T backbone features with AR(text) vectors."""

from typing import TYPE_CHECKING, Any

from nla.steering.action_delta import action_stats, policy_get_action, to_numpy
from nla.steering.backbone_steer import (
    BackboneFeaturesSteerHook,
    SteerPlacement,
    SteerSpec,
    attach_backbone_steer,
    resolve_steer_indices,
)
from nla.steering.groot_obs import build_observation_for_step, parse_observation_flat
from nla.steering.nla_vec import ar_text_to_backbone_vec

if TYPE_CHECKING:
    from nla.steering.sim_policy_wrapper import NlaSteerGr00tPolicy

__all__ = [
    "BackboneFeaturesSteerHook",
    "NlaSteerGr00tPolicy",
    "SteerPlacement",
    "SteerSpec",
    "action_stats",
    "ar_text_to_backbone_vec",
    "attach_backbone_steer",
    "build_observation_for_step",
    "parse_observation_flat",
    "policy_get_action",
    "resolve_steer_indices",
    "to_numpy",
]


def __getattr__(name: str) -> Any:
    # Lazy: NlaSteerGr00tPolicy needs `gr00t` installed (sim policy server side
    # only). Avoid importing it eagerly so the toy smoke test in
    # ``scripts/eval/nla_steer_ar_smoke.py`` keeps working without GR00T.
    if name == "NlaSteerGr00tPolicy":
        from nla.steering.sim_policy_wrapper import NlaSteerGr00tPolicy

        return NlaSteerGr00tPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

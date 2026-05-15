"""NLA steering: patch GR00T backbone features with AR(text) vectors."""

from nla.steering.backbone_steer import (
    BackboneFeaturesSteerHook,
    SteerPlacement,
    SteerSpec,
    attach_backbone_steer,
    resolve_steer_indices,
)
from nla.steering.groot_obs import build_observation_for_step, parse_observation_flat
from nla.steering.nla_vec import ar_text_to_backbone_vec

__all__ = [
    "BackboneFeaturesSteerHook",
    "SteerPlacement",
    "SteerSpec",
    "attach_backbone_steer",
    "ar_text_to_backbone_vec",
    "build_observation_for_step",
    "parse_observation_flat",
    "resolve_steer_indices",
]

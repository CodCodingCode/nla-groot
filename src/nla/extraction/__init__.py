"""Activation extraction for GR00T NLA.

See ``layer_spec`` for the canonical hook target. The public surface is::

    BackboneFeatureHook, attach_hooks            # hook.py
    CapturedActivation                            # hook.py
    ActivationShardWriter, ActivationShardReader  # storage.py
    ExampleRecord, RunManifest                    # storage.py
    sample_position, iter_image_positions         # sampler.py
    PositionType, SampledPosition                 # sampler.py
    compute_stats, save_stats, load_stats         # stats.py
    ActivationStats                               # stats.py
"""

from nla.extraction.hook import (
    BackboneFeatureHook,
    CapturedActivation,
    attach_hooks,
)
from nla.extraction.sampler import (
    PositionType,
    SampledPosition,
    iter_image_positions,
    sample_position,
)
from nla.extraction.stats import (
    ActivationStats,
    alpha_from_norms,
    compute_stats,
    load_stats,
    save_stats,
)
from nla.extraction.storage import (
    ActivationShardReader,
    ActivationShardWriter,
    ExampleRecord,
    RunManifest,
    iter_records,
)

__all__ = [
    "BackboneFeatureHook",
    "CapturedActivation",
    "attach_hooks",
    "PositionType",
    "SampledPosition",
    "sample_position",
    "iter_image_positions",
    "ActivationStats",
    "alpha_from_norms",
    "compute_stats",
    "save_stats",
    "load_stats",
    "ActivationShardReader",
    "ActivationShardWriter",
    "ExampleRecord",
    "RunManifest",
    "iter_records",
]

"""Steerability evaluation harness.

Public surface:

- :func:`load_config` reads a YAML config into :class:`SteerabilityConfig`.
- :func:`run_one_rollout` executes a single (condition, env, seed) and writes
  ``trajectory.parquet`` + ``rollout.mp4`` + ``summary.json``.
- :func:`aggregate_metrics` summarises a directory of summaries.
- :func:`render_report` builds ``report.md`` + ``report.html`` + figures.

The driver script ``scripts/eval/steerability_eval.py`` orchestrates them.
"""
from nla.eval.steerability.config import (
    SteerabilityConfig,
    ConditionConfig,
    SteerCfg,
    AvFidelityConfig,
    AvJudgeDatasetConfig,
    load_config,
)

__all__ = [
    "SteerabilityConfig",
    "ConditionConfig",
    "SteerCfg",
    "AvFidelityConfig",
    "AvJudgeDatasetConfig",
    "load_config",
]

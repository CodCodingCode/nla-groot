"""Tests for strict rollout JSON serialization."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nla.eval.steerability.json_utils import dumps_rollout_json, extract_rollout_json


def test_extract_rollout_json_skips_libero_info_lines():
    stdout = (
        "[info] using task orders [0, 1, 2, 3]\n"
        '[{"r_sim": 1.5, "n_steps": 3}]\n'
    )
    parsed = extract_rollout_json(stdout, expect_array=True)
    assert parsed[0]["r_sim"] == 1.5


def test_dumps_replaces_nan_with_null():
    raw = [{"r_sim": float("nan"), "ok": np.float32(1.0)}]
    text = dumps_rollout_json(raw)
    parsed = json.loads(text)
    assert parsed[0]["r_sim"] is None
    assert parsed[0]["ok"] == 1.0


def test_extract_rollout_json_handles_pretty_printed_nested_object():
    """Regression: rfind('{') used to land on an inner brace of the nested
    ``sim_score_breakdown`` dict and json.loads choked with
    ``Extra data: line N column 1`` because the outer closer was left over.
    The fixed extractor must return the OUTER object intact.
    """
    obj = {
        "r_sim": 2.1,
        "n_steps": 42,
        "sim_score_breakdown": {
            "predicate": 1.0,
            "r_dist": 0.7,
            "r_displace": 0.4,
            "r_contact": 0.2,
        },
        "success_any": False,
    }
    stdout = dumps_rollout_json(obj)  # pretty-printed by default
    parsed = extract_rollout_json(stdout, expect_array=False)
    assert parsed == obj


def test_extract_rollout_json_object_with_leading_log_lines():
    obj = {"r_sim": 0.5, "nested": {"a": {"b": 1}}}
    stdout = (
        "[info] something happened\n"
        "Some unrelated stderr-like message\n"
        + dumps_rollout_json(obj)
    )
    parsed = extract_rollout_json(stdout, expect_array=False)
    assert parsed == obj


def test_extract_rollout_json_pretty_array():
    arr = [{"r_sim": 1.0}, {"r_sim": 2.5, "nested": {"x": 1}}]
    stdout = (
        "[info] using task orders [0, 1]\n"
        + dumps_rollout_json(arr)
    )
    parsed = extract_rollout_json(stdout, expect_array=True)
    assert parsed == arr


def test_extract_rollout_json_empty_stdout_raises():
    with pytest.raises(ValueError):
        extract_rollout_json("", expect_array=False)


def test_extract_rollout_json_no_object_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_rollout_json("only stderr-like text\nno braces here", expect_array=False)

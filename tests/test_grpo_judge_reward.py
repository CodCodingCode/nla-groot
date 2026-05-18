"""Unit tests for the optional multimodal-judge reward (Workstream B).

These tests never hit the real OpenAI API: the grader is replaced via a
``grade_fn`` injection seam in ``_compute_judge_rewards`` so we can
deterministically validate cache hit/miss + verdict math.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from nla.labeling.grader import DEFAULT_GRADER_MODEL
from nla.training.grpo import (
    GRPOConfig,
    _blend_rewards,
    _compute_judge_rewards,
    _judge_cache_key,
    _load_judge_cache,
    _serialize_config,
    _validate_judge_config,
    _verdicts_to_scalar,
)


# ---------------------------------------------------------------------------
# Tiny stand-ins for nla.labeling.grader.AxisGrade / GradeResult
# ---------------------------------------------------------------------------

@dataclass
class _StubAxis:
    verdict: str
    reason: str = ""
    passed: bool = True


@dataclass
class _StubGradeResult:
    grounding: _StubAxis | None
    appropriateness: _StubAxis | None


def _make_stub_grader(verdicts: list[tuple[str | None, str | None]]):
    """Return a grade_fn that yields the given (B, C) verdicts in order.

    Records call count + the inputs it saw on the returned closure for
    assertions in tests.
    """
    state = {"calls": 0, "seen_inputs": []}

    def grade_fn(inputs, *, model, concurrency):
        state["calls"] += 1
        state["seen_inputs"].extend(inputs)
        out: list[_StubGradeResult] = []
        for i in range(len(inputs)):
            b, c = verdicts[i] if i < len(verdicts) else (None, None)
            out.append(_StubGradeResult(
                grounding=_StubAxis(b) if b is not None else None,
                appropriateness=_StubAxis(c) if c is not None else None,
            ))
        return out

    grade_fn.state = state  # type: ignore[attr-defined]
    return grade_fn


# Camera keys used by the synthetic fixtures below. The judge code now accepts
# any tokens via ``video_keys=``; we use LIBERO-style names to mirror what V3
# runs will see, but the same logic works with any caller-supplied keys.
_TEST_VIDEO_KEYS = ["image", "wrist_image"]


def _touch_frames(frames_dir: Path, source_id: str) -> None:
    """Create one frame file per ``_TEST_VIDEO_KEYS`` entry for ``source_id``."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for key in _TEST_VIDEO_KEYS:
        (frames_dir / f"{source_id}__{key}.jpg").write_bytes(b"fake")


# ---------------------------------------------------------------------------
# 1. weight=0 -> reward computation is byte-identical to baseline
# ---------------------------------------------------------------------------

def test_judge_reward_zero_is_byte_identical():
    r_recon = torch.tensor([-0.005, -0.003, -0.007, -0.001])
    r_judge = torch.tensor([1.5, -1.5, 0.5, -0.5])
    out = _blend_rewards(r_recon, r_judge, weight=0.0)
    assert torch.equal(out, r_recon), out

    # And that the saved-config view hides the new fields entirely.
    cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
    serialized = _serialize_config(cfg)
    for k in ("judge_reward_weight", "judge_concurrency", "judge_model",
              "judge_cache_path", "frames_cache", "judge_video_keys"):
        assert k not in serialized, f"{k} leaked into config.json layout"


# ---------------------------------------------------------------------------
# 2. blend math at w=0.5 matches the closed-form expectation
# ---------------------------------------------------------------------------

def test_blend_math():
    r_recon = torch.tensor([1.0, 2.0, 3.0, 4.0])
    r_judge = torch.tensor([1.5, -1.5, 0.5, -0.5])
    w = 0.5
    expected_norm = (r_recon - r_recon.mean()) / r_recon.std().clamp_min(1e-6)
    expected = (1.0 - w) * expected_norm + w * r_judge
    out = _blend_rewards(r_recon, r_judge, weight=w)
    assert torch.allclose(out, expected, atol=1e-6), (out, expected)


def test_verdicts_to_scalar():
    assert _verdicts_to_scalar("specific", "appropriate") == 1.5
    assert _verdicts_to_scalar("specific", "inappropriate") == 0.5
    assert _verdicts_to_scalar("generic", "appropriate") == -0.5
    assert _verdicts_to_scalar("generic", "inappropriate") == -1.5
    assert _verdicts_to_scalar(None, "appropriate") == 0.0
    assert _verdicts_to_scalar("specific", None) == 0.0


# ---------------------------------------------------------------------------
# 3. cache hit skips the grader entirely
# ---------------------------------------------------------------------------

def test_cache_hit_skips_grader(tmp_path):
    frames = tmp_path / "frames"
    _touch_frames(frames, "src1")
    cache_path = tmp_path / "cache.jsonl"
    text = "a small red cube on the table"
    key = _judge_cache_key("src1", text, grader_model=DEFAULT_GRADER_MODEL)
    cache = {key: {"key": key, "r_judge": 1.5,
                   "verdict_b": "specific", "verdict_c": "appropriate"}}

    grade_fn = _make_stub_grader([])
    rewards = _compute_judge_rewards(
        rollout_texts=[text],
        source_example_ids=["src1"],
        position_types=["image_patch"],
        frames_cache=str(frames),
        video_keys=_TEST_VIDEO_KEYS,
        judge_cache=cache,
        judge_cache_path=str(cache_path),
        grade_fn=grade_fn,
    )
    assert rewards == [1.5], rewards
    assert grade_fn.state["calls"] == 0  # type: ignore[attr-defined]
    assert not cache_path.exists() or cache_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# 4. cache miss writes a new entry to the JSONL file and updates the dict
# ---------------------------------------------------------------------------

def test_cache_miss_writes_entry(tmp_path):
    frames = tmp_path / "frames"
    _touch_frames(frames, "src2")
    cache_path = tmp_path / "cache.jsonl"
    cache: dict[str, dict] = {}
    text = "the gripper hovers over the green block"

    grade_fn = _make_stub_grader([("specific", "appropriate")])
    rewards = _compute_judge_rewards(
        rollout_texts=[text],
        source_example_ids=["src2"],
        position_types=["last_text"],
        frames_cache=str(frames),
        video_keys=_TEST_VIDEO_KEYS,
        judge_cache=cache,
        judge_cache_path=str(cache_path),
        grade_fn=grade_fn,
    )
    assert rewards == [1.5], rewards
    assert grade_fn.state["calls"] == 1  # type: ignore[attr-defined]

    expected_key = _judge_cache_key("src2", text, grader_model=DEFAULT_GRADER_MODEL)
    assert expected_key in cache
    assert cache[expected_key]["verdict_b"] == "specific"
    assert cache[expected_key]["verdict_c"] == "appropriate"

    # JSONL was appended.
    rows = [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 1, rows
    assert rows[0]["key"] == expected_key
    assert rows[0]["r_judge"] == 1.5

    # Reading the cache back from disk reproduces the dict.
    reloaded = _load_judge_cache(cache_path)
    assert expected_key in reloaded


# ---------------------------------------------------------------------------
# 5. missing frame -> neutral reward (no grader call) for that rollout only
# ---------------------------------------------------------------------------

def test_missing_frame_yields_neutral(tmp_path):
    frames = tmp_path / "frames"
    _touch_frames(frames, "have_frames")
    # "missing" deliberately not touched
    cache_path = tmp_path / "cache.jsonl"
    cache: dict[str, dict] = {}

    grade_fn = _make_stub_grader([("specific", "appropriate")])
    rewards = _compute_judge_rewards(
        rollout_texts=["text1", "text2"],
        source_example_ids=["have_frames", "missing"],
        position_types=["image_patch", "anchor"],
        frames_cache=str(frames),
        video_keys=_TEST_VIDEO_KEYS,
        judge_cache=cache,
        judge_cache_path=str(cache_path),
        grade_fn=grade_fn,
    )
    # have_frames was scored as +1.5, missing got the neutral 0.0
    assert rewards == [1.5, 0.0], rewards
    # Grader was called exactly once (only for have_frames)
    assert grade_fn.state["calls"] == 1  # type: ignore[attr-defined]
    assert len(grade_fn.state["seen_inputs"]) == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 6. CLI / config-level validation when judge enabled without frames_cache
# ---------------------------------------------------------------------------

def test_run_grpo_validates_judge_args(monkeypatch):
    # frames_cache missing -> ValueError
    cfg = GRPOConfig(
        sft_dir="x", activations_root="y", output_dir="z",
        judge_reward_weight=0.5,
    )
    with pytest.raises(ValueError, match="frames_cache|frames-cache"):
        _validate_judge_config(cfg)

    # frames_cache present but judge_video_keys missing -> ValueError
    cfg_no_keys = GRPOConfig(
        sft_dir="x", activations_root="y", output_dir="z",
        judge_reward_weight=0.5,
        frames_cache="/tmp/does/not/matter",
    )
    with pytest.raises(ValueError, match="judge.video.keys|judge_video_keys"):
        _validate_judge_config(cfg_no_keys)

    # frames_cache + judge_video_keys present but no OPENAI_API_KEY -> ValueError
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg2 = GRPOConfig(
        sft_dir="x", activations_root="y", output_dir="z",
        judge_reward_weight=0.5,
        frames_cache="/tmp/does/not/matter",
        judge_video_keys=["image", "wrist_image"],
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _validate_judge_config(cfg2)

    # All present -> no raise.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg3 = GRPOConfig(
        sft_dir="x", activations_root="y", output_dir="z",
        judge_reward_weight=0.5,
        frames_cache="/tmp/does/not/matter",
        judge_video_keys=["image", "wrist_image"],
    )
    _validate_judge_config(cfg3)


# ---------------------------------------------------------------------------
# Sanity: the cache key is stable across repeated calls + sensitive to text
# ---------------------------------------------------------------------------

def test_cache_key_is_deterministic_and_sensitive():
    k1 = _judge_cache_key("src", "hello world")
    k2 = _judge_cache_key("src", "hello world")
    assert k1 == k2
    assert _judge_cache_key("src", "hello world") != _judge_cache_key("src", "hello world!")
    assert _judge_cache_key("a", "x") != _judge_cache_key("b", "x")
    assert _judge_cache_key("s", "t", grader_model="m1") != _judge_cache_key("s", "t", grader_model="m2")
    assert _judge_cache_key("s", "t", grader_model="") == _judge_cache_key("s", "t")

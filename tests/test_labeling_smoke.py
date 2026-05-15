"""Smoke tests for the warm-start labeling stack.

Covers prompt rendering, text-context decoding, position-sampling integration
with the extraction reader, frame extraction from a synthetic MP4, and the
end-to-end async label runner (with a fake OpenAI client).
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import av
import numpy as np
import pytest
import torch

from nla.extraction import (
    ActivationShardReader,
    ActivationShardWriter,
    RunManifest,
)
from nla.labeling.context import (
    _draw_positions_for_example,
    decode_text_context,
    image_patch_meta,
    sample_one_position_per_example,
    sample_positions_per_example,
)
from nla.labeling.frames import DatasetInfo, EpisodeFrameLoader, save_jpeg
from nla.labeling.openai_client import _build_messages, label_many_async
from nla.labeling.prompts import (
    LabelInput,
    PositionLabelInput,
    build_position_prompt,
    build_step_prompt,
    build_strict_position_prompt,
)
from nla.layer_spec import BACKBONE_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def test_position_prompt_includes_position_clause():
    inp = PositionLabelInput(
        example_id="ex0@p042_image_patch",
        instruction="Pick up the red cube and place it in the bowl.",
        decoded_text_context="<system> stuff <image: 248 patches> <user> task </user>",
        position_index=42,
        position_type="image_patch",
        sequence_length=277,
        image_paths=["/tmp/fake.jpg"],
        image_patch_meta=(42, 248),
    )
    sys_p, user_p = build_position_prompt(inp)
    assert "GR00T" in sys_p
    assert "4-5" in sys_p and "bullets" in sys_p
    assert "position 42" in user_p
    assert "out of 277" in user_p
    assert "IMAGE-PATCH" in user_p
    assert "image patch 42 of 248" in user_p
    assert "Pick up the red cube" in user_p


def test_position_prompt_handles_missing_instruction():
    inp = PositionLabelInput(
        example_id="ex0@p0_anchor",
        instruction="",
        decoded_text_context="<text>",
        position_index=0,
        position_type="anchor",
        sequence_length=1,
        image_paths=[],
    )
    _, user_p = build_position_prompt(inp)
    assert "(no instruction provided)" in user_p
    assert "ANCHOR" in user_p


def test_position_prompt_image_patch_forbids_index_layout_inference():
    """The per-position prompt must explicitly forbid guessing screen
    quadrant / pixel coordinates from the (k, n) patch index, and must not
    model that failure mode in its own example bullets.

    Background: see ``docs/sft_plan/01_data_audit.md`` §3.2 (confabulated
    image_region content).  The labeler is shown ``image patch k of n`` as
    metadata but never told *which* patch is k, so any "upper-left" /
    "lower-right" claim derived from k alone is teacher hallucination.
    """
    inp = PositionLabelInput(
        example_id="ex0@p042_image_patch",
        instruction="Pick up the red cube and place it in the bowl.",
        decoded_text_context="<image: 248 patches>",
        position_index=42,
        position_type="image_patch",
        sequence_length=277,
        image_paths=["/tmp/fake.jpg"],
        image_patch_meta=(42, 248),
    )
    sys_p, _ = build_position_prompt(inp)
    assert "Rules for IMAGE-PATCH positions" in sys_p
    assert "Do NOT use it to guess" in sys_p
    assert "(k of n)" in sys_p
    assert "upper-left" in sys_p
    assert "upper-right of the table" not in sys_p


def test_strict_position_prompt_inherits_image_patch_rules():
    """Strict relabel prompt must also carry the anti-hallucination rules
    so re-labeled rows do not regress to confident layout claims."""
    inp = PositionLabelInput(
        example_id="ex_strict@p7_image_patch",
        instruction="Test",
        decoded_text_context="ctx",
        position_index=7,
        position_type="image_patch",
        sequence_length=100,
        image_paths=[],
        image_patch_meta=(7, 256),
    )
    sys_p, _ = build_strict_position_prompt(inp)
    assert "Rules for IMAGE-PATCH positions" in sys_p
    assert "Additional rules (strict):" in sys_p


def test_step_prompt_keeps_backcompat():
    inp = LabelInput(
        example_id="ex_step",
        instruction="Pick up the cup",
        image_path="/tmp/fake.jpg",
        state=[0.1, 0.2, 0.3],
    )
    sys_p, user_p = build_step_prompt(inp)
    assert "Pick up the cup" in user_p
    assert "0.100, 0.200, 0.300" in user_p
    assert "4-5" in sys_p


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def test_build_messages_attaches_all_images(tmp_path: Path):
    f1 = save_jpeg(np.zeros((8, 8, 3), dtype=np.uint8), tmp_path / "a.jpg")
    f2 = save_jpeg(np.full((8, 8, 3), 255, np.uint8), tmp_path / "b.jpg")
    inp = PositionLabelInput(
        example_id="ex0",
        instruction="task",
        decoded_text_context="ctx",
        position_index=3,
        position_type="anchor",
        sequence_length=10,
        image_paths=[str(f1), str(f2)],
    )
    messages, kind, meta = _build_messages(inp)
    assert kind == "position"
    assert meta["position_index"] == 3
    user_content = messages[1]["content"]
    image_blocks = [c for c in user_content if c.get("type") == "image_url"]
    assert len(image_blocks) == 2
    for b in image_blocks:
        assert b["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# Text-context decoding
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Trivial tokenizer that maps ids back via a fixed vocab."""

    def __init__(self) -> None:
        self.vocab = {1: "Hello", 2: " world", 3: ".", 999: "<image>"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab.get(int(i), f"<{int(i)}>") for i in ids)


def test_decode_text_context_collapses_image_runs():
    tok = _FakeTokenizer()
    input_ids = torch.tensor([1, 2, 999, 999, 999, 3])
    image_mask = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.bool)
    out = decode_text_context(input_ids, image_mask, tok, char_budget=10_000)
    assert "Hello world" in out
    assert "<image: 3 patches>" in out
    assert "<image>" not in out  # the raw special token text must not leak


def test_decode_text_context_truncates_long():
    tok = _FakeTokenizer()
    input_ids = torch.tensor([1] * 5000)
    image_mask = torch.zeros(5000, dtype=torch.bool)
    out = decode_text_context(input_ids, image_mask, tok, char_budget=100)
    assert "[elided]" in out
    assert len(out) <= 200


def test_image_patch_meta():
    img = torch.tensor([1, 1, 0, 1, 1, 1, 0], dtype=torch.bool)
    assert image_patch_meta(img, 0) == (0, 5)
    assert image_patch_meta(img, 1) == (1, 5)
    assert image_patch_meta(img, 3) == (2, 5)
    assert image_patch_meta(img, 2) is None  # not an image patch


# ---------------------------------------------------------------------------
# Extraction <-> labeling bridge
# ---------------------------------------------------------------------------

def _write_synthetic_dump(out_root: Path, n: int = 3, T: int = 20) -> ActivationShardReader:
    manifest = RunManifest(
        schema_version=1,
        model_repo="fake",
        layer_module_path="fake",
        hidden_size=BACKBONE_EMBEDDING_DIM,
        activation_dtype="float32",
        embodiment_tag=None,
    )
    writer = ActivationShardWriter(out_root, manifest, max_examples_per_shard=64)
    for i in range(n):
        f = torch.randn(T, BACKBONE_EMBEDDING_DIM)
        attn = torch.ones(T, dtype=torch.bool)
        img = torch.zeros(T, dtype=torch.bool)
        img[: T // 2] = True
        ids = torch.tensor([999] * (T // 2) + [1, 2, 3] + [1] * (T - T // 2 - 3))
        writer.write(
            example_id=f"traj0_step{i:04d}",
            features=f,
            attention_mask=attn,
            image_mask=img,
            input_ids=ids,
            episode_index=0,
            step_index=i,
            task_text="Test task",
            embodiment_tag="FAKE",
        )
    writer.close()
    return ActivationShardReader(out_root)


def test_sample_one_position_per_example_yields_all(tmp_path: Path):
    reader = _write_synthetic_dump(tmp_path / "act", n=5)
    tok = _FakeTokenizer()
    sampled = list(sample_one_position_per_example(reader, tok, seed=0))
    assert len(sampled) == 5
    assert {s.record.example_id for s in sampled} == {f"traj0_step{i:04d}" for i in range(5)}
    for s in sampled:
        # Every sampled position must be within the example's length.
        assert 0 <= s.position_index < s.record.seq_len
        assert s.position_type in ("last_text", "image_patch", "anchor", "fallback")


def test_sample_n_positions_per_example_yields_n_per_example(tmp_path: Path):
    reader = _write_synthetic_dump(tmp_path / "act", n=3, T=40)
    tok = _FakeTokenizer()
    sampled = list(sample_positions_per_example(reader, tok, n_per_example=4, seed=0))
    # 3 examples × 4 positions each = 12 sampled rows.
    assert len(sampled) == 12
    # Positions within one example must be distinct (no-replacement default).
    by_example: dict[str, list[int]] = {}
    for s in sampled:
        by_example.setdefault(s.record.example_id, []).append(s.position_index)
    for ex, positions in by_example.items():
        assert len(positions) == len(set(positions)), f"Duplicate positions for {ex}: {positions}"


def test_draw_positions_default_matches_sample_positions():
    """Without ``guarantee_strata`` the helper must reproduce the legacy
    ``sample_positions`` behavior bit-for-bit so existing label runs are
    unchanged."""
    from nla.extraction.sampler import sample_positions

    T = 16
    attn = torch.ones(T, dtype=torch.bool)
    img = torch.zeros(T, dtype=torch.bool)
    img[:8] = True

    sps_default = _draw_positions_for_example(
        attn, img, n=3, rng=np.random.default_rng(123), guarantee_strata=False,
    )
    sps_baseline = sample_positions(attn, img, n=3, rng=np.random.default_rng(123))
    assert [sp.index for sp in sps_default] == [sp.index for sp in sps_baseline]
    assert [sp.type for sp in sps_default] == [sp.type for sp in sps_baseline]


def test_draw_positions_guarantees_last_text_and_anchor():
    """When ``guarantee_strata`` is set and last_text/anchor are distinct
    indices, both must appear in the returned slots regardless of the
    POSITION_MIX draw."""
    T = 20
    attn = torch.ones(T, dtype=torch.bool)
    img = torch.zeros(T, dtype=torch.bool)
    img[:10] = True
    img[18:] = True

    sps = _draw_positions_for_example(
        attn, img, n=4, rng=np.random.default_rng(0), guarantee_strata=True,
    )
    assert len(sps) == 4
    indices = [sp.index for sp in sps]
    assert len(set(indices)) == 4, f"Expected distinct indices, got {indices}"
    assert 19 in indices  # anchor (final non-pad token, here an image token)
    assert 17 in indices  # last_text (last non-image, non-pad token)
    types = {sp.type.value for sp in sps}
    assert "anchor" in types
    assert "last_text" in types


def test_sample_skips_examples_without_input_ids(tmp_path: Path):
    out_root = tmp_path / "act"
    manifest = RunManifest(
        schema_version=1, model_repo="fake", layer_module_path="fake",
        hidden_size=BACKBONE_EMBEDDING_DIM, activation_dtype="float32", embodiment_tag=None,
    )
    writer = ActivationShardWriter(out_root, manifest)
    writer.write(
        example_id="no_ids",
        features=torch.randn(8, BACKBONE_EMBEDDING_DIM),
        attention_mask=torch.ones(8, dtype=torch.bool),
        image_mask=torch.zeros(8, dtype=torch.bool),
        input_ids=None,
        episode_index=0, step_index=0,
    )
    writer.close()
    reader = ActivationShardReader(out_root)
    tok = _FakeTokenizer()
    sampled = list(sample_one_position_per_example(reader, tok, seed=0))
    assert sampled == []


# ---------------------------------------------------------------------------
# Frame loader on a synthetic mp4
# ---------------------------------------------------------------------------

def _write_synthetic_video(path: Path, n_frames: int = 8, size: int = 32) -> None:
    """Encode a tiny mp4 with deterministic per-frame content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "23"}
        for i in range(n_frames):
            arr = np.full((size, size, 3), (i * 30) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_minimal_lerobot_dataset(root: Path, video_keys=("cam0",), n_frames: int = 8) -> None:
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 5,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))
    # Mimic the real LeRobot export: short key -> 'original_key' on disk.
    modality = {
        "video": {vk: {"original_key": f"observation.images.{vk}"} for vk in video_keys}
    }
    (meta / "modality.json").write_text(json.dumps(modality))
    for vk in video_keys:
        _write_synthetic_video(
            root / f"videos/chunk-000/observation.images.{vk}/episode_000000.mp4",
            n_frames=n_frames,
        )


def test_dataset_info_round_trip(tmp_path: Path):
    _make_minimal_lerobot_dataset(tmp_path / "ds", video_keys=("camA", "camB"))
    di = DatasetInfo.from_root(tmp_path / "ds")
    assert di.fps == 5
    assert set(di.video_keys) == {"camA", "camB"}


def test_episode_frame_loader_returns_correct_shape(tmp_path: Path):
    _make_minimal_lerobot_dataset(tmp_path / "ds", video_keys=("cam0",), n_frames=8)
    with EpisodeFrameLoader(tmp_path / "ds", episode_index=0) as loader:
        f0 = loader.frame("cam0", 0)
        f5 = loader.frame("cam0", 5)
    assert f0.shape == (32, 32, 3) and f0.dtype == np.uint8
    assert f5.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# End-to-end async labeler with a mocked OpenAI client
# ---------------------------------------------------------------------------

def _make_mock_completion(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    usage = MagicMock()
    usage.model_dump.return_value = {
        "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
    }
    resp.usage = usage
    return resp


def test_label_many_async_writes_jsonl_with_mocked_openai(tmp_path: Path, monkeypatch):
    fake_completion = _make_mock_completion(
        "- scene: small table with toys.\n"
        "- target: blue block on the right.\n"
        "- plan: prepare to grasp the blue block.\n"
        "- gripper: open, ready to close.\n"
        "- language: instruction names the blue block."
    )

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = AsyncMock(return_value=fake_completion)

        async def close(self):
            return None

    monkeypatch.setattr(
        "nla.labeling.openai_client._get_openai",
        lambda: (MagicMock(), FakeAsyncClient),
    )

    img_path = save_jpeg(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "img.jpg")
    inputs = [
        PositionLabelInput(
            example_id=f"ex_{i}",
            instruction="Test task",
            decoded_text_context="<image: 4 patches> some text",
            position_index=i,
            position_type="anchor",
            sequence_length=10,
            image_paths=[str(img_path)],
        )
        for i in range(3)
    ]

    out_jsonl = tmp_path / "out" / "labels.jsonl"
    n = asyncio.run(
        label_many_async(
            inputs, out_jsonl, model="fake-model", concurrency=2,
            api_key="fake", resume=False,
        )
    )
    assert n == 3
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    for row in rows:
        assert row["description"].startswith("- scene")
        assert row["error"] is None
        assert row["model"] == "fake-model"
        assert row["kind"] == "position"
        assert row["usage"]["total_tokens"] == 19


def test_label_many_async_resumes(tmp_path: Path, monkeypatch):
    fake_completion = _make_mock_completion("- scene: ok.")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = AsyncMock(return_value=fake_completion)

        async def close(self):
            return None

    monkeypatch.setattr(
        "nla.labeling.openai_client._get_openai",
        lambda: (MagicMock(), FakeAsyncClient),
    )

    img = save_jpeg(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "i.jpg")
    inputs = [
        PositionLabelInput(
            example_id=f"ex_{i}", instruction="t", decoded_text_context="c",
            position_index=0, position_type="anchor", sequence_length=1,
            image_paths=[str(img)],
        )
        for i in range(3)
    ]
    out_jsonl = tmp_path / "labels.jsonl"
    asyncio.run(label_many_async(inputs, out_jsonl, model="fake", api_key="x", resume=True))
    # Second run with one new input; the first three must be skipped.
    inputs.append(
        PositionLabelInput(
            example_id="ex_3", instruction="t", decoded_text_context="c",
            position_index=0, position_type="anchor", sequence_length=1,
            image_paths=[str(img)],
        )
    )
    n_new = asyncio.run(label_many_async(inputs, out_jsonl, model="fake", api_key="x", resume=True))
    assert n_new == 1
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    assert {r["example_id"] for r in rows} == {"ex_0", "ex_1", "ex_2", "ex_3"}

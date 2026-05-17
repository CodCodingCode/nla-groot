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
from nla.labeling.openai_client import (
    _build_messages,
    _select_position_builder,
    label_many_async,
)
from nla.labeling.prompts import (
    LabelInput,
    PositionLabelInput,
    V4_MOTOR_IMPERATIVE_PHRASES,
    V4_PLAN_PHASES,
    V4_SCAFFOLD_FORBIDDEN_PHRASES,
    build_position_prompt,
    build_step_prompt,
    build_strict_position_prompt,
    build_v4_position_prompt,
    infer_suite_from_example_id,
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


def test_position_prompt_hardens_against_anthropomorphic_phrasing():
    """May-2026 prompt hardening: the LIBERO pilot judge audit found that
    ``last_text`` bullets were emitting "instruction has been read" /
    "goal committed" phrasing because the system prompt itself said the
    last bullet describes "what the model is committing to".

    This test pins the new neutral wording: the production prompt must
    (a) not steer with anthropomorphic verbs in its own guidance text,
    (b) include an explicit Forbidden phrasings block listing
    "committing to" and "instruction has been read" as banned, and
    (c) not show "image_region" as the canonical last-bullet category for
    image_patch examples (it should suggest target/scene/spatial instead).
    """
    inp = PositionLabelInput(
        example_id="ex_hard@p9_last_text",
        instruction="put the bowl on the plate",
        decoded_text_context="ctx",
        position_index=9,
        position_type="last_text",
        sequence_length=143,
        image_paths=[],
    )
    sys_p, _ = build_position_prompt(inp)

    assert "Forbidden phrasings" in sys_p
    assert "committing to" in sys_p
    assert "instruction has been read" in sys_p
    assert "goal committed" in sys_p

    guidance_block, _, _ = sys_p.partition("Rules for IMAGE-PATCH positions:")
    assert "what the model is committing to" not in guidance_block
    assert "language: instruction has been read" not in guidance_block
    assert "goal is to grasp the blue cube" not in guidance_block

    image_patch_example_line = next(
        (line for line in guidance_block.splitlines()
         if line.lstrip().startswith("- image_patch:")),
        None,
    )
    assert image_patch_example_line is not None
    assert "image_region:" not in image_patch_example_line

    assert "Avoid the 'image_region' category" in sys_p


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


# ---------------------------------------------------------------------------
# V4 prompt — Phase-1 LIBERO corpus repair (SA1)
# ---------------------------------------------------------------------------

def _v4_inp(position_type, *, position_index=42, image_patch_meta=(42, 248)):
    """Helper: build a PositionLabelInput at a canonical sample position."""
    return PositionLabelInput(
        example_id=f"ex_v4@p{position_index}_{position_type}",
        instruction="put the bowl on the plate",
        decoded_text_context="<image: 248 patches> some text",
        position_index=position_index,
        position_type=position_type,
        sequence_length=277,
        image_paths=["/tmp/fake.jpg"],
        image_patch_meta=image_patch_meta if position_type == "image_patch" else None,
    )


def test_v4_position_prompt_forbids_scaffold_leakage():
    """The V4 system text must explicitly list the prompt-scaffolding
    phrases that V3 labelers regurgitated into captions ("action head",
    "this patch carries", "transformer", ...). The diversity audit flagged
    these as the dominant boilerplate; V4 must call them out by name so the
    labeler cannot reuse them."""
    sys_p, _ = build_v4_position_prompt(_v4_inp("image_patch"))
    assert "Scaffold-leakage ban" in sys_p
    for phrase in ("action head", "this patch carries", "transformer",
                   "embedding", "hidden state", "residual stream",
                   "token carries", "the patch carries", "carries the"):
        assert phrase in sys_p, f"V4 scaffold-leakage ban missing phrase: {phrase!r}"
    assert all(p in sys_p for p in V4_SCAFFOLD_FORBIDDEN_PHRASES)


def test_v4_position_prompt_forbids_motor_imperatives():
    """At least 3 second-person motor-imperative phrasings must be named as
    forbidden in the V4 system text. The Phase-1 plan requires the labeler
    to stop saying things like "grasp the bowl" / "align the gripper" and
    instead describe the scene in third person."""
    sys_p, _ = build_v4_position_prompt(_v4_inp("last_text", position_index=200,
                                                image_patch_meta=None))
    assert "Motor-imperative ban" in sys_p
    listed = [p for p in V4_MOTOR_IMPERATIVE_PHRASES if p in sys_p]
    assert len(listed) >= 3, (
        f"Expected >=3 motor-imperative phrases listed in V4 system text; "
        f"found {len(listed)}: {listed}"
    )
    for required in ("grasp the", "align the gripper", "reach toward"):
        assert required in sys_p, f"Missing required motor-imperative ban: {required!r}"
    assert "descriptive third-person" in sys_p
    assert "pickup" in sys_p and "approach" in sys_p
    for phase in V4_PLAN_PHASES[:5]:
        assert phase in sys_p


def test_v4_position_prompt_position_type_conditioning():
    """The position-type-conditional last-bullet rule must actually differ
    between ``image_patch`` and ``last_text`` system prompts. This is the
    big lever from Agent 4 against high cross-position Jaccard."""
    sys_ip, _ = build_v4_position_prompt(_v4_inp("image_patch"))
    sys_lt, _ = build_v4_position_prompt(_v4_inp("last_text", position_index=200,
                                                 image_patch_meta=None))
    sys_an, _ = build_v4_position_prompt(_v4_inp("anchor", position_index=276,
                                                 image_patch_meta=None))

    assert sys_ip != sys_lt, "image_patch and last_text V4 prompts must differ"
    assert sys_ip != sys_an
    assert sys_lt != sys_an

    assert "Position-conditional last bullet (IMAGE-PATCH)" in sys_ip
    assert "Position-conditional last bullet (LAST-TEXT)" in sys_lt
    assert "Position-conditional last bullet (ANCHOR)" in sys_an

    assert "Position-conditional last bullet (LAST-TEXT)" not in sys_ip
    assert "Position-conditional last bullet (IMAGE-PATCH)" not in sys_lt

    assert "Do NOT restate the task instruction" in sys_ip
    assert "next ~3 timesteps" in sys_lt
    assert "overall trajectory phase" in sys_an

    assert "language" in sys_ip.lower() and "OPTIONAL" in sys_ip


def test_v4_last_bullet_image_patch_vs_last_text_differs():
    """SA4 disambiguation gate: same instruction + same frame must yield
    DIFFERENT position-clause sections in the user prompt AND DIFFERENT
    last-bullet sections in the system prompt for image_patch vs
    last_text. This is the prompt-side guarantee that backs the Jaccard
    delta metric in ``scripts/eval/audit_ptype_disambiguation.py``."""
    # Two inputs identical except for position_type (and metadata that's
    # logically tied to the ptype: image_patch_meta).
    ip = PositionLabelInput(
        example_id="ex_v4@p042_image_patch",
        instruction="put the bowl on the plate",
        decoded_text_context="<image: 248 patches> some text",
        position_index=42,
        position_type="image_patch",
        sequence_length=277,
        image_paths=["/tmp/fake.jpg"],
        image_patch_meta=(42, 248),
    )
    lt = PositionLabelInput(
        example_id="ex_v4@p200_last_text",
        instruction="put the bowl on the plate",
        decoded_text_context="<image: 248 patches> some text",
        position_index=200,
        position_type="last_text",
        sequence_length=277,
        image_paths=["/tmp/fake.jpg"],
        image_patch_meta=None,
    )
    sys_ip, user_ip = build_v4_position_prompt(ip)
    sys_lt, user_lt = build_v4_position_prompt(lt)

    assert user_ip != user_lt, (
        "User prompts must differ between image_patch and last_text "
        "(the position clause embeds the ptype label)."
    )
    assert "IMAGE-PATCH" in user_ip and "LAST TEXT" in user_lt

    assert sys_ip != sys_lt, (
        "System prompts must differ between image_patch and last_text "
        "(the per-ptype last-bullet clause)."
    )

    # Isolate the per-ptype last-bullet clauses so we can test prescriptive
    # language without false positives from the shared V4-LEAK-1 rule.
    ip_clause = sys_ip.split("Position-conditional last bullet (IMAGE-PATCH)")[1]
    lt_clause = sys_lt.split("Position-conditional last bullet (LAST-TEXT)")[1]

    # image_patch clause: perceptual phrasing, restate-instruction ban,
    # canonical 'visible in this frame: <object> <state>' example.
    assert "restate" in ip_clause
    assert "perceptual" in ip_clause
    assert "visible in this frame" in ip_clause
    assert (
        "black wine bottle upright on the wooden tabletop next to the gripper"
        in ip_clause
    )

    # last_text clause: explicit temporal connector, plan-phase list,
    # canonical 'pickup phase over the next 3 timesteps: ...' example, and
    # NO 'perceptual' phrasing in the clause body itself.
    assert "over the next 3 timesteps" in lt_clause
    assert "plan-phase list" in lt_clause
    assert "perceptual" not in lt_clause
    assert (
        "pickup phase over the next 3 timesteps: gripper closes on the "
        "wine bottle, then lifts before placing on the rack"
        in lt_clause
    )


def test_v4_leak_rule_present():
    """The V4-LEAK-1 cross-leak rule must appear in the V4 system prompt
    (i.e. baked into ``_V4_POSITION_SYSTEM`` via ``_V4_EXTRA_RULES``) so
    every V4 row carries the discipline regardless of ptype."""
    inp = PositionLabelInput(
        example_id="ex_leak@p10_anchor",
        instruction="put the bowl on the plate",
        decoded_text_context="ctx",
        position_index=10,
        position_type="anchor",
        sequence_length=11,
        image_paths=[],
    )
    sys_p, _ = build_v4_position_prompt(inp)
    assert "Rule V4-LEAK-1" in sys_p
    assert "Position-type discipline" in sys_p
    assert "image_patch-style perceptual bullets" in sys_p
    assert "last_text-style temporal-plan bullets" in sys_p
    assert "over the next 3 timesteps" in sys_p
    assert "DIFFERENT last bullets for image_patch vs last_text" in sys_p


def test_v4_position_prompt_suite_hook():
    """Unknown suite names must remain a no-op (no error, no extra block).

    Once SA2 registers ``_V4_SUITE_ADDENDA["libero_spatial"]`` (see
    ``test_v4_libero_spatial_addendum_present``) the known-suite branch
    diverges from the default prompt; this test still pins the unknown-suite
    no-op contract so SA5's pipeline can pass arbitrary suite tags safely.
    """
    inp = _v4_inp("image_patch")
    sys_default, _ = build_v4_position_prompt(inp)
    sys_unknown_suite, _ = build_v4_position_prompt(inp, suite="not_a_real_suite")
    assert sys_default == sys_unknown_suite


def test_v4_libero_spatial_addendum_present():
    """When ``suite="libero_spatial"`` is passed, the V4 system prompt must
    surface the SP-1..SP-5 rule block: in-frame relation verification (SP-1),
    explicit frame-of-reference (SP-2), the named confabulation pairs (SP-3),
    visually verifiable landmark requirement (SP-4), and occlusion handling
    (SP-5). This addendum is the lever for fixing the V3 libero_spatial
    B-pass cluster (~73%)."""
    inp = _v4_inp("image_patch")
    sys_p, _ = build_v4_position_prompt(inp, suite="libero_spatial")
    assert "LIBERO-SPATIAL addendum" in sys_p
    assert "Rule SP-1" in sys_p
    assert "visible in the attached camera frame" in sys_p
    assert "Rule SP-2" in sys_p
    assert "frame of reference" in sys_p.lower()
    assert "Rule SP-3" in sys_p
    for pair in ("bowl", "plate", "mug", "shelf", "cube", "tray",
                 "wine bottle", "rack"):
        assert pair in sys_p, f"SP-3 confabulation pair missing: {pair!r}"
    assert "Rule SP-4" in sys_p
    assert "wooden tabletop" in sys_p or "silver gripper" in sys_p
    assert "Rule SP-5" in sys_p
    assert "occluded" in sys_p.lower() or "occlusion" in sys_p.lower()
    # SP-6 / SP-7 added in iteration 1 after observing instruction-anchored
    # color hallucinations on the V3-bad pilot ("pick up the BLACK bowl" ->
    # labeler asserts a black bowl is visible even when bowls are gray).
    assert "Rule SP-6" in sys_p
    assert "instruction" in sys_p.lower()
    assert "visually verify" in sys_p
    assert "Rule SP-7" in sys_p
    assert "metallic" in sys_p


def test_v4_libero_spatial_addendum_absent_for_other_suites():
    """The libero_spatial rule block must NOT appear when another suite is
    selected; the other 3 suites are healthy at V3 and do not need the extra
    constraints."""
    inp = _v4_inp("image_patch")
    # Use addendum-unique markers (the base prompt also says "visible in the
    # attached camera frame", so use SP-N rule headers + the confabulation-pair
    # lexicon as the disambiguator).
    for suite in ("libero_goal", "libero_object", "libero_10"):
        sys_p, _ = build_v4_position_prompt(inp, suite=suite)
        assert "LIBERO-SPATIAL addendum" not in sys_p, (
            f"libero_spatial block leaked into suite={suite!r}"
        )
        for marker in ("Rule SP-1", "Rule SP-2", "Rule SP-3",
                       "Rule SP-4", "Rule SP-5", "Rule SP-6", "Rule SP-7"):
            assert marker not in sys_p, (
                f"{marker} leaked into suite={suite!r}"
            )
        assert "confabulation" not in sys_p.lower()
    sys_none, _ = build_v4_position_prompt(inp)
    assert "LIBERO-SPATIAL addendum" not in sys_none
    assert "Rule SP-1" not in sys_none


def test_v4_suite_auto_inference_from_example_id():
    """If the caller does not pass ``suite=...`` but the input's example_id
    starts with ``libero_spatial::`` (the eval-style prefix), the spatial
    addendum must auto-activate. Also exercise the inference helper directly
    on the public surface."""
    inp = PositionLabelInput(
        example_id="libero_spatial::traj000017_step000014@p151_anchor",
        instruction="pick up the bowl",
        decoded_text_context="ctx",
        position_index=151,
        position_type="anchor",
        sequence_length=200,
        image_paths=["/tmp/fake.jpg"],
    )
    sys_inferred, _ = build_v4_position_prompt(inp)  # no suite= passed
    assert "LIBERO-SPATIAL addendum" in sys_inferred
    assert "Rule SP-1" in sys_inferred

    inp_goal = PositionLabelInput(
        example_id="libero_goal::traj000000_step000000@p0_anchor",
        instruction="t",
        decoded_text_context="c",
        position_index=0,
        position_type="anchor",
        sequence_length=1,
        image_paths=[],
    )
    sys_goal, _ = build_v4_position_prompt(inp_goal)
    assert "LIBERO-SPATIAL addendum" not in sys_goal

    assert infer_suite_from_example_id(
        "libero_spatial::traj1_step2@p3_anchor"
    ) == "libero_spatial"
    assert infer_suite_from_example_id(
        "libero_goal::x"
    ) == "libero_goal"
    assert infer_suite_from_example_id(
        "libero_object::x"
    ) == "libero_object"
    assert infer_suite_from_example_id(
        "libero_10::x"
    ) == "libero_10"
    assert infer_suite_from_example_id("traj1_step2@p3_anchor") is None
    assert infer_suite_from_example_id(None) is None
    assert infer_suite_from_example_id(
        "no_prefix", extra={"suite": "libero_spatial"},
    ) == "libero_spatial"
    assert infer_suite_from_example_id(
        "no_prefix", extra={"suite": "bogus_suite"},
    ) is None


def test_pipeline_dispatches_v4_when_mode_set(monkeypatch):
    """SA5 wiring: ``_select_position_builder`` must return the V4 builder
    when ``NLA_POSITION_PROMPT_MODE=v4`` (or any of the documented aliases),
    fall back to the strict builder for ``strict``/``v3_strict``, and stay
    on the V3 default otherwise.  This is the only contract production
    re-label runs depend on; the rest of the dispatch is plumbing."""
    monkeypatch.setenv("NLA_POSITION_PROMPT_MODE", "v4")
    assert _select_position_builder() is build_v4_position_prompt
    assert _select_position_builder("v4") is build_v4_position_prompt
    assert _select_position_builder("V4_position") is build_v4_position_prompt
    assert _select_position_builder("v4-position") is build_v4_position_prompt

    assert _select_position_builder("strict") is build_strict_position_prompt
    assert _select_position_builder("v3_strict") is build_strict_position_prompt

    assert _select_position_builder("v3") is build_position_prompt
    assert _select_position_builder("totally_unknown_mode") is build_position_prompt

    monkeypatch.setenv("NLA_POSITION_PROMPT_MODE", "v3")
    assert _select_position_builder() is build_position_prompt


def test_pipeline_threads_suite_through_to_v4_builder(monkeypatch, tmp_path: Path):
    """When ``NLA_POSITION_PROMPT_MODE=v4`` and the input carries a ``suite``
    attribute (set by SA5's pipeline plumbing), ``_build_messages`` must
    invoke the V4 builder *with* ``suite=...`` so the per-suite addendum
    fires.  Verifies the suite-aware dispatch end-to-end at unit scope."""
    monkeypatch.setenv("NLA_POSITION_PROMPT_MODE", "v4")
    img = save_jpeg(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "img.jpg")
    inp = PositionLabelInput(
        example_id="ex_suite@p0_image_patch",
        instruction="put the bowl on the plate",
        decoded_text_context="<image: 4 patches>",
        position_index=0,
        position_type="image_patch",
        sequence_length=10,
        image_paths=[str(img)],
        image_patch_meta=(0, 4),
        suite="libero_spatial",
    )
    messages, kind, _ = _build_messages(inp)
    assert kind == "position"
    sys_p = messages[0]["content"]
    # libero_spatial addendum must be present (proof that suite was threaded
    # through to build_v4_position_prompt).
    assert "LIBERO-SPATIAL addendum" in sys_p
    # And the V4 system text identity (so we know we did not fall back to V3).
    assert "Scaffold-leakage ban" in sys_p


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

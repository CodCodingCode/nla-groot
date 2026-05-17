"""Unit tests for pure helpers in ``scripts/eval/nla_steer_leverage_sweep.py``.

The CLI loads GR00T + AR lazily inside ``main()``; the dataclass/parsing/null
helpers stand alone and are worth exercising on CPU without those heavy deps.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SWEEP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "nla_steer_leverage_sweep.py"


def _load_sweep_module():
    name = "nla_steer_leverage_sweep_test_alias"
    spec = importlib.util.spec_from_file_location(name, SWEEP_PATH)
    assert spec and spec.loader, f"failed to load {SWEEP_PATH}"
    mod = importlib.util.module_from_spec(spec)
    # dataclass() looks up the class's owning module in sys.modules; register first.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep_module()


def test_parse_range_spec_basic_and_open_end():
    assert sweep._parse_range_spec("0:128:8") == (0, 128, 8)
    assert sweep._parse_range_spec("0::4") == (0, None, 4)
    assert sweep._parse_range_spec("16:64") == (16, 64, 1)
    assert sweep._parse_range_spec(None) is None
    assert sweep._parse_range_spec("") is None


def test_parse_range_spec_rejects_bad_stride():
    with pytest.raises(SystemExit):
        sweep._parse_range_spec("0:10:0")


def test_parse_int_and_float_lists():
    assert sweep._parse_int_list("0, 1, 2,") == [0, 1, 2]
    assert sweep._parse_int_list("") == []
    assert sweep._parse_float_list("0.25,0.5,1.0") == [0.25, 0.5, 1.0]


def test_build_conditions_grid():
    conds = sweep._build_conditions(
        placements=["last_text", "anchor", "image_patch", "fixed"],
        blends=[0.5, 1.0],
        image_patch_seeds=[0, 1],
        fixed_token_indices=[10, 20],
    )
    placements = sorted({c.placement for c in conds})
    assert placements == ["anchor", "fixed", "image_patch", "last_text"]
    # 2 blends × (1 last_text + 1 anchor + 2 image_patch + 2 fixed) = 12.
    assert len(conds) == 12

    image_conds = [c for c in conds if c.placement == "image_patch"]
    assert {c.image_patch_seed for c in image_conds} == {0, 1}

    fixed_conds = [c for c in conds if c.placement == "fixed"]
    assert {c.fixed_token_index for c in fixed_conds} == {10, 20}


def test_build_conditions_rejects_image_patch_without_seeds():
    with pytest.raises(SystemExit):
        sweep._build_conditions(
            placements=["image_patch"],
            blends=[1.0],
            image_patch_seeds=[],
            fixed_token_indices=[],
        )


def test_build_conditions_rejects_fixed_without_indices():
    with pytest.raises(SystemExit):
        sweep._build_conditions(
            placements=["fixed"],
            blends=[1.0],
            image_patch_seeds=[],
            fixed_token_indices=[],
        )


def test_matched_null_vec_preserves_l2_norm():
    real = torch.tensor([3.0, 4.0, 0.0])
    target_norm = float(torch.linalg.norm(real))
    null = sweep._matched_null_vec(real, seed=123)
    assert null.shape == real.shape
    assert float(torch.linalg.norm(null)) == pytest.approx(target_norm, rel=1e-6)


def test_matched_null_vec_is_deterministic_per_seed():
    real = torch.ones(8)
    a = sweep._matched_null_vec(real, seed=7)
    b = sweep._matched_null_vec(real, seed=7)
    c = sweep._matched_null_vec(real, seed=8)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_resolve_indices_matches_backbone_steer_helper():
    from nla.steering.backbone_steer import SteerSpec

    probe = sweep._ProbeHook()
    probe.attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    probe.image_mask = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.bool)

    last_text = sweep._resolve_indices(probe, SteerSpec("last_text"))
    assert last_text == [4]
    image_patch = sweep._resolve_indices(probe, SteerSpec("image_patch", image_patch_seed=42))
    assert image_patch and image_patch[0] in {2, 3}

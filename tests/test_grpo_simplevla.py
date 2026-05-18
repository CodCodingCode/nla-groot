"""Unit tests for the SimpleVLA-RL-inspired GRPO knobs (A1 workstream).

Covers ``GRPOConfig`` fields ``dynamic_sampling`` /
``dynamic_sampling_threshold``, ``use_ppo_clip`` / ``clip_eps_low`` /
``clip_eps_high``, ``disable_kl_anchor``, ``rollout_temperature_high`` and
their matching CLI flags in ``scripts/training/run_grpo.py``. CPU only;
no real Qwen3 model loads (stub pattern from ``test_grpo_sim_reward.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from nla.training.grpo import GRPOConfig, _serialize_config, grpo_step


# ---------------------------------------------------------------------------
# Stubs that mimic the AV / AR interface ``grpo_step`` needs.
# ---------------------------------------------------------------------------


class _StubTokenizer:
    eos_token_id = 99
    pad_token_id = 0


class _StubAV(nn.Module):
    """Minimal ActivationVerbalizer stand-in."""

    def __init__(self, *, canned_logp: torch.Tensor | None = None):
        super().__init__()
        self._pad_id = 0
        self.tokenizer = _StubTokenizer()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._canned_logp = canned_logp
        self._calls = {"generate": 0, "score_tokens": 0}
        self._last_kwargs: dict = {}

    def generate(self, acts, ptypes, **kwargs):
        self._calls["generate"] += 1
        self._last_kwargs = dict(kwargs)
        B = acts.shape[0]
        T = kwargs.get("max_new_tokens", 4)
        return {"text": [f"r{i}" for i in range(B)],
                "token_ids": torch.ones(B, T, dtype=torch.long)}

    def score_tokens(self, acts, ptypes, gen_ids, gen_mask, **kwargs):
        self._calls["score_tokens"] += 1
        B, T = gen_ids.shape
        base = (self._canned_logp.float() if self._canned_logp is not None
                else torch.zeros(B, T))
        assert base.shape == (B, T)
        return base + self.dummy * 0.0


class _StubARCfg:
    alpha = 1.0


class _StubAR(nn.Module):
    """Returns ``pred`` so ``-((target - pred)**2).mean(-1) == rewards``.

    Assumes ``activations`` are zeros and ``H=1`` (so target=0); rewards
    must be ``<= 0`` (mirrors the real recon reward).
    """

    def __init__(self, rewards: torch.Tensor):
        super().__init__()
        self.cfg = _StubARCfg()
        self._rewards = rewards.float()

    def __call__(self, texts, device):
        BK = len(texts)
        assert BK == self._rewards.numel()
        out = torch.zeros(BK, 1)
        for i, r in enumerate(self._rewards.tolist()):
            assert r <= 0.0, f"stub AR only supports non-positive rewards (got {r})"
            out[i, 0] = (-r) ** 0.5
        return out


def _make_inputs(rewards, *, B, K, canned_pol_logp=None, canned_ref_logp=None,
                 ptypes=None):
    assert len(rewards) == B * K
    acts = torch.zeros(B, 1)
    pol = _StubAV(canned_logp=canned_pol_logp)
    ref = _StubAV(canned_logp=canned_ref_logp)
    ar = _StubAR(torch.tensor(rewards, dtype=torch.float32))
    return acts, (ptypes or ["anchor"] * B), ar, pol, ref


# ---------------------------------------------------------------------------
# TestDynamicSampling
# ---------------------------------------------------------------------------


class TestDynamicSampling:
    def test_skip_zero_variance_group(self):
        B, K = 2, 2
        rewards = [-1.0, -1.0, -1.0, -4.0]  # group 0: std=0, group 1: mixed
        acts, ptypes, ar, pol, ref = _make_inputs(rewards, B=B, K=K,
                                                  ptypes=["anchor"] * 2)
        out = grpo_step(
            pol, ref, ar, acts, ptypes,
            K=K, beta=0.0, rollout_max_new_tokens=4,
            rollout_temperature=1.0, rollout_top_p=1.0, use_kl=False,
            dynamic_sampling=True, dynamic_sampling_threshold=1e-6,
        )
        diag = out["diagnostics"]
        assert "dynamic_sampling_drop_frac" in diag, diag.keys()
        assert diag["dynamic_sampling_drop_frac"] == pytest.approx(0.5, abs=1e-6)
        adv = out["advantages"].view(B, K)
        assert torch.allclose(adv[0], torch.zeros(K)), adv[0]
        assert adv[1].abs().sum() > 0.0, adv[1]

    def test_keep_mixed_reward_group(self):
        B, K = 2, 2
        rewards = [-1.0, -4.0, -2.0, -8.0]
        acts, ptypes, ar, pol, ref = _make_inputs(rewards, B=B, K=K,
                                                  ptypes=["anchor"] * 2)
        out = grpo_step(
            pol, ref, ar, acts, ptypes,
            K=K, beta=0.0, rollout_max_new_tokens=4,
            rollout_temperature=1.0, rollout_top_p=1.0, use_kl=False,
            dynamic_sampling=True, dynamic_sampling_threshold=1e-6,
        )
        assert out["diagnostics"]["dynamic_sampling_drop_frac"] == pytest.approx(0.0)

    def test_threshold_respected(self):
        B, K = 2, 2
        rewards = [-1.0, -2.0, -1.0, -8.0]  # group-0 std~0.5, group-1 std~3.5
        kwargs = dict(
            K=K, beta=0.0, rollout_max_new_tokens=4,
            rollout_temperature=1.0, rollout_top_p=1.0, use_kl=False,
            dynamic_sampling=True,
        )
        acts, ptypes, ar, pol, ref = _make_inputs(rewards, B=B, K=K,
                                                  ptypes=["anchor"] * 2)
        out_low = grpo_step(pol, ref, ar, acts, ptypes,
                            dynamic_sampling_threshold=0.1, **kwargs)
        acts, ptypes, ar, pol, ref = _make_inputs(rewards, B=B, K=K,
                                                  ptypes=["anchor"] * 2)
        out_high = grpo_step(pol, ref, ar, acts, ptypes,
                             dynamic_sampling_threshold=2.0, **kwargs)
        d_low = out_low["diagnostics"]["dynamic_sampling_drop_frac"]
        d_high = out_high["diagnostics"]["dynamic_sampling_drop_frac"]
        assert d_high > d_low, (d_low, d_high)

    def test_diagnostic_emitted(self):
        B, K = 1, 2
        acts, ptypes, ar, pol, ref = _make_inputs([-1.0, -4.0], B=B, K=K)
        out = grpo_step(
            pol, ref, ar, acts, ptypes,
            K=K, beta=0.0, rollout_max_new_tokens=4,
            rollout_temperature=1.0, rollout_top_p=1.0, use_kl=False,
            dynamic_sampling=True, dynamic_sampling_threshold=1e-6,
        )
        assert "dynamic_sampling_drop_frac" in out["diagnostics"]

    def test_default_on_when_sim_reward(self):
        """A1 stores ``dynamic_sampling=None`` as an auto-marker; the
        run_grpo rule is ON iff ``sim_reward_weight > 0``."""
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z",
                         sim_reward_weight=0.5)
        assert hasattr(cfg, "dynamic_sampling"), "field missing"
        eff = (bool(cfg.dynamic_sampling) if cfg.dynamic_sampling is not None
               else cfg.sim_reward_weight > 0.0)
        assert eff is True

    def test_default_off_when_recon_only(self):
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
        eff = (bool(cfg.dynamic_sampling) if cfg.dynamic_sampling is not None
               else cfg.sim_reward_weight > 0.0)
        assert eff is False


# ---------------------------------------------------------------------------
# TestClipHigher
# ---------------------------------------------------------------------------


class TestClipHigher:
    def test_clip_eps_low_default(self):
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
        assert hasattr(cfg, "clip_eps_low")
        assert cfg.clip_eps_low == pytest.approx(0.2)  # PPO standard

    def test_clip_eps_high_default(self):
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
        assert hasattr(cfg, "clip_eps_high")
        assert cfg.clip_eps_high > cfg.clip_eps_low  # "Clip-Higher"

    def test_use_ppo_clip_off_is_byte_identical(self):
        B, K = 1, 2
        rewards = [-1.0, -4.0]
        torch.manual_seed(0)
        acts, ptypes, ar, pol, ref = _make_inputs(rewards, B=B, K=K)
        out_a = grpo_step(
            pol, ref, ar, acts, ptypes, K=K, beta=0.0,
            rollout_max_new_tokens=4, rollout_temperature=1.0,
            rollout_top_p=1.0, use_kl=False, use_ppo_clip=False,
        )
        torch.manual_seed(0)
        acts, ptypes, ar, pol, ref = _make_inputs(rewards, B=B, K=K)
        out_b = grpo_step(
            pol, ref, ar, acts, ptypes, K=K, beta=0.0,
            rollout_max_new_tokens=4, rollout_temperature=1.0,
            rollout_top_p=1.0, use_kl=False,  # no PPO kwarg = baseline
        )
        assert torch.allclose(out_a["pg_loss"], out_b["pg_loss"], atol=1e-7), (
            out_a["pg_loss"], out_b["pg_loss"])
        assert torch.allclose(out_a["loss"], out_b["loss"], atol=1e-7)

    def test_use_ppo_clip_on_clamps_ratio(self):
        """A1 writes ratio = exp(new_logp - new_logp.detach()) so it is
        identically 1 in single-step GRPO -> the clip is a no-op and the
        loss MUST match the unclipped path (guards against e.g. someone
        using ``ref_logp`` as "old_logp" by mistake)."""
        B, K, T = 1, 2, 4
        rewards = [-1.0, -4.0]
        canned_pol = torch.full((B * K, T), 2.0)
        canned_ref = torch.zeros(B * K, T)
        acts, ptypes, ar, pol, ref = _make_inputs(
            rewards, B=B, K=K,
            canned_pol_logp=canned_pol, canned_ref_logp=canned_ref,
        )
        out_off = grpo_step(
            pol, ref, ar, acts, ptypes, K=K, beta=0.0,
            rollout_max_new_tokens=T, rollout_temperature=1.0,
            rollout_top_p=1.0, use_kl=False, use_ppo_clip=False,
        )
        acts, ptypes, ar, pol, ref = _make_inputs(
            rewards, B=B, K=K,
            canned_pol_logp=canned_pol, canned_ref_logp=canned_ref,
        )
        out_on = grpo_step(
            pol, ref, ar, acts, ptypes, K=K, beta=0.0,
            rollout_max_new_tokens=T, rollout_temperature=1.0,
            rollout_top_p=1.0, use_kl=False,
            use_ppo_clip=True, clip_eps_low=0.2, clip_eps_high=0.3,
        )
        assert torch.isfinite(out_on["pg_loss"]).item()
        assert torch.allclose(out_off["pg_loss"], out_on["pg_loss"], atol=1e-6), (
            f"In single-step GRPO the clip must be a no-op; got "
            f"off={out_off['pg_loss']} on={out_on['pg_loss']}"
        )


# ---------------------------------------------------------------------------
# TestNoKL
# ---------------------------------------------------------------------------


class TestNoKL:
    def test_disable_kl_anchor_skips_kl_term(self):
        B, K = 1, 2
        acts, ptypes, ar, pol, ref = _make_inputs([-1.0, -4.0], B=B, K=K)
        out = grpo_step(
            pol, ref, ar, acts, ptypes, K=K, beta=1.0,
            rollout_max_new_tokens=4, rollout_temperature=1.0,
            rollout_top_p=1.0, disable_kl_anchor=True,
        )
        assert float(out["kl_loss"].item()) == 0.0, out["kl_loss"]
        assert out["diagnostics"]["kl_loss"] == 0.0

    def test_disable_kl_anchor_skips_ref_forward(self):
        B, K = 1, 2
        acts, ptypes, ar, pol, ref = _make_inputs([-1.0, -4.0], B=B, K=K)
        grpo_step(
            pol, ref, ar, acts, ptypes, K=K, beta=0.5,
            rollout_max_new_tokens=4, rollout_temperature=1.0,
            rollout_top_p=1.0, disable_kl_anchor=True,
        )
        assert ref._calls["score_tokens"] == 0, (
            f"ref_av.score_tokens was called {ref._calls['score_tokens']} times "
            "even with disable_kl_anchor=True"
        )
        # And ref_av=None should also work.
        acts, ptypes, ar, pol, _ = _make_inputs([-1.0, -4.0], B=B, K=K)
        out2 = grpo_step(
            pol, None, ar, acts, ptypes, K=K, beta=0.5,
            rollout_max_new_tokens=4, rollout_temperature=1.0,
            rollout_top_p=1.0, disable_kl_anchor=True,
        )
        assert float(out2["kl_loss"].item()) == 0.0

    def test_serialize_config_hides_when_default(self):
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
        out = _serialize_config(cfg)
        assert "disable_kl_anchor" not in out, (
            "`disable_kl_anchor` leaked into saved config at default"
        )
        cfg_on = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z",
                            disable_kl_anchor=True)
        assert _serialize_config(cfg_on).get("disable_kl_anchor") is True


# ---------------------------------------------------------------------------
# TestTempOverride
# ---------------------------------------------------------------------------


def _resolve_rollout_temp(cfg: GRPOConfig) -> float:
    """Mirror the resolution rule that ``run_grpo`` applies."""
    return (cfg.rollout_temperature_high
            if cfg.rollout_temperature_high is not None
            else cfg.rollout_temperature)


class TestTempOverride:
    """``rollout_temperature_high`` is plumbed through ``run_grpo`` (not
    ``grpo_step``); we exercise the documented resolution rule directly."""

    def test_rollout_temperature_high_overrides(self):
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z",
                         rollout_temperature=1.0, rollout_temperature_high=1.7)
        assert _resolve_rollout_temp(cfg) == pytest.approx(1.7)
        assert cfg.rollout_temperature == pytest.approx(1.0)  # base untouched

    def test_rollout_temperature_high_none_uses_base_temperature(self):
        cfg = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z",
                         rollout_temperature=0.7, rollout_temperature_high=None)
        assert _resolve_rollout_temp(cfg) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# TestCliFlags
# ---------------------------------------------------------------------------


def _load_run_grpo_module():
    script = Path(__file__).resolve().parent.parent / "scripts" / "training" / "run_grpo.py"
    spec = importlib.util.spec_from_file_location("run_grpo_test_mod", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestCliFlags:
    def test_all_new_flags_parse(self):
        mod = _load_run_grpo_module()
        parser = mod._build_parser()
        args = parser.parse_args([
            "--sft-dir", "x", "--activations-root", "y", "--output-dir", "z",
            "--dynamic-sampling",
            "--dynamic-sampling-threshold", "0.05",
            "--use-ppo-clip",
            "--clip-eps-low", "0.15",
            "--clip-eps-high", "0.3",
            "--disable-kl-anchor",
            "--rollout-temperature-high", "1.8",
        ])
        assert getattr(args, "dynamic_sampling", None) is True
        assert getattr(args, "dynamic_sampling_threshold", None) == pytest.approx(0.05)
        assert getattr(args, "use_ppo_clip", None) is True
        assert getattr(args, "clip_eps_low", None) == pytest.approx(0.15)
        assert getattr(args, "clip_eps_high", None) == pytest.approx(0.3)
        assert getattr(args, "disable_kl_anchor", None) is True
        assert getattr(args, "rollout_temperature_high", None) == pytest.approx(1.8)

    def test_config_round_trip(self):
        cfg = GRPOConfig(
            sft_dir="x", activations_root="y", output_dir="z",
            dynamic_sampling=True, dynamic_sampling_threshold=0.05,
            use_ppo_clip=True, clip_eps_low=0.15, clip_eps_high=0.3,
            disable_kl_anchor=True, rollout_temperature_high=1.8,
        )
        out = _serialize_config(cfg)
        assert out["dynamic_sampling"] is True
        assert out["dynamic_sampling_threshold"] == pytest.approx(0.05)
        assert out["use_ppo_clip"] is True
        assert out["clip_eps_low"] == pytest.approx(0.15)
        assert out["clip_eps_high"] == pytest.approx(0.3)
        assert out["disable_kl_anchor"] is True
        assert out["rollout_temperature_high"] == pytest.approx(1.8)

        cfg_def = GRPOConfig(sft_dir="x", activations_root="y", output_dir="z")
        out_def = _serialize_config(cfg_def)
        for k in ("disable_kl_anchor", "rollout_temperature_high",
                  "use_ppo_clip", "clip_eps_low", "clip_eps_high"):
            assert k not in out_def, (
                f"{k!r} leaked into saved config at default"
            )

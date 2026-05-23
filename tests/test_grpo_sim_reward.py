"""Unit tests for the optional sim-success reward (Framing B sim-GRPO).

These never spawn a real LIBERO subprocess: ``SimRewardWorker._run_rollout_subprocess``
is monkeypatched to return canned JSON summaries so we can deterministically
exercise the caching, error-skipping, and three-way reward blending paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nla.training.grpo import (
    GRPOConfig,
    _blend_multi_rewards,
    _blend_rewards,
    _serialize_config,
    _validate_sim_config,
    _zscore,
)
from nla.training.sim_reward import (
    SimRewardJob,
    SimRewardWorker,
    assemble_jobs,
    load_sim_cache,
    sim_cache_key,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_sim_cache_key_is_stable_and_collision_free():
    k1 = sim_cache_key("env_A", "task_A", "src_1", "text one", 0, 100)
    k2 = sim_cache_key("env_A", "task_A", "src_1", "text one", 0, 100)
    assert k1 == k2
    # Any field change flips the key.
    assert sim_cache_key("env_B", "task_A", "src_1", "text one", 0, 100) != k1
    assert sim_cache_key("env_A", "task_B", "src_1", "text one", 0, 100) != k1
    assert sim_cache_key("env_A", "task_A", "src_2", "text one", 0, 100) != k1
    assert sim_cache_key("env_A", "task_A", "src_1", "text two", 0, 100) != k1
    assert sim_cache_key("env_A", "task_A", "src_1", "text one", 1, 100) != k1
    assert sim_cache_key("env_A", "task_A", "src_1", "text one", 0, 200) != k1


def test_sim_cache_key_includes_language_swap_and_steer_disabled_fields():
    """Regression: eval-v2 adds two new per-job knobs
    (``policy_language_override`` for the language_swap arm and
    ``steer_disabled`` for the no_steer arm) that share the
    (env, task, source, text, seed, steps, placement, steer_h) tuple
    with the matched arm. The cache key must flip when either is set,
    or compute() returns the matched arm's rollout for the new arms.
    """
    base_args = ("env_A", "task_A", "src_1", "text one", 0, 100)
    k_default = sim_cache_key(*base_args)
    k_lang = sim_cache_key(*base_args, policy_language_override="put the wine bottle")
    k_disabled = sim_cache_key(*base_args, steer_disabled=True)
    k_both = sim_cache_key(
        *base_args,
        policy_language_override="put the wine bottle",
        steer_disabled=True,
    )
    k_wpred = sim_cache_key(*base_args, w_predicate=1.0)
    assert len({k_default, k_lang, k_disabled, k_both, k_wpred}) == 5


def test_sim_cache_key_legacy_defaults_byte_identical():
    """The default eval-v2 args must hash to the same key the legacy
    (pre-2026-05) callers produced. This keeps GRPO sim caches written
    before the language_swap fields existed readable by the new code.
    """
    legacy = sim_cache_key("env_A", "task_A", "src_1", "text one", 0, 100)
    v2_defaults = sim_cache_key(
        "env_A", "task_A", "src_1", "text one", 0, 100,
        policy_language_override=None,
        steer_disabled=False,
        w_predicate=None,
    )
    assert legacy == v2_defaults


def test_sim_cache_key_includes_placement_and_steer_fingerprint():
    """Regression: causal-arm sweeps (semantic / matched_null / wrong_placement)
    share the (env, task, source, text, seed, steps) tuple but vary the steer
    vector and placement. The old key formula collided on all three arms,
    causing the in-memory cache to silently return the first arm's rollout
    for every subsequent arm. The new key must flip on either field.
    """
    base_args = ("env_A", "task_A", "src_1", "text one", 0, 100)
    k_semantic = sim_cache_key(*base_args, placement="image_patch", steer_h_fp="hash_real")
    k_wrong_place = sim_cache_key(*base_args, placement="last_text", steer_h_fp="hash_real")
    k_null_vec = sim_cache_key(*base_args, placement="image_patch", steer_h_fp="hash_null")
    assert len({k_semantic, k_wrong_place, k_null_vec}) == 3

    # Defaults still produce a stable key for callers that don't pass extras
    # (e.g. legacy training paths that always use the same placement/vector).
    legacy = sim_cache_key(*base_args)
    legacy_again = sim_cache_key(*base_args)
    assert legacy == legacy_again


def test_simrewardworker_compute_runs_each_causal_arm(monkeypatch, tmp_path):
    """Regression: with a shared (text, source) but different steer_h / placement,
    SimRewardWorker.compute() must invoke a fresh rollout per arm instead of
    silently returning the first arm's result via the in-memory cache.
    """
    calls: list[tuple[str, str]] = []  # (fingerprint, placement)

    def fake_rollout(job, **_kw):
        from nla.training.sim_reward import _steer_h_fingerprint
        fp = _steer_h_fingerprint(job.steer_h)
        calls.append((fp, job.placement))
        return {
            "r_sim": float(len(calls)),
            "n_steps": 1,
            "early_stopped": True,
            "success_any": False,
            "sim_score_breakdown": {
                "predicate": float(len(calls)) / 10.0,
                "r_dist": 0.0, "r_displace": 0.0, "r_contact": 0.0,
            },
        }

    monkeypatch.setattr(
        "nla.training.sim_reward._run_rollout_subprocess", fake_rollout
    )

    worker = SimRewardWorker(
        policy_host="x", policy_port=0,
        n_workers=1,
        rollout_python="python", rollout_script="x.py",
        cache_path=None,
        scratch_dir=str(tmp_path),
    )

    base = dict(
        env_name="env_A", target_task="task_A",
        source_id="src_1", text="text one",
        seed=0, sim_max_steps=10, blend=1.0,
    )
    job_semantic = SimRewardJob(
        steer_h=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        placement="image_patch", **base,
    )
    job_null = SimRewardJob(
        steer_h=np.array([-0.1, 0.5, 0.7], dtype=np.float32),
        placement="image_patch", **base,
    )
    job_wrong_place = SimRewardJob(
        steer_h=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        placement="last_text", **base,
    )

    results = worker.compute([job_semantic, job_null, job_wrong_place])
    assert len(results) == 3
    assert len(calls) == 3, f"each arm must invoke a fresh rollout, got {calls}"
    # Three distinct r_sim values prove no in-memory cache short-circuit.
    assert len({r.r_sim for r in results}) == 3


def test_load_sim_cache_handles_missing_and_corrupt(tmp_path: Path):
    assert load_sim_cache(None) == {}
    assert load_sim_cache(tmp_path / "nonexistent.jsonl") == {}
    p = tmp_path / "cache.jsonl"
    p.write_text(
        '{"key":"abc","r_sim":1.5}\n'
        'not-json\n'
        '\n'
        '{"key":"def","r_sim":-0.5,"predicate":0.0}\n'
    )
    cache = load_sim_cache(p)
    assert set(cache) == {"abc", "def"}
    assert cache["abc"]["r_sim"] == 1.5


# ---------------------------------------------------------------------------
# Reward blending math
# ---------------------------------------------------------------------------


def test_blend_multi_rewards_recon_only_is_identity():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = _blend_multi_rewards(r)
    assert torch.allclose(out, r)


def test_blend_multi_rewards_judge_only_matches_legacy_path():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rj = torch.tensor([0.5, -0.5, 1.0, 1.5])
    blended = _blend_multi_rewards(r, r_judge=rj, judge_weight=0.5)
    legacy = _blend_rewards(r, rj, 0.5)
    assert torch.allclose(blended, legacy), (blended, legacy)


def test_blend_multi_rewards_sim_term_is_zero_mean_per_step():
    # When only the sim term is on (judge_weight=0, sim_weight=1) the output
    # is exactly z(r_sim) -> mean 0, std 1 (within float tolerance).
    torch.manual_seed(0)
    r = torch.randn(8)
    rs = torch.randn(8) * 2.5 + 1.0
    out = _blend_multi_rewards(r, r_sim=rs, sim_weight=1.0)
    assert abs(float(out.mean())) < 1e-5
    assert abs(float(out.std()) - 1.0) < 1e-4


def test_blend_multi_rewards_three_way_recovers_base_weight():
    # With judge_weight=0.3 + sim_weight=0.2 the recon term coefficient must
    # be 0.5; we can recover it by setting both extras to zero and looking
    # at the residual.
    r = torch.tensor([1.0, -1.0, 0.5, -0.5])
    out = _blend_multi_rewards(
        r,
        r_judge=torch.zeros_like(r), judge_weight=0.3,
        r_sim=torch.zeros_like(r),   sim_weight=0.2,
    )
    expected = 0.5 * _zscore(r)
    assert torch.allclose(out, expected)


def test_blend_multi_rewards_overweight_clamps_base_to_zero(caplog):
    r = torch.tensor([1.0, 2.0, 3.0])
    rj = torch.tensor([0.0, 0.0, 0.0])
    rs = torch.tensor([1.0, -1.0, 0.0])
    with caplog.at_level("WARNING"):
        out = _blend_multi_rewards(
            r, r_judge=rj, judge_weight=0.7, r_sim=rs, sim_weight=0.7,
        )
    assert "clamped" in caplog.text or "clamped" in "".join(r.message for r in caplog.records)
    expected = 0.7 * _zscore(rj) + 0.7 * _zscore(rs)
    assert torch.allclose(out, expected)


def test_blend_multi_rewards_requires_r_judge_when_judge_weight_positive():
    r = torch.tensor([1.0, 2.0, 3.0])
    rs = torch.tensor([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="r_judge"):
        _blend_multi_rewards(r, judge_weight=0.3, r_sim=rs, sim_weight=0.3)


# ---------------------------------------------------------------------------
# Partial sim blending (per-row eligibility mask)
# ---------------------------------------------------------------------------


def test_blend_multi_rewards_sim_active_all_true_matches_classic_path():
    # ``sim_active == 1`` everywhere should reproduce the byte-identical
    # classic blend (modulo float associativity of the per-row vs scalar
    # multiplies, which is exact at this scale).
    torch.manual_seed(0)
    r = torch.randn(6)
    rs = torch.randn(6) * 1.5 + 0.5
    classic = _blend_multi_rewards(r, r_sim=rs, sim_weight=0.4)
    masked = _blend_multi_rewards(
        r, r_sim=rs, sim_weight=0.4, sim_active=torch.ones(6),
    )
    assert torch.allclose(classic, masked, atol=1e-6), (classic, masked)


def test_blend_multi_rewards_sim_active_all_false_collapses_to_recon():
    # No active sim rows -> sim contribution is zero and recon picks up
    # the full slack, so output is z(r_recon) (judge off).
    r = torch.tensor([1.0, -1.0, 2.0, -2.0])
    rs = torch.tensor([5.0, 5.0, 5.0, 5.0])  # would otherwise dominate
    out = _blend_multi_rewards(
        r, r_sim=rs, sim_weight=0.6, sim_active=torch.zeros(4),
    )
    expected = _zscore(r)
    assert torch.allclose(out, expected)


def test_blend_multi_rewards_sim_active_partial_zscores_only_active():
    # Active rows: 0,2,3; inactive: 1. The inactive sim entry has a wild
    # value (1000) that would blow up the mean/std if we let it through;
    # confirm we don't.
    r = torch.tensor([0.0, 0.0, 0.0, 0.0])
    rs = torch.tensor([1.0, 1000.0, -1.0, 0.0])
    active = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out = _blend_multi_rewards(
        r, r_sim=rs, sim_weight=1.0, sim_active=active,
    )
    # Active z-score is taken over [1.0, -1.0, 0.0].
    active_vals = torch.tensor([1.0, -1.0, 0.0])
    mu, sd = active_vals.mean(), active_vals.std().clamp_min(1e-6)
    z = (rs - mu) / sd
    # Inactive row contributes 0 from the sim branch; recon is 0 already.
    expected = torch.zeros_like(r)
    expected[0] = z[0]
    expected[2] = z[2]
    expected[3] = z[3]
    assert torch.allclose(out, expected, atol=1e-5)


def test_blend_multi_rewards_sim_active_per_row_recovers_recon_slack():
    # Inactive rows must keep their recon term at full strength (1.0 *
    # z(r_recon)), while active rows pay the sim toll (base = 1 - sim_w).
    torch.manual_seed(1)
    r = torch.randn(8)
    rs = torch.randn(8)
    active = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    sim_w = 0.5
    out = _blend_multi_rewards(
        r, r_sim=rs, sim_weight=sim_w, sim_active=active,
    )
    z_r = _zscore(r)
    # Inactive entries must equal z(r_recon) since judge off + sim off
    # for those rows.
    inactive_mask = active == 0.0
    assert torch.allclose(out[inactive_mask], z_r[inactive_mask], atol=1e-6)
    # No NaN anywhere.
    assert not torch.isnan(out).any()


def test_blend_multi_rewards_sim_active_with_judge_only_taxes_active_rows():
    # judge_weight=0.3, sim_weight=0.4. On inactive sim rows the base
    # (recon) coefficient should be 1 - 0.3 = 0.7; on active rows it
    # should be 1 - 0.3 - 0.4 = 0.3.
    r = torch.tensor([1.0, -1.0, 2.0, -2.0])
    rj = torch.tensor([0.0, 0.0, 0.0, 0.0])
    rs = torch.tensor([1.0, -1.0, 1.0, -1.0])
    active = torch.tensor([1.0, 0.0, 1.0, 0.0])
    out = _blend_multi_rewards(
        r,
        r_judge=rj, judge_weight=0.3,
        r_sim=rs, sim_weight=0.4,
        sim_active=active,
    )
    z_r = _zscore(r)
    # Build expected: inactive rows -> 0.7 * z(r); active rows -> 0.3 * z(r) + 0.4 * z(rs|active).
    active_vals = rs[active.bool()]
    mu = active_vals.mean()
    sd = active_vals.std().clamp_min(1e-6)
    z_s = (rs - mu) / sd
    expected = torch.where(
        active.bool(), 0.3 * z_r + 0.4 * z_s, 0.7 * z_r,
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_blend_multi_rewards_sim_active_shape_mismatch_raises():
    r = torch.tensor([1.0, 2.0, 3.0])
    rs = torch.tensor([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="sim_active shape"):
        _blend_multi_rewards(
            r, r_sim=rs, sim_weight=0.5,
            sim_active=torch.tensor([1.0, 0.0]),
        )


# ---------------------------------------------------------------------------
# Config validation + serialization
# ---------------------------------------------------------------------------


def test_validate_sim_config_off_is_noop():
    cfg = GRPOConfig(sim_reward_weight=0.0)
    _validate_sim_config(cfg)  # should not raise


def test_validate_sim_config_requires_pairs_path():
    cfg = GRPOConfig(sim_reward_weight=0.5, sim_counterfactual_pairs_path=None)
    with pytest.raises(ValueError, match="sim-counterfactual-pairs-path"):
        _validate_sim_config(cfg)


def test_validate_sim_config_rejects_missing_pairs_file(tmp_path: Path):
    cfg = GRPOConfig(
        sim_reward_weight=0.5,
        sim_counterfactual_pairs_path=str(tmp_path / "missing.jsonl"),
    )
    with pytest.raises(ValueError, match="does not exist"):
        _validate_sim_config(cfg)


def test_validate_sim_config_rejects_bad_blend_and_n_workers(tmp_path: Path):
    p = tmp_path / "pairs.jsonl"
    p.write_text("")
    cfg = GRPOConfig(
        sim_reward_weight=0.5,
        sim_counterfactual_pairs_path=str(p),
        sim_blend=-0.1,
    )
    with pytest.raises(ValueError, match="sim_blend"):
        _validate_sim_config(cfg)
    cfg = GRPOConfig(
        sim_reward_weight=0.5,
        sim_counterfactual_pairs_path=str(p),
        sim_n_workers=0,
    )
    with pytest.raises(ValueError, match="sim_n_workers"):
        _validate_sim_config(cfg)


def test_serialize_config_hides_sim_fields_when_off():
    cfg = GRPOConfig()  # sim off by default
    out = _serialize_config(cfg)
    for k in (
        "sim_reward_weight", "sim_counterfactual_pairs_path",
        "sim_policy_host", "sim_n_workers", "sim_placement",
    ):
        assert k not in out


def test_serialize_config_keeps_sim_fields_when_on(tmp_path: Path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("")
    cfg = GRPOConfig(
        sim_reward_weight=0.5,
        sim_counterfactual_pairs_path=str(pairs),
        sim_n_workers=2,
    )
    out = _serialize_config(cfg)
    assert out["sim_reward_weight"] == 0.5
    assert out["sim_n_workers"] == 2
    assert out["sim_counterfactual_pairs_path"] == str(pairs)


# ---------------------------------------------------------------------------
# SimRewardWorker (subprocess stubbed out)
# ---------------------------------------------------------------------------


def _make_job(text: str = "do the thing", seed: int = 0, H: int = 8) -> SimRewardJob:
    return SimRewardJob(
        env_name="LIBERO_GOAL_put_the_bowl_on_the_plate",
        target_task="put_the_bowl_on_the_plate",
        source_id="ep01_t005",
        text=text,
        seed=seed,
        steer_h=np.zeros(H, dtype=np.float32),
        sim_max_steps=10,
        placement="image_patch",
        blend=1.0,
    )


def test_worker_uses_cache_for_repeated_keys(monkeypatch, tmp_path: Path):
    """A repeated job key should not invoke the subprocess a second time."""
    cache_path = tmp_path / "sim_cache.jsonl"
    n_calls = {"count": 0}

    def fake_run(job, **kwargs):
        n_calls["count"] += 1
        return {
            "r_sim": 1.25, "n_steps": 7, "early_stopped": True,
            "success_any": True,
            "sim_score_breakdown": {
                "predicate": 1.0, "r_dist": 0.7, "r_displace": 0.4, "r_contact": 0.1,
            },
        }

    monkeypatch.setattr("nla.training.sim_reward._run_rollout_subprocess", fake_run)

    w = SimRewardWorker(
        n_workers=1, sim_max_steps=10, cache_path=cache_path,
        scratch_dir=tmp_path / "scratch",
    )
    job = _make_job()
    out1 = w.compute([job])
    out2 = w.compute([job])
    assert n_calls["count"] == 1, "second call should hit cache"
    assert out1[0].r_sim == 1.25 and not out1[0].cached
    assert out2[0].r_sim == 1.25 and out2[0].cached
    # Verify the cache file has exactly one entry persisted.
    lines = [l for l in cache_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    cached_obj = json.loads(lines[0])
    assert cached_obj["text"] == "do the thing"


def test_worker_subprocess_error_is_swallowed(monkeypatch, tmp_path: Path):
    """A subprocess crash should not poison the whole step."""
    def fake_run(job, **kwargs):
        raise RuntimeError("rollout subprocess died")

    monkeypatch.setattr("nla.training.sim_reward._run_rollout_subprocess", fake_run)

    w = SimRewardWorker(n_workers=1, scratch_dir=tmp_path / "scratch")
    out = w.compute([_make_job()])
    assert len(out) == 1
    assert out[0].r_sim == 0.0
    assert out[0].error is not None
    assert "rollout subprocess died" in out[0].error


def test_worker_parallel_multiple_jobs(monkeypatch, tmp_path: Path):
    """Several jobs at once should each get the right scalar back, in order."""
    scores = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}

    def fake_run(job, **kwargs):
        return {
            "r_sim": scores[int(job.seed)],
            "n_steps": 5,
            "early_stopped": False,
            "success_any": bool(scores[int(job.seed)] > 2.5),
            "sim_score_breakdown": {
                "predicate": 1.0 if scores[int(job.seed)] > 2.5 else 0.0,
                "r_dist": 0.0, "r_displace": 0.0, "r_contact": 0.0,
            },
        }

    monkeypatch.setattr("nla.training.sim_reward._run_rollout_subprocess", fake_run)

    w = SimRewardWorker(n_workers=4, scratch_dir=tmp_path / "scratch")
    jobs = [_make_job(text=f"r{i}", seed=i) for i in range(4)]
    out = w.compute(jobs)
    assert [r.r_sim for r in out] == [1.0, 2.0, 3.0, 4.0]
    assert sum(r.success_any for r in out) == 2  # seeds 2 and 3


# ---------------------------------------------------------------------------
# assemble_jobs
# ---------------------------------------------------------------------------


def test_assemble_jobs_length_mismatch_raises():
    with pytest.raises(AssertionError, match="length mismatch"):
        assemble_jobs(
            rollout_texts=["a", "b"],
            steer_vecs=torch.zeros(2, 4),
            target_tasks=["t"],  # wrong length
            target_env_names=["e", "e"],
            source_ids=["s", "s"],
            seeds=[0, 1],
            sim_max_steps=10,
            placement="image_patch",
            blend=1.0,
        )


def test_assemble_jobs_happy_path():
    jobs = assemble_jobs(
        rollout_texts=["a", "b", "c"],
        steer_vecs=torch.arange(12).float().reshape(3, 4),
        target_tasks=["t1", "t2", "t1"],
        target_env_names=["env_a", "env_b", "env_a"],
        source_ids=["s1", "s2", "s3"],
        seeds=[10, 20, 30],
        sim_max_steps=15,
        placement="anchor",
        blend=0.7,
    )
    assert len(jobs) == 3
    assert jobs[1].text == "b"
    assert jobs[1].seed == 20
    assert jobs[1].target_task == "t2"
    assert jobs[1].placement == "anchor"
    assert jobs[1].blend == pytest.approx(0.7)
    assert jobs[1].sim_max_steps == 15
    assert jobs[1].steer_h.shape == (4,)
    # steer_h should be an independent copy, not a view.
    jobs[1].steer_h[0] = 999.0
    assert jobs[1].steer_h[0] == 999.0

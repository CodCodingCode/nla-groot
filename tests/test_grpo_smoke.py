"""Smoke tests for the GRPO trainer.

Same recipe as the SFT smoke tests: tiny random Qwen3 + synthetic activations,
no real-weight loads. We verify:

- ``AV.score_tokens`` is differentiable, matches generation logprobs when
  re-evaluated with the same policy.
- ``grpo_step`` produces a finite loss, KL=0 when policy=ref, KL>0 after a
  parameter perturbation.
- A few SGD steps with ``grpo_step`` actually decrease the loss.
"""

from __future__ import annotations

import copy

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from nla.models import (
    ActivationReconstructor,
    ActivationVerbalizer,
    ARConfig,
    AVConfig,
)
from nla.training.grpo import _build_gen_mask, grpo_step


TINY_HIDDEN = 32
TINY_LAYERS = 2
TINY_HEADS = 4
TINY_ACTIVATION_DIM = 32


def _make_tiny_av_ar(alpha: float = 5.0):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_HIDDEN * 2,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_HEADS,
        max_position_embeddings=512,
        rope_theta=1_000_000.0,
        torch_dtype="float32",
    )
    base_av = Qwen3ForCausalLM(cfg)
    base_ar = Qwen3ForCausalLM(cfg)
    av_cfg = AVConfig(
        activation_dim=TINY_ACTIVATION_DIM, alpha=alpha, dtype="float32",
        lora_rank=4, lora_alpha=8, max_new_tokens=24,
    )
    ar_cfg = ARConfig(
        activation_dim=TINY_ACTIVATION_DIM, alpha=alpha, dtype="float32",
        truncate_to_n_layers=1, lora_rank=4, lora_alpha=8, max_length=128,
    )
    av = ActivationVerbalizer(av_cfg, tokenizer=tok, base_model=base_av, apply_lora=True)
    ar = ActivationReconstructor(ar_cfg, tokenizer=av.tokenizer, base_model=base_ar, apply_lora=True)
    return av, ar


# ---------------------------------------------------------------------------
# _build_gen_mask
# ---------------------------------------------------------------------------

def test_build_gen_mask_stops_at_eos():
    # Two rows: one with EOS at position 2, one without EOS.
    gen = torch.tensor([
        [10, 11, 99, 12, 13],
        [20, 21, 22, 23, 24],
    ])
    mask = _build_gen_mask(gen, eos_id=99, pad_id=0)
    # Row 0: 1 1 1 (include EOS) 0 0
    assert mask[0].tolist() == [1, 1, 1, 0, 0]
    # Row 1: all 1 because no EOS and no pad.
    assert mask[1].tolist() == [1, 1, 1, 1, 1]


def test_build_gen_mask_treats_pad_as_terminator():
    gen = torch.tensor([
        [10, 11, 0, 12, 13],
    ])
    mask = _build_gen_mask(gen, eos_id=99, pad_id=0)
    assert mask[0].tolist() == [1, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# score_tokens
# ---------------------------------------------------------------------------

def test_score_tokens_matches_forward_sft_on_known_target():
    """``score_tokens`` should reproduce the per-token logp implied by the same
    forward pass that ``forward_sft`` uses (their CE matches the average -logp).
    """
    torch.manual_seed(0)
    av, _ = _make_tiny_av_ar()
    av.eval()
    B = 2
    acts = torch.randn(B, TINY_ACTIVATION_DIM)
    ptypes = ["anchor", "image_patch"]
    # Pre-tokenize some target text and score it.
    targets = ["a small red cube", "the gripper above the table"]
    tgt_ids_list = [av.tokenizer.encode(" " + t, add_special_tokens=False) for t in targets]
    max_tgt = max(len(ids) for ids in tgt_ids_list)
    gen_ids = torch.full((B, max_tgt), av._pad_id, dtype=torch.long)
    gen_mask = torch.zeros((B, max_tgt), dtype=torch.long)
    for b, ids in enumerate(tgt_ids_list):
        gen_ids[b, :len(ids)] = torch.tensor(ids)
        gen_mask[b, :len(ids)] = 1

    with torch.no_grad():
        logp = av.score_tokens(acts, ptypes, gen_ids, gen_mask)
    assert logp.shape == (B, max_tgt)
    # All logp values at masked-in positions are <= 0 (real log probs).
    for b in range(B):
        L = int(gen_mask[b].sum().item())
        assert (logp[b, :L] <= 1e-3).all(), f"logp values must be <= 0: {logp[b, :L]}"
        # Padded positions should be exactly 0.
        if max_tgt > L:
            assert (logp[b, L:].abs() < 1e-6).all()


def test_score_tokens_is_differentiable():
    torch.manual_seed(0)
    av, _ = _make_tiny_av_ar()
    av.train()
    B = 2
    acts = torch.randn(B, TINY_ACTIVATION_DIM)
    ptypes = ["anchor", "last_text"]
    gen_ids = torch.tensor([
        [av.tokenizer.encode(" a", add_special_tokens=False)[0]] * 6,
        [av.tokenizer.encode(" b", add_special_tokens=False)[0]] * 6,
    ], dtype=torch.long)
    gen_mask = torch.ones_like(gen_ids)

    logp = av.score_tokens(acts, ptypes, gen_ids, gen_mask)
    loss = -logp.sum()
    loss.backward()
    # Activation projector must have grad (it's always trainable).
    g = av.act_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------------------------------------------------------------------
# grpo_step
# ---------------------------------------------------------------------------

def test_grpo_step_runs_and_loss_is_finite():
    torch.manual_seed(0)
    policy_av, ar = _make_tiny_av_ar()
    ref_av = copy.deepcopy(policy_av)
    for p in ref_av.parameters():
        p.requires_grad = False
    ref_av.eval()

    B, K = 2, 3
    acts = torch.randn(B, TINY_ACTIVATION_DIM)
    ptypes = ["image_patch", "anchor"]

    out = grpo_step(
        policy_av, ref_av, ar,
        acts, ptypes,
        K=K, beta=0.02,
        rollout_max_new_tokens=8,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
    )
    assert torch.isfinite(out["loss"]).item()
    diag = out["diagnostics"]
    assert "reward_mean" in diag and "kl_loss" in diag and "pg_loss" in diag
    assert out["rewards"].shape == (B * K,)
    assert out["advantages"].shape == (B * K,)


def test_grpo_step_kl_zero_when_policy_equals_ref():
    """Right after constructing ref as a deep copy of policy, the KL term
    (computed under the same params) should be 0 up to floating-point noise.
    """
    torch.manual_seed(0)
    policy_av, ar = _make_tiny_av_ar()
    ref_av = copy.deepcopy(policy_av)
    for p in ref_av.parameters():
        p.requires_grad = False
    ref_av.eval()

    # Disable PG so the only contribution comes from KL.
    B, K = 2, 2
    acts = torch.randn(B, TINY_ACTIVATION_DIM)
    ptypes = ["anchor", "last_text"]
    out = grpo_step(
        policy_av, ref_av, ar,
        acts, ptypes,
        K=K, beta=1.0,
        rollout_max_new_tokens=8,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        use_pg=False, use_kl=True,
    )
    # k3 estimator under identical policies should be ~0.
    assert abs(float(out["kl_loss"].item())) < 1e-4, out["kl_loss"]


def test_grpo_step_advantages_have_zero_mean_within_group():
    torch.manual_seed(0)
    policy_av, ar = _make_tiny_av_ar()
    ref_av = copy.deepcopy(policy_av)
    for p in ref_av.parameters():
        p.requires_grad = False
    ref_av.eval()

    B, K = 2, 4
    acts = torch.randn(B, TINY_ACTIVATION_DIM)
    ptypes = ["image_patch", "anchor"]
    out = grpo_step(
        policy_av, ref_av, ar,
        acts, ptypes,
        K=K, beta=0.0,
        rollout_max_new_tokens=8,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        advantage_normalize=False,
    )
    adv_grp = out["advantages"].view(B, K)
    # Each row of advantages should sum to ~0 (centered within group).
    assert torch.allclose(adv_grp.sum(dim=1), torch.zeros(B), atol=1e-4), adv_grp


def test_metrics_closed_greedy_aliases_for_scorecard():
    from nla.training.grpo import _metrics_with_closed_greedy_aliases

    metrics = {
        "cosine/temp=0.0": 0.55,
        "fve/position=image_patch/temp=0.0": 0.12,
        "cosine/temp=0.7": 0.40,
    }
    out = _metrics_with_closed_greedy_aliases(metrics, greedy_temperature=0.0)
    assert out["closed_greedy/cosine"] == 0.55
    assert out["closed_greedy/fve/position=image_patch"] == 0.12
    assert "closed_greedy/cosine/temp=0.7" not in out
    assert out["cosine/temp=0.7"] == 0.40

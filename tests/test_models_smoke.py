"""Smoke tests for the AV and AR modules using a tiny Qwen3 config.

These tests build a synthetic Qwen3 with hidden_size=64, 4 layers, vocab=256
so we exercise the real injection/pickoff/affine-head code paths without
downloading or loading the 4B Instruct weights. The real Qwen3-4B base is
only loaded in dedicated integration scripts.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from nla.models import (
    ActivationReconstructor,
    ActivationVerbalizer,
    AR_PROMPT_TEMPLATE,
    ARConfig,
    AV_PROMPT_TEMPLATE,
    AV_SLOT_PLACEHOLDER,
    AVConfig,
    ensure_slot_token,
    find_slot_token_id,
    render_ar_prompt,
    render_av_prompt,
)
from nla.models.ar import _resolve_backbone, _truncate_layers


TINY_HIDDEN = 64
TINY_LAYERS = 4
TINY_HEADS = 4


def _make_tiny_base():
    """A fresh, randomly initialized Qwen3ForCausalLM with the real tokenizer.

    We use the real Qwen tokenizer so the slot-id lookup hits a genuine
    reserved-token in vocabulary; pads/eos behave correctly.
    """
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
    model = Qwen3ForCausalLM(cfg)
    return model, tok


# ---------------------------------------------------------------------------
# Template-level
# ---------------------------------------------------------------------------

def test_av_template_has_unique_slot():
    rendered = render_av_prompt("image_patch")
    assert rendered.count(AV_SLOT_PLACEHOLDER) == 1
    assert "image_patch" in rendered
    assert "4-5 bullet" in rendered


def test_ar_template_is_exact():
    rendered = render_ar_prompt("hello world")
    assert rendered == (
        "Summary of the following text: <text>hello world</text> <summary>"
    )


def test_ar_template_constant_format():
    assert AR_PROMPT_TEMPLATE.format(explanation="X") == (
        "Summary of the following text: <text>X</text> <summary>"
    )


def test_ensure_slot_token_adds_and_resizes():
    base, tok = _make_tiny_base()
    vocab_before = base.get_input_embeddings().weight.shape[0]
    tok_len_before = len(tok)
    tid = ensure_slot_token(tok, base, "<|act_slot|>")
    assert tid >= tok_len_before
    assert tok.encode("<|act_slot|>", add_special_tokens=False) == [tid]
    vocab_after = base.get_input_embeddings().weight.shape[0]
    assert vocab_after == vocab_before + 1
    # Idempotent: a second call must not change anything.
    tid2 = ensure_slot_token(tok, base, "<|act_slot|>")
    assert tid2 == tid
    assert base.get_input_embeddings().weight.shape[0] == vocab_after


# ---------------------------------------------------------------------------
# AV
# ---------------------------------------------------------------------------

def test_av_sft_forward_returns_finite_loss():
    base, tok = _make_tiny_base()
    cfg = AVConfig(activation_dim=128, alpha=10.0, dtype="float32")
    av = ActivationVerbalizer(cfg, tokenizer=tok, base_model=base, apply_lora=False)
    B = 2
    acts = torch.randn(B, 128)
    out = av.forward_sft(
        activations=acts,
        position_types=["image_patch", "last_text"],
        target_texts=["- scene: a table\n- target: blue cube", "- scene: floor\n- target: mug"],
    )
    assert torch.isfinite(out.loss)
    # Loss should require grad and be linked to both LM and projector parameters.
    out.loss.backward()
    assert av.act_proj.weight.grad is not None
    assert av.act_proj.weight.grad.abs().sum() > 0


def test_av_injection_overwrites_slot_embedding():
    base, tok = _make_tiny_base()
    cfg = AVConfig(activation_dim=TINY_HIDDEN, alpha=5.0, dtype="float32")
    av = ActivationVerbalizer(cfg, tokenizer=tok, base_model=base, apply_lora=False)
    acts = torch.randn(1, TINY_HIDDEN)
    input_ids, attn, _, slot_indices = av._tokenize_prompts(["anchor"], target_texts=None)
    embeds_no_inj = av._embed()(input_ids)
    embeds_with_inj = av._embed_with_injection(input_ids, attn, slot_indices, acts)
    s = slot_indices[0]
    # All non-slot positions should be unchanged.
    mask = torch.ones(embeds_no_inj.shape[1], dtype=torch.bool)
    mask[s] = False
    assert torch.allclose(embeds_with_inj[0, mask], embeds_no_inj[0, mask])
    # Slot position must change unless the random projection happens to land
    # exactly on the original embedding (vanishingly unlikely).
    assert not torch.allclose(embeds_with_inj[0, s], embeds_no_inj[0, s])
    # Magnitude check: injected vector has norm ~= alpha.
    inj_norm = embeds_with_inj[0, s].norm().item()
    assert abs(inj_norm - cfg.alpha) / cfg.alpha < 1e-4


def test_av_generate_returns_text():
    base, tok = _make_tiny_base()
    cfg = AVConfig(
        activation_dim=TINY_HIDDEN,
        alpha=5.0,
        dtype="float32",
        max_new_tokens=8,
    )
    av = ActivationVerbalizer(cfg, tokenizer=tok, base_model=base, apply_lora=False)
    acts = torch.randn(2, TINY_HIDDEN)
    out = av.generate(
        activations=acts,
        position_types=["image_patch", "last_text"],
        do_sample=False,
    )
    assert "text" in out and len(out["text"]) == 2
    assert "token_ids" in out and out["token_ids"].shape[0] == 2


def test_av_lora_wrapping_works():
    base, tok = _make_tiny_base()
    cfg = AVConfig(
        activation_dim=TINY_HIDDEN, alpha=5.0, dtype="float32",
        lora_rank=4, lora_alpha=8,
    )
    av = ActivationVerbalizer(cfg, tokenizer=tok, base_model=base, apply_lora=True)
    # Should still be able to forward + backward.
    acts = torch.randn(1, TINY_HIDDEN)
    out = av.forward_sft(
        activations=acts, position_types=["anchor"],
        target_texts=["- scene: ok"],
    )
    out.loss.backward()
    # Verify some LoRA param has a gradient.
    grads = [p.grad for n, p in av.named_parameters() if "lora_" in n and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# AR
# ---------------------------------------------------------------------------

def test_ar_truncation_changes_layer_count():
    base, _ = _make_tiny_base()
    backbone = _resolve_backbone(base)
    n0 = len(backbone.layers)
    assert n0 == TINY_LAYERS
    _truncate_layers(base, keep=2)
    assert len(_resolve_backbone(base).layers) == 2


def test_ar_forward_shape_and_scaling():
    base, tok = _make_tiny_base()
    cfg = ARConfig(
        activation_dim=128, alpha=10.0, truncate_to_n_layers=2,
        lora_rank=4, lora_alpha=8, dtype="float32",
    )
    ar = ActivationReconstructor(cfg, tokenizer=tok, base_model=base, apply_lora=False)
    pred = ar(["this is a test", "another short explanation"])
    assert pred.shape == (2, 128)
    assert torch.isfinite(pred).all()


def test_ar_sft_loss_decreases_on_overfit():
    base, tok = _make_tiny_base()
    cfg = ARConfig(
        activation_dim=32, alpha=1.0, truncate_to_n_layers=2,
        lora_rank=4, lora_alpha=8, dtype="float32",
    )
    ar = ActivationReconstructor(cfg, tokenizer=tok, base_model=base, apply_lora=False)
    explanations = ["- scene: table\n- target: blue cube"]
    target = torch.randn(1, 32)
    optim = torch.optim.Adam(ar.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(20):
        optim.zero_grad()
        loss, _ = ar.forward_sft(explanations, target)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]


def test_ar_predict_unscales_by_alpha():
    base, tok = _make_tiny_base()
    cfg = ARConfig(
        activation_dim=16, alpha=7.5, truncate_to_n_layers=2, dtype="float32",
    )
    ar = ActivationReconstructor(cfg, tokenizer=tok, base_model=base, apply_lora=False)
    explanations = ["x"]
    scaled = ar.forward(explanations)
    unscaled = ar.predict(explanations, unscale=True)
    assert torch.allclose(unscaled.float(), scaled.detach().float() * cfg.alpha, atol=1e-4)


def test_ar_lora_wrapping_works():
    base, tok = _make_tiny_base()
    cfg = ARConfig(
        activation_dim=16, alpha=1.0, truncate_to_n_layers=2,
        lora_rank=4, lora_alpha=8, dtype="float32",
    )
    ar = ActivationReconstructor(cfg, tokenizer=tok, base_model=base, apply_lora=True)
    target = torch.randn(1, 16)
    loss, _ = ar.forward_sft(["sample"], target)
    loss.backward()
    grads = [p.grad for n, p in ar.named_parameters() if "lora_" in n and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)

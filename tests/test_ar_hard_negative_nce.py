"""Hard-negative InfoNCE tests for ``ActivationReconstructor.forward_sft``.

Workstream D1: when ``negative_explanations`` is ``None`` the contrastive
codepath must remain byte-identical to the legacy in-batch-only objective.
When provided, the augmented (B, B+K_neg) similarity matrix must change the
loss in the expected direction (negatives suck probability mass from the
positive).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from nla.models import ActivationReconstructor, ARConfig


TINY_HIDDEN = 32
TINY_LAYERS = 2
TINY_HEADS = 4


def _make_tiny_ar(activation_dim: int = 16, alpha: float = 5.0):
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
    base = Qwen3ForCausalLM(cfg)
    ar_cfg = ARConfig(
        activation_dim=activation_dim,
        alpha=alpha,
        truncate_to_n_layers=1,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.0,                 # determinism for byte-identical test
        dtype="float32",
        nce_temperature=0.1,
    )
    ar = ActivationReconstructor(ar_cfg, tokenizer=tok, base_model=base, apply_lora=False)
    ar.eval()                              # also for determinism
    return ar


def test_forward_sft_no_negatives_byte_identical():
    """``negative_explanations=None`` must produce *exactly* the legacy NCE loss.

    We compare the new code path against an inlined re-implementation of the
    pre-D1 contrastive computation on the same inputs. They must match
    bit-for-bit (the new branch only kicks in when negs are not None).
    """
    torch.manual_seed(0)
    ar = _make_tiny_ar(activation_dim=16, alpha=5.0)
    explanations = [
        "- scene: tabletop\n- target: blue cube",
        "- scene: floor\n- target: cup",
        "- scene: drawer\n- target: cloth",
        "- scene: shelf\n- target: marker",
    ]
    target = torch.randn(len(explanations), 16)

    with torch.no_grad():
        # New code path with no negatives.
        mse_new, nce_new, pred_new = ar.forward_sft(
            explanations, target, return_nce=True, negative_explanations=None,
        )
        # Re-implement the pre-D1 inner loop (what the legacy code did).
        pred_legacy = ar.forward(explanations, device=target.device)
        target_scaled = (target / ar.cfg.alpha).to(pred_legacy.dtype)
        sims = F.cosine_similarity(
            pred_legacy.unsqueeze(1),
            target_scaled.unsqueeze(0),
            dim=-1,
        )
        sims = sims / max(1e-6, ar.cfg.nce_temperature)
        labels = torch.arange(pred_legacy.shape[0], device=pred_legacy.device)
        nce_legacy = F.cross_entropy(sims.float(), labels)
        mse_legacy = F.mse_loss(pred_legacy, target_scaled)

    assert torch.equal(pred_new, pred_legacy)
    assert torch.equal(mse_new, mse_legacy)
    assert torch.equal(nce_new, nce_legacy)


def test_forward_sft_with_negatives_changes_loss():
    """A "known-bad" negative (== anchor caption) must INCREASE the NCE loss.

    Intuition: when the hard negative is the anchor caption itself,
    ``cos(pred[i], pred_neg[i,k]) == cos(pred[i], pred[i]) ≈ 1``. After
    temperature scaling that lifts those columns above the diagonal value,
    so the softmax mass concentrates on the negatives and the cross-entropy
    of the diagonal label rises.
    """
    torch.manual_seed(0)
    ar = _make_tiny_ar(activation_dim=16, alpha=5.0)
    explanations = [
        "- scene: tabletop\n- target: blue cube",
        "- scene: floor\n- target: cup",
        "- scene: drawer\n- target: cloth",
        "- scene: shelf\n- target: marker",
    ]
    target = torch.randn(len(explanations), 16)
    # Each anchor's "hard" negative is its own caption -> guaranteed to score
    # cos == 1 against pred[i], making it a strict-upper-bound competitor for
    # the positive's logit at the diagonal.
    negative_explanations = [[expl, expl] for expl in explanations]

    with torch.no_grad():
        _mse_no, nce_no, _pred_no = ar.forward_sft(
            explanations, target, return_nce=True, negative_explanations=None,
        )
        _mse_yes, nce_yes, _pred_yes = ar.forward_sft(
            explanations, target, return_nce=True,
            negative_explanations=negative_explanations,
        )
    # Adding a column whose logit equals (or nearly equals) the anchor's own
    # pred-pred similarity is a strict upper bound on the anchor self-cos
    # (cos == 1 / temp = 10), well above any cos(pred, target) in the
    # randomly initialized backbone. So the per-row positive softmax mass
    # falls and CE rises.
    assert nce_yes.item() > nce_no.item() + 1e-3, (
        f"hard-neg NCE {nce_yes.item()} should exceed in-batch-only "
        f"{nce_no.item()} by a noticeable margin"
    )


def test_negatives_shape():
    """Verify the augmented similarity matrix has shape (B, B+K_neg).

    We monkey-patch ``cross_entropy`` to capture the logits tensor it sees.
    """
    torch.manual_seed(0)
    ar = _make_tiny_ar(activation_dim=16, alpha=5.0)
    explanations = ["- a", "- b", "- c"]                       # B=3
    K_neg = 2
    negative_explanations = [["- x", "- y"] for _ in explanations]
    target = torch.randn(len(explanations), 16)

    captured = {}
    real_ce = F.cross_entropy

    def _capturing_ce(input, target, *args, **kwargs):
        captured["shape"] = tuple(input.shape)
        return real_ce(input, target, *args, **kwargs)

    with torch.no_grad():
        torch.nn.functional.cross_entropy = _capturing_ce
        try:
            ar.forward_sft(
                explanations, target, return_nce=True,
                negative_explanations=negative_explanations,
            )
        finally:
            torch.nn.functional.cross_entropy = real_ce

    B = len(explanations)
    assert "shape" in captured, "cross_entropy was not invoked"
    assert captured["shape"] == (B, B + K_neg), (
        f"expected augmented logits shape ({B}, {B + K_neg}); "
        f"got {captured['shape']}"
    )


def test_negative_explanations_rectangular_shape_validated():
    """Non-rectangular ``negative_explanations`` must raise loudly.

    Mixed K_neg per row would silently mis-align the cat-along-dim-1 step;
    we require the user to provide a rectangular shape and we check it.
    """
    import pytest

    torch.manual_seed(0)
    ar = _make_tiny_ar(activation_dim=16, alpha=5.0)
    explanations = ["- a", "- b"]
    target = torch.randn(2, 16)
    negative_explanations = [["- x", "- y"], ["- only one"]]

    with pytest.raises(ValueError, match="rectangular"):
        ar.forward_sft(
            explanations, target, return_nce=True,
            negative_explanations=negative_explanations,
        )


def test_negative_explanations_row_count_validated():
    """``len(negative_explanations) != B`` must raise."""
    import pytest

    torch.manual_seed(0)
    ar = _make_tiny_ar(activation_dim=16, alpha=5.0)
    explanations = ["- a", "- b", "- c"]
    target = torch.randn(3, 16)
    # Wrong row count.
    negative_explanations = [["- x"], ["- y"]]

    with pytest.raises(ValueError, match="one list per batch row"):
        ar.forward_sft(
            explanations, target, return_nce=True,
            negative_explanations=negative_explanations,
        )

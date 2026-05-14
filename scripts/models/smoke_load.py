#!/usr/bin/env python
"""End-to-end smoke load of AV + AR on the real Qwen3-4B-Instruct-2507.

Verifies that:
  - Both models load on the GPU
  - AV.forward_sft yields a finite loss with a real activation
  - AV.generate produces non-empty text
  - AR.forward predicts the right shape
  - AR.forward_sft yields a finite scaled-MSE loss

Run::

    PYTHONPATH=src python scripts/models/smoke_load.py
"""

from __future__ import annotations

import argparse
import logging
import time

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--alpha", type=float, default=196.15)
    p.add_argument("--ar-layers", type=int, default=10)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("smoke")

    from nla.models import (
        ActivationReconstructor,
        ActivationVerbalizer,
        ARConfig,
        AVConfig,
    )

    log.info("Loading AV (%s, dtype=%s)...", args.base_model, args.dtype)
    av_cfg = AVConfig(
        base_model=args.base_model,
        activation_dim=2048,
        alpha=args.alpha,
        dtype=args.dtype,
        max_new_tokens=64,
    )
    t0 = time.time()
    av = ActivationVerbalizer(av_cfg).to(args.device)
    log.info("AV loaded in %.1fs (slot id=%d, str=%r)",
             time.time() - t0, av.cfg.slot_token_id, av.cfg.slot_token_str)

    log.info("Loading AR (%s, %d layers)...", args.base_model, args.ar_layers)
    ar_cfg = ARConfig(
        base_model=args.base_model,
        activation_dim=2048,
        alpha=args.alpha,
        truncate_to_n_layers=args.ar_layers,
        dtype=args.dtype,
    )
    t0 = time.time()
    ar = ActivationReconstructor(ar_cfg, tokenizer=av.tokenizer).to(args.device)
    log.info("AR loaded in %.1fs", time.time() - t0)

    B = 2
    acts = torch.randn(B, 2048, device=args.device) * args.alpha
    pos_types = ["image_patch", "last_text"]
    targets = [
        "- scene: white round table with toys and a green bowl.\n"
        "- target: blue cube near the bowl rim.\n"
        "- gripper: open and approaching from the left.\n"
        "- plan: reach toward blue cube.\n"
        "- image_region: upper-right region containing cube and bowl rim.",
        "- scene: white round table with a wooden tray.\n"
        "- target: blue cube on the tray.\n"
        "- distractor: yellow banana, orange fruit.\n"
        "- spatial: bowl is left of cube; gripper approaches from left.\n"
        "- language: instruction read; goal = grasp blue cube and place in bowl.",
    ]

    log.info("AV forward_sft on B=%d ...", B)
    t0 = time.time()
    out = av.forward_sft(activations=acts, position_types=pos_types, target_texts=targets)
    log.info("  loss=%.4f  (%.1fs)", out.loss.item(), time.time() - t0)
    assert torch.isfinite(out.loss)

    log.info("AV generate on B=%d ...", B)
    t0 = time.time()
    gen = av.generate(activations=acts, position_types=pos_types, max_new_tokens=48, do_sample=False)
    log.info("  generation took %.1fs", time.time() - t0)
    for i, txt in enumerate(gen["text"]):
        log.info("  sample %d: %s", i, txt[:160].replace("\n", " ⏎ "))

    log.info("AR forward on B=%d ...", B)
    t0 = time.time()
    pred = ar([targets[0], targets[1]])
    log.info("  pred shape=%s  (%.1fs)", tuple(pred.shape), time.time() - t0)
    assert pred.shape == (2, 2048)

    log.info("AR forward_sft on B=%d ...", B)
    t0 = time.time()
    loss, _ = ar.forward_sft([targets[0], targets[1]], acts)
    log.info("  scaled MSE loss=%.4f  (%.1fs)", loss.item(), time.time() - t0)
    assert torch.isfinite(loss)

    log.info("Smoke PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

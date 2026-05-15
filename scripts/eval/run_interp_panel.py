#!/usr/bin/env python
"""Intervention evidence runner for the interpretability eval.

Reads the frozen ``eval_cases.jsonl`` produced by ``build_eval_cases.py`` and,
for each case, produces a deterministic evidence package:

    - **baseline_output**:  AV explanation on the original activation.
    - **edited_output**:    AV explanation on a counterfactually edited activation.
    - **control_output**:   AV explanation on a random-direction edit of the same
                            norm (matched-magnitude null).
    - **auto_metrics_inputs**: vectors needed by ``score_panel.py`` to compute
                               direction_match, effect_size, stability, and
                               (if AR available) AR reconstruction deltas.

Edit kinds (selected by ``case.edit_kind`` from the frozen set, or overridden
on the command line):

    * ``noise``       : ``h_edit = h + alpha * eps``, ``eps`` ~ N(0, I) scaled
                        to ``alpha * ||h||``.
    * ``swap``        : ``h_edit = h_other`` taken from a different case in the
                        same eval set (deterministic pair via ``--swap-seed``).
    * ``null``        : ``h_edit = 0`` (zeroing baseline).
    * ``paraphrase``  : no activation edit; ``h_edit = h``. The contrast comes
                        from re-sampling AV with a different seed; this is the
                        explanation-stability baseline (NLA paper-style).

Output schema (one row per case in ``panel_rows.jsonl``)::

    {
      "case_id": "...",
      "position_type": "...",
      "baseline_text":  "<AV explanation>",
      "edited_text":    "<AV on edited activation>",
      "control_text":   "<AV on random edit>",
      "auto_metrics_inputs": {
         "h_norm":             float,    # ||h||
         "edit_norm":          float,    # ||h_edit||
         "control_norm":       float,    # ||h_ctrl||
         "edit_delta_norm":    float,    # ||h_edit - h||
         "control_delta_norm": float,    # ||h_ctrl - h||
         "ar_present":         bool,
         "ar_baseline_mse":    float | null,  # MSE(AR(baseline_text), h)
         "ar_edited_mse":      float | null,  # MSE(AR(edited_text), h_edit)
         "ar_control_mse":     float | null,  # MSE(AR(control_text), h_ctrl)
         "ar_baseline_cos":    float | null,
         "ar_edited_cos":      float | null,
         "ar_control_cos":     float | null
      },
      "seed_stability_texts": ["<AV resample 1>", "<AV resample 2>", ...],
      "intervention_spec": {"edit_kind": "...", "edit_strength": ...}
    }

The script is **deterministic given the same seed and the same checkpoint**.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("nla.eval.panel")


def _set_seed(seed: int) -> None:
    import random as _random

    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _apply_edit(
    h: torch.Tensor,
    *,
    edit_kind: str,
    edit_strength: float,
    other_h: torch.Tensor | None,
    rng: torch.Generator,
) -> torch.Tensor:
    """Return ``h_edit`` for the requested counterfactual edit.

    All edits preserve the [1, H] shape of the activation slice.
    """
    h = h.clone()
    if edit_kind == "noise":
        eps = torch.randn(h.shape, generator=rng, device=h.device, dtype=h.dtype)
        eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return h + edit_strength * h.norm(dim=-1, keepdim=True) * eps
    if edit_kind == "swap":
        if other_h is None:
            raise ValueError("edit_kind='swap' requires a partner activation.")
        return other_h.to(h.device, dtype=h.dtype).clone()
    if edit_kind == "null":
        return torch.zeros_like(h)
    if edit_kind == "paraphrase":
        return h
    raise ValueError(f"Unknown edit_kind: {edit_kind}")


def _control_edit(
    h: torch.Tensor,
    *,
    edit_strength: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Random-direction edit with the same scaled magnitude as the noise edit.

    Used as the matched-magnitude null hypothesis: any change AV produces here
    is **not** explainable by the targeted edit, only by perturbation noise.
    """
    eps = torch.randn(h.shape, generator=rng, device=h.device, dtype=h.dtype)
    eps = eps / eps.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return h + edit_strength * h.norm(dim=-1, keepdim=True) * eps


def _scaled_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().item())


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cases", required=True, help="eval_cases.jsonl from build_eval_cases.py")
    p.add_argument("--activations-root", required=True)
    p.add_argument("--av-dir", required=True, help="Path to SFT av/ checkpoint.")
    p.add_argument(
        "--ar-dir",
        default=None,
        help="Optional path to SFT ar/ checkpoint. If present, AR reconstruction "
             "deltas are added to auto_metrics_inputs.",
    )
    p.add_argument("--out", required=True, help="panel_rows.jsonl output path.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="AV decode length (default 80; lower => faster, less detail).",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy decode for the primary baseline/edited/control samples.",
    )
    p.add_argument(
        "--n-stability-samples",
        type=int,
        default=2,
        help="Extra paraphrase samples (sampled, not greedy) for the seed-stability metric.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--swap-seed",
        type=int,
        default=1234,
        help="Seed for choosing per-case swap partners (deterministic).",
    )
    p.add_argument(
        "--edit-kind-override",
        default=None,
        help="Override the per-case edit_kind for ablations (otherwise uses case value).",
    )
    p.add_argument(
        "--edit-strength-override",
        type=float,
        default=None,
        help="Override the per-case edit_strength for ablations.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cases_path = Path(args.cases)
    out_path = Path(args.out)
    if not cases_path.is_file():
        logger.error("Cases file not found: %s", cases_path)
        return 2

    # Lazy imports.
    from nla.extraction.storage import ActivationShardReader
    from nla.training.checkpoint import load_av_from_sft, load_ar_from_sft

    _set_seed(args.seed)
    cases = _load_cases(cases_path)
    if not cases:
        logger.error("No cases in %s", cases_path)
        return 2

    reader = ActivationShardReader(Path(args.activations_root))
    av = load_av_from_sft(args.av_dir, device=args.device, freeze=True)
    ar = None
    if args.ar_dir:
        ar = load_ar_from_sft(args.ar_dir, device=args.device, freeze=True)

    # Pre-load activation slices so we can pick swap partners deterministically.
    slices: dict[str, torch.Tensor] = {}
    for c in cases:
        blob = reader.get(c["example_id"])
        feat = blob["features"]
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat[0]
        idx = int(c["token_index"])
        slices[c["case_id"]] = feat[idx : idx + 1].to(args.device, dtype=torch.float32)

    swap_rng = torch.Generator(device="cpu")
    swap_rng.manual_seed(args.swap_seed)
    case_ids = [c["case_id"] for c in cases]
    perm = torch.randperm(len(case_ids), generator=swap_rng).tolist()
    swap_partner = {case_ids[i]: case_ids[perm[i]] for i in range(len(case_ids))}

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as fout:
        for c in cases:
            case_id = c["case_id"]
            ptype = c["position_type"]
            edit_kind = args.edit_kind_override or c.get("edit_kind", "noise")
            edit_strength = (
                float(args.edit_strength_override)
                if args.edit_strength_override is not None
                else float(c.get("edit_strength", 0.5))
            )

            h = slices[case_id]
            other = slices.get(swap_partner.get(case_id, case_id))
            h_edit = _apply_edit(
                h,
                edit_kind=edit_kind,
                edit_strength=edit_strength,
                other_h=other,
                rng=rng,
            )
            h_ctrl = _control_edit(h, edit_strength=edit_strength, rng=rng)

            do_sample = not args.greedy
            with torch.no_grad():
                base_gen = av.generate(
                    h,
                    [ptype],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=do_sample,
                )
                edit_gen = av.generate(
                    h_edit,
                    [ptype],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=do_sample,
                )
                ctrl_gen = av.generate(
                    h_ctrl,
                    [ptype],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=do_sample,
                )

                stability_texts: list[str] = []
                for _ in range(max(0, args.n_stability_samples)):
                    g = av.generate(
                        h,
                        [ptype],
                        max_new_tokens=args.max_new_tokens,
                        temperature=max(0.4, args.temperature),
                        top_p=args.top_p,
                        do_sample=True,
                    )
                    stability_texts.append((g["text"][0] or "").strip())

            ar_inputs = {"ar_present": ar is not None}
            if ar is not None:
                pred_b = ar.predict([base_gen["text"][0]]).float()
                pred_e = ar.predict([edit_gen["text"][0]]).float()
                pred_c = ar.predict([ctrl_gen["text"][0]]).float()
                ar_inputs.update(
                    {
                        "ar_baseline_mse": _scaled_mse(pred_b, h),
                        "ar_edited_mse": _scaled_mse(pred_e, h_edit),
                        "ar_control_mse": _scaled_mse(pred_c, h_ctrl),
                        "ar_baseline_cos": float(
                            torch.nn.functional.cosine_similarity(
                                pred_b.flatten().unsqueeze(0),
                                h.flatten().unsqueeze(0),
                            ).item()
                        ),
                        "ar_edited_cos": float(
                            torch.nn.functional.cosine_similarity(
                                pred_e.flatten().unsqueeze(0),
                                h_edit.flatten().unsqueeze(0),
                            ).item()
                        ),
                        "ar_control_cos": float(
                            torch.nn.functional.cosine_similarity(
                                pred_c.flatten().unsqueeze(0),
                                h_ctrl.flatten().unsqueeze(0),
                            ).item()
                        ),
                    }
                )
            else:
                ar_inputs.update(
                    {
                        "ar_baseline_mse": None,
                        "ar_edited_mse": None,
                        "ar_control_mse": None,
                        "ar_baseline_cos": None,
                        "ar_edited_cos": None,
                        "ar_control_cos": None,
                    }
                )

            row = {
                "case_id": case_id,
                "position_type": ptype,
                "baseline_text": (base_gen["text"][0] or "").strip(),
                "edited_text": (edit_gen["text"][0] or "").strip(),
                "control_text": (ctrl_gen["text"][0] or "").strip(),
                "seed_stability_texts": stability_texts,
                "auto_metrics_inputs": {
                    "h_norm": float(h.norm().item()),
                    "edit_norm": float(h_edit.norm().item()),
                    "control_norm": float(h_ctrl.norm().item()),
                    "edit_delta_norm": float((h_edit - h).norm().item()),
                    "control_delta_norm": float((h_ctrl - h).norm().item()),
                    **ar_inputs,
                },
                "intervention_spec": {
                    "edit_kind": edit_kind,
                    "edit_strength": edit_strength,
                    "swap_partner": swap_partner.get(case_id),
                },
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_written += 1
            if n_written % 5 == 0:
                logger.info("  panel rows written: %d / %d", n_written, len(cases))

    logger.info("Wrote %d panel rows to %s", n_written, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Freeze a stratified, deterministic interpretability eval set.

Reads an activation dump (produced by ``scripts/extraction/run_extract.py``) and
writes an immutable ``eval_cases.jsonl`` file that downstream eval scripts
(``run_interp_panel.py``, ``run_llm_judge.py``, ``score_panel.py``) read.

Each case row is the **smallest atom of an interpretability test**: one
activation at one token position, plus a pre-registered hypothesis describing
what behavioral/latent change we expect from a counterfactual edit.

Schema (one row per case)::

    {
      "case_id":         "case_000007",
      "example_id":      "traj000001_step000042",
      "episode_index":   1,
      "step_index":      42,
      "position_type":   "last_text" | "image_patch" | "anchor",
      "token_index":     17,
      "seq_len":         128,
      "task_text":       "...optional original instruction...",
      "hypothesis":      "free-text predicted change (filled by hand or template)",
      "expected_direction": "+|-|none",
      "edit_kind":       "noise|swap|paraphrase|null",
      "edit_strength":   0.5,
      "control_kind":    "random_unit_vector",
      "metadata":        {...passthrough info...}
    }

The script is **deterministic given a seed** and **stratifies** across position
types so a small (10-20) eval set still contains a balanced mix.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("nla.eval.build")


DEFAULT_POSITION_MIX = (
    ("last_text", 0.4),
    ("image_patch", 0.4),
    ("anchor", 0.2),
)

DEFAULT_HYPOTHESIS_TEMPLATES = {
    "last_text": (
        "AV explanation should reference the language instruction or the "
        "next-action plan. Editing the explanation to a different action "
        "should shift behavior toward the edited action."
    ),
    "image_patch": (
        "AV explanation should reference scene/object content visible at "
        "this image patch. Editing the referenced object should change the "
        "spatial focus of the reconstructed activation."
    ),
    "anchor": (
        "AV explanation should summarize the model's pre-action plan. "
        "Editing the planned target object should change the predicted "
        "action distribution."
    ),
}


def _stratified_sample(
    records: list[dict[str, Any]],
    *,
    n: int,
    position_mix: list[tuple[str, float]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Pick ``n`` records, stratifying by ``position_type``.

    Each record must already have a ``position_type`` and ``token_index``
    chosen (we do this in the caller using the same TokenPositionSampler used
    in training, so eval positions look like training positions).
    """
    by_type: dict[str, list[dict[str, Any]]] = {pt: [] for pt, _ in position_mix}
    for r in records:
        pt = r.get("position_type")
        if pt in by_type:
            by_type[pt].append(r)

    # Compute target count per stratum.
    targets: dict[str, int] = {}
    remaining = n
    for pt, frac in position_mix[:-1]:
        k = max(0, int(round(frac * n)))
        targets[pt] = k
        remaining -= k
    targets[position_mix[-1][0]] = max(0, remaining)

    out: list[dict[str, Any]] = []
    for pt, k in targets.items():
        pool = by_type.get(pt, [])
        rng.shuffle(pool)
        out.extend(pool[:k])

    # If any stratum was short, fill from leftover.
    if len(out) < n:
        leftover: list[dict[str, Any]] = []
        for pt, pool in by_type.items():
            already = {id(x) for x in out}
            leftover.extend(x for x in pool if id(x) not in already)
        rng.shuffle(leftover)
        out.extend(leftover[: n - len(out)])

    return out[:n]


def _record_to_case(
    rec: dict[str, Any],
    *,
    case_idx: int,
    edit_kind: str,
    edit_strength: float,
    hypothesis_templates: dict[str, str],
) -> dict[str, Any]:
    pt = rec["position_type"]
    return {
        "case_id": f"case_{case_idx:06d}",
        "example_id": rec["example_id"],
        "episode_index": rec.get("episode_index"),
        "step_index": rec.get("step_index"),
        "position_type": pt,
        "token_index": int(rec["token_index"]),
        "seq_len": int(rec["seq_len"]),
        "task_text": rec.get("task_text"),
        "hypothesis": hypothesis_templates.get(pt, ""),
        "expected_direction": "+",
        "edit_kind": edit_kind,
        "edit_strength": float(edit_strength),
        "control_kind": "random_unit_vector",
        "metadata": {
            "embodiment_tag": rec.get("embodiment_tag"),
            "task_index": rec.get("task_index"),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--activations-root",
        required=True,
        help="Activation dump directory (must contain index.jsonl + shard_*).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Path to write eval_cases.jsonl (frozen).",
    )
    p.add_argument(
        "--n-cases",
        type=int,
        default=16,
        help="Total number of cases to freeze (default: 16).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for sampling.",
    )
    p.add_argument(
        "--episode-allow",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these episode indices (default: all).",
    )
    p.add_argument(
        "--edit-kind",
        choices=["noise", "swap", "paraphrase", "null"],
        default="noise",
        help="Default counterfactual edit kind for all cases.",
    )
    p.add_argument(
        "--edit-strength",
        type=float,
        default=0.5,
        help="Default edit strength alpha (interpretation depends on edit-kind).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing eval_cases.jsonl.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        logger.error(
            "%s already exists. Frozen eval sets must not change between runs. "
            "Pass --overwrite if you really mean to replace it.",
            out_path,
        )
        return 2

    # Lazy imports so --help works without the heavy stack.
    from nla.extraction.storage import ActivationShardReader
    from nla.training.sampling import TokenPositionSampler

    root = Path(args.activations_root)
    if not root.is_dir():
        logger.error("activations-root not a directory: %s", root)
        return 2

    reader = ActivationShardReader(root)
    sampler = TokenPositionSampler(seed=args.seed)
    rng = random.Random(args.seed)

    allow_ep: set[int] | None = (
        set(int(x) for x in args.episode_allow) if args.episode_allow else None
    )

    candidates: list[dict[str, Any]] = []
    for rec in reader.records:
        if rec.episode_index is None or rec.step_index is None:
            continue
        if allow_ep is not None and rec.episode_index not in allow_ep:
            continue
        # Materialize attn/img masks so we can pick a real token position.
        try:
            blob = reader.get(rec.example_id)
        except Exception as e:
            logger.warning("Skipping %s (read error: %s)", rec.example_id, e)
            continue
        attn = blob["attention_mask"].reshape(-1)
        img = blob["image_mask"].reshape(-1)
        ptype, tok_idx = sampler.sample(attn.cpu(), img.cpu())
        candidates.append(
            {
                "example_id": rec.example_id,
                "episode_index": rec.episode_index,
                "step_index": rec.step_index,
                "task_index": rec.task_index,
                "task_text": rec.task_text,
                "embodiment_tag": rec.embodiment_tag,
                "position_type": ptype,
                "token_index": int(tok_idx),
                "seq_len": int(rec.seq_len),
            }
        )

    if not candidates:
        logger.error("No usable examples in %s.", root)
        return 2

    chosen = _stratified_sample(
        candidates,
        n=args.n_cases,
        position_mix=list(DEFAULT_POSITION_MIX),
        rng=rng,
    )

    cases = [
        _record_to_case(
            rec,
            case_idx=i,
            edit_kind=args.edit_kind,
            edit_strength=args.edit_strength,
            hypothesis_templates=DEFAULT_HYPOTHESIS_TEMPLATES,
        )
        for i, rec in enumerate(chosen)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    by_pt: dict[str, int] = {}
    for c in cases:
        by_pt[c["position_type"]] = by_pt.get(c["position_type"], 0) + 1
    logger.info("Wrote %d cases to %s", len(cases), out_path)
    logger.info("Stratification: %s", by_pt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

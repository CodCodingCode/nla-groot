#!/usr/bin/env python
"""Choose the next iteration's mining CLI flags from a failed audit.json.

Reads a scorecard produced by ``audit_cf_pairs.py`` and writes a
space-separated set of flags to stdout that should be appended to the next
``mine_grpo_counterfactual_pairs.py`` invocation.

Decision rules (priority order):

  1. If ``counterfactual_fraction`` is outside [0.45, 0.55], adjust
     ``--matching-fraction`` to push it back. Mining draws ``rng.random() <
     matching_fraction`` for the "matching" half, so matching_fraction is
     just (1 - cf_target).

  2. If max rows-per-episode is >= 3x the mean, propose a tighter
     ``--max-per-episode`` (clamp into a sensible band).

  3. If any source-task share > 25% (gate cap), propose
     ``--max-per-source-task = floor(0.25 * n_rows)``. This requires the
     miner to support the flag; if the audit shows source skew but the
     previous mine did NOT pass --max-per-source-task, we print the
     NEEDS_MINER_EDIT sentinel so the orchestrator can fail fast and ask
     the user. The audit JSON path is preserved as ``--max-per-source-task``
     value as a hint regardless of whether the flag actually exists.

  4. If any target-task share > 25%, propose ``--balance-target-counts`` (a
     weighted target sampler that inverse-weights by current emit count).
     Same NEEDS_MINER_EDIT mechanism.

Output format:

    NEXT_FLAGS:<flags>
    NOTES:<comma-separated rationales>
    NEEDS_EDIT:<csv of required miner edits, or empty>

Exit code is always 0 (this is a pure helper). The orchestrator inspects the
NEEDS_EDIT line to decide whether to keep iterating or hand off to the user.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _round_match_frac(cf_target: float) -> float:
    return max(0.0, min(1.0, round(1.0 - cf_target, 2)))


def tune(audit: dict, *, prev_flags: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Return (next_flags_list, notes, needs_edits).

    ``prev_flags`` is the kwargs map the previous iteration used (string ->
    string). We respect any flag the user already pinned manually.
    """
    flags: dict[str, str] = dict(prev_flags)
    notes: list[str] = []
    needs: list[str] = []

    cf = float(audit.get("counterfactual_fraction", 0.0))
    src_pct = audit.get("source_task_pct", {}) or {}
    tgt_pct = audit.get("target_task_pct", {}) or {}
    n_rows = int(audit.get("n_rows", 0))
    ec = audit.get("episode_coverage", {}) or {}

    # Rule 1: counterfactual fraction.
    if not (0.45 <= cf <= 0.55):
        # Recenter to 0.5.
        new_mf = _round_match_frac(0.5)
        flags["--matching-fraction"] = f"{new_mf:.2f}"
        notes.append(f"cf={cf:.3f} out of band -> set --matching-fraction={new_mf:.2f}")

    # Rule 2: episode hog cap.
    mean_rpe = float(ec.get("mean_rows_per_episode", 0.0))
    max_rpe = int(ec.get("max_rows_per_episode", 0))
    if mean_rpe > 0 and max_rpe >= max(3, 3 * mean_rpe):
        # Target ~1.5x the mean, clamped to [1, 8] for small smokes.
        target = max(1, min(8, math.ceil(1.5 * mean_rpe)))
        cur = prev_flags.get("--max-per-episode")
        if cur is None or int(target) < int(cur):
            flags["--max-per-episode"] = str(int(target))
            notes.append(
                f"max_rows_per_episode={max_rpe} vs mean={mean_rpe:.2f} -> "
                f"--max-per-episode={target}"
            )

    # Rule 3: source-task skew (head trim).
    over_src = [t for t, p in src_pct.items() if p > 0.25]
    if over_src and n_rows > 0:
        cap = max(1, int(math.floor(0.25 * n_rows)))
        flags["--max-per-source-task"] = str(cap)
        notes.append(
            f"source skew on {over_src} -> --max-per-source-task={cap}"
        )
        needs.append("max_per_source_task")

    # Rule 4: target-task skew (weighted sampler).
    over_tgt = [t for t, p in tgt_pct.items() if p > 0.25]
    if over_tgt:
        flags["--balance-target-counts"] = ""  # flag-only, no value
        notes.append(
            f"target skew on {over_tgt} -> --balance-target-counts"
        )
        needs.append("balance_target_counts")

    # Materialize flags as an ordered list, preserving "--flag" or
    # "--flag value" form.
    flag_list: list[str] = []
    for k, v in flags.items():
        flag_list.append(k)
        if v:
            flag_list.append(v)
    return flag_list, notes, needs


def _parse_prev_flags(prev: str) -> dict[str, str]:
    """Parse a space-separated CLI fragment into a {flag: value} dict.

    A bare flag (e.g. ``--balance-target-counts``) maps to ``""``. We don't
    use argparse here because we don't have the miner's full schema and we
    want the helper to tolerate flags it doesn't know about.
    """
    out: dict[str, str] = {}
    if not prev:
        return out
    toks = prev.split()
    i = 0
    while i < len(toks):
        k = toks[i]
        if not k.startswith("--"):
            i += 1
            continue
        v = ""
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            v = toks[i + 1]
            i += 2
        else:
            i += 1
        out[k] = v
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--audit-json", required=True,
                   help="Path to the .audit.json produced by audit_cf_pairs.py.")
    p.add_argument("--prev-flags", default="",
                   help="Space-separated CLI fragment used on the previous "
                        "mining call (so this tuner can keep prior knobs).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audit_path = Path(args.audit_json)
    if not audit_path.exists():
        # No audit -> first iteration. Echo prev flags untouched.
        print(f"NEXT_FLAGS:{args.prev_flags.strip()}")
        print("NOTES:no audit_json yet, starting fresh")
        print("NEEDS_EDIT:")
        return 0

    audit = json.loads(audit_path.read_text())
    prev = _parse_prev_flags(args.prev_flags)
    next_flags, notes, needs = tune(audit, prev_flags=prev)
    print("NEXT_FLAGS:" + " ".join(next_flags))
    print("NOTES:" + " | ".join(notes))
    print("NEEDS_EDIT:" + ",".join(needs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

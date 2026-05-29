#!/usr/bin/env python
"""Populate a sim-rollout cache JSONL with the no_steer arms from a prior
compare_cf_steer_checkpoints.py run.

The no_steer cache key (steer_disabled=True) intentionally omits the
fields that depend on the codec checkpoint (text / placement /
steer_h_fp), so entries from any prior run with matching
(target_task, env_seed, policy_lang, sim_max_steps) are reusable.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/populate_no_steer_cache.py \\
        --prior-eval data/eval/v8_cf_steer_goal.json \\
        --cache-path data/eval/sim_rollout_cache.jsonl

After running, future ``compare_cf_steer_checkpoints.py --sim-cache-path
<cache>`` calls will hit the populated entries for matching seeds /
tasks instead of re-running them in sim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_prior(prior_path: Path) -> dict:
    obj = json.loads(prior_path.read_text())
    if "samples" not in obj or "config" not in obj:
        raise ValueError(
            f"{prior_path} does not look like a compare_cf_steer_checkpoints "
            "output (missing 'samples' or 'config' top-level keys)."
        )
    return obj


def _emit_entries(prior: dict) -> list[dict]:
    """Yield cache-JSONL entries for the no_steer arms in this prior run."""
    from nla.training.sim_reward import sim_cache_key

    cfg = prior["config"]
    base_seed = int(cfg.get("seed", 0))
    sim_max_steps = int(cfg.get("sim_max_steps", 100))

    # We need a key for each (sample, intent_arm) pair on the no_steer
    # causal arm. The compare script uses these record_keys for no_steer:
    #   sft_av__no_steer                       (matched / no_steer)
    #   sft_av__mismatched_source__no_steer    (mismatched / no_steer)
    # The policy_language_override differs across the two; otherwise the
    # rollout parameters are identical.
    no_steer_keys = [
        ("sft_av__no_steer", "matched", "target_intent"),
        ("sft_av__mismatched_source__no_steer", "mismatched_source", "source_intent"),
    ]

    out: list[dict] = []
    for i, sample in enumerate(prior["samples"]):
        env_name = sample.get("target_env_name") or ""
        target_task = sample.get("target_task") or ""
        source_id = sample.get("source_example_id") or ""
        seed = base_seed + i * 17   # matches compare script's seed formula
        conds = sample.get("conditions", {}) or {}
        for record_key, _arm, lang_field in no_steer_keys:
            cond = conds.get(record_key)
            if not cond:
                continue
            if cond.get("error") is not None:
                # Don't cache error rollouts -- let them re-run.
                continue
            if "skipped_reason" in cond:
                continue
            # Pull policy lang from the recorded value if it's there
            # (eval-v2 'language_swap' protocol records it explicitly).
            # Fall back to sample-level intent if not.
            policy_lang = (
                cond.get("policy_language_override")
                or (sample.get(lang_field) or None)
            )
            key = sim_cache_key(
                env_name=env_name,
                target_task=target_task,
                source_id=source_id,
                text="",  # ignored for steer_disabled=True under new key
                seed=seed,
                sim_max_steps=sim_max_steps,
                placement="",  # ignored
                steer_h_fp="",  # ignored
                policy_language_override=policy_lang,
                steer_disabled=True,
                w_predicate=None,
            )
            entry = {
                "key": key,
                "r_sim": float(cond.get("r_sim", 0.0)),
                "predicate": float(cond.get("predicate", 0.0)),
                "r_dist": float(cond.get("r_dist", 0.0)),
                "r_displace": float(cond.get("r_displace", 0.0)),
                "r_contact": float(cond.get("r_contact", 0.0)),
                "n_steps": int(cond.get("n_steps", 0)),
                "early_stopped": bool(cond.get("early_stopped", False)),
                "elapsed_s": float(cond.get("elapsed_s", 0.0)),
                "success_any": bool(cond.get("success_any", False)),
                "env_name": env_name,
                "target_task": target_task,
                "source_id": source_id,
                "text": "",
                "seed": seed,
                "sim_max_steps": sim_max_steps,
                "policy_language_override": policy_lang or "",
                "steer_disabled": True,
                "_provenance": "populated_from_prior_eval",
            }
            out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prior-eval", required=True,
                   help="JSON produced by compare_cf_steer_checkpoints.py.")
    p.add_argument("--cache-path", required=True,
                   help="Target sim-rollout cache JSONL (will be created or "
                        "appended to). Pass this same path to "
                        "compare_cf_steer_checkpoints.py --sim-cache-path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print entry count + sample keys, don't write.")
    args = p.parse_args(argv)

    prior_path = Path(args.prior_eval)
    if not prior_path.exists():
        print(f"FATAL: prior eval not found: {prior_path}", file=sys.stderr)
        return 2
    prior = _load_prior(prior_path)
    entries = _emit_entries(prior)
    print(f"Emitting {len(entries)} no_steer cache entries "
          f"(from {len(prior['samples'])} samples in {prior_path})")
    for e in entries[:3]:
        print(f"  sample preview: task={e['target_task']!r} seed={e['seed']} "
              f"lang={(e['policy_language_override'] or '')[:40]!r} "
              f"r_sim={e['r_sim']:.3f}")
    if args.dry_run:
        return 0

    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_dedup = 0
    # Deduplicate by key against any existing entries in the cache.
    existing_keys: set[str] = set()
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "key" in obj:
                        existing_keys.add(obj["key"])
                except Exception:
                    continue
    with cache_path.open("a") as f:
        for e in entries:
            if e["key"] in existing_keys:
                n_dedup += 1
                continue
            f.write(json.dumps(e) + "\n")
            existing_keys.add(e["key"])
            n_written += 1
    print(f"Wrote {n_written} new entries to {cache_path} "
          f"({n_dedup} duplicates skipped, total now {len(existing_keys)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

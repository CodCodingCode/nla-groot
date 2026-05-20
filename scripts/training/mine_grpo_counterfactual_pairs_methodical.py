#!/usr/bin/env python
"""Mine LIBERO Goal counterfactual pairs for sim-GRPO with *methodical* targets.

Compared to ``mine_grpo_counterfactual_pairs.py`` (uniform random alternate task),
this script assigns non-matching targets using **TaskSpec** metadata from
``nla.eval.steerability.predicates.GOAL_TASKS``:

* **preserve_behavior** — ``target_task == source_task`` (fraction ``--matching-fraction``).
* **site_swap** — same manipulated ``source_body``, different canonical task /
  placement (different ``destination`` in practice).
* **object_swap** — different ``source_body`` (swap to another manipulation family).

Fallback: if the preferred branch has no legal candidate (e.g. rare joint-proxy
tasks), we fall through to the other branch, then to **preserve_behavior**.

Requires **zero** GPT / API calls. Optional JSON-only provenance columns
(``cf_reason``, ``source_object_tokens``, ``constraint_note``) are preserved in
JSONL files but ignored by ``CounterfactualPairSampler``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.eval.steerability.bddl_bodies import (  # noqa: E402
    DEFAULT_GOAL_BDDL_DIR,
    filter_tasks_with_bodies_in_bddl,
    task_bodies_present_in_bddl,
    validate_cf_target_bodies,
)
from nla.eval.steerability.predicates import GOAL_TASKS, TaskSpec, resolve_task  # noqa: E402

logger = logging.getLogger("mine_grpo_cf_methodical")


def _resolve_canonical(instruction: str) -> str | None:
    try:
        return resolve_task(instruction).name
    except KeyError:
        return None


def _iter_goal_label_rows(labels_path: Path) -> Iterable[dict]:
    n_total = 0
    n_kept = 0
    n_skipped_suite = 0
    n_skipped_meta = 0
    n_skipped_intent = 0
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                continue
            meta = obj.get("meta") or {}
            if meta.get("suite") != "goal":
                n_skipped_suite += 1
                continue
            sid = meta.get("source_example_id")
            ep = meta.get("episode_index")
            st = meta.get("step_index")
            pos = meta.get("position_index")
            ptype = meta.get("position_type")
            instr = meta.get("instruction")
            if sid is None or ep is None or st is None or pos is None or ptype is None or not instr:
                n_skipped_meta += 1
                continue
            canon = _resolve_canonical(instr)
            if canon is None:
                n_skipped_intent += 1
                continue
            yield {
                "example_id": obj.get("example_id"),
                "source_example_id": str(sid),
                "episode_index": int(ep),
                "step_index": int(st),
                "position_index": int(pos),
                "position_type": str(ptype),
                "source_intent": str(instr),
                "source_task": canon,
            }
            n_kept += 1
    logger.info(
        "Goal label scan: total=%d kept=%d  skipped(suite!=goal)=%d  "
        "skipped(missing meta)=%d  skipped(unknown intent)=%d",
        n_total, n_kept, n_skipped_suite, n_skipped_meta, n_skipped_intent,
    )


def _canonical_to_instruction() -> dict[str, str]:
    return {
        "put_the_bowl_on_the_plate": "put the bowl on the plate",
        "put_the_bowl_on_the_stove": "put the bowl on the stove",
        "put_the_bowl_on_top_of_the_cabinet": "put the bowl on top of the cabinet",
        "put_the_wine_bottle_on_the_rack": "put the wine bottle on the rack",
        "put_the_wine_bottle_on_top_of_the_cabinet": "put the wine bottle on top of the cabinet",
        "put_the_cream_cheese_in_the_bowl": "put the cream cheese in the bowl",
        "push_the_plate_to_the_front_of_the_stove": "push the plate to the front of the stove",
        "open_the_top_drawer_and_put_the_bowl_inside": "open the top drawer and put the bowl inside",
        "open_the_middle_drawer_of_the_cabinet": "open the middle drawer of the cabinet",
        "turn_on_the_stove": "turn on the stove",
    }


def _site_swap_candidates(src: str, spec_src: TaskSpec) -> list[str]:
    sb = spec_src.source_body
    if sb is None:
        return []
    out = [
        t
        for t, sp in GOAL_TASKS.items()
        if t != src and sp.source_body == sb
    ]
    return sorted(out)


def _object_swap_candidates(
    src: str,
    spec_src: TaskSpec,
    *,
    exclude_joint_proxy_targets: bool,
) -> list[str]:
    """Alternate tasks whose *manipulated* body differs from ``src``."""
    sb = spec_src.source_body
    out: list[str] = []
    if sb is None:
        return out
    for t, sp in GOAL_TASKS.items():
        if t == src:
            continue
        other = sp.source_body
        if other is None or other == sb:
            continue
        if exclude_joint_proxy_targets and sp.predicate_kind in (
            "displacement_only",
            "contact_with_source",
        ):
            continue
        out.append(t)
    return sorted(out)


def validate_pair_row(
    row: dict,
    canon_to_instr: dict[str, str],
    *,
    bddl_dir: Path = DEFAULT_GOAL_BDDL_DIR,
    instance_cache: dict[str, frozenset[str]] | None = None,
) -> list[str]:
    issues: list[str] = []
    src = row.get("source_task")
    tgt = row.get("target_task")
    ti = row.get("target_intent") or ""
    env = row.get("target_env_name") or ""

    if src not in GOAL_TASKS:
        issues.append(f"bad source_task {src!r}")
    if tgt not in GOAL_TASKS:
        issues.append(f"bad target_task {tgt!r}")
    if not str(ti).strip():
        issues.append("empty target_intent")
    if tgt in canon_to_instr and canon_to_instr[tgt] != ti:
        issues.append("target_intent mismatch vs canon_to_instr")
    if tgt and env != f"libero_sim/{tgt}":
        issues.append(f"target_env_name {env!r} != libero_sim/{tgt}")

    iso = row.get("is_counterfactual")
    if iso is not None and bool(iso) != (src != tgt):
        issues.append("is_counterfactual inconsistent with source_task vs target_task")

    if tgt in GOAL_TASKS and env:
        issues.extend(
            validate_cf_target_bodies(
                str(tgt), str(env), bddl_dir, instance_cache=instance_cache
            )
        )
    return issues


def _pick_candidate(rng: random.Random, cand: list[str]) -> str:
    return cand[rng.randrange(len(cand))]


def pick_methodical_target(
    rng: random.Random,
    *,
    src: str,
    matching_fraction: float,
    prefer_site_swap_prob: float,
    exclude_joint_proxy_targets: bool,
    bddl_dir: Path = DEFAULT_GOAL_BDDL_DIR,
    instance_cache: dict[str, frozenset[str]] | None = None,
) -> tuple[str, str]:
    """Return ``(target_task, cf_reason)``.

    Counterfactual candidates are filtered so ``TaskSpec`` bodies for the
    chosen target exist in that target's BDDL scene before we emit the row.
    """
    cache = instance_cache if instance_cache is not None else {}

    def _valid_tasks(candidates: list[str]) -> list[str]:
        return filter_tasks_with_bodies_in_bddl(
            candidates, bddl_dir, instance_cache=cache
        )

    if rng.random() < matching_fraction:
        if task_bodies_present_in_bddl(src, bddl_dir, instance_cache=cache):
            return src, "preserve_behavior"
        # Should not happen for well-formed Goal tasks; fall through.

    spec_src = GOAL_TASKS[src]
    try_site_first = rng.random() < prefer_site_swap_prob

    def try_site_then_object() -> tuple[str, str] | None:
        site = _valid_tasks(_site_swap_candidates(src, spec_src))
        if site:
            return _pick_candidate(rng, site), "site_swap"
        obj = _valid_tasks(
            _object_swap_candidates(
                src, spec_src, exclude_joint_proxy_targets=exclude_joint_proxy_targets
            )
        )
        if obj:
            return _pick_candidate(rng, obj), "object_swap"
        return None

    def try_object_then_site() -> tuple[str, str] | None:
        obj = _valid_tasks(
            _object_swap_candidates(
                src, spec_src, exclude_joint_proxy_targets=exclude_joint_proxy_targets
            )
        )
        if obj:
            return _pick_candidate(rng, obj), "object_swap"
        site = _valid_tasks(_site_swap_candidates(src, spec_src))
        if site:
            return _pick_candidate(rng, site), "site_swap"
        return None

    seq = try_site_then_object if try_site_first else try_object_then_site
    got = seq()
    if got is not None:
        return got
    if task_bodies_present_in_bddl(src, bddl_dir, instance_cache=cache):
        return src, "fallback_preserve"
    raise RuntimeError(
        f"no BDDL-valid target for source_task {src!r}; check GOAL_TASKS vs BDDL"
    )


def mine_pairs_methodical(
    labels_path: Path,
    *,
    out_path: Path,
    seed: int = 0,
    matching_fraction: float = 0.5,
    max_per_episode: int | None = None,
    max_total: int | None = None,
    max_per_source_task: int | None = None,
    shuffle: bool = True,
    prefer_site_swap_prob: float = 0.5,
    exclude_joint_proxy_targets: bool = True,
) -> dict:
    rng = random.Random(seed)
    canon_to_instr = _canonical_to_instruction()
    all_canon = set(GOAL_TASKS.keys())
    if set(canon_to_instr.keys()) != all_canon:
        logger.warning(
            "canon_to_instruction keys differ from GOAL_TASKS keys: extra=%s missing=%s",
            set(canon_to_instr) - all_canon,
            all_canon - set(canon_to_instr),
        )

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.jsonl not found: {labels_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_per_episode: dict[int, int] = {}
    src_task_emitted: Counter[str] = Counter()
    cf_reason_counts: Counter[str] = Counter()
    src_task_counts: Counter[str] = Counter()
    tgt_task_counts: Counter[str] = Counter()
    counterfactual_count = 0
    written = 0
    n_rejected = 0
    instance_cache: dict[str, frozenset[str]] = {}
    bddl_dir = DEFAULT_GOAL_BDDL_DIR

    row_stream = list(_iter_goal_label_rows(labels_path))
    if shuffle:
        rng.shuffle(row_stream)

    with out_path.open("w") as fout:
        for row in row_stream:
            ep = row["episode_index"]
            if max_per_episode is not None:
                if rows_per_episode.get(ep, 0) >= max_per_episode:
                    continue
            src = row["source_task"]
            if max_per_source_task is not None and src_task_emitted[src] >= max_per_source_task:
                continue
            if max_per_episode is not None:
                rows_per_episode[ep] = rows_per_episode.get(ep, 0) + 1
            src_task_emitted[src] += 1
            src_task_counts[src] += 1

            tgt, cf_reason = pick_methodical_target(
                rng,
                src=src,
                matching_fraction=matching_fraction,
                prefer_site_swap_prob=prefer_site_swap_prob,
                exclude_joint_proxy_targets=exclude_joint_proxy_targets,
                bddl_dir=bddl_dir,
                instance_cache=instance_cache,
            )
            if tgt != src:
                counterfactual_count += 1
            tgt_task_counts[tgt] += 1
            cf_reason_counts[cf_reason] += 1

            spec = GOAL_TASKS[src]
            note = cf_reason if cf_reason == "preserve_behavior" else f"{cf_reason} from={src}"

            out_row = {
                **row,
                "target_intent": canon_to_instr[tgt],
                "target_task": tgt,
                "target_env_name": f"libero_sim/{tgt}",
                "is_counterfactual": bool(tgt != src),
                # Optional tracing (ignored by CounterfactualPairSampler)
                "cf_reason": cf_reason,
                "source_object_tokens": [
                    spec.source_body or "",
                    spec.destination or "",
                ],
                "constraint_note": note,
            }
            issues = validate_pair_row(
                out_row, canon_to_instr,
                bddl_dir=bddl_dir, instance_cache=instance_cache,
            )
            if issues:
                logger.warning(
                    "[validator] rejecting row sid=%s: %s",
                    row.get("source_example_id"),
                    issues,
                )
                n_rejected += 1
                continue

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            if max_total is not None and written >= max_total:
                break

    summary = {
        "labels_path": str(labels_path),
        "out_path": str(out_path),
        "seed": int(seed),
        "n_rows": int(written),
        "n_counterfactual": int(counterfactual_count),
        "counterfactual_fraction": float(counterfactual_count / max(1, written)),
        "source_task_counts": dict(src_task_counts),
        "target_task_counts": dict(tgt_task_counts),
        "cf_reason_counts": dict(cf_reason_counts),
        "n_validator_rejected": int(n_rejected),
        "prefer_site_swap_prob": prefer_site_swap_prob,
        "exclude_joint_proxy_targets": exclude_joint_proxy_targets,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote %d pairs to %s (counterfactual=%.1f%% cf_reason=%s); summary=%s rejects=%d",
        written,
        out_path,
        100.0 * summary["counterfactual_fraction"],
        dict(cf_reason_counts),
        summary_path,
        n_rejected,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--labels", default="data/labels/libero_4suite_combined/labels.jsonl",
        help="Combined labels.jsonl path.",
    )
    p.add_argument(
        "--out",
        default="data/grpo/libero_goal_counterfactual_pairs_methodical.jsonl",
        help="Output JSONL path (distinct from the uniform miner default).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--matching-fraction", type=float, default=0.5,
        help="Probability of preserve_behavior rows (same as sibling miner).",
    )
    p.add_argument(
        "--prefer-site-swap-probability", type=float, default=0.5,
        dest="prefer_site_swap_prob",
        help="Among counterfactual rows, tries site_swap before object_swap "
        "with this probability (fallback chain applies). Default 0.5.",
    )
    p.add_argument(
        "--allow-joint-proxy-object-swap-targets",
        action="store_true",
        dest="allow_joint_proxy",
        help="Allow tasks with displacement_only / contact_with_source predicates "
        "to be chosen as counterfactual *targets* in object_swap (often noisier).",
    )
    p.add_argument(
        "--max-per-episode", type=int, default=None,
        help="Optional cap rows per episode_index.",
    )
    p.add_argument(
        "--max-total", type=int, default=None,
        help="Optional cap total rows for smoke.",
    )
    p.add_argument(
        "--max-per-source-task", type=int, default=None,
        help="Hard cap emitted rows per source_task.",
    )
    p.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false", default=True,
        help="Iterate in file order (no shuffle). Default: shuffle seeded.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    exclude_joint = not getattr(args, "allow_joint_proxy", False)
    mine_pairs_methodical(
        Path(args.labels),
        out_path=Path(args.out),
        seed=args.seed,
        matching_fraction=float(args.matching_fraction),
        max_per_episode=args.max_per_episode,
        max_total=args.max_total,
        max_per_source_task=args.max_per_source_task,
        shuffle=bool(args.shuffle),
        prefer_site_swap_prob=float(args.prefer_site_swap_prob),
        exclude_joint_proxy_targets=exclude_joint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

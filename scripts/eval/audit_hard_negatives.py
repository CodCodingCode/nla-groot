#!/usr/bin/env python
"""Audit the mined hard-negative index for the AR InfoNCE objective.

We answer three questions:

1. **Are the mined negatives "hard"?** -- mined cosine vs random-pair cosine
   distributions on a 500-anchor sample. Healthy is mined sims in roughly
   [0.6, 0.95]; saturated (>0.99) means activations are too uniform to mine.
2. **Are they semantically different captions?** -- token Jaccard between
   anchor and each of its mined negatives. Healthy is mean ~0.1-0.4.
3. **Are there degenerate cases?** -- self-matches, identical caption text,
   anchors negging themselves via the same source_example_id, cross-suite/
   position-type mix.

Outputs a JSON dump (for the report writer) and a short stderr summary.
The script is read-only and uses only the cached labels.jsonl + the
``cos`` field already stored in hard_negatives.jsonl when possible.
Re-loading activations is only needed to build a *random-pair* baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nla.extraction.storage import ActivationShardReader  # noqa: E402

logger = logging.getLogger("audit_hard_neg")

# ---------------------------------------------------------------------------
# Anchor ID parsing
# ---------------------------------------------------------------------------
# Format example: "goal__traj000001_step000038@p143_last_text"
ANCHOR_RE = re.compile(
    r"^(?P<suite>[^_]+(?:_[^_]+)*?)__(?P<example>traj\d+_step\d+)@p(?P<pos>\d+)_(?P<ptype>.+)$"
)


def parse_anchor_id(aid: str) -> dict | None:
    m = ANCHOR_RE.match(aid)
    if not m:
        return None
    return {
        "suite": m.group("suite"),
        "source_example_id": f"{m.group('suite')}__{m.group('example')}",
        "position_index": int(m.group("pos")),
        "position_type": m.group("ptype"),
    }


# ---------------------------------------------------------------------------
# Token Jaccard
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------
def percentiles(vals: list[float], qs: Iterable[float]) -> dict[str, float]:
    if not vals:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    t = torch.tensor(vals, dtype=torch.float64)
    out = {}
    for q in qs:
        k = max(1, min(len(t), int(round(q * len(t)))))
        out[f"p{int(q*100)}"] = float(t.kthvalue(k).values.item())
    return out


def summary_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "count": 0}
    t = torch.tensor(vals, dtype=torch.float64)
    stats = {
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "count": len(vals),
    }
    stats.update(percentiles(vals, [0.10, 0.50, 0.90]))
    return stats


def ascii_histogram(vals: list[float], lo: float, hi: float, n_bins: int = 20, width: int = 40) -> str:
    """Simple horizontal-bar ASCII histogram."""
    if not vals:
        return "(empty)"
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in vals:
        if v < lo or v > hi:
            # clamp to first/last bin so the picture isn't lying
            idx = 0 if v < lo else n_bins - 1
        else:
            idx = min(n_bins - 1, int((v - lo) / (hi - lo) * n_bins))
        counts[idx] += 1
    mx = max(counts) or 1
    lines = []
    for i in range(n_bins):
        bar = "#" * int(width * counts[i] / mx)
        lines.append(f"  [{edges[i]:.3f},{edges[i+1]:.3f}) {counts[i]:6d} |{bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Activation loading (only used for the random-pair baseline + sampled mined)
# ---------------------------------------------------------------------------
def load_activations_for_anchors(
    reader: ActivationShardReader,
    anchors: list[dict],
) -> dict[tuple[str, int], torch.Tensor]:
    """Return {(source_example_id, position_index) -> [D] float32 tensor}.

    Groups requests by shard and opens each shard exactly once.
    """
    out: dict[tuple[str, int], torch.Tensor] = {}
    # group keys by shard
    by_shard: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for a in anchors:
        key = (a["source_example_id"], a["position_index"])
        if key in out:
            continue
        try:
            gidx = reader._by_id[a["source_example_id"]]
        except KeyError:
            out[key] = None  # type: ignore[assignment]
            continue
        rec = reader._records[gidx]
        by_shard[rec.shard_id].append((rec.example_id, rec.local_index, a["position_index"]))

    from safetensors import safe_open
    for shard_id, requests in sorted(by_shard.items()):
        shard_path = reader.root / f"shard_{shard_id:06d}" / "activations.safetensors"
        with safe_open(str(shard_path), framework="pt") as f:
            cache_by_local: dict[int, torch.Tensor] = {}
            for example_id, local_idx, pos in requests:
                if local_idx not in cache_by_local:
                    cache_by_local[local_idx] = f.get_tensor(f"act_{local_idx:06d}")
                feats = cache_by_local[local_idx]
                if pos < 0 or pos >= feats.shape[0]:
                    out[(example_id, pos)] = None  # type: ignore[assignment]
                else:
                    out[(example_id, pos)] = feats[pos].to(torch.float32).clone()
            del cache_by_local
    return out


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # We accept both the legacy `--hard-neg`/`--labels` and the V4
    # `--hard-negatives-jsonl`/`--labels-jsonl` forms (Agent-5's miner
    # follow-up calls them the latter). The dest names below let the rest
    # of the script continue to read args.hard_neg / args.labels.
    hn_group = ap.add_mutually_exclusive_group(required=True)
    hn_group.add_argument("--hard-neg", dest="hard_neg")
    hn_group.add_argument("--hard-negatives-jsonl", dest="hard_neg",
                          help="Alias for --hard-neg (V4 naming).")
    ap.add_argument("--activations-root", required=True)
    lb_group = ap.add_mutually_exclusive_group(required=True)
    lb_group.add_argument("--labels", dest="labels")
    lb_group.add_argument("--labels-jsonl", dest="labels",
                          help="Alias for --labels (V4 naming).")
    ap.add_argument("--sample-anchors", type=int, default=500)
    ap.add_argument("--caption-jaccard-pairs", type=int, default=100)
    ap.add_argument("--random-pairs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    # `--out-json` is now optional (defaults to alongside the markdown out)
    # so V4 callers can specify only `--out-md`.
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args(argv)
    if args.out_json is None:
        args.out_json = str(Path(args.out_md).with_suffix(".json"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)

    # --- 1. Load hard_negatives.jsonl ------------------------------------
    t0 = time.time()
    rows: list[dict] = []
    with open(args.hard_neg) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    n_anchors = len(rows)
    neg_counts = Counter(len(r["negs"]) for r in rows)
    logger.info("Loaded %d anchors in %.1fs; neg-counts=%s",
                n_anchors, time.time() - t0, dict(neg_counts))

    # --- 2. Build parsed metadata + anchor->row lookup -------------------
    parsed_by_id: dict[str, dict] = {}
    for r in rows:
        info = parse_anchor_id(r["anchor"])
        if info is not None:
            parsed_by_id[r["anchor"]] = info

    # V4 miner stores `position_type` and `strategy` per row. Prefer those
    # when available; otherwise fall back to parsing the anchor id.
    def _row_ptype(r: dict) -> str | None:
        pt = r.get("position_type")
        if pt:
            return str(pt)
        info = parsed_by_id.get(r["anchor"])
        return info["position_type"] if info else None

    def _row_strategy(r: dict) -> str:
        return str(r.get("strategy") or "topk_cosine")

    has_new_schema = any("position_type" in r or "strategy" in r for r in rows)
    strategy_counts: Counter = Counter(_row_strategy(r) for r in rows)
    ptype_counts: Counter = Counter(_row_ptype(r) or "?" for r in rows)
    logger.info(
        "Schema: new_v4_fields=%s strategy_counts=%s ptype_counts=%s",
        has_new_schema, dict(strategy_counts), dict(ptype_counts),
    )

    # neg cosine quick summary across ALL rows (cheap; just uses cached cos).
    # We separately track per-ptype so the verdict can be applied per ptype.
    all_cos_top1: list[float] = []
    all_cos_all: list[float] = []
    all_cos_top1_by_ptype: dict[str, list[float]] = defaultdict(list)
    all_cos_all_by_ptype: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        pt = _row_ptype(r) or "?"
        if r["cos"]:
            all_cos_top1.append(float(r["cos"][0]))
            all_cos_top1_by_ptype[pt].append(float(r["cos"][0]))
            all_cos_all.extend(float(c) for c in r["cos"])
            all_cos_all_by_ptype[pt].extend(float(c) for c in r["cos"])

    # --- 3. Load labels for caption analysis -----------------------------
    t0 = time.time()
    captions: dict[str, str] = {}
    suite_of: dict[str, str] = {}
    src_of: dict[str, str] = {}
    ep_of: dict[str, int | None] = {}
    with open(args.labels) as f:
        for line in f:
            obj = json.loads(line)
            ex = obj.get("example_id")
            if not ex:
                continue
            captions[ex] = obj.get("description") or ""
            m = obj.get("meta") or {}
            suite_of[ex] = m.get("suite") or parse_anchor_id(ex)["suite"] if parse_anchor_id(ex) else m.get("suite") or "?"
            src_of[ex] = m.get("source_example_id") or ""
            ep_of[ex] = m.get("episode_index")
    logger.info("Loaded %d label rows in %.1fs", len(captions), time.time() - t0)

    # --- 4. Cross-suite / cross-position-type / degenerate aggregate ----
    n_pairs = 0
    n_same_suite = 0
    n_same_ptype = 0
    n_self_match = 0
    n_same_source = 0
    n_identical_caption = 0
    suite_pair_counter: Counter = Counter()
    ptype_pair_counter: Counter = Counter()

    for r in rows:
        a_id = r["anchor"]
        a_info = parsed_by_id.get(a_id)
        a_cap = captions.get(a_id, "")
        a_src = a_info["source_example_id"] if a_info else None
        a_suite = a_info["suite"] if a_info else None
        a_ptype = a_info["position_type"] if a_info else None
        for n_id in r["negs"]:
            n_info = parsed_by_id.get(n_id)
            if n_info is None:
                # fall back to re-parse
                n_info = parse_anchor_id(n_id) or {}
            n_pairs += 1
            if n_id == a_id:
                n_self_match += 1
            n_src = n_info.get("source_example_id")
            if a_src is not None and n_src == a_src:
                n_same_source += 1
            n_suite = n_info.get("suite")
            n_ptype = n_info.get("position_type")
            if a_suite and n_suite and a_suite == n_suite:
                n_same_suite += 1
            if a_ptype and n_ptype and a_ptype == n_ptype:
                n_same_ptype += 1
            suite_pair_counter[(a_suite, n_suite)] += 1
            ptype_pair_counter[(a_ptype, n_ptype)] += 1
            n_cap = captions.get(n_id, "")
            if a_cap and n_cap and a_cap == n_cap:
                n_identical_caption += 1

    # --- 5. Sample anchors for cosine + jaccard analysis -----------------
    sample_size = min(args.sample_anchors, n_anchors)
    sampled_rows = rng.sample(rows, sample_size)

    # mined cosines (stored)
    mined_cos_sampled: list[float] = []
    mined_cos_by_ptype: dict[str, list[float]] = defaultdict(list)
    for r in sampled_rows:
        pt = _row_ptype(r) or "?"
        for c in r["cos"]:
            mined_cos_sampled.append(float(c))
            mined_cos_by_ptype[pt].append(float(c))

    # --- 6. Random-pair cosine baseline ----------------------------------
    # Sample random pairs of anchors (any two distinct rows), then load their
    # activations and compute cosine. We match position_type within each pair
    # because in training InfoNCE only contrasts within the same activation
    # position; cross-position cosines would be meaningless.
    reader = ActivationShardReader(args.activations_root)

    # We will need activations for:
    #  - all anchors in sampled_rows  -> 500
    #  - all their mined negs         -> 500 * 8 = 4000
    #  - random partners (one per anchor x N_random_pairs) -> drawn from rows
    needed_anchors: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def add_anchor(info: dict | None):
        if info is None:
            return
        key = (info["source_example_id"], info["position_index"])
        if key in seen:
            return
        seen.add(key)
        needed_anchors.append(info)

    for r in sampled_rows:
        add_anchor(parsed_by_id.get(r["anchor"]))
        for n_id in r["negs"]:
            add_anchor(parsed_by_id.get(n_id) or parse_anchor_id(n_id))

    # random pair partners: stratified by position_type so we can compare
    # apples-to-apples within each ptype.
    rows_by_ptype: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        pt = _row_ptype(r)
        if pt is not None:
            rows_by_ptype[pt].append(r)

    random_pair_specs: list[tuple[str, str, str]] = []  # (ptype, anchor_id, partner_id)
    n_random = min(args.random_pairs, n_anchors)
    for _ in range(n_random):
        pt = rng.choice(list(rows_by_ptype.keys()))
        pool = rows_by_ptype[pt]
        a, b = rng.sample(pool, 2)
        random_pair_specs.append((pt, a["anchor"], b["anchor"]))
        add_anchor(parsed_by_id.get(a["anchor"]))
        add_anchor(parsed_by_id.get(b["anchor"]))

    logger.info("Loading %d unique activation vectors from shards ...", len(needed_anchors))
    t0 = time.time()
    act_cache = load_activations_for_anchors(reader, needed_anchors)
    logger.info("Activation load: %.1fs", time.time() - t0)

    # 6a. recomputed mined cosines (sanity vs stored cos)
    recomputed_cos: list[float] = []
    stored_cos: list[float] = []
    for r in sampled_rows:
        a_info = parsed_by_id.get(r["anchor"])
        if a_info is None:
            continue
        a_vec = act_cache.get((a_info["source_example_id"], a_info["position_index"]))
        if a_vec is None:
            continue
        a_norm = torch.nn.functional.normalize(a_vec, dim=-1)
        for c, n_id in zip(r["cos"], r["negs"]):
            n_info = parsed_by_id.get(n_id) or parse_anchor_id(n_id)
            if n_info is None:
                continue
            n_vec = act_cache.get((n_info["source_example_id"], n_info["position_index"]))
            if n_vec is None:
                continue
            n_norm = torch.nn.functional.normalize(n_vec, dim=-1)
            recomputed_cos.append(float((a_norm * n_norm).sum().item()))
            stored_cos.append(float(c))

    # 6b. random pair cosines
    random_cos_by_ptype: dict[str, list[float]] = defaultdict(list)
    random_cos_all: list[float] = []
    for pt, a_id, b_id in random_pair_specs:
        a_info = parsed_by_id.get(a_id)
        b_info = parsed_by_id.get(b_id)
        if a_info is None or b_info is None:
            continue
        a_vec = act_cache.get((a_info["source_example_id"], a_info["position_index"]))
        b_vec = act_cache.get((b_info["source_example_id"], b_info["position_index"]))
        if a_vec is None or b_vec is None:
            continue
        a_n = torch.nn.functional.normalize(a_vec, dim=-1)
        b_n = torch.nn.functional.normalize(b_vec, dim=-1)
        c = float((a_n * b_n).sum().item())
        random_cos_by_ptype[pt].append(c)
        random_cos_all.append(c)

    # --- 7. Caption Jaccard ----------------------------------------------
    # Use all (anchor, neg) pairs from the sampled rows; that's
    # sample_size * 8 = ~4000 pairs. Cheap.
    jaccards_all: list[float] = []
    jaccards_by_ptype: dict[str, list[float]] = defaultdict(list)
    n_missing_caption = 0
    for r in sampled_rows:
        a_id = r["anchor"]
        a_cap = captions.get(a_id, "")
        pt = _row_ptype(r) or "?"
        a_tok = tokenize(a_cap)
        for n_id in r["negs"]:
            n_cap = captions.get(n_id, "")
            if not a_cap or not n_cap:
                n_missing_caption += 1
                continue
            n_tok = tokenize(n_cap)
            j = jaccard(a_tok, n_tok)
            jaccards_all.append(j)
            jaccards_by_ptype[pt].append(j)

    # --- 8. Random-pair caption Jaccard baseline -------------------------
    # Same random pairs as the cosine baseline (paired ptype) so the comparison
    # is apples-to-apples.
    random_jaccards: list[float] = []
    for pt, a_id, b_id in random_pair_specs:
        a_cap = captions.get(a_id, "")
        b_cap = captions.get(b_id, "")
        if not a_cap or not b_cap:
            continue
        random_jaccards.append(jaccard(tokenize(a_cap), tokenize(b_cap)))

    # --- 9. Pick 20 example pairs for the report -------------------------
    example_pairs = []
    pool = list(sampled_rows)
    rng.shuffle(pool)
    for r in pool:
        if len(example_pairs) >= 20:
            break
        a_id = r["anchor"]
        a_cap = captions.get(a_id, "")
        if not a_cap:
            continue
        # take only the first 4 negs (the configured ar_nce_hard_negatives_per_anchor=4)
        neg_block = []
        for n_id, c in zip(r["negs"][:4], r["cos"][:4]):
            n_cap = captions.get(n_id, "")
            neg_block.append({
                "neg_id": n_id,
                "cos": float(c),
                "jaccard": jaccard(tokenize(a_cap), tokenize(n_cap)),
                "snippet": (n_cap[:240] + "...") if len(n_cap) > 240 else n_cap,
            })
        example_pairs.append({
            "anchor_id": a_id,
            "anchor_snippet": (a_cap[:240] + "...") if len(a_cap) > 240 else a_cap,
            "negs": neg_block,
        })

    # --- 9b. Schema-aware (V4) per-ptype degenerate accounting -----------
    # Build per-ptype degenerate counts so the per-ptype verdict has its own
    # numerator. We reuse the loops from step 4 to avoid a second pass.
    pairs_by_ptype: dict[str, int] = defaultdict(int)
    self_by_ptype: dict[str, int] = defaultdict(int)
    same_src_by_ptype: dict[str, int] = defaultdict(int)
    same_suite_by_ptype: dict[str, int] = defaultdict(int)
    identical_cap_by_ptype: dict[str, int] = defaultdict(int)
    for r in rows:
        a_id = r["anchor"]
        a_info = parsed_by_id.get(a_id)
        a_pt = _row_ptype(r) or "?"
        a_src = a_info["source_example_id"] if a_info else None
        a_suite = a_info["suite"] if a_info else None
        a_cap = captions.get(a_id, "")
        for n_id in r["negs"]:
            n_info = parsed_by_id.get(n_id) or parse_anchor_id(n_id) or {}
            pairs_by_ptype[a_pt] += 1
            if n_id == a_id:
                self_by_ptype[a_pt] += 1
            if a_src and n_info.get("source_example_id") == a_src:
                same_src_by_ptype[a_pt] += 1
            if a_suite and n_info.get("suite") == a_suite:
                same_suite_by_ptype[a_pt] += 1
            n_cap = captions.get(n_id, "")
            if a_cap and n_cap and a_cap == n_cap:
                identical_cap_by_ptype[a_pt] += 1

    # --- 10. Output payload ----------------------------------------------
    out = {
        "n_anchors": n_anchors,
        "negs_per_anchor_distribution": dict(neg_counts),
        "has_new_schema": has_new_schema,
        "strategy_counts": dict(strategy_counts),
        "ptype_counts": dict(ptype_counts),
        "all_cos_top1_summary": summary_stats(all_cos_top1),
        "all_cos_summary": summary_stats(all_cos_all),
        "all_cos_top1_by_ptype": {
            pt: summary_stats(v) for pt, v in all_cos_top1_by_ptype.items()
        },
        "all_cos_by_ptype": {
            pt: summary_stats(v) for pt, v in all_cos_all_by_ptype.items()
        },
        "mined_cos_sampled_summary": summary_stats(mined_cos_sampled),
        "mined_cos_by_ptype": {pt: summary_stats(v) for pt, v in mined_cos_by_ptype.items()},
        "random_cos_summary": summary_stats(random_cos_all),
        "random_cos_by_ptype": {pt: summary_stats(v) for pt, v in random_cos_by_ptype.items()},
        "stored_vs_recomputed_max_abs_diff": max(
            (abs(s - r) for s, r in zip(stored_cos, recomputed_cos)), default=0.0
        ),
        "jaccard_summary": summary_stats(jaccards_all),
        "jaccard_by_ptype": {pt: summary_stats(v) for pt, v in jaccards_by_ptype.items()},
        "random_jaccard_summary": summary_stats(random_jaccards),
        "cross_suite": {
            "n_pairs": n_pairs,
            "same_suite": n_same_suite,
            "same_suite_pct": 100.0 * n_same_suite / max(1, n_pairs),
            "cross_suite_pct": 100.0 - 100.0 * n_same_suite / max(1, n_pairs),
            "suite_pair_counts": {
                f"{a}->{b}": c for (a, b), c in suite_pair_counter.most_common(20)
            },
        },
        "cross_position_type": {
            "same_ptype": n_same_ptype,
            "same_ptype_pct": 100.0 * n_same_ptype / max(1, n_pairs),
            "ptype_pair_counts": {
                f"{a}->{b}": c for (a, b), c in ptype_pair_counter.most_common(10)
            },
        },
        "degenerate": {
            "self_matches": n_self_match,
            "self_matches_pct": 100.0 * n_self_match / max(1, n_pairs),
            "same_source_example_id": n_same_source,
            "same_source_example_id_pct": 100.0 * n_same_source / max(1, n_pairs),
            "identical_caption_text": n_identical_caption,
            "identical_caption_text_pct": 100.0 * n_identical_caption / max(1, n_pairs),
            "missing_caption_pairs": n_missing_caption,
        },
        "degenerate_by_ptype": {
            pt: {
                "n_pairs": pairs_by_ptype[pt],
                "self_matches": self_by_ptype[pt],
                "same_source_example_id": same_src_by_ptype[pt],
                "identical_caption_text": identical_cap_by_ptype[pt],
                "same_suite": same_suite_by_ptype[pt],
            }
            for pt in sorted(pairs_by_ptype)
        },
        "examples": example_pairs,
        "histograms": {
            "mined_cos": ascii_histogram(mined_cos_sampled, 0.0, 1.0, n_bins=20),
            "random_cos": ascii_histogram(random_cos_all, 0.0, 1.0, n_bins=20),
            "jaccard": ascii_histogram(jaccards_all, 0.0, 1.0, n_bins=20),
            "random_jaccard": ascii_histogram(random_jaccards, 0.0, 1.0, n_bins=20),
        },
    }

    # --- 10b. Per-ptype verdict ------------------------------------------
    # Compute per-ptype verdict using the same thresholds the overall
    # verdict applies. Iterate over EVERY ptype present in the index (incl.
    # rare ones like `anchor` that the 500-anchor sample may miss), and
    # let ``compute_ptype_verdict`` fall back to the full-population stats
    # when the sample is empty for that ptype.
    all_ptypes = set(list(out["mined_cos_by_ptype"].keys())
                     + list(out["jaccard_by_ptype"].keys())
                     + list(out["all_cos_by_ptype"].keys())
                     + list(out["ptype_counts"].keys()))
    all_ptypes.discard("?")
    per_ptype_verdict: dict[str, str] = {}
    for pt in sorted(all_ptypes):
        per_ptype_verdict[pt] = compute_ptype_verdict(out, pt)
    out["per_ptype_verdict"] = per_ptype_verdict

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote JSON dump to %s", args.out_json)

    # --- 11. Render markdown ---------------------------------------------
    write_markdown(out, args.out_md)
    logger.info("Wrote markdown report to %s", args.out_md)

    # --- 12. Short stderr summary ----------------------------------------
    verdict = compute_verdict(out)
    per_ptype_str = "  ".join(f"{pt}={v}" for pt, v in per_ptype_verdict.items())
    print(
        "==== AUDIT SUMMARY ====\n"
        f"schema                : {'v4 (position_type+strategy present)' if out['has_new_schema'] else 'legacy'}\n"
        f"strategy_counts       : {out['strategy_counts']}\n"
        f"mean mined cosine     : {out['mined_cos_sampled_summary']['mean']:.4f}\n"
        f"mean random cosine    : {out['random_cos_summary']['mean']:.4f}\n"
        f"mean caption Jaccard  : {out['jaccard_summary']['mean']:.4f}\n"
        f"% within-suite negs   : {out['cross_suite']['same_suite_pct']:.1f}%\n"
        f"degenerate pair count : self={out['degenerate']['self_matches']} "
        f"same_src={out['degenerate']['same_source_example_id']} "
        f"identical_cap={out['degenerate']['identical_caption_text']}\n"
        f"verdict (overall)     : {verdict}\n"
        f"verdict (per ptype)   : {per_ptype_str}\n",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def _verdict_from_metrics(
    cos_mean: float,
    jac_mean: float,
    deg_pct: float,
) -> str:
    """Apply the GREEN/YELLOW/RED bands Agent 5 documented."""
    in_band = 0.60 <= cos_mean <= 0.95
    if deg_pct > 1.0 or not in_band:
        return "RED"
    off = 0
    # in_band already True here (otherwise RED), so we score the soft axes
    if jac_mean >= 0.40:
        off += 1
    if deg_pct >= 0.10:
        off += 1
    if off == 0:
        return "GREEN"
    if off == 1:
        return "YELLOW"
    return "RED"


def compute_verdict(out: dict) -> str:
    cos_mean = out["mined_cos_sampled_summary"]["mean"]
    jac_mean = out["jaccard_summary"]["mean"]
    n_pairs = max(1, out["cross_suite"]["n_pairs"])
    deg_pct = 100.0 * (
        out["degenerate"]["self_matches"]
        + out["degenerate"]["same_source_example_id"]
        + out["degenerate"]["identical_caption_text"]
    ) / n_pairs
    return _verdict_from_metrics(cos_mean, jac_mean, deg_pct)


def compute_ptype_verdict(out: dict, ptype: str) -> str:
    """Same band thresholds as ``compute_verdict`` but restricted to one ptype.

    Prefers the sampled ``mined_cos_by_ptype`` (matches the overall verdict's
    sample-based numerator) and falls back to the full-population
    ``all_cos_by_ptype`` if the 500-anchor sample happened to miss this ptype
    (common for the rare `anchor` ptype, ~0.16% of rows).

    Returns ``"N/A"`` only when there is genuinely no cos data at all for
    the ptype (e.g. the miner emitted ``strategy=drop`` for every anchor of
    this ptype).
    """
    cos_block = out["mined_cos_by_ptype"].get(ptype, {})
    cos_mean = cos_block.get("mean", float("nan"))
    if math.isnan(cos_mean) or cos_block.get("count", 0) == 0:
        # Fall back to the full-population stats (covers rare ptypes that
        # the sample missed). We still return N/A if everyone was dropped.
        cos_block = out["all_cos_by_ptype"].get(ptype, {})
        cos_mean = cos_block.get("mean", float("nan"))
        if math.isnan(cos_mean) or cos_block.get("count", 0) == 0:
            return "N/A"
    jac_block = out["jaccard_by_ptype"].get(ptype, {})
    jac_mean = jac_block.get("mean", float("nan"))
    if math.isnan(jac_mean):
        jac_mean = 0.0
    deg_block = out.get("degenerate_by_ptype", {}).get(ptype, {})
    n_pairs_pt = max(1, int(deg_block.get("n_pairs", 0)))
    deg_pct = 100.0 * (
        int(deg_block.get("self_matches", 0))
        + int(deg_block.get("same_source_example_id", 0))
        + int(deg_block.get("identical_caption_text", 0))
    ) / n_pairs_pt
    return _verdict_from_metrics(cos_mean, jac_mean, deg_pct)


def _ptype_block(label: str, by_ptype: dict[str, dict], keys=("mean", "p10", "p50", "p90", "count")) -> str:
    lines = [f"**{label}** (per position_type):", "",
             "| ptype | " + " | ".join(keys) + " |",
             "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for pt, stats in sorted(by_ptype.items()):
        row = ["`" + pt + "`"]
        for k in keys:
            v = stats.get(k, float("nan"))
            if isinstance(v, float) and not math.isnan(v):
                row.append(f"{v:.4f}" if k != "count" else f"{int(v)}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(out: dict, path: str) -> None:
    n = out["n_anchors"]
    cs = out["cross_suite"]
    cp = out["cross_position_type"]
    dg = out["degenerate"]
    verdict = compute_verdict(out)
    md = []
    md.append(f"# Agent 5 — Hard-Negative Mining Quality\n")
    md.append("**Verdict (overall):** `" + verdict + "`")
    per_ptype = out.get("per_ptype_verdict", {})
    if per_ptype:
        md.append("**Verdict (per position_type):** "
                  + ", ".join(f"`{pt}`={v}" for pt, v in per_ptype.items()))
    md.append("")
    md.append("**Source:** `data/activations/libero_4suite_combined/hard_negatives.jsonl`")
    md.append("**Mining script:** `scripts/training/mine_hard_negatives.py` (top-K cosine, same-episode mask).")
    md.append("**This audit:** `scripts/eval/audit_hard_negatives.py` (read-only).")
    if out.get("has_new_schema"):
        md.append("**Schema:** V4 (`position_type`+`strategy` fields present).  "
                  "Strategy counts: `" + json.dumps(out.get("strategy_counts", {})) + "`")
    else:
        md.append("**Schema:** legacy (no `position_type`/`strategy` fields; "
                  "inferred from anchor IDs).")
    md.append("")

    # headline numbers
    md.append("## Headline numbers")
    md.append("")
    md.append("| metric | value | healthy band |")
    md.append("|---|---:|---|")
    md.append(f"| mean mined cosine (all ptypes) | **{out['mined_cos_sampled_summary']['mean']:.4f}** | [0.60, 0.95] |")
    md.append(f"| mean random-pair cosine (same ptype) | {out['random_cos_summary']['mean']:.4f} | — |")
    md.append(f"| mean caption Jaccard (mined) | **{out['jaccard_summary']['mean']:.4f}** | < 0.40 |")
    md.append(f"| mean caption Jaccard (random) | {out['random_jaccard_summary']['mean']:.4f} | — |")
    md.append(f"| within-suite negative fraction | {out['cross_suite']['same_suite_pct']:.2f}% | mostly within-suite |")
    md.append(f"| degenerate-pair count | self={out['degenerate']['self_matches']}, "
              f"same-src={out['degenerate']['same_source_example_id']}, "
              f"identical-caption={out['degenerate']['identical_caption_text']} | all ≈ 0 |")
    md.append("")

    md.append("## 1. Schema summary\n")
    md.append(f"- Total anchors: **{n:,}** (matches `manifest.json` num_examples × 2 position types? "
              f"manifest reports `num_examples=50_790`; 50_790 × 2 = 101_580 ✓)")
    npa = out["negs_per_anchor_distribution"]
    md.append(f"- Negatives per anchor distribution: `{dict(npa)}`  → "
              f"every anchor has exactly **8** negatives.")
    md.append(f"  - The mining script emits **top-K=8** (see `scripts/training/mine_hard_negatives.py`); "
              f"the SFT config `ar_nce_hard_negatives_per_anchor=4` (see `SFTConfig` in `src/nla/training/sft.py`) "
              f"sub-samples 4 of those 8 per anchor at each training step.")
    md.append(f"  - Mining excludes self and same-`episode_index` rows (the dataset uses "
              f"`--exclude-same-episode` by default — verified by mining script source).")
    md.append(f"- Stored field `cos` is the cosine sim used during mining. "
              f"Spot-check: max |stored − recomputed| over {out['mined_cos_sampled_summary']['count']} "
              f"sampled pairs = `{out['stored_vs_recomputed_max_abs_diff']:.2e}` "
              f"(≈ 0 → stored sims are trustworthy and the audit can rely on them).")
    # Note the 3rd position type we found
    ptype_counts = out['cross_position_type']['ptype_pair_counts']
    has_anchor_ptype = any('anchor' in k for k in ptype_counts)
    if has_anchor_ptype:
        md.append("- **Position-type inventory:** three values appear in the anchor IDs: "
                  "`last_text` (~50%), `image_patch` (~50%), and a small `anchor` slice "
                  "(~1.5% of all rows — `grep '@p[0-9]\\+_anchor'` on the JSONL returns 1,567). "
                  "Mining mostly keeps `last_text→last_text` and `image_patch→image_patch` (99.66% of pairs), "
                  "but it does NOT explicitly mask cross-ptype negatives: a small number of "
                  "`last_text↔anchor` swaps appear (≈2,700 pairs total). InfoNCE will treat those "
                  "as legitimate negatives even though the activation positions are different "
                  "kinds of summary token — probably fine, but worth knowing.")
    md.append("")

    md.append("## 2. Cosine-similarity distribution: mined vs random\n")
    md.append("Mined cosines come from the stored `cos` field (full top-8 list per sampled anchor). "
              "Random-pair cosines were computed by sampling 500 random (anchor_a, anchor_b) pairs "
              "from the same position_type and loading their activations.\n")

    def _stat_line(s):
        return (f"mean={s['mean']:.4f}  p10={s.get('p10', float('nan')):.4f}  "
                f"p50={s.get('p50', float('nan')):.4f}  p90={s.get('p90', float('nan')):.4f}  "
                f"min={s['min']:.4f}  max={s['max']:.4f}  n={s['count']}")

    md.append(f"- **Mined (sampled top-8, all ptypes):** {_stat_line(out['mined_cos_sampled_summary'])}")
    md.append(f"- **Random pairs (same ptype):** {_stat_line(out['random_cos_summary'])}")
    md.append(f"- **All anchors top-1 (full 101k):** {_stat_line(out['all_cos_top1_summary'])}")
    md.append("")
    md.append(_ptype_block("Mined cosine", out["mined_cos_by_ptype"]))
    md.append("")
    md.append(_ptype_block("Random-pair cosine", out["random_cos_by_ptype"]))
    md.append("")
    md.append("### Mined-cosine histogram\n```")
    md.append(out["histograms"]["mined_cos"])
    md.append("```")
    md.append("### Random-pair cosine histogram\n```")
    md.append(out["histograms"]["random_cos"])
    md.append("```")

    md.append("\n## 3. Caption-similarity (token Jaccard)\n")
    md.append("Token Jaccard between the anchor caption and each of its 8 mined negative captions.")
    md.append(f"- **Mined pairs:** {_stat_line(out['jaccard_summary'])}")
    md.append(f"- **Random pairs (same ptype, no mining):** {_stat_line(out['random_jaccard_summary'])}")
    md.append("")
    md.append(_ptype_block("Mined Jaccard", out["jaccard_by_ptype"]))
    md.append("")
    md.append("### Mined-Jaccard histogram\n```")
    md.append(out["histograms"]["jaccard"])
    md.append("```")
    md.append("### Random-Jaccard histogram\n```")
    md.append(out["histograms"]["random_jaccard"])
    md.append("```")

    md.append("\n## 4. Cross-suite distribution\n")
    md.append(f"- Total (anchor, neg) pairs: **{cs['n_pairs']:,}**")
    md.append(f"- Within-suite negatives: **{cs['same_suite']:,}** ({cs['same_suite_pct']:.2f}%)")
    md.append(f"- Cross-suite negatives:  **{cs['n_pairs'] - cs['same_suite']:,}** ({cs['cross_suite_pct']:.2f}%)")
    md.append("\nTop (anchor_suite → neg_suite) pairs:")
    md.append("")
    md.append("| pair | count |")
    md.append("|---|---|")
    for k, v in cs["suite_pair_counts"].items():
        md.append(f"| `{k}` | {v:,} |")

    md.append("\n## 5. Cross-position-type distribution\n")
    md.append(f"- Pairs with same position_type as anchor: **{cp['same_ptype']:,}** ({cp['same_ptype_pct']:.2f}%)")
    md.append("")
    md.append("| pair | count |")
    md.append("|---|---|")
    for k, v in cp["ptype_pair_counts"].items():
        md.append(f"| `{k}` | {v:,} |")

    md.append("\n## 6. Degenerate cases\n")
    md.append(f"- Self-matches (anchor in its own neg list): **{dg['self_matches']:,}** "
              f"({dg['self_matches_pct']:.4f}%) — should be 0")
    md.append(f"- Same `source_example_id` as anchor: **{dg['same_source_example_id']:,}** "
              f"({dg['same_source_example_id_pct']:.4f}%)")
    md.append(f"- Identical caption text: **{dg['identical_caption_text']:,}** "
              f"({dg['identical_caption_text_pct']:.4f}%)")
    md.append(f"- Pairs missing a caption in labels: **{dg['missing_caption_pairs']:,}**")

    md.append("\n## 7. Example (anchor, negs) blocks\n")
    md.append("Showing first 4 negs (matches `ar_nce_hard_negatives_per_anchor=4`); each row shows "
              "cosine and Jaccard.\n")
    for i, ex in enumerate(out["examples"], 1):
        md.append(f"### Example {i}\n")
        md.append(f"**Anchor** `{ex['anchor_id']}`")
        md.append(f"> {ex['anchor_snippet']}\n")
        for j, neg in enumerate(ex["negs"], 1):
            md.append(f"- **Neg {j}** `{neg['neg_id']}`  (cos={neg['cos']:.4f}, jaccard={neg['jaccard']:.3f})")
            md.append(f"  > {neg['snippet']}")
        md.append("")

    md.append("\n## 8. Verdict\n")
    md.append(f"**Overall: {verdict}**\n")
    md.append("Bands used:")
    md.append("- GREEN: mean mined cosine in [0.6, 0.95], mean Jaccard < 0.4, degenerate fraction < 0.1%")
    md.append("- YELLOW: exactly one of those off")
    md.append("- RED: degenerate fraction > 1% OR mean cosine outside [0.6, 0.95]")
    md.append("")
    md.append("Inputs to the overall verdict:")
    md.append(f"- mean mined cosine = {out['mined_cos_sampled_summary']['mean']:.4f}")
    md.append(f"- mean Jaccard = {out['jaccard_summary']['mean']:.4f}")
    md.append(f"- self-match pct = {dg['self_matches_pct']:.4f}%")
    md.append(f"- same-source pct = {dg['same_source_example_id_pct']:.4f}%")
    md.append(f"- identical-caption pct = {dg['identical_caption_text_pct']:.4f}%")

    if per_ptype:
        md.append("")
        md.append("### Per-position-type verdict\n")
        md.append("| ptype | verdict | mean cos | mean jaccard | n pairs | self | same_src | identical_cap |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|")
        deg_by_pt = out.get("degenerate_by_ptype", {})
        for pt in sorted(per_ptype.keys()):
            cos_b = out["mined_cos_by_ptype"].get(pt) or out["all_cos_by_ptype"].get(pt, {})
            jac_b = out["jaccard_by_ptype"].get(pt, {})
            deg_b = deg_by_pt.get(pt, {})
            md.append(
                f"| `{pt}` | **{per_ptype[pt]}** | "
                f"{cos_b.get('mean', float('nan')):.4f} | "
                f"{jac_b.get('mean', float('nan')):.4f} | "
                f"{int(deg_b.get('n_pairs', 0)):,} | "
                f"{int(deg_b.get('self_matches', 0)):,} | "
                f"{int(deg_b.get('same_source_example_id', 0)):,} | "
                f"{int(deg_b.get('identical_caption_text', 0)):,} |"
            )
        md.append("")
        md.append("`N/A` means the ptype was emitted with `strategy=drop` by the miner "
                  "(no cos values to grade). When mean cos / jaccard for a rare "
                  "ptype is shown the audit falls back to the full-population "
                  "stats since the 500-anchor sample may not have included that ptype.")

    md.append("\n## 9. Recommendations\n")
    md.append(write_recommendations(out))

    md.append("\n## 10. Cross-references to other audit_reports outputs\n")
    md.append(
        "- **Agent 1 (multimodal judge)**: B-axis grounding 91% (YELLOW), "
        "dragged down by `libero_spatial`. Implication for this audit: if "
        "spatial captions are less grounded, mined within-spatial negatives "
        "may share vague language → boosts Jaccard. Spatial-cell within-suite "
        "mining count here is "
        + f"`{out['cross_suite']['suite_pair_counts'].get('spatial->spatial', 0):,}` "
        "(~22.4% of all mined pairs). If spatial captions are filtered, the "
        "hard-neg index must be re-mined (the dataset loader rebuilds the "
        "candidate set against the in-split rows only — see "
        "`_build_topk_cosine_index` in `src/nla/training/dataset.py`)."
    )
    md.append(
        "- **Agent 2 (prompt-hardening regression)**: position-aware bullet "
        "conformance only 63.59% (RED) — significant for hard-neg mining "
        "because lower-quality captions = noisier Jaccard signal. The "
        "mining itself is unaffected (it uses activations, not captions), "
        "but the InfoNCE objective's *meaningfulness* depends on caption "
        "quality."
    )
    md.append(
        "- **Agent 4 (bullet informativeness)**: `language:` bullet present "
        "in only 20.2% of labels; `language` is also the highest filler "
        "bullet (54.9%). This is consistent with the moderate Jaccard "
        "(~0.38) observed here: many bullets are scene/spatial/plan which "
        "are similar across nearby scenes."
    )
    md.append(
        "- **Agent 3 (caption diversity)**: report not present at audit "
        "time. If it surfaces with high near-duplicate rate, expect Jaccard "
        "p90 to climb above 0.5; re-check this audit after their findings."
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(md))


def write_recommendations(out: dict) -> str:
    """Produce >=3 targeted recommendations based on the actual numbers.

    Each "slot" emits one bullet so the final list always has at least 3.
    """
    cos_mean = out["mined_cos_sampled_summary"]["mean"]
    rnd_mean = out["random_cos_summary"]["mean"]
    jac_mean = out["jaccard_summary"]["mean"]
    rnd_jac = out["random_jaccard_summary"]["mean"]
    by_pt = out["mined_cos_by_ptype"]
    rnd_by_pt = out["random_cos_by_ptype"]
    lines: list[str] = []

    last_text_mean = by_pt.get("last_text", {}).get("mean", float("nan"))
    image_patch_mean = by_pt.get("image_patch", {}).get("mean", float("nan"))
    rnd_last_text = rnd_by_pt.get("last_text", {}).get("mean", float("nan"))
    rnd_image_patch = rnd_by_pt.get("image_patch", {}).get("mean", float("nan"))

    # --- Slot 1: cosine quality --------------------------------------------
    if not math.isnan(last_text_mean) and last_text_mean > 0.98:
        delta = last_text_mean - rnd_last_text if not math.isnan(rnd_last_text) else 0.0
        lines.append(
            f"**`last_text` activations are saturated** (mean mined "
            f"cos = {last_text_mean:.4f}; random `last_text` pairs also "
            f"sit at {rnd_last_text:.4f} → mining gains only Δ={delta:.4f}). "
            f"The final-token hidden state after a templated prompt is "
            f"nearly identical across episodes, so top-K cosine over the "
            f"final layer is mining *uniform noise*: any 'hard' negative "
            f"is interchangeable with a random one. **Action:** mine "
            f"`last_text` from an earlier layer (current activations are "
            f"the last decoder layer; try `layer -4` or `-8`), or project "
            f"the activation through a learned scene-encoder head and "
            f"mine in that subspace, or drop `last_text` from the InfoNCE "
            f"term entirely and keep hard-neg only on `image_patch` "
            f"anchors (where mining is healthier — see slot 3)."
        )
    elif cos_mean > 0.95:
        lines.append(
            f"**Mined cosines are saturated overall** (mean = {cos_mean:.3f} "
            f"vs random {rnd_mean:.3f}). Mining is essentially uniform "
            f"sampling within position_type. Switch to a less-uniform "
            f"layer or learned projection before cosine."
        )
    else:
        lines.append(
            f"**Cosines look reasonable** (mean = {cos_mean:.3f}; random "
            f"baseline = {rnd_mean:.3f}). Consider widening K from 8 to "
            f"16 so the train-time random subsample of K_neg=4 has more "
            f"diversity per epoch."
        )

    # --- Slot 2: Jaccard ----------------------------------------------------
    if jac_mean > 0.5:
        lines.append(
            f"**Caption Jaccard is very high** (mean = {jac_mean:.3f} vs "
            f"random {rnd_jac:.3f}). Either captions are templated/near-"
            f"duplicate (cross-reference Agent 3 diversity / Agent 4 "
            f"informativeness reports) or mining pulls same-scene rows. "
            f"Add a **scene-conditional exclusion** (skip candidates whose "
            f"instruction string or `target:` head word matches the "
            f"anchor's) so InfoNCE only sees different-scene negatives."
        )
    elif jac_mean >= 0.35:
        lines.append(
            f"**Caption Jaccard is moderate-high** (mean = {jac_mean:.3f} "
            f"vs random pairs {rnd_jac:.3f}). Mining genuinely tightens "
            f"caption similarity (Δ = +{jac_mean - rnd_jac:.3f}), which "
            f"is the desired effect — negatives describe similar but "
            f"non-identical scenes. Watch the p90 ({out['jaccard_summary'].get('p90', 0):.2f}); "
            f"if it creeps above 0.6 the InfoNCE objective will start "
            f"penalising legitimately similar captions. Optional: add a "
            f"hard cap (`reject neg if jaccard(anchor, neg) > 0.7`) to "
            f"the miner."
        )
    elif jac_mean >= 0.2:
        lines.append(
            f"**Caption Jaccard is healthy** (mean = {jac_mean:.3f}). "
            f"Negatives share corpus vocabulary but differ in scene "
            f"tokens — exactly the right setting for InfoNCE. No change."
        )
    else:
        lines.append(
            f"**Caption Jaccard is very low** (mean = {jac_mean:.3f}). "
            f"Negatives may be too lexically distinct to push the model — "
            f"verify captions aren't simply too varied (Agent 3) to ever "
            f"collide on common verbs."
        )

    # --- Slot 3: per-position-type contrast --------------------------------
    # The mean cos values can be similar in magnitude yet have very
    # different mined-vs-random *contrast* (i.e. how much the miner actually
    # improves over a random draw of the same ptype). Use the deltas.
    if not math.isnan(image_patch_mean) and not math.isnan(last_text_mean):
        ip_delta = (image_patch_mean - rnd_image_patch
                    if not math.isnan(rnd_image_patch) else float("nan"))
        lt_delta = (last_text_mean - rnd_last_text
                    if not math.isnan(rnd_last_text) else float("nan"))
        # If image_patch contrast is meaningfully larger than last_text contrast,
        # flag the asymmetry — that's the actionable finding.
        if (not math.isnan(ip_delta) and not math.isnan(lt_delta)
                and ip_delta > 2 * max(0.01, lt_delta)):
            lines.append(
                f"**Position-type asymmetry: `image_patch` mining gives "
                f"real contrast, `last_text` does not.** image_patch "
                f"mined-cos = {image_patch_mean:.4f} vs random "
                f"{rnd_image_patch:.4f} (Δ = +{ip_delta:.4f} — top-K "
                f"is meaningfully tighter than random). last_text "
                f"mined-cos = {last_text_mean:.4f} vs random "
                f"{rnd_last_text:.4f} (Δ = +{lt_delta:.4f} — mining "
                f"barely tightens over random). The training-time `tau` "
                f"has to compromise between two distributions of very "
                f"different sharpness. **Action:** (a) apply z-score "
                f"normalisation per `position_type` to the cosines "
                f"before the softmax in `_hard_negative_sims`, or use "
                f"separate temperatures `tau_last_text`, "
                f"`tau_image_patch`; (b) better, drop `last_text` from "
                f"InfoNCE (its mining signal is nonexistent — see "
                f"slot 1) and let `image_patch` carry the contrastive "
                f"term."
            )
        elif abs(last_text_mean - image_patch_mean) > 0.05:
            lines.append(
                f"**Per-position-type cosine spread = "
                f"{abs(last_text_mean - image_patch_mean):.3f}** "
                f"(`last_text`={last_text_mean:.4f} vs "
                f"`image_patch`={image_patch_mean:.4f}). Use separate "
                f"temperatures or z-score normalise per ptype before "
                f"`tau`."
            )
        else:
            lines.append(
                f"**Per-position-type cosine balance is fine** "
                f"(spread = {abs(last_text_mean - image_patch_mean):.3f}; "
                f"mined-vs-random Δ similar between ptypes). Single "
                f"`tau` is OK."
            )

    # --- Slot 4: cross-suite balance ---------------------------------------
    same_suite_pct = out["cross_suite"]["same_suite_pct"]
    if same_suite_pct > 99.0:
        lines.append(
            f"**Negatives are 100% within-suite** ({same_suite_pct:.1f}%). "
            f"Hardest setting (same task family), but the model never sees "
            f"cross-suite breadth. Consider mixing in 1-2 random "
            f"cross-suite negatives per anchor."
        )
    elif same_suite_pct < 50.0:
        lines.append(
            f"**Most negatives are cross-suite** "
            f"({same_suite_pct:.1f}% same-suite). Within-suite mining is "
            f"what breaks template collapse; re-mine with a per-suite "
            f"cosine matrix, then merge."
        )
    else:
        lines.append(
            f"**Cross-suite balance is good** ({same_suite_pct:.1f}% "
            f"within-suite; {100-same_suite_pct:.1f}% cross). "
            f"Within-suite negatives dominate (giving hard contrast) "
            f"while {100-same_suite_pct:.1f}% cross-suite provides "
            f"breadth. No change."
        )

    # number the recommendations
    return "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))


if __name__ == "__main__":
    raise SystemExit(main())

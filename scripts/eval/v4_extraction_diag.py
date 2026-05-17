#!/usr/bin/env python
"""V4 extraction diagnostics — Stage 0 of the image-patch A/B sweep plan.

Three measurements on already-extracted layer-15 activations, no GPU
extraction needed. Pins (or flips) the V3 image_patch collapse hypothesis
before we spend GPU on the layer/strategy sweep.

The three checks (numbered to match
``.cursor/plans/v4_image-patch_a_b_sweep_628ee13b.plan.md``):

* 0a — Per-position-type hard-negative cosine distribution. Parses
  ``hard_negatives*.jsonl`` and reports ``median_cos_top1 / p5 / p95`` for
  each of ``{last_text, image_patch, anchor}``.

* 0b — Suite logistic probe on raw ``h``, stratified by ``position_type``.
  Predicts ``suite ∈ {goal, spatial, object, libero_10}`` from the
  activation alone, using the existing episode-stratified
  ``fit_linear_probe`` from
  ``scripts/eval/probe_h_attributes.py``.

* 0c — Same-episode vs cross-episode cosine gap per ``position_type``. For
  each ptype we sample ``--n-cosine-pairs`` random same-episode pairs and
  the same number of cross-episode pairs, then report the mean cosine in
  each pool and the (same − cross) gap. The gap measures "are activations
  scene-specific in raw vector space?"

Decision rule (per the plan):
  * ``image_patch`` cos_top1 median > 0.97  ⇒  input-side collapse
    hypothesis stands; layer / hook sweep is the priority.
  * ``image_patch`` cos_top1 median < 0.95  ⇒  collapse is in AV training,
    not inputs; position-strategy is the priority.
  * In-between is reported as "ambiguous; run sweep anyway".

Writes a single JSON to ``--out-json`` (default
``data/sft/libero_4suite_v3/extraction_diag.json``) with all three
sections plus a one-paragraph verdict.

Example::

    PYTHONPATH=src .venv/bin/python scripts/eval/v4_extraction_diag.py \\
        --activations-root data/activations/libero_4suite_combined \\
        --labels-jsonl     data/labels/libero_4suite_combined/labels.jsonl \\
        --hard-negatives   data/activations/libero_4suite_combined/hard_negatives_v4.jsonl \\
        --n-per-ptype      1500 \\
        --n-cosine-pairs   500 \\
        --out-json         data/sft/libero_4suite_v3/extraction_diag.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

logger = logging.getLogger("nla.v4_extraction_diag")


PTYPES = ("last_text", "image_patch", "anchor")
# Anchor pattern lifted from ``LabeledPositionDataset._anchor_id``:
#   "<source_example_id>@p<NNN>_<position_type>"
_ANCHOR_RE = re.compile(r"@p\d+_(?P<ptype>[a-zA-Z_]+)$")


# ---------------------------------------------------------------------------
# 0a: per-ptype hard-negative cosine distribution.
# ---------------------------------------------------------------------------

def hard_neg_per_ptype(path: Path) -> dict:
    """Parse hard_negatives.jsonl and return per-ptype cos_top1 stats.

    Each row carries either an explicit ``position_type`` field (v4 schema)
    or the ptype is encoded in the ``anchor`` string. We accept either and
    silently skip rows we can't classify.
    """
    by_ptype: dict[str, list[float]] = defaultdict(list)
    n_total = 0
    n_unclassified = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ptype = row.get("position_type")
            if not ptype:
                m = _ANCHOR_RE.search(str(row.get("anchor", "")))
                ptype = m.group("ptype") if m else None
            if ptype is None:
                n_unclassified += 1
                continue
            cos = row.get("cos") or []
            if not cos:
                continue
            by_ptype[ptype].append(float(cos[0]))
            n_total += 1

    summary: dict[str, dict] = {}
    for ptype in sorted(by_ptype.keys()):
        arr = np.asarray(by_ptype[ptype], dtype=np.float64)
        summary[ptype] = {
            "n": int(arr.size),
            "median_cos_top1": float(np.median(arr)),
            "mean_cos_top1": float(arr.mean()),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return {
        "source_file": str(path),
        "n_rows_total": int(n_total),
        "n_unclassified": int(n_unclassified),
        "by_ptype": summary,
    }


# ---------------------------------------------------------------------------
# Activation loading shared by 0b/0c.
# ---------------------------------------------------------------------------

def load_h_per_ptype(
    activations_root: Path,
    labels_jsonl: Path,
    n_per_ptype: int,
    *,
    seed: int = 0,
    min_bullet_lines: int = 3,
) -> dict[str, dict]:
    """Sample ``n_per_ptype`` activation vectors per position_type.

    Returns ``{ptype: {"h": [N, D] float32, "suite": [N] str,
    "episode": [N] int, "source": [N] str, "pos": [N] int}}``.

    We sample labels uniformly per ptype (without replacement), then group
    by ``source_example_id`` and pull all positions per example in a
    single shard read. This keeps the I/O cost ~ ``n_shards`` open() calls
    instead of ``3 * n_per_ptype``.
    """
    from nla.extraction.storage import ActivationShardReader
    from nla.training.dataset import load_labels_jsonl

    reader = ActivationShardReader(str(activations_root))
    by_id = {rec.example_id: rec for rec in reader.records}

    labels = load_labels_jsonl(str(labels_jsonl), min_bullet_lines=min_bullet_lines)

    by_ptype: dict[str, list] = defaultdict(list)
    for entry in labels:
        rec = by_id.get(entry.source_example_id)
        if rec is None:
            continue
        if entry.position_index >= int(rec.seq_len):
            continue
        suite = entry.raw.get("meta", {}).get("suite")
        if suite is None:
            continue
        by_ptype[entry.position_type].append(
            (entry.source_example_id, int(entry.position_index), str(suite),
             None if rec.episode_index is None else int(rec.episode_index))
        )

    rng = random.Random(seed)
    chosen_by_ptype: dict[str, list] = {}
    for ptype in PTYPES:
        pool = by_ptype.get(ptype, [])
        if not pool:
            continue
        rng.shuffle(pool)
        chosen_by_ptype[ptype] = pool[: int(n_per_ptype)]
        logger.info("ptype=%s n_pool=%d n_sample=%d",
                    ptype, len(pool), len(chosen_by_ptype[ptype]))

    # Group the union of chosen rows by source_example_id to batch shard reads.
    needed_positions: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for ptype, rows in chosen_by_ptype.items():
        for sid, pos, _suite, _ep in rows:
            needed_positions[sid].append((ptype, pos))

    feats_cache: dict[tuple[str, int], np.ndarray] = {}
    record_filter = lambda rec: rec.example_id in needed_positions  # noqa: E731

    for item in reader.iter_examples(record_filter=record_filter):
        rec = item["_record"]
        features = item["features"]
        seq_len = int(features.shape[0])
        for ptype, pos in needed_positions[rec.example_id]:
            if pos >= seq_len:
                continue
            feats_cache[(rec.example_id, pos)] = (
                features[pos].detach().cpu().float().numpy()
            )

    out: dict[str, dict] = {}
    for ptype, rows in chosen_by_ptype.items():
        h_list, suite_list, ep_list, sid_list, pos_list = [], [], [], [], []
        for sid, pos, suite, ep in rows:
            v = feats_cache.get((sid, pos))
            if v is None:
                continue
            h_list.append(v)
            suite_list.append(suite)
            ep_list.append(-1 if ep is None else int(ep))
            sid_list.append(sid)
            pos_list.append(pos)
        if not h_list:
            continue
        out[ptype] = {
            "h": np.stack(h_list, axis=0).astype(np.float32),
            "suite": np.asarray(suite_list),
            "episode": np.asarray(ep_list, dtype=np.int64),
            "source": np.asarray(sid_list),
            "pos": np.asarray(pos_list, dtype=np.int64),
        }
        logger.info("ptype=%s loaded h shape=%s", ptype, out[ptype]["h"].shape)
    return out


# ---------------------------------------------------------------------------
# 0b: suite logistic probe per ptype.
# ---------------------------------------------------------------------------

def suite_probe_per_ptype(
    pools: dict[str, dict],
    *,
    seed: int = 0,
    held_out_fraction: float = 0.2,
) -> dict:
    """Fit a logistic probe predicting ``suite`` from ``h`` per ptype.

    Episode-stratified train/val split so the probe can't memorize trajectory
    timesteps. We re-use ``fit_linear_probe`` from probe_h_attributes so the
    classifier settings (class_weight, max_iter) stay aligned with the
    rest of the pipeline.
    """
    try:
        from scripts.eval.probe_h_attributes import fit_linear_probe
    except ImportError:
        # ``scripts`` is on sys.path via REPO/scripts at module import time;
        # fall back to the relative form so direct invocation also works.
        from eval.probe_h_attributes import fit_linear_probe  # type: ignore

    out: dict[str, dict] = {}
    for ptype in PTYPES:
        if ptype not in pools:
            continue
        h = pools[ptype]["h"]
        suite = pools[ptype]["suite"]
        ep = pools[ptype]["episode"]

        rng = np.random.default_rng(seed)
        unique_eps = np.unique(ep)
        rng.shuffle(unique_eps)
        n_held = max(1, int(round(len(unique_eps) * held_out_fraction)))
        held_eps = set(unique_eps[:n_held].tolist())
        is_val = np.array([int(e) in held_eps for e in ep], dtype=bool)

        if is_val.sum() < 4 or (~is_val).sum() < 4:
            # Fallback to row split if too few episodes
            idx = np.arange(len(h))
            rng.shuffle(idx)
            n_val = max(2, int(round(len(idx) * held_out_fraction)))
            is_val = np.zeros(len(h), dtype=bool)
            is_val[idx[:n_val]] = True

        X_train, y_train = h[~is_val], suite[~is_val]
        X_val,   y_val   = h[is_val],  suite[is_val]
        acc, f1, _ = fit_linear_probe(X_train, y_train, X_val, y_val, seed=seed)
        # Majority baseline on val for sanity.
        if len(y_val):
            vals, counts = np.unique(y_val, return_counts=True)
            maj_acc = float(counts.max() / counts.sum())
        else:
            maj_acc = float("nan")
        out[ptype] = {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "majority_baseline_acc": maj_acc,
            "n_train": int((~is_val).sum()),
            "n_val": int(is_val.sum()),
            "n_classes": int(len(np.unique(suite))),
            "classes": [str(c) for c in sorted(np.unique(suite).tolist())],
        }
        logger.info("ptype=%s suite_probe acc=%.3f macro_f1=%.3f maj=%.3f",
                    ptype, acc, f1, maj_acc)
    return out


# ---------------------------------------------------------------------------
# 0c: same-episode vs cross-episode cosine gap per ptype.
# ---------------------------------------------------------------------------

def episode_cosine_gap(
    pools: dict[str, dict],
    *,
    n_pairs: int = 500,
    seed: int = 0,
) -> dict:
    """For each ptype, mean cosine of same-episode pairs vs cross-episode."""
    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for ptype in PTYPES:
        if ptype not in pools:
            continue
        h = pools[ptype]["h"]
        ep = pools[ptype]["episode"]
        # L2-normalize for cosine = dot.
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)

        same_ep_pairs: list[tuple[int, int]] = []
        cross_ep_pairs: list[tuple[int, int]] = []
        ep_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, e in enumerate(ep.tolist()):
            ep_to_idx[int(e)].append(i)

        episodes = [e for e, ix in ep_to_idx.items() if len(ix) >= 2 and e != -1]
        if not episodes:
            out[ptype] = {"error": "no episode has >= 2 samples"}
            continue

        attempts = 0
        max_attempts = n_pairs * 20
        while len(same_ep_pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            e = episodes[rng.integers(0, len(episodes))]
            ix = ep_to_idx[e]
            a, b = rng.choice(ix, size=2, replace=False)
            same_ep_pairs.append((int(a), int(b)))

        all_idx = np.arange(len(h))
        attempts = 0
        while len(cross_ep_pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            a, b = rng.choice(all_idx, size=2, replace=False)
            if int(ep[a]) != int(ep[b]):
                cross_ep_pairs.append((int(a), int(b)))

        if not same_ep_pairs or not cross_ep_pairs:
            out[ptype] = {"error": "could not sample enough pairs"}
            continue

        same_a = np.asarray([p[0] for p in same_ep_pairs])
        same_b = np.asarray([p[1] for p in same_ep_pairs])
        cross_a = np.asarray([p[0] for p in cross_ep_pairs])
        cross_b = np.asarray([p[1] for p in cross_ep_pairs])

        same_cos = (hn[same_a] * hn[same_b]).sum(axis=1)
        cross_cos = (hn[cross_a] * hn[cross_b]).sum(axis=1)

        out[ptype] = {
            "n_same_pairs": int(len(same_ep_pairs)),
            "n_cross_pairs": int(len(cross_ep_pairs)),
            "same_episode_cos_mean": float(same_cos.mean()),
            "same_episode_cos_std":  float(same_cos.std()),
            "cross_episode_cos_mean": float(cross_cos.mean()),
            "cross_episode_cos_std":  float(cross_cos.std()),
            "gap": float(same_cos.mean() - cross_cos.mean()),
        }
        logger.info("ptype=%s same=%.3f cross=%.3f gap=%.3f",
                    ptype, same_cos.mean(), cross_cos.mean(),
                    same_cos.mean() - cross_cos.mean())
    return out


# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------

def render_verdict(diag: dict) -> str:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

    by_ptype = diag.get("hard_negatives", {}).get("by_ptype", {})
    ip_med = by_ptype.get("image_patch", {}).get("median_cos_top1")
    lt_med = by_ptype.get("last_text", {}).get("median_cos_top1")
    an_med = by_ptype.get("anchor", {}).get("median_cos_top1")

    sp = diag.get("suite_probe", {})
    ip_acc = sp.get("image_patch", {}).get("accuracy")
    lt_acc = sp.get("last_text", {}).get("accuracy")
    an_acc = sp.get("anchor", {}).get("accuracy")

    gap = diag.get("episode_cosine_gap", {})
    ip_gap = gap.get("image_patch", {}).get("gap")
    lt_gap = gap.get("last_text", {}).get("gap")
    an_gap = gap.get("anchor", {}).get("gap")

    parts: list[str] = []
    parts.append(
        f"Per-ptype hard-negative median cos_top1: "
        f"image_patch={fmt(ip_med)}, last_text={fmt(lt_med)}, anchor={fmt(an_med)}."
    )

    if isinstance(ip_med, (int, float)):
        if ip_med > 0.97:
            hyp = (
                "image_patch median > 0.97 ⇒ input-side collapse hypothesis stands; "
                "the layer / hook sweep is the priority."
            )
        elif ip_med < 0.95:
            hyp = (
                "image_patch median < 0.95 ⇒ collapse is in AV training, "
                "not inputs; position-strategy is the priority."
            )
        else:
            hyp = (
                "image_patch median in [0.95, 0.97] ⇒ ambiguous; run the full sweep."
            )
        parts.append(hyp)

    if isinstance(ip_acc, (int, float)) or isinstance(lt_acc, (int, float)):
        parts.append(
            f"Suite logistic probe accuracy on raw h: "
            f"image_patch={fmt(ip_acc)}, last_text={fmt(lt_acc)}, anchor={fmt(an_acc)}."
        )
        if isinstance(ip_acc, (int, float)) and ip_acc >= 0.80:
            parts.append(
                "image_patch suite-probe accuracy >= 0.80 with AV grounding=0% "
                "is hard evidence the failure is at the AV head, not the activations."
            )

    if isinstance(ip_gap, (int, float)) or isinstance(lt_gap, (int, float)):
        parts.append(
            f"Same-vs-cross episode cosine gap: "
            f"image_patch={fmt(ip_gap)}, last_text={fmt(lt_gap)}, anchor={fmt(an_gap)}. "
            "Larger gap = activations are more scene-specific in raw space."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations-root", type=Path,
                   default=REPO / "data/activations/libero_4suite_combined")
    p.add_argument("--labels-jsonl", type=Path,
                   default=REPO / "data/labels/libero_4suite_combined/labels.jsonl")
    p.add_argument("--hard-negatives", type=Path,
                   default=REPO / "data/activations/libero_4suite_combined/hard_negatives_v4.jsonl",
                   help="hard_negatives.jsonl path; pre-v4 file accepted (we parse ptype from anchor).")
    p.add_argument("--out-json", type=Path,
                   default=REPO / "data/sft/libero_4suite_v3/extraction_diag.json")
    p.add_argument("--n-per-ptype", type=int, default=1500,
                   help="Number of activations to sample per position_type for 0b/0c.")
    p.add_argument("--n-cosine-pairs", type=int, default=500,
                   help="Number of pair samples per ptype for 0c.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-bullet-lines", type=int, default=3)
    p.add_argument("--skip-suite-probe", action="store_true")
    p.add_argument("--skip-cosine-gap", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if not args.hard_negatives.exists():
        logger.error("Hard-negatives file does not exist: %s", args.hard_negatives)
        return 2
    if not args.activations_root.exists():
        logger.error("Activations root does not exist: %s", args.activations_root)
        return 2
    if not args.labels_jsonl.exists():
        logger.error("Labels jsonl does not exist: %s", args.labels_jsonl)
        return 2

    logger.info("Stage 0a: per-ptype hard-negative cos_top1 from %s", args.hard_negatives)
    hn = hard_neg_per_ptype(args.hard_negatives)

    pools: dict[str, dict] = {}
    if not (args.skip_suite_probe and args.skip_cosine_gap):
        logger.info(
            "Loading %d activations per ptype from %s",
            args.n_per_ptype, args.activations_root,
        )
        pools = load_h_per_ptype(
            args.activations_root, args.labels_jsonl,
            n_per_ptype=args.n_per_ptype, seed=args.seed,
            min_bullet_lines=args.min_bullet_lines,
        )

    sp: dict = {}
    if not args.skip_suite_probe:
        logger.info("Stage 0b: suite logistic probe per ptype")
        sp = suite_probe_per_ptype(pools, seed=args.seed)

    eg: dict = {}
    if not args.skip_cosine_gap:
        logger.info("Stage 0c: same-vs-cross episode cosine gap per ptype")
        eg = episode_cosine_gap(pools, n_pairs=args.n_cosine_pairs, seed=args.seed)

    diag = {
        "stage": "0_diagnosis",
        "config": {
            "activations_root": str(args.activations_root),
            "labels_jsonl": str(args.labels_jsonl),
            "hard_negatives": str(args.hard_negatives),
            "n_per_ptype": int(args.n_per_ptype),
            "n_cosine_pairs": int(args.n_cosine_pairs),
            "seed": int(args.seed),
            "min_bullet_lines": int(args.min_bullet_lines),
        },
        "hard_negatives": hn,
        "suite_probe": sp,
        "episode_cosine_gap": eg,
    }
    diag["verdict"] = render_verdict(diag)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(diag, indent=2) + "\n")
    logger.info("Wrote %s", args.out_json)
    print("VERDICT:", diag["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

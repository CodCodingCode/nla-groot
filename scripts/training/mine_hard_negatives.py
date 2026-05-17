#!/usr/bin/env python
"""Offline top-K cosine hard-negative mining for AR's InfoNCE term.

Output is consumed by ``LabeledPositionDataset(hard_negative_source="topk_cosine",
hard_negative_index_path=<this file>)`` during SFT.

Background
----------
The V2 SFT diagnosis (``docs/evals/v2_lessons_learned.md``) showed that
in-batch InfoNCE at B=4 with random negatives is too easy: AR can reconstruct
generic template captions to almost any activation in the batch, so the
contrastive term collapses toward ``ln(B)`` and AV faces no pressure to
write scene-specific prose. The fix is **hard negatives with the SAME-LOOKING
activation but a DIFFERENT scene caption** — those are exactly the captions
the model would template-collapse onto, so penalizing them is what breaks
the collapse.

Algorithm
---------
1. Load every kept label (``source_example_id``, ``position_index``,
   ``position_type``, ``description``) from labels.jsonl. We respect
   ``--min-bullet-lines`` so the mining set matches what SFT will train on.
2. Materialize the ``[N, D]`` activation slice ``h_i`` for each kept label
   (one ``[D]`` vector per ``(source_example_id, position_index)``).
3. L2-normalize. Compute pairwise cosine ``H @ H.T`` in chunks on GPU.
4. Mask the diagonal and (by default) every row that shares the anchor's
   ``episode_index`` — same-episode adjacent steps have ~identical
   activations and ~identical captions, so they are *trivial* negatives,
   not hard ones.
5. Take top-K per anchor. Emit one JSONL row per anchor:
   ``{"anchor": "<label_example_id>", "negs": ["...", ...], "cos": [...]}``

Notes
-----
* We do NOT store the caption strings in the index — only the negative
  anchor IDs. The dataset side joins back to captions at init time so the
  index stays small and survives label edits as long as IDs are stable.
* Self-cosine is masked to ``-inf``.  ``--exclude-same-episode`` (default
  ``True``) additionally masks rows whose ``episode_index`` equals the
  anchor's; turn it off if your dataset has only a handful of episodes
  (e.g. smoke tests) where every row is "same episode" and excluding
  would leave an empty pool.
* Chunked GPU matmul: ``--chunk-size 2048`` keeps peak memory ~ ``chunk *
  N * 4 bytes``; safe up to N ~ 200k on a 24GB GPU.

Example (LIBERO)
----------------
::

    PYTHONPATH=src python scripts/training/mine_hard_negatives.py \\
        --activations-root data/activations/libero_goal_pilot \\
        --labels-jsonl     data/labels/libero_goal_pilot/labels.jsonl \\
        --min-bullet-lines 3 \\
        --top-k            8 \\
        --out              data/activations/libero_goal_pilot/hard_negatives.jsonl

The script is corpus-agnostic; substitute any extraction root + labels file
produced by the standard pipeline (``scripts/extraction/run_extract.py`` and
``scripts/labeling/run_label.py``). For small corpora with < ~30 episodes
add ``--no-exclude-same-episode`` so the candidate pool isn't starved.

Output schema (one JSON object per line)::

    {
      "anchor": "<label_example_id>",
      "negs":   ["<label_example_id>", ...],   # length <= --top-k
      "cos":    [0.913, 0.901, ...],           # cosine sims (in [-1, 1])
      "anchor_episode": 12,                    # for diagnostics; may be null
      "position_type": "last_text",            # added in V4 mining
      "strategy": "topk_cosine"                # added in V4 mining
    }

V4 additions (Agent 5 follow-up)
--------------------------------
* ``--per-position-type``: restrict each anchor's candidate pool to rows
  sharing its position_type. Cleaner setting for InfoNCE.
* ``--jaccard-cap CAP``: post-filter top-K to drop candidates whose caption
  Jaccard vs the anchor exceeds CAP (recommended 0.7). The miner walks the
  next-most-similar cosine candidate to fill dropped slots.
* ``--last-text-strategy {topk_cosine,random_same_ptype,drop}``: how to
  populate negs for ``last_text`` anchors specifically. Use
  ``random_same_ptype`` (or ``drop``) when the audit shows
  ``last_text`` cosine is saturated (mined ≈ random).
* New per-row fields ``position_type`` and ``strategy`` so the audit can
  stratify without rejoining labels.

Sanity check (printed to stderr at the end)::

    n_anchors=49382  median_cos_top1=0.78  p5=0.61  p95=0.92  n_skipped=0

A median ``cos_top1`` in ``[0.5, 0.9]`` is healthy.  Outside that range:

* Too high (>0.95) → activations are too uniform; mining will be noise.
  Investigate label diversity or extraction layer.
* Too low (<0.3)  → activations are very far apart; either the corpus is
  tiny or hidden states are not aligned. Hard-neg may not help.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from nla.extraction.storage import ActivationShardReader
from nla.training.dataset import load_labels_jsonl

logger = logging.getLogger("nla.mine_hard_neg")


# Bullet headers we strip when computing the Jaccard token set so that
# headers like "scene:", "target:" don't dominate the overlap.
_BULLET_HEADERS = frozenset({
    "scene", "target", "distractor", "language", "spatial",
    "plan", "motion", "gripper", "action", "instruction",
})
_PUNCT_STRIP = ".,;:!?\"'`()[]{}<>"


def _caption_tokens(caption: str) -> set[str]:
    """Whitespace-tokenized lowercased word set, stripped of `-` and bullet headers."""
    tokens: set[str] = set()
    if not caption:
        return tokens
    for line in caption.splitlines():
        s = line.strip()
        if s.startswith("- "):
            s = s[2:].strip()
        elif s == "-":
            continue
        # Strip a leading "header:" if header is one of the known bullet keys.
        if ":" in s:
            head, _, body = s.partition(":")
            if head.strip().lower() in _BULLET_HEADERS:
                s = body
        for tok in s.lower().split():
            tok = tok.strip(_PUNCT_STRIP)
            if tok:
                tokens.add(tok)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass(frozen=True)
class _KeptAnchor:
    label_id: str
    activation_global_index: int
    position_index: int
    position_type: str
    source_example_id: str
    episode_index: int | None
    description: str


def _label_key(source_example_id: str, position_index: int, position_type: str) -> str:
    """Canonical anchor key used by both the miner and the dataset loader.

    Mirrors the synthetic ``example_id`` that the labeling pipeline writes;
    we fall back to this form if a label row is missing the explicit
    ``example_id`` field. The dataset side accepts either form.
    """
    return f"{source_example_id}@p{position_index:03d}_{position_type}"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-root", required=True,
                   help="Phase-1 extraction root (must contain manifest.json + index.jsonl).")
    p.add_argument("--labels-jsonl", required=True,
                   help="Phase-2 labels.jsonl. Only labels whose source_example_id "
                        "exists in the activation index are kept.")
    p.add_argument("--out", required=True,
                   help="Output JSONL path. One row per kept anchor.")
    p.add_argument("--top-k", type=int, default=8,
                   help="K per anchor. Default 8. The training-time loader can "
                        "sample any K_neg <= this value with --ar-nce-hard-negatives-per-anchor.")
    p.add_argument("--min-bullet-lines", type=int, default=None,
                   help="Drop labels with fewer than this many '-' bullet lines. "
                        "Should mirror SFTConfig.min_bullet_lines so mining and "
                        "training see the same anchor set.")
    p.add_argument("--exclude-same-episode", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Mask out negatives sharing the anchor's episode_index. "
                        "Default True (same-episode rows have near-identical h and "
                        "near-identical caption, so they're trivial negatives). "
                        "Pass --no-exclude-same-episode for tiny corpora.")
    p.add_argument("--chunk-size", type=int, default=2048,
                   help="Anchors per chunk during the GPU matmul. Lower if OOM.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for the pairwise cosine matmul.")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"],
                   help="Compute dtype for the matmul. fp32 is safest; bf16 saves "
                        "memory but the cosine ranking is less stable on near-ties.")
    p.add_argument("--log-every", type=int, default=2048,
                   help="Log progress every N anchors processed.")
    p.add_argument("--per-position-type", action="store_true",
                   help="Restrict each anchor's candidate pool to rows sharing "
                        "its position_type. With this on, an `image_patch` anchor "
                        "only ever sees `image_patch` candidates and likewise for "
                        "`last_text` and `anchor`. This is the cleaner setting for "
                        "InfoNCE because cross-ptype cosines are not comparable. "
                        "Default off for back-compat with V3 mining.")
    p.add_argument("--jaccard-cap", type=float, default=None,
                   help="Optional float in [0, 1]. When set, post-filter each "
                        "anchor's top-K mined negatives: drop any candidate whose "
                        "whitespace-token Jaccard vs the anchor caption (after "
                        "stripping `-` and bullet headers like `scene:`/`target:`) "
                        "exceeds this cap, then walk the next-most-similar cosine "
                        "candidate to fill the slot. Recommended 0.7. Default off.")
    p.add_argument("--last-text-strategy", default="topk_cosine",
                   choices=["topk_cosine", "random_same_ptype", "drop"],
                   help="How to populate negatives for `last_text` anchors only "
                        "(image_patch and anchor anchors always use topk_cosine). "
                        "`topk_cosine` keeps existing behavior. "
                        "`random_same_ptype` samples K random other `last_text` "
                        "rows (still excluding same-episode); use this when "
                        "Agent-5 audits show `last_text` activations are "
                        "saturated (mined cosine ~= random cosine). "
                        "`drop` emits an empty negs list for `last_text` anchors; "
                        "the dataset loader falls back to its own behavior.")
    p.add_argument("--jaccard-oversample", type=int, default=4,
                   help="Multiplicative factor for top-K oversampling when "
                        "--jaccard-cap is set (we mine K*oversample candidates "
                        "and walk them in cosine order until K admissible remain). "
                        "Higher = more robust to dense scenes at the cost of GPU "
                        "memory. Default 4.")
    return p


def _build_kept_anchors(
    *,
    activations_root: str,
    labels_jsonl: str,
    min_bullet_lines: int | None,
) -> tuple[list[_KeptAnchor], ActivationShardReader]:
    """Load activations index + labels and produce the list of kept anchors.

    Kept anchors are returned in the same order they appear in ``labels.jsonl``
    after dropping rows with missing activations or short captions.  Order
    determines the row index in the ``H`` matrix and the per-row top-K rank
    is deterministic given the same input.
    """
    reader = ActivationShardReader(activations_root)
    index_by_id = {rec.example_id: i for i, rec in enumerate(reader.records)}
    logger.info("Activation corpus: %d examples at %s", len(reader.records), activations_root)

    labels = load_labels_jsonl(labels_jsonl, min_bullet_lines=min_bullet_lines)
    kept: list[_KeptAnchor] = []
    n_missing = 0
    for entry in labels:
        if entry.source_example_id not in index_by_id:
            n_missing += 1
            continue
        gidx = int(index_by_id[entry.source_example_id])
        rec = reader.records[gidx]
        label_id = entry.raw.get("example_id") or _label_key(
            entry.source_example_id, entry.position_index, entry.position_type,
        )
        ep = None if rec.episode_index is None else int(rec.episode_index)
        kept.append(_KeptAnchor(
            label_id=str(label_id),
            activation_global_index=gidx,
            position_index=int(entry.position_index),
            position_type=str(entry.position_type),
            source_example_id=str(entry.source_example_id),
            episode_index=ep,
            description=str(entry.description or ""),
        ))
    logger.info(
        "Mining set: %d kept labels (skipped %d with no matching activation).",
        len(kept), n_missing,
    )
    return kept, reader


def _materialize_h_matrix(
    *,
    kept: list[_KeptAnchor],
    reader: ActivationShardReader,
) -> torch.Tensor:
    """Return an ``[N, D]`` CPU tensor of float32 ``h_i`` for each kept anchor.

    We group anchors by their underlying activation example so each shard's
    safetensors file is opened only once per example, amortizing the I/O.
    """
    n = len(kept)
    D = int(reader.manifest.hidden_size)
    H = torch.empty(n, D, dtype=torch.float32)

    by_global: dict[int, list[int]] = {}
    for kept_i, ka in enumerate(kept):
        by_global.setdefault(ka.activation_global_index, []).append(kept_i)

    t0 = time.time()
    n_loaded = 0
    for gidx, kept_indices in by_global.items():
        item = reader[gidx]
        feats = item["features"]
        for kept_i in kept_indices:
            pos = kept[kept_i].position_index
            if pos >= feats.shape[0]:
                raise ValueError(
                    f"position_index {pos} >= seq_len {feats.shape[0]} for "
                    f"example {kept[kept_i].source_example_id!r}. Did you forget "
                    "--min-bullet-lines? Mining set must match the SFT filter."
                )
            H[kept_i].copy_(feats[pos].to(torch.float32))
        n_loaded += len(kept_indices)
        if n_loaded == n or (n_loaded % 4096) < len(kept_indices):
            logger.info(
                "Loaded %d/%d label-row activations in %.1fs",
                n_loaded, n, time.time() - t0,
            )
    return H


def _topk_cosine_chunked(
    *,
    H_norm_dev: torch.Tensor,
    episode_array_dev: torch.Tensor | None,
    ptype_array_dev: torch.Tensor | None,
    exclude_same_episode: bool,
    per_position_type: bool,
    top_k: int,
    chunk_size: int,
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(top_idx, top_vals)`` of shape ``[N, K]`` each.

    ``top_idx[i]`` are the K most cosine-similar row indices to row ``i``;
    ``top_vals[i]`` their cosine sims (descending).  Self and (optionally)
    same-episode and cross-ptype rows are masked to ``-inf`` *before* topk.

    When ``per_position_type=True`` the effective K is capped by the
    smallest in-pool count minus one; callers should be aware that
    ``out_idx`` may contain padding entries with ``-inf`` cosine.
    """
    n, _D = H_norm_dev.shape
    device = H_norm_dev.device
    K = min(top_k, max(0, n - 1))
    out_idx = torch.empty(n, K, dtype=torch.long)
    out_vals = torch.empty(n, K, dtype=torch.float32)

    t_mm = time.time()
    n_done = 0
    with torch.inference_mode():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = H_norm_dev[start:end]                              # (c, D)
            sims = chunk @ H_norm_dev.T                                # (c, N)
            sims[torch.arange(end - start, device=device),
                 torch.arange(start, end, device=device)] = float("-inf")
            if exclude_same_episode and episode_array_dev is not None:
                anchor_ep = episode_array_dev[start:end].unsqueeze(1)  # (c, 1)
                ep_row = episode_array_dev.unsqueeze(0)                # (1, N)
                mask = (anchor_ep != -1) & (ep_row == anchor_ep)
                sims = sims.masked_fill(mask, float("-inf"))
            if per_position_type and ptype_array_dev is not None:
                anchor_pt = ptype_array_dev[start:end].unsqueeze(1)    # (c, 1)
                pt_row = ptype_array_dev.unsqueeze(0)                  # (1, N)
                sims = sims.masked_fill(anchor_pt != pt_row, float("-inf"))
            top_vals, top_idx = torch.topk(sims, k=K, dim=-1)
            out_idx[start:end] = top_idx.cpu()
            out_vals[start:end] = top_vals.float().cpu()
            n_done = end
            if n_done % log_every == 0 or n_done == n:
                logger.info(
                    "Cosine top-%d for %d/%d anchors in %.1fs",
                    K, n_done, n, time.time() - t_mm,
                )
    return out_idx, out_vals


def _sample_random_same_ptype(
    *,
    kept: list[_KeptAnchor],
    pool_indices: list[int],
    anchor_kept_idx: int,
    K: int,
    exclude_same_episode: bool,
    rng: random.Random,
) -> list[int]:
    """Return up to K kept-indices sampled uniformly from the same-ptype pool.

    Excludes self always; excludes same-episode rows when requested.
    Uses rejection sampling so it stays ``O(K)`` per anchor even when the
    pool is large (the alternative -- enumerating eligible candidates --
    would be ``O(|pool|^2)`` over all anchors).
    """
    if K <= 0 or not pool_indices:
        return []
    anchor = kept[anchor_kept_idx]
    anchor_ep = anchor.episode_index
    chosen: list[int] = []
    seen: set[int] = {anchor_kept_idx}
    max_attempts = K * 20 + 32
    attempts = 0
    pool_n = len(pool_indices)
    while len(chosen) < K and attempts < max_attempts:
        cand = pool_indices[rng.randrange(pool_n)]
        attempts += 1
        if cand in seen:
            continue
        if (exclude_same_episode and anchor_ep is not None
                and kept[cand].episode_index == anchor_ep):
            continue
        chosen.append(cand)
        seen.add(cand)
    return chosen


def _build_negs_for_anchor(
    *,
    kept: list[_KeptAnchor],
    anchor_kept_idx: int,
    top_idx_row: list[int],
    top_vals_row: list[float],
    K: int,
    jaccard_cap: float | None,
    anchor_tokens: set[str] | None,
    token_cache: dict[int, set[str]],
) -> tuple[list[int], list[float], int]:
    """Walk the candidate list and return up to K (idx, cos) admitted by Jaccard.

    Returns ``(neg_kept_indices, cos_values, n_jaccard_dropped)``.
    """
    out_idx: list[int] = []
    out_cos: list[float] = []
    n_dropped = 0
    for j, c in zip(top_idx_row, top_vals_row):
        if len(out_idx) >= K:
            break
        if c == float("-inf"):
            continue
        j = int(j)
        if jaccard_cap is not None and anchor_tokens is not None:
            tok = token_cache.get(j)
            if tok is None:
                tok = _caption_tokens(kept[j].description)
                token_cache[j] = tok
            jac = _jaccard(anchor_tokens, tok)
            if jac > jaccard_cap:
                n_dropped += 1
                continue
        out_idx.append(j)
        out_cos.append(float(c))
    return out_idx, out_cos, n_dropped


def _cosines_for_pairs(
    *,
    H_norm_dev: torch.Tensor,
    anchor_indices: list[int],
    neg_indices_per_anchor: list[list[int]],
    chunk_size: int = 4096,
) -> list[list[float]]:
    """Compute cosine sim between each anchor and its sampled negatives.

    All inputs are kept-row indices into ``H_norm_dev``. Returns one list of
    cosines per anchor, aligned with ``neg_indices_per_anchor``.
    """
    out: list[list[float]] = []
    if not anchor_indices:
        return out
    device = H_norm_dev.device
    with torch.inference_mode():
        for start in range(0, len(anchor_indices), chunk_size):
            end = min(start + chunk_size, len(anchor_indices))
            a_block = anchor_indices[start:end]
            a_vecs = H_norm_dev[torch.tensor(a_block, device=device)]
            for local_i, anchor_idx in enumerate(a_block):
                negs = neg_indices_per_anchor[start + local_i]
                if not negs:
                    out.append([])
                    continue
                n_vecs = H_norm_dev[torch.tensor(negs, device=device)]
                sims = (a_vecs[local_i:local_i + 1] @ n_vecs.T).squeeze(0).float().cpu().tolist()
                out.append([float(s) for s in sims])
    return out


def _write_jsonl(
    *,
    out_path: Path,
    kept: list[_KeptAnchor],
    rows_to_write: list[dict],
) -> dict[str, list[float]]:
    """Write the mining JSONL.

    Returns ``cos_top1_by_ptype`` (per-ptype list of top-1 cosines, for
    diagnostics). Anchors with empty negs (e.g. last_text strategy=drop, or
    saturated jaccard filter) contribute no top-1 entry.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cos_top1_by_ptype: dict[str, list[float]] = defaultdict(list)
    with out_path.open("w") as f:
        for row in rows_to_write:
            if row["cos"]:
                cos_top1_by_ptype[row["position_type"]].append(float(row["cos"][0]))
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return cos_top1_by_ptype


# Healthy band Agent 5 uses for the per-ptype `median_cos_top1` check.
_HEALTHY_COS_LOW = 0.60
_HEALTHY_COS_HIGH = 0.95


def _quantile(vals: list[float], q: float) -> float:
    """Approximate quantile using ``torch.kthvalue`` (no extra deps)."""
    if not vals:
        return float("nan")
    t = torch.tensor(vals, dtype=torch.float64)
    k = max(1, min(len(t), int(round(q * len(t)))))
    return float(t.kthvalue(k).values.item())


def _print_diagnostics(
    *,
    cos_top1_by_ptype: dict[str, list[float]],
    n_anchors: int,
    eff_K: int,
    per_position_type: bool,
    n_anchors_with_empty_negs: int,
    n_jaccard_dropped: int,
    strategy_counts: dict[str, int],
) -> None:
    cos_top1_all = [c for lst in cos_top1_by_ptype.values() for c in lst]
    if not cos_top1_all:
        print(
            "[mine_hard_negatives] WARNING: no anchor produced any admissible "
            "negative. Try --no-exclude-same-episode or use a larger corpus.",
            file=sys.stderr,
        )
        return
    median = _quantile(cos_top1_all, 0.50)
    p5 = _quantile(cos_top1_all, 0.05)
    p95 = _quantile(cos_top1_all, 0.95)
    print(
        f"[mine_hard_negatives] n_anchors={n_anchors}  K={eff_K}  "
        f"median_cos_top1={median:.3f}  p5={p5:.3f}  p95={p95:.3f}  "
        f"n_anchors_with_empty_negs={n_anchors_with_empty_negs}  "
        f"n_jaccard_dropped={n_jaccard_dropped}  "
        f"strategy_counts={dict(strategy_counts)}",
        file=sys.stderr,
    )
    if per_position_type:
        for pt in sorted(cos_top1_by_ptype.keys()):
            vals = cos_top1_by_ptype[pt]
            if not vals:
                print(
                    f"[mine_hard_negatives] per-ptype: {pt} n=0 (no admissible negs)",
                    file=sys.stderr,
                )
                continue
            pt_med = _quantile(vals, 0.50)
            pt_p5 = _quantile(vals, 0.05)
            pt_p95 = _quantile(vals, 0.95)
            print(
                f"[mine_hard_negatives] per-ptype: {pt} n={len(vals)} "
                f"median_cos_top1={pt_med:.3f} p5={pt_p5:.3f} p95={pt_p95:.3f}",
                file=sys.stderr,
            )
            if pt_med < _HEALTHY_COS_LOW or pt_med > _HEALTHY_COS_HIGH:
                print(
                    f"[mine_hard_negatives] WARNING: per-ptype median_cos_top1 for "
                    f"{pt}={pt_med:.3f} is OUTSIDE healthy band "
                    f"[{_HEALTHY_COS_LOW:.2f}, {_HEALTHY_COS_HIGH:.2f}] — mining "
                    f"is likely saturated or unreliable for this ptype.",
                    file=sys.stderr,
                )
    else:
        if median > _HEALTHY_COS_HIGH:
            print(
                "[mine_hard_negatives] WARNING: median_cos_top1 > 0.95. "
                "Activations are very uniform; hard-neg mining will be near-uniform "
                "noise. Investigate label diversity or extraction layer, or pass "
                "--per-position-type to segment by ptype.",
                file=sys.stderr,
            )
        if median < 0.30:
            print(
                "[mine_hard_negatives] WARNING: median_cos_top1 < 0.3. Activations "
                "are very far apart; hard-neg may not help on this corpus.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    out_path = Path(args.out)

    if args.jaccard_cap is not None:
        if not (0.0 <= float(args.jaccard_cap) <= 1.0):
            logger.error("--jaccard-cap must be in [0, 1]; got %r", args.jaccard_cap)
            return 2

    kept, reader = _build_kept_anchors(
        activations_root=args.activations_root,
        labels_jsonl=args.labels_jsonl,
        min_bullet_lines=args.min_bullet_lines,
    )
    n = len(kept)
    if n == 0:
        logger.error("No kept labels after filtering. Nothing to mine.")
        return 2
    if int(args.top_k) <= 0:
        logger.error("--top-k must be >= 1; got %d", args.top_k)
        return 2

    # Inventory ptypes (for per-ptype mask + last_text strategy routing).
    indices_by_ptype: dict[str, list[int]] = defaultdict(list)
    for i, ka in enumerate(kept):
        indices_by_ptype[ka.position_type].append(i)
    logger.info(
        "Position-type inventory: %s",
        {pt: len(v) for pt, v in sorted(indices_by_ptype.items())},
    )

    H = _materialize_h_matrix(kept=kept, reader=reader)
    H_norm = torch.nn.functional.normalize(H, dim=-1)
    del H

    dtype = {
        "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)
    H_norm_dev = H_norm.to(device=device, dtype=dtype)

    episode_array_dev: torch.Tensor | None = None
    if args.exclude_same_episode:
        episode_array_dev = torch.tensor(
            [-1 if ka.episode_index is None else ka.episode_index for ka in kept],
            dtype=torch.long, device=device,
        )

    ptype_array_dev: torch.Tensor | None = None
    if args.per_position_type:
        ptype_to_id = {pt: i for i, pt in enumerate(sorted(indices_by_ptype.keys()))}
        ptype_array_dev = torch.tensor(
            [ptype_to_id[ka.position_type] for ka in kept],
            dtype=torch.long, device=device,
        )

    # Oversample top-K when --jaccard-cap is on so we have replacements for
    # dropped candidates. Anchors whose ptype uses random_same_ptype or drop
    # don't need the topk pool, but we still compute it once for the rest.
    target_K = int(args.top_k)
    mining_K = target_K
    if args.jaccard_cap is not None:
        mining_K = max(target_K, target_K * max(1, int(args.jaccard_oversample)))
    mining_K = max(1, min(mining_K, max(1, n - 1)))

    top_idx, top_vals = _topk_cosine_chunked(
        H_norm_dev=H_norm_dev,
        episode_array_dev=episode_array_dev,
        ptype_array_dev=ptype_array_dev,
        exclude_same_episode=args.exclude_same_episode,
        per_position_type=bool(args.per_position_type),
        top_k=mining_K,
        chunk_size=max(1, int(args.chunk_size)),
        log_every=int(args.log_every),
    )
    if mining_K < target_K:
        logger.warning(
            "--top-k=%d but pool only supports %d candidates per anchor; "
            "will emit up to that many.",
            target_K, mining_K,
        )

    # Resolve per-anchor strategies + assemble the rows. We first walk the
    # topk_cosine path for everyone, then override last_text anchors with
    # the requested last_text strategy.
    last_text_strategy = str(args.last_text_strategy)
    strategy_counts: dict[str, int] = defaultdict(int)
    n_anchors_with_empty_negs = 0
    n_jaccard_dropped_total = 0
    token_cache: dict[int, set[str]] = {}

    # Pre-compute random_same_ptype assignments and anchors that need cos
    # computed for those random partners; we batch the cos calc afterwards.
    random_anchor_kept_idx: list[int] = []
    random_neg_indices_per_anchor: list[list[int]] = []
    rng = random.Random(0xC0FFEE ^ int(args.top_k) ^ int(n))

    rows_to_write: list[dict] = []
    top_idx_list = top_idx.tolist()
    top_vals_list = top_vals.tolist()

    for kept_i in range(n):
        ka = kept[kept_i]
        if ka.position_type == "last_text":
            strategy = last_text_strategy
        else:
            strategy = "topk_cosine"

        if strategy == "drop":
            strategy_counts[strategy] += 1
            n_anchors_with_empty_negs += 1
            rows_to_write.append({
                "anchor": ka.label_id,
                "negs": [],
                "cos": [],
                "anchor_episode": ka.episode_index,
                "position_type": ka.position_type,
                "strategy": strategy,
            })
            continue

        if strategy == "random_same_ptype":
            strategy_counts[strategy] += 1
            pool = indices_by_ptype.get(ka.position_type, [])
            chosen = _sample_random_same_ptype(
                kept=kept,
                pool_indices=pool,
                anchor_kept_idx=kept_i,
                K=target_K,
                exclude_same_episode=bool(args.exclude_same_episode),
                rng=rng,
            )
            random_anchor_kept_idx.append(kept_i)
            random_neg_indices_per_anchor.append(chosen)
            rows_to_write.append({
                "anchor": ka.label_id,
                "negs": [kept[j].label_id for j in chosen],
                # cos placeholder, filled in below
                "cos": [None] * len(chosen),
                "anchor_episode": ka.episode_index,
                "position_type": ka.position_type,
                "strategy": strategy,
            })
            if not chosen:
                n_anchors_with_empty_negs += 1
            continue

        # default: topk_cosine
        strategy_counts["topk_cosine"] += 1
        anchor_tokens = None
        if args.jaccard_cap is not None:
            anchor_tokens = _caption_tokens(ka.description)
        neg_kept_indices, cos_values, n_drop = _build_negs_for_anchor(
            kept=kept,
            anchor_kept_idx=kept_i,
            top_idx_row=top_idx_list[kept_i],
            top_vals_row=top_vals_list[kept_i],
            K=target_K,
            jaccard_cap=float(args.jaccard_cap) if args.jaccard_cap is not None else None,
            anchor_tokens=anchor_tokens,
            token_cache=token_cache,
        )
        n_jaccard_dropped_total += n_drop
        if not neg_kept_indices:
            n_anchors_with_empty_negs += 1
        rows_to_write.append({
            "anchor": ka.label_id,
            "negs": [kept[j].label_id for j in neg_kept_indices],
            "cos": cos_values,
            "anchor_episode": ka.episode_index,
            "position_type": ka.position_type,
            "strategy": "topk_cosine",
        })

    # Fill in cosines for random_same_ptype anchors in a batched GPU pass.
    if random_anchor_kept_idx:
        logger.info(
            "Computing cosines for %d random_same_ptype anchors ...",
            len(random_anchor_kept_idx),
        )
        cos_per_anchor = _cosines_for_pairs(
            H_norm_dev=H_norm_dev,
            anchor_indices=random_anchor_kept_idx,
            neg_indices_per_anchor=random_neg_indices_per_anchor,
        )
        # Map (anchor_kept_idx -> row index in rows_to_write) is identity by
        # construction.
        random_lookup = dict(zip(random_anchor_kept_idx, cos_per_anchor))
        for kept_i, row in enumerate(rows_to_write):
            if row["strategy"] == "random_same_ptype":
                row["cos"] = list(random_lookup.get(kept_i, []))

    cos_top1_by_ptype = _write_jsonl(
        out_path=out_path,
        kept=kept,
        rows_to_write=rows_to_write,
    )
    _print_diagnostics(
        cos_top1_by_ptype=cos_top1_by_ptype,
        n_anchors=n,
        eff_K=target_K,
        per_position_type=bool(args.per_position_type),
        n_anchors_with_empty_negs=n_anchors_with_empty_negs,
        n_jaccard_dropped=n_jaccard_dropped_total,
        strategy_counts=strategy_counts,
    )

    logger.info("Wrote %d rows to %s", n, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

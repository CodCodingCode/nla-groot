#!/usr/bin/env python
"""V4 image-patch A/B sweep — proxy-eval driver (Stages 1c + 2 + 3).

For a given ``(layer, position_strategy)`` extraction config produce one JSON
row of cheap proxy metrics measured on raw activations (no SFT, no AR/AV
training). These metrics are the basis on which we pick the winning
extraction config for V4 SFT.

The four metrics (per the plan)
-------------------------------

* ``knn_caption_at1`` / ``knn_caption_at5``: For each anchor ``h_i`` find
  its 1- or top-5 nearest neighbour(s) in cosine space (excluding self
  and same-episode). Report the mean Jaccard similarity of the
  bag-of-words captions for those neighbours. Higher = ``h`` is more
  aligned with caption content (i.e. extraction surfaces signal that
  AV will be able to verbalise). A random-pair baseline is reported
  alongside; the *lift* is what matters.

* ``suite_probe_acc``: Logistic probe predicting ``suite ∈ {goal,
  spatial, object, libero_10}`` from ``h`` alone, episode-stratified
  train/val split. Re-uses ``fit_linear_probe`` from
  ``scripts/eval/probe_h_attributes.py``.

* ``same_ep_cosine_gap``: Mean cosine of same-episode pairs minus mean
  cosine of cross-episode pairs. Measures scene-specificity in raw
  vector space.

* ``anisotropy_floor``: Median pairwise cosine over a random sample of
  pairs. Higher = activations cluster in a narrower cone (worse for
  contrastive training).

Layer axis
----------

* ``--layer 16`` (default for "cached" mode): read existing on-disk
  activations from ``--activations-root`` and apply the chosen
  ``position_strategy`` over the stored ``[T, H]`` block. No GPU needed.
* ``--layer 8`` / ``--layer 12``: re-run the GR00T backbone forward on
  the same examples and capture decoder-block output via
  ``IntermediateLayerHook``. Requires the per-suite GR00T checkpoint
  and the source LeRobot dataset. We avoid re-extracting layer 16 in
  this path: it's already on disk.

CLI (driver mode)
-----------------

Run all 12 configs in one go (slow path, includes GR00T forwards)::

    PYTHONPATH=src .venv/bin/python scripts/eval/probe_extraction_sweep.py \\
        --layers 8 12 16 \\
        --strategies random_one mean_pool_image strided_image center_image \\
        --n-samples 1000 \\
        --out-root data/sft/libero_4suite_v3/extraction_sweep

Run only the cached layer-16 configs (fast, no GPU)::

    PYTHONPATH=src .venv/bin/python scripts/eval/probe_extraction_sweep.py \\
        --layers 16 --strategies random_one mean_pool_image strided_image center_image \\
        --n-samples 1000 \\
        --out-root data/sft/libero_4suite_v3/extraction_sweep

Run a single named config (workhorse used by the driver loop)::

    PYTHONPATH=src .venv/bin/python scripts/eval/probe_extraction_sweep.py \\
        --layer 8 --strategy mean_pool_image --n-samples 1000 \\
        --out-json data/sft/libero_4suite_v3/extraction_sweep/8__mean_pool_image.json

Outputs
-------
One JSON per config under ``--out-root``::

    {
      "config": {"layer": 16, "strategy": "mean_pool_image"},
      "n_samples": 1000,
      "metrics": {
        "knn_caption_at1":   {"value": 0.41, "baseline_random": 0.12, "lift": 0.29},
        "knn_caption_at5":   {"value": 0.35, "baseline_random": 0.12, "lift": 0.23},
        "suite_probe_acc":   {"value": 0.81, "macro_f1": 0.80,
                              "majority": 0.27, "n_train": 800, "n_val": 200},
        "same_ep_cosine_gap":{"same": 0.97, "cross": 0.96, "gap": 0.01},
        "anisotropy_floor":  {"value": 0.94, "n_pairs": 2000}
      },
      ...
    }
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

logger = logging.getLogger("nla.v4_sweep")


# ---------------------------------------------------------------------------
# Sample selection.
# ---------------------------------------------------------------------------

_TRAJ_STEP_RE = re.compile(r"^(?P<suite>[a-zA-Z0-9_]+?)__traj(?P<traj>\d+)_step(?P<step>\d+)$")


@dataclass
class SampleSpec:
    """One sample's pointer into both the cached shards and the source dataset."""

    source_example_id: str          # e.g. "goal__traj000001_step000038"
    suite: str                      # e.g. "goal"
    traj_id: int
    step_idx: int
    position_index: int             # the image_patch token chosen at labeling time
    caption: str                    # gold caption
    episode_index: int | None


def _parse_source_id(source_example_id: str) -> tuple[str, int, int] | None:
    m = _TRAJ_STEP_RE.match(source_example_id)
    if m is None:
        return None
    return m["suite"], int(m["traj"]), int(m["step"])


def select_balanced_samples(
    activations_root: Path,
    labels_jsonl: Path,
    n_per_suite: int,
    *,
    seed: int = 0,
    min_bullet_lines: int = 3,
    suites: tuple[str, ...] = ("goal", "spatial", "object", "10"),
    position_types: tuple[str, ...] = ("image_patch",),
) -> list[SampleSpec]:
    """Choose ``n_per_suite`` random image-patch labels per suite, balanced."""
    from nla.extraction.storage import ActivationShardReader
    from nla.training.dataset import load_labels_jsonl

    reader = ActivationShardReader(str(activations_root))
    rec_by_id = {rec.example_id: rec for rec in reader.records}

    labels = load_labels_jsonl(str(labels_jsonl), min_bullet_lines=min_bullet_lines)

    pool_by_suite: dict[str, list[SampleSpec]] = defaultdict(list)
    n_unparsed = 0
    n_orphan = 0
    for entry in labels:
        if entry.position_type not in position_types:
            continue
        suite = entry.raw.get("meta", {}).get("suite") or ""
        if suite not in suites:
            continue
        rec = rec_by_id.get(entry.source_example_id)
        if rec is None:
            n_orphan += 1
            continue
        parsed = _parse_source_id(entry.source_example_id)
        if parsed is None:
            n_unparsed += 1
            continue
        _, traj_id, step_idx = parsed
        pool_by_suite[suite].append(SampleSpec(
            source_example_id=entry.source_example_id,
            suite=suite,
            traj_id=traj_id,
            step_idx=step_idx,
            position_index=int(entry.position_index),
            caption=entry.description,
            episode_index=None if rec.episode_index is None else int(rec.episode_index),
        ))

    rng = random.Random(seed)
    chosen: list[SampleSpec] = []
    for suite in suites:
        pool = pool_by_suite.get(suite, [])
        rng.shuffle(pool)
        take = pool[: int(n_per_suite)]
        logger.info("suite=%s pool=%d take=%d", suite, len(pool), len(take))
        chosen.extend(take)
    logger.info("Selected %d total samples (orphan=%d, unparsed=%d).",
                len(chosen), n_orphan, n_unparsed)
    return chosen


# ---------------------------------------------------------------------------
# Layer-16: cached path.
# ---------------------------------------------------------------------------

@dataclass
class LayerTHCache:
    """In-memory ``[T, H]`` + masks cache for one layer's forward outputs.

    Keys are source_example_ids; values are tuples of CPU torch tensors.
    Used so that applying multiple position-strategies to the same layer's
    activations does not require redoing the (expensive) GR00T forward
    pass.
    """

    layer: int
    by_example: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def has(self, example_id: str) -> bool:
        return example_id in self.by_example


def build_layer16_cache_from_disk(
    activations_root: Path,
    samples: list[SampleSpec],
) -> LayerTHCache:
    """Stream the on-disk shards once and stash each example's ``[T, H]``."""
    from nla.extraction.storage import ActivationShardReader

    reader = ActivationShardReader(str(activations_root))
    needed = {s.source_example_id for s in samples}
    record_filter = lambda rec: rec.example_id in needed  # noqa: E731

    by_example: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for item in reader.iter_examples(record_filter=record_filter):
        rec = item["_record"]
        by_example[rec.example_id] = (
            item["features"].detach().cpu().to(torch.float32).contiguous(),
            item["image_mask"].detach().cpu().bool().contiguous(),
            item["attention_mask"].detach().cpu().bool().contiguous(),
        )
    logger.info("Layer 16 cache built from disk: %d examples.", len(by_example))
    return LayerTHCache(layer=16, by_example=by_example)


def apply_strategy_over_cache(
    cache: LayerTHCache,
    samples: list[SampleSpec],
    strategy: str,
    *,
    seed: int = 0,
    strided_k: int = 4,
) -> tuple[np.ndarray, list[SampleSpec]]:
    """Apply ``strategy`` to every cached ``[T, H]`` and return ``[N, H]``.

    Returns the stacked array and the list of samples whose cache entry
    survived (mirrors the order of the array).
    """
    from nla.extraction.position_strategies import apply as apply_strategy

    rng = np.random.default_rng(seed)
    out_vecs: list[np.ndarray] = []
    keep_samples: list[SampleSpec] = []
    for s in samples:
        slot = cache.by_example.get(s.source_example_id)
        if slot is None:
            continue
        features, image_mask, attention_mask = slot
        try:
            v = apply_strategy(
                strategy, features, image_mask, attention_mask,
                rng=rng, k=strided_k,
            )
        except ValueError as e:
            logger.debug("strategy %s skipped %s: %s", strategy, s.source_example_id, e)
            continue
        out_vecs.append(v.detach().cpu().to(torch.float32).numpy())
        keep_samples.append(s)
    h = (np.stack(out_vecs, axis=0).astype(np.float32, copy=False)
         if out_vecs else np.zeros((0,), dtype=np.float32))
    return h, keep_samples


# ---------------------------------------------------------------------------
# Layer-8 / layer-12: fresh-extraction path.
# ---------------------------------------------------------------------------

def build_forward_layer_caches(
    samples: list[SampleSpec],
    layer_indices: Iterable[int],
    *,
    checkpoint_template: str,
    dataset_template: str,
    device: str = "cuda:0",
    select_layer: int = 16,
) -> dict[int, LayerTHCache]:
    """One forward pass per example; capture every requested layer at once.

    Loads one ``Gr00tPolicy`` per suite, attaches the wrapper
    ``BackboneFeatureHook`` plus one ``IntermediateLayerHook(layer_idx)``
    for each ``layer_idx`` in ``layer_indices``, runs the backbone
    forward, and stashes the per-example ``[T, H]`` block for each
    requested layer (plus the shared masks). Returns
    ``{layer_idx: LayerTHCache}``.

    ``checkpoint_template``: ``str.format``-able with ``{suite}`` placeholder.
    ``dataset_template``: same convention.
    """
    from copy import deepcopy

    from nla.extraction._compat import apply_all as _apply_groot_compat
    _apply_groot_compat()
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    import tree as _tree

    from nla.extraction.hook import (
        BackboneFeatureHook,
        IntermediateLayerHook,
        attach_hooks,
    )

    embodiment_tag = EmbodimentTag.resolve("LIBERO_PANDA")
    layer_indices = sorted(set(int(x) for x in layer_indices))
    if not layer_indices:
        return {}

    caches: dict[int, LayerTHCache] = {
        L: LayerTHCache(layer=int(L), by_example={}) for L in layer_indices
    }

    by_suite: dict[str, list[SampleSpec]] = defaultdict(list)
    for s in samples:
        by_suite[s.suite].append(s)

    for suite, suite_samples in by_suite.items():
        ckpt = checkpoint_template.format(suite=suite)
        dataset_path = dataset_template.format(suite=suite)
        logger.info(
            "Loading GR00T policy for suite=%s from %s (probe_layers=%s, select_layer=%d)",
            suite, ckpt, layer_indices, select_layer,
        )
        t0 = time.time()
        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag, model_path=ckpt, device=device,
        )
        loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=policy.modality_configs,
            video_backend="torchcodec",
        )
        language_keys = list(policy.modality_configs["language"].modality_keys)
        modality_configs = deepcopy(policy.modality_configs)
        if "action" in modality_configs:
            modality_configs.pop("action")
        logger.info("  policy + loader ready in %.1fs", time.time() - t0)

        wrap_hook = BackboneFeatureHook(to_cpu=True, store_dtype=torch.float32)
        layer_hooks: list[IntermediateLayerHook] = [
            IntermediateLayerHook(layer_idx=L, to_cpu=True, store_dtype=torch.float32)
            for L in layer_indices
        ]

        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(attach_hooks(policy.model.backbone, wrap_hook))
            for lh in layer_hooks:
                stack.enter_context(lh.attach(policy.model.backbone))

            traj_cache: dict[int, object] = {}
            tF0 = time.time()
            for i, s in enumerate(suite_samples):
                if i and i % 25 == 0:
                    elapsed = time.time() - tF0
                    rate = i / max(elapsed, 1e-3)
                    logger.info(
                        "  suite=%s progress=%d/%d (%.2f ex/s)",
                        suite, i, len(suite_samples), rate,
                    )
                try:
                    traj = traj_cache.get(s.traj_id)
                    if traj is None:
                        traj = loader[s.traj_id]
                        if len(traj_cache) >= 4:
                            traj_cache.clear()
                        traj_cache[s.traj_id] = traj
                    observation = _build_observation(
                        traj, s.step_idx, modality_configs, embodiment_tag,
                        language_keys, extract_step_data,
                    )
                    _run_backbone_forward(policy, observation, _tree, MessageType)
                except Exception as e:
                    logger.warning("  forward failed for %s: %s", s.source_example_id, e)
                    continue

                if wrap_hook.last is None:
                    logger.warning("  no wrapper capture for %s", s.source_example_id)
                    continue
                image_mask = wrap_hook.last.image_mask[0].contiguous()
                attention_mask = wrap_hook.last.attention_mask[0].contiguous()

                for L, lh in zip(layer_indices, layer_hooks):
                    if lh.last is None:
                        continue
                    feats = lh.last[0]
                    if feats.shape[0] != image_mask.shape[0]:
                        logger.warning(
                            "  layer=%d T mismatch (feats=%d, masks=%d) for %s; skipping",
                            L, feats.shape[0], image_mask.shape[0], s.source_example_id,
                        )
                        continue
                    caches[L].by_example[s.source_example_id] = (
                        feats.contiguous(),
                        image_mask.bool(),
                        attention_mask.bool(),
                    )

        del policy
        torch.cuda.empty_cache()
    for L in layer_indices:
        logger.info("Layer %d cache: %d examples.", L, len(caches[L].by_example))
    return caches


def _build_observation(
    traj, step_idx, modality_configs, embodiment_tag, language_keys, extract_step_data,
):
    """Mirror of ``run_extract._prepare_step_obs`` for one (traj, step)."""
    data_point = extract_step_data(
        traj, step_idx, modality_configs, embodiment_tag, allow_padding=True,
    )
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for language_key in language_keys:
        obs[language_key] = data_point.text
    # Flat -> nested
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def _run_backbone_forward(policy, observation, tree_mod, MessageType):
    """Mirror of ``run_extract._run_backbone_forward``."""
    unbatched = policy._unbatch_observation(observation)
    processed = []
    for obs in unbatched:
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed.append(policy.processor(messages))
    collated = policy.collate_fn(processed)

    def _to_dtype(x):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(dtype=policy.model.dtype)
        return x

    collated = tree_mod.map_structure(_to_dtype, collated)
    if "inputs" in collated and "input_ids" not in collated:
        inner = collated["inputs"]
    else:
        inner = collated
    with torch.inference_mode():
        backbone_inputs, _ = policy.model.prepare_input(inner)
        _ = policy.model.backbone(backbone_inputs)
    return backbone_inputs


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

_CAPTION_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")
_CAPTION_STOPWORDS = frozenset({
    "the", "and", "for", "with", "are", "is", "of", "in", "to", "on",
    "at", "by", "an", "this", "that", "from", "into", "as", "be",
    # Bullet-list scaffolding from the labels.
    "language", "target", "scene", "spatial", "plan",
    "instruction", "specifies", "phase", "task", "active",
    "place", "places", "placing", "pickup", "and-place",
})


def _caption_tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in (m.group(0).lower() for m in _CAPTION_TOKEN_RE.finditer(text))
        if t not in _CAPTION_STOPWORDS
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def knn_caption_match(
    h: np.ndarray,
    captions: list[str],
    episode_ids: list[int],
    *,
    k_list: tuple[int, ...] = (1, 5),
    seed: int = 0,
    max_anchors: int = 1000,
) -> dict[str, dict]:
    """For each anchor, retrieve top-K cosine-NN and measure caption Jaccard.

    Returns ``{f"at{k}": {value, baseline_random, lift}, ...}``.
    Pairs that share ``episode_index`` are excluded from the NN pool so the
    metric measures cross-scene caption alignment, not trajectory-step
    overlap.
    """
    N, D = h.shape
    if N < 5:
        return {f"at{k}": {"value": 0.0, "baseline_random": 0.0, "lift": 0.0} for k in k_list}

    rng = np.random.default_rng(seed)
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    sims = hn @ hn.T

    tokens = [_caption_tokens(c) for c in captions]
    ep = np.asarray(episode_ids, dtype=np.int64)

    anchor_idx = np.arange(N)
    if N > max_anchors:
        anchor_idx = rng.choice(N, size=max_anchors, replace=False)
        anchor_idx.sort()

    per_anchor: dict[int, list[float]] = {k: [] for k in k_list}
    random_pair_jaccs: list[float] = []
    for i in anchor_idx:
        forbid = (ep == ep[i])
        scores = sims[i].copy()
        scores[i] = -np.inf
        scores[forbid] = -np.inf
        valid = np.where(np.isfinite(scores))[0]
        if valid.size < max(k_list):
            continue
        order = valid[np.argsort(-scores[valid])]
        for k in k_list:
            top = order[:k]
            jaccs = [_jaccard(tokens[i], tokens[j]) for j in top]
            per_anchor[k].append(float(np.mean(jaccs)) if jaccs else 0.0)
        # Random-pair baseline: pick one random index (different episode, not self)
        candidates = np.where(~forbid)[0]
        candidates = candidates[candidates != i]
        if candidates.size:
            j = int(rng.choice(candidates))
            random_pair_jaccs.append(_jaccard(tokens[i], tokens[j]))

    out: dict[str, dict] = {}
    for k in k_list:
        vals = per_anchor[k]
        v = float(np.mean(vals)) if vals else 0.0
        bl = float(np.mean(random_pair_jaccs)) if random_pair_jaccs else 0.0
        out[f"at{k}"] = {
            "value": v,
            "baseline_random": bl,
            "lift": v - bl,
            "n_anchors": int(len(vals)),
        }
    return out


def suite_probe_metric(
    h: np.ndarray,
    suites: list[str],
    episode_ids: list[int],
    *,
    seed: int = 0,
    held_out_fraction: float = 0.2,
) -> dict:
    try:
        from scripts.eval.probe_h_attributes import fit_linear_probe
    except ImportError:
        from eval.probe_h_attributes import fit_linear_probe  # type: ignore

    suites_arr = np.asarray(suites)
    ep = np.asarray(episode_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(ep)
    rng.shuffle(unique_eps)
    n_held = max(1, int(round(len(unique_eps) * held_out_fraction)))
    held_eps = set(unique_eps[:n_held].tolist())
    is_val = np.array([int(e) in held_eps for e in ep], dtype=bool)
    if is_val.sum() < 4 or (~is_val).sum() < 4:
        idx = np.arange(len(h))
        rng.shuffle(idx)
        n_val = max(2, int(round(len(idx) * held_out_fraction)))
        is_val = np.zeros(len(h), dtype=bool)
        is_val[idx[:n_val]] = True
    X_train, y_train = h[~is_val], suites_arr[~is_val]
    X_val, y_val = h[is_val], suites_arr[is_val]
    acc, f1, _ = fit_linear_probe(X_train, y_train, X_val, y_val, seed=seed)
    if len(y_val):
        _, counts = np.unique(y_val, return_counts=True)
        maj = float(counts.max() / counts.sum())
    else:
        maj = float("nan")
    return {
        "value": float(acc),
        "macro_f1": float(f1),
        "majority": maj,
        "n_train": int((~is_val).sum()),
        "n_val": int(is_val.sum()),
    }


def episode_cosine_gap_metric(
    h: np.ndarray, episode_ids: list[int],
    *, n_pairs: int = 500, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    ep = np.asarray(episode_ids, dtype=np.int64)
    ep_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(ep.tolist()):
        ep_to_idx[int(e)].append(i)
    eps = [e for e, ix in ep_to_idx.items() if len(ix) >= 2 and e != -1]
    if not eps:
        return {"same": float("nan"), "cross": float("nan"), "gap": float("nan")}

    same_pairs = []
    attempts = 0
    while len(same_pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        e = eps[rng.integers(0, len(eps))]
        ix = ep_to_idx[e]
        a, b = rng.choice(ix, size=2, replace=False)
        same_pairs.append((int(a), int(b)))
    cross_pairs = []
    attempts = 0
    while len(cross_pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        a, b = rng.choice(len(h), size=2, replace=False)
        if int(ep[a]) != int(ep[b]):
            cross_pairs.append((int(a), int(b)))
    if not same_pairs or not cross_pairs:
        return {"same": float("nan"), "cross": float("nan"), "gap": float("nan")}
    sa = np.asarray([p[0] for p in same_pairs]); sb = np.asarray([p[1] for p in same_pairs])
    ca = np.asarray([p[0] for p in cross_pairs]); cb = np.asarray([p[1] for p in cross_pairs])
    same_cos = (hn[sa] * hn[sb]).sum(axis=1)
    cross_cos = (hn[ca] * hn[cb]).sum(axis=1)
    return {
        "same": float(same_cos.mean()),
        "cross": float(cross_cos.mean()),
        "gap": float(same_cos.mean() - cross_cos.mean()),
        "n_pairs": int(min(len(same_pairs), len(cross_pairs))),
    }


def anisotropy_floor_metric(
    h: np.ndarray, *, n_pairs: int = 2000, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
    N = len(h)
    if N < 4:
        return {"value": float("nan"), "n_pairs": 0}
    a = rng.integers(0, N, size=n_pairs)
    b = rng.integers(0, N, size=n_pairs)
    same = (a == b)
    if same.any():
        # Replace self-pairs with a deterministic shift.
        b[same] = (b[same] + 1) % N
    cos = (hn[a] * hn[b]).sum(axis=1)
    return {
        "value": float(np.median(cos)),
        "mean": float(cos.mean()),
        "p95": float(np.percentile(cos, 95)),
        "n_pairs": int(n_pairs),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def evaluate_config(
    layer: int,
    strategy: str,
    samples: list[SampleSpec],
    layer_caches: dict[int, LayerTHCache],
    *,
    args: argparse.Namespace,
) -> dict:
    """Gather ``h`` for one (layer, strategy) and compute the proxy metrics."""
    if layer not in layer_caches:
        return {
            "config": {"layer": layer, "strategy": strategy},
            "n_samples": 0,
            "error": f"no cache built for layer {layer}",
        }
    t0 = time.time()
    h, local_samples = apply_strategy_over_cache(
        layer_caches[layer], samples, strategy,
        seed=args.seed, strided_k=args.strided_k,
    )
    apply_s = time.time() - t0
    logger.info(
        "config layer=%d strategy=%s apply=%.1fs N=%d",
        layer, strategy, apply_s, len(h),
    )
    if len(h) < 10:
        return {
            "config": {"layer": layer, "strategy": strategy},
            "n_samples": int(len(h)),
            "error": "too few samples; skipped",
            "apply_seconds": apply_s,
        }

    captions = [s.caption for s in local_samples]
    suites = [s.suite for s in local_samples]
    episodes = [(-1 if s.episode_index is None else int(s.episode_index))
                for s in local_samples]

    t1 = time.time()
    knn = knn_caption_match(h, captions, episodes,
                            k_list=(1, 5), seed=args.seed, max_anchors=args.knn_max_anchors)
    sp = suite_probe_metric(h, suites, episodes, seed=args.seed)
    eg = episode_cosine_gap_metric(h, episodes,
                                   n_pairs=args.episode_pairs, seed=args.seed)
    an = anisotropy_floor_metric(h, n_pairs=args.anisotropy_pairs, seed=args.seed)
    metrics_s = time.time() - t1
    logger.info("  metrics=%.1fs knn@1=%.3f suite=%.3f gap=%.3f aniso=%.3f",
                metrics_s, knn["at1"]["value"], sp["value"], eg["gap"], an["value"])

    return {
        "config": {"layer": layer, "strategy": strategy},
        "n_samples": int(len(h)),
        "metrics": {
            "knn_caption_at1":   knn["at1"],
            "knn_caption_at5":   knn["at5"],
            "suite_probe_acc":   sp,
            "same_ep_cosine_gap": eg,
            "anisotropy_floor":  an,
        },
        "timing": {
            "apply_seconds": apply_s,
            "metrics_seconds": metrics_s,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations-root", type=Path,
                   default=REPO / "data/activations/libero_4suite_combined")
    p.add_argument("--labels-jsonl", type=Path,
                   default=REPO / "data/labels/libero_4suite_combined/labels.jsonl")
    p.add_argument("--out-root", type=Path,
                   default=REPO / "data/sft/libero_4suite_v3/extraction_sweep")
    p.add_argument("--out-json", type=Path, default=None,
                   help="If set, run a single (--layer, --strategy) config and write here.")
    p.add_argument("--layers", type=int, nargs="+", default=[16])
    p.add_argument("--strategies", type=str, nargs="+",
                   default=["random_one", "mean_pool_image",
                            "strided_image", "center_image"])
    p.add_argument("--layer", type=int, default=None,
                   help="(workhorse mode) single layer; used with --strategy")
    p.add_argument("--strategy", type=str, default=None,
                   help="(workhorse mode) single strategy; used with --layer")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Total samples (split equally across the 4 suites).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strided-k", type=int, default=4)
    p.add_argument("--knn-max-anchors", type=int, default=1000)
    p.add_argument("--episode-pairs", type=int, default=500)
    p.add_argument("--anisotropy-pairs", type=int, default=2000)
    p.add_argument("--min-bullet-lines", type=int, default=3)
    p.add_argument("--checkpoint-template", type=str,
                   default=str(REPO / "checkpoints/GR00T-N1.7-LIBERO/libero_{suite}"),
                   help="Path template for the per-suite GR00T checkpoint.")
    p.add_argument("--dataset-template", type=str,
                   default=str(REPO / "third_party/Isaac-GR00T/examples/LIBERO/libero_{suite}_no_noops_1.0.0_lerobot"),
                   help="Path template for the per-suite LeRobot dataset.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--select-layer", type=int, default=16,
                   help="GR00T wrapper select_layer; must be > max requested probe layer.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a (layer, strategy) config if its JSON already exists.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.layer is not None and args.strategy is not None:
        configs: list[tuple[int, str]] = [(int(args.layer), str(args.strategy))]
    else:
        configs = [(int(L), str(S)) for L in args.layers for S in args.strategies]

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    n_per_suite = max(1, args.n_samples // 4)
    samples = select_balanced_samples(
        args.activations_root, args.labels_jsonl,
        n_per_suite=n_per_suite, seed=args.seed,
        min_bullet_lines=args.min_bullet_lines,
    )
    if not samples:
        logger.error("No samples selected; aborting.")
        return 2

    requested_layers = sorted({L for L, _ in configs})
    layer_caches: dict[int, LayerTHCache] = {}

    if 16 in requested_layers:
        logger.info("Building layer-16 cache from on-disk activations...")
        t0 = time.time()
        layer_caches[16] = build_layer16_cache_from_disk(args.activations_root, samples)
        logger.info("Layer-16 cache built in %.1fs", time.time() - t0)

    forward_layers = [L for L in requested_layers if L != 16]
    if forward_layers:
        logger.info(
            "Running GR00T forward to build caches for layers %s "
            "(one pass per example, all layers captured in parallel)...",
            forward_layers,
        )
        t0 = time.time()
        fwd_caches = build_forward_layer_caches(
            samples=samples,
            layer_indices=forward_layers,
            checkpoint_template=args.checkpoint_template,
            dataset_template=args.dataset_template,
            device=args.device,
            select_layer=args.select_layer,
        )
        layer_caches.update(fwd_caches)
        logger.info("Forward-layer caches built in %.1fs", time.time() - t0)

    results = []
    for layer, strategy in configs:
        out_path = (args.out_json
                    if args.out_json and len(configs) == 1
                    else out_root / f"{layer}__{strategy}.json")
        if args.skip_existing and out_path.exists():
            logger.info("skip-existing: %s already present", out_path)
            try:
                results.append(json.loads(out_path.read_text()))
            except Exception:
                pass
            continue
        row = evaluate_config(layer, strategy, samples, layer_caches, args=args)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2) + "\n")
        logger.info("wrote %s", out_path)
        results.append(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())

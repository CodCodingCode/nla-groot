#!/usr/bin/env python
"""Linear/MLP probes on the GR00T residual ``h`` for scene attributes.

For each (attribute, position_type, probe_kind) triple this script:

  1. Joins ``LabeledPositionDataset`` rows with attributes.jsonl on
     ``(source_example_id, position_index)``.
  2. Splits episodes (held_out_fraction=0.1) and stratifies by attribute.
  3. Trains a linear (sklearn ``LogisticRegression``) and a 1-hidden-layer
     MLP probe on the train side, evaluates top-1 accuracy and macro-F1 on
     the held-out val side.
  4. Computes a majority-class baseline and -- if ``--av-samples-jsonl`` is
     passed -- the AV verbalisation accuracy (extract the same attribute
     from AV's caption with the same regex extractors).

The output is a JSONL row per probe + a markdown table for the paper.

This is the headline interpretability number: the probe accuracy is the
**upper bound** on what AV could ever verbalise about each attribute.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/probe_h_attributes.py \\
        --activations-root data/activations/libero_goal_pilot \\
        --labels-jsonl     data/labels/libero_goal_pilot/labels.jsonl \\
        --attributes-jsonl data/eval/attributes.jsonl \\
        --attributes target_object_class gripper_state scene_type target_visible task_phase \\
        --probe-kind both \\
        --out-jsonl data/eval/probe_results.jsonl \\
        --out-md    data/eval/probe_table.md \\
        --max-rows-per-attr 5000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# Local imports kept lazy where they pull in heavy deps (torch, sklearn) so
# that ``--help`` stays import-cheap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Re-use the deterministic attribute extractors so AV captions are scored
# with the *same* mapping as the gold labels.
try:
    from scripts.eval.extract_attributes import (
        extract_attributes as _extract_attrs_from_caption,
        known_attributes as _known_attrs,
    )
except ImportError:
    # When invoked as ``python scripts/eval/probe_h_attributes.py`` the
    # ``scripts`` package isn't always on the path; fall back to a relative
    # import from the file's directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_attributes import (  # type: ignore
        extract_attributes as _extract_attrs_from_caption,
        known_attributes as _known_attrs,
    )


logger = logging.getLogger("nla.probe_h")


PROBE_KIND_CHOICES = ("linear", "mlp", "both")
POSITION_TYPES_DEFAULT = ("last_text", "image_patch", "anchor")


# ---------------------------------------------------------------------------
# Pipeline data structures
# ---------------------------------------------------------------------------

@dataclass
class JoinedRow:
    """One (h, label, episode) tuple ready for the probe."""

    activation: np.ndarray
    label: str
    episode_index: int | None
    position_type: str
    source_example_id: str
    position_index: int


# ---------------------------------------------------------------------------
# Build joined rows from real activations + labels + attributes.
# ---------------------------------------------------------------------------

def build_joined_rows(
    activations_root: str,
    labels_jsonl: str,
    attributes_jsonl: str,
    *,
    attributes: list[str],
    seed: int = 0,
    max_input_rows: int | None = None,
    early_exit_cap: int | None = None,
) -> dict[str, list[JoinedRow]]:
    """Return ``{attribute: [JoinedRow, ...]}`` for the requested attributes.

    Reads activations via ``ActivationShardReader`` and walks the matching
    labels in the dataset's iteration order. Each yielded row carries its
    episode_index so downstream code can do an episode-stratified split.

    Performance notes:
      * ``max_input_rows`` caps the *input* row count (after the
        dataset's deterministic shuffle), which is the only way to avoid
        scanning all ~100k labels when running a small smoke probe.
      * ``early_exit_cap``: when every attribute has at least this many
        rows, stop iterating. Useful when the per-class cap is small and
        you don't want to load more activations than you'll actually use.
    """
    from nla.training.dataset import LabeledPositionDataset

    ds = LabeledPositionDataset(
        activations_root, labels_jsonl,
        seed=seed, held_out_fraction=0.0,
        max_items=max_input_rows,
    )

    attr_index: dict[tuple[str, int], dict[str, object]] = {}
    with Path(attributes_jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (str(obj["source_example_id"]), int(obj["position_index"]))
            attr_index[key] = obj.get("attributes") or {}

    out: dict[str, list[JoinedRow]] = {a: [] for a in attributes}
    n_rows = len(ds)

    log_every = max(1, n_rows // 20)
    for i in range(n_rows):
        sample = ds[i]
        key = (sample.example_id, sample.position_index)
        attrs_for_row = attr_index.get(key)
        if attrs_for_row is None:
            continue
        h = sample.activation.detach().cpu().numpy().astype(np.float32, copy=False)
        for a in attributes:
            if a not in attrs_for_row:
                continue
            out[a].append(JoinedRow(
                activation=h,
                label=str(attrs_for_row[a]),
                episode_index=sample.episode_index,
                position_type=sample.position_type,
                source_example_id=sample.example_id,
                position_index=sample.position_index,
            ))
        if i and i % log_every == 0:
            logger.info(
                "Joined %d / %d rows (%s)",
                i, n_rows,
                ", ".join(f"{a}:{len(out[a])}" for a in attributes),
            )
        if (
            early_exit_cap is not None
            and all(len(out[a]) >= early_exit_cap for a in attributes)
        ):
            logger.info(
                "Early-exit at row %d: every attribute has >= %d rows.",
                i, early_exit_cap,
            )
            break
    return out


# ---------------------------------------------------------------------------
# Episode-stratified train/val split.
# ---------------------------------------------------------------------------

def episode_split_indices(
    rows: list[JoinedRow],
    *,
    seed: int,
    held_out_fraction: float,
) -> tuple[list[int], list[int]]:
    """Return (train_idx, val_idx) holding *whole episodes* out.

    Mirrors ``_split_episode_aware`` semantics from the training dataset but
    works on the much smaller probe-side row list. Rows whose
    ``episode_index`` is None get bucketed under a sentinel which always
    lands in train (so the held-out set never relies on ungrouped rows).
    """
    if held_out_fraction <= 0.0:
        return list(range(len(rows))), []

    SENTINEL = "__no_episode__"
    by_ep: dict[object, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        key = SENTINEL if r.episode_index is None else int(r.episode_index)
        by_ep[key].append(i)

    ep_keys = sorted(by_ep.keys(), key=lambda k: (k == SENTINEL, str(k)))
    if len(ep_keys) < 2 or ep_keys == [SENTINEL]:
        # Not enough episodes to do an episode-level holdout. Fall back to
        # row split so tests with synthetic 1-episode dumps still work.
        rng = np.random.default_rng(seed)
        idx = np.arange(len(rows))
        rng.shuffle(idx)
        n_held = max(1, int(round(len(idx) * held_out_fraction)))
        return idx[n_held:].tolist(), idx[:n_held].tolist()

    rng = np.random.default_rng(seed)
    shuffled = list(ep_keys)
    rng.shuffle(shuffled)
    n_ep = len(shuffled)
    n_held_ep = max(1, int(round(n_ep * held_out_fraction)))
    held_keys = set(shuffled[:n_held_ep])
    train, val = [], []
    for i, r in enumerate(rows):
        key = SENTINEL if r.episode_index is None else int(r.episode_index)
        (val if key in held_keys else train).append(i)
    return train, val


# ---------------------------------------------------------------------------
# Probes.
# ---------------------------------------------------------------------------

def fit_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if len(np.unique(y_train)) < 2:
        # sklearn refuses single-class training; the probe is trivially the
        # majority class. Return its val accuracy so callers can still see
        # the row.
        pred = np.full_like(y_val, fill_value=y_train[0]) if len(y_train) else y_val
        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        return acc, f1, pred

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced", random_state=seed, n_jobs=1,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, pred))
    f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
    return acc, f1, pred


def fit_mlp_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    hidden: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    early_stop_patience: int = 5,
) -> tuple[float, float, np.ndarray]:
    """Tiny MLP probe in pure torch. Single hidden layer, AdamW, early stop.

    We fit class-balanced sample weights so a 95/5 imbalance can't trivially
    win by predicting the majority class.
    """
    from sklearn.metrics import accuracy_score, f1_score

    classes = np.unique(np.concatenate([y_train, y_val]))
    if len(classes) < 2:
        pred = np.full_like(y_val, fill_value=classes[0]) if len(classes) else y_val
        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        return acc, f1, pred

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    yt_idx = np.array([cls_to_idx[c] for c in y_train], dtype=np.int64)
    yv_idx = np.array([cls_to_idx[c] for c in y_val], dtype=np.int64)

    counts = np.bincount(yt_idx, minlength=len(classes)).astype(np.float64)
    inv_freq = np.where(counts > 0, 1.0 / counts, 0.0)
    inv_freq /= max(inv_freq.sum(), 1e-9)
    cls_weight = torch.tensor(inv_freq * len(classes), dtype=torch.float32)

    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_train.astype(np.float32))
    Yt = torch.from_numpy(yt_idx)
    Xv = torch.from_numpy(X_val.astype(np.float32))
    Yv = torch.from_numpy(yv_idx)

    in_dim = Xt.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hidden, len(classes)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss(weight=cls_weight)

    batch_size = min(256, len(Xt))
    best_val = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    n_train = len(Xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed + epoch))
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = crit(logits, Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(Xv)
            v_pred = v_logits.argmax(dim=-1)
            v_acc = (v_pred == Yv).float().mean().item()
        if v_acc > best_val + 1e-6:
            best_val = v_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        v_logits = model(Xv)
        v_pred_idx = v_logits.argmax(dim=-1).cpu().numpy()
    pred = np.array([classes[i] for i in v_pred_idx])
    acc = float(accuracy_score(y_val, pred))
    f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
    return acc, f1, pred


def majority_baseline(
    y_train: np.ndarray, y_val: np.ndarray,
) -> tuple[float, str]:
    if len(y_train) == 0:
        return 0.0, ""
    from sklearn.metrics import accuracy_score
    counter = Counter(y_train.tolist())
    majority = counter.most_common(1)[0][0]
    pred = np.full_like(y_val, fill_value=majority)
    acc = float(accuracy_score(y_val, pred))
    return acc, str(majority)


# ---------------------------------------------------------------------------
# AV verbalisation accuracy: re-extract attribute from AV caption.
# ---------------------------------------------------------------------------

def load_av_predictions(
    av_samples_jsonl: str | None,
) -> dict[tuple[str, int], str]:
    """Return ``{(source_example_id, position_index): av_caption}``.

    The AV samples file is the dump produced by ``scripts/eval/dump_av_samples.py``;
    we tolerate either ``meta.source_example_id`` style or top-level keys.
    """
    if av_samples_jsonl is None:
        return {}
    out: dict[tuple[str, int], str] = {}
    with Path(av_samples_jsonl).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("meta") or {}
            src = obj.get("source_example_id") or meta.get("source_example_id")
            pidx = obj.get("position_index")
            if pidx is None:
                pidx = meta.get("position_index")
            if src is None or pidx is None:
                continue
            text = obj.get("av_text") or obj.get("text") or obj.get("caption") or ""
            if not text:
                continue
            out[(str(src), int(pidx))] = str(text)
    return out


def av_accuracy_for_attribute(
    val_rows: list[JoinedRow],
    attr: str,
    av_predictions: dict[tuple[str, int], str],
) -> tuple[float, int]:
    """Score AV's text-based attribute prediction on the held-out val rows.

    Returns ``(accuracy, n_scored)`` where ``n_scored`` is the count of val
    rows that *had* an AV caption available.
    """
    if not av_predictions:
        return float("nan"), 0
    n_correct = 0
    n_scored = 0
    for r in val_rows:
        text = av_predictions.get((r.source_example_id, r.position_index))
        if not text:
            continue
        try:
            extracted = _extract_attrs_from_caption(text, [attr])
        except ValueError:
            continue
        pred = extracted.get(attr)
        if pred is None:
            continue
        if str(pred) == r.label:
            n_correct += 1
        n_scored += 1
    if n_scored == 0:
        return float("nan"), 0
    return n_correct / n_scored, n_scored


# ---------------------------------------------------------------------------
# Main probe sweep.
# ---------------------------------------------------------------------------

def _stack(rows: list[JoinedRow]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=object)
    X = np.stack([r.activation for r in rows], axis=0).astype(np.float32, copy=False)
    y = np.array([r.label for r in rows], dtype=object)
    return X, y


def _cap_per_class(rows: list[JoinedRow], cap: int, seed: int) -> list[JoinedRow]:
    """Cap to at most ``cap`` rows total, sampling per-class to keep balance.

    If ``cap`` is None or len(rows) <= cap, return rows unchanged.
    """
    if cap is None or len(rows) <= cap:
        return rows
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[r.label].append(i)
    rng = np.random.default_rng(seed)
    classes = sorted(by_label.keys())
    per_class = max(1, cap // len(classes))
    chosen: list[int] = []
    for c in classes:
        idx = by_label[c]
        if len(idx) > per_class:
            sub = rng.choice(idx, size=per_class, replace=False)
            chosen.extend(int(x) for x in sub)
        else:
            chosen.extend(idx)
    chosen.sort()
    return [rows[i] for i in chosen[:cap]]


def run_probe_sweep(
    rows_by_attr: dict[str, list[JoinedRow]],
    *,
    probe_kinds: tuple[str, ...],
    position_types: tuple[str, ...],
    held_out_fraction: float,
    max_rows_per_attr: int | None,
    seed: int,
    av_predictions: dict[tuple[str, int], str] | None = None,
) -> list[dict]:
    """For each (attribute, position_type, probe_kind) train+evaluate.

    ``position_types`` is augmented with the literal string ``"all"`` which
    aggregates every position_type (the headline number for the paper table).
    """
    av_predictions = av_predictions or {}
    out: list[dict] = []
    pt_keys = list(position_types) + ["all"]

    for attr, all_rows in rows_by_attr.items():
        if not all_rows:
            logger.warning("No rows for attribute %s -- skipping.", attr)
            continue

        for pt in pt_keys:
            rows = (
                all_rows if pt == "all"
                else [r for r in all_rows if r.position_type == pt]
            )
            if len(rows) < 4:
                logger.warning(
                    "Attribute=%s position_type=%s: only %d rows -- skipping.",
                    attr, pt, len(rows),
                )
                continue

            rows = _cap_per_class(rows, max_rows_per_attr, seed)

            train_idx, val_idx = episode_split_indices(
                rows, seed=seed, held_out_fraction=held_out_fraction,
            )
            if not val_idx or not train_idx:
                logger.warning(
                    "Attribute=%s position_type=%s: empty split (n_train=%d, n_val=%d).",
                    attr, pt, len(train_idx), len(val_idx),
                )
                continue
            train_rows = [rows[i] for i in train_idx]
            val_rows = [rows[i] for i in val_idx]
            X_train, y_train = _stack(train_rows)
            X_val, y_val = _stack(val_rows)

            base_acc, base_class = majority_baseline(y_train, y_val)
            av_acc, av_n = av_accuracy_for_attribute(val_rows, attr, av_predictions)

            train_classes = Counter(y_train.tolist())
            val_classes = Counter(y_val.tolist())

            base_record = {
                "attribute": attr,
                "position_type": pt,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_classes_train": int(len(train_classes)),
                "n_classes_val": int(len(val_classes)),
                "majority_class": base_class,
                "majority_baseline_acc": base_acc,
                "av_extract_acc": None if np.isnan(av_acc) else float(av_acc),
                "av_n_scored": int(av_n),
                "seed": int(seed),
            }

            for kind in probe_kinds:
                if kind == "linear":
                    acc, f1, _ = fit_linear_probe(
                        X_train, y_train, X_val, y_val, seed=seed,
                    )
                elif kind == "mlp":
                    acc, f1, _ = fit_mlp_probe(
                        X_train, y_train, X_val, y_val, seed=seed,
                    )
                else:
                    raise ValueError(f"Unknown probe kind: {kind!r}")
                out.append({
                    **base_record,
                    "probe_kind": kind,
                    "probe_acc": float(acc),
                    "probe_macro_f1": float(f1),
                    "gap_vs_majority": float(acc - base_acc),
                    "gap_vs_av": (
                        None if np.isnan(av_acc) else float(acc - av_acc)
                    ),
                })

    return out


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

def write_results_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def render_markdown_table(rows: list[dict]) -> str:
    """Return a markdown table summarising the probe sweep.

    Columns: ``attribute | position_type | probe | n_val | majority_acc |
    probe_acc | macro_f1 | av_extract_acc | gap_vs_av``.
    """
    headers = [
        "attribute", "position_type", "probe", "n_val", "majority_acc",
        "probe_acc", "macro_f1", "av_extract_acc", "gap_vs_av",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    sorted_rows = sorted(
        rows,
        key=lambda r: (r["attribute"], r["position_type"], r.get("probe_kind", "")),
    )
    for r in sorted_rows:
        av = r.get("av_extract_acc")
        gap = r.get("gap_vs_av")
        lines.append(
            "| "
            + " | ".join([
                str(r["attribute"]),
                str(r["position_type"]),
                str(r.get("probe_kind", "")),
                str(r.get("n_val", 0)),
                f"{r.get('majority_baseline_acc', 0.0):.3f}",
                f"{r.get('probe_acc', 0.0):.3f}",
                f"{r.get('probe_macro_f1', 0.0):.3f}",
                "n/a" if av is None else f"{av:.3f}",
                "n/a" if gap is None else f"{gap:+.3f}",
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_table(rows))


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activations-root", required=True, type=Path)
    p.add_argument("--labels-jsonl", required=True, type=Path)
    p.add_argument("--attributes-jsonl", required=True, type=Path)
    p.add_argument(
        "--attributes", nargs="+",
        default=_known_attrs(),
        choices=_known_attrs(),
    )
    p.add_argument(
        "--probe-kind", default="both", choices=PROBE_KIND_CHOICES,
        help="Which probes to fit.",
    )
    p.add_argument(
        "--position-types", nargs="+", default=list(POSITION_TYPES_DEFAULT),
        help="Per-position breakdown buckets (we always also report 'all').",
    )
    p.add_argument(
        "--held-out-fraction", type=float, default=0.1,
        help="Fraction of *episodes* to hold out for the val split.",
    )
    p.add_argument(
        "--max-rows-per-attr", type=int, default=5000,
        help="Cap rows per attribute (per-class balanced sampling).",
    )
    p.add_argument(
        "--max-input-rows", type=int, default=None,
        help=(
            "Cap on raw label rows scanned from the dataset. Use to keep "
            "smoke runs cheap; default scans the full dataset."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument(
        "--av-samples-jsonl", default=None, type=Path,
        help="Optional dump_av_samples.py output to score AV verbalisation.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.probe_kind == "both":
        probe_kinds: tuple[str, ...] = ("linear", "mlp")
    else:
        probe_kinds = (args.probe_kind,)

    print(
        f"Building joined (h, attribute) rows from "
        f"{args.activations_root} <-> {args.attributes_jsonl} ..."
    )
    early_exit = (
        max(1, int(args.max_rows_per_attr) * 4)
        if args.max_rows_per_attr is not None else None
    )
    rows_by_attr = build_joined_rows(
        str(args.activations_root),
        str(args.labels_jsonl),
        str(args.attributes_jsonl),
        attributes=list(args.attributes),
        seed=args.seed,
        max_input_rows=args.max_input_rows,
        early_exit_cap=early_exit,
    )
    for a, rs in rows_by_attr.items():
        print(f"  attr={a}: {len(rs)} joined rows.")

    av_predictions = load_av_predictions(
        str(args.av_samples_jsonl) if args.av_samples_jsonl else None,
    )
    if av_predictions:
        print(f"Loaded AV predictions for {len(av_predictions)} (src,pos) pairs.")

    results = run_probe_sweep(
        rows_by_attr,
        probe_kinds=probe_kinds,
        position_types=tuple(args.position_types),
        held_out_fraction=float(args.held_out_fraction),
        max_rows_per_attr=int(args.max_rows_per_attr),
        seed=int(args.seed),
        av_predictions=av_predictions,
    )

    write_results_jsonl(results, args.out_jsonl)
    write_markdown_table(results, args.out_md)
    print(f"Wrote {len(results)} probe rows to {args.out_jsonl}")
    print(f"Wrote markdown table to {args.out_md}")

    print("\n" + render_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

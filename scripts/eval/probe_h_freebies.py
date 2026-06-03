"""Linear probes on cached GR00T layer-16 activations.

Tests whether information GR00T's representation should plausibly contain is
linearly readable from h. Loads activations once, probes both the last_text
and image_patch_mean positions side-by-side. Task identity is recovered from
labels.jsonl (record.task_text is None in this cache).

For each probe target we fit a linear classifier / regressor with
cross-entropy / MSE and report val accuracy / R^2 vs the trivial baseline
(majority class or mean prediction).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/eval/probe_h_freebies.py \
    --activations-root data/activations/libero_4suite_v4_combined \
    --labels-jsonl     data/labels/libero_4suite_v6_with_task/labels.jsonl \
    --max-samples 30000 \
    --out-json data/eval/probe_freebies.json
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe")


def _h_at_positions(item):
    """Return (h_last_text, h_image_patch_mean) as float32 (2048,) each.

    Returns (None, None) if attention/image masks are empty.
    """
    feats = item["features"].numpy()
    attn = item["attention_mask"].numpy().astype(bool)
    img = item["image_mask"].numpy().astype(bool)
    h_lt = None
    text_positions = np.where(attn & ~img)[0]
    if len(text_positions) > 0:
        h_lt = feats[text_positions[-1]]
    h_ip = None
    img_positions = np.where(img)[0]
    if len(img_positions) > 0:
        h_ip = feats[img_positions].mean(axis=0)
    return h_lt, h_ip


_TASK_RE = re.compile(r"- task:\s*(.+?)\s*$", re.MULTILINE)


def _build_task_lookup(labels_jsonl: Path) -> dict[str, str]:
    """Map source_example_id -> task string by parsing labels.jsonl."""
    lookup: dict[str, str] = {}
    with labels_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            sid = d.get("meta", {}).get("source_example_id")
            desc = d.get("description", "")
            m = _TASK_RE.search(desc)
            if sid and m:
                lookup.setdefault(sid, m.group(1).strip())
    return lookup


def _train_classifier(X_train, y_train, X_val, y_val, n_classes, n_epochs=25, lr=5e-3, device="cuda"):
    in_dim = X_train.shape[1]
    model = nn.Linear(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)
    bs = 4096
    n = X_train_t.shape[0]
    best = 0.0
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(-1) == y_val_t).float().mean().item()
            if val_acc > best:
                best = val_acc
    cnt = Counter(y_val.tolist())
    majority = max(cnt.values()) / len(y_val)
    return {"val_acc": best, "majority_baseline": majority, "n_classes": n_classes,
            "n_train": int(n), "n_val": int(len(y_val))}


def _train_regressor(X_train, y_train, X_val, y_val, n_epochs=25, lr=5e-3, device="cuda"):
    in_dim = X_train.shape[1]
    model = nn.Linear(in_dim, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    y_mean, y_std = float(y_train.mean()), float(y_train.std() + 1e-6)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std
    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train_n, dtype=torch.float32, device=device).unsqueeze(-1)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val_n, dtype=torch.float32, device=device).unsqueeze(-1)
    bs = 4096
    n = X_train_t.shape[0]
    best_r2 = -1e9
    best_mse = float("inf")
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(X_val_t)
            mse_n = float(((pred - y_val_t) ** 2).mean().item())
            mse = mse_n * (y_std ** 2)
            ss_res = float(((pred - y_val_t) ** 2).sum().item())
            ss_tot = float(((y_val_t - y_val_t.mean()) ** 2).sum().item())
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            if r2 > best_r2:
                best_r2 = r2; best_mse = mse
    return {"val_r2": best_r2, "val_mse": best_mse,
            "baseline_mse": float(np.var(y_val)),
            "y_mean": y_mean, "y_std": y_std,
            "n_train": int(n), "n_val": int(len(y_val))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-root", required=True)
    ap.add_argument("--labels-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=30000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    from nla.extraction.storage import ActivationShardReader
    t0 = time.time()
    reader = ActivationShardReader(args.activations_root)
    n = len(reader)
    logger.info("Loaded reader: n=%d records (%.1fs)", n, time.time() - t0)

    logger.info("Building task lookup from labels...")
    task_lookup = _build_task_lookup(Path(args.labels_jsonl))
    logger.info("Task lookup size=%d, %d unique task strings",
                len(task_lookup), len(set(task_lookup.values())))

    rng = np.random.RandomState(args.seed)
    n_take = min(args.max_samples, n)
    idx = rng.permutation(n)[:n_take]

    H_lt = []; H_ip = []
    tasks = []; steps = []; suites = []; eps = []
    t0 = time.time()
    n_skipped = 0
    for j, i in enumerate(idx):
        if j % 5000 == 0:
            logger.info("loading %d/%d ...", j, n_take)
        rec = reader.records[i]
        if rec.example_id not in task_lookup:
            n_skipped += 1
            continue
        item = reader[i]
        h_lt, h_ip = _h_at_positions(item)
        if h_lt is None or h_ip is None:
            n_skipped += 1
            continue
        H_lt.append(h_lt)
        H_ip.append(h_ip)
        tasks.append(task_lookup[rec.example_id])
        steps.append(int(rec.step_index))
        eps.append(int(rec.episode_index))
        sid = rec.example_id
        suite = sid.split("__", 1)[0] if "__" in sid else "unknown"
        suites.append(suite)
    H_lt = np.asarray(H_lt, dtype=np.float32)
    H_ip = np.asarray(H_ip, dtype=np.float32)
    logger.info("Activations: last_text=%s image_patch_mean=%s (skipped=%d) (%.1fs)",
                H_lt.shape, H_ip.shape, n_skipped, time.time() - t0)

    eps_arr = np.asarray(eps)
    unique_eps = np.unique(eps_arr)
    rng2 = np.random.RandomState(args.seed + 1)
    rng2.shuffle(unique_eps)
    n_val_eps = max(1, int(args.val_frac * len(unique_eps)))
    val_eps = set(unique_eps[:n_val_eps].tolist())
    train_mask = np.array([e not in val_eps for e in eps_arr])
    val_mask = ~train_mask
    logger.info("episode-stratified split: %d val episodes; %d train rows / %d val rows",
                n_val_eps, train_mask.sum(), val_mask.sum())

    task_uniq = sorted(set(tasks))
    task_to_id = {t: i for i, t in enumerate(task_uniq)}
    y_task = np.array([task_to_id[t] for t in tasks])
    suite_uniq = sorted(set(suites))
    suite_to_id = {s: i for i, s in enumerate(suite_uniq)}
    y_suite = np.array([suite_to_id[s] for s in suites])
    y_step = np.asarray(steps, dtype=np.float32)

    logger.info("task_identity: %d unique tasks   suite: %d suites   step: range [%d, %d]",
                len(task_uniq), len(suite_uniq), int(y_step.min()), int(y_step.max()))

    results = {}
    for pos_name, H in [("last_text", H_lt), ("image_patch_mean", H_ip)]:
        logger.info("=== position: %s ===", pos_name)
        r = {}
        # task identity (categorical, ~10-40 classes)
        out = _train_classifier(H[train_mask], y_task[train_mask], H[val_mask],
                                 y_task[val_mask], n_classes=len(task_uniq), device=args.device)
        logger.info("  task_identity val_acc=%.3f (baseline %.3f, %d classes)",
                    out["val_acc"], out["majority_baseline"], len(task_uniq))
        r["task_identity"] = out

        out = _train_classifier(H[train_mask], y_suite[train_mask], H[val_mask],
                                 y_suite[val_mask], n_classes=len(suite_uniq), device=args.device)
        logger.info("  suite val_acc=%.3f (baseline %.3f)", out["val_acc"], out["majority_baseline"])
        r["suite"] = out

        out = _train_regressor(H[train_mask], y_step[train_mask], H[val_mask],
                                y_step[val_mask], device=args.device)
        out["val_rmse"] = float(np.sqrt(out["val_mse"]))
        out["baseline_rmse"] = float(np.sqrt(out["baseline_mse"]))
        logger.info("  step_index R^2=%.3f val_rmse=%.2f (baseline rmse %.2f)",
                    out["val_r2"], out["val_rmse"], out["baseline_rmse"])
        r["step_index"] = out

        results[pos_name] = r

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "n_rows_used": int(H_lt.shape[0]),
        "task_classes": len(task_uniq),
        "suite_classes": len(suite_uniq),
        "step_range": [int(y_step.min()), int(y_step.max())],
        "probes": results,
    }, indent=2))
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

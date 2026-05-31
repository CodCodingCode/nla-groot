#!/usr/bin/env python3
"""Export a JSON snapshot for the technical writeup website.

v9_combined-era exporter. Reads:

  * SFT training metrics    (run_sft.py -> metrics.jsonl, val + final rows)
  * Caption diagnostic JSON (av_caption_intent_diff.py -> *_paired_captions.json)
  * CF eval JSON            (compare_cf_steer_checkpoints.py -> *_cf_strided_cached.json)

Computes derived absolute values (cosine -> angle degrees, sqrt(mse) ->
RMS raw units, etc.) and emits:

  * website/src/data/snapshot.json   (committed, drives all charts)

The CF eval is the only currently-pending artifact at "I'm running this
right now" snapshot time. When the JSON isn't present yet, status is
set to "pending" so the site can render placeholders. Re-run this
exporter after the CF JSON lands to fill in the final numbers.

Run::

    python scripts/website/export_site_data.py \\
        --run-name v9_combined_12k \\
        --sft-log data/sft/v9_combined_12k_launch.log \\
        --captions-json data/eval/v9_combined_12k_paired_captions.json \\
        --cf-json data/eval/v9_combined_12k_cf_strided_cached.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAME = "v9_combined_12k"
DEFAULT_SFT_LOG = REPO / "data" / "sft" / f"{DEFAULT_RUN_NAME}_launch.log"
DEFAULT_CAPTIONS = REPO / "data" / "eval" / f"{DEFAULT_RUN_NAME}_paired_captions.json"
DEFAULT_CF = REPO / "data" / "eval" / f"{DEFAULT_RUN_NAME}_cf_strided_cached.json"
DEFAULT_OUT = REPO / "website" / "src" / "data" / "snapshot.json"
DEFAULT_ALPHA = 203.97713487289315  # P75 norm from libero_4suite_v4_combined stats
DEFAULT_TOTAL_STEPS = 12000
DEFAULT_N_TARGET = 32


_VAL_RE = re.compile(
    r"\[(step (\d+)|final)\].*"
    r"val (?P<rest>fve=.+?_n_rows=\d+\.\d+)"
)


def _parse_sft_log(log_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Pull every (step, val/closed_greedy_*, train) point from the SFT log.

    Returns (training_points, final_eval_dict).
    """
    if not log_path.exists():
        return [], None
    training: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    def _parse_kvs(rest: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for tok in rest.split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out

    last_train_metrics: dict[str, float] = {}
    with log_path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            # Train line
            mt = re.search(
                r"\[step (\d+)\] train  loss=([\d.]+) +ce=([\d.]+) +"
                r"ar_mse=([\d.]+) +ar_nce=([\d.]+)",
                line,
            )
            if mt:
                last_train_metrics = {
                    "step": int(mt.group(1)),
                    "train_loss": float(mt.group(2)),
                    "train_ce": float(mt.group(3)),
                    "train_ar_mse": float(mt.group(4)),
                }
                continue
            # Val line (periodic or final)
            mv = re.search(
                r"\[(step (\d+)|final)\].*val (?P<rest>fve=.+?)$",
                line,
            )
            if mv:
                rest = mv.group("rest")
                kvs = _parse_kvs(rest)
                step = int(mv.group(2)) if mv.group(2) else None
                if step is None and "_n_rows" not in rest:
                    # final line; use last train step as proxy if no step number
                    step = last_train_metrics.get("step", 0)
                rec = {
                    "step": step if step is not None else 0,
                    "val_cosine": kvs.get("cosine"),
                    "val_mse": kvs.get("mse"),
                    "val_fve": kvs.get("fve"),
                    "closed_greedy_cosine": kvs.get("closed_greedy/cosine"),
                    "closed_greedy_mse": kvs.get("closed_greedy/mse"),
                    "closed_greedy_fve": kvs.get("closed_greedy/fve"),
                    "train_loss": last_train_metrics.get("train_loss"),
                    "train_ar_mse": last_train_metrics.get("train_ar_mse"),
                    "train_ce": last_train_metrics.get("train_ce"),
                }
                training.append(rec)
                if "final" in mv.group(1):
                    final = rec
    # Subsample if too long (keep all if <= 40)
    if len(training) > 40:
        idxs = [round(i * (len(training) - 1) / 39) for i in range(40)]
        training = [training[i] for i in idxs]
    return training, final


def _codec_final(
    final_eval: dict[str, Any] | None,
    alpha: float,
    total_steps: int,
) -> dict[str, Any] | None:
    if not final_eval:
        return None
    cg_cos = final_eval.get("closed_greedy_cosine")
    cg_mse = final_eval.get("closed_greedy_mse")
    cg_fve = final_eval.get("closed_greedy_fve")
    val_cos = final_eval.get("val_cosine")
    val_mse = final_eval.get("val_mse")
    val_fve = final_eval.get("val_fve")
    val_ce = final_eval.get("train_ce")  # this is fallback; actual val ce comes from re-parse

    rms = math.sqrt(cg_mse) if isinstance(cg_mse, (int, float)) else None
    rel_pct = (rms / alpha * 100.0) if rms is not None else None
    angle_deg = (
        math.degrees(math.acos(max(-1.0, min(1.0, cg_cos))))
        if isinstance(cg_cos, (int, float))
        else None
    )
    return {
        "val_cosine": val_cos,
        "val_mse": val_mse,
        "val_fve": val_fve,
        "val_ce": val_ce,
        "closed_greedy_cosine": cg_cos,
        "closed_greedy_mse": cg_mse,
        "closed_greedy_fve": cg_fve,
        "alpha_norm": alpha,
        "rms_error_raw_units": rms,
        "relative_norm_error_pct": rel_pct,
        "angle_degrees": angle_deg,
        "step": final_eval.get("step"),
        "total_steps": total_steps,
    }


def _caption_diag(captions_path: Path) -> dict[str, Any] | None:
    if not captions_path.exists():
        return None
    blob = json.loads(captions_path.read_text())
    samples = blob.get("samples", []) or []
    paired = []
    for s in samples[:10]:  # cap for the site
        paired.append({
            "source_id": s.get("source_example_id") or "",
            "matched_intent": s.get("matched_intent") or "",
            "mismatched_intent": s.get("mismatched_intent") or "",
            "char_overlap": float(s.get("char_overlap_ratio") or 0.0),
            "bullet_diff": int(s.get("bullet_difference_count") or 0),
            "caption_matched_preview": (s.get("caption_matched") or "")[:600],
            "caption_mismatched_preview": (s.get("caption_mismatched") or "")[:600],
        })
    mean_overlap = float(blob.get("mean_char_overlap") or 0.0)
    mean_bd = float(blob.get("mean_bullet_diff_count") or 0.0)
    if mean_overlap > 0.9:
        interp = (
            "AV is NOT intent-conditional. Captions barely change with intent; "
            "expect minimal steering signal."
        )
    elif mean_overlap > 0.7:
        interp = (
            "AV is weakly intent-conditional. Only the task bullet shifts; "
            "expect modest steering improvement."
        )
    elif mean_overlap > 0.5:
        interp = (
            "AV is moderately intent-conditional. Multiple bullets shift; "
            "expect measurable steering improvement."
        )
    else:
        interp = (
            "AV is strongly intent-conditional. Captions substantively differ; "
            "high chance of clear publishable steering."
        )
    return {
        "n_samples": int(blob.get("n_samples") or len(samples)),
        "mean_char_overlap": mean_overlap,
        "mean_bullet_diff_count": mean_bd,
        "interpretation": interp,
        "paired_samples": paired,
    }


def _cf_eval(cf_path: Path, n_target: int) -> dict[str, Any]:
    if not cf_path.exists():
        return {
            "status": "pending",
            "n_complete": 0,
            "n_target": n_target,
            "steer_lift": None,
            "sem_gap": None,
            "lang_swap": None,
            "codec_above_lang": None,
            "predicate_rates": None,
            "per_sample": [],
        }
    blob = json.loads(cf_path.read_text())
    samples = blob.get("samples", []) or []
    arms = {
        "M_sem": "sft_av",
        "M_nost": "sft_av__no_steer",
        "Mm_sem": "sft_av__mismatched_source",
        "Mm_nost": "sft_av__mismatched_source__no_steer",
    }

    def _r(s, key):
        c = s["conditions"].get(arms[key])
        if c and c.get("error") is None and "skipped_reason" not in c:
            return float(c.get("r_sim", 0.0))
        return None

    def _pred(s, key):
        c = s["conditions"].get(arms[key])
        if c and c.get("error") is None and "skipped_reason" not in c:
            return int(float(c.get("predicate", 0.0)) > 0)
        return None

    per_sample: list[dict[str, Any]] = []
    sl, sg, lsw = [], [], []
    pred_M_sem, pred_M_nost, pred_Mm_sem, pred_Mm_nost = [], [], [], []
    for i, s in enumerate(samples):
        msem = _r(s, "M_sem")
        mnost = _r(s, "M_nost")
        mmsem = _r(s, "Mm_sem")
        mmnost = _r(s, "Mm_nost")
        if None in (msem, mnost, mmsem, mmnost):
            continue
        sl.append(msem - mnost)
        sg.append(msem - mmsem)
        lsw.append(mnost - mmnost)
        per_sample.append({
            "sample_index": i,
            "target_task": s.get("target_task") or "",
            "m_sem": msem,
            "m_nost": mnost,
            "mm_sem": mmsem,
            "mm_nost": mmnost,
            "steer_lift": msem - mnost,
            "sem_gap": msem - mmsem,
            "lang_swap": mnost - mmnost,
        })
        for arm, sink in (
            ("M_sem", pred_M_sem),
            ("M_nost", pred_M_nost),
            ("Mm_sem", pred_Mm_sem),
            ("Mm_nost", pred_Mm_nost),
        ):
            p = _pred(s, arm)
            if p is not None:
                sink.append(p)

    def _summary(xs: list[float]) -> dict[str, Any] | None:
        if not xs:
            return None
        n = len(xs)
        m = sum(xs) / n
        s = statistics.pstdev(xs) if n > 1 else 0.0
        se = s / math.sqrt(n - 1) if n > 1 else 0.0
        t = m / se if se > 0 else 0.0
        w = sum(1 for x in xs if x > 0)
        l = sum(1 for x in xs if x < 0)
        ties = n - w - l
        return {
            "mean": m, "std": s, "se": se, "t": t,
            "wins": w, "losses": l, "ties": ties, "n": n,
        }

    sl_sum = _summary(sl)
    sg_sum = _summary(sg)
    lsw_sum = _summary(lsw)
    n_complete = len(per_sample)
    n_target_effective = max(n_target, n_complete)
    status = "complete" if n_complete >= n_target else "running"
    codec_above = None
    if sg_sum and lsw_sum:
        codec_above = sg_sum["mean"] - lsw_sum["mean"]
    predicate_rates = None
    if pred_M_sem:
        predicate_rates = {
            "matched_semantic": sum(pred_M_sem) / max(1, len(pred_M_sem)),
            "matched_no_steer": sum(pred_M_nost) / max(1, len(pred_M_nost)),
            "mismatched_semantic": sum(pred_Mm_sem) / max(1, len(pred_Mm_sem)),
            "mismatched_no_steer": sum(pred_Mm_nost) / max(1, len(pred_Mm_nost)),
        }
    return {
        "status": status,
        "n_complete": n_complete,
        "n_target": n_target_effective,
        "steer_lift": sl_sum,
        "sem_gap": sg_sum,
        "lang_swap": lsw_sum,
        "codec_above_lang": codec_above,
        "predicate_rates": predicate_rates,
        "per_sample": per_sample,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--sft-log", type=Path, default=DEFAULT_SFT_LOG)
    p.add_argument("--captions-json", type=Path, default=DEFAULT_CAPTIONS)
    p.add_argument("--cf-json", type=Path, default=DEFAULT_CF)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--n-target", type=int, default=DEFAULT_N_TARGET)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    print(f"[export] reading SFT log: {args.sft_log}", file=sys.stderr)
    training, final_eval = _parse_sft_log(args.sft_log)
    print(f"[export] parsed {len(training)} val rows; final eval: "
          f"{'yes' if final_eval else 'no'}", file=sys.stderr)

    codec_final = _codec_final(final_eval, args.alpha, args.total_steps)

    print(f"[export] reading caption diagnostic: {args.captions_json}", file=sys.stderr)
    caption_diag = _caption_diag(args.captions_json)
    if caption_diag:
        print(f"[export]   mean_char_overlap = {caption_diag['mean_char_overlap']:.4f}",
              file=sys.stderr)

    print(f"[export] reading CF eval: {args.cf_json}", file=sys.stderr)
    cf = _cf_eval(args.cf_json, args.n_target)
    print(f"[export]   status={cf['status']}  n_complete={cf['n_complete']}/{cf['n_target']}",
          file=sys.stderr)
    if cf.get("steer_lift"):
        sl = cf["steer_lift"]
        print(f"[export]   steer_lift mean={sl['mean']:+.4f} t={sl['t']:+.2f}",
              file=sys.stderr)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "codec_final": codec_final,
        "training": training,
        "caption_diag": caption_diag,
        "cf_eval": cf,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"[ok] wrote {args.out}  ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

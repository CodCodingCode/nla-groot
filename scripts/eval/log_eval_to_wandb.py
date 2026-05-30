#!/usr/bin/env python
"""Consolidate caption-diagnostic + CF-eval JSON outputs into one W&B run.

Run at the end of auto_cf_eval_after_sft.sh so the eval results appear on
the W&B dashboard alongside the SFT training metrics. Loads the JSON
files written by av_caption_intent_diff.py and
compare_cf_steer_checkpoints.py, computes the headline aggregates, logs
them to a single W&B run, and builds two tables (per-sample caption
overlaps + per-sample CF arm r_sim values) for drill-down.

Run name defaults to ``<sft_run_name>_eval`` so the eval run sits next to
the SFT run in the project list.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/log_eval_to_wandb.py \\
        --captions-json data/eval/v9_combined_12k_paired_captions.json \\
        --cf-json       data/eval/v9_combined_12k_cf_strided_cached.json \\
        --project       nla-groot \\
        --run-name      v9_combined_12k_eval
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path


def _load_dotenv() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k, v)


def _cf_aggregates(cf: dict) -> dict:
    """Compute steer_lift / sem_gap / lang_swap aggregates from the CF JSON."""
    samples = cf.get("samples", [])
    arms = {
        "M_sem":   "sft_av",
        "M_nost":  "sft_av__no_steer",
        "Mm_sem":  "sft_av__mismatched_source",
        "Mm_nost": "sft_av__mismatched_source__no_steer",
    }

    def _get(s, key):
        c = s["conditions"].get(key)
        if c is None or c.get("error") is not None or "skipped_reason" in c:
            return None
        return c.get("r_sim")

    def _stat(xs):
        if not xs:
            return {}
        n = len(xs)
        m = sum(xs) / n
        s = statistics.pstdev(xs) if n > 1 else 0.0
        se = s / max(1, n - 1) ** 0.5 if n > 1 else 0.0
        t = m / se if se > 0 else 0.0
        return {
            "mean": m, "std": s, "se": se, "t": t,
            "wins": sum(1 for x in xs if x > 0),
            "losses": sum(1 for x in xs if x < 0),
            "n": n,
        }

    sl, sg, lsw = [], [], []
    pred_M_sem, pred_M_nost, pred_Mm_sem, pred_Mm_nost = [], [], [], []
    for s in samples:
        vals = {k: _get(s, arms[k]) for k in arms}
        if None in vals.values():
            continue
        sl.append(vals["M_sem"] - vals["M_nost"])
        sg.append(vals["M_sem"] - vals["Mm_sem"])
        lsw.append(vals["M_nost"] - vals["Mm_nost"])
        # Predicate rates
        for arm_key, sink in (
            ("sft_av", pred_M_sem),
            ("sft_av__no_steer", pred_M_nost),
            ("sft_av__mismatched_source", pred_Mm_sem),
            ("sft_av__mismatched_source__no_steer", pred_Mm_nost),
        ):
            c = s["conditions"].get(arm_key)
            if c and c.get("error") is None and "skipped_reason" not in c:
                sink.append(int(c.get("predicate", 0) > 0))

    return {
        "n_complete": len(sl),
        "steer_lift": _stat(sl),
        "sem_gap": _stat(sg),
        "lang_swap": _stat(lsw),
        "codec_above_lang": (
            (sum(sg) / len(sg)) - (sum(lsw) / len(lsw))
            if sg and lsw else 0.0
        ),
        "predicate_rate_M_sem":   sum(pred_M_sem)  / max(1, len(pred_M_sem)),
        "predicate_rate_M_nost":  sum(pred_M_nost) / max(1, len(pred_M_nost)),
        "predicate_rate_Mm_sem":  sum(pred_Mm_sem) / max(1, len(pred_Mm_sem)),
        "predicate_rate_Mm_nost": sum(pred_Mm_nost)/ max(1, len(pred_Mm_nost)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--captions-json", required=True,
                   help="Output of av_caption_intent_diff.py")
    p.add_argument("--cf-json", required=True,
                   help="Output of compare_cf_steer_checkpoints.py")
    p.add_argument("--project", default="nla-groot")
    p.add_argument("--entity", default="nathanyan2008p-personal")
    p.add_argument("--run-name", required=True,
                   help="W&B run name (suggest <sft_run_name>_eval).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + print aggregates without calling wandb.")
    args = p.parse_args()

    _load_dotenv()

    cap = json.loads(Path(args.captions_json).read_text()) if Path(args.captions_json).exists() else None
    cf = json.loads(Path(args.cf_json).read_text())

    cf_agg = _cf_aggregates(cf)

    print(f"\n=== CF eval aggregates ===")
    for k in ("steer_lift", "sem_gap", "lang_swap"):
        v = cf_agg[k]
        if v:
            print(f"  {k:<14} mean={v['mean']:+.4f}  se={v['se']:.4f}  t={v['t']:+.2f}  w/l={v['wins']}/{v['losses']}  n={v['n']}")
    print(f"  codec_above_lang  {cf_agg['codec_above_lang']:+.4f}")
    print(f"  predicate_rate_M_sem={cf_agg['predicate_rate_M_sem']:.2%}  M_nost={cf_agg['predicate_rate_M_nost']:.2%}")
    if cap:
        print(f"\n=== Caption diag aggregate ===")
        print(f"  n_samples            = {cap.get('n_samples')}")
        print(f"  mean_char_overlap    = {cap.get('mean_char_overlap'):.4f}")
        print(f"  mean_bullet_diff     = {cap.get('mean_bullet_diff_count'):.2f}")

    if args.dry_run:
        return 0

    import wandb
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        settings=wandb.Settings(_disable_stats=True),
    )

    # ---- Caption diagnostic logging
    if cap:
        wandb.define_metric("caption_diag/mean_char_overlap", summary="last")
        wandb.define_metric("caption_diag/mean_bullet_diff_count", summary="last")
        run.log({
            "caption_diag/mean_char_overlap": cap.get("mean_char_overlap"),
            "caption_diag/mean_bullet_diff_count": cap.get("mean_bullet_diff_count"),
            "caption_diag/n_samples": cap.get("n_samples"),
        })
        # Per-sample table for drill-down
        table = wandb.Table(columns=[
            "source_id", "matched_intent", "mismatched_intent",
            "char_overlap", "bullet_diff", "caption_matched", "caption_mismatched",
        ])
        for s in cap.get("samples", []):
            table.add_data(
                s.get("source_example_id", ""),
                s.get("matched_intent", "")[:120],
                s.get("mismatched_intent", "")[:120],
                float(s.get("char_overlap_ratio", 0.0)),
                int(s.get("bullet_difference_count", 0)),
                s.get("caption_matched", "")[:1000],
                s.get("caption_mismatched", "")[:1000],
            )
        run.log({"caption_diag/per_sample": table})

    # ---- CF eval logging
    for label in ("steer_lift", "sem_gap", "lang_swap"):
        wandb.define_metric(f"cf/{label}_mean", summary="last")
    wandb.define_metric("cf/codec_above_lang", summary="last")
    wandb.define_metric("cf/predicate_rate_M_sem", summary="max")
    wandb.define_metric("cf/predicate_rate_M_nost", summary="max")
    wandb.define_metric("cf/predicate_rate_Mm_sem", summary="max")
    wandb.define_metric("cf/predicate_rate_Mm_nost", summary="max")

    cf_log = {"cf/n_complete": cf_agg["n_complete"]}
    for label in ("steer_lift", "sem_gap", "lang_swap"):
        v = cf_agg[label]
        if v:
            cf_log[f"cf/{label}_mean"] = v["mean"]
            cf_log[f"cf/{label}_se"] = v["se"]
            cf_log[f"cf/{label}_t"] = v["t"]
            cf_log[f"cf/{label}_wins"] = v["wins"]
            cf_log[f"cf/{label}_losses"] = v["losses"]
    cf_log["cf/codec_above_lang"] = cf_agg["codec_above_lang"]
    cf_log["cf/predicate_rate_M_sem"]   = cf_agg["predicate_rate_M_sem"]
    cf_log["cf/predicate_rate_M_nost"]  = cf_agg["predicate_rate_M_nost"]
    cf_log["cf/predicate_rate_Mm_sem"]  = cf_agg["predicate_rate_Mm_sem"]
    cf_log["cf/predicate_rate_Mm_nost"] = cf_agg["predicate_rate_Mm_nost"]
    run.log(cf_log)

    # Per-sample CF table
    cf_table = wandb.Table(columns=[
        "source_id", "target_task", "target_intent",
        "M_sem_rsim", "M_nost_rsim", "Mm_sem_rsim", "Mm_nost_rsim",
        "steer_lift", "sem_gap", "lang_swap",
    ])
    arms = {
        "M_sem":   "sft_av",
        "M_nost":  "sft_av__no_steer",
        "Mm_sem":  "sft_av__mismatched_source",
        "Mm_nost": "sft_av__mismatched_source__no_steer",
    }
    for s in cf.get("samples", []):
        def _r(k):
            c = s["conditions"].get(arms[k])
            return float(c["r_sim"]) if c and c.get("error") is None and "skipped_reason" not in c else None
        msem  = _r("M_sem")
        mnost = _r("M_nost")
        mmsem = _r("Mm_sem")
        mmnost= _r("Mm_nost")
        if None in (msem, mnost, mmsem, mmnost):
            continue
        cf_table.add_data(
            s.get("source_example_id", ""),
            s.get("target_task", ""),
            (s.get("target_intent") or "")[:120],
            msem, mnost, mmsem, mmnost,
            msem - mnost, msem - mmsem, mnost - mmnost,
        )
    run.log({"cf/per_sample": cf_table})

    # ---- Final summary text in the run config
    summary_text = (
        f"v9 combined CF eval summary:\n"
        f"  steer_lift mean = {cf_agg['steer_lift'].get('mean', 0):+.4f} "
        f"(t={cf_agg['steer_lift'].get('t', 0):+.2f}, n={cf_agg['n_complete']})\n"
        f"  sem_gap         = {cf_agg['sem_gap'].get('mean', 0):+.4f}\n"
        f"  lang_swap       = {cf_agg['lang_swap'].get('mean', 0):+.4f}\n"
        f"  codec_above_lang= {cf_agg['codec_above_lang']:+.4f}\n"
    )
    if cap:
        summary_text += (
            f"  caption overlap = {cap.get('mean_char_overlap'):.4f}\n"
            f"  bullet diff     = {cap.get('mean_bullet_diff_count'):.2f}\n"
        )
    run.notes = summary_text
    print(summary_text)

    print(f"W&B run: {run.url}")
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

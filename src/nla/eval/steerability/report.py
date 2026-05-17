"""Render report.md + report.html + matplotlib bar charts for the harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PRETTY_BODY = {
    "akita_black_bowl_1_main": "bowl",
    "plate_1_main": "plate",
    "wine_bottle_1_main": "wine bottle",
    "cream_cheese_1_main": "cream cheese",
    "wooden_cabinet_1_main": "cabinet",
    "flat_stove_1_main": "stove",
    "wine_rack_1_main": "wine rack",
}


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "—"
        return f"{x:.{digits}f}"
    return str(x)


def _bar_chart(
    path: Path,
    series: dict[str, dict[str, float | None]],
    title: str,
    ylabel: str,
    pretty_keys: dict[str, str] | None = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return  # silently skip plots if matplotlib not installed

    all_keys: list[str] = []
    for d in series.values():
        for k in d.keys():
            if k not in all_keys:
                all_keys.append(k)
    n_cond = len(series)
    n_key = len(all_keys)
    if n_cond == 0 or n_key == 0:
        return
    pretty = pretty_keys or {}
    labels = [pretty.get(k, k) for k in all_keys]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_key * n_cond), 4))
    bar_w = 0.8 / n_cond
    x = np.arange(n_key)
    for i, (cond, d) in enumerate(series.items()):
        ys = [d.get(k) if d.get(k) is not None else 0.0 for k in all_keys]
        offset = (i - n_cond / 2) * bar_w + bar_w / 2
        ax.bar(x + offset, ys, bar_w, label=cond)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_markdown_report(
    output_dir: Path,
    config_name: str,
    metrics: dict[str, Any],
    av_metrics: dict[str, Any] | None = None,
    comparison_videos: list[Path] | None = None,
) -> Path:
    md_lines: list[str] = []
    md_lines.append(f"# Steerability eval — {config_name}\n")

    # Wrapper-transparency callout: compare any steered condition's success
    # rate to baseline. If a *matching-prompt* steer drops success rate the
    # AR+steer wrapper is not transparent at these settings.
    base = metrics["conditions"].get("baseline", {}).get("overall", {})
    if base:
        baseline_sa = base.get("success_any_rate")
        callout: list[str] = []
        for cond_name, cd in metrics["conditions"].items():
            if cond_name == "baseline":
                continue
            sa = cd["overall"].get("success_any_rate")
            if baseline_sa is None or sa is None:
                continue
            delta = sa - baseline_sa
            verdict = (
                "transparent ✓" if abs(delta) <= 0.1
                else ("breaks task" if delta < -0.1 else "boosts task")
            )
            callout.append(f"- **{cond_name}**: success {_fmt(sa)} vs baseline "
                           f"{_fmt(baseline_sa)} (Δ={_fmt(delta)}) — {verdict}")
        if callout:
            md_lines.append("## Wrapper transparency\n")
            md_lines.append(
                "Does the steer wrapper keep the policy able to do the "
                "*original* task? A condition is 'transparent' if its "
                "success rate is within 10 pp of baseline.\n"
            )
            md_lines.extend(callout)
            md_lines.append("")

    md_lines.append("## Per-condition summary\n")

    overall_rows = []
    body_set: list[str] = []
    for cond_name, cond_data in metrics["conditions"].items():
        o = cond_data["overall"]
        overall_rows.append(
            (
                cond_name,
                _fmt(o.get("success_any_rate")),
                _fmt(o.get("success_final_rate")),
                _fmt(o.get("mean_n_steps"), 1),
                _fmt(o.get("mean_target_displacement")),
                _fmt(o.get("mean_target_min_ee_distance")),
                _fmt(o.get("target_winner_rate")),
            )
        )
        for k in (o.get("displacement") or {}).keys():
            if k not in body_set:
                body_set.append(k)
    md_lines.append(
        "| condition | success_any | success_final | mean_steps | target_disp (m) | target_min_ee (m) | target_winner_rate |"
    )
    md_lines.append("|---|---|---|---|---|---|---|")
    for row in overall_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    md_lines.append("## Per-object end-of-episode displacement (mean over seeds)\n")
    md_lines.append("Larger ⇒ that object moved further during the episode.\n")
    md_lines.append("| condition | " + " | ".join(PRETTY_BODY.get(b, b) for b in body_set) + " |")
    md_lines.append("|---|" + "|".join(["---"] * len(body_set)) + "|")
    for cond_name, cond_data in metrics["conditions"].items():
        d = cond_data["overall"].get("displacement", {})
        row = [cond_name] + [_fmt(d.get(b)) for b in body_set]
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    md_lines.append("## Per-object min gripper distance (mean over seeds)\n")
    md_lines.append("Smaller ⇒ the gripper went *near* that object at least once.\n")
    md_lines.append("| condition | " + " | ".join(PRETTY_BODY.get(b, b) for b in body_set) + " |")
    md_lines.append("|---|" + "|".join(["---"] * len(body_set)) + "|")
    for cond_name, cond_data in metrics["conditions"].items():
        d = cond_data["overall"].get("min_ee_distance", {})
        row = [cond_name] + [_fmt(d.get(b)) for b in body_set]
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")

    md_lines.append("## Displacement winner counts (which object moved most each episode)\n")
    md_lines.append("| condition | most-moved object → count |")
    md_lines.append("|---|---|")
    for cond_name, cond_data in metrics["conditions"].items():
        wc = cond_data["overall"].get("winner_counts", {})
        pretty = ", ".join(
            f"{PRETTY_BODY.get(k, k)}: {v}" for k, v in sorted(wc.items(), key=lambda kv: -kv[1])
        )
        md_lines.append(f"| {cond_name} | {pretty or '—'} |")
    md_lines.append("")

    if av_metrics:
        md_lines.append("## AV text-fidelity\n")
        md_lines.append(
            "Generated AV captions vs. gold labels, graded by an LLM against the "
            "cached camera frames. ``grounding``=specific-to-this-scene, "
            "``appropriateness``=stays at the right abstraction, "
            "``template_distinguishable``=not a generic template. "
            "Numbers are pass-rates (higher = better). "
            "``av_pred − gold`` is the drop from ground-truth to AV's caption: "
            "anything strongly negative means the AV is losing fidelity.\n"
        )
        last_ds: str | None = None
        for ar_name in sorted(av_metrics.keys(), key=lambda k: (k.split("@")[-1], k)):
            ds_label = None
            if "@" in ar_name:
                ds_label = ar_name.split("@", 1)[1]
            if ds_label and ds_label != last_ds:
                md_lines.append(f"#### Hold-out: `{ds_label}`\n")
                last_ds = ds_label
            av = av_metrics[ar_name]
            md_lines.append(f"### `{ar_name}`")
            md_lines.append(f"({av.get('n_rows', 0)} graded rows from "
                            f"`{av.get('jsonl_path', '?')}`)")
            per_var = av.get("per_variant_mean", {})
            if per_var:
                axes = sorted({a for d in per_var.values() for a in d.keys()})
                md_lines.append("| variant | " + " | ".join(axes) + " |")
                md_lines.append("|---|" + "|".join(["---"] * len(axes)) + "|")
                for variant, d in sorted(per_var.items()):
                    md_lines.append(
                        f"| {variant} | "
                        + " | ".join(_fmt(d.get(a)) for a in axes)
                        + " |"
                    )
                diff = av.get("av_pred_minus_gold") or {}
                if diff:
                    md_lines.append(
                        "| av_pred − gold | "
                        + " | ".join(_fmt(diff.get(a)) for a in axes)
                        + " |"
                    )
            md_lines.append("")

    figs_dir = output_dir / "figures"
    if figs_dir.exists():
        for png in sorted(figs_dir.glob("*.png")):
            md_lines.append(f"![{png.stem}](figures/{png.name})\n")

    if comparison_videos:
        md_lines.append("## Comparison videos\n")
        for v in comparison_videos:
            rel = v.relative_to(output_dir)
            md_lines.append(f"- [{v.stem}]({rel})")

    md_path = output_dir / "report.md"
    md_path.write_text("\n".join(md_lines))

    # Minimal HTML wrapper (lets the user open report.html in a browser).
    html_lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Steerability eval — {config_name}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;}",
        "table{border-collapse:collapse;}th,td{padding:6px 10px;border:1px solid #ddd;}",
        "img{max-width:100%;}h1,h2{border-bottom:1px solid #eee;padding-bottom:4px;}</style>",
        "</head><body>",
    ]
    try:
        import markdown
        html_lines.append(markdown.markdown(md_path.read_text(), extensions=["tables"]))
    except Exception:
        html_lines.append("<pre>")
        html_lines.append(md_path.read_text().replace("<", "&lt;"))
        html_lines.append("</pre>")
    html_lines.append("</body></html>")
    (output_dir / "report.html").write_text("\n".join(html_lines))

    return md_path


def render_bar_charts(output_dir: Path, metrics: dict[str, Any]) -> None:
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    disp_series = {
        c: cd["overall"].get("displacement", {}) for c, cd in metrics["conditions"].items()
    }
    minee_series = {
        c: cd["overall"].get("min_ee_distance", {}) for c, cd in metrics["conditions"].items()
    }
    _bar_chart(
        figs_dir / "per_object_displacement.png",
        disp_series,
        "Per-object end-of-episode displacement (m)",
        ylabel="displacement (m)",
        pretty_keys=PRETTY_BODY,
    )
    _bar_chart(
        figs_dir / "per_object_min_ee_distance.png",
        minee_series,
        "Per-object min gripper distance during episode (m)",
        ylabel="min distance (m)",
        pretty_keys=PRETTY_BODY,
    )

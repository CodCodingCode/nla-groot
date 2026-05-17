#!/usr/bin/env python
"""Steerability eval harness driver.

Reads a YAML config (:class:`nla.eval.steerability.SteerabilityConfig`),
brings up the NLA-steered GR00T policy server once per condition, drives
rollouts via :mod:`nla.eval.steerability.rollout` (running inside the
LIBERO ``libero_uv`` venv as a subprocess), aggregates metrics, optionally
runs the AV-text-fidelity sub-eval, and writes a self-contained
report under ``output_dir``.

Run from the main NLA venv::

    PYTHONPATH=src python scripts/eval/steerability_eval.py \\
        --config scripts/eval/steerability_v1.yaml
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import tyro
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIBERO_VENV = ROOT / "third_party" / "Isaac-GR00T" / "gr00t" / "eval" / "sim" / "LIBERO" / "libero_uv" / ".venv"
LIBERO_PYTHON = LIBERO_VENV / "bin" / "python"
ROLLOUT_MODULE = "nla.eval.steerability.rollout"
SERVER_SCRIPT = ROOT / "scripts" / "eval" / "run_gr00t_server_nla_steer.py"
SERVER_VENV_PY = ROOT / ".venv" / "bin" / "python"


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_server(host: str, port: int, timeout_s: float = 600.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if _port_open(host, port):
            return True
        time.sleep(3)
    return False


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _start_server(
    *,
    model_path: str,
    embodiment_tag: str,
    port: int,
    host: str,
    ar_dir: str | None,
    steer_prompt_file: Path | None,
    placement: str,
    blend: float,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        str(SERVER_VENV_PY),
        str(SERVER_SCRIPT),
        "--model-path", model_path,
        "--embodiment-tag", embodiment_tag,
        "--use-sim-policy-wrapper",
        "--host", host,
        "--port", str(port),
    ]
    if ar_dir is not None and steer_prompt_file is not None:
        cmd.extend([
            "--ar-dir", ar_dir,
            "--steer-text-file", str(steer_prompt_file),
            "--placement", placement,
            "--blend", str(blend),
        ])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("ab")
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}\n  (logs: {log_path})", flush=True)
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def _run_rollout_subprocess(
    *,
    env_name: str,
    seed: int,
    output_dir: Path,
    tracked_bodies: list[str],
    target_body: str | None,
    policy_host: str,
    policy_port: int,
    n_action_steps: int,
    max_episode_steps: int,
    fps: int,
    steps_per_render: int,
    log_path: Path,
) -> dict[str, Any] | None:
    cmd = [
        str(LIBERO_PYTHON), "-m", ROLLOUT_MODULE,
        "--env-name", env_name,
        "--seed", str(seed),
        "--policy-host", policy_host,
        "--policy-port", str(policy_port),
        "--output-dir", str(output_dir),
        "--tracked-bodies", *tracked_bodies,
        "--n-action-steps", str(n_action_steps),
        "--max-episode-steps", str(max_episode_steps),
        "--fps", str(fps),
        "--steps-per-render", str(steps_per_render),
    ]
    if target_body:
        cmd.extend(["--target-body", target_body])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        print(f"    rollout failed (rc={proc.returncode}); see {log_path}", flush=True)
        return None
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def _run_av_fidelity_subeval(cfg, output_dir: Path) -> dict[str, Any] | None:
    if not cfg.av_eval.enabled:
        return None
    datasets_list = getattr(cfg.av_eval, "datasets", []) or []
    if not datasets_list:
        print(
            "  AV eval enabled but no datasets (populate av_eval.datasets or the "
            "legacy activations_root/labels_jsonl/frames_cache triple); skipping",
            flush=True,
        )
        return None
    out: dict[str, Any] = {}
    for ds in datasets_list:
        per_pos = ds.per_position if ds.per_position is not None else cfg.av_eval.per_position
        frac = (
            ds.held_out_fraction
            if ds.held_out_fraction is not None
            else cfg.av_eval.held_out_fraction
        )
        split = ds.split_by if ds.split_by is not None else cfg.av_eval.split_by

        ar_list = ds.ar_dirs if ds.ar_dirs else cfg.av_eval.ar_dirs
        if not ar_list:
            print(f"  AV dataset {ds.name!r}: empty ar_dirs; skipping", flush=True)
            continue
        ds_slug = ds.name.replace("/", "_").replace(".", "_")
        for ar_dir in ar_list:
            ar_dir_p = Path(ar_dir)
            ar_name = ar_dir_p.parent.name + "/" + ar_dir_p.name
            slug = ar_name.replace("/", "__")
            ar_out = output_dir / "av_fidelity" / f"{slug}__ds__{ds_slug}"
            ar_out.mkdir(parents=True, exist_ok=True)
            out_jsonl = ar_out / "llm_judge.jsonl"
            metric_key = f"{slug}@{ds.name}"
            if out_jsonl.exists() and out_jsonl.stat().st_size > 0:
                print(
                    f"  AV judge → {metric_key}  (reusing existing {out_jsonl})", flush=True
                )
                out[metric_key] = _summarise_av_judge_jsonl(out_jsonl)
                continue
            cmd = [
                str(SERVER_VENV_PY),
                str(ROOT / "scripts" / "eval" / "llm_judge_av_captions.py"),
                "--ckpt-dir", str(ar_dir_p.parent),
                "--activations-root", ds.activations_root,
                "--labels-jsonl", ds.labels_jsonl,
                "--frames-cache", ds.frames_cache,
                "--video-keys", *cfg.av_eval.video_keys,
                "--per-position", str(per_pos),
                "--held-out-fraction", str(frac),
                "--split-by", split,
                "--out-jsonl", str(out_jsonl),
            ]
            if cfg.av_eval.judge_model:
                cmd.extend(["--judge-model", cfg.av_eval.judge_model])
            log = ar_out / "judge.log"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
            print(f"  AV judge → {metric_key}  (logs: {log})", flush=True)
            with log.open("ab") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            if r.returncode != 0:
                print(f"    AV judge failed (rc={r.returncode})", flush=True)
                continue
            out[metric_key] = _summarise_av_judge_jsonl(out_jsonl)
    return out or None


_AV_AXES = ("grounding", "appropriateness", "template_distinguishable")


def _summarise_av_judge_jsonl(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    agg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        v = r.get("variant_id") or r.get("variant", "?")
        for axis in _AV_AXES:
            ax = r.get(axis)
            if isinstance(ax, dict) and "passed" in ax:
                agg.setdefault(v, {}).setdefault(axis + "_pass_rate", []).append(
                    1.0 if ax["passed"] else 0.0
                )
    summary = {
        "n_rows": len(rows),
        "per_variant_mean": {
            v: {k: (sum(xs) / len(xs)) for k, xs in d.items() if xs}
            for v, d in agg.items()
        },
        "jsonl_path": str(path),
    }
    # Also report a flat "gold vs av_pred" diff if both variants are present
    pred = summary["per_variant_mean"].get("av_pred", {})
    gold = summary["per_variant_mean"].get("gold", {})
    if pred and gold:
        summary["av_pred_minus_gold"] = {
            k: round(pred.get(k, 0.0) - gold.get(k, 0.0), 3)
            for k in set(pred) | set(gold)
        }
    return summary


def _scorecard_verdict(
    val: float | None, *, higher: bool, t_pass: float, t_warn: float
) -> str:
    if val is None:
        return "NA"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "NA"
    if higher:
        if v >= t_pass:
            return "PASS"
        if v >= t_warn:
            return "WARN"
        return "FAIL"
    if v <= t_pass:
        return "PASS"
    if v <= t_warn:
        return "WARN"
    return "FAIL"


def _v3_four_suite_av_pred_rates(av_metrics: dict[str, Any] | None) -> dict[str, float | None]:
    if not av_metrics:
        return {}
    needle = "libero_4suite_v3__ar@"
    for key, blob in av_metrics.items():
        if needle in key and key.endswith("@libero_4suite_holdout"):
            pv = (blob.get("per_variant_mean") or {}).get("av_pred") or {}
            return {
                "grounding": pv.get("grounding_pass_rate"),
                "anti_template": pv.get("template_distinguishable_pass_rate"),
                "appropriateness": pv.get("appropriateness_pass_rate"),
            }
    return {}


def patch_v3_style_scorecard(
    scorecard_path: Path,
    steering_metrics: dict[str, Any],
    av_metrics: dict[str, Any] | None,
    bench_notes: dict[str, str],
) -> None:
    """Fill ``sim_*`` / ``judge_*`` holes in ``v3_scorecard.json`` style scorecards."""

    if not scorecard_path.exists():
        print(f"  patch_scorecard: missing {scorecard_path}, skipping", flush=True)
        return
    blob = json.loads(scorecard_path.read_text())
    cond = steering_metrics.get("conditions") or {}
    baseline_sa = float(
        (cond.get("baseline") or {}).get("overall", {}).get("success_any_rate") or 0
    )
    corr_sa = float(
        (cond.get("steer_bowl_plate_v3") or {})
        .get("overall", {})
        .get("success_any_rate") or 0
    )
    wrong_sa = float(
        (cond.get("steer_wine_rack_v3") or {})
        .get("overall", {})
        .get("success_any_rate") or 0
    )

    j = _v3_four_suite_av_pred_rates(av_metrics)
    updates: dict[str, tuple[float | None, dict[str, Any]]] = {
        "judge_grounding_specific_pct": (
            j.get("grounding"),
            {"higher_is_better": True, "t_pass": 0.55, "t_warn": 0.4},
        ),
        "judge_anti_template_specific_pct": (
            j.get("anti_template"),
            {"higher_is_better": True, "t_pass": 0.5, "t_warn": 0.3},
        ),
        "judge_appropriateness_pct": (
            j.get("appropriateness"),
            {"higher_is_better": True, "t_pass": 0.8, "t_warn": 0.6},
        ),
        "sim_correct_success": (
            corr_sa,
            {"higher_is_better": True, "t_pass": 0.3, "t_warn": 0.15},
        ),
        "sim_correct_minus_baseline_floor": (
            corr_sa - baseline_sa,
            {"higher_is_better": True, "t_pass": -0.1, "t_warn": -0.3},
        ),
        "sim_wrong_minus_baseline": (
            wrong_sa - baseline_sa,
            {"higher_is_better": False, "t_pass": 0.0, "t_warn": 0.05},
        ),
        "sim_correct_minus_wrong": (
            corr_sa - wrong_sa,
            {"higher_is_better": True, "t_pass": 0.05, "t_warn": 0.0},
        ),
    }

    idx = {m["name"]: m for m in blob.get("metrics", [])}
    for name, (val, tspec) in updates.items():
        if name not in idx:
            continue
        idx[name]["value"] = val
        idx[name]["verdict"] = _scorecard_verdict(
            val,
            higher=tspec["higher_is_better"],
            t_pass=tspec["t_pass"],
            t_warn=tspec["t_warn"],
        )

    src = blob.setdefault("sources", {})
    src.setdefault("scorecard_benchmark", {}).update(bench_notes)
    blob.setdefault("config", {})
    blob["config"]["sim_present"] = True
    blob["config"]["benchmark_note"] = "patched_via_steerability_eval"

    scorecard_path.write_text(json.dumps(blob, indent=2, default=float) + "\n")
    print(f"  patched scorecard: {scorecard_path}", flush=True)


def _import_modules() -> tuple:
    sys.path.insert(0, str(SRC))
    from nla.eval.steerability import SteerabilityConfig, load_config
    from nla.eval.steerability.metrics import aggregate_all
    from nla.eval.steerability.report import render_bar_charts, render_markdown_report
    from nla.eval.steerability.video import hstack_conditions_for_seed
    return (
        SteerabilityConfig,
        load_config,
        aggregate_all,
        render_bar_charts,
        render_markdown_report,
        hstack_conditions_for_seed,
    )


def main(
    config: str,
    *,
    skip_rollouts: bool = False,
    skip_av: bool = False,
    skip_report: bool = False,
) -> None:
    (
        _SteerabilityConfig,
        load_config,
        aggregate_all,
        render_bar_charts,
        render_markdown_report,
        hstack_conditions_for_seed,
    ) = _import_modules()

    cfg = load_config(config)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(Path(config).read_text())

    if not skip_rollouts:
        for cond in cfg.conditions:
            print(f"\n==== condition: {cond.name} ====", flush=True)
            steer_prompt_file = None
            placement = "image_patch_all"
            blend = 1.0
            ar_dir = cond.ar_dir
            if cond.steer is not None:
                steer_prompt_file = output_dir / "prompts" / f"{cond.name}.txt"
                steer_prompt_file.parent.mkdir(parents=True, exist_ok=True)
                steer_prompt_file.write_text(cond.steer.resolved_prompt())
                placement = cond.steer.placement
                blend = float(cond.steer.blend)
            server_log = output_dir / "conditions" / cond.name / "server.log"
            proc = _start_server(
                model_path=cfg.model_path,
                embodiment_tag=cfg.embodiment_tag,
                port=cfg.policy_port,
                host=cfg.policy_host,
                ar_dir=ar_dir,
                steer_prompt_file=steer_prompt_file,
                placement=placement,
                blend=blend,
                log_path=server_log,
            )
            try:
                ready = _wait_for_server("localhost", cfg.policy_port, timeout_s=600)
                if not ready:
                    print(f"  server failed to come up; see {server_log}", flush=True)
                    continue
                print(f"  server ready on {cfg.policy_host}:{cfg.policy_port}", flush=True)
                for env_name in cfg.envs:
                    env_dir = output_dir / "conditions" / cond.name / env_name.replace("/", "__")
                    for seed in cfg.seeds:
                        seed_dir = env_dir / f"seed_{seed}"
                        if (seed_dir / "summary.json").exists():
                            print(f"  skip existing: {seed_dir}", flush=True)
                            continue
                        seed_dir.mkdir(parents=True, exist_ok=True)
                        print(f"  rollout: env={env_name} seed={seed}", flush=True)
                        log = seed_dir / "rollout.log"
                        summary = _run_rollout_subprocess(
                            env_name=env_name,
                            seed=seed,
                            output_dir=seed_dir,
                            tracked_bodies=cfg.tracked_bodies,
                            target_body=cond.target_body,
                            policy_host="localhost",
                            policy_port=cfg.policy_port,
                            n_action_steps=cfg.n_action_steps,
                            max_episode_steps=cfg.max_episode_steps,
                            fps=cfg.fps,
                            steps_per_render=cfg.steps_per_render,
                            log_path=log,
                        )
                        if summary is None:
                            continue
                        print(
                            "    -> success_any={success_any}  steps={n_steps}  winner={displacement_winner}".format(
                                **summary
                            ),
                            flush=True,
                        )
            finally:
                _kill_server(proc)
                # tiny wait so port frees before next condition
                time.sleep(2)

    # Aggregate + AV + report
    metrics = aggregate_all(
        output_dir,
        condition_names=[c.name for c in cfg.conditions],
        env_names=cfg.envs,
        tracked_bodies=cfg.tracked_bodies,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nWrote metrics to {output_dir / 'metrics.json'}", flush=True)

    av_metrics = None
    if not skip_av:
        av_metrics = _run_av_fidelity_subeval(cfg, output_dir)
        if av_metrics:
            (output_dir / "av_metrics.json").write_text(json.dumps(av_metrics, indent=2, default=float))

    if getattr(cfg, "patch_scorecard", None):
        score_p = Path(cfg.patch_scorecard)
        notes: dict[str, str | None] = {
            "steering_metrics_path": str((output_dir / "metrics.json").resolve()),
            "av_metrics_path": (
                str((output_dir / "av_metrics.json").resolve()) if av_metrics else None
            ),
        }
        if score_p.exists():
            backup = output_dir / "v3_scorecard.before_patch.json"
            shutil.copy2(score_p, backup)
            notes["scorecard_backup"] = str(backup.resolve())
        patch_v3_style_scorecard(score_p, metrics, av_metrics, notes)

    comparison_videos: list[Path] = []
    if not skip_report:
        render_bar_charts(output_dir, metrics)
        # Build one hstack per (env, seed) covering all condition rollouts
        for env_name in cfg.envs:
            for seed in cfg.seeds:
                vids: dict[str, Path] = {}
                for cond in cfg.conditions:
                    p = (
                        output_dir
                        / "conditions"
                        / cond.name
                        / env_name.replace("/", "__")
                        / f"seed_{seed}"
                        / "rollout.mp4"
                    )
                    if p.exists():
                        vids[cond.name] = p
                if len(vids) >= 2:
                    try:
                        out = hstack_conditions_for_seed(
                            output_dir, vids, seed=seed, env_name=env_name, fps=cfg.fps,
                        )
                        if out:
                            comparison_videos.append(out)
                            print(f"  built comparison: {out}", flush=True)
                    except Exception as e:
                        print(f"  comparison build failed: {e}", flush=True)
        md = render_markdown_report(
            output_dir,
            config_name=cfg.name,
            metrics=metrics,
            av_metrics=av_metrics,
            comparison_videos=comparison_videos,
        )
        print(f"\nReport at: {md}")
        print(f"HTML at:   {output_dir / 'report.html'}")


if __name__ == "__main__":
    tyro.cli(main)

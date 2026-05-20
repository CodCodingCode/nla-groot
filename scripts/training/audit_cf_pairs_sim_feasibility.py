#!/usr/bin/env python
"""Sim-feasibility audit for the production GRPO counterfactual pairs file.

For every unique ``target_env_name`` in
``data/grpo/libero_goal_counterfactual_pairs.jsonl`` (or any other CF-pairs
JSONL) this script verifies, ahead of GRPO sim time, that:

  1. The target task resolves through
     :func:`nla.eval.steerability.predicates.resolve_task` (so the predicate
     machinery would actually fire on a rollout for that task).
  2. A matching BDDL file exists in the LIBERO Goal BDDL folder
     (``libero/libero/bddl_files/libero_goal/<canonical>.bddl``). This is the
     filename that ``register_libero_envs()`` in
     ``third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py`` will hand
     to ``OffScreenRenderEnv`` at GRPO start.
  3. (Optional) ``LiberoEnv(env_name=target_env_name)`` actually instantiates
     and ``.reset()`` returns. This is the "real" feasibility test but it
     pulls in ``robosuite`` + ``mujoco`` and requires a working OpenGL stack
     (``MUJOCO_GL=osmesa`` or ``egl``). It is gated behind ``--probe-env``
     and is run in a subprocess with a per-env timeout so a single hanging
     reset can't take the whole audit down.

Outputs (always under the input file's directory):

    <pairs>.sim_audit.json   - machine-readable results
    <pairs>.sim_audit.md     - human-readable summary

Exit code:
    0 - every unique target_env_name has BDDL, AND env_loads if probed.
    1 - any target_env_name has missing BDDL / unresolvable task
        (or env_load=False when probed).

The script is read-only with respect to the inputs and never touches
``src/nla/`` -- it only adds ``src/`` to ``sys.path`` so it can import
``nla.eval.steerability.predicates``.

Pure CPU. Without ``--probe-env`` it runs in well under a second on the 5k
production file; with ``--probe-env`` it fans out one subprocess per *unique*
env (~10 for libero_goal), each timed at ``--env-timeout-s`` seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nla.eval.steerability.bddl_bodies import (  # noqa: E402
    DEFAULT_GOAL_BDDL_DIR,
    missing_bodies_for_task,
    parse_bddl_instance_names,
)
from nla.eval.steerability.predicates import GOAL_TASKS, resolve_task  # noqa: E402


logger = logging.getLogger("audit_cf_pairs_sim_feasibility")


# Where ``register_libero_envs()`` looks. We replicate the same path so the
# audit doesn't have to import LIBERO just to find the BDDL files.
DEFAULT_BDDL_DIR = (
    _REPO_ROOT
    / "third_party/Isaac-GR00T/external_dependencies/LIBERO/libero/libero"
    / "bddl_files/libero_goal"
)
DEFAULT_PAIRS = _REPO_ROOT / "data/grpo/libero_goal_counterfactual_pairs.jsonl"
DEFAULT_VENV_PY = _REPO_ROOT / ".venv/bin/python"
DEFAULT_LIBERO_PY = (
    _REPO_ROOT / "third_party/Isaac-GR00T/external_dependencies/LIBERO"
)
DEFAULT_GR00T_PY = _REPO_ROOT / "third_party/Isaac-GR00T"


# ---------------------------------------------------------------------------
# Pairs reader
# ---------------------------------------------------------------------------


def _read_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _task_from_env_name(env_name: str) -> str:
    """``libero_sim/<task>`` -> ``<task>``. No-op if the prefix is missing."""
    if env_name.startswith("libero_sim/"):
        return env_name[len("libero_sim/"):]
    return env_name


# ---------------------------------------------------------------------------
# Optional in-subprocess env-load probe
# ---------------------------------------------------------------------------


# This script body is what the env-load subprocess executes. It imports
# LIBERO, registers ``libero_sim/...`` envs, and instantiates the requested
# one. We only print a single JSON line so the parent can parse it
# unambiguously regardless of LIBERO's spammy stdout.
_ENV_PROBE_SCRIPT = r"""
import json, os, sys, time, traceback

env_name = os.environ["AUDIT_ENV_NAME"]
t0 = time.time()
result = {"env_name": env_name, "ok": False, "stage": "import", "elapsed_s": 0.0}
try:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    sys.path.insert(0, os.environ["AUDIT_LIBERO_PY"])
    sys.path.insert(0, os.environ["AUDIT_GR00T_PY"])
    import gymnasium as gym
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    result["stage"] = "register"
    register_libero_envs()
    result["stage"] = "make"
    env = gym.make(env_name)
    result["stage"] = "reset"
    env.reset()
    result["stage"] = "close"
    env.close()
    result["ok"] = True
    result["stage"] = "done"
except BaseException as e:
    result["error_type"] = type(e).__name__
    result["error"] = str(e)[:500]
    result["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
finally:
    result["elapsed_s"] = round(time.time() - t0, 2)
    sys.stdout.write("__AUDIT_RESULT__" + json.dumps(result) + "\n")
    sys.stdout.flush()
"""


def _probe_env_dependency(python_exe: Path, libero_py: Path, gr00t_py: Path) -> dict:
    """One-shot import probe so we don't fire 10 doomed subprocesses."""
    env = os.environ.copy()
    env["AUDIT_ENV_NAME"] = "libero_sim/_dependency_check_"
    env["AUDIT_LIBERO_PY"] = str(libero_py)
    env["AUDIT_GR00T_PY"] = str(gr00t_py)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    code = (
        "import os, sys, json\n"
        "sys.path.insert(0, os.environ['AUDIT_LIBERO_PY'])\n"
        "sys.path.insert(0, os.environ['AUDIT_GR00T_PY'])\n"
        "out={'ok': False}\n"
        "try:\n"
        "    import gymnasium as gym\n"
        "    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs\n"
        "    out['ok']=True\n"
        "except BaseException as e:\n"
        "    out['error_type']=type(e).__name__\n"
        "    out['error']=str(e)[:300]\n"
        "print('__AUDIT_RESULT__'+json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "TimeoutExpired", "error": "import probe timed out"}
    for line in proc.stdout.splitlines():
        if line.startswith("__AUDIT_RESULT__"):
            try:
                return json.loads(line[len("__AUDIT_RESULT__"):])
            except Exception:
                pass
    return {
        "ok": False,
        "error_type": "ProbeFailed",
        "error": (proc.stderr or proc.stdout or "no output")[:500],
    }


def _run_env_load(
    env_name: str,
    python_exe: Path,
    libero_py: Path,
    gr00t_py: Path,
    timeout_s: float,
) -> dict:
    env = os.environ.copy()
    env["AUDIT_ENV_NAME"] = env_name
    env["AUDIT_LIBERO_PY"] = str(libero_py)
    env["AUDIT_GR00T_PY"] = str(gr00t_py)
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", _ENV_PROBE_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "env_name": env_name,
            "ok": False,
            "stage": "timeout",
            "elapsed_s": round(time.time() - t0, 2),
            "error_type": "TimeoutExpired",
            "error": f"env load exceeded {timeout_s}s",
        }
    parsed: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("__AUDIT_RESULT__"):
            try:
                parsed = json.loads(line[len("__AUDIT_RESULT__"):])
                break
            except Exception:
                pass
    if parsed is None:
        parsed = {
            "env_name": env_name,
            "ok": False,
            "stage": "no_result",
            "elapsed_s": round(time.time() - t0, 2),
            "error_type": "NoResultLine",
            "error": (proc.stderr or proc.stdout or "no output")[:500],
        }
    return parsed


# ---------------------------------------------------------------------------
# Audit core
# ---------------------------------------------------------------------------


def audit(
    pairs_path: Path,
    bddl_dir: Path,
    *,
    probe_env: bool,
    python_exe: Path,
    libero_py: Path,
    gr00t_py: Path,
    env_timeout_s: float,
) -> dict[str, Any]:
    rows = _read_pairs(pairs_path)
    n_rows = len(rows)

    env_name_counts: Counter[str] = Counter()
    for r in rows:
        en = r.get("target_env_name") or ""
        if en:
            env_name_counts[en] += 1

    env_load_skipped_reason: str | None = None
    if probe_env:
        probe = _probe_env_dependency(python_exe, libero_py, gr00t_py)
        if not probe.get("ok"):
            env_load_skipped_reason = (
                f"dependency import failed: "
                f"{probe.get('error_type', '?')}: {probe.get('error', '?')}"
            )
            logger.warning(
                "Env-load probe disabled: %s. Falling back to BDDL-only audit.",
                env_load_skipped_reason,
            )
            probe_env = False
    else:
        env_load_skipped_reason = "not requested (run with --probe-env to enable)"

    per_env: list[dict[str, Any]] = []
    n_bddl_missing = 0
    n_unresolvable = 0
    n_missing_bodies = 0
    n_env_load_failed = 0
    rows_at_risk = 0

    for env_name, n_use in sorted(env_name_counts.items()):
        target_task = _task_from_env_name(env_name)
        rec: dict[str, Any] = {
            "target_env_name": env_name,
            "target_task_raw": target_task,
            "n_rows_using_it": int(n_use),
            "task_canonical": None,
            "predicate_kind": None,
            "source_body": None,
            "destination": None,
            "missing_predicate_bodies": [],
            "bddl_found": False,
            "bddl_path": None,
            "env_loads": None,
            "env_load_stage": None,
            "env_load_elapsed_s": None,
            "env_load_error": None,
            "errors": [],
        }

        try:
            spec = resolve_task(target_task)
            rec["task_canonical"] = spec.name
            rec["predicate_kind"] = spec.predicate_kind
            rec["source_body"] = spec.source_body
            rec["destination"] = spec.destination
        except KeyError as e:
            rec["errors"].append(f"resolve_task failed: {e}")
            n_unresolvable += 1
            rows_at_risk += int(n_use)
            per_env.append(rec)
            continue

        bddl_path = bddl_dir / f"{spec.name}.bddl"
        rec["bddl_path"] = str(bddl_path)
        if bddl_path.exists():
            rec["bddl_found"] = True
            missing = missing_bodies_for_task(spec.name, bddl_dir)
            rec["missing_predicate_bodies"] = missing
            if missing:
                rec["errors"].append(
                    "predicate bodies missing from BDDL: "
                    + ", ".join(missing)
                )
                n_missing_bodies += 1
                rows_at_risk += int(n_use)
        else:
            rec["bddl_found"] = False
            rec["errors"].append(f"missing BDDL at {bddl_path}")
            n_bddl_missing += 1
            rows_at_risk += int(n_use)

        if probe_env and rec["bddl_found"] and not rec.get("missing_predicate_bodies"):
            t0 = time.time()
            res = _run_env_load(
                env_name,
                python_exe=python_exe,
                libero_py=libero_py,
                gr00t_py=gr00t_py,
                timeout_s=env_timeout_s,
            )
            rec["env_loads"] = bool(res.get("ok"))
            rec["env_load_stage"] = res.get("stage")
            rec["env_load_elapsed_s"] = res.get("elapsed_s", round(time.time() - t0, 2))
            if not res.get("ok"):
                rec["env_load_error"] = (
                    f"{res.get('error_type', '?')}: {res.get('error', '?')}"
                )
                rec["errors"].append(f"env load failed: {rec['env_load_error']}")
                n_env_load_failed += 1
                rows_at_risk += int(n_use)

        per_env.append(rec)

    canon_seen = {r["task_canonical"] for r in per_env if r["task_canonical"]}
    canon_missing_from_pairs = sorted(set(GOAL_TASKS) - canon_seen)

    summary = {
        "pairs_path": str(pairs_path),
        "bddl_dir": str(bddl_dir),
        "n_rows": n_rows,
        "n_unique_target_env_names": len(env_name_counts),
        "n_bddl_missing": n_bddl_missing,
        "n_missing_predicate_bodies": n_missing_bodies,
        "n_unresolvable": n_unresolvable,
        "env_load_probed": bool(probe_env),
        "env_load_skipped_reason": env_load_skipped_reason,
        "env_timeout_s": env_timeout_s if probe_env else None,
        "n_env_load_failed": n_env_load_failed,
        "rows_at_risk": rows_at_risk,
        "rows_at_risk_pct": (rows_at_risk / n_rows) if n_rows else 0.0,
        "canonical_tasks_in_pairs": sorted(canon_seen),
        "canonical_tasks_missing_from_pairs": canon_missing_from_pairs,
        "per_env": per_env,
    }
    return summary


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Sim-feasibility audit: `{Path(report['pairs_path']).name}`")
    lines.append("")
    lines.append(f"- Pairs file: `{report['pairs_path']}`")
    lines.append(f"- BDDL dir: `{report['bddl_dir']}`")
    lines.append(f"- Rows: {report['n_rows']}")
    lines.append(
        f"- Unique target_env_names: {report['n_unique_target_env_names']}"
    )
    lines.append(f"- BDDL missing: {report['n_bddl_missing']}")
    lines.append(f"- Predicate bodies missing from BDDL: {report['n_missing_predicate_bodies']}")
    lines.append(f"- Unresolvable target_task: {report['n_unresolvable']}")
    lines.append(
        f"- Env-load probed: **{'yes' if report['env_load_probed'] else 'no (skipped)'}**"
    )
    if not report["env_load_probed"]:
        lines.append(f"  - Skip reason: {report['env_load_skipped_reason']}")
        lines.append(
            "  - Fallback: BDDL existence + `predicates.resolve_task` only. "
            "Real LIBERO env instantiation was NOT exercised in this run, so a "
            "missing C-extension dependency (e.g. `robosuite`/`mujoco`) at GRPO "
            "time would not be caught by this audit. Re-run with `--probe-env` "
            "from a Python that can import "
            "`gr00t.eval.sim.LIBERO.libero_env.register_libero_envs` after "
            "installing the LIBERO sim deps."
        )
    else:
        lines.append(f"  - Per-env timeout: {report['env_timeout_s']}s")
        lines.append(f"  - Env-load failures: {report['n_env_load_failed']}")
    lines.append(
        f"- Rows at risk (any failure): {report['rows_at_risk']} "
        f"({100 * report['rows_at_risk_pct']:.2f}%)"
    )
    missing = report["canonical_tasks_missing_from_pairs"]
    if missing:
        lines.append(
            f"- Canonical tasks NOT seen in pairs ({len(missing)}): "
            + ", ".join(f"`{m}`" for m in missing)
        )
    lines.append("")

    pass_bddl = (
        report["n_bddl_missing"] == 0
        and report["n_unresolvable"] == 0
        and report["n_missing_predicate_bodies"] == 0
    )
    pass_env = (
        not report["env_load_probed"] or report["n_env_load_failed"] == 0
    )
    overall = "PASS" if (pass_bddl and pass_env) else "FAIL"
    lines.append(f"**GATE: {overall}**")
    lines.append("")

    lines.append("## Per `target_env_name`")
    lines.append("")
    if report["env_load_probed"]:
        lines.append(
            "| target_env_name | task_canonical | predicate | n_rows | bddl_found | env_loads | stage | t (s) | error |"
        )
        lines.append(
            "|---|---|---|---:|:---:|:---:|---|---:|---|"
        )
    else:
        lines.append(
            "| target_env_name | task_canonical | predicate | n_rows | bddl_found | error |"
        )
        lines.append("|---|---|---|---:|:---:|---|")

    for r in report["per_env"]:
        env_name = r["target_env_name"]
        canon = r["task_canonical"] or "?"
        pred = r["predicate_kind"] or "?"
        n_rows = r["n_rows_using_it"]
        bddl = "YES" if r["bddl_found"] else "no"
        err = "; ".join(r.get("errors") or []) or ""
        if report["env_load_probed"]:
            if r["env_loads"] is None:
                env_cell = "skip"
            else:
                env_cell = "YES" if r["env_loads"] else "no"
            stage = r.get("env_load_stage") or ""
            t_s = r.get("env_load_elapsed_s")
            t_str = f"{t_s:.2f}" if isinstance(t_s, (int, float)) else ""
            lines.append(
                f"| `{env_name}` | `{canon}` | `{pred}` | {n_rows} | {bddl} | "
                f"{env_cell} | {stage} | {t_str} | {err} |"
            )
        else:
            lines.append(
                f"| `{env_name}` | `{canon}` | `{pred}` | {n_rows} | {bddl} | {err} |"
            )
    lines.append("")

    risky = [r for r in report["per_env"] if r.get("errors")]
    if risky:
        rows_drop = sum(r["n_rows_using_it"] for r in risky)
        lines.append("## CF-pair rows that would FAIL at GRPO sim time")
        lines.append("")
        lines.append(
            f"Drop / filter the {rows_drop} rows whose `target_env_name` is one of:"
        )
        lines.append("")
        for r in risky:
            lines.append(
                f"- `{r['target_env_name']}` "
                f"({r['n_rows_using_it']} rows): "
                + "; ".join(r["errors"])
            )
        lines.append("")
    else:
        lines.append("## CF-pair rows that would FAIL at GRPO sim time")
        lines.append("")
        lines.append("None. All `target_env_name`s pass every check that was run.")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS,
        help="CF pairs JSONL (default: %(default)s).",
    )
    p.add_argument(
        "--bddl-dir",
        type=Path,
        default=DEFAULT_BDDL_DIR,
        help="LIBERO Goal BDDL dir (default: %(default)s).",
    )
    p.add_argument(
        "--probe-env",
        action="store_true",
        help="Also try to instantiate LiberoEnv via gym.make + reset() per "
             "unique target_env_name. Heavier; needs robosuite + mujoco. If "
             "the import probe fails, falls back to BDDL-only with a clear "
             "skip reason in the report.",
    )
    p.add_argument(
        "--python-exe",
        type=Path,
        default=DEFAULT_VENV_PY if DEFAULT_VENV_PY.exists() else Path(sys.executable),
        help="Python used to spawn env-load subprocesses "
             "(default: %(default)s).",
    )
    p.add_argument(
        "--libero-py",
        type=Path,
        default=DEFAULT_LIBERO_PY,
        help="Path prepended to PYTHONPATH so `import libero` works in the "
             "env-load subprocess (default: %(default)s).",
    )
    p.add_argument(
        "--gr00t-py",
        type=Path,
        default=DEFAULT_GR00T_PY,
        help="Path prepended to PYTHONPATH so `import gr00t...` works in the "
             "env-load subprocess (default: %(default)s).",
    )
    p.add_argument(
        "--env-timeout-s",
        type=float,
        default=30.0,
        help="Per-env reset timeout in seconds (default: %(default)s).",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override output JSON path. Default: <pairs>.sim_audit.json .",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Override output Markdown path. Default: <pairs>.sim_audit.md .",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pairs_path: Path = args.pairs
    if not pairs_path.exists():
        logger.error("CF pairs file not found: %s", pairs_path)
        return 2
    bddl_dir: Path = args.bddl_dir
    if not bddl_dir.exists():
        logger.error("BDDL dir not found: %s", bddl_dir)
        return 2

    out_json = args.out_json or pairs_path.with_suffix(pairs_path.suffix + ".sim_audit.json")
    out_md = args.out_md or pairs_path.with_suffix(pairs_path.suffix + ".sim_audit.md")

    t0 = time.time()
    report = audit(
        pairs_path,
        bddl_dir,
        probe_env=bool(args.probe_env),
        python_exe=args.python_exe,
        libero_py=args.libero_py,
        gr00t_py=args.gr00t_py,
        env_timeout_s=float(args.env_timeout_s),
    )
    report["audit_elapsed_s"] = round(time.time() - t0, 2)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=False))
    out_md.write_text(render_markdown(report))

    pass_bddl = (
        report["n_bddl_missing"] == 0
        and report["n_unresolvable"] == 0
        and report["n_missing_predicate_bodies"] == 0
    )
    pass_env = (
        not report["env_load_probed"] or report["n_env_load_failed"] == 0
    )
    overall = pass_bddl and pass_env

    logger.info(
        "Audit done in %.2fs. unique_envs=%d bddl_missing=%d unresolvable=%d "
        "env_load_probed=%s env_load_failed=%d rows_at_risk=%d (%.2f%%)",
        report["audit_elapsed_s"],
        report["n_unique_target_env_names"],
        report["n_bddl_missing"],
        report["n_unresolvable"],
        report["env_load_probed"],
        report["n_env_load_failed"],
        report["rows_at_risk"],
        100 * report["rows_at_risk_pct"],
    )
    print(f"GATE: {'PASS' if overall else 'FAIL'} (rows_at_risk={report['rows_at_risk']})")
    print(f"json: {out_json}")
    print(f"md:   {out_md}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

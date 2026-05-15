#!/usr/bin/env python
"""Constrained JSON LLM judge for the interpretability panel.

Given:
    - ``eval_cases.jsonl``  (frozen hypotheses)
    - ``panel_rows.jsonl``  (baseline/edited/control AV outputs from
                              ``run_interp_panel.py``)

The judge produces a strict-schema ``judge_rows.jsonl`` where every row
matches the schema declared in ``rubric.py``. The judge:

    * uses ``temperature=0`` and ``seed=<arg>`` for determinism;
    * is forced to JSON via OpenAI's structured-output (json_schema) interface;
    * is asked to provide verbatim quotes from baseline/edited/control as
      evidence (verified after parse against the actual source strings);
    * has its output passed through ``rubric.validate_judge_row`` before save.

The judge sees the panel evidence and the case hypothesis but does **not** see
the activation vector itself, the hidden state, or the model's parameters --
those go through deterministic auto metrics in ``score_panel.py``. The judge
is asked to score *the explanation*, never the model.

Optional dual-judge mode (``--dual-judge``) runs a second model on the same
inputs and persists both rows; agreement is computed by ``score_panel.py``.

Output schema
-------------
``judge_rows.jsonl`` rows::

    {
      "case_id":                  "case_000003",
      "specificity_0_3":          0..3,
      "consistency_0_3":          0..3,
      "confabulation_0_3":        0..3,
      "overall_faithfulness_0_3": 0..3,
      "confidence_0_1":           0..1,
      "evidence_spans":           [...],
      "rationale":                "<=2 sentences",
      "judge_model":              "<model id>",
      "judge_seed":               <int>,
      "_warnings":                [<validation notes>]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# We import rubric and judge_prompt next to this file. Append the script
# directory to sys.path so ``from rubric import ...`` works regardless of
# CWD when the script is invoked via ``python scripts/eval/run_llm_judge.py``.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rubric import (  # type: ignore  # noqa: E402
    ALLOWED_EVIDENCE_SOURCES,
    output_json_schema,
    render_rubric_for_prompt,
    validate_judge_row,
)

logger = logging.getLogger("nla.eval.judge")


SYSTEM_PROMPT_HEADER = (
    "You are an interpretability evaluator. Your job is to score the quality "
    "of natural-language explanations produced by an Activation Verbalizer "
    "(AV) for a vision-language-action (VLA) robot model.\n\n"
    "You will be shown a single case with three explanations:\n"
    "  - baseline_text: AV explanation on the original activation.\n"
    "  - edited_text:   AV explanation after a counterfactual edit to the "
    "activation (specified in 'intervention_spec').\n"
    "  - control_text:  AV explanation after a random matched-magnitude edit.\n\n"
    "You also receive the case 'hypothesis' and 'expected_direction'.\n\n"
    "Score the rubric dimensions below according to their **anchored "
    "definitions**. You MUST output strictly valid JSON matching the schema, "
    "with verbatim 'evidence_spans' quoted from baseline_text / edited_text / "
    "control_text. Do not invent quotes."
)


def _build_user_prompt(case: dict[str, Any], panel_row: dict[str, Any]) -> str:
    """Assemble the per-case prompt the judge sees."""
    return (
        f"CASE_ID: {case['case_id']}\n"
        f"POSITION_TYPE: {case.get('position_type')}\n"
        f"HYPOTHESIS: {case.get('hypothesis')}\n"
        f"EXPECTED_DIRECTION: {case.get('expected_direction')}\n"
        f"INTERVENTION_SPEC: {json.dumps(panel_row.get('intervention_spec', {}))}\n\n"
        f"baseline_text:\n{panel_row.get('baseline_text', '')}\n\n"
        f"edited_text:\n{panel_row.get('edited_text', '')}\n\n"
        f"control_text:\n{panel_row.get('control_text', '')}\n\n"
        "Score the case using the rubric. Output JSON only."
    )


def _build_system_prompt() -> str:
    return (
        SYSTEM_PROMPT_HEADER
        + "\n\nRUBRIC (use these anchors verbatim):\n\n"
        + render_rubric_for_prompt()
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _existing_judge_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["case_id"])
            except Exception:
                continue
    return out


def _call_openai_judge(
    *,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Invoke the OpenAI Chat Completions API with structured JSON output.

    Falls back gracefully if the model doesn't accept ``response_format`` of
    type ``json_schema`` (older models): we then ask for ``json_object`` and
    rely on the rubric validator to clean things up.
    """
    common = dict(
        model=model,
        temperature=0.0,
        seed=seed,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        resp = client.chat.completions.create(
            **common,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "interp_judge",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
    except Exception as e:
        logger.warning(
            "json_schema response_format failed (%s); falling back to json_object", e,
        )
        resp = client.chat.completions.create(
            **common,
            response_format={"type": "json_object"},
        )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Judge returned non-JSON content: %r", content[:200])
        return {}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cases", required=True, help="eval_cases.jsonl")
    p.add_argument("--panel", required=True, help="panel_rows.jsonl")
    p.add_argument("--out", required=True, help="judge_rows.jsonl output path")
    p.add_argument(
        "--model",
        default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-2024-08-06"),
        help="OpenAI judge model (default from $OPENAI_JUDGE_MODEL).",
    )
    p.add_argument(
        "--dual-judge-model",
        default=None,
        help="Optional second judge model for inter-judge agreement (writes "
             "rows with judge_model=<this> alongside primary).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip case_ids that already have a judged row in --out.",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cases_path = Path(args.cases)
    panel_path = Path(args.panel)
    out_path = Path(args.out)
    if not cases_path.is_file():
        logger.error("Cases file not found: %s", cases_path); return 2
    if not panel_path.is_file():
        logger.error("Panel file not found: %s", panel_path); return 2

    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "openai SDK not installed. `pip install 'openai>=1.50'`."
        )
        return 2

    client = OpenAI()  # honors OPENAI_API_KEY

    cases = {c["case_id"]: c for c in _load_jsonl(cases_path)}
    panel = {r["case_id"]: r for r in _load_jsonl(panel_path)}

    # Order judging by case order in eval_cases.jsonl for stable output.
    case_order = [c["case_id"] for c in _load_jsonl(cases_path)]
    if args.resume:
        already = _existing_judge_ids(out_path)
        case_order = [cid for cid in case_order if cid not in already]
        logger.info("Resume: skipping %d already-judged cases", len(already))

    if not case_order:
        logger.info("Nothing to do.")
        return 0

    schema = output_json_schema()
    system_prompt = _build_system_prompt()
    judge_models = [args.model] + ([args.dual_judge_model] if args.dual_judge_model else [])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and out_path.exists() else "w"

    n_written = 0
    with out_path.open(mode) as fout:
        for cid in case_order:
            case = cases.get(cid)
            row = panel.get(cid)
            if case is None or row is None:
                logger.warning("Skipping %s: missing case or panel row", cid)
                continue
            user_prompt = _build_user_prompt(case, row)
            sources = {
                "baseline_text": row.get("baseline_text", ""),
                "edited_text": row.get("edited_text", ""),
                "control_text": row.get("control_text", ""),
            }
            for judge_model in judge_models:
                raw = _call_openai_judge(
                    client=client,
                    model=judge_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    seed=args.seed,
                    schema=schema,
                )
                cleaned = validate_judge_row(
                    raw, sources=sources, case_id=cid, strict=False,
                )
                cleaned["judge_model"] = judge_model
                cleaned["judge_seed"] = args.seed
                fout.write(json.dumps(cleaned) + "\n")
                fout.flush()
                n_written += 1
            if n_written % 5 == 0:
                logger.info("  judged rows written: %d", n_written)

    logger.info("Wrote %d judge rows to %s", n_written, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

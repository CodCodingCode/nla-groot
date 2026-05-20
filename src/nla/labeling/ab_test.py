"""End-to-end A/B test orchestrator for prompt variants.

Round flow:

    1. Load the frozen eval set (data/prompt_ab/eval_set.jsonl).
    2. For each variant in the round:
       a. Run gpt-5.1-mini labeling against the eval set (with the variant's
          system prompt + response_format).
       b. Run deterministic scorers (qa_metrics) for axes (a) + (b)-auto + (c)-auto.
       c. Run the GPT-5.1 grader for (b)-LLM and (c)-LLM.
       d. Export 30 stratified samples for the Claude eye-check.
    3. Aggregate per-axis pass rates per variant.
    4. Emit scores.json and (variant -> labels.jsonl, grades.jsonl) under the
       round dir.
    5. Decide: is any variant at >=95% on all three combined axes?

Combined pass rates per axis:

  axis_a    = qa_metrics format       (auto-only)
  axis_b    = (b)-auto AND (b)-LLM    -- both must pass per the plan
  axis_c    = (c)-auto AND (c)-LLM    -- both must pass per the plan

A variant "passes" iff all three combined pass rates are >= 0.95.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from nla.labeling.grader import (
    DEFAULT_GRADER_MODEL,
    GradeInput,
    GradeResult,
    aggregate_llm_grades,
    export_claude_samples,
    grade_many_async,
    load_claude_grades,
)
from nla.labeling.openai_client import DEFAULT_MODEL as DEFAULT_LABELING_MODEL
from nla.labeling.prompts import PositionLabelInput
from nla.labeling.prompt_variants import VariantOutput, get_variant
from nla.labeling.qa_metrics import (
    LabelScores,
    VariantScorecard,
    score_and_aggregate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval-set loading
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    eval_id: str
    source: str
    example_id: str
    instruction: str
    decoded_text_context: str
    position_index: int
    position_type: str
    sequence_length: int
    image_patch_meta: tuple[int, int] | None
    image_paths: list[str]
    episode_index: int | None
    step_index: int | None
    state: list[float] | None = None
    state_name: str | None = None


def load_eval_set(path: str | Path) -> list[EvalRow]:
    path = Path(path)
    rows: list[EvalRow] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ipm = obj.get("image_patch_meta")
            rows.append(EvalRow(
                eval_id=obj["eval_id"],
                source=obj["source"],
                example_id=obj["example_id"],
                instruction=obj.get("instruction") or "",
                decoded_text_context=obj.get("decoded_text_context") or "",
                position_index=int(obj["position_index"]),
                position_type=obj["position_type"],
                sequence_length=int(obj["sequence_length"]),
                image_patch_meta=tuple(ipm) if ipm is not None else None,
                image_paths=list(obj["image_paths"]),
                episode_index=obj.get("episode_index"),
                step_index=obj.get("step_index"),
                state=obj.get("state"),
                state_name=obj.get("state_name"),
            ))
    return rows


def eval_row_to_position_input(row: EvalRow) -> PositionLabelInput:
    return PositionLabelInput(
        example_id=row.eval_id,
        instruction=row.instruction,
        decoded_text_context=row.decoded_text_context,
        position_index=row.position_index,
        position_type=row.position_type,  # type: ignore[arg-type]
        sequence_length=row.sequence_length,
        image_paths=list(row.image_paths),
        image_patch_meta=row.image_patch_meta,
        state=row.state,
        state_name=row.state_name,
        episode_index=row.episode_index,
        step_index=row.step_index,
        extra={"source": row.source, "source_example_id": row.example_id},
    )


# ---------------------------------------------------------------------------
# Variant labeling: a thin wrapper over openai_client that lets a variant
# inject its own (system_prompt, response_format, post_process).
# ---------------------------------------------------------------------------

@dataclass
class LabelRow:
    example_id: str
    description: str
    raw_response: str
    model: str
    elapsed_ms: float
    usage: dict
    error: str | None
    meta: dict


async def _label_one_variant_async(
    client,
    variant_id: str,
    variant_out: VariantOutput,
    inp: PositionLabelInput,
    model: str,
    sem: asyncio.Semaphore,
    max_retries: int,
    base_backoff: float,
) -> LabelRow:
    from nla.labeling.openai_client import _img_data_url

    content: list[dict] = [{"type": "text", "text": variant_out.user_prompt}]
    for p in inp.image_paths:
        content.append({"type": "image_url", "image_url": {"url": _img_data_url(p)}})
    messages = [
        {"role": "system", "content": variant_out.system_prompt},
        {"role": "user", "content": content},
    ]
    create_kwargs: dict = {"model": model, "messages": messages}
    if variant_out.response_format is not None:
        create_kwargs["response_format"] = variant_out.response_format
    temp = variant_out.meta.get("temperature")
    if temp is not None:
        create_kwargs["temperature"] = float(temp)

    last_err = "no attempt"
    backoff = base_backoff
    for attempt in range(max_retries):
        async with sem:
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(**create_kwargs)
                raw = (resp.choices[0].message.content or "").strip()
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                post_err: str | None = None
                if variant_out.post_process is not None:
                    try:
                        desc = variant_out.post_process(raw)
                    except Exception as e:
                        post_err = f"{type(e).__name__}: {e}"
                        desc = ""
                else:
                    desc = raw
                return LabelRow(
                    example_id=inp.example_id,
                    description=desc,
                    raw_response=raw,
                    model=model,
                    elapsed_ms=(time.time() - t0) * 1000,
                    usage=usage,
                    error=post_err,
                    meta={
                        "variant_id": variant_id,
                        "position_type": inp.position_type,
                        "position_index": inp.position_index,
                        "instruction": inp.instruction,
                        "source": inp.extra.get("source"),
                        "source_example_id": inp.extra.get("source_example_id"),
                        "temperature": variant_out.meta.get("temperature"),
                    },
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "variant %s label %s attempt %d failed: %s",
                    variant_id, inp.example_id, attempt + 1, last_err,
                )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
    return LabelRow(
        example_id=inp.example_id, description="", raw_response="",
        model=model, elapsed_ms=0.0, usage={}, error=last_err,
        meta={
            "variant_id": variant_id,
            "position_type": inp.position_type,
            "position_index": inp.position_index,
            "instruction": inp.instruction,
            "source": inp.extra.get("source"),
            "source_example_id": inp.extra.get("source_example_id"),
            "temperature": variant_out.meta.get("temperature"),
        },
    )


async def label_variant_async(
    variant_id: str,
    eval_rows: Sequence[EvalRow],
    output_jsonl: str | Path,
    *,
    model: str = DEFAULT_LABELING_MODEL,
    concurrency: int = 16,
    api_key: str | None = None,
    resume: bool = True,
    max_retries: int = 4,
    base_backoff: float = 1.0,
    progress_every: int = 25,
) -> int:
    """Label one variant over the entire eval set, with resume."""
    from openai import AsyncOpenAI

    variant_fn = get_variant(variant_id)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if resume and output_jsonl.exists():
        with output_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("description") and not obj.get("error"):
                    done_ids.add(obj["example_id"])

    todo: list[tuple[VariantOutput, PositionLabelInput]] = []
    for row in eval_rows:
        if row.eval_id in done_ids:
            continue
        inp = eval_row_to_position_input(row)
        vo = variant_fn(inp)
        todo.append((vo, inp))
    logger.info(
        "[%s] labeling: %d new, %d previously done -> %s",
        variant_id, len(todo), len(done_ids), output_jsonl,
    )
    if not todo:
        return 0

    client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    n_new = 0
    f = output_jsonl.open("a")
    try:
        async def run_one(vo, inp):
            nonlocal n_new
            res = await _label_one_variant_async(
                client, variant_id, vo, inp, model, sem, max_retries, base_backoff,
            )
            f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % progress_every == 0:
                logger.info("[%s]   %d / %d labeled", variant_id, n_new, len(todo))
            return res
        await asyncio.gather(*(run_one(vo, inp) for vo, inp in todo))
    finally:
        f.close()
        await client.close()
    return n_new


def load_label_rows(path: str | Path) -> list[dict]:
    """Load streaming label JSONL (one variant's output)."""
    path = Path(path)
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Per-variant grade-input construction (joins eval rows with label rows)
# ---------------------------------------------------------------------------

def build_grade_inputs(
    variant_id: str,
    eval_rows: Sequence[EvalRow],
    label_rows: Sequence[dict],
) -> list[GradeInput]:
    by_id = {r.eval_id: r for r in eval_rows}
    out: list[GradeInput] = []
    for lr in label_rows:
        eid = lr["example_id"]
        if eid not in by_id:
            continue
        if not lr.get("description") or lr.get("error"):
            continue
        er = by_id[eid]
        out.append(GradeInput(
            example_id=eid,
            variant_id=variant_id,
            description=lr["description"],
            instruction=er.instruction,
            position_type=er.position_type,
            image_paths=er.image_paths,
            seq_len=er.sequence_length,
            position_index=er.position_index,
        ))
    return out


def load_grade_rows(path: str | Path) -> list[GradeResult]:
    """Re-hydrate GradeResult objects from a saved grades.jsonl."""
    from nla.labeling.grader import AxisGrade
    path = Path(path)
    out: list[GradeResult] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            g = obj.get("grounding")
            a = obj.get("appropriateness")
            d = obj.get("template_distinguishable")
            out.append(GradeResult(
                example_id=obj["example_id"],
                variant_id=obj["variant_id"],
                grader=obj.get("grader", "gpt-5.1"),
                model=obj.get("model", DEFAULT_GRADER_MODEL),
                grounding=(AxisGrade(**g) if g else None),
                appropriateness=(AxisGrade(**a) if a else None),
                template_distinguishable=(AxisGrade(**d) if d else None),
                elapsed_ms=float(obj.get("elapsed_ms", 0.0)),
                usage=obj.get("usage", {}),
                error=obj.get("error"),
                raw_response=obj.get("raw_response"),
            ))
    return out


# ---------------------------------------------------------------------------
# Combined scorecard: deterministic + LLM grader for axes b and c
# ---------------------------------------------------------------------------

@dataclass
class CombinedVariantScorecard:
    variant_id: str
    round: int
    n_labels: int
    n_graded_llm: int
    # Per-axis combined pass rates (the criterion the plan checks against).
    pass_rate_a: float
    pass_rate_b_combined: float
    pass_rate_c_combined: float
    # Sub-rollups for diagnostics.
    auto: dict
    llm: dict
    per_position_type: dict
    passes_95: bool
    pass_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def combined_pass(
    label_scores: Sequence[LabelScores],
    grade_results: Sequence[GradeResult],
    *,
    pass_threshold: float = 0.95,
    variant_id: str,
    round_idx: int,
) -> CombinedVariantScorecard:
    """Combine deterministic (a/b_auto/c_auto) + LLM (b_llm/c_llm)."""
    auto_card = _auto_scorecard(label_scores, variant_id)
    llm_card = aggregate_llm_grades(variant_id, grade_results)

    by_id_grade = {(g.variant_id, g.example_id): g for g in grade_results}

    n = len(label_scores)
    n_pass_a = sum(1 for s in label_scores if s.passes_a)

    n_pass_b_combined = 0
    n_pass_c_combined = 0
    for s in label_scores:
        g = by_id_grade.get((variant_id, s.example_id))
        if g is None:
            continue
        if s.passes_b_auto and g.passes_b_llm:
            n_pass_b_combined += 1
        if s.passes_c_auto and g.passes_c_llm:
            n_pass_c_combined += 1

    pass_rate_a = (n_pass_a / n) if n else 0.0
    pass_rate_b_combined = (n_pass_b_combined / n) if n else 0.0
    pass_rate_c_combined = (n_pass_c_combined / n) if n else 0.0

    passes_95 = (
        pass_rate_a >= pass_threshold
        and pass_rate_b_combined >= pass_threshold
        and pass_rate_c_combined >= pass_threshold
    )

    # Per-position-type rollup using combined criteria.
    per_pos: dict[str, dict] = {}
    by_pos: dict[str, list[LabelScores]] = {}
    for s in label_scores:
        by_pos.setdefault(s.position_type, []).append(s)
    for pos, group in by_pos.items():
        ng = len(group)
        a_pass = sum(1 for s in group if s.passes_a) / ng if ng else 0.0
        b_pass = 0; c_pass = 0
        for s in group:
            g = by_id_grade.get((variant_id, s.example_id))
            if g is None: continue
            if s.passes_b_auto and g.passes_b_llm: b_pass += 1
            if s.passes_c_auto and g.passes_c_llm: c_pass += 1
        per_pos[pos] = {
            "n": ng,
            "a": a_pass,
            "b_combined": (b_pass / ng) if ng else 0.0,
            "c_combined": (c_pass / ng) if ng else 0.0,
        }

    return CombinedVariantScorecard(
        variant_id=variant_id,
        round=round_idx,
        n_labels=n,
        n_graded_llm=len(grade_results),
        pass_rate_a=pass_rate_a,
        pass_rate_b_combined=pass_rate_b_combined,
        pass_rate_c_combined=pass_rate_c_combined,
        auto=auto_card,
        llm={
            "pass_rate_b_llm": llm_card.pass_rate_b_llm,
            "pass_rate_c_llm": llm_card.pass_rate_c_llm,
            "top_b_failures": llm_card.top_b_failures,
            "top_c_failures": llm_card.top_c_failures,
        },
        per_position_type=per_pos,
        passes_95=passes_95,
        pass_threshold=pass_threshold,
    )


def _auto_scorecard(
    label_scores: Sequence[LabelScores], variant_id: str,
) -> dict:
    from nla.labeling.qa_metrics import aggregate_variant
    card = aggregate_variant(variant_id, label_scores)
    return card.to_dict()


# ---------------------------------------------------------------------------
# Round orchestrator
# ---------------------------------------------------------------------------

@dataclass
class RoundConfig:
    round_idx: int
    variants: list[str]
    eval_set_path: Path
    out_dir: Path
    label_model: str = DEFAULT_LABELING_MODEL
    grader_model: str = DEFAULT_GRADER_MODEL
    label_concurrency: int = 16
    grade_concurrency: int = 16
    skip_distinctness: bool = False
    pass_threshold: float = 0.95
    api_key: str | None = None
    claude_n_per_position: int = 10


async def run_round_async(cfg: RoundConfig) -> dict[str, CombinedVariantScorecard]:
    eval_rows = load_eval_set(cfg.eval_set_path)
    logger.info("Round %d: loaded %d eval rows", cfg.round_idx, len(eval_rows))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Single embedder reused across variants.
    embedder = None
    if not cfg.skip_distinctness:
        from nla.labeling.qa_metrics import _load_embedder
        embedder = _load_embedder()

    scorecards: dict[str, CombinedVariantScorecard] = {}

    for variant in cfg.variants:
        variant_dir = cfg.out_dir / f"variant_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        labels_jsonl = variant_dir / "labels.jsonl"
        grades_jsonl = variant_dir / "grades.jsonl"

        logger.info("[%s] STEP 1/3: labeling", variant)
        await label_variant_async(
            variant, eval_rows, labels_jsonl,
            model=cfg.label_model,
            concurrency=cfg.label_concurrency,
            api_key=cfg.api_key,
        )
        label_rows = load_label_rows(labels_jsonl)

        logger.info("[%s] STEP 2/3: deterministic scoring", variant)
        label_scores, _auto_card = score_and_aggregate(
            variant, label_rows,
            embedder=embedder,
            skip_distinctness=cfg.skip_distinctness,
        )

        logger.info("[%s] STEP 3/3: GPT-5.1 grading", variant)
        grade_inputs = build_grade_inputs(variant, eval_rows, label_rows)
        await grade_many_async(
            grade_inputs, grades_jsonl,
            model=cfg.grader_model,
            concurrency=cfg.grade_concurrency,
            api_key=cfg.api_key,
        )
        grade_rows = load_grade_rows(grades_jsonl)

        # Claude eye-check sample export.
        claude_dir = cfg.out_dir / "claude_samples"
        claude_dir.mkdir(parents=True, exist_ok=True)
        export_claude_samples(
            grade_inputs, claude_dir,
            n_per_position_type=cfg.claude_n_per_position,
            seed=cfg.round_idx * 1000,
        )

        card = combined_pass(
            label_scores, grade_rows,
            pass_threshold=cfg.pass_threshold,
            variant_id=variant,
            round_idx=cfg.round_idx,
        )
        scorecards[variant] = card
        logger.info(
            "[%s] DONE  a=%.3f  b=%.3f  c=%.3f  pass95=%s",
            variant, card.pass_rate_a, card.pass_rate_b_combined,
            card.pass_rate_c_combined, card.passes_95,
        )

    # Save scores.json
    scores_path = cfg.out_dir / "scores.json"
    scores_path.write_text(json.dumps(
        {v: card.to_dict() for v, card in scorecards.items()},
        indent=2, default=str,
    ))
    logger.info("wrote %s", scores_path)

    return scorecards


def run_round_sync(cfg: RoundConfig) -> dict[str, CombinedVariantScorecard]:
    return asyncio.run(run_round_async(cfg))


# ---------------------------------------------------------------------------
# Disagreement detection (post-Claude grading)
# ---------------------------------------------------------------------------

def detect_round_disagreements(round_dir: str | Path) -> dict[str, list[dict]]:
    """For each variant in a round, compare GPT-5.1 vs (filled-in) Claude grades.

    Returns ``{variant_id: [disagreement_dicts]}``.
    """
    from nla.labeling.grader import find_disagreements

    round_dir = Path(round_dir)
    claude_dir = round_dir / "claude_samples"
    claude_grades = load_claude_grades(claude_dir)
    if not claude_grades:
        return {}
    out: dict[str, list[dict]] = {}
    for vdir in sorted(round_dir.glob("variant_*")):
        variant = vdir.name[len("variant_"):]
        grade_rows = load_grade_rows(vdir / "grades.jsonl")
        if not grade_rows:
            continue
        disagreements = find_disagreements(grade_rows, claude_grades)
        if disagreements:
            out[variant] = [asdict(d) for d in disagreements]
    return out


__all__ = [
    "EvalRow",
    "load_eval_set",
    "eval_row_to_position_input",
    "LabelRow",
    "label_variant_async",
    "load_label_rows",
    "build_grade_inputs",
    "load_grade_rows",
    "CombinedVariantScorecard",
    "combined_pass",
    "RoundConfig",
    "run_round_async",
    "run_round_sync",
    "detect_round_disagreements",
]

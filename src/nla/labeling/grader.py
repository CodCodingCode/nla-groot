"""LLM graders for the prompt A/B test.

Two graders ship here:

1. ``GPT51Grader``: programmatic grader that calls ``gpt-5.1`` (full model, not
   mini) with the same image(s) the labeler saw, plus the candidate label, and
   asks three pass/fail questions per the plan's operational definitions:

   (b) grounding:                 Specific vs Generic           (axis b -- LLM half)
   (c) appropriateness:           Appropriate vs Inappropriate  (axis c -- LLM half)
   (d) template_distinguishable:  Specific vs Template          (axis d -- anti-collapse)

   Axis (d) is a stricter, V2-collapse-specific version of axis (b). Axis (b)
   asks "could this describe a different scene?" and is satisfied by any
   minimally grounded language. Axis (d) asks "would this same caption,
   verbatim, be a good label for many different but similar manipulation
   scenes?" — a Yes flags the template-collapse failure mode where V2 emitted
   reusable boilerplate that happened to mention the correct workspace.

   Structured JSON output is enforced via ``response_format={"type": "json_object"}``
   so we can parse without regex hacks. ``gpt-5.1`` supports the json_object mode
   when images are attached; if a future endpoint upgrade brings full json_schema
   for multimodal payloads we switch over -- in the meantime the user-prompt
   pins the shape and we validate it client-side.

2. ``ClaudeSampleExporter``: deterministically samples 30 labels per variant
   (10 per position type) and writes them to ``claude_samples/<variant>.jsonl``
   in a render-ready shape (label + image paths + instruction + a blank grade
   slot per axis) for Claude to fill in interactively.

Both graders return a uniform ``GradeResult`` shape so the A/B orchestrator can
merge them with the deterministic scorers in ``qa_metrics``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRADER_MODEL = os.environ.get("OPENAI_GRADER_MODEL", "gpt-5.1")

GRADE_SYSTEM = """You are a strict, calibrated grader for interpretability \
labels of a vision-language-action robot model (GR00T N1.7).

You will see:
- The robot's task instruction.
- One or more camera frames from the same step the label describes.
- The token-position type the label is targeting (last_text, image_patch, \
anchor, fallback).
- A candidate label: 4-5 bullets describing what the model is internally \
tracking at that token position.

You must grade the label on exactly three axes and return a single JSON \
object with the keys described below. Be terse and decisive.

Axis B -- GROUNDING: does this label describe THIS scene specifically, or \
could it plausibly describe a *different* manipulation scene without changes? \
Pass = "specific" (concrete objects, colors, named distractors, or spatial \
relations grounded in the visible image). Fail = "generic" (vague templates \
that would describe any tabletop manipulation, or content that contradicts \
the visible scene).

Axis C -- APPROPRIATENESS: does the label describe what a VLM's middle layer \
would plausibly track for the next action -- scene / object / spatial / plan \
content -- WITHOUT inventing precise numeric measurements, ascribing affect \
(feels/wants/thinks/decides/believes/hopes), or specifying actuator-level \
commands (joint angles, force percentages, torque, motor commands)? \
Pass = "appropriate". Fail = "inappropriate".

Axis D -- TEMPLATE_DISTINGUISHABLE (anti-collapse, stricter than B): would \
this exact same label, verbatim, also be a reasonable description for many \
*different but similar* manipulation scenes (e.g. another LIBERO task with a \
different target object on the same workspace)? \
Pass = "specific" (the label commits to scene-fingerprinting details -- a \
named target object/colour, a specific distractor, a one-of-a-kind spatial \
relation -- that would NOT generalise to a different similar scene). \
Fail = "template" (the label is reusable boilerplate that happens to mention \
the workspace; swap one object name and it would describe a different task \
just as well, or the bullets follow a fixed schema rather than describing \
unique features of *this* frame).

Return strictly valid JSON of the form:
{
  "grounding": {"verdict": "specific" | "generic", "reason": "<one short sentence>"},
  "appropriateness": {"verdict": "appropriate" | "inappropriate", "reason": "<one short sentence>"},
  "template_distinguishable": {"verdict": "specific" | "template", "reason": "<one short sentence>"}
}

No preamble, no markdown fences, no extra keys.
"""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AxisGrade:
    verdict: str        # "specific"/"generic" for B, "appropriate"/"inappropriate" for C
    reason: str
    passed: bool        # True if verdict is the positive class.


@dataclass
class GradeResult:
    example_id: str
    variant_id: str
    grader: str               # "gpt-5.1", "claude", etc.
    model: str
    grounding: AxisGrade | None
    appropriateness: AxisGrade | None
    elapsed_ms: float
    usage: dict = field(default_factory=dict)
    error: str | None = None
    raw_response: str | None = None
    # Optional 3rd axis added for the V3 anti-template-collapse eval. Older
    # grade JSONL rows pre-V3 lack this field and load with ``None``.
    template_distinguishable: AxisGrade | None = None

    @property
    def passes_b_llm(self) -> bool:
        return self.grounding is not None and self.grounding.passed

    @property
    def passes_c_llm(self) -> bool:
        return self.appropriateness is not None and self.appropriateness.passed

    @property
    def passes_d_llm(self) -> bool:
        return (
            self.template_distinguishable is not None
            and self.template_distinguishable.passed
        )


# ---------------------------------------------------------------------------
# OpenAI client helpers (mirror nla.labeling.openai_client style)
# ---------------------------------------------------------------------------

def _img_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def _get_openai():
    try:
        from openai import AsyncOpenAI, OpenAI
        return OpenAI, AsyncOpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "openai is not installed in this environment. "
            "Install with `pip install 'openai>=1.50'`."
        ) from e


# Pre-compiled fallback parser: pull the first balanced JSON object out of a
# response that *should* already be JSON but may be wrapped in stray prose.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_grade(
    text: str,
) -> tuple[AxisGrade | None, AxisGrade | None, AxisGrade | None, str | None]:
    """Try to parse a grade JSON object out of model output. Returns
    ``(grounding, appropriateness, template_distinguishable, error_or_None)``.

    Tolerant of stray markdown fences but rejects ill-typed JSON.
    The 3rd axis is optional: if absent in the JSON it parses as ``None``
    rather than as an error (backward compatibility with B/C-only graders).
    """
    s = text.strip()
    if s.startswith("```"):
        # strip ``` ... ``` (with or without a language hint)
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    obj: dict | None = None
    try:
        obj = json.loads(s)
    except Exception:
        m = _JSON_OBJECT_RE.search(s)
        if m is not None:
            try:
                obj = json.loads(m.group(0))
            except Exception as e2:
                return None, None, None, f"json_parse: {e2}"
        else:
            return None, None, None, "no_json_object"
    if not isinstance(obj, dict):
        return None, None, None, "json_not_object"

    assert obj is not None
    obj_dict = obj  # narrow type for closure below

    def _axis(key: str, pos: str, neg: str) -> tuple[AxisGrade | None, str | None]:
        v = obj_dict.get(key)
        if not isinstance(v, dict):
            return None, f"missing:{key}"
        verdict = (v.get("verdict") or "").strip().lower()
        reason = (v.get("reason") or "").strip()
        if verdict not in (pos, neg):
            return None, f"bad_verdict:{key}:{verdict!r}"
        return AxisGrade(verdict=verdict, reason=reason, passed=(verdict == pos)), None

    g, e1 = _axis("grounding", "specific", "generic")
    a, e2 = _axis("appropriateness", "appropriate", "inappropriate")
    # Axis D is optional for backward compatibility: legacy graders return
    # only B/C; we treat its absence as ``None`` (i.e. "not graded on this
    # axis") rather than as an error.
    d, e3 = (None, None)
    if "template_distinguishable" in obj:
        d, e3 = _axis("template_distinguishable", "specific", "template")
    err = e1 or e2 or e3
    return g, a, d, err


# ---------------------------------------------------------------------------
# Per-label payload construction
# ---------------------------------------------------------------------------

@dataclass
class GradeInput:
    """Everything the grader needs about one (label, eval_row) pair."""
    example_id: str
    variant_id: str
    description: str
    instruction: str
    position_type: str
    image_paths: list[str]
    seq_len: int | None = None
    position_index: int | None = None


def _build_grade_messages(inp: GradeInput) -> list[dict]:
    pos_clause = (
        f"Token position: {inp.position_type}"
        + (f" (index {inp.position_index} of {inp.seq_len})" if inp.position_index is not None else "")
    )
    user_text = (
        f'Task instruction: "{inp.instruction or "(no instruction provided)"}"\n'
        f"{pos_clause}\n\n"
        "Candidate label (4-5 bullets):\n"
        f"<label>\n{inp.description.strip()}\n</label>\n\n"
        "Grade the label on (B) grounding, (C) appropriateness, and "
        "(D) template_distinguishable per the system instructions, and return "
        "the JSON object."
    )
    content: list[dict] = [{"type": "text", "text": user_text}]
    for p in inp.image_paths:
        content.append({"type": "image_url", "image_url": {"url": _img_data_url(p)}})
    return [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# GPT-5.1 grader
# ---------------------------------------------------------------------------

async def _grade_one_async(
    client,
    inp: GradeInput,
    model: str,
    sem: asyncio.Semaphore,
    max_retries: int,
    base_backoff: float,
) -> GradeResult:
    msgs = _build_grade_messages(inp)
    last_err = "no attempt"
    backoff = base_backoff
    for attempt in range(max_retries):
        async with sem:
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    response_format={"type": "json_object"},
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                g, a, d, parse_err = _parse_grade(text)
                return GradeResult(
                    example_id=inp.example_id, variant_id=inp.variant_id,
                    grader="gpt-5.1", model=model,
                    grounding=g, appropriateness=a,
                    template_distinguishable=d,
                    elapsed_ms=(time.time() - t0) * 1000, usage=usage,
                    error=parse_err, raw_response=text,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "grade_one %s attempt %d failed: %s",
                    inp.example_id, attempt + 1, last_err,
                )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
    return GradeResult(
        example_id=inp.example_id, variant_id=inp.variant_id,
        grader="gpt-5.1", model=model,
        grounding=None, appropriateness=None,
        elapsed_ms=0.0, usage={}, error=last_err, raw_response=None,
    )


async def grade_many_async(
    inputs: Iterable[GradeInput],
    output_jsonl: str | Path,
    *,
    model: str = DEFAULT_GRADER_MODEL,
    concurrency: int = 16,
    api_key: str | None = None,
    resume: bool = True,
    max_retries: int = 4,
    base_backoff: float = 1.0,
    progress_every: int = 25,
) -> int:
    """Grade many labels concurrently; stream JSONL with resume on (variant_id, example_id)."""
    _, AsyncOpenAI = _get_openai()
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple[str, str]] = set()
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
                # Resume policy: a row is "done" if all axes we currently
                # grade are present and the row didn't error out. Legacy
                # B/C-only rows pre-V3 are *not* re-graded (we treat axis D
                # as absent rather than failing), to preserve historical
                # judge outputs; if you want to upgrade them, delete the
                # JSONL and rerun.
                if (
                    obj.get("grounding")
                    and obj.get("appropriateness")
                    and not obj.get("error")
                ):
                    done_keys.add((obj["variant_id"], obj["example_id"]))

    todo = [
        i for i in inputs
        if (i.variant_id, i.example_id) not in done_keys
    ]
    logger.info(
        "Grading: %d new, %d previously done -> %s",
        len(todo), len(done_keys), output_jsonl,
    )
    if not todo:
        return 0

    client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(concurrency)
    n_new = 0
    f = output_jsonl.open("a")
    try:
        async def run_one(inp):
            nonlocal n_new
            res = await _grade_one_async(client, inp, model, sem, max_retries, base_backoff)
            row = _grade_to_row(res)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % progress_every == 0:
                logger.info("  %d / %d graded", n_new, len(todo))
            return res

        await asyncio.gather(*(run_one(i) for i in todo))
    finally:
        f.close()
        await client.close()
    return n_new


def _grade_to_row(res: GradeResult) -> dict:
    return {
        "example_id": res.example_id,
        "variant_id": res.variant_id,
        "grader": res.grader,
        "model": res.model,
        "grounding": asdict(res.grounding) if res.grounding else None,
        "appropriateness": asdict(res.appropriateness) if res.appropriateness else None,
        "template_distinguishable": (
            asdict(res.template_distinguishable)
            if res.template_distinguishable
            else None
        ),
        "elapsed_ms": res.elapsed_ms,
        "usage": res.usage,
        "error": res.error,
        "raw_response": res.raw_response,
    }


def grade_to_row(res: GradeResult) -> dict:
    """Public alias for grade serialization."""
    return _grade_to_row(res)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class LLMVariantRollup:
    variant_id: str
    grader: str
    n_graded: int
    pass_rate_b_llm: float
    pass_rate_c_llm: float
    top_b_failures: list[tuple[str, int]]
    top_c_failures: list[tuple[str, int]]
    # Axis D was added for V3 anti-template-collapse eval; older rollups
    # produced before V3 carry zero counts here.
    pass_rate_d_llm: float = 0.0
    n_graded_d: int = 0
    top_d_failures: list[tuple[str, int]] = field(default_factory=list)


def aggregate_llm_grades(
    variant_id: str,
    grades: Sequence[GradeResult],
    *,
    top_k: int = 6,
    grader: str = "gpt-5.1",
) -> LLMVariantRollup:
    """Roll a list of GradeResult up to per-variant pass rates."""
    from collections import Counter
    n = len(grades)
    n_b = sum(g.passes_b_llm for g in grades)
    n_c = sum(g.passes_c_llm for g in grades)
    # Axis D pass-rate is computed only over rows that actually have axis D
    # populated, so legacy B/C-only grades don't artificially deflate it.
    d_grades = [g for g in grades if g.template_distinguishable is not None]
    n_d_total = len(d_grades)
    n_d = sum(g.passes_d_llm for g in d_grades)
    b_reasons: Counter = Counter()
    c_reasons: Counter = Counter()
    d_reasons: Counter = Counter()
    for g in grades:
        if not g.passes_b_llm and g.grounding is not None:
            b_reasons[g.grounding.reason[:80] or "no_reason"] += 1
        if not g.passes_c_llm and g.appropriateness is not None:
            c_reasons[g.appropriateness.reason[:80] or "no_reason"] += 1
        if g.template_distinguishable is not None and not g.passes_d_llm:
            d_reasons[g.template_distinguishable.reason[:80] or "no_reason"] += 1
    return LLMVariantRollup(
        variant_id=variant_id,
        grader=grader,
        n_graded=n,
        pass_rate_b_llm=(n_b / n) if n else 0.0,
        pass_rate_c_llm=(n_c / n) if n else 0.0,
        pass_rate_d_llm=(n_d / n_d_total) if n_d_total else 0.0,
        n_graded_d=n_d_total,
        top_b_failures=b_reasons.most_common(top_k),
        top_c_failures=c_reasons.most_common(top_k),
        top_d_failures=d_reasons.most_common(top_k),
    )


# ---------------------------------------------------------------------------
# Claude stratified-sample exporter
# ---------------------------------------------------------------------------

@dataclass
class ClaudeSampleRow:
    """A single row exported for Claude eye-checking.

    Layout intentionally minimal so the eye-check in-session is fast: image
    paths first, then label, with two blank grade slots.
    """
    variant_id: str
    example_id: str
    position_type: str
    instruction: str
    image_paths: list[str]
    description: str
    grade_b_grounding: dict = field(
        default_factory=lambda: {"verdict": None, "reason": ""}
    )
    grade_c_appropriateness: dict = field(
        default_factory=lambda: {"verdict": None, "reason": ""}
    )


def export_claude_samples(
    grade_inputs: Sequence[GradeInput],
    out_dir: str | Path,
    *,
    n_per_position_type: int = 10,
    position_types: Sequence[str] = ("last_text", "image_patch", "anchor"),
    seed: int = 0,
) -> Path:
    """Stratified sample writer: 10 (default) per position type per variant.

    ``grade_inputs`` should already contain only one variant's labels; the
    writer creates ``out_dir/<variant_id>.jsonl`` and refuses to overwrite an
    existing file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = sorted({g.variant_id for g in grade_inputs})
    written_paths: list[Path] = []
    for variant in variants:
        rng = np.random.default_rng(seed + hash(variant) % 100_000)
        v_inputs = [g for g in grade_inputs if g.variant_id == variant]
        by_pos: dict[str, list[GradeInput]] = {p: [] for p in position_types}
        for g in v_inputs:
            if g.position_type in by_pos:
                by_pos[g.position_type].append(g)
        sampled: list[GradeInput] = []
        for pos in position_types:
            pool = by_pos[pos]
            if not pool:
                logger.warning(
                    "variant %s has no labels for position_type=%s; "
                    "Claude sample will undershoot.", variant, pos,
                )
                continue
            k = min(n_per_position_type, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            sampled.extend(pool[i] for i in sorted(idx.tolist()))
        out_path = out_dir / f"{variant}.jsonl"
        with out_path.open("w") as f:
            for g in sampled:
                row = ClaudeSampleRow(
                    variant_id=g.variant_id,
                    example_id=g.example_id,
                    position_type=g.position_type,
                    instruction=g.instruction,
                    image_paths=g.image_paths,
                    description=g.description,
                )
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        written_paths.append(out_path)
        logger.info("wrote %s (%d rows)", out_path, len(sampled))
    return out_dir


def load_claude_grades(claude_dir: str | Path) -> dict[tuple[str, str], dict]:
    """Load Claude's filled-in grades from ``<claude_dir>/<variant>.jsonl``.

    Returns a dict keyed by ``(variant_id, example_id)`` -> the parsed row.

    Rows with both grade verdicts still ``None`` are skipped (not yet graded).
    """
    claude_dir = Path(claude_dir)
    out: dict[tuple[str, str], dict] = {}
    if not claude_dir.exists():
        return out
    for jsonl in sorted(claude_dir.glob("*.jsonl")):
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                b = obj.get("grade_b_grounding") or {}
                c = obj.get("grade_c_appropriateness") or {}
                if b.get("verdict") is None and c.get("verdict") is None:
                    continue
                out[(obj["variant_id"], obj["example_id"])] = obj
    return out


# ---------------------------------------------------------------------------
# Disagreement detection (GPT-5.1 vs Claude)
# ---------------------------------------------------------------------------

@dataclass
class GraderDisagreement:
    variant_id: str
    example_id: str
    axis: str               # "grounding" or "appropriateness"
    gpt_verdict: str | None
    claude_verdict: str | None


def find_disagreements(
    gpt_grades: Sequence[GradeResult],
    claude_grades: dict[tuple[str, str], dict],
) -> list[GraderDisagreement]:
    """Compare GPT-5.1 vs Claude on the labels Claude also graded."""
    out: list[GraderDisagreement] = []
    by_key: dict[tuple[str, str], GradeResult] = {
        (g.variant_id, g.example_id): g for g in gpt_grades
    }
    for key, claude_row in claude_grades.items():
        gpt = by_key.get(key)
        if gpt is None:
            continue
        cb = (claude_row.get("grade_b_grounding") or {}).get("verdict")
        cc = (claude_row.get("grade_c_appropriateness") or {}).get("verdict")
        gb = gpt.grounding.verdict if gpt.grounding else None
        gc = gpt.appropriateness.verdict if gpt.appropriateness else None
        if cb is not None and gb is not None and cb != gb:
            out.append(GraderDisagreement(
                variant_id=key[0], example_id=key[1], axis="grounding",
                gpt_verdict=gb, claude_verdict=cb,
            ))
        if cc is not None and gc is not None and cc != gc:
            out.append(GraderDisagreement(
                variant_id=key[0], example_id=key[1], axis="appropriateness",
                gpt_verdict=gc, claude_verdict=cc,
            ))
    return out


__all__ = [
    "DEFAULT_GRADER_MODEL",
    "GRADE_SYSTEM",
    "AxisGrade",
    "GradeResult",
    "GradeInput",
    "grade_many_async",
    "grade_to_row",
    "aggregate_llm_grades",
    "LLMVariantRollup",
    "ClaudeSampleRow",
    "export_claude_samples",
    "load_claude_grades",
    "GraderDisagreement",
    "find_disagreements",
]

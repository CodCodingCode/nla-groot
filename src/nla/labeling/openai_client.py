"""Async + sync OpenAI labeling client.

Submits per-position (or legacy per-step) labeling jobs to a multimodal model
(default ``gpt-5.1-mini`` via ``OPENAI_LABELING_MODEL``).  Streams results
to a JSONL file with resume support.

Result schema (one row per ``LabelResult.example_id``)::

    {
      "example_id": ...,
      "description": "<bulleted text>",
      "model": ...,
      "elapsed_ms": ...,
      "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
      "error": null | "<str>",
      "kind": "position" | "step",
      "meta": {<input metadata for traceability>}
    }
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from nla.labeling.prompts import (
    LabelInput,
    PositionLabelInput,
    build_position_prompt,
    build_step_prompt,
    build_strict_position_prompt,
)
from nla.labeling.prompts_v5 import StepLabelInput, V5_NESTED_JSON_SCHEMA, build_v5_step_prompt

logger = logging.getLogger(__name__)


DEFAULT_MODEL = os.environ.get("OPENAI_LABELING_MODEL", "gpt-5.1-mini")


# ---------------------------------------------------------------------------
# Position-prompt builder dispatch.
#
# Production labeling has historically called ``build_position_prompt`` (V3)
# unconditionally.  Phase-1 LIBERO repair (May-2026; see
# ``docs/sft_plan/v4_repair/``) introduces a V4 builder that drops the
# ``gripper`` / ``motion`` / ``image_region`` headers, bans scaffold-leakage
# phrases, and conditions the last bullet on the position type.  We dispatch
# to the right builder based on ``NLA_POSITION_PROMPT_MODE`` (or the
# ``--prompt-mode`` CLI flag, which sets the env var before pipeline import)
# so the same labeling pipeline can re-label data under either prompt.
# ---------------------------------------------------------------------------

_PROMPT_MODE = os.environ.get("NLA_POSITION_PROMPT_MODE", "v3").lower()


def _select_position_builder(mode: str | None = None) -> Callable:
    """Return the position-prompt builder for the requested mode.

    ``mode`` defaults to the current ``NLA_POSITION_PROMPT_MODE`` env var (or
    ``"v3"``) so callers and tests can override it dynamically without
    re-importing the module.  Unknown modes silently fall back to V3 so a
    typo in the env var never crashes a long-running label job.
    """
    if mode is None:
        mode = os.environ.get("NLA_POSITION_PROMPT_MODE", _PROMPT_MODE)
    mode = (mode or "v3").lower()
    if mode in ("v4", "v4_position", "v4-position"):
        from nla.labeling.prompts import build_v4_position_prompt
        return build_v4_position_prompt
    if mode in ("v5", "v5_step", "v5-step"):
        return _v5_step_builder_stub
    if mode in ("strict", "v3_strict"):
        return build_strict_position_prompt
    return build_position_prompt


def _v5_step_builder_stub(inp: PositionLabelInput) -> tuple[str, str]:
    """Dispatch V5 step labeling from a position-shaped input (pipeline stub)."""
    from nla.labeling.prompts_v5 import build_v5_step_prompt, position_input_to_step

    return build_v5_step_prompt(position_input_to_step(inp))


def _call_position_builder(
    builder: Callable,
    inp: PositionLabelInput,
) -> tuple[str, str]:
    """Invoke a position-prompt builder, threading ``suite`` if it accepts one.

    V3 builders (``build_position_prompt``, ``build_strict_position_prompt``)
    take only ``(inp,)``; the V4 builder also accepts ``suite=...``.  We probe
    the signature once and fall back to the unary call on a ``TypeError`` so
    a custom builder that rejects unexpected kwargs still works.
    """
    suite = getattr(inp, "suite", None)
    if suite is not None:
        try:
            params = inspect.signature(builder).parameters
        except (TypeError, ValueError):
            params = None
        accepts_suite = (
            params is not None
            and (
                "suite" in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        )
        if accepts_suite:
            try:
                return builder(inp, suite=suite)
            except TypeError:
                pass
    return builder(inp)


# ---------------------------------------------------------------------------
# Lazy openai import so the rest of the module loads without the SDK present.
# ---------------------------------------------------------------------------

def _get_openai():
    try:
        from openai import AsyncOpenAI, OpenAI
        return OpenAI, AsyncOpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "openai is not installed in this environment. "
            "Install with `pip install 'openai>=1.50'`."
        ) from e


# ---------------------------------------------------------------------------
# Result + small helpers
# ---------------------------------------------------------------------------

@dataclass
class LabelResult:
    example_id: str
    description: str
    model: str
    elapsed_ms: float
    usage: dict
    error: str | None = None
    kind: str = "position"
    meta: dict = field(default_factory=dict)


def _is_v5_label_mode() -> bool:
    mode = os.environ.get("NLA_POSITION_PROMPT_MODE", "v3").lower()
    return mode in ("v5", "v5_step", "v5-step")


def _completion_kwargs(model: str, messages: list[dict]) -> dict:
    kwargs: dict = {"model": model, "messages": messages}
    if _is_v5_label_mode():
        kwargs["response_format"] = V5_NESTED_JSON_SCHEMA
    return kwargs


def _img_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{b64}"


def _build_messages(inp) -> tuple[list[dict], str, dict]:
    """Construct the OpenAI ``messages`` payload for a single input.

    Returns ``(messages, kind, meta_for_logging)``.
    """
    if isinstance(inp, StepLabelInput):
        sys_p, user_p = build_v5_step_prompt(inp)
        image_paths = inp.image_paths
        kind = "step"
        meta = {
            "episode_index": inp.episode_index,
            "step_index": inp.step_index,
            "instruction": inp.instruction,
            "source_example_id": inp.example_id,
            "label_schema": "v5",
        }
    elif isinstance(inp, PositionLabelInput):
        builder = _select_position_builder()
        sys_p, user_p = _call_position_builder(builder, inp)
        image_paths = inp.image_paths
        kind = "position"
        meta = {
            "position_index": inp.position_index,
            "position_type": inp.position_type,
            "seq_len": inp.sequence_length,
            "image_patch_meta": list(inp.image_patch_meta)
            if inp.image_patch_meta is not None
            else None,
            "episode_index": inp.episode_index,
            "step_index": inp.step_index,
            "instruction": inp.instruction,
            "source_example_id": inp.extra.get("source_example_id"),
        }
    elif isinstance(inp, LabelInput):
        sys_p, user_p = build_step_prompt(inp)
        image_paths = [inp.image_path]
        kind = "step"
        meta = {
            "episode_id": inp.episode_id,
            "timestep": inp.timestep,
            "instruction": inp.instruction,
        }
    else:
        raise TypeError(f"Unsupported labeling input type: {type(inp).__name__}")

    content: list[dict] = [{"type": "text", "text": user_p}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _img_data_url(p)}})

    return (
        [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": content},
        ],
        kind,
        meta,
    )


# ---------------------------------------------------------------------------
# Sync (one-shot, for tests / small batches)
# ---------------------------------------------------------------------------

def label_one(
    inp,
    *,
    model: str = DEFAULT_MODEL,
    client=None,
) -> LabelResult:
    OpenAI, _ = _get_openai()
    client = client or OpenAI()
    messages, kind, meta = _build_messages(inp)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**_completion_kwargs(model, messages))
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return LabelResult(
            example_id=inp.example_id, description=text, model=model,
            elapsed_ms=(time.time() - t0) * 1000, usage=usage,
            kind=kind, meta=meta,
        )
    except Exception as e:
        return LabelResult(
            example_id=inp.example_id, description="", model=model,
            elapsed_ms=(time.time() - t0) * 1000, usage={},
            error=str(e), kind=kind, meta=meta,
        )


# ---------------------------------------------------------------------------
# Async streaming runner with resume
# ---------------------------------------------------------------------------

async def _label_one_async(client, inp, model: str, sem: asyncio.Semaphore,
                           max_retries: int, base_backoff: float) -> LabelResult:
    messages, kind, meta = _build_messages(inp)
    last_err = "no attempt"
    backoff = base_backoff
    for attempt in range(max_retries):
        async with sem:
            t0 = time.time()
            try:
                resp = await client.chat.completions.create(
                    **_completion_kwargs(model, messages),
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                return LabelResult(
                    example_id=inp.example_id, description=text, model=model,
                    elapsed_ms=(time.time() - t0) * 1000, usage=usage,
                    kind=kind, meta=meta,
                )
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.warning(
                    "label_one_async %s attempt %d failed: %s",
                    inp.example_id, attempt + 1, last_err,
                )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
    return LabelResult(
        example_id=inp.example_id, description="", model=model,
        elapsed_ms=0.0, usage={}, error=last_err, kind=kind, meta=meta,
    )


def _position_resume_key_from_row(obj: dict) -> tuple[str, int, str] | None:
    """Match (source_example_id, position_index, position_type) for position labels."""
    if obj.get("kind") != "position":
        return None
    m = obj.get("meta") or {}
    sid = m.get("source_example_id")
    pidx = m.get("position_index")
    pt = m.get("position_type")
    if sid is None or pidx is None or pt is None:
        return None
    return (str(sid), int(pidx), str(pt))


def _position_resume_key_from_input(inp) -> tuple[str, int, str] | None:
    if not isinstance(inp, PositionLabelInput):
        return None
    sid = inp.extra.get("source_example_id")
    if sid is None:
        return None
    return (str(sid), int(inp.position_index), str(inp.position_type))


async def label_many_async(
    inputs: Iterable,
    output_jsonl: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 16,
    api_key: str | None = None,
    resume: bool = True,
    max_retries: int = 4,
    base_backoff: float = 1.0,
    progress_every: int = 25,
) -> int:
    """Run labeling concurrently, streaming JSONL with resume.

    Resume skips when either ``example_id`` was already written successfully *or*
    the canonical position key ``(source_example_id, position_index, position_type)``
    is present (prevents duplicate / conflicting rows if ``example_id`` drifted).
    """
    _, AsyncOpenAI = _get_openai()
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    done_pos_keys: set[tuple[str, int, str]] = set()
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
                    if obj.get("kind") != "step":
                        pk = _position_resume_key_from_row(obj)
                        if pk is not None:
                            done_pos_keys.add(pk)

    todo = []
    for i in inputs:
        if i.example_id in done_ids:
            continue
        if not isinstance(i, StepLabelInput):
            pk = _position_resume_key_from_input(i)
            if pk is not None and pk in done_pos_keys:
                continue
        todo.append(i)
    logger.info(
        "Labeling: %d new, %d example_ids done, %d position keys done -> %s",
        len(todo), len(done_ids), len(done_pos_keys), output_jsonl,
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
            res = await _label_one_async(client, inp, model, sem, max_retries, base_backoff)
            row = asdict(res)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % progress_every == 0:
                logger.info("  %d / %d labeled", n_new, len(todo))
            return res

        await asyncio.gather(*(run_one(i) for i in todo))
    finally:
        f.close()
        await client.close()

    return n_new

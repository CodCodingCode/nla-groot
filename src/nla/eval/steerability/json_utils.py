"""JSON helpers for rollout subprocess stdout (strict JSON, no NaN/Inf)."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, str):
        return obj
    return obj


def dumps_rollout_json(obj: Any, *, indent: int | None = 2) -> str:
    """Serialize rollout summaries for ``stdout`` (valid strict JSON)."""
    return json.dumps(_sanitize(obj), indent=indent, allow_nan=False)


def extract_rollout_json(text: str, *, expect_array: bool = False) -> Any:
    """Parse JSON from rollout subprocess stdout (tolerates leading LIBERO logs).

    LIBERO prints lines like ``[info] using task orders [0, 1, 2, 3]`` which
    must not be confused with the JSON payload. The summary is pretty-printed
    (``json.dumps(indent=2)``), so a naive ``rfind('{')`` lands on an *inner*
    brace of a nested object and the loader chokes on the outer closer with
    ``Extra data: line N column 1``. ``[info]`` lines also embed a literal
    bracketed array that ``find('[')`` would happily parse as a 4-element
    list of ints, hiding the real summary entirely.

    Strategy: scan all lines whose first non-whitespace char matches the
    expected opening token, dropping lines that start with ``[info]``. The
    LAST such candidate is the rollout summary's opening line; everything
    from there onward is fed to :meth:`json.JSONDecoder.raw_decode` so we
    consume exactly one outermost JSON value.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty rollout stdout")

    target = "[" if expect_array else "{"
    decoder = json.JSONDecoder()
    lines = text.splitlines(keepends=True)

    # Compute byte offset of each line in ``text`` so we can slice from the
    # candidate line. Pretty-printed JSON's opening line is by definition
    # column 0 (the token alone), so this scan is exact.
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    candidate_indices: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(target):
            continue
        if stripped.startswith("[info]"):
            continue
        candidate_indices.append(i)

    last_err: json.JSONDecodeError | None = None
    # Try the latest candidate first (rollout summary is last on stdout),
    # then fall back to earlier candidates if it fails.
    for i in reversed(candidate_indices):
        chunk = text[offsets[i]:]
        try:
            obj, _end = decoder.raw_decode(chunk)
            return obj
        except json.JSONDecodeError as e:
            last_err = e
            continue

    kind = "array" if expect_array else "object"
    if last_err is not None:
        raise json.JSONDecodeError(
            f"no parseable JSON {kind} in rollout stdout "
            f"(last error: {last_err.msg} at pos {last_err.pos})",
            text, last_err.pos,
        )
    raise json.JSONDecodeError(
        f"no JSON {kind} found in rollout stdout", text, 0,
    )

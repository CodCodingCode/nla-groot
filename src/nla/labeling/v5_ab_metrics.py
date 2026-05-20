"""V5 nested-label A/B metrics (schema validity, slot granularity, template collapse)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nla.labeling.prompts_v5 import parse_v5_response
from nla.labeling.schema_v5 import (
    SLOT_NAMES,
    cross_slot_jaccard,
    validate_nested,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 2]


def _plan_phrases(nested: dict[str, dict[str, str]]) -> list[str]:
    out: list[str] = []
    for slot in SLOT_NAMES:
        plan = str(nested.get(slot, {}).get("plan", "")).strip()
        if not plan or plan.upper() == "NA":
            continue
        phase = plan.split(":", 1)[0].strip().lower()
        if phase:
            out.append(phase)
        out.append(plan.lower()[:80])
    return out


def score_v5_label_row(raw_response: str, description: str | None = None) -> dict[str, Any]:
    """Score one labeled row for V5 nested JSON quality."""
    nested = None
    parse_err: str | None = None
    for blob in (raw_response, description or ""):
        if not blob or not str(blob).strip():
            continue
        try:
            nested = parse_v5_response(str(blob))
            break
        except Exception as e:
            parse_err = str(e)

    if nested is None:
        return {
            "valid_schema": False,
            "cross_slot_jaccard_mean": None,
            "distinct_scene_target_tokens": 0,
            "top_phrase_df": [],
            "errors": [parse_err or "empty response"],
        }

    ok, errors, norm = validate_nested(nested)
    jac_mean = None
    distinct = 0
    top_phrases: list[tuple[str, int]] = []

    if ok and norm is not None:
        jac = cross_slot_jaccard(norm)
        jac_mean = jac.get("mean")
        tokens: set[str] = set()
        for slot in SLOT_NAMES:
            for key in ("scene", "target"):
                tokens.update(_tokenize(str(norm[slot].get(key, ""))))
        distinct = len(tokens)
        phrase_counts = Counter(_plan_phrases(norm))
        top_phrases = phrase_counts.most_common(10)

    return {
        "valid_schema": ok,
        "cross_slot_jaccard_mean": jac_mean,
        "distinct_scene_target_tokens": distinct,
        "top_phrase_df": top_phrases,
        "errors": errors if not ok else [],
    }


def _load_label_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def aggregate_v5_ab(labels_jsonl: dict[str, str | Path]) -> dict[str, Any]:
    """Aggregate V5 metrics across variant label files.

    Parameters
    ----------
    labels_jsonl:
        ``{variant_id: path}`` mapping to each variant's ``labels.jsonl``.
    """
    out: dict[str, Any] = {}
    for variant_id, path in labels_jsonl.items():
        path = Path(path)
        rows = _load_label_rows(path)
        n = len(rows)
        valid = 0
        jac_vals: list[float] = []
        distinct_vals: list[int] = []
        phrase_global: Counter[str] = Counter()
        n_errors = 0

        for row in rows:
            if row.get("error"):
                n_errors += 1
                continue
            raw = row.get("raw_response") or row.get("description") or ""
            desc = row.get("description")
            scores = score_v5_label_row(raw, desc)
            if scores["valid_schema"]:
                valid += 1
            if scores["cross_slot_jaccard_mean"] is not None:
                jac_vals.append(float(scores["cross_slot_jaccard_mean"]))
            distinct_vals.append(int(scores["distinct_scene_target_tokens"]))
            for phrase, cnt in scores["top_phrase_df"]:
                phrase_global[phrase] += cnt

        valid_rate = (valid / n) if n else 0.0
        mean_jac = (sum(jac_vals) / len(jac_vals)) if jac_vals else None
        mean_distinct = (sum(distinct_vals) / len(distinct_vals)) if distinct_vals else 0.0
        top_phrase_df = phrase_global.most_common(15)

        out[variant_id] = {
            "n_rows": n,
            "n_errors": n_errors,
            "valid_schema_rate": valid_rate,
            "cross_slot_jaccard_mean": mean_jac,
            "mean_distinct_scene_target_tokens": mean_distinct,
            "top_phrase_df": top_phrase_df,
            "granularity": {
                "valid_schema_rate": valid_rate,
                "cross_slot_jaccard_mean": mean_jac,
                "mean_distinct_scene_target_tokens": mean_distinct,
                "template_collapse_top_phrases": top_phrase_df[:5],
            },
        }
    return out


__all__ = ["score_v5_label_row", "aggregate_v5_ab"]

"""Tests for ``scripts/eval/compare_judge_runs.py``.

Covers:
- aggregation by (variant, position_type) with synthetic counts
- markdown table rendering: percentages, n, missing-cell N/A
- delta computation against the first run
- example_id-based position_type fallback when the row omits the field
- end-to-end CLI run that writes a comparison markdown file
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "eval" / "compare_judge_runs.py"
    spec = importlib.util.spec_from_file_location("compare_judge_runs", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _judge_row(
    *,
    example_id: str,
    variant_id: str,
    grounding: str | None,
    appropriateness: str | None,
    position_type: str | None = None,
) -> dict:
    row: dict = {
        "example_id": example_id,
        "variant_id": variant_id,
        "grader": "gpt-5.1",
        "model": "gpt-5.1",
        "grounding": (
            None
            if grounding is None
            else {"verdict": grounding, "reason": "x", "passed": grounding == "specific"}
        ),
        "appropriateness": (
            None
            if appropriateness is None
            else {
                "verdict": appropriateness,
                "reason": "x",
                "passed": appropriateness == "appropriate",
            }
        ),
        "elapsed_ms": 1.0,
        "usage": {},
        "error": None,
        "raw_response": "{}",
    }
    if position_type is not None:
        row["position_type"] = position_type
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_basic_counts(mod):
    rows = [
        # gold/image_patch: 4 total, 3 specific, 4 appropriate
        _judge_row(example_id="t1@p10_image_patch", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
        _judge_row(example_id="t2@p11_image_patch", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
        _judge_row(example_id="t3@p12_image_patch", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
        _judge_row(example_id="t4@p13_image_patch", variant_id="gold",
                   grounding="generic", appropriateness="appropriate"),
        # av_pred/last_text: 2 total, 0 specific, 1 appropriate
        _judge_row(example_id="t5@p20_last_text", variant_id="av_pred",
                   grounding="generic", appropriateness="appropriate"),
        _judge_row(example_id="t6@p21_last_text", variant_id="av_pred",
                   grounding="generic", appropriateness="inappropriate"),
    ]
    agg = mod.aggregate(rows)
    assert agg[("gold", "image_patch")]["n"] == 4
    assert agg[("gold", "image_patch")]["b_specific_pct"] == pytest.approx(75.0)
    assert agg[("gold", "image_patch")]["c_appropriate_pct"] == pytest.approx(100.0)
    assert agg[("av_pred", "last_text")]["n"] == 2
    assert agg[("av_pred", "last_text")]["b_specific_pct"] == pytest.approx(0.0)
    assert agg[("av_pred", "last_text")]["c_appropriate_pct"] == pytest.approx(50.0)


def test_aggregate_position_type_from_example_id(mod):
    """Rows that omit ``position_type`` fall back to parsing the example_id."""
    rows = [
        _judge_row(example_id="trajX_step000@p7_anchor", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
    ]
    agg = mod.aggregate(rows)
    assert ("gold", "anchor") in agg
    assert agg[("gold", "anchor")]["n"] == 1


def test_aggregate_skips_ungraded(mod):
    rows = [
        _judge_row(example_id="t1@p1_anchor", variant_id="gold",
                   grounding=None, appropriateness=None),
    ]
    agg = mod.aggregate(rows)
    assert agg == {}


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_table_and_deltas(mod, tmp_path):
    run1 = tmp_path / "run1.jsonl"
    run2 = tmp_path / "run2.jsonl"
    _write_jsonl(run1, [
        _judge_row(example_id=f"t{i}@p{i}_image_patch", variant_id="gold",
                   grounding="specific" if i < 8 else "generic",
                   appropriateness="appropriate")
        for i in range(10)
    ])
    _write_jsonl(run2, [
        _judge_row(example_id=f"t{i}@p{i}_image_patch", variant_id="gold",
                   grounding="specific" if i < 9 else "generic",
                   appropriateness="appropriate")
        for i in range(10)
    ])

    rows1 = mod.load_judge_jsonl(run1)
    rows2 = mod.load_judge_jsonl(run2)
    aggs = [mod.aggregate(rows1), mod.aggregate(rows2)]
    md = mod.render_markdown([("V2", run1), ("cleaned", run2)], aggs)

    assert "| V2 | gold | image_patch | 10 | 80.0 | 100.0 |" in md
    assert "| cleaned | gold | image_patch | 10 | 90.0 | 100.0 |" in md
    assert "## Deltas vs first run" in md
    assert "| cleaned vs V2 | gold | image_patch | +10.0 | +0.0 |" in md


def test_render_markdown_missing_cell_is_NA(mod, tmp_path):
    run1 = tmp_path / "run1.jsonl"
    run2 = tmp_path / "run2.jsonl"
    _write_jsonl(run1, [
        _judge_row(example_id="t1@p1_image_patch", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
        _judge_row(example_id="t2@p2_anchor", variant_id="gold",
                   grounding="generic", appropriateness="appropriate"),
    ])
    _write_jsonl(run2, [
        _judge_row(example_id="t1@p1_image_patch", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
    ])

    aggs = [mod.aggregate(mod.load_judge_jsonl(run1)),
            mod.aggregate(mod.load_judge_jsonl(run2))]
    md = mod.render_markdown([("V2", run1), ("cleaned", run2)], aggs)

    assert "| cleaned | gold | anchor | 0 | N/A | N/A |" in md
    assert "| cleaned vs V2 | gold | anchor | N/A | N/A |" in md


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_runs_and_writes_md(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    run1 = tmp_path / "v2.jsonl"
    run2 = tmp_path / "cleaned.jsonl"
    _write_jsonl(run1, [
        _judge_row(example_id="t1@p1_image_patch", variant_id="gold",
                   grounding="generic", appropriateness="appropriate"),
        _judge_row(example_id="t2@p2_image_patch", variant_id="gold",
                   grounding="specific", appropriateness="appropriate"),
    ])
    _write_jsonl(run2, [
        _judge_row(example_id="t1@p1_image_patch", variant_id="gold_cleaned",
                   grounding="specific", appropriateness="appropriate"),
        _judge_row(example_id="t2@p2_image_patch", variant_id="gold_cleaned",
                   grounding="specific", appropriateness="appropriate"),
    ])

    out_md = tmp_path / "compare.md"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "eval" / "compare_judge_runs.py"),
        "--runs",
        f"V2={run1}",
        f"cleaned={run2}",
        "--out-md",
        str(out_md),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Aggregate verdicts" in res.stdout
    assert out_md.exists()
    text = out_md.read_text()
    assert "| V2 | gold | image_patch | 2 | 50.0 | 100.0 |" in text
    assert "| cleaned | gold_cleaned | image_patch | 2 | 100.0 | 100.0 |" in text


def test_cli_help_works():
    repo = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo / "scripts" / "eval" / "compare_judge_runs.py"),
        "--help",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "--runs" in res.stdout
    assert "--out-md" in res.stdout

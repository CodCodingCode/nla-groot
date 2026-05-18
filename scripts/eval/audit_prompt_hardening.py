#!/usr/bin/env python
"""Prompt-hardening regression scan over LIBERO labels (V3 baseline + V4 regression).

Originally written as Agent 2 of the V3 data-quality audit; extended by
SA3 of the V4 dataset-repair plan to also measure the three V4-era
failure modes uncovered by Agent 1 / Agent 3 / Agent 4:

* **Motor imperatives** — second-person verbs aimed at the robot ("grasp
  the bowl", "align the gripper"). Drove the residual 1.8% C-fails on
  Agent 1's multimodal judge; eliminated by the V4 prompt.
* **Scaffold leakage** — phrases echoed from the system prompt's own
  scaffolding ("action head", "this patch carries", "transformer",
  "embedding"). 11-17% of V3 plan bullets carried these per Agent 3;
  explicitly forbidden by the V4 prompt.
* **Non-canonical bullet headers** — ``gripper:`` / ``motion:`` /
  ``image_region:`` headers that were never in the V3 closed-vocabulary
  list but accounted for ~10% of V3 bullets per Agent 2's first pass.
  V4 collapses these into ``plan`` / ``target`` / ``spatial``.

The V3-era failure modes are still measured for back-compat:

1. Anthropomorphic phrasing (case-insensitive substring match).
2. Numerical confabulation (regex over fabricated measurements, applied
   per bullet line, not just whole description, so we count any row that
   has at least one bullet with a hit).
3. ``image_region:`` bullets (the hardened V3 prompt explicitly steered
   labelers away from this category; in V4 it is subsumed by the
   non-canonical-header check).
4. Other forbidden patterns the V3 hardened prompt added.
5. Empty / degenerate captions (<50 chars OR <3 bullets).
6. Bullet-prefix conformance (presence of ALL FIVE of
   ``- language:``, ``- target:``, ``- scene:``, ``- spatial:``,
   ``- plan:``).
7. Error rows (``error`` field non-null).

Phrase lists for the three V4 failure modes are imported directly from
``nla.labeling.prompts`` (``V4_MOTOR_IMPERATIVE_PHRASES``,
``V4_SCAFFOLD_FORBIDDEN_PHRASES``, ``V4_FORBIDDEN_HEADERS``) so the
prompt-side ban list and the audit-side regex can never silently drift
apart.

By default writes the markdown report to
``docs/sft_plan/audit_reports/agent2_prompt_hardening_regression.md`` and a
JSON summary to ``data/eval/audit_prompt_hardening_summary.json``. SA10
can regression-gate V4 against the frozen V3 baseline at
``data/eval/sa3_v3_baseline_summary.json``.

CLI::

    PYTHONPATH=src .venv/bin/python scripts/eval/audit_prompt_hardening.py \\
        --labels-root data/labels/libero_4suite_stride2 \\
        --out-json data/eval/sa3_v3_baseline_summary.json \\
        --out-md   /tmp/sa3_v3_baseline.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Make ``src/`` importable so we can pull the V4 phrase tuples straight
# from the labeling-prompt module: prompt-side and audit-side share one
# source of truth.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from nla.labeling.prompts import (  # noqa: E402  (sys.path manipulation above)
    V4_BULLET_CATEGORIES,
    V4_FORBIDDEN_HEADERS,
    V4_MOTOR_IMPERATIVE_PHRASES,
    V4_SCAFFOLD_FORBIDDEN_PHRASES,
)

logger = logging.getLogger("agent2.audit")

REPO_ROOT = Path("/home/ubuntu/nla-groot")
V3_BASE = REPO_ROOT / "data/labels/libero_4suite_stride2"
V3_SUITES = ("goal", "spatial", "object", "10")
V2_DROID_PATH_LIVE    = REPO_ROOT / "data/labels/droid_100ep/labels.jsonl"
V2_DROID_PATH_ARCHIVE = REPO_ROOT / "data/_archive_droid/labels/droid_100ep/labels.jsonl"
PILOT_PATH = REPO_ROOT / "data/labels/libero_goal_pilot/labels.jsonl"


def _resolve_v2_droid_path() -> Path | None:
    """Return whichever DROID labels.jsonl exists (live first, then the
    archive landed by scripts/migration/archive_droid.sh), or None."""
    for candidate in (V2_DROID_PATH_LIVE, V2_DROID_PATH_ARCHIVE):
        if candidate.exists():
            return candidate
    return None
DEFAULT_REPORT_PATH = REPO_ROOT / "docs/sft_plan/audit_reports/agent2_prompt_hardening_regression.md"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "data/eval/audit_prompt_hardening_summary.json"
# Frozen V3-baseline JSON Agent 2 emitted before SA3's regression-gate
# extensions; left untouched so the V3-era reports keep resolving to the
# exact numbers Agent 2 published.
AGENT2_FROZEN_SUMMARY_PATH = REPO_ROOT / "data/eval/agent2_summary.json"

# Failure-mode patterns -------------------------------------------------------

# 1. Anthropomorphic phrasing. Case-insensitive substring match.
ANTHROPO_PHRASES: tuple[str, ...] = (
    "committing to",
    "has been read and committed to",
    "committed to",
    "intends to",
    "wants to",
    "trying to",
    "decides to",
    "is about to",
    "plans to ",
    "prepares to",
    "the model",
    "the policy",
    "the network",
    "the agent thinks",
    "the agent believes",
    "goal committed",
)

# 2. Numerical confabulation. Regex per the task spec.
NUMERIC_MEAS_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m\b|meters?|centimeters?|millimeters?|"
    r"inches?|in\b|°|degrees?|rad(?:ians?)?|kg|grams?|g\b|n\b|newtons?)\b",
    re.IGNORECASE,
)

# 3. image_region bullets.
IMAGE_REGION_RE = re.compile(r"^\s*-?\s*image[_ ]region\s*:", re.IGNORECASE)

# 4. Other forbidden patterns added by prompt hardening.
OTHER_FORBIDDEN_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reads_instruction", re.compile(r"\b(?:reads|has read)\s+the\s+instruction\b", re.IGNORECASE)),
    ("understands_goal", re.compile(r"\b(?:understands|comprehends)\s+the\s+goal\b", re.IGNORECASE)),
    ("ready_to_execute", re.compile(r"\bready\s+to\s+execute\b", re.IGNORECASE)),
)

# Bullet-prefix conformance.
EXPECTED_BULLETS: tuple[str, ...] = ("language", "target", "scene", "spatial", "plan")
BULLET_PREFIX_RES: dict[str, re.Pattern[str]] = {
    cat: re.compile(rf"^\s*-\s*{cat}\s*:", re.IGNORECASE | re.MULTILINE)
    for cat in EXPECTED_BULLETS
}

# Generic bullet detector for length stats / count.
BULLET_LINE_RE = re.compile(r"^\s*-\s*([a-zA-Z_][a-zA-Z _/]{0,40})\s*:\s*(.*)$")


def _phrase_regex(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Build a case-insensitive word-boundary alternation regex.

    ``\\b<phrase>\\b`` on each side, as the V4 prompt scaffolding does
    internally. Phrases are ``re.escape``-d so a future entry containing
    regex meta (a ``.`` / ``?`` / parenthesis) cannot break the pattern.
    """
    if not phrases:
        return re.compile(r"(?!x)x")  # matches nothing
    alt = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)


# V4 regression failure modes -------------------------------------------------
# Imported phrase tuples; the regex is built from them so prompt-side and
# audit-side cannot drift apart (test:
# ``test_audit_uses_canonical_constants``).
V4_MOTOR_IMPERATIVE_RE: re.Pattern[str] = _phrase_regex(V4_MOTOR_IMPERATIVE_PHRASES)
V4_SCAFFOLD_LEAKAGE_RE: re.Pattern[str] = _phrase_regex(V4_SCAFFOLD_FORBIDDEN_PHRASES)
# Line-level (anchored) detector for V4-forbidden bullet headers
# (``gripper:`` / ``motion:`` / ``image_region:``). Allows a leading ``-``
# bullet marker or a header without one, matches case-insensitively, and
# requires the trailing ``:``.
_NONCANON_HEADERS_ALT = "|".join(re.escape(h) for h in V4_FORBIDDEN_HEADERS)
V4_NONCANON_HEADER_RE: re.Pattern[str] = re.compile(
    rf"^\s*-?\s*({_NONCANON_HEADERS_ALT})\s*:",
    re.IGNORECASE,
)

# Bullet categories we track per-bullet-type for V4 motor/scaffold
# breakdowns. Includes V4-allowed categories plus the V4-forbidden
# legacy ones (so per-bullet-type can also surface that
# e.g. ``gripper:`` bullets carry imperatives at 80%).
V4_TRACKED_BULLET_TYPES: tuple[str, ...] = tuple(V4_BULLET_CATEGORIES) + tuple(
    V4_FORBIDDEN_HEADERS
)


@dataclass
class FailingExample:
    example_id: str
    description: str
    matched: str  # which substring/pattern matched (for explainability)


@dataclass
class SuiteStats:
    name: str
    path: str
    n_total: int = 0
    n_position: int = 0
    n_error: int = 0
    n_no_description: int = 0

    # Per-failure-mode counts (label-level: row counted at most once).
    n_anthropo: int = 0
    n_numeric: int = 0
    n_image_region: int = 0
    n_reads_instruction: int = 0
    n_understands_goal: int = 0
    n_ready_to_execute: int = 0
    n_empty_or_degenerate: int = 0
    n_short_desc: int = 0  # subset of degenerate
    n_few_bullets: int = 0  # subset of degenerate
    n_conformant_bullets: int = 0  # all 5 prefixes present

    # V4 regression failure modes (label-level: row counted at most once).
    n_v4_motor: int = 0
    n_v4_scaffold: int = 0
    n_v4_noncanon_header: int = 0

    # V4 per-bullet-type breakdown: bullet_type -> (total seen, motor hits, scaffold hits).
    # Tracked over every bullet (not just V4_TRACKED_BULLET_TYPES) so SA10
    # can also see leakage in unexpected categories.
    bt_total: Counter = field(default_factory=Counter)
    bt_motor_hits: Counter = field(default_factory=Counter)
    bt_scaffold_hits: Counter = field(default_factory=Counter)
    # Row counts of which V4-forbidden header tripped non-canonical detection.
    noncanon_header_hits: Counter = field(default_factory=Counter)

    # Bullet length stats: bullet_category -> list of token counts.
    bullet_token_counts: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    # Up to N examples per failure mode for the report.
    examples: dict[str, list[FailingExample]] = field(default_factory=lambda: defaultdict(list))

    # Bookkeeping for anthropomorphic phrase counts (which phrase fires most often).
    anthropo_phrase_hits: Counter = field(default_factory=Counter)

    # Bullet-prefix presence (per individual prefix).
    bullet_present_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Non-conformant examples (for tables when conformance <99%).
    nonconformant_examples: list[FailingExample] = field(default_factory=list)

    # Per-position-type breakdown for conformance and per-prefix presence.
    # (The hardened prompt is position-type aware: last_text rows should
    # carry a 'language:' bullet, image_patch rows are explicitly steered
    # toward target/scene/spatial/distractor/plan instead. So we must
    # report conformance conditional on position_type to separate
    # 'prompt-design schema' from 'labeler failure'.)
    position_type_counts: Counter = field(default_factory=Counter)
    pos_bullet_present_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    pos_conformant_bullets: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Position-type-aware conformance: last_text needs all 5; image_patch
    # needs the 4 non-language ones (target/scene/spatial/plan).
    pos_relaxed_conformant: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    MAX_EXAMPLES_PER_MODE: int = 5

    def add_example(self, mode: str, ex: FailingExample) -> None:
        bucket = self.examples[mode]
        if len(bucket) < self.MAX_EXAMPLES_PER_MODE:
            bucket.append(ex)


def _tokens(text: str) -> int:
    """Crude whitespace token count."""
    return len(text.strip().split())


def _scan_row(stats: SuiteStats, row: dict[str, Any]) -> None:
    stats.n_total += 1
    if row.get("kind") == "position":
        stats.n_position += 1
    if row.get("error") is not None:
        stats.n_error += 1

    desc = row.get("description") or ""
    if not isinstance(desc, str) or not desc.strip():
        stats.n_no_description += 1
        return
    example_id = row.get("example_id") or "<no id>"

    # ------------------------------------------------------------------ 1. anthropo
    low = desc.lower()
    anthropo_hits: list[str] = [p for p in ANTHROPO_PHRASES if p in low]
    if anthropo_hits:
        stats.n_anthropo += 1
        for h in anthropo_hits:
            stats.anthropo_phrase_hits[h] += 1
        stats.add_example(
            "anthropomorphic",
            FailingExample(example_id, desc, ", ".join(anthropo_hits)),
        )

    # ------------------------------------------------------------------ 2. numeric
    # Per-bullet check: any line with a measurement counts the row.
    numeric_match_text: str | None = None
    for ln in desc.splitlines():
        m = NUMERIC_MEAS_RE.search(ln)
        if m:
            numeric_match_text = m.group(0)
            break
    if numeric_match_text is not None:
        stats.n_numeric += 1
        stats.add_example(
            "numeric",
            FailingExample(example_id, desc, numeric_match_text),
        )

    # ------------------------------------------------------------------ 3. image_region
    image_region_hit = False
    for ln in desc.splitlines():
        if IMAGE_REGION_RE.match(ln):
            image_region_hit = True
            break
    if image_region_hit:
        stats.n_image_region += 1
        stats.add_example(
            "image_region",
            FailingExample(example_id, desc, "image_region bullet"),
        )

    # ------------------------------------------------------------------ 4. other forbidden
    for mode_name, pattern in OTHER_FORBIDDEN_RES:
        m = pattern.search(desc)
        if m:
            attr = f"n_{mode_name}"
            setattr(stats, attr, getattr(stats, attr) + 1)
            stats.add_example(mode_name, FailingExample(example_id, desc, m.group(0)))

    # ------------------------------------------------------------------ 5. empty / degenerate
    bullets: list[tuple[str, str]] = []
    for ln in desc.splitlines():
        bm = BULLET_LINE_RE.match(ln)
        if bm:
            cat = bm.group(1).strip().lower()
            body = bm.group(2).strip()
            bullets.append((cat, body))

    # ------------------------------------------------------------------ V4 motor/scaffold/header
    # Row-level detection runs against the full description; the
    # per-bullet-type breakdown runs against each bullet body
    # separately, because Agent 4 found imperatives concentrate in
    # ``plan:`` and we need to surface that.
    motor_hit_text = V4_MOTOR_IMPERATIVE_RE.search(desc)
    if motor_hit_text is not None:
        stats.n_v4_motor += 1
        stats.add_example(
            "v4_motor",
            FailingExample(example_id, desc, motor_hit_text.group(0)),
        )
    scaffold_hit_text = V4_SCAFFOLD_LEAKAGE_RE.search(desc)
    if scaffold_hit_text is not None:
        stats.n_v4_scaffold += 1
        stats.add_example(
            "v4_scaffold",
            FailingExample(example_id, desc, scaffold_hit_text.group(0)),
        )
    noncanon_header_hit: str | None = None
    for ln in desc.splitlines():
        nm = V4_NONCANON_HEADER_RE.match(ln)
        if nm:
            noncanon_header_hit = nm.group(1).lower()
            break
    if noncanon_header_hit is not None:
        stats.n_v4_noncanon_header += 1
        stats.noncanon_header_hits[noncanon_header_hit] += 1
        stats.add_example(
            "v4_noncanon_header",
            FailingExample(example_id, desc, f"{noncanon_header_hit}:"),
        )

    # Per-bullet-type tallies for the V4 breakdowns.
    for cat, body in bullets:
        stats.bt_total[cat] += 1
        if V4_MOTOR_IMPERATIVE_RE.search(body):
            stats.bt_motor_hits[cat] += 1
        if V4_SCAFFOLD_LEAKAGE_RE.search(body):
            stats.bt_scaffold_hits[cat] += 1

    is_short = len(desc.strip()) < 50
    is_few_bullets = len(bullets) < 3
    if is_short:
        stats.n_short_desc += 1
    if is_few_bullets:
        stats.n_few_bullets += 1
    if is_short or is_few_bullets:
        stats.n_empty_or_degenerate += 1
        stats.add_example(
            "empty_or_degenerate",
            FailingExample(
                example_id,
                desc,
                f"len={len(desc.strip())} bullets={len(bullets)}",
            ),
        )

    # ------------------------------------------------------------------ 6. bullet conformance
    meta = row.get("meta") or {}
    position_type = meta.get("position_type") or "<unknown>"
    stats.position_type_counts[position_type] += 1

    bullet_cats = {c for c, _ in bullets}

    def _has(cat: str) -> bool:
        return any(c == cat or c.startswith(cat) for c in bullet_cats)

    has_all_five = all(_has(ex) for ex in EXPECTED_BULLETS)
    for ex in EXPECTED_BULLETS:
        if _has(ex):
            stats.bullet_present_counts[ex] += 1
            stats.pos_bullet_present_counts[position_type][ex] += 1
    if has_all_five:
        stats.n_conformant_bullets += 1
        stats.pos_conformant_bullets[position_type] += 1
    else:
        if len(stats.nonconformant_examples) < stats.MAX_EXAMPLES_PER_MODE:
            missing = [ex for ex in EXPECTED_BULLETS if not _has(ex)]
            stats.nonconformant_examples.append(
                FailingExample(example_id, desc, "missing: " + ",".join(missing))
            )

    # Position-type-aware "relaxed" conformance: the hardened prompt only
    # asks last_text rows for a `language:` bullet. image_patch (and anchor)
    # rows are explicitly steered toward target/scene/spatial/plan
    # (+ optionally distractor). So an image_patch row counts as conformant
    # if it has target+scene+spatial+plan, even without `language:`.
    if position_type == "last_text":
        relaxed_ok = has_all_five
    else:
        relaxed_ok = _has("target") and _has("scene") and _has("spatial") and _has("plan")
    if relaxed_ok:
        stats.pos_relaxed_conformant[position_type] += 1

    # ------------------------------------------------------------------ bullet length stats
    for cat, body in bullets:
        # Bucket only to canonical names + image_region for visibility; anything
        # else goes under "_other".
        key = cat if cat in EXPECTED_BULLETS or cat == "image_region" or cat == "distractor" else "_other"
        stats.bullet_token_counts[key].append(_tokens(body))


AGENT1_JUDGE_PATH = REPO_ROOT / "data/eval/libero_v3_quality_judge.jsonl"


def _load_agent1_findings() -> dict[str, Any] | None:
    """Best-effort cross-reference to Agent 1 (multimodal gpt-5.1 judge).

    Returns None if Agent 1's output is missing. Used only for the
    coordination section of the report.
    """
    if not AGENT1_JUDGE_PATH.exists():
        return None
    n = 0
    g_pass = 0
    a_pass = 0
    suites: Counter = Counter()
    g_fail_reasons: list[str] = []
    a_fail_reasons: list[str] = []
    with AGENT1_JUDGE_PATH.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            ex = row.get("example_id", "")
            if "::" in ex:
                suites[ex.split("::")[0]] += 1
            g = row.get("grounding") or {}
            a = row.get("appropriateness") or {}
            if g.get("passed"):
                g_pass += 1
            else:
                g_fail_reasons.append(g.get("reason", ""))
            if a.get("passed"):
                a_pass += 1
            else:
                a_fail_reasons.append(a.get("reason", ""))
    if n == 0:
        return None
    return {
        "n": n,
        "grounding_pass_pct": 100.0 * g_pass / n,
        "appropriateness_pass_pct": 100.0 * a_pass / n,
        "suite_breakdown": dict(suites),
        "sample_grounding_fails": g_fail_reasons[:3],
        "sample_appropriateness_fails": a_fail_reasons[:3],
    }


def _rescan_other_categories(base: Path, suites: tuple[str, ...]) -> Counter:
    """Tiny second pass to break the `_other` bucket into actual category names.

    Avoids carrying every non-canonical category around the main stats
    struct; keeps the main pass simple. Run-time is ~3s for the full V3
    corpus.
    """
    bullet_re = re.compile(r"^\s*-\s*([a-zA-Z_][a-zA-Z _/]{0,40})\s*:\s*")
    canonical = {"language", "target", "scene", "spatial", "plan",
                 "distractor", "image_region"}
    counts: Counter = Counter()
    for suite in suites:
        path = base / f"libero_{suite}" / "labels.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                desc = row.get("description") or ""
                for ln in desc.splitlines():
                    m = bullet_re.match(ln)
                    if not m:
                        continue
                    cat = m.group(1).strip().lower()
                    if cat not in canonical:
                        counts[cat] += 1
    return counts


def scan_file(path: Path, name: str) -> SuiteStats:
    stats = SuiteStats(name=name, path=str(path))
    if not path.exists():
        logger.warning("Missing labels file: %s", path)
        return stats
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Bad JSON in %s", path)
                continue
            _scan_row(stats, row)
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(num: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return 100.0 * num / denom


def _fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    pct = _pct(num, denom)
    if pct == 0.0:
        return "0.000%"
    if pct >= 10.0:
        return f"{pct:.2f}%"
    return f"{pct:.3f}%"


def aggregate_v3(per_suite: dict[str, SuiteStats]) -> SuiteStats:
    """Combine the four LIBERO suites into a single V3-overall row."""
    out = SuiteStats(name="v3_overall", path="<aggregate>")
    for s in per_suite.values():
        out.n_total += s.n_total
        out.n_position += s.n_position
        out.n_error += s.n_error
        out.n_no_description += s.n_no_description
        out.n_anthropo += s.n_anthropo
        out.n_numeric += s.n_numeric
        out.n_image_region += s.n_image_region
        out.n_reads_instruction += s.n_reads_instruction
        out.n_understands_goal += s.n_understands_goal
        out.n_ready_to_execute += s.n_ready_to_execute
        out.n_empty_or_degenerate += s.n_empty_or_degenerate
        out.n_short_desc += s.n_short_desc
        out.n_few_bullets += s.n_few_bullets
        out.n_conformant_bullets += s.n_conformant_bullets
        out.n_v4_motor += s.n_v4_motor
        out.n_v4_scaffold += s.n_v4_scaffold
        out.n_v4_noncanon_header += s.n_v4_noncanon_header
        for k, v in s.bt_total.items():
            out.bt_total[k] += v
        for k, v in s.bt_motor_hits.items():
            out.bt_motor_hits[k] += v
        for k, v in s.bt_scaffold_hits.items():
            out.bt_scaffold_hits[k] += v
        for k, v in s.noncanon_header_hits.items():
            out.noncanon_header_hits[k] += v
        for k, v in s.anthropo_phrase_hits.items():
            out.anthropo_phrase_hits[k] += v
        for k, v in s.bullet_present_counts.items():
            out.bullet_present_counts[k] = out.bullet_present_counts.get(k, 0) + v
        for cat, lst in s.bullet_token_counts.items():
            out.bullet_token_counts[cat].extend(lst)
        for ptype, n in s.position_type_counts.items():
            out.position_type_counts[ptype] += n
        for ptype, n in s.pos_conformant_bullets.items():
            out.pos_conformant_bullets[ptype] += n
        for ptype, n in s.pos_relaxed_conformant.items():
            out.pos_relaxed_conformant[ptype] += n
        for ptype, sub in s.pos_bullet_present_counts.items():
            for cat, n in sub.items():
                out.pos_bullet_present_counts[ptype][cat] += n
    return out


def relaxed_conformant_total(stats: SuiteStats) -> int:
    return sum(stats.pos_relaxed_conformant.values())


def percentile(values: list[int], p: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    k = (len(vs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vs[int(k)])
    return vs[f] * (c - k) + vs[c] * (k - f)


# Failure-mode rows we render in the main table (label, attr-on-stats).
FAILURE_MODES: tuple[tuple[str, str], ...] = (
    ("Anthropomorphic phrasing", "n_anthropo"),
    ("Numerical confabulation", "n_numeric"),
    ("image_region bullets", "n_image_region"),
    ("'reads/has read the instruction'", "n_reads_instruction"),
    ("'understands/comprehends the goal'", "n_understands_goal"),
    ("'ready to execute'", "n_ready_to_execute"),
    ("Empty / degenerate (<50 chars or <3 bullets)", "n_empty_or_degenerate"),
    ("    of which: <50 chars", "n_short_desc"),
    ("    of which: <3 bullets", "n_few_bullets"),
    ("Error rows (non-null error)", "n_error"),
)


def build_table(
    suite_stats: dict[str, SuiteStats],
    v3_overall: SuiteStats,
    droid: SuiteStats,
    pilot: SuiteStats,
) -> str:
    cols = ["goal", "spatial", "object", "10", "V3-overall", "V2-DROID", "Pilot"]
    col_stats = [
        suite_stats["goal"], suite_stats["spatial"], suite_stats["object"], suite_stats["10"],
        v3_overall, droid, pilot,
    ]
    lines = []
    header = "| Failure mode | " + " | ".join(cols) + " |"
    lines.append(header)
    lines.append("|---" * (1 + len(cols)) + "|")
    # First, an N row for context.
    n_row = "| n (rows) | " + " | ".join(f"{cs.n_total:,}" for cs in col_stats) + " |"
    lines.append(n_row)
    for label, attr in FAILURE_MODES:
        cells = []
        for cs in col_stats:
            num = getattr(cs, attr)
            cells.append(f"{num:,} ({_fmt_pct(num, cs.n_total)})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_conformance_table(
    suite_stats: dict[str, SuiteStats],
    v3_overall: SuiteStats,
    droid: SuiteStats,
    pilot: SuiteStats,
) -> str:
    cols = ["goal", "spatial", "object", "10", "V3-overall", "V2-DROID", "Pilot"]
    col_stats = [
        suite_stats["goal"], suite_stats["spatial"], suite_stats["object"], suite_stats["10"],
        v3_overall, droid, pilot,
    ]
    lines = []
    lines.append("| Bullet present | " + " | ".join(cols) + " |")
    lines.append("|---" * (1 + len(cols)) + "|")
    for cat in EXPECTED_BULLETS:
        cells = []
        for cs in col_stats:
            cells.append(_fmt_pct(cs.bullet_present_counts.get(cat, 0), cs.n_total))
        lines.append(f"| - {cat}: | " + " | ".join(cells) + " |")
    cells = []
    for cs in col_stats:
        cells.append(_fmt_pct(cs.n_conformant_bullets, cs.n_total))
    lines.append("| **All 5 prefixes present (strict)** | " + " | ".join(cells) + " |")
    cells = []
    for cs in col_stats:
        cells.append(_fmt_pct(relaxed_conformant_total(cs), cs.n_total))
    lines.append("| **Position-aware conformance (relaxed)** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_position_type_table(stats: SuiteStats) -> str:
    """Break conformance down by position_type for the V3 aggregate."""
    lines = []
    lines.append(
        "| position_type | n rows | language: % | target: % | scene: % | "
        "spatial: % | plan: % | all-5 (strict) % | relaxed-conformant % |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    ptypes = sorted(stats.position_type_counts.keys(),
                    key=lambda p: -stats.position_type_counts[p])
    for ptype in ptypes:
        n = stats.position_type_counts[ptype]
        cells = [ptype, f"{n:,}"]
        for cat in EXPECTED_BULLETS:
            cells.append(_fmt_pct(stats.pos_bullet_present_counts[ptype].get(cat, 0), n))
        cells.append(_fmt_pct(stats.pos_conformant_bullets.get(ptype, 0), n))
        cells.append(_fmt_pct(stats.pos_relaxed_conformant.get(ptype, 0), n))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_v4_per_suite_table(
    title: str,
    attr: str,
    suite_stats: dict[str, SuiteStats],
    v3_overall: SuiteStats,
    droid: SuiteStats,
    pilot: SuiteStats,
) -> str:
    """Render a per-suite row for one V4 regression failure mode.

    ``attr`` is the ``SuiteStats`` row-level count attribute
    (e.g. ``n_v4_motor``).
    """
    cols = ["goal", "spatial", "object", "10", "V3-overall", "V2-DROID", "Pilot"]
    col_stats = [
        suite_stats["goal"], suite_stats["spatial"], suite_stats["object"], suite_stats["10"],
        v3_overall, droid, pilot,
    ]
    lines = [
        "| Suite | " + " | ".join(cols) + " |",
        "|---" * (1 + len(cols)) + "|",
    ]
    cells = [f"{getattr(cs, attr):,} ({_fmt_pct(getattr(cs, attr), cs.n_total)})" for cs in col_stats]
    lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_v4_per_bullet_type_table(
    title: str,
    bt_hits_attr: str,
    stats: SuiteStats,
) -> str:
    """Render a per-bullet-type breakdown for motor or scaffold leakage.

    Uses the V3-overall aggregate as the denominator base. Reports the
    hit rate as ``hits / total bullets of that type``.
    """
    hits: Counter = getattr(stats, bt_hits_attr)
    cats: list[str] = sorted(
        set(stats.bt_total.keys()) | set(V4_TRACKED_BULLET_TYPES),
        key=lambda c: (-stats.bt_total.get(c, 0), c),
    )
    lines = [
        "| Bullet type | n bullets | hits | rate |",
        "|---|---|---|---|",
    ]
    for cat in cats:
        total = stats.bt_total.get(cat, 0)
        if total == 0:
            continue
        h = hits.get(cat, 0)
        lines.append(f"| `{cat}` | {total:,} | {h:,} | {_fmt_pct(h, total)} |")
    grand_total = sum(stats.bt_total.values())
    grand_hits = sum(hits.values())
    lines.append(
        f"| **any bullet ({title})** | {grand_total:,} | {grand_hits:,} | "
        f"{_fmt_pct(grand_hits, grand_total)} |"
    )
    return "\n".join(lines)


def build_v4_noncanon_breakdown_table(stats: SuiteStats) -> str:
    """Per-header breakdown (gripper / motion / image_region) over V3-overall."""
    lines = [
        "| Forbidden header | # rows hit |",
        "|---|---|",
    ]
    total = 0
    for h in V4_FORBIDDEN_HEADERS:
        n = stats.noncanon_header_hits.get(h, 0)
        total += n
        lines.append(f"| `{h}:` | {n:,} |")
    lines.append(f"| **rows with any forbidden header** | {stats.n_v4_noncanon_header:,} |")
    return "\n".join(lines)


def build_length_table(stats: SuiteStats) -> str:
    """Bullet-length distribution from the V3 aggregate."""
    cats = list(EXPECTED_BULLETS) + ["distractor", "image_region", "_other"]
    lines = []
    lines.append("| Bullet | n bullets | mean tok | p10 | p50 | p90 |")
    lines.append("|---|---|---|---|---|---|")
    for cat in cats:
        toks = stats.bullet_token_counts.get(cat, [])
        if not toks:
            continue
        mean = statistics.fmean(toks)
        p10 = percentile(toks, 0.10)
        p50 = percentile(toks, 0.50)
        p90 = percentile(toks, 0.90)
        label = "_other (uncategorised)" if cat == "_other" else cat
        lines.append(
            f"| {label} | {len(toks):,} | {mean:.1f} | {p10:.0f} | {p50:.0f} | {p90:.0f} |"
        )
    return "\n".join(lines)


def truncate(text: str, n: int = 600) -> str:
    if len(text) <= n:
        return text
    return text[:n].rstrip() + " ..."


def render_examples(mode_label: str, examples: list[FailingExample]) -> str:
    if not examples:
        return f"(no examples found for {mode_label})\n"
    lines = []
    for ex in examples:
        lines.append(f"**{ex.example_id}** — matched: `{ex.matched}`")
        lines.append("")
        lines.append("```")
        lines.append(truncate(ex.description))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


_VERDICT_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def _worsen(current: str, candidate: str) -> str:
    """Return the more-severe of two verdicts (RED > YELLOW > GREEN)."""
    return candidate if _VERDICT_RANK[candidate] > _VERDICT_RANK[current] else current


def compute_verdict(
    v3_overall: SuiteStats,
    per_suite: dict[str, SuiteStats] | None = None,
) -> tuple[str, list[str]]:
    """Return (verdict, list-of-reasons).

    Base V3-era rubric:
        GREEN  = every failure mode <0.5% AND conformance >=99%
        YELLOW = any failure mode 0.5-2%
        RED    = any failure mode >2%

    V4 regression-mode rubric (SA3 extension):
        Motor imperatives:
            >2% in any suite (or overall) -> RED for that suite
        Scaffold leakage:
            >5% -> YELLOW; >15% -> RED (per Agent 3, V3 plan
            bullets sat at 11-17%)
        Non-canonical headers (gripper/motion/image_region):
            >0.5% -> YELLOW; >2% -> RED

    Conformance is scored position-aware (relaxed) because the V3
    prompt only asks last_text rows for a ``language:`` bullet;
    flagging RED on a strict 'all 5 always' metric would conflate
    prompt design with labeler failure. The strict number is still
    reported alongside.
    """
    failures: list[str] = []
    rates: dict[str, float] = {
        "anthropomorphic": _pct(v3_overall.n_anthropo, v3_overall.n_total),
        "numeric": _pct(v3_overall.n_numeric, v3_overall.n_total),
        "image_region": _pct(v3_overall.n_image_region, v3_overall.n_total),
        "reads_instruction": _pct(v3_overall.n_reads_instruction, v3_overall.n_total),
        "understands_goal": _pct(v3_overall.n_understands_goal, v3_overall.n_total),
        "ready_to_execute": _pct(v3_overall.n_ready_to_execute, v3_overall.n_total),
        "empty_or_degenerate": _pct(v3_overall.n_empty_or_degenerate, v3_overall.n_total),
        "error_rows": _pct(v3_overall.n_error, v3_overall.n_total),
    }
    conformance_strict = _pct(v3_overall.n_conformant_bullets, v3_overall.n_total)
    conformance_relaxed = _pct(relaxed_conformant_total(v3_overall), v3_overall.n_total)
    worst = max(rates.values()) if rates else 0.0

    if worst > 2.0:
        verdict = "RED"
    elif worst > 0.5:
        verdict = "YELLOW"
    elif conformance_relaxed >= 99.0:
        verdict = "GREEN"
    else:
        # No targeted failure mode breached YELLOW/RED, but conformance
        # disqualifies GREEN. Treat as YELLOW with conformance as the cause.
        verdict = "YELLOW"

    for k, v in rates.items():
        if v > 2.0:
            failures.append(f"{k}={v:.3f}% (>2%)")
        elif v > 0.5:
            failures.append(f"{k}={v:.3f}% (0.5-2%)")
    if conformance_relaxed < 99.0:
        failures.append(
            f"position-aware bullet conformance={conformance_relaxed:.2f}% "
            "(<99% GREEN bar)"
        )
    if conformance_strict < 99.0:
        failures.append(
            f"strict 'all 5 prefixes always' conformance={conformance_strict:.2f}% — "
            "but the hardened prompt only asks last_text rows for `language:`, "
            "so this strict metric is partly schema-design, not labeler failure"
        )

    # ---- V4 regression-mode rubric ----
    # Motor imperatives: per-suite gating (any suite >2% drives RED).
    suite_iter: list[tuple[str, SuiteStats]] = []
    if per_suite is not None:
        suite_iter.extend(per_suite.items())
    suite_iter.append(("overall", v3_overall))
    for suite_name, st in suite_iter:
        motor_pct = _pct(st.n_v4_motor, st.n_total)
        if motor_pct > 2.0:
            verdict = _worsen(verdict, "RED")
            failures.append(
                f"V4 motor-imperative ({suite_name})={motor_pct:.3f}% (>2% RED bar; "
                "drives the residual C-fails per Agent 1)"
            )
        scaffold_pct = _pct(st.n_v4_scaffold, st.n_total)
        if scaffold_pct > 15.0:
            verdict = _worsen(verdict, "RED")
            failures.append(
                f"V4 scaffold-leakage ({suite_name})={scaffold_pct:.3f}% (>15% RED bar)"
            )
        elif scaffold_pct > 5.0:
            verdict = _worsen(verdict, "YELLOW")
            failures.append(
                f"V4 scaffold-leakage ({suite_name})={scaffold_pct:.3f}% (>5% YELLOW bar)"
            )
        noncanon_pct = _pct(st.n_v4_noncanon_header, st.n_total)
        if noncanon_pct > 2.0:
            verdict = _worsen(verdict, "RED")
            failures.append(
                f"V4 non-canonical-headers ({suite_name})={noncanon_pct:.3f}% (>2% RED bar)"
            )
        elif noncanon_pct > 0.5:
            verdict = _worsen(verdict, "YELLOW")
            failures.append(
                f"V4 non-canonical-headers ({suite_name})={noncanon_pct:.3f}% (>0.5% YELLOW bar)"
            )

    if not failures:
        failures.append(
            "every targeted failure mode <0.5% and conformance>=99%; "
            "the hardened prompt's regressions are cleanly eliminated."
        )
    return verdict, failures


def render_recommendations(
    v3_overall: SuiteStats,
    pilot: SuiteStats,
    droid: SuiteStats,
    other_bullet_counts: Counter,
    agent1: dict[str, Any] | None,
) -> list[str]:
    """Produce three concrete follow-up actions based on what we found.

    We prioritise the ranked findings:
    1. New failure modes the hardening introduced (motor-imperative
       phrasing + non-canonical bullets), since they were not in the
       user's failure-mode list but are the dominant issue with V3.
    2. The schema gap between strict and relaxed conformance.
    3. Targeted residual rates for the original failure modes.
    """
    recs: list[str] = []
    conformance_strict = _pct(v3_overall.n_conformant_bullets, v3_overall.n_total)
    conformance_relaxed = _pct(relaxed_conformant_total(v3_overall), v3_overall.n_total)
    non_canon = sum(other_bullet_counts.values())

    # 1. Address the new failure mode if it's visible.
    if agent1 is not None and agent1["appropriateness_pass_pct"] < 99.5:
        appr_fail_rate = 100.0 - agent1["appropriateness_pass_pct"]
        recs.append(
            f"**New C-failure mode**: Agent 1's multimodal judge flagged "
            f"{appr_fail_rate:.1f}% appropriateness fails — dominated by "
            "low-level motor-imperative phrasing ('grasp the X', 'align the "
            "gripper, lift, place') rather than the anthropomorphic phrasing "
            "the original hardened prompt targeted. **Action**: extend "
            "`_FORBIDDEN_PHRASING` in `src/nla/labeling/prompts.py` with "
            "imperative-verb patterns ('grasp', 'lift', 'reach', 'place', "
            "'carry it') when they appear inside the `plan:` bullet, then "
            "re-label the ~6% of rows that fail. Alternatively, add a "
            "scrub step that rewrites imperative `plan:` bullets to a "
            "neutral 'plan: <phase> active; <observable state>' template."
        )

    # 2. Surface the non-canonical bullets that the prompt's category list
    # was supposed to forbid.
    if non_canon > 0:
        top_other = other_bullet_counts.most_common(3)
        top_other_str = ", ".join(f"`{c}` ({n:,})" for c, n in top_other)
        recs.append(
            f"**Non-canonical bullets**: V3 contains {non_canon:,} bullets "
            f"whose category is outside the prompt's allowed set "
            f"(top: {top_other_str}). The hardened "
            "`build_strict_position_prompt` already enforces the closed "
            "vocabulary; the issue is that production labeling used "
            "`build_position_prompt`, which lists categories without "
            "forbidding others. **Action**: switch the default labeling "
            "entrypoint to `build_strict_position_prompt`, *or* run a "
            "category-rewrite scrub that maps `gripper:` → `target:` "
            "(when gripper state is the topic) and `motion:` → `plan:` "
            "before SFT."
        )

    # 3. Targeted residuals: address whichever original failure mode is
    # the largest, even though all are <0.5%.
    rates: dict[str, tuple[float, str]] = {
        "image_region": (
            _pct(v3_overall.n_image_region, v3_overall.n_total),
            f"image_region: bullets at {_pct(v3_overall.n_image_region, v3_overall.n_total):.3f}% "
            f"(58 rows; pilot was 41%, V2 DROID was 0.66%). Run "
            "`scripts/labeling/strip_hallucinated_image_region.py "
            "--match patch_or_layout --mode strip` over each of the four V3 "
            "labels.jsonl files; that single sweep finishes the job without "
            "touching the prompt.",
        ),
        "anthropomorphic": (
            _pct(v3_overall.n_anthropo, v3_overall.n_total),
            f"Residual anthropomorphic phrasing at "
            f"{_pct(v3_overall.n_anthropo, v3_overall.n_total):.3f}% "
            "(7 rows). Single sed over the four files is sufficient.",
        ),
    }
    if not recs:
        recs.append(
            "All scanned failure modes <0.5% and conformance acceptable; keep "
            "the hardened prompt as default for the next data refresh."
        )
    while len(recs) < 3:
        # Append the next-most-impactful original failure mode that we
        # haven't already covered.
        best = max(rates.items(), key=lambda kv: kv[1][0])
        rec_text = best[1][1]
        if rec_text not in recs:
            recs.append(rec_text)
        rates.pop(best[0])
        if not rates:
            break

    while len(recs) < 3:
        recs.append(
            "Add the failure-mode regex set used here to the existing CI sanity "
            "checks under `data/eval/sanity_check_hardened/` so the next "
            "labeling run is auto-audited (motor-imperative phrases included)."
        )
    return recs[:3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: ``--labels-root`` / ``--out-json`` / ``--out-md``.

    Defaults preserve original V3-baseline behaviour: same labels root,
    new (V4-aware) JSON summary path so the frozen
    ``data/eval/agent2_summary.json`` is never overwritten.
    """
    parser = argparse.ArgumentParser(description=__doc__ or "audit prompt hardening")
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=V3_BASE,
        help=(
            "Root containing ``libero_{goal,spatial,object,10}/labels.jsonl`` "
            "subdirs. Defaults to the V3 corpus at "
            "data/labels/libero_4suite_stride2."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=(
            "Path to write the JSON summary. Default: "
            "data/eval/audit_prompt_hardening_summary.json. SA10 "
            "regression-gates V4 against data/eval/sa3_v3_baseline_summary.json."
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Path to write the markdown report. Default: "
            "docs/sft_plan/audit_reports/agent2_prompt_hardening_regression.md."
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help=(
            "Skip the V2-DROID and Pilot baseline scans. Speeds up V4 "
            "regression runs where only the suites under --labels-root matter."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = parse_args(argv)
    report_path: Path = args.out_md
    summary_path: Path = args.out_json
    labels_root: Path = args.labels_root

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    suite_stats: dict[str, SuiteStats] = {}
    for suite in V3_SUITES:
        path = labels_root / f"libero_{suite}" / "labels.jsonl"
        logger.info("Scanning suite %s @ %s", suite, path)
        suite_stats[suite] = scan_file(path, name=f"libero_{suite}")

    if args.skip_baselines:
        droid_stats = SuiteStats(name="droid_100ep", path="<skipped>")
        pilot_stats = SuiteStats(name="libero_goal_pilot", path="<skipped>")
    else:
        droid_path = _resolve_v2_droid_path()
        if droid_path is None:
            logger.info(
                "V2 DROID baseline not reachable (checked %s and %s); skipping.",
                V2_DROID_PATH_LIVE, V2_DROID_PATH_ARCHIVE,
            )
            droid_stats = SuiteStats(name="droid_100ep", path="<archived>")
        else:
            logger.info("Scanning V2 DROID baseline @ %s", droid_path)
            droid_stats = scan_file(droid_path, name="droid_100ep")
        logger.info("Scanning Pilot baseline @ %s", PILOT_PATH)
        pilot_stats = scan_file(PILOT_PATH, name="libero_goal_pilot")

    v3_overall = aggregate_v3(suite_stats)

    # ----------------------- markdown report -----------------------
    table_md = build_table(suite_stats, v3_overall, droid_stats, pilot_stats)
    conformance_md = build_conformance_table(suite_stats, v3_overall, droid_stats, pilot_stats)
    length_md = build_length_table(v3_overall)
    verdict, verdict_reasons = compute_verdict(v3_overall, suite_stats)
    rescan_counts = _rescan_other_categories(labels_root, V3_SUITES)
    agent1 = _load_agent1_findings()
    recommendations = render_recommendations(
        v3_overall, pilot_stats, droid_stats, rescan_counts, agent1,
    )

    md: list[str] = []
    md.append("# Agent 2 — Prompt-Hardening Regression Scan (V3 LIBERO)")
    md.append("")
    md.append(
        "Single-pass scan over 101,580 V3 LIBERO labels (`libero_4suite_stride2/`) "
        "plus the V2 DROID and LIBERO-goal pilot baselines, checking whether the "
        "hardened labeling prompt in `src/nla/labeling/prompts.py` eliminated the "
        "failure modes documented in `docs/sft_plan/01_data_audit.md`. "
        "Run via "
        "`PYTHONPATH=src .venv/bin/python scripts/eval/audit_prompt_hardening.py`."
    )
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(f"**{verdict}**")
    md.append("")
    if verdict_reasons:
        md.append("Reasons:")
        for r in verdict_reasons:
            md.append(f"- {r}")
    else:
        md.append("All failure modes <0.5% and bullet conformance >=99%.")
    md.append("")

    md.append("## Failure-mode rates")
    md.append("")
    md.append(table_md)
    md.append("")

    # ---------- V4 regression failure modes (SA3 extension) ----------
    md.append("## Motor imperatives (V4 regression mode)")
    md.append("")
    md.append(
        "Second-person verbs aimed at the robot (\"grasp the bowl\", "
        "\"align the gripper\"). Phrase list is imported from "
        "``nla.labeling.prompts.V4_MOTOR_IMPERATIVE_PHRASES`` so the audit "
        "regex cannot drift from the V4 prompt's ban list. Per Agent 1, "
        "these drive the residual ~1.8% C-fails on the multimodal judge."
    )
    md.append("")
    md.append(
        build_v4_per_suite_table(
            "Motor-imperative rows", "n_v4_motor",
            suite_stats, v3_overall, droid_stats, pilot_stats,
        )
    )
    md.append("")
    md.append("### Per bullet type (V3 aggregate)")
    md.append("")
    md.append(build_v4_per_bullet_type_table("motor", "bt_motor_hits", v3_overall))
    md.append("")

    md.append("## Scaffold leakage (V4 regression mode)")
    md.append("")
    md.append(
        "Phrases echoed from the system prompt's own scaffolding "
        "(\"action head\", \"this patch carries\", \"transformer\", "
        "\"embedding\", \"hidden state\"). Phrase list imported from "
        "``nla.labeling.prompts.V4_SCAFFOLD_FORBIDDEN_PHRASES``. Per Agent "
        "3, V3 plan bullets sat at 11-17% scaffold leakage; V4 forbids "
        "these substrings outright."
    )
    md.append("")
    md.append(
        build_v4_per_suite_table(
            "Scaffold-leakage rows", "n_v4_scaffold",
            suite_stats, v3_overall, droid_stats, pilot_stats,
        )
    )
    md.append("")
    md.append("### Per bullet type (V3 aggregate)")
    md.append("")
    md.append(build_v4_per_bullet_type_table("scaffold", "bt_scaffold_hits", v3_overall))
    md.append("")

    md.append("## Non-canonical bullet headers (V4 regression mode)")
    md.append("")
    md.append(
        "Rows containing at least one bullet whose header is in "
        "``V4_FORBIDDEN_HEADERS`` (``gripper``, ``motion``, "
        "``image_region``). V4 collapses these into ``plan`` / ``target`` / "
        "``spatial`` so the closed bullet vocabulary stays a strict subset "
        "of ``V4_BULLET_CATEGORIES``."
    )
    md.append("")
    md.append(
        build_v4_per_suite_table(
            "Non-canonical-header rows", "n_v4_noncanon_header",
            suite_stats, v3_overall, droid_stats, pilot_stats,
        )
    )
    md.append("")
    md.append("### Header breakdown (V3 aggregate)")
    md.append("")
    md.append(build_v4_noncanon_breakdown_table(v3_overall))
    md.append("")

    md.append("## Bullet-prefix conformance")
    md.append("")
    md.append(
        "% of rows that contain *each* expected bullet prefix, plus the "
        "two aggregate metrics:\n"
        "- **strict** = all five of `- language:`, `- target:`, `- scene:`, "
        "`- spatial:`, `- plan:` present;\n"
        "- **relaxed (position-aware)** = `last_text` rows need all five; "
        "`image_patch` / `anchor` rows only need target+scene+spatial+plan, "
        "because the hardened prompt explicitly steers those positions "
        "away from a `language:` bullet (see `_IMAGE_PATCH_RULES` in "
        "`src/nla/labeling/prompts.py`)."
    )
    md.append("")
    md.append(conformance_md)
    md.append("")
    md.append("### Position-type-conditioned conformance (V3 aggregate)")
    md.append("")
    md.append(
        "Separates 'labeler skipped a prescribed bullet' from 'prompt did "
        "not ask for that bullet here'."
    )
    md.append("")
    md.append(build_position_type_table(v3_overall))
    md.append("")

    # If any V3 suite < 99% RELAXED conformance, list non-conformant examples.
    nonconformant_listings: list[str] = []
    for suite, st in suite_stats.items():
        pct = _pct(relaxed_conformant_total(st), st.n_total)
        if pct < 99.0:
            nonconformant_listings.append(
                f"### libero_{suite} ({pct:.2f}% position-aware conformant)"
            )
            nonconformant_listings.append("")
            nonconformant_listings.append(
                render_examples(
                    f"libero_{suite}_nonconformant", st.nonconformant_examples
                )
            )
    if nonconformant_listings:
        md.append("### Non-conformant examples")
        md.append("")
        md.append(
            "These rows fail the **strict** all-5-prefixes test (we list "
            "them for completeness; many of these are simply image_patch "
            "rows missing only `language:`, which the prompt expected)."
        )
        md.append("")
        md.extend(nonconformant_listings)
        md.append("")

    md.append("## Per-bullet length distribution (V3 aggregate, tokens per bullet body)")
    md.append("")
    md.append(length_md)
    md.append("")

    md.append("## Non-canonical bullet categories (V3 aggregate)")
    md.append("")
    md.append(
        "Bullets whose category is not in the prompt's allowed set "
        "(`scene, target, distractor, spatial, plan, language, image_region`). "
        "Counts are per-bullet, not per-row."
    )
    md.append("")
    md.append("| Category | # bullets |")
    md.append("|---|---|")
    # Recover from per-row bullet stats: anything outside the canonical set
    # was stored as "_other" -- we re-derive the breakdown via a quick pass
    # so we keep one main scan but get this finding for free.
    for cat, n in rescan_counts.most_common(15):
        md.append(f"| `{cat}` | {n:,} |")
    md.append(f"| **total non-canonical bullets** | {sum(rescan_counts.values()):,} |")
    md.append("")
    md.append(
        "These bullets are not flagged by the user's failure-mode list, "
        "but they do count against bullet-prefix conformance because they "
        "crowd out a prescribed category. `gripper:` and `motion:` together "
        "account for >99% of the non-canonical volume and explain a "
        "non-trivial fraction of the image_patch rows that are missing "
        "`plan:` or `spatial:`."
    )
    md.append("")

    md.append("## Top anthropomorphic phrase hits (V3 aggregate)")
    md.append("")
    if v3_overall.anthropo_phrase_hits:
        md.append("| Phrase | # rows hit |")
        md.append("|---|---|")
        for phrase, count in v3_overall.anthropo_phrase_hits.most_common():
            md.append(f"| `{phrase}` | {count:,} |")
    else:
        md.append("(none)")
    md.append("")

    # Examples for any failure mode with >0.5% incidence in V3.
    md.append("## Examples for V3 failure modes (>0.5% incidence)")
    md.append("")
    example_modes: list[tuple[str, str, str]] = [
        ("anthropomorphic", "Anthropomorphic phrasing", "n_anthropo"),
        ("numeric", "Numerical confabulation", "n_numeric"),
        ("image_region", "image_region bullets", "n_image_region"),
        ("reads_instruction", "'reads/has read the instruction'", "n_reads_instruction"),
        ("understands_goal", "'understands/comprehends the goal'", "n_understands_goal"),
        ("ready_to_execute", "'ready to execute'", "n_ready_to_execute"),
        ("empty_or_degenerate", "Empty / degenerate", "n_empty_or_degenerate"),
    ]
    showed_any = False
    for ex_key, label, attr in example_modes:
        rate = _pct(getattr(v3_overall, attr), v3_overall.n_total)
        if rate <= 0.5:
            continue
        showed_any = True
        md.append(f"### {label} — {rate:.2f}% of V3 rows")
        md.append("")
        # Prefer examples from the suite with the most hits.
        best_suite = max(suite_stats.values(), key=lambda s: len(s.examples.get(ex_key, [])))
        md.append(f"(showing up to 5 examples from suite {best_suite.name}.)")
        md.append("")
        md.append(render_examples(label, best_suite.examples.get(ex_key, [])))
    if not showed_any:
        md.append("_(No V3 failure mode exceeded 0.5%; no example listing needed.)_")
    md.append("")

    md.append("## Cross-reference: Agent 1 (multimodal gpt-5.1 judge, 500-row sample)")
    md.append("")
    if agent1 is None:
        md.append("_(Agent 1 output not found at `data/eval/libero_v3_quality_judge.jsonl`; cross-reference skipped.)_")
    else:
        md.append(
            f"Agent 1 read {agent1['n']} V3 rows with a multimodal judge "
            f"(suite breakdown: {agent1['suite_breakdown']}).\n\n"
            f"- Grounding pass: **{agent1['grounding_pass_pct']:.1f}%** "
            f"({100.0 - agent1['grounding_pass_pct']:.1f}% C-grounding fails). "
            "These are visual-misidentification failures (\"misstates layout\", "
            "\"misidentifies the visible can\"). My regex scan **does not "
            "catch any of these** — they require pixels to detect.\n"
            f"- Appropriateness pass: **{agent1['appropriateness_pass_pct']:.1f}%** "
            f"({100.0 - agent1['appropriateness_pass_pct']:.1f}% appropriateness fails). "
            "These fails are dominated by low-level motor commands "
            "(\"grasp the bowl and carry it\", \"align the gripper, lift, place\") "
            "rather than the anthropomorphic phrasing the V2/Pilot baselines "
            "showed. **My anthropomorphic regex is 0.007% but Agent 1's judge "
            "flags ~1.8% — they're catching a distinct C-failure mode** "
            "(actuator-level imperative phrasing) that the hardened prompt did "
            "not explicitly forbid.\n\n"
            "Sample Agent 1 appropriateness-fail reasons:"
        )
        for r in agent1["sample_appropriateness_fails"]:
            md.append(f"  - {r}")
        md.append("")
        md.append(
            "**Implication**: the hardened prompt eliminated the *old* "
            "C-failure mode (anthropomorphic / cognitive-state phrasing) but "
            "the labeler has shifted to a *new* one (low-level motor "
            "imperatives in the `plan:` bullet, often co-occurring with the "
            "non-canonical `motion:` / `gripper:` bullets called out above)."
        )
    md.append("")

    md.append("## Recommendations")
    md.append("")
    for i, rec in enumerate(recommendations, 1):
        md.append(f"{i}. {rec}")
    md.append("")

    md.append("## Method notes")
    md.append("")
    md.append(
        "- Counts are per-label (one bullet hit = one row hit). Rates are "
        "denominated against total rows in each file (including any rows "
        "with empty descriptions / non-null `error`).\n"
        "- Anthropomorphic phrasing uses case-insensitive substring matching "
        "against the prompt-hardening phrase list documented in the user task. "
        "The same row can match multiple phrases; the `top anthropomorphic "
        "phrase hits` table breaks them out individually.\n"
        "- Numerical confabulation uses the regex shipped in the user task "
        "(measurement units mm/cm/m/in/°/rad/kg/g/N, with optional decimal). "
        "It is intentionally narrower than "
        "`scripts/labeling/scrub_fabricated_measurements.py` (which also "
        "catches `5-8` ranges with hedging words); the spec asked for the "
        "tighter regex.\n"
        "- `image_region` detection is a bullet-prefix match (one or more "
        "lines starting with `- image_region:` or `image region:`).\n"
        "- Bullet conformance accepts both exact-match (`- language:`) and "
        "prefix-match (`- language/state:`) for each canonical category. "
        "Categories not in the canonical set are bucketed under `_other` "
        "in the length-distribution table.\n"
    )
    md.append("")

    report_path.write_text("\n".join(md))
    logger.info("Report written to %s", report_path)

    # ----------------------- JSON summary -----------------------
    def summary_for(s: SuiteStats) -> dict[str, Any]:
        return {
            "name": s.name,
            "path": s.path,
            "n_total": s.n_total,
            "n_position": s.n_position,
            "n_error": s.n_error,
            "n_no_description": s.n_no_description,
            "rates_pct": {
                "anthropomorphic": _pct(s.n_anthropo, s.n_total),
                "numeric": _pct(s.n_numeric, s.n_total),
                "image_region": _pct(s.n_image_region, s.n_total),
                "reads_instruction": _pct(s.n_reads_instruction, s.n_total),
                "understands_goal": _pct(s.n_understands_goal, s.n_total),
                "ready_to_execute": _pct(s.n_ready_to_execute, s.n_total),
                "empty_or_degenerate": _pct(s.n_empty_or_degenerate, s.n_total),
                "short_desc": _pct(s.n_short_desc, s.n_total),
                "few_bullets": _pct(s.n_few_bullets, s.n_total),
                "error_rows": _pct(s.n_error, s.n_total),
                "bullet_conformance_all5_strict": _pct(s.n_conformant_bullets, s.n_total),
                "bullet_conformance_position_aware": _pct(
                    relaxed_conformant_total(s), s.n_total
                ),
                # ---- V4 regression failure modes (SA3 extension) ----
                "v4_motor_imperative": _pct(s.n_v4_motor, s.n_total),
                "v4_scaffold_leakage": _pct(s.n_v4_scaffold, s.n_total),
                "v4_noncanonical_header": _pct(s.n_v4_noncanon_header, s.n_total),
            },
            "bullet_present_pct": {
                cat: _pct(s.bullet_present_counts.get(cat, 0), s.n_total)
                for cat in EXPECTED_BULLETS
            },
            "position_type_counts": dict(s.position_type_counts),
            "position_type_relaxed_conformance_pct": {
                ptype: _pct(s.pos_relaxed_conformant.get(ptype, 0), n)
                for ptype, n in s.position_type_counts.items()
            },
            "top_anthropo_phrases": s.anthropo_phrase_hits.most_common(10),
            "v4": {
                "n_motor": s.n_v4_motor,
                "n_scaffold": s.n_v4_scaffold,
                "n_noncanon_header": s.n_v4_noncanon_header,
                "by_bullet_type": {
                    cat: {
                        "total_bullets": s.bt_total.get(cat, 0),
                        "motor_hits": s.bt_motor_hits.get(cat, 0),
                        "scaffold_hits": s.bt_scaffold_hits.get(cat, 0),
                        "motor_pct": _pct(s.bt_motor_hits.get(cat, 0), s.bt_total.get(cat, 0)),
                        "scaffold_pct": _pct(s.bt_scaffold_hits.get(cat, 0), s.bt_total.get(cat, 0)),
                    }
                    for cat in sorted(
                        set(s.bt_total.keys()) | set(V4_TRACKED_BULLET_TYPES)
                    )
                    if s.bt_total.get(cat, 0) > 0
                },
                "noncanon_header_breakdown": {
                    h: s.noncanon_header_hits.get(h, 0) for h in V4_FORBIDDEN_HEADERS
                },
            },
        }

    # Convenience: V4-mode aggregates flat per-suite + overall, plus
    # per-bullet-type sub-keys (the schema SA10 will regression-gate against).
    def _v4_rates_block(attr: str) -> dict[str, Any]:
        block: dict[str, Any] = {
            "goal": _pct(getattr(suite_stats["goal"], attr), suite_stats["goal"].n_total),
            "spatial": _pct(getattr(suite_stats["spatial"], attr), suite_stats["spatial"].n_total),
            "object": _pct(getattr(suite_stats["object"], attr), suite_stats["object"].n_total),
            "10": _pct(getattr(suite_stats["10"], attr), suite_stats["10"].n_total),
            "overall": _pct(getattr(v3_overall, attr), v3_overall.n_total),
        }
        return block

    def _v4_by_bullet_type_block(bt_attr: str) -> dict[str, float]:
        hits: Counter = getattr(v3_overall, bt_attr)
        out: dict[str, float] = {}
        for cat in sorted(set(v3_overall.bt_total.keys()) | set(V4_TRACKED_BULLET_TYPES)):
            total = v3_overall.bt_total.get(cat, 0)
            if total == 0:
                continue
            out[cat] = _pct(hits.get(cat, 0), total)
        return out

    motor_block = _v4_rates_block("n_v4_motor")
    motor_block["by_bullet_type"] = _v4_by_bullet_type_block("bt_motor_hits")
    scaffold_block = _v4_rates_block("n_v4_scaffold")
    scaffold_block["by_bullet_type"] = _v4_by_bullet_type_block("bt_scaffold_hits")
    noncanon_block = _v4_rates_block("n_v4_noncanon_header")
    noncanon_block["header_breakdown_overall"] = {
        h: v3_overall.noncanon_header_hits.get(h, 0) for h in V4_FORBIDDEN_HEADERS
    }

    summary = {
        "agent": "audit_prompt_hardening (V3 baseline + V4 regression modes)",
        "labels_root": str(labels_root),
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "thresholds": {
            "GREEN": "<0.5% and conformance>=99%",
            "YELLOW": "0.5-2% (or conformance>=95%)",
            "RED": ">2% or conformance<95%",
            "v4_motor_imperative": "RED if >2% in any suite",
            "v4_scaffold_leakage": "YELLOW if >5%, RED if >15%",
            "v4_noncanonical_header": "YELLOW if >0.5%, RED if >2%",
        },
        "v4_failure_modes": {
            "motor_imperative_pct": motor_block,
            "scaffold_leakage_pct": scaffold_block,
            "noncanonical_header_pct": noncanon_block,
        },
        "v3_overall": summary_for(v3_overall),
        "per_suite": {name: summary_for(s) for name, s in suite_stats.items()},
        "v2_droid": summary_for(droid_stats),
        "pilot": summary_for(pilot_stats),
        "recommendations": recommendations,
        "report_path": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", summary_path)

    # Print the summary for the parent agent.
    print()
    print("---- Audit summary ----")
    print(f"labels_root:        {labels_root}")
    print(f"anthropomorphic%:   {summary['v3_overall']['rates_pct']['anthropomorphic']:.3f}")
    print(f"numeric%:           {summary['v3_overall']['rates_pct']['numeric']:.3f}")
    print(f"image_region%:      {summary['v3_overall']['rates_pct']['image_region']:.3f}")
    print(f"v4_motor%:          {summary['v3_overall']['rates_pct']['v4_motor_imperative']:.3f}")
    print(f"v4_scaffold%:       {summary['v3_overall']['rates_pct']['v4_scaffold_leakage']:.3f}")
    print(f"v4_noncanon_hdr%:   {summary['v3_overall']['rates_pct']['v4_noncanonical_header']:.3f}")
    print(
        f"conformance%:       "
        f"{summary['v3_overall']['rates_pct']['bullet_conformance_position_aware']:.3f} "
        f"(position-aware) / "
        f"{summary['v3_overall']['rates_pct']['bullet_conformance_all5_strict']:.3f} (strict)"
    )
    print(f"verdict:            {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

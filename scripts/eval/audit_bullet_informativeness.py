"""Bullet-level grounding & informativeness audit for V3 LIBERO labels.

Agent 4 of the V3 data-quality audit suite. Streams each suite's labels.jsonl,
parses the structured 5-bullet captions, and computes per-bullet:

    * length stats (whitespace tokens) per suite
    * presence rate
    * concrete-noun rate (LIBERO vocab + colors + parts)
    * cross-bullet Jaccard redundancy (5x5 matrix, mean over rows)
    * position-type sensitivity (image_patch vs last_text top n-grams)
    * filler-bullet rate (bullet contains >=80% of instruction's content words)
    * plan-phase taxonomy per suite (top phases by frequency)

Emits a markdown report to docs/sft_plan/audit_reports/agent4_bullet_informativeness.md.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/audit_bullet_informativeness.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Iterator

# -----------------------------------------------------------------------------
# Config / vocab
# -----------------------------------------------------------------------------

LABELS_ROOT = Path("data/labels/libero_4suite_stride2")
SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")
OUT_PATH = Path("docs/sft_plan/audit_reports/agent4_bullet_informativeness.md")

# The five bullet types this audit is scoped to.
BULLETS = ("language", "target", "scene", "spatial", "plan")

# LIBERO object vocabulary. Multi-word phrases (checked first as substrings) and
# single-word nouns (checked as tokens).  Conservative list drawn from the
# LIBERO-{goal, spatial, object, 10} task specs.
LIBERO_MULTIWORD = (
    "wine bottle", "tomato sauce", "cream cheese", "chocolate pudding",
    "salad dressing", "alphabet soup", "bbq sauce", "orange juice",
    "milk carton", "butter stick", "cookie box", "soup can",
    "ketchup bottle", "mustard bottle", "soda can",
)
LIBERO_OBJECTS = (
    # containers / placements
    "bowl", "plate", "cup", "mug", "tray", "basket", "caddy", "rack",
    "pot", "pan", "box", "carton", "can", "jar", "drawer", "cabinet",
    "microwave", "oven", "stove", "shelf", "table", "tabletop",
    # graspable items
    "bottle", "cube", "block", "milk", "butter", "book", "tomato",
    "ketchup", "mustard", "pudding", "cheese", "soup", "soda", "juice",
    "sauce", "dressing", "lid", "handle", "opening", "rim", "surface",
    # robot parts
    "gripper", "arm", "jaws", "claw", "fingers",
)
LIBERO_COLORS = (
    "red", "blue", "green", "yellow", "black", "white", "brown",
    "pink", "orange", "purple", "gray", "grey", "tan", "beige",
    "wooden", "metallic", "transparent",
)
CONCRETE_VOCAB_SINGLE = set(LIBERO_OBJECTS) | set(LIBERO_COLORS)

# Plan-phase taxonomy. Healthy distribution would cover all phases.
PHASE_KEYWORDS = {
    "approach": ("approach", "approaching", "advance", "advancing"),
    "reach": ("reach", "reaching", "extend", "extending"),
    "align": ("align", "aligning", "position over", "center over", "hover"),
    "grasp": ("grasp", "grasping", "grip", "gripping", "close gripper", "pick up", "pick-up", "pickup"),
    "lift": ("lift", "lifting", "raise", "raising", "elevate"),
    "carry": ("carry", "carrying", "transport", "transporting", "move toward", "move it toward"),
    "place": ("place", "placing", "deposit", "depositing", "set down", "lower onto", "put"),
    "release": ("release", "releasing", "open gripper", "let go", "drop"),
    "retract": ("retract", "retracting", "withdraw", "withdrawing", "pull back"),
    "idle": ("idle", "wait", "waiting", "hold", "holding", "stationary", "stay"),
    "open": ("open the drawer", "open the cabinet", "open the door", "open drawer", "open cabinet", "pull open"),
    "close": ("close the drawer", "close the cabinet", "close the door", "close drawer", "close cabinet", "push closed"),
    "pick-and-place": ("pick-and-place", "pick and place", "pick-up-and-place", "pickup-and-place"),
}

# Stopwords for the "content words" computation used by the filler heuristic.
STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "in", "on", "to", "for", "with", "at",
    "is", "are", "was", "were", "be", "been", "being", "by", "from", "into",
    "it", "its", "this", "that", "these", "those", "as", "or", "but", "if",
    "then", "so", "than", "such", "which", "what", "who", "whom", "whose",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "must", "near", "next", "over", "under",
    "above", "below", "between", "among", "out", "up", "down",
})

# Categories the bullet parser knows about (so an unrecognised "scen:" doesn't
# silently become content of the previous bullet). We only score the five
# scoped ones, but we recognise the broader set so the parser respects bullet
# boundaries.
ALL_BULLET_PATTERNS = (
    "language", "target", "scene", "spatial", "plan",
    "distractor", "motion", "gripper", "image_region",
)
BULLET_RE = re.compile(
    r"^\s*-\s*(" + "|".join(ALL_BULLET_PATTERNS) + r")\s*:\s*(.*)$",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenisation with basic punctuation stripping."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return text.split()


def content_words(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def parse_bullets(description: str) -> dict[str, list[str]]:
    """Return {bullet_category: [text, ...]} from a 4-5 bullet description.

    Same category appearing twice (e.g., two `- target:` lines) is preserved as
    a list, but downstream we concatenate within a row.
    """
    bullets: dict[str, list[str]] = defaultdict(list)
    if not description:
        return bullets
    for line in description.splitlines():
        m = BULLET_RE.match(line)
        if not m:
            continue
        cat = m.group(1).lower()
        body = m.group(2).strip().rstrip(".")
        if body:
            bullets[cat].append(body)
    return bullets


def concrete_noun_hit(text: str) -> bool:
    """True if the text contains at least one concrete noun/color from vocab."""
    if not text:
        return False
    lower = text.lower()
    for phrase in LIBERO_MULTIWORD:
        if phrase in lower:
            return True
    toks = set(tokenize(lower))
    return bool(toks & CONCRETE_VOCAB_SINGLE)


def detect_phases(text: str) -> list[str]:
    """Return list of phase tags found in this plan bullet (any keyword match)."""
    if not text:
        return []
    lower = text.lower()
    hits = []
    for phase, kws in PHASE_KEYWORDS.items():
        for kw in kws:
            if kw in lower:
                hits.append(phase)
                break
    return hits


def ngrams(tokens: list[str], n: int) -> Iterator[tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = q * (len(sorted_xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (idx - lo) * (sorted_xs[hi] - sorted_xs[lo])


# -----------------------------------------------------------------------------
# Streaming aggregator
# -----------------------------------------------------------------------------


class Stats:
    def __init__(self) -> None:
        # Per-suite per-bullet length lists.
        self.lengths: dict[str, dict[str, list[int]]] = {
            s: {b: [] for b in BULLETS} for s in SUITES
        }
        # Per-suite per-bullet presence counts.
        self.presence: dict[str, dict[str, int]] = {
            s: {b: 0 for b in BULLETS} for s in SUITES
        }
        self.row_counts: dict[str, int] = {s: 0 for s in SUITES}
        # Concrete-noun hit counts per suite per bullet.
        self.concrete: dict[str, dict[str, int]] = {
            s: {b: 0 for b in BULLETS} for s in SUITES
        }
        # Filler-bullet counts per suite per bullet.
        self.filler: dict[str, dict[str, int]] = {
            s: {b: 0 for b in BULLETS} for s in SUITES
        }
        # Cross-bullet Jaccard sums + counts per pair.
        self.jaccard_sum: dict[tuple[str, str], float] = {
            (a, b): 0.0 for a, b in combinations(BULLETS, 2)
        }
        self.jaccard_n: dict[tuple[str, str], int] = {
            (a, b): 0 for a, b in combinations(BULLETS, 2)
        }
        # Plan-phase counts per suite.
        self.phase_counts: dict[str, Counter] = {s: Counter() for s in SUITES}
        self.plan_total: dict[str, int] = {s: 0 for s in SUITES}
        # Position-type → bullet → unigram + bigram counters (for top n-grams).
        self.pos_unigrams: dict[str, dict[str, Counter]] = defaultdict(
            lambda: {b: Counter() for b in BULLETS}
        )
        self.pos_bigrams: dict[str, dict[str, Counter]] = defaultdict(
            lambda: {b: Counter() for b in BULLETS}
        )
        self.pos_bullet_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {b: 0 for b in BULLETS}
        )
        # Example collection for the well/poorly-grounded showcase.
        self.well_examples: list[dict] = []
        self.poor_examples: list[dict] = []

    def update(self, suite: str, record: dict) -> None:
        self.row_counts[suite] += 1
        desc = record.get("description") or ""
        meta = record.get("meta") or {}
        instruction = (meta.get("instruction") or "").strip()
        position_type = meta.get("position_type") or "unknown"
        bullets = parse_bullets(desc)

        # Per-bullet text (concatenate if same category appears multiple times).
        bullet_text: dict[str, str] = {b: " ".join(bullets.get(b, [])) for b in BULLETS}
        bullet_tokens: dict[str, list[str]] = {
            b: tokenize(bullet_text[b]) for b in BULLETS
        }
        bullet_token_sets: dict[str, set[str]] = {
            b: set(bullet_tokens[b]) for b in BULLETS
        }
        instr_content = content_words(tokenize(instruction))

        # Per-bullet tallies.
        for b in BULLETS:
            present = bool(bullets.get(b))
            if not present:
                continue
            self.presence[suite][b] += 1
            tlen = len(bullet_tokens[b])
            self.lengths[suite][b].append(tlen)
            if concrete_noun_hit(bullet_text[b]):
                self.concrete[suite][b] += 1
            # Filler: bullet covers >=80% of instruction's content words.
            if instr_content:
                bullet_content = content_words(bullet_tokens[b])
                covered = len(instr_content & bullet_content) / len(instr_content)
                if covered >= 0.8:
                    self.filler[suite][b] += 1
            # Position-type n-grams.
            self.pos_bullet_counts[position_type][b] += 1
            self.pos_unigrams[position_type][b].update(
                t for t in bullet_tokens[b] if t not in STOPWORDS and len(t) > 2
            )
            self.pos_bigrams[position_type][b].update(
                " ".join(g) for g in ngrams(bullet_tokens[b], 2)
            )

        # Cross-bullet Jaccard for bullets that are both present.
        for a, b in combinations(BULLETS, 2):
            sa = bullet_token_sets[a]
            sb = bullet_token_sets[b]
            if not sa or not sb:
                continue
            self.jaccard_sum[(a, b)] += jaccard(sa, sb)
            self.jaccard_n[(a, b)] += 1

        # Plan phase taxonomy (count UNIQUE phases per plan bullet, so a plan
        # describing multiple phases credits each once).
        plan_text = bullet_text.get("plan", "")
        if plan_text:
            self.plan_total[suite] += 1
            phases = set(detect_phases(plan_text))
            if not phases:
                self.phase_counts[suite]["(other)"] += 1
            else:
                for p in phases:
                    self.phase_counts[suite][p] += 1

        # Example showcase: well = all 5 bullets present, all 5 concrete-hit, no
        # filler. poor = >=2 bullets missing OR >=2 bullets are filler OR <=1
        # bullets concrete-hit.
        present_count = sum(1 for b in BULLETS if bullets.get(b))
        concrete_count = sum(
            1 for b in BULLETS if bullets.get(b) and concrete_noun_hit(bullet_text[b])
        )
        filler_count = 0
        if instr_content:
            for b in BULLETS:
                if bullets.get(b):
                    bullet_content = content_words(bullet_tokens[b])
                    covered = len(instr_content & bullet_content) / len(instr_content)
                    if covered >= 0.8:
                        filler_count += 1
        if (
            present_count == 5
            and concrete_count == 5
            and filler_count == 0
            and len(self.well_examples) < 5
        ):
            self.well_examples.append(
                {
                    "suite": suite,
                    "example_id": record.get("example_id"),
                    "instruction": instruction,
                    "description": desc.strip(),
                }
            )
        if (
            (present_count <= 3 or filler_count >= 2 or concrete_count <= 1)
            and len(self.poor_examples) < 5
        ):
            self.poor_examples.append(
                {
                    "suite": suite,
                    "example_id": record.get("example_id"),
                    "instruction": instruction,
                    "description": desc.strip(),
                    "present_count": present_count,
                    "concrete_count": concrete_count,
                    "filler_count": filler_count,
                }
            )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return " - "
    return f"{100.0 * num / denom:5.1f}%"


def length_row(stats: Stats, bullet: str) -> str:
    parts = []
    for suite in SUITES:
        xs = sorted(stats.lengths[suite][bullet])
        if not xs:
            parts.append("-/-/-/- ")
        else:
            p10 = quantile(xs, 0.10)
            p50 = quantile(xs, 0.50)
            p90 = quantile(xs, 0.90)
            avg = mean(xs)
            parts.append(f"{p10:.0f}/{p50:.0f}/{p90:.0f} (μ={avg:.1f})")
    return " | ".join(parts)


def build_report(stats: Stats) -> str:
    lines: list[str] = []
    lines.append("# V3 Bullet Informativeness Audit (Agent 4)\n")
    lines.append(
        "Per-bullet grounding, redundancy, and informativeness for the V3 "
        "5-bullet captions on LIBERO 4-suite (stride-2). Scope: the five "
        "scoped bullet types `language / target / scene / spatial / plan`. "
        "Other categories (`distractor`, `motion`, `gripper`, `image_region`) "
        "are recognised by the bullet parser so they do not bleed into "
        "scoped-bullet content, but are not reported here.\n"
    )
    total_rows = sum(stats.row_counts.values())
    lines.append(
        f"**Corpus**: {total_rows:,} labels across "
        f"{', '.join(f'{s}={stats.row_counts[s]:,}' for s in SUITES)}.\n"
    )

    # -- Length stats table --
    lines.append("## 1. Length, presence, concrete-noun (per suite)\n")
    lines.append(
        "Length cells are `p10/p50/p90 (μ=mean)` whitespace tokens, including "
        "the bullet header words.\n"
    )
    lines.append(
        "| bullet | "
        + " | ".join(f"len[{s}]" for s in SUITES)
        + " | presence | concrete-noun |"
    )
    lines.append("|---|" + "|".join("---" for _ in SUITES) + "|---|---|")
    for b in BULLETS:
        total_present = sum(stats.presence[s][b] for s in SUITES)
        total_concrete = sum(stats.concrete[s][b] for s in SUITES)
        pres = fmt_pct(total_present, total_rows)
        conc = (
            fmt_pct(total_concrete, total_present)
            if total_present
            else " - "
        )
        lines.append(f"| **{b}** | {length_row(stats, b)} | {pres} | {conc} |")
    lines.append("")
    # Highlight presence anomalies (e.g., a bullet that the labeler frequently skips).
    low_presence: list[tuple[str, float]] = []
    for b in BULLETS:
        total_present = sum(stats.presence[s][b] for s in SUITES)
        rate = total_present / total_rows if total_rows else 0.0
        if rate < 0.80:
            low_presence.append((b, rate))
    if low_presence:
        lines.append("")
        lines.append(
            "**Presence anomaly**: the following bullets are missing from a "
            "non-trivial share of labels, suggesting the prompt template does "
            "not require them strongly enough:"
        )
        for b, rate in sorted(low_presence, key=lambda kv: kv[1]):
            lines.append(f"- `{b}` present in only {rate*100:.1f}% of rows")
        lines.append("")
    lines.append("Per-suite concrete-noun rate (denominator = present bullets):\n")
    lines.append("| bullet | " + " | ".join(SUITES) + " |")
    lines.append("|---|" + "|".join("---" for _ in SUITES) + "|")
    for b in BULLETS:
        cells = []
        for s in SUITES:
            cells.append(fmt_pct(stats.concrete[s][b], stats.presence[s][b]))
        lines.append(f"| **{b}** | " + " | ".join(cells) + " |")
    lines.append("")

    # -- Filler --
    lines.append("## 2. Filler-bullet rate (bullet covers ≥80% of instruction content words)\n")
    lines.append(
        "Filler means the bullet's content-word set covers the instruction's "
        "content-word set almost completely — i.e., it just paraphrases the "
        "task text without adding visual or plan information. Computed only on "
        "rows where the instruction is non-empty.\n"
    )
    lines.append("| bullet | " + " | ".join(SUITES) + " | overall |")
    lines.append("|---|" + "|".join("---" for _ in SUITES) + "|---|")
    for b in BULLETS:
        cells = []
        total_f = 0
        total_p = 0
        for s in SUITES:
            cells.append(fmt_pct(stats.filler[s][b], stats.presence[s][b]))
            total_f += stats.filler[s][b]
            total_p += stats.presence[s][b]
        cells.append(fmt_pct(total_f, total_p))
        lines.append(f"| **{b}** | " + " | ".join(cells) + " |")
    lines.append("")

    # -- Cross-bullet Jaccard 5x5 --
    lines.append("## 3. Cross-bullet Jaccard redundancy (mean over rows where both bullets present)\n")
    lines.append(
        "Token-set Jaccard on whitespace tokens (after lowercase, punctuation "
        "strip). Diagonals are 1.00 by definition; only upper triangle is "
        "computed and mirrored.\n"
    )
    pair_mean: dict[tuple[str, str], float] = {}
    for (a, b), s in stats.jaccard_sum.items():
        n = stats.jaccard_n[(a, b)]
        pair_mean[(a, b)] = s / n if n else 0.0

    header = "| | " + " | ".join(BULLETS) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join("---" for _ in BULLETS) + "|")
    for r in BULLETS:
        row = [f"**{r}**"]
        for c in BULLETS:
            if r == c:
                row.append("1.00")
            else:
                key = (r, c) if (r, c) in pair_mean else (c, r)
                row.append(f"{pair_mean[key]:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    # Redundancy flags (>0.4).
    flagged = sorted(
        (((a, b), v) for (a, b), v in pair_mean.items()),
        key=lambda x: -x[1],
    )
    lines.append("Pairs ranked by mean Jaccard (descending):\n")
    lines.append("| pair | mean Jaccard | flag (>0.4) |")
    lines.append("|---|---|---|")
    for (a, b), v in flagged:
        flag = "REDUNDANT" if v > 0.4 else ""
        lines.append(f"| {a} ↔ {b} | {v:.3f} | {flag} |")
    lines.append("")

    # -- Plan phase --
    lines.append("## 4. Plan-bullet phase taxonomy (per suite)\n")
    lines.append(
        "Each plan bullet is matched against a curated keyword list; a single "
        "bullet may contribute to multiple phases (e.g., `pick-and-place phase "
        "active; reach over the bowl, then place` → `pick-and-place`, `reach`, "
        "`place`). `(other)` = no keyword matched.\n"
    )
    lines.append("| suite | plan-bullets | top phases (count, %) |")
    lines.append("|---|---|---|")
    for s in SUITES:
        total = stats.plan_total[s]
        top = stats.phase_counts[s].most_common(10)
        if not top:
            lines.append(f"| {s} | 0 | - |")
            continue
        rendered = "; ".join(
            f"{ph} ({cnt}, {100.0*cnt/total:.0f}%)" for ph, cnt in top
        )
        lines.append(f"| {s} | {total} | {rendered} |")
    lines.append("")

    # Overall top-phase share across all suites combined.
    overall_phase = Counter()
    overall_plan = 0
    for s in SUITES:
        overall_phase.update(stats.phase_counts[s])
        overall_plan += stats.plan_total[s]
    lines.append("Overall plan-phase share (all suites combined):\n")
    lines.append("| phase | count | share |")
    lines.append("|---|---|---|")
    for ph, cnt in overall_phase.most_common(15):
        lines.append(f"| {ph} | {cnt} | {100.0*cnt/overall_plan:.1f}% |")
    lines.append("")

    # -- Position-type sensitivity --
    lines.append("## 5. Position-type sensitivity (image_patch vs last_text)\n")
    lines.append(
        "Top content unigrams per bullet for each position type. If "
        "`image_patch` bullets do their job (visually-grounded), their top "
        "n-grams should be more concrete (objects, colors, parts) than the "
        "matched `last_text` bullets.\n"
    )
    pos_types = [pt for pt in ("last_text", "image_patch", "anchor") if pt in stats.pos_bullet_counts]
    counts_summary = ", ".join(
        f"{pt}={sum(stats.pos_bullet_counts[pt].values())}" for pt in pos_types
    )
    lines.append(f"Per-position bullet counts (summed across bullets): {counts_summary}.\n")

    if "last_text" in pos_types and "image_patch" in pos_types:
        lines.append(
            "**Quantitative summary**: Jaccard overlap of top-N content "
            "unigrams between `last_text` and `image_patch` per bullet. "
            "1.00 = identical vocabulary; high values mean the labeler is "
            "producing similar text regardless of position type.\n"
        )
        lines.append("| bullet | top-10 Jaccard | top-30 Jaccard |")
        lines.append("|---|---|---|")
        for b in BULLETS:
            lt10 = {tok for tok, _ in stats.pos_unigrams["last_text"][b].most_common(10)}
            ip10 = {tok for tok, _ in stats.pos_unigrams["image_patch"][b].most_common(10)}
            lt30 = {tok for tok, _ in stats.pos_unigrams["last_text"][b].most_common(30)}
            ip30 = {tok for tok, _ in stats.pos_unigrams["image_patch"][b].most_common(30)}
            lines.append(
                f"| **{b}** | {jaccard(lt10, ip10):.2f} | {jaccard(lt30, ip30):.2f} |"
            )
        lines.append("")

    for b in BULLETS:
        lines.append(f"### `{b}` bullet — top 10 content unigrams\n")
        lines.append("| rank | " + " | ".join(pos_types) + " |")
        lines.append("|---|" + "|".join("---" for _ in pos_types) + "|")
        tops = {
            pt: stats.pos_unigrams[pt][b].most_common(10) for pt in pos_types
        }
        max_rank = max((len(t) for t in tops.values()), default=0)
        for i in range(max_rank):
            row = [str(i + 1)]
            for pt in pos_types:
                if i < len(tops[pt]):
                    tok, cnt = tops[pt][i]
                    row.append(f"{tok} ({cnt})")
                else:
                    row.append("")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # Concrete-vocab share among top-30 unigrams per position type.
        shares = []
        for pt in pos_types:
            top30 = stats.pos_unigrams[pt][b].most_common(30)
            if not top30:
                shares.append(f"{pt}: n/a")
                continue
            conc_hits = sum(
                1 for tok, _ in top30 if tok in CONCRETE_VOCAB_SINGLE
            )
            shares.append(f"{pt}: {conc_hits}/30 concrete")
        lines.append("Concrete-vocab share in top-30 unigrams — " + "; ".join(shares) + "\n")

    # -- Showcase examples --
    lines.append("## 6. Examples\n")
    lines.append("### Well-grounded (all 5 bullets present, all concrete, no filler)\n")
    if not stats.well_examples:
        lines.append("_None matched the strict criteria._\n")
    for ex in stats.well_examples:
        lines.append(f"- **{ex['example_id']}** ({ex['suite']}) — instruction: _{ex['instruction']}_")
        for line in ex["description"].splitlines():
            lines.append(f"    {line}")
        lines.append("")
    lines.append("### Poorly-grounded (≤3 bullets, or ≥2 filler bullets, or ≤1 concrete-hit)\n")
    if not stats.poor_examples:
        lines.append("_None matched the criteria._\n")
    for ex in stats.poor_examples:
        lines.append(
            f"- **{ex['example_id']}** ({ex['suite']}) — "
            f"present={ex['present_count']} concrete={ex['concrete_count']} "
            f"filler={ex['filler_count']} — instruction: _{ex['instruction']}_"
        )
        for line in ex["description"].splitlines():
            lines.append(f"    {line}")
        lines.append("")

    # -- Verdict --
    lines.append("## 7. Verdict\n")
    issues: list[str] = []
    # Concrete-noun rate per bullet.
    bullet_conc_rate = {}
    for b in BULLETS:
        tp = sum(stats.presence[s][b] for s in SUITES)
        tc = sum(stats.concrete[s][b] for s in SUITES)
        rate = tc / tp if tp else 0.0
        bullet_conc_rate[b] = rate
    overall_conc = mean(bullet_conc_rate.values())
    low_conc = [b for b, r in bullet_conc_rate.items() if r < 0.60]
    if low_conc:
        issues.append(
            f"bullets below 60% concrete-noun: {', '.join(f'{b}={bullet_conc_rate[b]*100:.0f}%' for b in low_conc)}"
        )
    # Redundancy.
    high_red = [((a, b), v) for (a, b), v in pair_mean.items() if v > 0.4]
    if high_red:
        issues.append(
            "Jaccard >0.4 pairs: "
            + ", ".join(f"{a}↔{b}={v:.2f}" for (a, b), v in high_red)
        )
    # Plan phase concentration.
    top_phase = overall_phase.most_common(1)
    top_phase_share = (
        100.0 * top_phase[0][1] / overall_plan if top_phase and overall_plan else 0.0
    )
    if top_phase and top_phase_share > 40.0:
        issues.append(
            f"plan-phase concentrated: '{top_phase[0][0]}' = {top_phase_share:.0f}%"
        )

    if len(issues) == 0:
        verdict = "GREEN"
    elif len(issues) <= 2:
        verdict = "YELLOW"
    else:
        verdict = "RED"
    lines.append(f"**Verdict: {verdict}** ({len(issues)} issue(s))")
    if issues:
        for i in issues:
            lines.append(f"- {i}")
    else:
        lines.append(
            "- All scoped bullets ≥60% concrete-noun, no bullet-pair Jaccard "
            "exceeded 0.4, and top plan phase below 40%."
        )
    lines.append("")
    lines.append(
        f"**Overall concrete-noun rate (mean across 5 bullets)**: "
        f"{overall_conc*100:.1f}%"
    )
    lines.append("")

    # -- Recommendations --
    lines.append("## 8. Top recommendations for prompt tightening\n")
    recs = []
    # Build data-driven recommendations.
    # rec 1: address most redundant pair.
    if high_red:
        a, b = high_red[0][0]
        v = high_red[0][1]
        recs.append(
            f"**Differentiate `{a}` and `{b}` bullets.** Mean Jaccard "
            f"{v:.2f} > 0.4: the two bullets share most of their tokens and "
            f"are doing the same job. Concrete fix: change the prompt clause "
            f"for `{a}` to require a property the `{b}` bullet does NOT carry "
            f"(e.g., `{a}` = instruction parsing / next-step intent; `{b}` = "
            f"current visible object + color + state)."
        )
    # rec 2: address filler bullet
    worst_filler_bullet = None
    worst_filler_rate = 0.0
    for b in BULLETS:
        tp = sum(stats.presence[s][b] for s in SUITES)
        tf = sum(stats.filler[s][b] for s in SUITES)
        rate = tf / tp if tp else 0.0
        if rate > worst_filler_rate:
            worst_filler_rate = rate
            worst_filler_bullet = b
    if worst_filler_bullet and worst_filler_rate > 0.10:
        recs.append(
            f"**Force `{worst_filler_bullet}` to add information beyond the instruction.** "
            f"{worst_filler_rate*100:.0f}% of `{worst_filler_bullet}` bullets cover ≥80% "
            f"of the instruction's content words — i.e., they are paraphrases of the task "
            f"text. Add a prompt rule: '`{worst_filler_bullet}` must include at least one "
            f"property NOT in the instruction text (color, spatial relation, gripper state, "
            f"or next-step verb).'"
        )
    # rec 3: plan-phase diversity if top phase >40%.
    if top_phase and top_phase_share > 40.0:
        recs.append(
            f"**Diversify plan-phase taxonomy.** Plan bullets are dominated by "
            f"'{top_phase[0][0]}' ({top_phase_share:.0f}%). The labeler isn't "
            f"tracking step-within-episode. Either (a) give the labeler the "
            f"step index out of total, or (b) explicitly enumerate phases in the "
            f"prompt (approach / grasp / lift / carry / place / release / retract / idle) "
            f"and ask which one is active NOW."
        )
    # rec 4: low-concrete bullets.
    if low_conc:
        recs.append(
            "**Boost concreteness of "
            + ", ".join(f"`{b}`" for b in low_conc)
            + " bullets.** These are below the 60% concrete-noun threshold. "
            "Append to the prompt: 'every bullet must reference at least one "
            "concrete object name (mug, bowl, plate, bottle, drawer, ...) or "
            "color, except for the `language` bullet which may quote the "
            "instruction verbatim.'"
        )
    # rec 5: tighten position-type behaviour, citing the top-30 Jaccard.
    pos_jaccards: dict[str, float] = {}
    if "last_text" in stats.pos_bullet_counts and "image_patch" in stats.pos_bullet_counts:
        for b in BULLETS:
            lt30 = {t for t, _ in stats.pos_unigrams["last_text"][b].most_common(30)}
            ip30 = {t for t, _ in stats.pos_unigrams["image_patch"][b].most_common(30)}
            pos_jaccards[b] = jaccard(lt30, ip30)
    high_pos_overlap = [b for b, v in pos_jaccards.items() if v > 0.6 and b != "language"]
    if high_pos_overlap:
        cited = ", ".join(
            f"`{b}` top-30 Jaccard={pos_jaccards[b]:.2f}" for b in high_pos_overlap
        )
        recs.append(
            "**Make `image_patch` bullets visually distinct from `last_text` bullets.** "
            f"Top-30 unigram Jaccard between `last_text` and `image_patch` is very high for: {cited} "
            "(see \u00a75) -- the labeler is producing essentially the same caption "
            "regardless of which token is highlighted. Add a prompt rule that, for "
            "`image_patch` positions, the relevant bullet(s) must describe the object/region "
            "visible *at the gripper or directly under the current frame attention*, while "
            "for `last_text` positions the same bullets describe the parsed plan phase or "
            "instruction-level intent."
        )
    elif not recs or len(recs) < 3:
        recs.append(
            "**Make `image_patch` bullets visually distinct from `last_text` bullets.** "
            "Compare position-type top n-grams (\u00a75): if they look near-identical, "
            "the labeler is producing the same caption regardless of which token is "
            "highlighted. Add a prompt rule that, for `image_patch` positions, the "
            "last bullet must describe a visible object/region at the gripper, while "
            "for `last_text` positions it must describe the parsed plan phase."
        )

    for i, r in enumerate(recs[:3], 1):
        lines.append(f"{i}. {r}\n")

    # -- Coordination notes --
    lines.append("## 9. Coordination notes\n")
    lines.append(
        "- **Agent 3 (diversity)**: if they flag a boilerplate plan phrase "
        "(e.g., `pickup-and-place phase active`) appearing in a large share "
        "of plan bullets, that complements §4 here — plan-phase concentration "
        "and plan-bullet boilerplate are the same failure mode.\n"
        "- **Agent 2 (forbidden phrasing)**: our filler detection (§2) is "
        "orthogonal to their phrasing scan. A bullet can be filler without "
        "tripping forbidden phrasing, and vice versa.\n"
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    cwd = Path.cwd()
    labels_root = (cwd / LABELS_ROOT).resolve()
    if not labels_root.exists():
        print(f"ERROR: labels root not found: {labels_root}", file=sys.stderr)
        return 1

    stats = Stats()
    for suite in SUITES:
        path = labels_root / suite / "labels.jsonl"
        if not path.exists():
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        for rec in iter_jsonl(path):
            stats.update(suite, rec)
        print(
            f"[{suite}] {stats.row_counts[suite]:,} rows; "
            f"presence: "
            + ", ".join(
                f"{b}={stats.presence[suite][b]}" for b in BULLETS
            ),
            file=sys.stderr,
        )

    out_path = (cwd / OUT_PATH).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(stats)
    out_path.write_text(report)
    print(f"\nReport written to: {out_path}", file=sys.stderr)

    # --- 5-line stdout summary for the parent agent ---
    total_rows = sum(stats.row_counts.values())
    # concrete-noun rate (mean across 5 bullets, weighted by presence)
    bullet_conc_rate = {}
    for b in BULLETS:
        tp = sum(stats.presence[s][b] for s in SUITES)
        tc = sum(stats.concrete[s][b] for s in SUITES)
        bullet_conc_rate[b] = (tc / tp) if tp else 0.0
    overall_conc = mean(bullet_conc_rate.values())
    # Most redundant pair
    pair_mean = {
        k: (v / stats.jaccard_n[k]) if stats.jaccard_n[k] else 0.0
        for k, v in stats.jaccard_sum.items()
    }
    worst_pair, worst_jac = max(pair_mean.items(), key=lambda kv: kv[1])
    # Top plan phase
    overall_phase = Counter()
    overall_plan = 0
    for s in SUITES:
        overall_phase.update(stats.phase_counts[s])
        overall_plan += stats.plan_total[s]
    top_phase, top_phase_cnt = overall_phase.most_common(1)[0]
    top_phase_share = 100.0 * top_phase_cnt / overall_plan
    # Filler%: overall worst bullet
    worst_filler_bullet = None
    worst_filler_rate = 0.0
    overall_filler = 0
    overall_present = 0
    for b in BULLETS:
        tp = sum(stats.presence[s][b] for s in SUITES)
        tf = sum(stats.filler[s][b] for s in SUITES)
        overall_filler += tf
        overall_present += tp
        rate = (tf / tp) if tp else 0.0
        if rate > worst_filler_rate:
            worst_filler_rate = rate
            worst_filler_bullet = b
    overall_filler_rate = (overall_filler / overall_present) if overall_present else 0.0
    # Verdict
    issues = 0
    if any(r < 0.60 for r in bullet_conc_rate.values()):
        issues += 1
    if any(v > 0.4 for v in pair_mean.values()):
        issues += 1
    if top_phase_share > 40.0:
        issues += 1
    verdict = "GREEN" if issues == 0 else ("YELLOW" if issues <= 2 else "RED")

    summary_lines = [
        f"mean_concrete_noun_rate={overall_conc*100:.1f}%",
        f"most_redundant_pair={worst_pair[0]}<->{worst_pair[1]} jaccard={worst_jac:.2f}",
        f"top_plan_phase={top_phase} share={top_phase_share:.1f}%",
        f"filler_overall={overall_filler_rate*100:.1f}% worst_bullet={worst_filler_bullet}={worst_filler_rate*100:.1f}%",
        f"verdict={verdict} ({issues} issue(s); rows={total_rows:,})",
    ]
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Position-type disambiguation audit for warm-start LIBERO labels.

Measures how distinguishable ``image_patch`` captions are from ``last_text``
captions in any labels.jsonl corpus. Used as the V4-vs-V3 regression gate
for the LIBERO dataset repair (SA4 of the V4 repair plan).

The V3 audit (`docs/sft_plan/audit_reports/agent4_bullet_informativeness.md`,
§5) found top-30 unigram Jaccard between ``last_text`` and ``image_patch``
captions of 0.71-0.76 on the ``target:`` and ``scene:`` bullets — meaning
the labeler produced near-identical captions regardless of which token
position was highlighted, killing the AV's ability to learn
position-specific content. V4's fix is the position-type-conditional last
bullet (``image_patch`` -> perceptual; ``last_text`` -> temporal plan).
This script measures whether the fix is working.

Metrics
-------

1. **Top-30 unigram Jaccard** between the pooled ``image_patch`` vs pooled
   ``last_text`` token sets, per bullet type (``target``, ``scene``,
   ``spatial``, ``plan``). Mirrors Agent 4's V3 numbers (same tokenization
   conventions as ``audit_bullet_informativeness.py``).
2. **Mean pairwise Jaccard** between random pairs sampled across ptype
   (``image_patch`` row vs ``last_text`` row from same suite, same
   ``source_example_id`` whenever possible — falls back to same-suite if
   no matching source row exists). Sample 2000 such pairs; report mean +
   p10/p50/p90.
3. **Per-bullet-type token entropy** within each ptype (low-entropy ptype
   is templated; high is diverse).
4. **Last-bullet category mix**: per ptype, what % of last bullets are
   ``target:`` vs ``scene:`` vs ``plan:`` vs ``spatial:``? V4 expects
   ``image_patch`` -> mostly ``target/scene``, ``last_text`` -> mostly
   ``plan``.

Verdict
-------

- **GREEN**: every suite + bullet has top-30 Jaccard <= 0.45 AND
  ``image_patch`` last_bullet_mix['plan'] <= 30% AND ``last_text``
  last_bullet_mix['plan'] >= 60%.
- **YELLOW**: 1-2 cells violate the above.
- **RED**: >=3 cells violate.

Usage
-----

::

    PYTHONPATH=src python scripts/eval/audit_ptype_disambiguation.py \\
        --labels-root data/labels/libero_4suite_stride2 \\
        --out-json data/eval/ptype_jaccard_summary.json \\
        --out-md /tmp/ptype_jaccard.md \\
        [--suite libero_spatial]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterator


# -----------------------------------------------------------------------------
# Tokenization (mirror audit_bullet_informativeness.py exactly)
# -----------------------------------------------------------------------------

BULLETS = ("target", "scene", "spatial", "plan")
"""Bullet types we measure Jaccard / entropy on. We exclude ``language``
because V4 makes it OPTIONAL on image_patch/anchor (so its presence rate
is a separate axis, not a disambiguation axis)."""

# Same broader category list as the V3 audit so we recognise (and respect
# the boundaries of) every category a V3 row could carry, even ones V4
# removes from the allowed set.
ALL_BULLET_PATTERNS = (
    "language", "target", "scene", "spatial", "plan",
    "distractor", "motion", "gripper", "image_region",
)
BULLET_RE = re.compile(
    r"^\s*-\s*(" + "|".join(ALL_BULLET_PATTERNS) + r")\s*:\s*(.*)$",
    re.IGNORECASE,
)

# Same stopword list as audit_bullet_informativeness.py so top-30 numbers
# are directly comparable to Agent 4's V3 report.
STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "in", "on", "to", "for", "with", "at",
    "is", "are", "was", "were", "be", "been", "being", "by", "from", "into",
    "it", "its", "this", "that", "these", "those", "as", "or", "but", "if",
    "then", "so", "than", "such", "which", "what", "who", "whom", "whose",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "must", "near", "next", "over", "under",
    "above", "below", "between", "among", "out", "up", "down",
})


def tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenisation, mirrors V3 audit."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return text.split()


def content_tokens(tokens: list[str]) -> list[str]:
    """Drop stopwords and tokens shorter than 3 chars (matches the V3
    audit's top-N convention)."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def parse_bullets(description: str) -> dict[str, list[str]]:
    """Return {bullet_category: [text, ...]} from a multi-bullet caption.

    Same parser as ``audit_bullet_informativeness.py`` so n-gram counts and
    Jaccard numbers are directly comparable to Agent 4's V3 baseline.
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


def last_bullet_category(description: str) -> str | None:
    """Return the category name of the LAST bullet in this caption.

    Used to measure the last-bullet mix per ptype (V4 expects image_patch
    -> mostly target/scene, last_text -> mostly plan).
    """
    if not description:
        return None
    last: str | None = None
    for line in description.splitlines():
        m = BULLET_RE.match(line)
        if not m:
            continue
        last = m.group(1).lower()
    return last


def quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = q * (len(sorted_xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (idx - lo) * (sorted_xs[hi] - sorted_xs[lo])


def entropy_bits(counter: Counter) -> float:
    """Shannon entropy in bits over a token frequency distribution."""
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# -----------------------------------------------------------------------------
# Per-suite accumulator
# -----------------------------------------------------------------------------


class SuiteAcc:
    """Per-suite aggregator. We pool tokens per (ptype, bullet) for the
    top-30 Jaccard and entropy; we also retain a per-row token-set list per
    (ptype, bullet) for the pairwise-Jaccard sampler."""

    PTYPES = ("image_patch", "last_text", "anchor", "fallback")

    def __init__(self) -> None:
        # (ptype, bullet) -> Counter of content unigrams.
        self.unigrams: dict[tuple[str, str], Counter] = {
            (pt, b): Counter() for pt in self.PTYPES for b in BULLETS
        }
        # (ptype, bullet) -> int count of rows where that bullet was
        # present (used as the denominator for last_bullet_mix etc.).
        self.bullet_present: dict[tuple[str, str], int] = {
            (pt, b): 0 for pt in self.PTYPES for b in BULLETS
        }
        # Per (ptype) -> per source_example_id -> per bullet -> token set.
        # Used by the pairwise-Jaccard sampler so we can pair an
        # image_patch row to a last_text row from the same trajectory step.
        self.per_row_tokens: dict[
            str, dict[str, dict[str, set[str]]]
        ] = {pt: defaultdict(dict) for pt in self.PTYPES}
        # Per ptype -> Counter of last-bullet categories.
        self.last_bullet_categories: dict[str, Counter] = {
            pt: Counter() for pt in self.PTYPES
        }
        # Per ptype -> total rows seen.
        self.ptype_rows: Counter = Counter()
        self.total_rows: int = 0

    def update(self, record: dict) -> None:
        desc = record.get("description") or ""
        meta = record.get("meta") or {}
        ptype = (meta.get("position_type") or "").strip()
        if ptype not in self.PTYPES:
            return
        src = meta.get("source_example_id") or record.get("example_id") or ""
        bullets = parse_bullets(desc)

        self.total_rows += 1
        self.ptype_rows[ptype] += 1

        # Last-bullet category for the mix table.
        last_cat = last_bullet_category(desc)
        if last_cat is not None:
            self.last_bullet_categories[ptype][last_cat] += 1

        # Pool tokens per (ptype, bullet) AND remember per-row token sets
        # for the pairwise sampler.
        for b in BULLETS:
            if not bullets.get(b):
                continue
            bullet_text = " ".join(bullets[b])
            toks = content_tokens(tokenize(bullet_text))
            if not toks:
                continue
            self.bullet_present[(ptype, b)] += 1
            self.unigrams[(ptype, b)].update(toks)
            self.per_row_tokens[ptype][src][b] = set(toks)


# -----------------------------------------------------------------------------
# Pairwise sampling
# -----------------------------------------------------------------------------


def _sample_pair_jaccards(
    suite_acc: SuiteAcc, n_pairs: int, rng: random.Random
) -> dict[str, list[float]]:
    """Sample ``n_pairs`` (image_patch row, last_text row) pairs from the
    accumulator.

    Strategy:
      * Prefer pairs with the SAME ``source_example_id`` (same trajectory
        step). The V3 corpus tends to have image_patch and last_text rows
        per source step, so most pairs will be matched.
      * Fallback: pick a random image_patch row + a random last_text row
        from anywhere in the suite. This still measures ptype-vocabulary
        overlap; it just averages over uninvolved scenes.

    Returns: ``{bullet -> [pairwise_jaccard, ...]}`` lists, ready for
    mean/quantile reduction.
    """
    by_bullet: dict[str, list[float]] = {b: [] for b in BULLETS}

    ip_rows = list(suite_acc.per_row_tokens["image_patch"].items())
    lt_rows = list(suite_acc.per_row_tokens["last_text"].items())
    if not ip_rows or not lt_rows:
        return by_bullet

    lt_by_src = dict(lt_rows)
    matched_sources = [src for src, _ in ip_rows if src in lt_by_src]

    # Decide how many pairs to draw "matched" vs "random".
    # We try to fill matched first, then random for the rest.
    n_matched = min(n_pairs, len(matched_sources) * 5)  # at most 5x reuse

    drawn = 0
    while drawn < n_matched:
        src = rng.choice(matched_sources)
        ip_b = suite_acc.per_row_tokens["image_patch"][src]
        lt_b = lt_by_src[src]
        for b in BULLETS:
            if b in ip_b and b in lt_b:
                by_bullet[b].append(jaccard(ip_b[b], lt_b[b]))
        drawn += 1

    # Random fallback.
    while drawn < n_pairs:
        ip_src, ip_b = rng.choice(ip_rows)
        lt_src, lt_b = rng.choice(lt_rows)
        for b in BULLETS:
            if b in ip_b and b in lt_b:
                by_bullet[b].append(jaccard(ip_b[b], lt_b[b]))
        drawn += 1

    return by_bullet


# -----------------------------------------------------------------------------
# Summarisation
# -----------------------------------------------------------------------------


def _top30_jaccard(suite_acc: SuiteAcc) -> dict[str, float]:
    """Return {bullet: top-30 unigram Jaccard between image_patch and
    last_text pooled tokens for that bullet}."""
    out: dict[str, float] = {}
    for b in BULLETS:
        ip30 = {t for t, _ in suite_acc.unigrams[("image_patch", b)].most_common(30)}
        lt30 = {t for t, _ in suite_acc.unigrams[("last_text", b)].most_common(30)}
        out[b] = jaccard(ip30, lt30)
    return out


def _entropy_per_ptype_bullet(suite_acc: SuiteAcc) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for pt in suite_acc.PTYPES:
        ent: dict[str, float] = {}
        for b in BULLETS:
            ent[b] = entropy_bits(suite_acc.unigrams[(pt, b)])
        out[pt] = ent
    return out


def _last_bullet_mix(suite_acc: SuiteAcc) -> dict[str, dict[str, float]]:
    """Return ``{ptype: {category: pct}}`` for the LAST bullet of each
    caption, restricted to the 4 categories we care about (target, scene,
    plan, spatial)."""
    out: dict[str, dict[str, float]] = {}
    cats = ("target", "scene", "plan", "spatial")
    for pt in suite_acc.PTYPES:
        cnt = suite_acc.last_bullet_categories[pt]
        total = sum(cnt[c] for c in cats)  # ignore "language", etc.
        if total == 0:
            out[pt] = {c: 0.0 for c in cats}
            continue
        out[pt] = {c: 100.0 * cnt[c] / total for c in cats}
    return out


def _pair_summary(
    pair_jaccs: dict[str, list[float]],
) -> dict[str, float]:
    """Flatten {bullet: [j, ...]} into mean/p10/p50/p90 over all pairs."""
    pooled: list[float] = []
    for vals in pair_jaccs.values():
        pooled.extend(vals)
    pooled.sort()
    if not pooled:
        return {
            "n_pairs": 0, "mean": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
        }
    return {
        "n_pairs": len(pooled),
        "mean": mean(pooled),
        "p10": quantile(pooled, 0.10),
        "p50": quantile(pooled, 0.50),
        "p90": quantile(pooled, 0.90),
    }


def _violation_count(
    top30: dict[str, float],
    last_mix: dict[str, dict[str, float]],
) -> int:
    """Count cells violating V4 disambiguation targets:
      * any bullet (target/scene/spatial/plan) top-30 Jaccard > 0.45 -> 1 cell each.
      * image_patch last_bullet_mix['plan'] > 30 -> 1 cell.
      * last_text  last_bullet_mix['plan'] < 60 -> 1 cell.
    Returns total cell count.
    """
    n = 0
    for v in top30.values():
        if v > 0.45:
            n += 1
    if last_mix.get("image_patch", {}).get("plan", 0.0) > 30.0:
        n += 1
    if last_mix.get("last_text", {}).get("plan", 0.0) < 60.0:
        n += 1
    return n


def _verdict_from_violations(violations: int) -> str:
    if violations == 0:
        return "GREEN"
    if violations <= 2:
        return "YELLOW"
    return "RED"


# -----------------------------------------------------------------------------
# Summary builders
# -----------------------------------------------------------------------------


def _suite_summary(suite_acc: SuiteAcc, n_pairs: int, seed: int) -> dict:
    top30 = _top30_jaccard(suite_acc)
    pair_jaccs = _sample_pair_jaccards(suite_acc, n_pairs, random.Random(seed))
    pair_stats = _pair_summary(pair_jaccs)
    last_mix = _last_bullet_mix(suite_acc)
    entropy = _entropy_per_ptype_bullet(suite_acc)
    n_violations = _violation_count(top30, last_mix)
    verdict = _verdict_from_violations(n_violations)

    return {
        "n_rows": suite_acc.total_rows,
        "n_rows_per_ptype": dict(suite_acc.ptype_rows),
        "top30_jaccard": {b: round(v, 4) for b, v in top30.items()},
        "mean_pairwise_jaccard": round(pair_stats["mean"], 4),
        "p10_pairwise_jaccard": round(pair_stats["p10"], 4),
        "p50_pairwise_jaccard": round(pair_stats["p50"], 4),
        "p90_pairwise_jaccard": round(pair_stats["p90"], 4),
        "n_pairwise_jaccard": pair_stats["n_pairs"],
        "last_bullet_mix": {
            pt: {c: round(v, 2) for c, v in m.items()}
            for pt, m in last_mix.items()
        },
        "token_entropy_bits": {
            pt: {b: round(v, 3) for b, v in inner.items()}
            for pt, inner in entropy.items()
        },
        "violations": n_violations,
        "verdict": verdict,
    }


def _overall_acc(per_suite: dict[str, SuiteAcc]) -> SuiteAcc:
    """Concatenate per-suite accumulators into a single combined one. We
    rebuild from the per-row token sets so the overall pairwise sampler
    can also draw cross-suite pairs from a single pool."""
    overall = SuiteAcc()
    for suite, acc in per_suite.items():
        overall.total_rows += acc.total_rows
        overall.ptype_rows.update(acc.ptype_rows)
        for key, c in acc.unigrams.items():
            overall.unigrams[key].update(c)
        for key, n in acc.bullet_present.items():
            overall.bullet_present[key] += n
        for pt, by_src in acc.per_row_tokens.items():
            # Prefix sources with suite to avoid clobbering keys.
            for src, by_b in by_src.items():
                overall.per_row_tokens[pt][f"{suite}::{src}"] = by_b
        for pt, cnt in acc.last_bullet_categories.items():
            overall.last_bullet_categories[pt].update(cnt)
    return overall


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x:5.1f}%"


def _build_markdown(summary: dict, labels_root: Path, suite_filter: str | None) -> str:
    lines: list[str] = []
    lines.append("# Position-type disambiguation audit\n")
    lines.append(
        f"**Labels root**: `{labels_root}`  \n"
        f"**Suite filter**: `{suite_filter or '(all suites)'}`  \n"
        f"**Targets** (V4 GREEN): top-30 Jaccard ≤ 0.45 for every (suite, "
        "bullet); image_patch last_bullet_mix['plan'] ≤ 30%; "
        "last_text last_bullet_mix['plan'] ≥ 60%.\n"
    )

    overall = summary.get("overall", {})
    if overall:
        lines.append(
            f"**Overall verdict**: **{overall.get('verdict','?')}** "
            f"({overall.get('violations','?')} cell(s) violating); "
            f"corpus {overall.get('n_rows','?'):,} rows.\n"
        )

    # -- Top-30 Jaccard table --
    per_suite = summary.get("per_suite", {})
    if per_suite:
        suites_present = sorted(per_suite.keys())
        lines.append("\n## 1. Top-30 unigram Jaccard (image_patch vs last_text)\n")
        lines.append("Pooled per (suite, bullet). Lower is better; V4 target ≤ 0.45.\n")
        header_cells = "| suite | " + " | ".join(BULLETS) + " | verdict |"
        sep = "|---|" + "|".join("---" for _ in BULLETS) + "|---|"
        lines.append(header_cells)
        lines.append(sep)
        for suite in suites_present:
            row = per_suite[suite]
            cells = []
            for b in BULLETS:
                v = row["top30_jaccard"].get(b, 0.0)
                flag = " ⚠️" if v > 0.45 else ""
                cells.append(f"{v:.2f}{flag}")
            lines.append(f"| {suite} | " + " | ".join(cells) + f" | {row['verdict']} |")
        if overall:
            cells = []
            for b in BULLETS:
                v = overall["top30_jaccard"].get(b, 0.0)
                flag = " ⚠️" if v > 0.45 else ""
                cells.append(f"{v:.2f}{flag}")
            lines.append(
                f"| **overall** | " + " | ".join(cells)
                + f" | **{overall['verdict']}** |"
            )

        # -- Pairwise Jaccard --
        lines.append("\n## 2. Pairwise Jaccard (random image_patch vs last_text row pairs)\n")
        lines.append("Sampled per suite from rows with the same `source_example_id` when possible.\n")
        lines.append("| suite | n_pairs | mean | p10 | p50 | p90 |")
        lines.append("|---|---|---|---|---|---|")
        for suite in suites_present:
            r = per_suite[suite]
            lines.append(
                f"| {suite} | {r['n_pairwise_jaccard']:,} | "
                f"{r['mean_pairwise_jaccard']:.2f} | {r['p10_pairwise_jaccard']:.2f} | "
                f"{r['p50_pairwise_jaccard']:.2f} | {r['p90_pairwise_jaccard']:.2f} |"
            )
        if overall:
            lines.append(
                f"| **overall** | {overall['n_pairwise_jaccard']:,} | "
                f"{overall['mean_pairwise_jaccard']:.2f} | "
                f"{overall['p10_pairwise_jaccard']:.2f} | "
                f"{overall['p50_pairwise_jaccard']:.2f} | "
                f"{overall['p90_pairwise_jaccard']:.2f} |"
            )

        # -- Last-bullet mix --
        lines.append("\n## 3. Last-bullet category mix per ptype\n")
        lines.append(
            "Per ptype, what fraction of captions end with each category. "
            "V4 expects `image_patch` → mostly `target`/`scene`, "
            "`last_text` → mostly `plan`.\n"
        )
        lines.append("| suite | ptype | target | scene | plan | spatial |")
        lines.append("|---|---|---|---|---|---|")
        for suite in suites_present + (["overall"] if overall else []):
            row = per_suite.get(suite) if suite != "overall" else overall
            for pt in ("image_patch", "last_text", "anchor", "fallback"):
                mix = row["last_bullet_mix"].get(pt, {})
                if not mix:
                    continue
                lines.append(
                    f"| {suite} | {pt} | {_fmt_pct(mix.get('target', 0.0))} | "
                    f"{_fmt_pct(mix.get('scene', 0.0))} | "
                    f"{_fmt_pct(mix.get('plan', 0.0))} | "
                    f"{_fmt_pct(mix.get('spatial', 0.0))} |"
                )

        # -- Entropy --
        lines.append("\n## 4. Per-bullet token entropy (bits) per ptype\n")
        lines.append(
            "Higher entropy = more lexical diversity. A ptype that's "
            "heavily templated will have low entropy.\n"
        )
        lines.append("| suite | ptype | target | scene | spatial | plan |")
        lines.append("|---|---|---|---|---|---|")
        for suite in suites_present + (["overall"] if overall else []):
            row = per_suite.get(suite) if suite != "overall" else overall
            for pt in ("image_patch", "last_text"):
                ent = row["token_entropy_bits"].get(pt, {})
                if not ent:
                    continue
                lines.append(
                    f"| {suite} | {pt} | {ent.get('target', 0.0):.2f} | "
                    f"{ent.get('scene', 0.0):.2f} | "
                    f"{ent.get('spatial', 0.0):.2f} | "
                    f"{ent.get('plan', 0.0):.2f} |"
                )

    # -- Verdict --
    lines.append("\n## 5. Verdict logic\n")
    lines.append(
        "- **GREEN**: every (suite, bullet) cell has top-30 Jaccard ≤ 0.45, "
        "AND `image_patch` last_bullet_mix[plan] ≤ 30%, "
        "AND `last_text` last_bullet_mix[plan] ≥ 60%.\n"
        "- **YELLOW**: 1-2 cells violate.\n"
        "- **RED**: ≥3 cells violate.\n"
    )
    if overall:
        lines.append(
            f"\n**Final**: `{overall.get('verdict')}` "
            f"(corpus violations: {overall.get('violations')})\n"
        )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# I/O
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


def _discover_suites(labels_root: Path, suite_filter: str | None) -> list[str]:
    if suite_filter:
        return [suite_filter]
    suites: list[str] = []
    for p in sorted(labels_root.iterdir()):
        if p.is_dir() and (p / "labels.jsonl").exists():
            suites.append(p.name)
    return suites


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--labels-root", type=Path, required=True,
        help="Directory containing one subdir per suite, each with labels.jsonl.",
    )
    p.add_argument(
        "--out-json", type=Path, required=True,
        help="Where to write the JSON summary.",
    )
    p.add_argument(
        "--out-md", type=Path, default=None,
        help="Optional markdown report path.",
    )
    p.add_argument(
        "--suite", type=str, default=None,
        help="Restrict to one suite (subdir name).",
    )
    p.add_argument(
        "--n-pairs", type=int, default=2000,
        help="Number of (image_patch, last_text) pairs to sample per suite.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for pairwise sampling.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels_root = args.labels_root.resolve()
    if not labels_root.exists():
        print(f"ERROR: labels root not found: {labels_root}", file=sys.stderr)
        return 1

    suites = _discover_suites(labels_root, args.suite)
    if not suites:
        print(
            f"ERROR: no suite subdirectories with labels.jsonl found under "
            f"{labels_root}",
            file=sys.stderr,
        )
        return 1

    per_suite_acc: dict[str, SuiteAcc] = {}
    for suite in suites:
        path = labels_root / suite / "labels.jsonl"
        if not path.exists():
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        acc = SuiteAcc()
        for rec in iter_jsonl(path):
            acc.update(rec)
        per_suite_acc[suite] = acc
        print(
            f"[{suite}] {acc.total_rows:,} rows; "
            "ptypes=" + ", ".join(
                f"{pt}={acc.ptype_rows[pt]}" for pt in SuiteAcc.PTYPES
                if acc.ptype_rows[pt] > 0
            ),
            file=sys.stderr,
        )

    per_suite_summary: dict[str, dict] = {}
    for suite, acc in per_suite_acc.items():
        per_suite_summary[suite] = _suite_summary(acc, args.n_pairs, args.seed)

    overall_acc = _overall_acc(per_suite_acc)
    overall_summary = _suite_summary(overall_acc, args.n_pairs, args.seed)

    summary = {
        "labels_root": str(labels_root),
        "suite_filter": args.suite,
        "n_pairs_target": args.n_pairs,
        "seed": args.seed,
        "per_suite": per_suite_summary,
        "overall": overall_summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nJSON summary -> {args.out_json}", file=sys.stderr)

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        md = _build_markdown(summary, labels_root, args.suite)
        args.out_md.write_text(md)
        print(f"Markdown report -> {args.out_md}", file=sys.stderr)

    # --- terse stdout summary for parent agent ---
    overall_top30 = overall_summary["top30_jaccard"]
    mean_top30 = mean(overall_top30.values())
    print(
        "\n=== ptype disambiguation summary ===\n"
        f"verdict={overall_summary['verdict']} "
        f"violations={overall_summary['violations']} "
        f"rows={overall_summary['n_rows']:,}\n"
        "top30_jaccard "
        + ", ".join(f"{b}={overall_top30[b]:.2f}" for b in BULLETS) + "\n"
        f"top30_jaccard_mean={mean_top30:.3f}\n"
        f"mean_pairwise={overall_summary['mean_pairwise_jaccard']:.3f} "
        f"(p10={overall_summary['p10_pairwise_jaccard']:.2f}, "
        f"p50={overall_summary['p50_pairwise_jaccard']:.2f}, "
        f"p90={overall_summary['p90_pairwise_jaccard']:.2f})\n"
        "last_bullet_mix.image_patch.plan="
        f"{overall_summary['last_bullet_mix'].get('image_patch', {}).get('plan', 0.0):.1f}% "
        "(target <=30%)\n"
        "last_bullet_mix.last_text.plan="
        f"{overall_summary['last_bullet_mix'].get('last_text', {}).get('plan', 0.0):.1f}% "
        "(target >=60%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

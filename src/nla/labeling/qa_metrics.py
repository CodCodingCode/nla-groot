"""Deterministic quality-axis scorers for the prompt A/B test.

Three axes, each with a per-label boolean ``pass`` flag and a structured
diagnostics record so we can attribute failures down the line:

(a) Format consistency
    -- bullet count is exactly 4 or 5
    -- every bullet matches the canonical regex
    -- no combined categories ("gripper/spatial:" fails)
    -- no preamble / no conclusion outside the bullet block
    -- per-bullet length cap (35 words)

(b) Per-example specificity (auto half)
    -- sentence-transformer embed each label; flag labels whose mean cosine
       distance to others *in the same position-type subset* is <= 0.40.

(c) Right vocabulary (auto half)
    -- anti-pattern regex (RGB, pixels, hex, numeric measurements,
       affective verbs, actuator commands)
    -- category whitelist (covered by (a) but recorded separately so we can
       see vocab-only failures)

The aggregator returns a per-variant scorecard: per-axis pass rate plus a
breakdown by position type and the top failure modes per axis.

Pure-Python where possible; the only "fat" dependency is
``sentence-transformers`` for (b). We import it lazily so importing this
module doesn't drag the model in.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from nla.labeling.prompts import BULLET_CATEGORIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / patterns
# ---------------------------------------------------------------------------

# The plan's canonical bullet regex. Tightened with explicit char-class on the
# trailing content so it matches the literal "- category: content" shape we ask
# for in the system prompt.
_CATEGORY_GROUP = "|".join(re.escape(c) for c in BULLET_CATEGORIES)
BULLET_RE = re.compile(rf"^- ({_CATEGORY_GROUP}):\s+(.{{20,}})$")

# A bullet line that *looks like* a bullet (starts with "- ") but doesn't match
# the canonical shape -- useful for diagnostics on near-misses.
LOOSE_BULLET_RE = re.compile(r"^- ([A-Za-z_/ ]+):\s*(.*)$")

# Combined-category sniffer: "gripper/spatial:" or "scene & target:" etc.
COMBINED_CATEGORY_RE = re.compile(r"^- [A-Za-z_]+\s*[/&,+]\s*[A-Za-z_]+:")

# Anti-pattern regexes for the vocabulary axis. Each is named so we can
# attribute failures to a category.
_ANTI_PATTERNS: dict[str, re.Pattern[str]] = {
    "rgb_or_pixels": re.compile(r"\b(rgb|pixel|pixels|hex)\b", re.IGNORECASE),
    "numeric_measurement": re.compile(
        # number followed by a unit; common unit aliases included
        r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|inches|deg|degrees?|°|n|newton|newtons|kg|g)\b",
        re.IGNORECASE,
    ),
    "affective_verb": re.compile(
        r"\b(feels?|wants?|thinks?|decides?|believes?|hopes?|wishes?|desires?|"
        r"intends?|prefers?|considers?)\b",
        re.IGNORECASE,
    ),
    "actuator_command": re.compile(
        r"\b(apply\s+\d+\s*%?\s*(?:force|torque)|torque|joint\s*angle|"
        r"actuator|servo|pwm|motor\s*command)\b",
        re.IGNORECASE,
    ),
}

# Threshold for (b) auto distinctness. The plan pins this at 0.40 mean cosine
# distance within the per-position-type subset.
DISTINCTNESS_THRESHOLD: float = 0.40

# Per-bullet word cap (axis a).
BULLET_WORD_CAP: int = 35

# Default sentence-transformer model for (b) auto.
DEFAULT_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Per-label structured results
# ---------------------------------------------------------------------------

@dataclass
class FormatResult:
    passed: bool
    n_bullets: int
    bad_bullets: list[tuple[int, str, str]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class VocabResult:
    passed: bool
    matched_patterns: dict[str, list[str]] = field(default_factory=dict)
    unknown_categories: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class DistinctnessResult:
    passed: bool
    mean_cosine_distance: float
    n_peers: int


@dataclass
class LabelScores:
    """All deterministic scores for one label."""
    example_id: str
    position_type: str
    description: str
    format: FormatResult
    vocab: VocabResult
    distinctness: DistinctnessResult | None = None

    @property
    def passes_a(self) -> bool:
        return self.format.passed

    @property
    def passes_b_auto(self) -> bool:
        # If we never ran (b)-auto, treat it as "not failing" but flag the
        # missing scorer at aggregation time. (b)-LLM provides the other half.
        return self.distinctness is None or self.distinctness.passed

    @property
    def passes_c_auto(self) -> bool:
        return self.vocab.passed


# ---------------------------------------------------------------------------
# Axis (a): format
# ---------------------------------------------------------------------------

def _split_bullets(description: str) -> tuple[list[str], list[str]]:
    """Return (bullet_lines, non_bullet_lines) from the raw label text.

    A "bullet line" is any line that starts with "- " after .strip().
    Anything else that's non-empty is a non-bullet line (preamble/conclusion).
    """
    bullets: list[str] = []
    non_bullets: list[str] = []
    for raw in description.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            bullets.append(line.lstrip())
        else:
            non_bullets.append(line)
    return bullets, non_bullets


def check_format(description: str) -> FormatResult:
    """Axis (a): deterministic format checker. Returns a structured verdict.

    A label passes iff:
      * bullet count is 4 or 5
      * every bullet matches ``BULLET_RE``
      * no bullet's category line includes a combined-category separator
      * no preamble/conclusion lines outside the bullet block
      * no bullet exceeds ``BULLET_WORD_CAP`` words of content
    """
    reasons: list[str] = []
    bad: list[tuple[int, str, str]] = []

    bullets, non_bullets = _split_bullets(description.strip())

    if non_bullets:
        reasons.append(f"non_bullet_lines:{len(non_bullets)}")

    n = len(bullets)
    if n < 4 or n > 5:
        reasons.append(f"bullet_count:{n}")

    for i, b in enumerate(bullets):
        if COMBINED_CATEGORY_RE.match(b):
            bad.append((i, "combined_category", b))
            continue
        m = BULLET_RE.match(b)
        if not m:
            # Diagnose: is the category unknown, or just malformed?
            lm = LOOSE_BULLET_RE.match(b)
            if lm is None:
                bad.append((i, "not_a_bullet", b))
            else:
                cat = lm.group(1).strip().lower()
                if cat not in BULLET_CATEGORIES:
                    bad.append((i, f"unknown_category:{cat}", b))
                else:
                    rest = lm.group(2)
                    if len(rest) < 20:
                        bad.append((i, "content_too_short", b))
                    else:
                        bad.append((i, "regex_mismatch", b))
            continue
        content = m.group(2)
        n_words = len(content.split())
        if n_words > BULLET_WORD_CAP:
            bad.append((i, f"too_long:{n_words}w", b))

    if bad:
        reasons.append(f"bad_bullets:{len(bad)}")

    passed = (
        not reasons
        and not bad
        and 4 <= n <= 5
    )
    return FormatResult(passed=passed, n_bullets=n, bad_bullets=bad, reasons=reasons)


# ---------------------------------------------------------------------------
# Axis (c) auto: vocabulary anti-patterns
# ---------------------------------------------------------------------------

def check_vocab(description: str) -> VocabResult:
    """Axis (c) auto: anti-pattern regex + category-allow-list checker.

    Anti-patterns are scanned over the *entire label* (after stripping the
    leading "- " bullet markers). Category-allow-list checks run separately
    on each bullet's category token.

    Both subaxes must pass.
    """
    matched: dict[str, list[str]] = {}
    unknown: list[str] = []

    bullets, _ = _split_bullets(description)
    for b in bullets:
        # Categorize each bullet (canonical or loose) so we can flag
        # out-of-vocab category labels.
        m = BULLET_RE.match(b)
        if m is None:
            lm = LOOSE_BULLET_RE.match(b)
            if lm is not None:
                cat = lm.group(1).strip().lower()
                if cat not in BULLET_CATEGORIES and cat not in unknown:
                    unknown.append(cat)

    # Anti-patterns scan the entire label content (bullets concatenated),
    # not just canonical bullets -- "the label contains none of ..."
    full_text = "\n".join(bullets) if bullets else description
    for name, pat in _ANTI_PATTERNS.items():
        hits = pat.findall(full_text)
        if hits:
            flat = [h if isinstance(h, str) else " ".join(h).strip() for h in hits]
            matched[name] = flat

    reasons: list[str] = []
    if matched:
        reasons.extend(f"anti:{k}" for k in matched)
    if unknown:
        reasons.append(f"unknown_cats:{','.join(unknown)}")

    passed = not matched and not unknown
    return VocabResult(
        passed=passed, matched_patterns=matched, unknown_categories=unknown,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Axis (b) auto: distinctness within position-type subset
# ---------------------------------------------------------------------------

def _load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Lazy-load the sentence transformer used for distinctness scoring."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for axis (b) auto. "
            "Install with `pip install sentence-transformers`."
        ) from e
    return SentenceTransformer(model_name)


def compute_distinctness(
    descriptions_by_subset: dict[str, list[str]],
    *,
    threshold: float = DISTINCTNESS_THRESHOLD,
    model_name: str = DEFAULT_EMBED_MODEL,
    embedder=None,
) -> dict[str, list[DistinctnessResult]]:
    """For each subset, embed all descriptions and score each by mean cosine
    distance to other descriptions in the same subset.

    Returns a dict mapping subset-name -> list-of-DistinctnessResult in the
    *same order* as the input list.

    A label passes (b)-auto iff its mean cosine distance to peers > threshold.

    Subsets with <2 labels can't be scored; we mark them as passed but with
    ``n_peers = 0`` so callers can see this in diagnostics.
    """
    import numpy as np

    embedder = embedder or _load_embedder(model_name)

    out: dict[str, list[DistinctnessResult]] = {}
    for subset_name, descs in descriptions_by_subset.items():
        if len(descs) < 2:
            out[subset_name] = [
                DistinctnessResult(passed=True, mean_cosine_distance=float("nan"),
                                   n_peers=0)
                for _ in descs
            ]
            continue
        embs = embedder.encode(
            descs, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine *similarity* matrix in [-1, 1]; subtract from 1 to get distance.
        sim = embs @ embs.T
        n = len(descs)
        # Zero out self-similarity, sum & divide by (n-1).
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        mean_dist = dist.sum(axis=1) / (n - 1)
        out[subset_name] = [
            DistinctnessResult(
                passed=bool(mean_dist[i] > threshold),
                mean_cosine_distance=float(mean_dist[i]),
                n_peers=n - 1,
            )
            for i in range(n)
        ]
    return out


# ---------------------------------------------------------------------------
# End-to-end per-label scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredLabel:
    example_id: str
    position_type: str
    description: str
    format: FormatResult
    vocab: VocabResult
    distinctness: DistinctnessResult | None = None


def score_label(
    description: str,
    *,
    example_id: str,
    position_type: str,
) -> LabelScores:
    """Compute the deterministic (a)+(c)-auto verdicts for a single label.

    (b)-auto is set later by ``score_variant``, which embeds the whole batch
    at once for efficiency.
    """
    fmt = check_format(description)
    voc = check_vocab(description)
    return LabelScores(
        example_id=example_id,
        position_type=position_type,
        description=description,
        format=fmt,
        vocab=voc,
        distinctness=None,
    )


def score_variant(
    labels: Sequence[dict],
    *,
    embedder=None,
    distinctness_threshold: float = DISTINCTNESS_THRESHOLD,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    skip_distinctness: bool = False,
) -> list[LabelScores]:
    """Score an entire variant's output set (labels.jsonl rows).

    Each input row must have ``example_id``, ``description``, and
    ``meta.position_type``.

    Distinctness is computed per position_type subset across the whole batch.

    If ``skip_distinctness=True``, only deterministic (a) and (c)-auto scores
    are filled in; (b)-auto stays None. Useful for fast unit tests.
    """
    scored: list[LabelScores] = []
    by_subset: dict[str, list[int]] = defaultdict(list)
    for row in labels:
        pos_type = row.get("meta", {}).get("position_type", "unknown")
        s = score_label(
            row["description"],
            example_id=row["example_id"],
            position_type=pos_type,
        )
        scored.append(s)
        by_subset[pos_type].append(len(scored) - 1)

    if skip_distinctness:
        return scored

    descs_by_subset = {
        subset: [scored[i].description for i in idxs]
        for subset, idxs in by_subset.items()
    }
    dist_by_subset = compute_distinctness(
        descs_by_subset,
        threshold=distinctness_threshold,
        model_name=embed_model_name,
        embedder=embedder,
    )
    for subset, idxs in by_subset.items():
        results = dist_by_subset[subset]
        for i, dr in zip(idxs, results):
            scored[i].distinctness = dr
    return scored


# ---------------------------------------------------------------------------
# Variant aggregation
# ---------------------------------------------------------------------------

@dataclass
class AxisRollup:
    pass_rate: float
    n_passed: int
    n_total: int
    # Top reason strings -> count, for failure-mode triage.
    top_failures: list[tuple[str, int]]


@dataclass
class VariantScorecard:
    variant_id: str
    n_labels: int
    overall_pass_rate: float
    axis_a: AxisRollup
    axis_b_auto: AxisRollup
    axis_c_auto: AxisRollup
    per_position_type: dict[str, dict[str, float]]
    passes_95: bool

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "n_labels": self.n_labels,
            "overall_pass_rate": self.overall_pass_rate,
            "passes_95": self.passes_95,
            "axis_a": {
                "pass_rate": self.axis_a.pass_rate,
                "n_passed": self.axis_a.n_passed,
                "n_total": self.axis_a.n_total,
                "top_failures": self.axis_a.top_failures,
            },
            "axis_b_auto": {
                "pass_rate": self.axis_b_auto.pass_rate,
                "n_passed": self.axis_b_auto.n_passed,
                "n_total": self.axis_b_auto.n_total,
                "top_failures": self.axis_b_auto.top_failures,
            },
            "axis_c_auto": {
                "pass_rate": self.axis_c_auto.pass_rate,
                "n_passed": self.axis_c_auto.n_passed,
                "n_total": self.axis_c_auto.n_total,
                "top_failures": self.axis_c_auto.top_failures,
            },
            "per_position_type": self.per_position_type,
        }


def _rollup_axis(
    scored: Sequence[LabelScores],
    accessor,
    failure_reasons,
    *,
    top_k: int = 6,
) -> AxisRollup:
    n_total = len(scored)
    n_passed = sum(1 for s in scored if accessor(s))
    failures: Counter = Counter()
    for s in scored:
        if not accessor(s):
            for r in failure_reasons(s):
                failures[r] += 1
    return AxisRollup(
        pass_rate=(n_passed / n_total) if n_total else 0.0,
        n_passed=n_passed,
        n_total=n_total,
        top_failures=failures.most_common(top_k),
    )


def aggregate_variant(
    variant_id: str,
    scored: Sequence[LabelScores],
    *,
    pass_threshold: float = 0.95,
) -> VariantScorecard:
    """Roll a list of LabelScores up to one VariantScorecard."""
    axis_a = _rollup_axis(
        scored,
        accessor=lambda s: s.passes_a,
        failure_reasons=lambda s: s.format.reasons + [r for _, r, _ in s.format.bad_bullets],
    )
    axis_c = _rollup_axis(
        scored,
        accessor=lambda s: s.passes_c_auto,
        failure_reasons=lambda s: s.vocab.reasons,
    )
    axis_b = _rollup_axis(
        scored,
        accessor=lambda s: s.passes_b_auto,
        failure_reasons=lambda s: (
            [f"low_distinctness:{s.distinctness.mean_cosine_distance:.2f}"]
            if s.distinctness is not None and not s.distinctness.passed
            else []
        ),
    )

    per_pos: dict[str, dict[str, float]] = {}
    by_pos: dict[str, list[LabelScores]] = defaultdict(list)
    for s in scored:
        by_pos[s.position_type].append(s)
    for pos, group in by_pos.items():
        n = len(group)
        per_pos[pos] = {
            "n": float(n),
            "a": sum(g.passes_a for g in group) / n,
            "b_auto": sum(g.passes_b_auto for g in group) / n,
            "c_auto": sum(g.passes_c_auto for g in group) / n,
        }

    n_all = len(scored)
    overall_passed = sum(
        1 for s in scored
        if s.passes_a and s.passes_b_auto and s.passes_c_auto
    )

    passes_95 = (
        axis_a.pass_rate >= pass_threshold
        and axis_b.pass_rate >= pass_threshold
        and axis_c.pass_rate >= pass_threshold
    )
    return VariantScorecard(
        variant_id=variant_id,
        n_labels=n_all,
        overall_pass_rate=(overall_passed / n_all) if n_all else 0.0,
        axis_a=axis_a,
        axis_b_auto=axis_b,
        axis_c_auto=axis_c,
        per_position_type=per_pos,
        passes_95=passes_95,
    )


# ---------------------------------------------------------------------------
# Convenience: end-to-end on a list of label rows
# ---------------------------------------------------------------------------

def score_and_aggregate(
    variant_id: str,
    label_rows: Sequence[dict],
    *,
    embedder=None,
    distinctness_threshold: float = DISTINCTNESS_THRESHOLD,
    skip_distinctness: bool = False,
) -> tuple[list[LabelScores], VariantScorecard]:
    """End-to-end: deterministic scoring + variant rollup."""
    scored = score_variant(
        label_rows,
        embedder=embedder,
        distinctness_threshold=distinctness_threshold,
        skip_distinctness=skip_distinctness,
    )
    card = aggregate_variant(variant_id, scored)
    return scored, card


__all__ = [
    "BULLET_RE",
    "LOOSE_BULLET_RE",
    "COMBINED_CATEGORY_RE",
    "BULLET_WORD_CAP",
    "DISTINCTNESS_THRESHOLD",
    "DEFAULT_EMBED_MODEL",
    "FormatResult",
    "VocabResult",
    "DistinctnessResult",
    "LabelScores",
    "AxisRollup",
    "VariantScorecard",
    "check_format",
    "check_vocab",
    "compute_distinctness",
    "score_label",
    "score_variant",
    "aggregate_variant",
    "score_and_aggregate",
]

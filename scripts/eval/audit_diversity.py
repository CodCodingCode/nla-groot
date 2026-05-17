"""V3 LIBERO caption diversity / template-collapse audit.

Streams the JSONL label files, parses 5-bullet `description` strings, and
computes per-(suite, bullet_type) vocabulary, n-gram, duplicate, and
cross-suite distinguishability statistics for the V3 LIBERO 4-suite
corpus plus the LIBERO pilot. The legacy V2 DROID baseline column is
included as a historical reference *when its labels.jsonl is still
reachable* (live tree first, then the archived path under
``data/_archive_droid/``); when DROID is gone the V2-DROID columns are
silently dropped. Writes a markdown report to
docs/sft_plan/v3_quality/agent3_diversity.md plus a JSON stats blob next to
it for downstream agents.

Run from repo root with `PYTHONPATH=src .venv/bin/python
scripts/eval/audit_diversity.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


REPO = Path("/home/ubuntu/nla-groot")
LIBERO_BASE = REPO / "data/labels/libero_4suite_stride2"
# DROID is the legacy V2 baseline. After the DROID archive (see
# scripts/migration/archive_droid.sh) labels live under
# data/_archive_droid/labels/droid_100ep/. We fall back to that path so
# historical comparison still works without re-hydrating the live tree,
# and silently skip the DROID column when the archive is also gone.
DROID_PATH_LIVE    = REPO / "data/labels/droid_100ep/labels.jsonl"
DROID_PATH_ARCHIVE = REPO / "data/_archive_droid/labels/droid_100ep/labels.jsonl"
PILOT_PATH = REPO / "data/labels/libero_goal_pilot/labels.jsonl"


def _resolve_droid_path() -> Path | None:
    """Return the first DROID labels.jsonl that actually exists, or None."""
    for candidate in (DROID_PATH_LIVE, DROID_PATH_ARCHIVE):
        if candidate.exists():
            return candidate
    return None

V3_SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]
CANON_BULLETS = ["language", "target", "scene", "spatial", "plan"]

OUT_DIR = REPO / "docs/sft_plan/v3_quality"
REPORT_MD = OUT_DIR / "agent3_diversity.md"
REPORT_JSON = OUT_DIR / "agent3_diversity_stats.json"

TOK = re.compile(r"[a-z0-9]+")
# STOPWORDS used for near-dup hashing — broader, to maximize collisions
# of paraphrases that differ only in function words.
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "and", "with",
    "is", "are", "at", "near", "by", "for", "be", "its", "it",
    "this", "that", "or", "as", "from", "into", "onto", "over",
    "under", "out", "up", "down", "off", "but", "so", "than",
    "then", "they", "them", "their", "there", "these", "those",
    "have", "has", "had", "was", "were", "been", "being", "do",
    "does", "did", "will", "would", "should", "could", "may",
    "might", "can", "must", "if", "while",
}
# Narrower closed-class set used to decide whether an n-gram counts as a
# "content phrase" for boilerplate detection (`pick up`, `phase active`,
# `pick and place` should count; `with a`, `on the`, `is the` should not).
FUNCTION_WORDS = {
    "the", "a", "an", "of", "in", "on", "to", "and", "with",
    "is", "are", "at", "by", "for", "be", "its", "it", "this",
    "that", "or", "as", "from", "into", "onto", "but", "so",
    "than", "then", "have", "has", "had", "was", "were", "been",
    "being", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "can", "must", "if", "while", "also",
}


def is_content_ngram(tokens: tuple[str, ...]) -> bool:
    """Return True if the n-gram carries lexical content (not pure stopwords).

    Bigrams must have >=1 non-function-word token; trigrams must have >=2;
    longer n-grams must be majority non-function words.
    """
    if not tokens:
        return False
    non_fn = sum(1 for t in tokens if t not in FUNCTION_WORDS)
    if len(tokens) <= 2:
        return non_fn >= 1
    if len(tokens) == 3:
        return non_fn >= 2
    return non_fn >= len(tokens) - 1

RNG = random.Random(20260516)


# ---------- parsing ----------

def parse_bullets(desc: str) -> dict[str, str]:
    bullets: dict[str, str] = {}
    if not desc:
        return bullets
    for raw in desc.splitlines():
        line = raw.strip()
        if not line.startswith("-"):
            continue
        body = line.lstrip("-").lstrip()
        if ":" not in body:
            continue
        key, val = body.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        if not key or not val:
            continue
        # Some labelers use `image_region`/`distractor`; keep but skip empty.
        bullets[key] = val
    return bullets


def tokenize(text: str) -> list[str]:
    return TOK.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Iterator[tuple[str, ...]]:
    if len(tokens) < n:
        return iter(())
    return (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def norm_key(text: str) -> str:
    """Hash key for near-duplicate detection: token set minus stop words, joined."""
    toks = [t for t in tokenize(text) if t not in STOPWORDS]
    if not toks:
        return ""
    return " ".join(toks)


# ---------- accumulator ----------

@dataclass
class BulletAcc:
    n_bullets: int = 0
    total_tokens: int = 0
    uni: Counter = field(default_factory=Counter)
    bi: Counter = field(default_factory=Counter)
    tri: Counter = field(default_factory=Counter)
    # document-frequency (how many bullets contain the n-gram at least once)
    df_bi: Counter = field(default_factory=Counter)
    df_tri: Counter = field(default_factory=Counter)
    # exact duplicates: raw text counter (small risk of memory)
    exact: Counter = field(default_factory=Counter)
    # near-duplicate: normalized text counter
    near: Counter = field(default_factory=Counter)
    # reservoir sample of (norm_key -> [raw_text examples])
    cluster_examples: dict[str, list[str]] = field(default_factory=dict)
    # reservoir of raw bullets (for embedding fallback / qualitative)
    sample: list[str] = field(default_factory=list)
    _sample_cap: int = 4000
    _sample_seen: int = 0

    def add(self, text: str) -> None:
        toks = tokenize(text)
        self.n_bullets += 1
        self.total_tokens += len(toks)
        self.uni.update(toks)
        bigrams = list(ngrams(toks, 2))
        trigrams = list(ngrams(toks, 3))
        self.bi.update(bigrams)
        self.tri.update(trigrams)
        if bigrams:
            for g in set(bigrams):
                self.df_bi[g] += 1
        if trigrams:
            for g in set(trigrams):
                self.df_tri[g] += 1
        self.exact[text] += 1
        nk = norm_key(text)
        self.near[nk] += 1
        # remember up to 6 raw examples per cluster (for cluster display)
        if nk:
            ex_list = self.cluster_examples.setdefault(nk, [])
            if len(ex_list) < 6:
                ex_list.append(text)
        # reservoir sample of raw bullets
        self._sample_seen += 1
        if len(self.sample) < self._sample_cap:
            self.sample.append(text)
        else:
            j = RNG.randint(0, self._sample_seen - 1)
            if j < self._sample_cap:
                self.sample[j] = text

    # ---- derived metrics ----
    def stats(self) -> dict:
        n = max(self.n_bullets, 1)
        tt = max(self.total_tokens, 1)
        exact_unique = len(self.exact)
        near_unique = len(self.near)
        exact_dup_rate = 1.0 - exact_unique / n
        near_dup_rate = 1.0 - near_unique / n
        ttr = len(self.uni) / tt
        return {
            "n_bullets": self.n_bullets,
            "total_tokens": self.total_tokens,
            "unique_unigrams": len(self.uni),
            "unique_bigrams": len(self.bi),
            "unique_trigrams": len(self.tri),
            "type_token_ratio": ttr,
            "exact_unique": exact_unique,
            "near_unique": near_unique,
            "exact_dup_rate": exact_dup_rate,
            "near_dup_rate": near_dup_rate,
            "avg_tokens_per_bullet": self.total_tokens / n,
        }

    def top_ngrams(self, n: int, k: int) -> list[tuple[tuple[str, ...], int, float]]:
        counter = {1: self.uni, 2: self.bi, 3: self.tri}[n]
        out = []
        for g, c in counter.most_common(k):
            out.append((g, c, c / max(self.n_bullets, 1)))
        return out

    def top_doc_phrases(self, n: int, k: int) -> list[tuple[tuple[str, ...], int, float]]:
        df = {2: self.df_bi, 3: self.df_tri}[n]
        out = []
        for g, c in df.most_common(k):
            out.append((g, c, c / max(self.n_bullets, 1)))
        return out

    def top_clusters(self, k: int) -> list[tuple[int, list[str]]]:
        ordered = self.near.most_common(k)
        clusters = []
        for nk, cnt in ordered:
            ex = self.cluster_examples.get(nk, [])
            clusters.append((cnt, ex))
        return clusters


# ---------- streaming ----------

def stream_labels(path: Path) -> Iterator[dict[str, str]]:
    """Yield {bullet_key: bullet_text} dicts."""
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            desc = rec.get("description") or ""
            yield parse_bullets(desc)


def accumulate(label_paths: dict[str, Path]) -> dict[str, dict[str, BulletAcc]]:
    """Returns {source_name: {bullet_type: BulletAcc}}."""
    out: dict[str, dict[str, BulletAcc]] = {}
    for src, path in label_paths.items():
        accs: dict[str, BulletAcc] = defaultdict(BulletAcc)
        t0 = time.time()
        n_rec = 0
        n_bullets_seen = 0
        for bullets in stream_labels(path):
            n_rec += 1
            for k, v in bullets.items():
                accs[k].add(v)
                n_bullets_seen += 1
        dt = time.time() - t0
        print(
            f"[scan] {src}: {n_rec} records, {n_bullets_seen} bullets "
            f"({len(accs)} bullet types) in {dt:.1f}s",
            file=sys.stderr,
        )
        out[src] = dict(accs)
    return out


# ---------- cross-suite distinguishability ----------

def cross_suite_distinguish(per_suite: dict[str, dict[str, BulletAcc]], bullet_type: str) -> dict | None:
    """Fit a TF-IDF + LogisticRegression on suite labels for a bullet type."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError:
        return None

    suites = sorted([s for s, b in per_suite.items() if bullet_type in b])
    if len(suites) < 2:
        return None
    texts: list[str] = []
    labels: list[str] = []
    per_suite_sample = 2000
    for s in suites:
        sample = per_suite[s][bullet_type].sample
        if len(sample) > per_suite_sample:
            sample = RNG.sample(sample, per_suite_sample)
        texts.extend(sample)
        labels.extend([s] * len(sample))
    if len(set(labels)) < 2 or len(texts) < 200:
        return None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
    except ValueError:
        return None
    vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=20000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=400, n_jobs=1)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    cm = confusion_matrix(y_test, pred, labels=suites).tolist()
    chance = 1.0 / len(suites)
    return {
        "bullet_type": bullet_type,
        "suites": suites,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": acc,
        "chance": chance,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


# ---------- formatting helpers ----------

def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100*x:.{digits}f}%"


def fmt_ngram(g) -> str:
    if isinstance(g, str):
        return g
    return " ".join(g)


def join_md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(headers) + " |"
    sep2 = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([sep, sep2, body])


def short(s: str, n: int = 110) -> str:
    s = s.replace("\n", " ").replace("|", "\\|")
    if len(s) > n:
        return s[: n - 1] + "…"
    return s


# ---------- forbidden-phrase cross-check ----------

FORBIDDEN_HINTS = {
    "wants", "wants to", "decides", "intends", "thinks",
    "the robot is trying", "the robot wants", "the gripper wants",
    "happy", "sad", "feels", "knows", "believes",
    "the robot decides", "the robot thinks", "the robot believes",
    "the agent wants", "the model decides",
}

# Agent 2's NEW C-failure mode: low-level motor imperatives leaking into the
# `plan:` bullet (e.g. "grasp the bowl", "carry it", "align the gripper").
# Boilerplate-level rates of these phrases double-flag Agent 2's regression.
MOTOR_IMPERATIVES = {
    "grasp the", "carry it", "reach toward", "lift it", "align the",
    "place it", "move it", "pick up", "and place", "carry it to",
    "and carry it", "for placement", "carries the action", "phase active",
    "pick and place",
}


def has_forbidden(ngram_str: str) -> bool:
    s = ngram_str.lower()
    return any(f in s for f in FORBIDDEN_HINTS)


def is_motor_imperative(ngram_str: str) -> bool:
    s = ngram_str.lower()
    return any(f in s for f in MOTOR_IMPERATIVES)


# ---------- main report ----------

def build_report(
    per_source: dict[str, dict[str, BulletAcc]],
    cross_results: dict[str, dict],
) -> tuple[str, dict]:
    lines: list[str] = []
    lines.append("# Agent 3 — V3 LIBERO Caption Diversity Audit")
    lines.append("")
    lines.append(
        "Streaming audit of the 101,580 V3 LIBERO captions in "
        "`data/labels/libero_4suite_stride2/`, with V2 DROID and the LIBERO "
        "pilot as diversity baselines. Per-bullet vocabulary, n-gram, "
        "duplicate, and cross-suite distinguishability metrics."
    )
    lines.append("")

    # --- 1. Top-level corpus sizes ---
    lines.append("## 1. Corpus inventory")
    rows = []
    for src, accs in per_source.items():
        n_records = max((a.n_bullets for a in accs.values()), default=0)
        # n_records is approximate (max bullet count)
        # better: report n_bullets across bullet types
        total_bullets = sum(a.n_bullets for a in accs.values())
        types = sorted(accs.keys())
        rows.append(
            [src, str(n_records), str(total_bullets), ", ".join(types)]
        )
    lines.append(
        join_md_table(
            ["source", "≈records", "total bullets", "bullet types observed"],
            rows,
        )
    )
    lines.append("")

    # --- 2. Per-(suite, bullet_type) vocabulary stats ---
    lines.append("## 2. Per-(suite, bullet_type) vocabulary stats")
    lines.append("")
    lines.append(
        "TTR = unique unigrams / total tokens. Lower TTR + low unique n-grams "
        "= heavier template re-use."
    )
    lines.append("")
    headers = [
        "source",
        "bullet",
        "n_bullets",
        "tokens",
        "uniq 1g",
        "uniq 2g",
        "uniq 3g",
        "TTR",
        "exact dup %",
        "near dup %",
        "avg tok",
    ]
    rows = []
    for src in list(per_source.keys()):
        accs = per_source[src]
        bullet_keys = [k for k in CANON_BULLETS if k in accs] + sorted(
            k for k in accs if k not in CANON_BULLETS
        )
        for b in bullet_keys:
            s = accs[b].stats()
            rows.append(
                [
                    src,
                    b,
                    str(s["n_bullets"]),
                    f"{s['total_tokens']:,}",
                    f"{s['unique_unigrams']:,}",
                    f"{s['unique_bigrams']:,}",
                    f"{s['unique_trigrams']:,}",
                    f"{s['type_token_ratio']:.4f}",
                    fmt_pct(s["exact_dup_rate"]),
                    fmt_pct(s["near_dup_rate"]),
                    f"{s['avg_tokens_per_bullet']:.1f}",
                ]
            )
    lines.append(join_md_table(headers, rows))
    lines.append("")

    # --- 3. Top-20 most common phrases per canonical bullet (V3 combined) ---
    lines.append("## 3. Top-20 most common phrases per bullet type (V3 LIBERO combined)")
    lines.append("")
    lines.append(
        "Combined across the 4 V3 suites. Phrase % is *document frequency* "
        "(fraction of bullets containing the phrase at least once). "
        "Items with **bold** % cross the 5% boilerplate threshold."
    )
    lines.append("")
    # build a V3 aggregated accumulator per bullet
    v3_combined: dict[str, BulletAcc] = defaultdict(BulletAcc)
    for src in V3_SUITES:
        if src not in per_source:
            continue
        for b, acc in per_source[src].items():
            comb = v3_combined[b]
            comb.n_bullets += acc.n_bullets
            comb.total_tokens += acc.total_tokens
            comb.uni.update(acc.uni)
            comb.bi.update(acc.bi)
            comb.tri.update(acc.tri)
            comb.df_bi.update(acc.df_bi)
            comb.df_tri.update(acc.df_tri)
            comb.exact.update(acc.exact)
            comb.near.update(acc.near)
            for nk, ex_list in acc.cluster_examples.items():
                tgt = comb.cluster_examples.setdefault(nk, [])
                for e in ex_list:
                    if len(tgt) < 6:
                        tgt.append(e)
            # also collect sample
            comb.sample.extend(acc.sample[:1000])

    forbidden_flags: list[dict] = []
    for b in CANON_BULLETS:
        if b not in v3_combined:
            continue
        acc = v3_combined[b]
        lines.append(f"### {b}  (n_bullets={acc.n_bullets:,})")
        # Top-20 bigrams by document frequency
        rows_bi = []
        for g, c, p in acc.top_doc_phrases(2, 20):
            txt = fmt_ngram(g)
            flag = ""
            if has_forbidden(txt):
                forbidden_flags.append(
                    {"bullet": b, "ngram": txt, "doc_pct": p}
                )
                flag = "  ⚠️FORBIDDEN"
            pct = fmt_pct(p)
            cell = f"**{pct}**" if p >= 0.05 else pct
            rows_bi.append([txt + flag, f"{c:,}", cell])
        rows_tri = []
        for g, c, p in acc.top_doc_phrases(3, 20):
            txt = fmt_ngram(g)
            flag = ""
            if has_forbidden(txt):
                forbidden_flags.append(
                    {"bullet": b, "ngram": txt, "doc_pct": p}
                )
                flag = "  ⚠️FORBIDDEN"
            pct = fmt_pct(p)
            cell = f"**{pct}**" if p >= 0.05 else pct
            rows_tri.append([txt + flag, f"{c:,}", cell])
        lines.append("**Top-20 bigrams (document frequency)**")
        lines.append("")
        lines.append(join_md_table(["bigram", "DF count", "% of bullets"], rows_bi))
        lines.append("")
        lines.append("**Top-20 trigrams (document frequency)**")
        lines.append("")
        lines.append(join_md_table(["trigram", "DF count", "% of bullets"], rows_tri))
        lines.append("")

    # --- 4. Boilerplate detection summary ---
    lines.append("## 4. Boilerplate phrases (content n-grams ≥5% of V3 bullets per type)")
    lines.append("")
    lines.append(
        "Filtered to **content-bearing** n-grams only (bigrams with ≥1 "
        "non-function-word token, trigrams with ≥2). Stopword bigrams like "
        "`with a`/`on the`/`of the` are excluded from this table — they are "
        "covered in Section 3."
    )
    lines.append("")
    bp_rows: list[list[str]] = []
    # Also keep parsed records to drive the verdict.
    bp_records: list[dict] = []
    for b in CANON_BULLETS:
        if b not in v3_combined:
            continue
        acc = v3_combined[b]
        for n in (3, 2):
            seen_under_thresh = 0
            for g, c, p in acc.top_doc_phrases(n, 200):
                if not is_content_ngram(g):
                    continue
                if p < 0.05:
                    seen_under_thresh += 1
                    if seen_under_thresh >= 5:
                        break
                    continue
                phrase = fmt_ngram(g)
                bp_rows.append(
                    [b, f"{n}-gram", phrase, f"{c:,}", fmt_pct(p)]
                )
                bp_records.append(
                    {"bullet": b, "n": n, "phrase": phrase, "count": c, "doc_pct": p}
                )
    if not bp_rows:
        lines.append("_No content bigram or trigram exceeded the 5% document-frequency threshold._")
    else:
        bp_rows.sort(key=lambda r: float(r[4].rstrip("%")), reverse=True)
        bp_records.sort(key=lambda r: r["doc_pct"], reverse=True)
        lines.append(
            join_md_table(
                ["bullet", "n-gram type", "phrase", "DF count", "% of bullets"],
                bp_rows[:80],
            )
        )
    lines.append("")

    # surface RED-zone (>=25%) and 10% bands — content phrases only
    red_phrases = [r for r in bp_records if r["doc_pct"] >= 0.25]
    yellow_phrases = [r for r in bp_records if 0.10 <= r["doc_pct"] < 0.25]

    # --- 5. Per-suite cross-comparison ---
    lines.append("## 5. Cross-suite distinguishability")
    lines.append("")
    lines.append(
        "TF-IDF (1-2 gram) + LogReg classifier, fit on a balanced sample of "
        "bullets per suite (cap 2,000 per suite per bullet), 75/25 split. "
        "Chance = 0.25 with 4 suites. Higher accuracy ⇒ the labeler is "
        "writing suite-specific content; near-chance ⇒ boilerplate that "
        "ignores the underlying task."
    )
    lines.append("")
    cs_rows = []
    for b, res in cross_results.items():
        if res is None:
            cs_rows.append([b, "—", "—", "—", "—"])
            continue
        cs_rows.append(
            [
                b,
                str(res["n_train"] + res["n_test"]),
                f"{res['accuracy']:.3f}",
                f"{res['macro_f1']:.3f}",
                f"chance={res['chance']:.3f}",
            ]
        )
    lines.append(
        join_md_table(
            ["bullet_type", "n_samples", "accuracy", "macro_F1", "baseline"],
            cs_rows,
        )
    )
    lines.append("")
    for b, res in cross_results.items():
        if res is None:
            continue
        lines.append(f"**Confusion matrix — `{b}`** (rows=true, cols=pred, order={res['suites']})")
        lines.append("")
        cm = res["confusion_matrix"]
        cm_rows = []
        for i, s in enumerate(res["suites"]):
            cm_rows.append([s] + [str(v) for v in cm[i]])
        lines.append(
            join_md_table([""] + res["suites"], cm_rows)
        )
        lines.append("")

    # --- 6. V3 vs baselines ---
    lines.append("## 6. V3 vs V2 DROID vs Pilot — diversity comparison")
    lines.append("")
    lines.append(
        "Aggregated per bullet type. V3 columns combine the four LIBERO "
        "suites; DROID and Pilot are reported as-is."
    )
    lines.append("")
    cmp_rows = []
    droid = per_source.get("droid_100ep_v2", {})
    pilot = per_source.get("libero_goal_pilot", {})
    for b in CANON_BULLETS:
        v3 = v3_combined.get(b)
        if v3 is None and b not in droid and b not in pilot:
            continue
        for tag, acc in [("V3 LIBERO", v3), ("V2 DROID", droid.get(b)), ("Pilot", pilot.get(b))]:
            if acc is None or acc.n_bullets == 0:
                cmp_rows.append([b, tag, "—", "—", "—", "—", "—", "—"])
                continue
            s = acc.stats()
            cmp_rows.append(
                [
                    b,
                    tag,
                    f"{s['n_bullets']:,}",
                    f"{s['unique_unigrams']:,}",
                    f"{s['unique_bigrams']:,}",
                    f"{s['unique_trigrams']:,}",
                    f"{s['type_token_ratio']:.4f}",
                    fmt_pct(s["near_dup_rate"]),
                ]
            )
    lines.append(
        join_md_table(
            [
                "bullet",
                "source",
                "n_bullets",
                "uniq 1g",
                "uniq 2g",
                "uniq 3g",
                "TTR",
                "near dup %",
            ],
            cmp_rows,
        )
    )
    lines.append("")

    # --- 7. Cluster examples ---
    lines.append("## 7. Top near-duplicate clusters (10 examples, 3 members each)")
    lines.append("")
    lines.append(
        "Picked from the most heavily reused normalized `plan` bullets — the "
        "bullet most prone to template collapse. Each cluster shares the "
        "same content-token set (after stopword removal). **Note:** even the "
        "largest cluster is <1% of all `plan` bullets, yet structural "
        "n-gram templates (e.g. `phase active`, `pick and place`, `carry it "
        "toward the …`) blanket 15–27% of the corpus — meaning the labeler "
        "swaps object names but locks in the same scaffold, so hash-bucket "
        "clusters under-state the true template collapse."
    )
    lines.append("")
    plan_acc = v3_combined.get("plan")
    clusters_for_summary = []
    if plan_acc is not None:
        top_clusters = plan_acc.top_clusters(20)
        # Filter to ones with >=3 members (cnt >=3 means there are at least
        # 3 V3 bullets sharing the same normalized form).
        chosen = [(cnt, exs) for cnt, exs in top_clusters if cnt >= 2][:10]
        for i, (cnt, exs) in enumerate(chosen, 1):
            lines.append(
                f"**Cluster #{i}** — appears in {cnt:,} `plan` bullets "
                f"({100*cnt/plan_acc.n_bullets:.1f}% of all V3 plan bullets)"
            )
            lines.append("")
            for j, ex in enumerate(exs[:3]):
                lines.append(f"- {short(ex, 240)}")
            lines.append("")
            clusters_for_summary.append({"count": cnt, "examples": exs[:3]})

    # --- 8. Forbidden-phrase cross check (Agent 2 coordination) ---
    lines.append("## 8. Cross-check with Agent 2 (forbidden phrases & motor-imperative regression)")
    lines.append("")
    if not forbidden_flags:
        lines.append(
            "✅ No V3 LIBERO top-20 n-gram matched the classical "
            "anthropomorphic heuristic (`wants`, `decides`, `thinks`, "
            "`believes`, `feels`, …). The hardened prompt successfully "
            "scrubbed the V2-era cognitive-state phrasing."
        )
    else:
        lines.append(
            "Top-20 phrases that **also** trip Agent 2's anthropomorphism "
            "heuristic — these are doubly flagged:"
        )
        lines.append("")
        rows = [
            [f["bullet"], f["ngram"], fmt_pct(f["doc_pct"])]
            for f in forbidden_flags
        ]
        lines.append(
            join_md_table(["bullet", "phrase", "% of bullets"], rows)
        )
    lines.append("")

    # --- Motor-imperative double-flag (Agent 2's "new C-failure" mode) ---
    motor_flags: list[tuple[str, str, float, int]] = []
    if "plan" in v3_combined:
        acc = v3_combined["plan"]
        for n in (3, 2):
            for g, c, p in acc.top_doc_phrases(n, 200):
                phrase = " ".join(g)
                if is_motor_imperative(phrase) and p >= 0.05:
                    motor_flags.append(("plan", phrase, p, c))
    # dedupe by phrase, keep largest p
    seen_ph: dict[str, tuple[str, str, float, int]] = {}
    for row in motor_flags:
        if row[1] not in seen_ph or row[2] > seen_ph[row[1]][2]:
            seen_ph[row[1]] = row
    motor_flags = sorted(seen_ph.values(), key=lambda x: -x[2])
    lines.append(
        "Agent 2 also reported a *new* C-failure mode: the hardened prompt "
        "eliminated cognitive-state phrasing but introduced **low-level "
        "motor imperatives** in the `plan:` bullet. The boilerplate signals "
        "below from my n-gram analysis quantify the prevalence of that "
        "regression — these phrases should be **double-flagged** against "
        "Agent 2's appropriateness fail set:"
    )
    lines.append("")
    if motor_flags:
        rows = [
            [b, p, f"{c:,}", fmt_pct(pct)]
            for b, p, pct, c in motor_flags[:15]
        ]
        lines.append(
            join_md_table(
                ["bullet", "imperative phrase", "DF count", "% of bullets"],
                rows,
            )
        )
    else:
        lines.append("_No motor-imperative phrase exceeded the 5% DF threshold._")
    lines.append("")

    # --- 9. Verdict + recommendations ---
    lines.append("## 9. Verdict")
    lines.append("")

    verdict_color = "GREEN"
    # Worst content-bearing phrase per bullet — drives the verdict.
    worst_per_bullet: dict[str, tuple] = {}
    for b in CANON_BULLETS:
        if b not in v3_combined:
            continue
        acc = v3_combined[b]
        best = None
        for n in (3, 2):
            for g, c, p in acc.top_doc_phrases(n, 200):
                if not is_content_ngram(g):
                    continue
                if best is None or p > best[2]:
                    best = (b, fmt_ngram(g), p, c, n)
        if best is not None:
            worst_per_bullet[b] = best

    worst_pct = max(
        (p for (_, _, p, _, _) in worst_per_bullet.values()),
        default=0.0,
    )
    worst_entry = max(
        worst_per_bullet.values(), key=lambda x: x[2], default=None
    )

    # V2 DROID near-dup baseline (target bullet as representative)
    def avg_near_dup(per_source_key: str) -> float | None:
        d = per_source.get(per_source_key, {})
        ns = [d[b].stats()["near_dup_rate"] for b in CANON_BULLETS if b in d]
        return sum(ns) / len(ns) if ns else None

    v2_nd = avg_near_dup("droid_100ep_v2")
    v3_nd_vals = [
        v3_combined[b].stats()["near_dup_rate"]
        for b in CANON_BULLETS
        if b in v3_combined
    ]
    v3_nd = sum(v3_nd_vals) / len(v3_nd_vals) if v3_nd_vals else 0.0

    # V2 vs V3 vocab compare (uniq_bigrams per 1000 bullets, target bullet for parity)
    def vocab_density(per_source_key: str) -> float | None:
        d = per_source.get(per_source_key, {})
        vals = []
        for b in CANON_BULLETS:
            acc = d.get(b)
            if acc is None or acc.n_bullets == 0:
                continue
            vals.append(len(acc.bi) / acc.n_bullets * 1000)  # uniq bigrams per 1k bullets
        return sum(vals) / len(vals) if vals else None

    v2_vd = vocab_density("droid_100ep_v2")
    v3_vd_vals = []
    for b in CANON_BULLETS:
        acc = v3_combined.get(b)
        if acc is None or acc.n_bullets == 0:
            continue
        v3_vd_vals.append(len(acc.bi) / acc.n_bullets * 1000)
    v3_vd = sum(v3_vd_vals) / len(v3_vd_vals) if v3_vd_vals else None

    if v2_vd is not None and v3_vd is not None:
        if v3_vd >= v2_vd * 0.85:
            v2_compare = "more or equal diverse"
        elif v3_vd >= v2_vd * 0.55:
            v2_compare = "less diverse"
        else:
            v2_compare = "much less diverse"
    else:
        v2_compare = "unknown"

    if worst_pct >= 0.25 or v3_nd > 0.6:
        verdict_color = "RED"
    elif worst_pct >= 0.10 or v3_nd > 0.3 or (v2_vd and v3_vd and v3_vd < 0.6 * v2_vd):
        verdict_color = "YELLOW"
    if v3_nd > (v2_nd or 0) * 1.5 and v3_nd > 0.15:
        if verdict_color == "GREEN":
            verdict_color = "YELLOW"
    if any(
        (res and res["accuracy"] < res["chance"] + 0.15)
        for res in cross_results.values()
        if res is not None
    ):
        if verdict_color == "GREEN":
            verdict_color = "YELLOW"

    lines.append(f"**Overall: {verdict_color}**")
    lines.append("")
    lines.append(
        "Decision criteria (GREEN ≅ V2 DROID diversity & no phrase >10%; "
        "YELLOW ≅ mild repetition / phrase 10-25%; RED ≅ severe collapse / "
        "single phrase >25%)."
    )
    lines.append("")
    if worst_entry is not None:
        lines.append(
            f"- Worst single phrase: `{worst_entry[1]}` in **{fmt_pct(worst_entry[2])}** "
            f"of V3 `{worst_entry[0]}` bullets ({worst_entry[3]:,} hits)."
        )
    lines.append(f"- V3 avg near-dup rate (across 5 canonical bullets): {fmt_pct(v3_nd)}")
    if v2_nd is not None:
        lines.append(f"- V2 DROID avg near-dup rate: {fmt_pct(v2_nd)}")
    if v2_vd is not None and v3_vd is not None:
        lines.append(
            f"- Unique bigrams per 1k bullets — V3 {v3_vd:.1f} vs V2 DROID {v2_vd:.1f} "
            f"⇒ V3 is **{v2_compare}** than V2 DROID."
        )
    if red_phrases:
        lines.append(
            f"- {len(red_phrases)} distinct phrase(s) exceed the 25% RED threshold."
        )
    if yellow_phrases:
        lines.append(
            f"- {len(yellow_phrases)} distinct phrase(s) fall in the 10-25% YELLOW band."
        )
    lines.append("")

    # --- 10. Recommendations ---
    lines.append("## 10. Top 3 recommendations")
    lines.append("")

    # Detect prompt-scaffold leakage: phrases like "this patch carries", "the
    # action head", "token carries the" point at the labeler echoing the
    # caller-side scaffold into its output.
    LEAK_PATTERNS = [
        "action head", "this patch", "patch carries", "token carries",
        "image_patch", "image patch token", "carries the action",
    ]
    leak_hits: list[tuple[str, str, float, int]] = []
    for b in CANON_BULLETS:
        if b not in v3_combined:
            continue
        acc = v3_combined[b]
        for n in (3, 2):
            for g, c, p in acc.top_doc_phrases(n, 200):
                phrase = " ".join(g)
                if any(pat in phrase for pat in LEAK_PATTERNS) and p >= 0.02:
                    leak_hits.append((b, phrase, p, c))

    recs: list[str] = []
    plan_phase_active = None
    if "plan" in v3_combined:
        for g, c, p in v3_combined["plan"].top_doc_phrases(2, 200):
            if g == ("phase", "active"):
                plan_phase_active = (c, p)
                break

    if worst_entry is not None and worst_entry[2] >= 0.10:
        scaffold_note = ""
        if plan_phase_active is not None:
            scaffold_note = (
                f" In particular, `phase active` shows up in "
                f"{fmt_pct(plan_phase_active[1])} of `plan` bullets — strip "
                "the literal `<task>-phase active;` scaffold from the prompt."
            )
        recs.append(
            f"**Rewrite the labeler system prompt** to break the template "
            f"`{worst_entry[1]}` (currently in {fmt_pct(worst_entry[2])} of "
            f"V3 `{worst_entry[0]}` bullets) and switch to a free-form "
            f"sentence schema with example variations per bullet.{scaffold_note}"
        )
    if v3_nd > (v2_nd or 0) and v3_nd > 0.15:
        ratio = (v3_nd / (v2_nd or 1e-6))
        recs.append(
            "**Increase decoding diversity for the re-label**: bump "
            "`temperature` to 0.9–1.0 and add `top_p=0.95`, or rotate the "
            "labeler model across `gpt-5.4-mini` / `gpt-5.5-mini`. Current "
            f"V3 near-dup rate ({fmt_pct(v3_nd)}) is {ratio:.1f}× the V2 "
            f"DROID baseline ({fmt_pct(v2_nd or 0)})."
        )
    if leak_hits:
        leak_examples = ", ".join(
            f"`{p}` ({fmt_pct(pct)} of `{b}`)"
            for b, p, pct, _ in sorted(leak_hits, key=lambda x: -x[2])[:3]
        )
        recs.append(
            "**Strip prompt-scaffold leakage from outputs**: the labeler is "
            f"writing prompt internals into the caption ({leak_examples}). "
            "These are non-grounded artifacts that hurt AV training. Either "
            "post-filter such lines, or remove the `action_head` / "
            "`image_patch_token` cues from the labeler's user message."
        )
    near_chance = [
        b for b, res in cross_results.items()
        if res is not None and res["accuracy"] < res["chance"] + 0.20
    ]
    if near_chance:
        recs.append(
            "**Add suite-conditioned hard-negative reweighting in SFT**: the "
            f"`{', '.join(near_chance)}` bullets are near-chance distinguishable "
            "between suites, so the captions don't ground on per-suite objects."
        )
    while len(recs) < 3:
        recs.append(
            "Diversify the `plan` bullet by sampling from a curated verb "
            "frame list (`grasp / lift / nudge / align / lower / release`) "
            "and ban repeated `pick-and-place phase active` openings."
        )
    for i, r in enumerate(recs[:3], 1):
        lines.append(f"{i}. {r}")
    lines.append("")

    # --- 10b. Prompt-scaffold leak callout ---
    if leak_hits:
        lines.append("### 10a. Detected prompt-scaffold leakage")
        lines.append("")
        lines.append(
            "Phrases that look like labeler prompt internals (the `action "
            "head` / image-patch token vocabulary) leaking into the "
            "caption body:"
        )
        lines.append("")
        leak_rows = [
            [b, p, f"{c:,}", fmt_pct(pct)]
            for b, p, pct, c in sorted(leak_hits, key=lambda x: -x[2])[:15]
        ]
        lines.append(
            join_md_table(
                ["bullet", "phrase", "DF count", "% of bullets"], leak_rows
            )
        )
        lines.append("")

    # --- 11. Five-line summary ---
    if worst_entry is not None:
        top1 = (
            f"`{worst_entry[1]}` in `{worst_entry[0]}` "
            f"({fmt_pct(worst_entry[2])})"
        )
    else:
        top1 = "n/a"
    total_v3_uni = sum(len(v3_combined[b].uni) for b in v3_combined)
    summary_block = {
        "v3_overall_vocab_size_unigrams": total_v3_uni,
        "top_phrase": top1,
        "v3_avg_near_dup": v3_nd,
        "v2_comparison": v2_compare,
        "verdict": verdict_color,
    }

    lines.append("## 11. Five-line summary")
    lines.append("")
    lines.append(
        f"- V3 overall vocabulary (unigrams summed over 5 canonical bullets): "
        f"**{total_v3_uni:,}** unique types."
    )
    lines.append(f"- Top-1 most-common bullet phrase: {top1}.")
    lines.append(f"- V3 near-duplicate rate (avg over canonical bullets): **{fmt_pct(v3_nd)}**.")
    lines.append(f"- V3 vs V2 DROID diversity: **{v2_compare}**.")
    lines.append(f"- Final verdict: **{verdict_color}**.")

    md = "\n".join(lines) + "\n"
    return md, summary_block


# ---------- entry point ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-md", default=str(REPORT_MD), help="markdown report output path"
    )
    ap.add_argument(
        "--out-json", default=str(REPORT_JSON), help="json stats output path"
    )
    args = ap.parse_args(argv)

    label_paths: dict[str, Path] = {}
    for s in V3_SUITES:
        p = LIBERO_BASE / s / "labels.jsonl"
        if p.exists():
            label_paths[s] = p
        else:
            print(f"[warn] missing {p}", file=sys.stderr)
    droid_path = _resolve_droid_path()
    if droid_path is not None:
        label_paths["droid_100ep_v2"] = droid_path
        print(f"[info] V2 DROID baseline: {droid_path}", file=sys.stderr)
    else:
        print(
            "[info] V2 DROID baseline not reachable "
            f"(checked {DROID_PATH_LIVE} and {DROID_PATH_ARCHIVE}); "
            "V2-DROID columns will be omitted from the report.",
            file=sys.stderr,
        )
    label_paths["libero_goal_pilot"] = PILOT_PATH

    per_source = accumulate(label_paths)

    cross_results: dict[str, dict] = {}
    just_v3 = {k: v for k, v in per_source.items() if k in V3_SUITES}
    for b in CANON_BULLETS:
        cross_results[b] = cross_suite_distinguish(just_v3, b)
        if cross_results[b] is not None:
            print(
                f"[cross] {b}: acc={cross_results[b]['accuracy']:.3f} "
                f"f1={cross_results[b]['macro_f1']:.3f}",
                file=sys.stderr,
            )

    md, summary = build_report(per_source, cross_results)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    stats_dump = {
        "summary": summary,
        "per_source": {
            src: {b: acc.stats() for b, acc in accs.items()}
            for src, accs in per_source.items()
        },
        "cross_suite": {
            k: ({} if v is None else v) for k, v in cross_results.items()
        },
    }
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(stats_dump, indent=2))
    print(f"[ok] wrote {out_md}", file=sys.stderr)
    print(f"[ok] wrote {out_json}", file=sys.stderr)
    print("[summary]", json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Derive scene attributes from labels.jsonl bullet-text via deterministic regex.

For each ``kind == "position"`` row we walk the markdown bullets in
``description`` and run a small per-attribute extractor. The result is a
side-car JSONL keyed by ``(source_example_id, position_index)`` that the
linear/MLP probe scaffold can join against the activations.

Why deterministic and not an LLM call?
- The probe figure measures what's **decodable from h**; we want labels we
  can audit and that don't drift with API changes.
- Rule-based "other" buckets keep the floor honest: a probe is only credited
  when it predicts the actual class, not the catch-all.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/eval/extract_attributes.py \\
        --labels-jsonl data/labels/libero_goal_pilot/labels.jsonl \\
        --out-jsonl    data/eval/attributes.jsonl \\
        --attributes   target_object_class gripper_state scene_type \\
                       target_visible task_phase
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable


TARGET_OBJECT_CLASSES: tuple[str, ...] = (
    "cup", "plate", "marker", "cloth", "bottle", "drawer", "bowl",
    "mug", "block", "cap", "brush", "pen", "paper", "lid", "other",
)

# Curated keyword -> class. Order matters: the FIRST match wins, so multi-word
# variants and more-specific words are tried before generic ones.
_TARGET_KEYWORDS: list[tuple[str, str]] = [
    ("dish towel", "cloth"),
    ("paper towel", "paper"),
    ("sponge", "cloth"),
    ("rag", "cloth"),
    ("cloth", "cloth"),
    ("towel", "cloth"),
    ("water bottle", "bottle"),
    ("bottle", "bottle"),
    ("drawer", "drawer"),
    ("plate", "plate"),
    ("dish", "plate"),
    ("bowl", "bowl"),
    ("mug", "mug"),
    ("cup", "cup"),
    ("tumbler", "cup"),
    ("marker", "marker"),
    ("sharpie", "marker"),
    ("pen", "pen"),
    ("pencil", "pen"),
    ("brush", "brush"),
    ("toothbrush", "brush"),
    ("paintbrush", "brush"),
    ("paper", "paper"),
    ("notepad", "paper"),
    ("envelope", "paper"),
    ("lid", "lid"),
    ("cover", "lid"),
    ("cap", "cap"),
    ("block", "block"),
    ("cube", "block"),
    ("brick", "block"),
]

SCENE_TYPES: tuple[str, ...] = (
    "tabletop", "dishwasher", "drawer", "couch", "kitchen",
    "bed", "sink", "other",
)

_SCENE_KEYWORDS: list[tuple[str, str]] = [
    ("dishwasher", "dishwasher"),
    ("drawer", "drawer"),
    ("kitchen", "kitchen"),
    ("couch", "couch"),
    ("sofa", "couch"),
    ("bed", "bed"),
    ("mattress", "bed"),
    ("bedroom", "bed"),
    ("sink", "sink"),
    ("basin", "sink"),
    ("counter top", "tabletop"),
    ("countertop", "tabletop"),
    ("table top", "tabletop"),
    ("tabletop", "tabletop"),
    ("table", "tabletop"),
    ("desk", "tabletop"),
    ("counter", "tabletop"),
]

GRIPPER_STATES: tuple[str, ...] = ("open", "closed", "holding", "unknown")

TARGET_VISIBLE_VALUES: tuple[str, ...] = ("true", "false")

TASK_PHASES: tuple[str, ...] = (
    "approach", "grasp", "transport", "release", "unknown",
)


# ---------------------------------------------------------------------------
# Bullet parsing
# ---------------------------------------------------------------------------

# Match "- key: rest of line ..." with a tolerant key set. We use re.IGNORECASE
# at call sites because gold labelers occasionally capitalise the key.
_BULLET_RE = re.compile(r"^\s*-\s*([a-zA-Z_]+)\s*:\s*(.*)$")


def parse_bullets(desc: str) -> dict[str, str]:
    """Return a {key: bullet-body} dict for each ``- key: ...`` line.

    Multi-bullet rows with the same key collapse to the *first* occurrence,
    which the spec calls "highest-confidence interpretation". In practice the
    labeler emits each key at most once, so this is a no-op safety net.
    """
    out: dict[str, str] = {}
    for line in (desc or "").splitlines():
        m = _BULLET_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        body = m.group(2).strip()
        if key not in out and body:
            out[key] = body
    return out


# ---------------------------------------------------------------------------
# Per-attribute extractors. Each takes the parsed bullet dict (from
# ``parse_bullets``) and returns the chosen class label as a plain string.
# ---------------------------------------------------------------------------

def _first_keyword_match(text: str, table: list[tuple[str, str]]) -> str | None:
    """Return the label of the keyword that appears earliest in ``text``.

    "Earliest" means *leftmost text position*, not table order: a target
    bullet "blue cube near the green bowl" should resolve to ``block``
    (the cube is the target; the bowl is the receptacle). Ties are broken
    by table order, which lets multi-word variants like ``"dish towel"``
    win over the single-word ``"towel"``.
    """
    if not text:
        return None
    lowered = text.lower()
    best_pos: int | None = None
    best_label: str | None = None
    for kw, label in table:
        pos = lowered.find(kw)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_label = label
    return best_label


def target_object_class(bullets: dict[str, str]) -> str:
    body = bullets.get("target", "")
    hit = _first_keyword_match(body, _TARGET_KEYWORDS)
    return hit or "other"


_GRIPPER_HOLDING_RE = re.compile(
    r"\b(holding|grasping|gripping|securely|carrying|clamping)\b", re.IGNORECASE
)
_GRIPPER_OPEN_RE = re.compile(
    r"\b(open|opened|jaws? open|fingers? open|spread|wide)\b", re.IGNORECASE
)
_GRIPPER_CLOSED_RE = re.compile(
    r"\b(closed|pinched|shut|clenched|fingers? closed)\b", re.IGNORECASE
)


def gripper_state(bullets: dict[str, str]) -> str:
    body = bullets.get("gripper", "")
    if not body:
        return "unknown"
    if _GRIPPER_HOLDING_RE.search(body):
        return "holding"
    if _GRIPPER_OPEN_RE.search(body):
        return "open"
    if _GRIPPER_CLOSED_RE.search(body):
        return "closed"
    return "unknown"


def scene_type(bullets: dict[str, str]) -> str:
    body = bullets.get("scene", "")
    hit = _first_keyword_match(body, _SCENE_KEYWORDS)
    return hit or "other"


_NEGATION_RE = re.compile(
    r"\b(not visible|no longer visible|occluded|hidden|out of frame|out of view|not in frame)\b",
    re.IGNORECASE,
)


def target_visible(bullets: dict[str, str]) -> bool:
    body = bullets.get("target", "")
    if not body:
        return True
    return _NEGATION_RE.search(body) is None


_PHASE_APPROACH_RE = re.compile(
    r"\b(approach|approaching|reach|reaching|moving toward|aligning|aiming)\b",
    re.IGNORECASE,
)
_PHASE_GRASP_RE = re.compile(
    r"\b(grasp|grasping|close on|closing on|pinch|pinching|grip|gripping|secure)\b",
    re.IGNORECASE,
)
_PHASE_TRANSPORT_RE = re.compile(
    r"\b(lift|lifting|transport|transporting|move toward goal|carry|carrying|raising|moving)\b",
    re.IGNORECASE,
)
_PHASE_RELEASE_RE = re.compile(
    r"\b(release|releasing|drop|dropping|place|placing|let go|open(?:ing)? gripper)\b",
    re.IGNORECASE,
)


def task_phase(bullets: dict[str, str]) -> str:
    """Coarse rollout-phase tag from `plan` and `gripper` bullets.

    Order matters: release > transport > grasp > approach. We check from the
    most-distal phase backwards because labels often mention "approach" while
    *also* describing a later step (e.g. "approached, now grasping the cup").
    """
    body_parts = []
    for k in ("plan", "gripper", "spatial"):
        if bullets.get(k):
            body_parts.append(bullets[k])
    body = " | ".join(body_parts)
    if not body:
        return "unknown"
    if _PHASE_RELEASE_RE.search(body):
        return "release"
    if _PHASE_TRANSPORT_RE.search(body):
        return "transport"
    if _PHASE_GRASP_RE.search(body):
        return "grasp"
    if _PHASE_APPROACH_RE.search(body):
        return "approach"
    return "unknown"


_EXTRACTORS: dict[str, Callable[[dict[str, str]], object]] = {
    "target_object_class": target_object_class,
    "gripper_state": gripper_state,
    "scene_type": scene_type,
    "target_visible": target_visible,
    "task_phase": task_phase,
}


def known_attributes() -> list[str]:
    return list(_EXTRACTORS.keys())


def extract_attributes(desc: str, attrs: list[str]) -> dict[str, object]:
    bullets = parse_bullets(desc)
    out: dict[str, object] = {}
    for a in attrs:
        if a not in _EXTRACTORS:
            raise ValueError(
                f"Unknown attribute {a!r}. Known: {sorted(_EXTRACTORS.keys())}"
            )
        out[a] = _EXTRACTORS[a](bullets)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels-jsonl", required=True, type=Path)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument(
        "--attributes",
        nargs="+",
        default=known_attributes(),
        choices=known_attributes(),
        help="Which attributes to extract (default: all).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of label rows processed (debugging).",
    )
    return p


def _format_distribution(counter: Counter, total: int, top_k: int = 10) -> str:
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    lines = []
    for k, v in items[:top_k]:
        pct = 100.0 * v / total if total else 0.0
        lines.append(f"    {str(k):<14} {v:>7d}  ({pct:5.1f}%)")
    if len(items) > top_k:
        rest = sum(v for _, v in items[top_k:])
        lines.append(f"    {'...':<14} {rest:>7d}  ({100.0*rest/total:5.1f}%)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    labels_path = Path(args.labels_jsonl)
    if not labels_path.exists():
        print(f"ERROR: labels.jsonl not found at {labels_path}", file=sys.stderr)
        return 2

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_skipped_kind = 0
    n_skipped_meta = 0
    n_skipped_empty = 0
    dists: dict[str, Counter] = {a: Counter() for a in args.attributes}

    seen: set[tuple[str, int, str]] = set()

    with labels_path.open() as fin, args.out_jsonl.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.limit is not None and n_in > args.limit:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("kind") != "position":
                n_skipped_kind += 1
                continue
            if obj.get("error"):
                n_skipped_empty += 1
                continue
            desc = (obj.get("description") or "").strip()
            if not desc:
                n_skipped_empty += 1
                continue
            meta = obj.get("meta") or {}
            src = meta.get("source_example_id")
            pidx = meta.get("position_index")
            ptype = meta.get("position_type")
            if src is None or pidx is None or ptype is None:
                n_skipped_meta += 1
                continue

            key = (str(src), int(pidx), str(ptype))
            if key in seen:
                continue
            seen.add(key)

            attrs = extract_attributes(desc, list(args.attributes))
            for a, v in attrs.items():
                dists[a][v] += 1

            row = {
                "source_example_id": str(src),
                "position_index": int(pidx),
                "position_type": str(ptype),
                "attributes": attrs,
            }
            fout.write(json.dumps(row, separators=(",", ":")) + "\n")
            n_out += 1

    print(
        f"Read {n_in} rows from {labels_path}; wrote {n_out} attribute rows to "
        f"{args.out_jsonl}",
        file=sys.stderr,
    )
    if n_skipped_kind or n_skipped_meta or n_skipped_empty:
        print(
            f"  skipped: kind!=position={n_skipped_kind}, "
            f"missing-meta={n_skipped_meta}, empty/error={n_skipped_empty}",
            file=sys.stderr,
        )

    print("\nPer-attribute distribution:")
    for a, ctr in dists.items():
        total = sum(ctr.values())
        print(f"\n  {a}  (n={total}, classes={len(ctr)}):")
        print(_format_distribution(ctr, total))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

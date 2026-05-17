"""Tests for ``scripts/eval/audit_prompt_hardening.py``.

Pins three properties of the SA3 V4 regression-mode extension:

1. The motor-imperative regex flags a synthetic
   ``- plan: grasp the bowl`` bullet (motor mode is detected).
2. The scaffold-leakage regex flags a synthetic
   ``- plan: action head selects the next motion`` bullet (scaffold
   mode is detected).
3. The audit imports the V4 phrase tuples directly from
   ``nla.labeling.prompts`` rather than re-deriving them, so prompt-side
   and audit-side cannot silently drift apart when SA2/SA4 update the
   ban lists.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_audit_module():
    """Load ``scripts/eval/audit_prompt_hardening.py`` as a module.

    The script lives outside the ``nla`` package so we import it by
    file path. ``src/`` is added to ``sys.path`` first so the script's
    ``from nla.labeling.prompts import ...`` resolves. The module is
    registered in ``sys.modules`` BEFORE ``exec_module`` because the
    audit script uses ``@dataclass``, and dataclass annotation
    resolution looks the module up in ``sys.modules``.
    """
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if "audit_prompt_hardening" in sys.modules:
        return sys.modules["audit_prompt_hardening"]
    script = repo / "scripts" / "eval" / "audit_prompt_hardening.py"
    spec = importlib.util.spec_from_file_location(
        "audit_prompt_hardening", script,
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["audit_prompt_hardening"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("audit_prompt_hardening", None)
        raise
    return mod


def test_audit_detects_motor_imperative() -> None:
    """A ``- plan: grasp the bowl`` bullet must trip the motor-imperative
    regex SA10 will use to regression-gate V4."""
    audit = _load_audit_module()
    bullet = "- plan: grasp the bowl"
    assert audit.V4_MOTOR_IMPERATIVE_RE.search(bullet) is not None
    # Also pin the per-row counting path so we exercise the same code
    # SA10 will call against V4 corpus.
    stats = audit.SuiteStats(name="t", path="<fake>")
    audit._scan_row(stats, {
        "example_id": "ex@p0_last_text",
        "description": (
            "- language: instruction says pick up the bowl.\n"
            "- target: yellow bowl near gripper.\n"
            "- scene: tabletop with bowl.\n"
            "- spatial: bowl directly under fingers.\n"
            "- plan: grasp the bowl and lift.\n"
        ),
        "kind": "position",
        "error": None,
        "meta": {"position_type": "last_text"},
    })
    assert stats.n_v4_motor == 1
    assert stats.bt_motor_hits["plan"] == 1
    # Sanity: the description does NOT trip scaffold or non-canonical header.
    assert stats.n_v4_scaffold == 0
    assert stats.n_v4_noncanon_header == 0


def test_audit_detects_scaffold_leakage() -> None:
    """A ``- plan: action head selects the next motion`` bullet must
    trip the scaffold-leakage regex (and not falsely fire motor or
    non-canonical-header detection on the canonical headers used).
    """
    audit = _load_audit_module()
    bullet = "- plan: action head selects the next motion"
    assert audit.V4_SCAFFOLD_LEAKAGE_RE.search(bullet) is not None
    stats = audit.SuiteStats(name="t", path="<fake>")
    audit._scan_row(stats, {
        "example_id": "ex@p0_last_text",
        "description": (
            "- language: instruction parsed.\n"
            "- target: bowl in view.\n"
            "- scene: tabletop scene.\n"
            "- spatial: bowl right of gripper.\n"
            "- plan: action head selects the next motion.\n"
        ),
        "kind": "position",
        "error": None,
        "meta": {"position_type": "last_text"},
    })
    assert stats.n_v4_scaffold == 1
    assert stats.bt_scaffold_hits["plan"] == 1
    assert stats.n_v4_motor == 0


def test_audit_detects_noncanonical_header() -> None:
    """A row with a ``- gripper:`` bullet must trip the V4
    non-canonical-header detector."""
    audit = _load_audit_module()
    stats = audit.SuiteStats(name="t", path="<fake>")
    audit._scan_row(stats, {
        "example_id": "ex@p0_image_patch",
        "description": (
            "- target: bowl near center.\n"
            "- scene: tabletop scene.\n"
            "- spatial: bowl right of gripper.\n"
            "- gripper: closing on the bowl.\n"
            "- plan: pickup phase active.\n"
        ),
        "kind": "position",
        "error": None,
        "meta": {"position_type": "image_patch"},
    })
    assert stats.n_v4_noncanon_header == 1
    assert stats.noncanon_header_hits["gripper"] == 1


def test_audit_uses_canonical_constants() -> None:
    """Audit-side phrase regex must be built from the same tuples the V4
    prompt scaffolding uses; otherwise the prompt ban list and the audit
    can silently drift apart and a future SA6 relabel could pass the
    audit while emitting forbidden phrases.
    """
    audit = _load_audit_module()
    from nla.labeling import prompts as label_prompts

    assert audit.V4_MOTOR_IMPERATIVE_PHRASES is label_prompts.V4_MOTOR_IMPERATIVE_PHRASES
    assert audit.V4_SCAFFOLD_FORBIDDEN_PHRASES is label_prompts.V4_SCAFFOLD_FORBIDDEN_PHRASES
    assert audit.V4_FORBIDDEN_HEADERS is label_prompts.V4_FORBIDDEN_HEADERS

    # And: every imperative / scaffold phrase in the canonical tuple
    # must actually match its own surface form via the audit regex
    # (catches accidental escaping mistakes when the regex is rebuilt).
    for phrase in label_prompts.V4_MOTOR_IMPERATIVE_PHRASES:
        assert audit.V4_MOTOR_IMPERATIVE_RE.search(phrase) is not None, (
            f"motor imperative phrase not matched by audit regex: {phrase!r}"
        )
    for phrase in label_prompts.V4_SCAFFOLD_FORBIDDEN_PHRASES:
        assert audit.V4_SCAFFOLD_LEAKAGE_RE.search(phrase) is not None, (
            f"scaffold phrase not matched by audit regex: {phrase!r}"
        )
    for header in label_prompts.V4_FORBIDDEN_HEADERS:
        assert audit.V4_NONCANON_HEADER_RE.match(f"- {header}: foo") is not None, (
            f"forbidden header not matched by audit regex: {header!r}"
        )

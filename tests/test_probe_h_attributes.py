"""Tests for the linear/MLP probe scaffold and its attribute extractors.

The extractor tests exercise the deterministic regex paths in
``scripts.eval.extract_attributes``. The probe-pipeline tests build a tiny
synthetic activation dump + matching attributes file and verify:

  - probes recover an attribute when it's linearly readable from h
  - probes hover near majority baseline when h is independent of the label
  - the markdown table writer emits the columns the paper needs
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_EVAL = REPO_ROOT / "scripts" / "eval"


def _load_module(mod_name: str, file_name: str):
    """Load a module from scripts/eval/ without making it a real package.

    The scripts/ tree isn't on ``sys.path`` by default (it's a CLI tree), so
    we import via ``importlib.util.spec_from_file_location`` to keep the
    tests self-contained and free of cross-script package wiring.
    """
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = SCRIPTS_EVAL / file_name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extract_attrs():
    return _load_module("extract_attributes", "extract_attributes.py")


@pytest.fixture(scope="module")
def probe_mod():
    # extract_attributes must load first so the probe module's relative
    # import of it succeeds.
    _load_module("extract_attributes", "extract_attributes.py")
    return _load_module("probe_h_attributes", "probe_h_attributes.py")


# ---------------------------------------------------------------------------
# Attribute extractor unit tests
# ---------------------------------------------------------------------------

def test_target_object_class_known_words(extract_attrs):
    desc = "- target: white plate at center, slightly off-center"
    assert extract_attrs.target_object_class(extract_attrs.parse_bullets(desc)) == "plate"

    desc2 = "- target: blue cup near the sink"
    assert extract_attrs.target_object_class(extract_attrs.parse_bullets(desc2)) == "cup"

    desc3 = "- target: blue cube on tray"
    assert extract_attrs.target_object_class(extract_attrs.parse_bullets(desc3)) == "block"


def test_target_object_class_other_fallback(extract_attrs):
    desc = "- target: weird gizmo with no known noun"
    assert extract_attrs.target_object_class(extract_attrs.parse_bullets(desc)) == "other"

    desc2 = "- scene: tabletop\n- target: shiny doohickey thing"
    assert extract_attrs.target_object_class(extract_attrs.parse_bullets(desc2)) == "other"


def test_gripper_state_open_closed_holding(extract_attrs):
    parse = extract_attrs.parse_bullets

    open_desc = "- gripper: jaws open, fingers spread, ready to reach"
    assert extract_attrs.gripper_state(parse(open_desc)) == "open"

    closed_desc = "- gripper: closed, pinched on empty air"
    assert extract_attrs.gripper_state(parse(closed_desc)) == "closed"

    holding_desc = "- gripper: holding the small blue cube securely"
    assert extract_attrs.gripper_state(parse(holding_desc)) == "holding"

    none_desc = "- target: blue block"
    assert extract_attrs.gripper_state(parse(none_desc)) == "unknown"


def test_scene_type_keywords(extract_attrs):
    parse = extract_attrs.parse_bullets

    assert extract_attrs.scene_type(parse("- scene: tabletop with cluttered toys")) == "tabletop"
    assert extract_attrs.scene_type(parse("- scene: kitchen counter near the sink")) == "kitchen"
    assert extract_attrs.scene_type(parse("- scene: open dishwasher rack")) == "dishwasher"
    assert extract_attrs.scene_type(parse("- scene: an open drawer in a wooden cabinet")) == "drawer"
    assert extract_attrs.scene_type(parse("- scene: somewhere unspecified mysterious place")) == "other"


def test_target_visible_negation(extract_attrs):
    parse = extract_attrs.parse_bullets

    assert extract_attrs.target_visible(parse("- target: blue cup on the rim")) is True

    assert extract_attrs.target_visible(parse("- target: cup is not visible behind the bowl")) is False

    assert extract_attrs.target_visible(parse("- target: marker fully occluded by cloth")) is False

    assert extract_attrs.target_visible(parse("- target: brush hidden under the towel")) is False


def test_task_phase_keywords(extract_attrs):
    parse = extract_attrs.parse_bullets

    desc_approach = "- plan: approach the blue cube from the left side"
    assert extract_attrs.task_phase(parse(desc_approach)) == "approach"

    desc_grasp = "- plan: grasp the cup with a firm pinch closure"
    assert extract_attrs.task_phase(parse(desc_grasp)) == "grasp"

    desc_transport = "- plan: lift the block and move toward the green bowl"
    assert extract_attrs.task_phase(parse(desc_transport)) == "transport"

    desc_release = "- plan: place the cube into the bowl, then release"
    assert extract_attrs.task_phase(parse(desc_release)) == "release"

    desc_unknown = "- plan: study the table"
    assert extract_attrs.task_phase(parse(desc_unknown)) == "unknown"


def test_extract_attributes_dispatch(extract_attrs):
    desc = (
        "- scene: tabletop with assorted toys\n"
        "- target: bright blue cube near the green bowl\n"
        "- gripper: jaws open, ready to reach\n"
        "- plan: approach the blue cube from the left"
    )
    out = extract_attrs.extract_attributes(
        desc,
        ["target_object_class", "gripper_state", "scene_type", "target_visible", "task_phase"],
    )
    assert out["target_object_class"] == "block"
    assert out["gripper_state"] == "open"
    assert out["scene_type"] == "tabletop"
    assert out["target_visible"] is True
    assert out["task_phase"] == "approach"


def test_extract_attributes_unknown_attribute_raises(extract_attrs):
    with pytest.raises(ValueError):
        extract_attrs.extract_attributes("- scene: tabletop", ["nonexistent_attr"])


# ---------------------------------------------------------------------------
# Probe-pipeline tests
# ---------------------------------------------------------------------------

def _make_synthetic_rows(probe_mod, n_per_class=120, dim=16, sep=4.0, seed=0,
                         independent=False, position_type="anchor",
                         n_episodes=20):
    """Build JoinedRow lists with attribute either readable or not from h.

    ``sep`` controls the linear-separability margin: positive vs negative
    along dim 0 with a Gaussian cloud of std=1.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cls_idx, label in enumerate(["pos", "neg"]):
        for k in range(n_per_class):
            x = rng.normal(loc=0.0, scale=1.0, size=(dim,)).astype(np.float32)
            if not independent:
                # Encode label as the SIGN of the first dim.
                shift = sep if cls_idx == 0 else -sep
                x[0] = rng.normal(loc=shift, scale=1.0)
            rows.append(probe_mod.JoinedRow(
                activation=x,
                label=label,
                episode_index=k % n_episodes,
                position_type=position_type,
                source_example_id=f"src_{cls_idx}_{k:04d}",
                position_index=k,
            ))
    rng.shuffle(rows)
    return rows


def test_probe_recovers_linearly_separable_attribute(probe_mod):
    rows = _make_synthetic_rows(probe_mod, n_per_class=200, dim=16, sep=4.0, seed=0)
    train_idx, val_idx = probe_mod.episode_split_indices(
        rows, seed=0, held_out_fraction=0.2,
    )
    assert train_idx and val_idx
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    X_train, y_train = probe_mod._stack(train_rows)
    X_val, y_val = probe_mod._stack(val_rows)

    acc, f1, _ = probe_mod.fit_linear_probe(X_train, y_train, X_val, y_val, seed=0)
    assert acc >= 0.95, f"linear probe acc was {acc:.3f} on a linearly-separable signal"
    assert f1 >= 0.9


def test_probe_chance_on_random_attribute(probe_mod):
    rows = _make_synthetic_rows(
        probe_mod, n_per_class=200, dim=16, seed=1, independent=True,
    )
    train_idx, val_idx = probe_mod.episode_split_indices(
        rows, seed=0, held_out_fraction=0.2,
    )
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    X_train, y_train = probe_mod._stack(train_rows)
    X_val, y_val = probe_mod._stack(val_rows)

    base_acc, _ = probe_mod.majority_baseline(y_train, y_val)
    acc, _, _ = probe_mod.fit_linear_probe(X_train, y_train, X_val, y_val, seed=0)

    n_val = len(y_val)
    sigma = float(np.sqrt(0.25 / max(n_val, 1)))
    assert acc <= base_acc + 1.5 * sigma + 0.05, (
        f"probe acc {acc:.3f} should be near majority {base_acc:.3f} "
        f"(sigma={sigma:.3f}) when h is independent of the label."
    )


def test_episode_split_holds_out_whole_episodes(probe_mod):
    rng = np.random.default_rng(0)
    rows = []
    for ep in range(10):
        for step in range(20):
            rows.append(probe_mod.JoinedRow(
                activation=rng.normal(size=(8,)).astype(np.float32),
                label="x",
                episode_index=ep,
                position_type="anchor",
                source_example_id=f"e{ep}_s{step}",
                position_index=step,
            ))
    train_idx, val_idx = probe_mod.episode_split_indices(
        rows, seed=0, held_out_fraction=0.2,
    )
    train_eps = {rows[i].episode_index for i in train_idx}
    val_eps = {rows[i].episode_index for i in val_idx}
    assert train_eps.isdisjoint(val_eps), (
        f"episode leak: train eps={train_eps}, val eps={val_eps}"
    )


def test_run_probe_sweep_writes_markdown_table(probe_mod, tmp_path: Path):
    rows = _make_synthetic_rows(probe_mod, n_per_class=120, dim=8, sep=3.0, seed=2)
    rows_by_attr = {"target_object_class": rows}

    results = probe_mod.run_probe_sweep(
        rows_by_attr,
        probe_kinds=("linear",),
        position_types=("anchor",),
        held_out_fraction=0.2,
        max_rows_per_attr=None,
        seed=0,
    )
    assert results, "probe sweep produced no rows"
    assert any(r["position_type"] == "all" for r in results)

    md = probe_mod.render_markdown_table(results)
    for col in ("attribute", "position_type", "probe", "n_val",
                "majority_acc", "probe_acc", "macro_f1"):
        assert col in md, f"markdown missing column {col!r}"

    md_path = tmp_path / "probe_table.md"
    probe_mod.write_markdown_table(results, md_path)
    assert md_path.exists() and md_path.read_text().startswith("|")

    jsonl_path = tmp_path / "probe_results.jsonl"
    probe_mod.write_results_jsonl(results, jsonl_path)
    parsed = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert parsed and all("probe_acc" in r for r in parsed)


def test_av_accuracy_uses_same_extractor(probe_mod):
    rows = [
        probe_mod.JoinedRow(
            activation=np.zeros(4, dtype=np.float32),
            label="cup",
            episode_index=0,
            position_type="anchor",
            source_example_id="src_a",
            position_index=0,
        ),
        probe_mod.JoinedRow(
            activation=np.zeros(4, dtype=np.float32),
            label="plate",
            episode_index=0,
            position_type="anchor",
            source_example_id="src_b",
            position_index=0,
        ),
    ]
    av_predictions = {
        ("src_a", 0): "- target: small blue cup near the rim",
        ("src_b", 0): "- target: weird gizmo with no known noun",
    }
    acc, n = probe_mod.av_accuracy_for_attribute(
        rows, "target_object_class", av_predictions,
    )
    assert n == 2
    assert acc == pytest.approx(0.5)


def test_majority_baseline_picks_most_common(probe_mod):
    y_train = np.array(["a", "a", "a", "b"], dtype=object)
    y_val = np.array(["a", "b", "b"], dtype=object)
    acc, cls = probe_mod.majority_baseline(y_train, y_val)
    assert cls == "a"
    assert acc == pytest.approx(1.0 / 3.0)

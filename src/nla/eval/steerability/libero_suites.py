"""LIBERO benchmark suite metadata for multi-suite CF mining.

Label rows carry ``meta.suite`` in ``{goal, spatial, object, 10}``. Sim-success
GRPO predicates currently score **Goal** tasks only, so non-goal suites are
mined with Goal ``target_task`` / ``target_env_name`` while preserving the
source row's suite-specific ``source_task`` for provenance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# ``meta.suite`` value in labels.jsonl -> LIBERO benchmark dict key.
SUITE_LABEL_TO_BENCHMARK: dict[str, str] = {
    "goal": "libero_goal",
    "spatial": "libero_spatial",
    "object": "libero_object",
    "10": "libero_10",
}

BDDL_SUBDIR: dict[str, str] = {
    "goal": "libero_goal",
    "spatial": "libero_spatial",
    "object": "libero_object",
    "10": "libero_10",
}

BDDL_ROOT = (
    _REPO_ROOT
    / "third_party/Isaac-GR00T/external_dependencies/LIBERO/libero/libero"
    / "bddl_files"
)


def bddl_dir_for_suite(suite_label: str) -> Path:
    if suite_label not in BDDL_SUBDIR:
        raise KeyError(f"unknown suite label {suite_label!r}")
    return BDDL_ROOT / BDDL_SUBDIR[suite_label]


@lru_cache(maxsize=8)
def _language_maps_for_benchmark(benchmark_key: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (lang_lower->task_name, task_name->lang) for one benchmark suite."""
    import sys

    libero_root = _REPO_ROOT / "third_party/Isaac-GR00T/external_dependencies/LIBERO"
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[benchmark_key]()
    lang_to_task: dict[str, str] = {}
    task_to_lang: dict[str, str] = {}
    for task_id in range(task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        lang = str(task.language).strip()
        name = str(task.name).strip()
        lang_to_task[lang.lower()] = name
        task_to_lang[name] = lang
    return lang_to_task, task_to_lang


def resolve_suite_instruction(instruction: str, suite_label: str) -> str | None:
    """Map a demo instruction to the canonical LIBERO task name for ``suite_label``."""
    if suite_label not in SUITE_LABEL_TO_BENCHMARK:
        return None
    benchmark_key = SUITE_LABEL_TO_BENCHMARK[suite_label]
    lang_to_task, task_to_lang = _language_maps_for_benchmark(benchmark_key)
    key = instruction.strip().lower()
    if key in lang_to_task:
        return lang_to_task[key]
    # Tolerant: underscores vs spaces for task id passthrough.
    alt = key.replace(" ", "_")
    if alt in task_to_lang:
        return alt
    return None


def canonical_to_instruction_for_suite(suite_label: str) -> dict[str, str]:
    """task_name -> demo language string for one suite."""
    if suite_label not in SUITE_LABEL_TO_BENCHMARK:
        raise KeyError(suite_label)
    _, task_to_lang = _language_maps_for_benchmark(SUITE_LABEL_TO_BENCHMARK[suite_label])
    return dict(task_to_lang)


def all_task_names_for_suite(suite_label: str) -> list[str]:
    canon = canonical_to_instruction_for_suite(suite_label)
    return sorted(canon.keys())

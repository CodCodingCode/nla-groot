#!/usr/bin/env python
"""Lightweight Isaac-GR00T / nla-groot environment sanity check.

The vendored ``third_party/Isaac-GR00T`` package pins ``requires-python == 3.10.*``
and a fixed torch/transformers stack. Use a 3.10 venv with ``gr00t`` installed
(``pip install -e third_party/Isaac-GR00T``) for extraction. The overlay script
``overlay_av_video.py`` does **not** import GR00t; it only needs torch + peft +
Qwen weights in whichever env you load the AV checkpoint.

Checks (informational; exit code 0 unless ``--strict``):
  - Python version (warn if not 3.10)
  - ``import gr00t`` and ``Gr00tPolicy`` (same entry points as extraction)
  - Whether ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` is set (gated Cosmos processor)

With ``--strict``, exits 1 if Python is not 3.10 or gr00t imports fail (useful for CI
or full extraction envs). Overlay-only workflows may omit gr00t; run without ``--strict``.

Also suggests optional follow-ups::

    PYTHONPATH=src python scripts/models/smoke_load.py
    python scripts/extraction/run_extract.py --help
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if Python is not 3.10 or gr00t / Gr00tPolicy cannot be imported",
    )
    args = p.parse_args(argv)

    print("python:", sys.version.replace("\n", " "))

    py310 = sys.version_info[:2] == (3, 10)
    if not py310:
        print(
            "WARNING: Isaac-GR00T pyproject pins requires-python == 3.10.* "
            f"(you have {sys.version_info.major}.{sys.version_info.minor})",
            file=sys.stderr,
        )

    gr00t_ok = True
    try:
        import gr00t  # noqa: F401

        print("import gr00t: OK")
    except Exception as e:
        gr00t_ok = False
        print(f"import gr00t: FAILED ({type(e).__name__}: {e})")

    try:
        from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: F401

        print("Gr00tPolicy: OK")
    except Exception as e:
        gr00t_ok = False
        print(f"Gr00tPolicy: FAILED ({type(e).__name__}: {e})")

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print("HF_TOKEN or HUGGING_FACE_HUB_TOKEN set:", bool(tok))
    if not tok:
        print(
            "NOTE: Gated models (e.g. nvidia/Cosmos-Reason2-2B for the processor) "
            "need a token; see https://huggingface.co/nvidia/Cosmos-Reason2-2B",
            file=sys.stderr,
        )

    print()
    print("Optional: PYTHONPATH=src python scripts/models/smoke_load.py")
    print("Optional: python scripts/extraction/run_extract.py --help")

    if args.strict and (not py310 or not gr00t_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

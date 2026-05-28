#!/usr/bin/env bash
# Launch the warm interactive LIBERO REPL.
#
# Picks the libero venv (which has gr00t + libero + robosuite + mujoco
# installed), sets PYTHONPATH to include our src/, and execs ipython -i
# scripts/eval/play_repl.py. After startup you have a warm `client`,
# `play(task, init_id, steer_text)` helper, and a cached LiberoEnv per task.
#
# Usage:
#   scripts/eval/play.sh                  # open the REPL
#   scripts/eval/play.sh -c "play('put_the_bowl_on_the_plate')"   # one-shot
#
# Environment overrides:
#   LIBERO_PY  path to the libero venv python
#              (default: third_party/Isaac-GR00T/.../libero_uv/.venv/bin/python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_LIBERO_PY="${REPO_ROOT}/third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
LIBERO_PY="${LIBERO_PY:-${DEFAULT_LIBERO_PY}}"
LIBERO_DIR="$(dirname "${LIBERO_PY}")"
IPYTHON_BIN="${LIBERO_DIR}/ipython"

if [[ ! -x "${LIBERO_PY}" ]]; then
    echo "ERROR: libero venv python missing at ${LIBERO_PY}" >&2
    echo "       set LIBERO_PY=/path/to/python and re-run" >&2
    exit 2
fi
if [[ ! -x "${IPYTHON_BIN}" ]]; then
    # ipython isn't strictly required — fall back to python -i.
    IPYTHON_BIN="${LIBERO_PY}"
    IPY_FLAGS=(-i)
else
    IPY_FLAGS=(-i)
fi

PLAY_REPL="${REPO_ROOT}/scripts/eval/play_repl.py"
if [[ ! -f "${PLAY_REPL}" ]]; then
    echo "ERROR: missing ${PLAY_REPL}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Forward any extra args (e.g. `-c "play('...')"`) verbatim.
exec "${IPYTHON_BIN}" "${IPY_FLAGS[@]}" "${PLAY_REPL}" "$@"

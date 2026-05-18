#!/usr/bin/env bash
# Launcher for the NlaSteerGr00tPolicy ZMQ server.
#
# Thin wrapper around scripts/eval/run_gr00t_server_nla_steer.py that:
#   - auto-resolves --ar-dir from $SFT_DIR/ar/
#   - validates SFT_DIR, AR_DIR, and that host:port is free
#   - launches the server in the background under nohup with the requested GPU
#   - tees stdout/stderr to $LOG_DIR/server_<unix-ts>.log and saves the PID to
#     $LOG_DIR/server.pid
#   - polls the log for "Server is ready and listening" (emitted by
#     gr00t.policy.server_client.PolicyServer.run after the socket binds and
#     the request loop starts)
#   - on success prints "READY host=...:port pid=..." and exits 0; the server
#     stays running in the background.
#   - on failure within --ready-timeout seconds (default 120) it SIGTERM/SIGKILLs
#     the child, removes the PID file, and exits 1 with a tail of the log.
#
# Usage:
#   scripts/eval/launch_steer_server.sh \
#       --sft-dir   data/sft/libero_goal_pilot_v3 \
#       [--port     5555]                          \
#       [--host     localhost]                     \
#       [--gpu-id   0]                             \
#       [--log-dir  <SFT_DIR>/steer_server_logs/]  \
#       [--ready-timeout 120]                      \
#       --                                         \
#       --model-path     checkpoints/.../libero_goal \
#       --embodiment-tag LIBERO_PANDA               \
#       --steer-text-file my_steer_bullets.txt
#
# Everything after `--` is forwarded verbatim to the python launcher, so the
# orchestrator can pass --model-path, --embodiment-tag, --steer-text(-file),
# --placement, --blend, etc. without this script having to know about them.
#
# Environment overrides:
#   PY_BIN  path to the python interpreter (default: <repo>/.venv/bin/python)
#   Repo .env is sourced when present (e.g. GROOT_MODEL_PATH, OPENAI_API_KEY).
#   If no --model-path is passed after --, defaults from GROOT_MODEL_PATH or built-in LIBERO ckpt dir.

set -euo pipefail

PORT=5555
HOST=localhost
GPU_ID=0
SFT_DIR=""
LOG_DIR=""
READY_TIMEOUT=120
PASSTHROUGH=()

usage() {
    grep -E '^#( |$)' "$0" | sed -E 's/^#( |$)//'
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sft-dir)        SFT_DIR="$2";        shift 2 ;;
        --port)           PORT="$2";           shift 2 ;;
        --host)           HOST="$2";           shift 2 ;;
        --gpu-id)         GPU_ID="$2";         shift 2 ;;
        --log-dir)        LOG_DIR="$2";        shift 2 ;;
        --ready-timeout)  READY_TIMEOUT="$2";  shift 2 ;;
        -h|--help)        usage ;;
        --)               shift; PASSTHROUGH=("$@"); break ;;
        *)                echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${SFT_DIR}" ]]; then
    echo "ERROR: --sft-dir is required" >&2
    exit 2
fi
if [[ ! -d "${SFT_DIR}" ]]; then
    echo "ERROR: --sft-dir does not exist: ${SFT_DIR}" >&2
    exit 2
fi
AR_DIR="${SFT_DIR%/}/ar"
if [[ ! -d "${AR_DIR}" ]]; then
    echo "ERROR: ${AR_DIR} does not exist (expected <SFT_DIR>/ar/ — produced by run_sft)" >&2
    exit 2
fi
if [[ -z "${LOG_DIR}" ]]; then
    LOG_DIR="${SFT_DIR%/}/steer_server_logs"
fi
mkdir -p "${LOG_DIR}"

# ----- port-in-use check (ss -> nc -> /dev/tcp) -----
port_in_use() {
    local host="$1" port="$2"
    if command -v ss >/dev/null 2>&1; then
        if ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE "[:.]${port}\$"; then
            return 0
        fi
    fi
    if command -v nc >/dev/null 2>&1; then
        if nc -z "${host}" "${port}" >/dev/null 2>&1; then
            return 0
        fi
    fi
    if (exec 3<>"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

if port_in_use "${HOST}" "${PORT}"; then
    echo "ERROR: port ${HOST}:${PORT} is already in use" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env" || true
    set +a
fi

PY_LAUNCHER="${REPO_ROOT}/scripts/eval/run_gr00t_server_nla_steer.py"
PY_BIN="${PY_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -f "${PY_LAUNCHER}" ]]; then
    echo "ERROR: python launcher missing at ${PY_LAUNCHER}" >&2
    exit 2
fi
if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: python binary not found / not executable at ${PY_BIN}" >&2
    echo "       set PY_BIN=/path/to/python and re-run" >&2
    exit 2
fi

TS=$(date +%s)
LOG_FILE="${LOG_DIR}/server_${TS}.log"
PID_FILE="${LOG_DIR}/server.pid"

if [[ -f "${PID_FILE}" ]]; then
    OLD_PID="$(cat "${PID_FILE}" || true)"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "ERROR: ${PID_FILE} already names a live process (pid=${OLD_PID}); stop it first" >&2
        exit 2
    fi
    rm -f "${PID_FILE}"
fi

_default_ckpt_rel="checkpoints/GR00T-N1.7-LIBERO/libero_goal"
_have_mp=0
for ((i = 0; i < ${#PASSTHROUGH[@]}; i++)); do
    if [[ "${PASSTHROUGH[$i]}" == "--model-path" ]]; then
        _have_mp=1
        break
    fi
done
if [[ "${_have_mp}" -eq 0 ]]; then
    _gp="${GROOT_MODEL_PATH:-}"
    if [[ -z "${_gp}" && -d "${REPO_ROOT}/${_default_ckpt_rel}" ]]; then
        _gp="${REPO_ROOT}/${_default_ckpt_rel}"
    fi
    if [[ -n "${_gp}" ]]; then
        if [[ "${_gp}" != /* ]]; then
            _gp="${REPO_ROOT}/${_gp#./}"
        fi
        PASSTHROUGH+=(--model-path "${_gp}")
    fi
fi
unset _have_mp _gp _default_ckpt_rel

_have_mp=0
for ((i = 0; i < ${#PASSTHROUGH[@]}; i++)); do
    if [[ "${PASSTHROUGH[$i]}" == "--model-path" ]]; then
        _have_mp=1
        break
    fi
done
if [[ "${_have_mp}" -eq 0 ]]; then
    echo "ERROR: GR00T --model-path is required." >&2
    echo "  Pass --model-path after '--', or export GROOT_MODEL_PATH in repo .env (or shell)." >&2
    echo "  Default lookup: repo_root/checkpoints/GR00T-N1.7-LIBERO/libero_goal" >&2
    exit 2
fi
unset _have_mp

echo "Launching NlaSteerGr00tPolicy server"
echo "  SFT dir:        ${SFT_DIR}"
echo "  AR dir:         ${AR_DIR}"
echo "  Host:port:      ${HOST}:${PORT}"
echo "  GPU id:         ${GPU_ID}"
echo "  Log file:       ${LOG_FILE}"
echo "  PID file:       ${PID_FILE}"
echo "  Ready timeout:  ${READY_TIMEOUT}s"
echo "  Forwarded args: ${PASSTHROUGH[*]:-<none>}"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
nohup "${PY_BIN}" "${PY_LAUNCHER}" \
    --ar-dir "${AR_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${PASSTHROUGH[@]}" \
    >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
disown "${SERVER_PID}" 2>/dev/null || true

READY_PATTERN='Server is ready and listening'
deadline=$(( $(date +%s) + READY_TIMEOUT ))
ready=0
while [[ $(date +%s) -lt ${deadline} ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: server process exited before becoming ready" >&2
        echo "---- tail of ${LOG_FILE} ----" >&2
        tail -n 80 "${LOG_FILE}" >&2 || true
        rm -f "${PID_FILE}"
        exit 1
    fi
    if grep -q "${READY_PATTERN}" "${LOG_FILE}" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 1
done

if [[ ${ready} -ne 1 ]]; then
    echo "ERROR: server did not become ready within ${READY_TIMEOUT}s" >&2
    echo "---- tail of ${LOG_FILE} ----" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    exit 1
fi

echo "READY host=${HOST}:${PORT} pid=${SERVER_PID}"
exit 0

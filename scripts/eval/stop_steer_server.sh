#!/usr/bin/env bash
# Stop an NlaSteerGr00tPolicy server previously started with
# scripts/eval/launch_steer_server.sh.
#
# Reads the PID from $LOG_DIR/server.pid, sends SIGTERM, waits up to
# --grace-seconds (default 10) for a clean exit, then escalates to SIGKILL.
# Removes the PID file on success. Treats a missing/stale PID file as a no-op
# (exit 0) so the orchestrator can call this idempotently.
#
# Usage:
#   scripts/eval/stop_steer_server.sh --log-dir DIR [--grace-seconds 10]
#
# Exit codes:
#   0  process was already gone, or was successfully stopped
#   1  process still alive after SIGKILL, or PID file is malformed
#   2  bad arguments

set -euo pipefail

LOG_DIR=""
GRACE_SECONDS=10

usage() {
    grep -E '^#( |$)' "$0" | sed -E 's/^#( |$)//'
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)        LOG_DIR="$2";        shift 2 ;;
        --grace-seconds)  GRACE_SECONDS="$2";  shift 2 ;;
        -h|--help)        usage ;;
        *)                echo "unknown arg: $1" >&2; usage ;;
    esac
done

if [[ -z "${LOG_DIR}" ]]; then
    echo "ERROR: --log-dir is required" >&2
    exit 2
fi

PID_FILE="${LOG_DIR}/server.pid"
if [[ ! -f "${PID_FILE}" ]]; then
    echo "no PID file at ${PID_FILE}; nothing to stop"
    exit 0
fi

PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
if ! [[ "${PID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: malformed PID '${PID}' in ${PID_FILE}" >&2
    exit 1
fi

if ! kill -0 "${PID}" 2>/dev/null; then
    echo "no live process with PID ${PID}; removing stale ${PID_FILE}"
    rm -f "${PID_FILE}"
    exit 0
fi

echo "sending SIGTERM to PID ${PID}"
kill -TERM "${PID}" 2>/dev/null || true

deadline=$(( $(date +%s) + GRACE_SECONDS ))
while [[ $(date +%s) -lt ${deadline} ]]; do
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "PID ${PID} exited cleanly after SIGTERM"
        rm -f "${PID_FILE}"
        exit 0
    fi
    sleep 1
done

echo "PID ${PID} still alive after ${GRACE_SECONDS}s; sending SIGKILL"
kill -KILL "${PID}" 2>/dev/null || true
sleep 1
if kill -0 "${PID}" 2>/dev/null; then
    echo "ERROR: PID ${PID} still alive after SIGKILL" >&2
    exit 1
fi
rm -f "${PID_FILE}"
echo "PID ${PID} killed"
exit 0

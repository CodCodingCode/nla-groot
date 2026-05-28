#!/usr/bin/env bash
# Status / cleanup helper for NlaSteerGr00tPolicy ZMQ servers.
#
# Scans data/sft/*/steer_server_logs/server.pid, reports per-server: pid,
# alive/stale, port, SFT dir, uptime, log path. Optionally health-checks
# (--check) and cleans stale pidfiles (--clean).
#
# Usage:
#   scripts/eval/steer_server_status.sh           # list state of all servers
#   scripts/eval/steer_server_status.sh --check   # also ping each live server
#   scripts/eval/steer_server_status.sh --clean   # remove stale pidfiles
#
# Exits 0 if at least one live server is found (and, with --check, healthy);
# exits 1 if nothing is running or a checked server fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_BIN="${PY_BIN:-${REPO_ROOT}/.venv/bin/python}"

DO_CHECK=0
DO_CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) DO_CHECK=1; shift ;;
        --clean) DO_CLEAN=1; shift ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed -E 's/^#( |$)//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mapfile -t PIDFILES < <(find "${REPO_ROOT}/data/sft" -maxdepth 4 -name "server.pid" 2>/dev/null | sort)

if [[ ${#PIDFILES[@]} -eq 0 ]]; then
    echo "no server.pid files under data/sft/*/steer_server_logs/"
    exit 1
fi

printf "%-8s %-6s %-12s %-7s %s\n" "PID" "STATE" "UPTIME" "PORT" "SFT_DIR"

any_live=0
any_check_fail=0
for pf in "${PIDFILES[@]}"; do
    pid="$(cat "${pf}" 2>/dev/null || true)"
    sft_dir="$(cd "$(dirname "${pf}")/.." && pwd)"
    rel_sft="${sft_dir#${REPO_ROOT}/}"
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        printf "%-8s %-6s %-12s %-7s %s\n" "${pid:-?}" "stale" "-" "-" "${rel_sft}"
        if [[ ${DO_CLEAN} -eq 1 ]]; then
            rm -f "${pf}"
            echo "  cleaned ${pf}"
        fi
        continue
    fi
    any_live=1
    uptime="$(ps -p "${pid}" -o etime= 2>/dev/null | tr -d ' ' || echo '?')"
    # Pull --port from the running process's argv (defaults 5555).
    port="$(tr '\0' ' ' < /proc/${pid}/cmdline 2>/dev/null \
        | grep -oE -- '--port [0-9]+' | awk '{print $2}' || true)"
    port="${port:-5555}"
    printf "%-8s %-6s %-12s %-7s %s\n" "${pid}" "live" "${uptime}" "${port}" "${rel_sft}"

    if [[ ${DO_CHECK} -eq 1 ]]; then
        if PYTHONPATH="${REPO_ROOT}/src" "${PY_BIN}" \
            "${REPO_ROOT}/scripts/eval/health_check_steer_server.py" \
            --host localhost --port "${port}" --skip-action >/dev/null 2>&1; then
            echo "  health: ok"
        else
            echo "  health: FAIL (port ${port})"
            any_check_fail=1
        fi
    fi
done

if [[ ${any_live} -eq 0 ]]; then
    exit 1
fi
if [[ ${DO_CHECK} -eq 1 && ${any_check_fail} -eq 1 ]]; then
    exit 1
fi
exit 0

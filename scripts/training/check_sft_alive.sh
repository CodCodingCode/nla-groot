#!/usr/bin/env bash
# Exit 0 if <sft_dir>/metrics.jsonl was modified within the last N seconds, else exit 1.
# Usage: check_sft_alive.sh <sft_dir> [max_age_seconds=120]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <sft_dir> [max_age_seconds=120]" >&2
  exit 2
fi

SFT_DIR="$1"
MAX_AGE="${2:-120}"
METRICS="${SFT_DIR%/}/metrics.jsonl"

if [[ ! -f "$METRICS" ]]; then
  echo "[check_sft_alive] missing: $METRICS" >&2
  exit 1
fi

NOW="$(date +%s)"
MTIME="$(stat -c %Y "$METRICS" 2>/dev/null || stat -f %m "$METRICS")"
AGE=$(( NOW - MTIME ))

if (( AGE <= MAX_AGE )); then
  echo "[check_sft_alive] alive age=${AGE}s metrics=$METRICS"
  exit 0
fi

echo "[check_sft_alive] stale age=${AGE}s (max=${MAX_AGE}s) metrics=$METRICS" >&2
exit 1

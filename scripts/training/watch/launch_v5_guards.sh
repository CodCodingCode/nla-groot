#!/usr/bin/env bash
# Launch all three V5 watchdog scripts in the background.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
GUARD_DIR="${REPO_ROOT}/logs/v5_guard"
mkdir -p "$GUARD_DIR"

for s in watch_v5_labels_guard.sh watch_v5_sft_guard.sh watch_v5_post_guard.sh; do
  chmod +x "scripts/training/watch/$s"
done

launch() {
  local name="$1"
  local script="$2"
  local pidf="${GUARD_DIR}/${name}.pid"
  local logf="${GUARD_DIR}/${name}.boot"
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "already running: $name pid=$(cat "$pidf")"
    return
  fi
  nohup bash "scripts/training/watch/$script" >> "$logf" 2>&1 &
  echo $! > "$pidf"
  echo "started $name pid=$(cat "$pidf") log=$logf"
}

launch labels_guard watch_v5_labels_guard.sh
launch sft_guard watch_v5_sft_guard.sh
launch post_guard watch_v5_post_guard.sh

echo "---"
echo "Monitor:"
echo "  tail -f logs/v5_guard/labels_guard.log"
echo "  tail -f logs/v5_guard/sft_guard.log"
echo "  tail -f logs/v5_guard/post_guard.log"
echo "Flags: labels_ready.flag | sft_started.flag | sft_success.flag | pipeline_complete.flag"

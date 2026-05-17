#!/usr/bin/env bash
#
# Archive all DROID-prefixed artifacts out of the live data tree.
#
# What this does:
#   For each top-level data subdir in {sft, labels, activations, grpo, grpo_ab}
#   we move every entry that matches the literal glob `droid_*` into
#   data/_archive_droid/<subdir>/. We rely on `mv` so the operation is
#   instant when source and destination share a filesystem (typical: the
#   data tree is a single mount).
#
#   A MANIFEST.txt is written under data/_archive_droid/ recording, for
#   each archived entry: origin path, destination path, size (du -sh),
#   and the timestamp of the move. Existing MANIFEST.txt is appended to
#   so the script is safe to re-run.
#
# What this does NOT do:
#   - It does NOT delete anything. Recovery is `mv` back to the original
#     parent. If you want to free the GB, run something like
#     `rm -rf data/_archive_droid` AFTER you have re-confirmed everything
#     V3 needs lives elsewhere.
#   - It does NOT touch directories that don't literally start with
#     "droid_" (e.g. data/activations/bridge_pilot is left in place).
#   - It does NOT touch loose files under data/grpo_ab/ that lack the
#     droid_ prefix; those files (judge_v2.jsonl, leverage_v2.csv, ...)
#     are V2/GRPO A/B artifacts that may still be referenced by reports.
#
# Flags:
#   --dry-run   Print what would happen but make no on-disk changes.
#   --yes       Skip the interactive confirmation prompt.
#
# Example:
#   ./scripts/migration/archive_droid.sh --dry-run
#   ./scripts/migration/archive_droid.sh --yes
#
# This script is intentionally conservative: a single `mv` failure on any
# entry causes the whole script to abort (set -e) so we never end up with
# a half-archived tree silently.

set -euo pipefail

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=1 ;;
    --yes|-y)   ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "[archive_droid] unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/../.."

DATA_ROOT="data"
ARCHIVE_ROOT="${DATA_ROOT}/_archive_droid"
MANIFEST="${ARCHIVE_ROOT}/MANIFEST.txt"
SUBDIRS=(sft labels activations grpo grpo_ab)

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[archive_droid] FATAL: ${DATA_ROOT}/ not found (run from repo root)" >&2
  exit 1
fi

run() {
  # Echo + execute (or echo only under dry-run).
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Pass 1: enumerate candidates so we can print a plan + ask for confirmation.
# ---------------------------------------------------------------------------
declare -a PLAN_SRCS=()
declare -a PLAN_DSTS=()
declare -a PLAN_SIZES=()
for sub in "${SUBDIRS[@]}"; do
  src_parent="${DATA_ROOT}/${sub}"
  if [[ ! -d "$src_parent" ]]; then
    continue
  fi
  # shellcheck disable=SC2231
  for entry in ${src_parent}/droid_*; do
    [[ -e "$entry" ]] || continue
    name="$(basename "$entry")"
    dst="${ARCHIVE_ROOT}/${sub}/${name}"
    sz="$(du -sh "$entry" 2>/dev/null | awk '{print $1}')"
    PLAN_SRCS+=("$entry")
    PLAN_DSTS+=("$dst")
    PLAN_SIZES+=("$sz")
  done
done

total="${#PLAN_SRCS[@]}"
if [[ "$total" -eq 0 ]]; then
  echo "[archive_droid] no droid_* entries found under data/{sft,labels,activations,grpo,grpo_ab} -- nothing to do."
  exit 0
fi

echo "[archive_droid] plan: ${total} entries to archive"
for i in "${!PLAN_SRCS[@]}"; do
  printf "  %-6s  %s  ->  %s\n" "${PLAN_SIZES[$i]}" "${PLAN_SRCS[$i]}" "${PLAN_DSTS[$i]}"
done
echo "[archive_droid] destination root: ${ARCHIVE_ROOT}/"
echo "[archive_droid] manifest:         ${MANIFEST}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[archive_droid] dry-run mode: no on-disk changes will be made."
  exit 0
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  printf "[archive_droid] proceed? [y/N] "
  read -r reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "[archive_droid] aborted." ; exit 1 ;;
  esac
fi

# ---------------------------------------------------------------------------
# Pass 2: execute. mkdir the destination subdir, mv each entry, append to
# MANIFEST.txt. We mv into a subdir-of-archive root so the original
# data/<sub>/ shape is preserved for easy restore.
# ---------------------------------------------------------------------------
run mkdir -p "${ARCHIVE_ROOT}"

# Make sure MANIFEST exists with a header on first run.
if [[ ! -f "$MANIFEST" ]]; then
  {
    echo "# DROID archive manifest"
    echo "# Format: <timestamp_utc>\t<size>\t<origin>\t<destination>"
    echo "# Created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Host:    $(hostname)"
    echo "# Repo:    $(git -C . rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
  } > "$MANIFEST"
fi

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
moved=0
for i in "${!PLAN_SRCS[@]}"; do
  src="${PLAN_SRCS[$i]}"
  dst="${PLAN_DSTS[$i]}"
  sz="${PLAN_SIZES[$i]}"

  # mkdir parent first (e.g. data/_archive_droid/sft/).
  dst_parent="$(dirname "$dst")"
  run mkdir -p "$dst_parent"

  # If the destination already exists, refuse rather than clobber.
  if [[ -e "$dst" ]]; then
    echo "[archive_droid] SKIP (destination exists): $dst" >&2
    continue
  fi

  run mv "$src" "$dst"
  printf "%s\t%s\t%s\t%s\n" "$ts" "$sz" "$src" "$dst" >> "$MANIFEST"
  moved=$((moved + 1))
done

echo "[archive_droid] done: ${moved}/${total} entries archived."
echo "[archive_droid] manifest: $MANIFEST"

#!/usr/bin/env bash
# A/B sweep for compare_cf_steer_checkpoints.py. Iterates (sim_batch_size,
# sim_n_workers) cells against the live GR00T steer server, captures peak GPU
# memory + host RSS + wall-clock, parses per-cell predicate rates, picks a
# winner, writes results.csv + REPORT.md.
#
# Designed to be safe to run while the GR00T steer server is up; does NOT
# touch the server process. Cells run strictly sequentially so wall-clock
# numbers are comparable.
#
# Env overrides:
#   N_SAMPLES (default 4), STEER_PORT (default 5556), STEER_PID (default
#   auto-detected from `pgrep -f run_gr00t_server_nla_steer.py`),
#   OUT_ROOT (default data/eval/ab_sweep_<UTC_ts>), CELLS_OVERRIDE
#   ("bs w" pairs newline-separated; replaces the default grid).
set -o pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-src}"

PYTHON=".venv/bin/python"
LIBERO_PY="third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
SFT_DIR="data/sft/libero_4suite_v5_base_qwen"
GRPO_AV_DIR="data/grpo/libero_4suite_v5_sim_grpo_v2_pilot/av"
ACT_ROOT="data/activations/libero_4suite_v4_combined"
PAIRS="data/eval/grpo_steer_holdout/libero_4suite_cf_eval_all_pairs.jsonl"
EXCLUDE="data/eval/grpo_steer_holdout/libero_4suite_cf_eval_all_train_manifest.json"

STEER_PORT="${STEER_PORT:-5556}"
N_SAMPLES="${N_SAMPLES:-2}"
TS=$(date -u +%Y%m%d_%H%M%S)
OUT_ROOT="${OUT_ROOT:-data/eval/ab_sweep_${TS}}"
mkdir -p "$OUT_ROOT"

# Auto-detect steer server PID if not provided.
if [[ -z "${STEER_PID:-}" ]]; then
  STEER_PID=$(pgrep -f run_gr00t_server_nla_steer.py | head -1)
fi
if [[ -z "$STEER_PID" ]] || ! kill -0 "$STEER_PID" 2>/dev/null; then
  echo "[ab-sweep] FATAL: GR00T steer server not running (set STEER_PID or start the server first)" >&2
  exit 2
fi

# Tight default grid: each cell at n=2 takes ~5-15 min. Full sweep ~50-70 min.
# bs=1,w=4 is the "no GR00T batching, only parallel workers" control. bs=4,w=2
# is the current launcher default. bs=8,w=2 is the projected winner. bs=16/w=2
# and bs=32/w=1 are OOM probes — placed last so a crash doesn't lose other
# cells.
DEFAULT_CELLS="\
1 4
4 2
8 1
8 2
16 2
32 1"
CELLS="${CELLS_OVERRIDE:-$DEFAULT_CELLS}"

OOM_GPU_MIB=75000          # cell flagged as OOM above this
WINNER_GPU_MIB=70000       # winner selection threshold (stricter)

CSV="${OUT_ROOT}/results.csv"
REPORT="${OUT_ROOT}/REPORT.md"
echo "cell_id,bs,w,wall_s,exit_code,peak_gpu_mib,peak_host_rss_mib,sft_pred,grpo_pred,delta_pred,error_count,oom_flag" > "$CSV"

echo "[ab-sweep] out=${OUT_ROOT} steer_pid=${STEER_PID} n_samples=${N_SAMPLES}"
echo "[ab-sweep] cells:"
printf '  %s\n' $CELLS | paste -d ' ' - -

# poll_resources <compare_pid> <out_file>
# Appends "<gpu_mib> <host_rss_mib>" once per second while compare is alive.
poll_resources() {
  local cpid="$1"
  local f="$2"
  : > "$f"
  while kill -0 "$cpid" 2>/dev/null; do
    local gpu_mib
    gpu_mib=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
              | awk -F',' -v p="$STEER_PID" '$1+0==p+0 {sub(/^[ \t]+/, "", $2); print $2+0; exit}')
    [[ -z "$gpu_mib" ]] && gpu_mib=0
    local rss_kb
    rss_kb=$(awk '/^VmRSS:/{print $2; exit}' "/proc/${cpid}/status" 2>/dev/null)
    [[ -z "$rss_kb" ]] && rss_kb=0
    local rss_mib=$((rss_kb / 1024))
    echo "$gpu_mib $rss_mib" >> "$f"
    sleep 1
  done
}

# parse_one_json <json_path> <key>
parse_field() {
  local jpath="$1"
  local key="$2"
  "$PYTHON" - <<PY 2>/dev/null
import json, sys
try:
    d = json.load(open("${jpath}"))
    v = d.get("${key}", "")
    print(v if v != "" else "")
except Exception:
    print("")
PY
}

count_errors() {
  local jpath="$1"
  "$PYTHON" - <<PY 2>/dev/null
import json
try:
    d = json.load(open("${jpath}"))
    n = 0
    for s in d.get("samples", []) or []:
        for c in (s.get("conditions") or {}).values():
            if c.get("error"):
                n += 1
    print(n)
except Exception:
    print("")
PY
}

# Loop over cells.
echo "$CELLS" | while read -r BS W; do
  [[ -z "$BS" ]] && continue
  CID="bs${BS}_w${W}"
  CELL_DIR="${OUT_ROOT}/cell_${CID}"
  mkdir -p "$CELL_DIR"
  POLL="${CELL_DIR}/poll.txt"
  LOG="${CELL_DIR}/log.txt"
  JSON="${CELL_DIR}/out.json"

  # Steer-server liveness check; abort sweep if dead.
  if ! kill -0 "$STEER_PID" 2>/dev/null; then
    echo "[ab-sweep] FATAL: steer server PID ${STEER_PID} died before cell ${CID}; aborting" >&2
    echo "${CID},${BS},${W},0,255,0,0,,,,,1" >> "$CSV"
    break
  fi

  echo "[ab-sweep] cell ${CID} starting (bs=${BS}, w=${W})" | tee -a "$LOG"
  t0=$(date +%s)
  "$PYTHON" scripts/eval/compare_cf_steer_checkpoints.py \
    --sft-dir "$SFT_DIR" \
    --grpo-av-dir "$GRPO_AV_DIR" \
    --pairs-path "$PAIRS" \
    --activations-root "$ACT_ROOT" \
    --exclude-ids-path "$EXCLUDE" \
    --require-held-out --deterministic-order --forbid-sim-cache \
    --n-samples "$N_SAMPLES" --seed 0 \
    --conditions sft_av,grpo_av \
    --intent-arms matched,mismatched_source \
    --causal-arms semantic,matched_null,wrong_placement \
    --policy-port "$STEER_PORT" \
    --sim-rollout-python "$LIBERO_PY" \
    --sim-batch-size "$BS" \
    --sim-n-workers "$W" \
    --out-json "$JSON" >> "$LOG" 2>&1 &
  CMP_PID=$!
  poll_resources "$CMP_PID" "$POLL" &
  POLL_PID=$!
  wait "$CMP_PID"
  RC=$?
  t1=$(date +%s)
  WALL=$((t1 - t0))
  kill "$POLL_PID" 2>/dev/null || true
  wait "$POLL_PID" 2>/dev/null || true

  PEAK_GPU=$(awk '{if($1+0>m)m=$1+0} END{print m+0}' "$POLL" 2>/dev/null)
  [[ -z "$PEAK_GPU" ]] && PEAK_GPU=0
  PEAK_RSS=$(awk '{if($2+0>m)m=$2+0} END{print m+0}' "$POLL" 2>/dev/null)
  [[ -z "$PEAK_RSS" ]] && PEAK_RSS=0

  SFT_PRED=$(parse_field "$JSON" "sft_av_predicate_rate")
  GRPO_PRED=$(parse_field "$JSON" "grpo_av_predicate_rate")
  DELTA_PRED=$(parse_field "$JSON" "delta_predicate_rate_grpo_minus_sft")
  ERR=$(count_errors "$JSON")
  [[ -z "$ERR" ]] && ERR=""

  OOM=0
  if [[ "$RC" -ne 0 ]] || [[ "$PEAK_GPU" -gt "$OOM_GPU_MIB" ]]; then
    OOM=1
  fi

  echo "${CID},${BS},${W},${WALL},${RC},${PEAK_GPU},${PEAK_RSS},${SFT_PRED},${GRPO_PRED},${DELTA_PRED},${ERR},${OOM}" >> "$CSV"
  echo "[ab-sweep] cell ${CID}: wall=${WALL}s rc=${RC} gpu_peak=${PEAK_GPU}MiB rss_peak=${PEAK_RSS}MiB pred(sft/grpo/delta)=${SFT_PRED}/${GRPO_PRED}/${DELTA_PRED} errs=${ERR} oom=${OOM}"
  sleep 10
done

# Build REPORT.md with sorted/annotated table + winner.
"$PYTHON" - "$CSV" "$REPORT" "$WINNER_GPU_MIB" "$N_SAMPLES" "$OUT_ROOT" "$STEER_PID" "$SFT_DIR" "$GRPO_AV_DIR" <<'PY'
import csv, sys, os
csv_path, report_path, winner_gpu_mib, n_samples, out_root, steer_pid, sft_dir, grpo_av_dir = sys.argv[1:9]
winner_gpu_mib = int(winner_gpu_mib)

rows = []
with open(csv_path) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        rows.append(r)

def _f(x):
    try: return float(x)
    except: return None
def _i(x):
    try: return int(x)
    except: return None

# Pick the lowest-bs, lowest-w cell as the predicate-rate reference.
ref = None
for r in rows:
    if r["exit_code"] == "0" and r["sft_pred"]:
        if ref is None or (_i(r["bs"]), _i(r["w"])) < (_i(ref["bs"]), _i(ref["w"])):
            ref = r

ref_sft = _f(ref["sft_pred"]) if ref else None
ref_grpo = _f(ref["grpo_pred"]) if ref else None

# Annotate rows with correctness vs reference + winner eligibility.
for r in rows:
    sft = _f(r["sft_pred"]); grpo = _f(r["grpo_pred"])
    if ref is None:
        r["pred_match"] = "no_ref"
    elif sft is None or grpo is None:
        r["pred_match"] = "no_data"
    elif sft == ref_sft and grpo == ref_grpo:
        r["pred_match"] = "ok"
    else:
        r["pred_match"] = f"DRIFT sft={sft - ref_sft:+.3f} grpo={grpo - ref_grpo:+.3f}"
    eligible = (
        r["exit_code"] == "0"
        and r.get("oom_flag") == "0"
        and r["pred_match"] == "ok"
        and _i(r["peak_gpu_mib"]) is not None
        and _i(r["peak_gpu_mib"]) <= winner_gpu_mib
    )
    r["eligible"] = "yes" if eligible else "no"

# Winner: lowest wall_s among eligible; tie-break smaller w then smaller bs.
elig = [r for r in rows if r["eligible"] == "yes"]
elig.sort(key=lambda r: (_i(r["wall_s"]), _i(r["w"]), _i(r["bs"])))
winner = elig[0] if elig else None

# Render markdown.
lines = []
lines.append(f"# A/B sweep CF eval batching")
lines.append("")
lines.append(f"- Generated: `{os.environ.get('USER','?')}@{os.uname().nodename}`, out=`{out_root}`")
lines.append(f"- Samples per cell: {n_samples} (full arm matrix; 48 rollouts/cell)")
lines.append(f"- Steer server PID: {steer_pid}")
lines.append(f"- SFT: `{sft_dir}`; GRPO AV: `{grpo_av_dir}`")
lines.append(f"- Reference cell (lowest bs/w with valid output): `{ref['cell_id'] if ref else 'none'}`")
lines.append(f"- Winner GPU cap: {winner_gpu_mib} MiB")
lines.append("")
lines.append("## Cells (sorted by wall_s)")
lines.append("")
lines.append("| cell | bs | w | wall (s) | rc | peak GPU MiB | peak RSS MiB | sft pred | grpo pred | delta | err | oom | pred match | eligible |")
lines.append("|------|----|----|----------|----|--------------|--------------|----------|-----------|-------|-----|-----|------------|----------|")
for r in sorted(rows, key=lambda x: (int(x["wall_s"] or 999999),)):
    lines.append("| {cell_id} | {bs} | {w} | {wall_s} | {exit_code} | {peak_gpu_mib} | {peak_host_rss_mib} | {sft_pred} | {grpo_pred} | {delta_pred} | {error_count} | {oom_flag} | {pred_match} | {eligible} |".format(**r))
lines.append("")
if winner:
    lines.append(f"## Winner: `{winner['cell_id']}` (bs={winner['bs']}, w={winner['w']})")
    lines.append("")
    lines.append(f"- Wall-clock: **{winner['wall_s']}s** ({float(winner['wall_s'])/60:.1f} min)")
    lines.append(f"- Peak GPU: {winner['peak_gpu_mib']} MiB (cap {winner_gpu_mib})")
    lines.append(f"- Peak host RSS: {winner['peak_host_rss_mib']} MiB")
    lines.append(f"- Predicate rates: sft={winner['sft_pred']} grpo={winner['grpo_pred']} delta={winner['delta_pred']}")
    bsl = next((r for r in rows if r["cell_id"] == "bs1_w1"), None)
    if bsl and bsl["exit_code"] == "0":
        bsl_wall = max(1, int(bsl["wall_s"]))
        win_wall = max(1, int(winner["wall_s"]))
        lines.append(f"- Speedup vs bs1_w1 baseline: **{bsl_wall / win_wall:.2f}x** ({bsl_wall}s -> {win_wall}s)")
    lines.append("")
    lines.append("## Reproducer (screen-tier validation)")
    lines.append("")
    lines.append("```bash")
    lines.append(f"EVAL_TIER=screen \\")
    lines.append(f"SIM_BATCH_SIZE={winner['bs']} SIM_N_WORKERS={winner['w']} \\")
    lines.append(f"OUT_DIR=data/eval/grpo_steer_holdout_screen_<ts> \\")
    lines.append(f"bash scripts/eval/run_grpo_steer_holdout.sh")
    lines.append("```")
else:
    lines.append("## Winner: NONE (no eligible cells)")
    lines.append("")
    lines.append("All cells failed correctness gate, OOM'd, or crashed. Inspect cell_*/log.txt.")
lines.append("")
open(report_path, "w").write("\n".join(lines))
print(f"[ab-sweep] wrote {report_path}")
if winner:
    print(f"[ab-sweep] WINNER cell={winner['cell_id']} bs={winner['bs']} w={winner['w']} wall={winner['wall_s']}s")
PY

echo
echo "[ab-sweep] DONE. CSV: $CSV   Report: $REPORT"

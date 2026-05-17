#!/usr/bin/env bash
# Build a 2-row x 4-col v1-vs-v3 steerability comparison video for one seed.
#
# Row 1 (v1 AR, data/sft/libero_goal_pilot): baseline | bowl->plate | wine bottle | bowl->stove
# Row 2 (v3 AR, data/sft/libero_4suite_v3):  baseline | bowl->plate | wine bottle | bowl->stove
#
# Includes a title band on top and a takeaway band on the bottom.
#
# Usage: bash scripts/eval/make_v1_vs_v3_grid.sh <seed> [out_mp4]
set -euo pipefail

SEED="${1:-0}"
ROOT="${ROOT:-data/eval/steerability_v1_vs_v3}"
OUT="${2:-${ROOT}/comparisons/v1_vs_v3_seed${SEED}.mp4}"
ENV_DIR="libero_sim__put_the_bowl_on_the_plate"

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PANEL_W=512
PANEL_H=256
CAP_H=44        # per-panel caption strip
TITLE_H=80      # top title bar
FOOTER_H=110    # bottom takeaway band

V_BASE="${ROOT}/conditions/baseline/${ENV_DIR}/seed_${SEED}/rollout.mp4"
V1_BP="${ROOT}/conditions/steer_bowl_plate/${ENV_DIR}/seed_${SEED}/rollout.mp4"
V1_WR="${ROOT}/conditions/steer_wine_rack/${ENV_DIR}/seed_${SEED}/rollout.mp4"
V1_BS="${ROOT}/conditions/steer_bowl_stove/${ENV_DIR}/seed_${SEED}/rollout.mp4"
V3_BP="${ROOT}/conditions/steer_bowl_plate_v3/${ENV_DIR}/seed_${SEED}/rollout.mp4"
V3_WR="${ROOT}/conditions/steer_wine_rack_v3/${ENV_DIR}/seed_${SEED}/rollout.mp4"
V3_BS="${ROOT}/conditions/steer_bowl_stove_v3/${ENV_DIR}/seed_${SEED}/rollout.mp4"

for f in "$V_BASE" "$V1_BP" "$V1_WR" "$V1_BS" "$V3_BP" "$V3_WR" "$V3_BS"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")"
CAP_DIR="$(mktemp -d)"
trap 'rm -rf "$CAP_DIR"' EXIT

# Top title (single short line).
echo "GR00T + NLA steerability  ::  v1 AR vs v3 AR  ::  seed ${SEED}" \
  > "$CAP_DIR/title.txt"

# Per-cell captions kept short so they fit a 512px panel at 16px.
# Numbers below are means across all 3 seeds so they're stable across grids.
echo "v1 baseline (no steer)   any=1.0  final=0.33   bowl bumped off plate"            > "$CAP_DIR/cap_v1_base.txt"
echo "v1 'bowl->plate' (match) succ 0/3  min_ee 0.204m   wrapper kills plan"           > "$CAP_DIR/cap_v1_bp.txt"
echo "v1 'wine->rack' (redir)  succ 0/3  min_ee 0.203m   does NOT go to bottle"        > "$CAP_DIR/cap_v1_wr.txt"
echo "v1 'bowl->stove' (redir) succ 0/3  min_ee 0.238m   does NOT go to stove"         > "$CAP_DIR/cap_v1_bs.txt"
echo "v3 baseline (no steer)   any=1.0  final=0.33   bowl bumped off plate"            > "$CAP_DIR/cap_v3_base.txt"
echo "v3 'bowl->plate' (match) succ 0/3  min_ee 0.108m   2x closer to bowl, no grasp"  > "$CAP_DIR/cap_v3_bp.txt"
echo "v3 'wine->rack' (redir)  succ 0/3  min_ee 0.188m   trajectory shifts, not to bottle" > "$CAP_DIR/cap_v3_wr.txt"
echo "v3 'bowl->stove' (redir) succ 0/3  min_ee 0.221m   trajectory shifts, not to stove"  > "$CAP_DIR/cap_v3_bs.txt"

# Bottom takeaway -- two lines so it stays readable.
cat > "$CAP_DIR/footer.txt" <<'FOOT'
v3 IS steering: matching-prompt brings the gripper 2x closer to the bowl, redirect prompts shift the EE trajectory in a prompt-specific way (min_ee 0.108 / 0.188 / 0.221 m).
But steering is DISRUPTIVE, not REDIRECTIVE -- it destroys the baseline grasp plan rather than re-targeting it. No steered cell achieves success at the new (or original) goal.
FOOT

# Build one labeled panel from input index N with caption file cap_X.txt.
panel () {
  local idx="$1" cap="$2" out="$3"
  local file="$CAP_DIR/${cap}.txt"
  printf '[%s:v]scale=%s:%s:flags=lanczos,setsar=1,pad=%s:%s+%s:0:%s:black,drawtext=fontfile=%s:textfile=%s:x=(w-tw)/2:y=12:fontsize=16:fontcolor=white[%s]' \
    "$idx" "$PANEL_W" "$PANEL_H" "$PANEL_W" "$PANEL_H" "$CAP_H" "$CAP_H" "$FONT" "$file" "$out"
}

FILT="$(panel 0 cap_v1_base P0)"
FILT+=";$(panel 1 cap_v1_bp   P1)"
FILT+=";$(panel 2 cap_v1_wr   P2)"
FILT+=";$(panel 3 cap_v1_bs   P3)"
FILT+=";$(panel 4 cap_v3_base P4)"
FILT+=";$(panel 5 cap_v3_bp   P5)"
FILT+=";$(panel 6 cap_v3_wr   P6)"
FILT+=";$(panel 7 cap_v3_bs   P7)"

FILT+=";[P0][P1][P2][P3]hstack=inputs=4[ROW1]"
FILT+=";[P4][P5][P6][P7]hstack=inputs=4[ROW2]"
FILT+=";[ROW1][ROW2]vstack=inputs=2[STACK]"

# Add title at top + takeaway at bottom in a single pad+drawtext chain.
FILT+=";[STACK]pad=iw:ih+${TITLE_H}+${FOOTER_H}:0:${TITLE_H}:black,"
FILT+="drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/title.txt:x=(w-tw)/2:y=24:fontsize=28:fontcolor=0xFFD700,"
FILT+="drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/footer.txt:x=(w-tw)/2:y=h-${FOOTER_H}+18:fontsize=18:fontcolor=white[FINAL]"

ffmpeg -y -hide_banner -loglevel error \
  -i "$V_BASE" -i "$V1_BP" -i "$V1_WR" -i "$V1_BS" \
  -i "$V_BASE" -i "$V3_BP" -i "$V3_WR" -i "$V3_BS" \
  -filter_complex "$FILT" \
  -map "[FINAL]" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart \
  "$OUT"

echo "wrote: $OUT"
ffprobe -v error -show_entries stream=width,height,duration -of default=noprint_wrappers=1 "$OUT" | head

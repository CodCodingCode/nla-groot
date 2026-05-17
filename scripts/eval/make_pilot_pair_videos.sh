#!/usr/bin/env bash
# Build three side-by-side comparison videos from the LIBERO Goal pilot rollouts
# and a vertical 3-up stack. Output goes under data/sim_rollouts/libero_goal_pilot/.
set -euo pipefail

ROOT=/home/ubuntu/nla-groot
ROLL_DIR="$ROOT/data/sim_rollouts/libero_goal_pilot"
OUT_DIR="$ROLL_DIR"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

BASELINE="$ROLL_DIR/baseline/40280b06-e710-4860-a1f2-0edb8a734dc5_s1.mp4"
STEER_A="$ROLL_DIR/steer_a/c00d7eae-41f3-44aa-b091-51c4508dbb87_s1.mp4"
STEER_B="$ROLL_DIR/steer_b/f29fb14c-3e7a-485a-89d3-0ffddf02cd39_s1.mp4"
STEER_BAGG="$ROLL_DIR/steer_b_agg/d957843f-e202-4ff2-9b5b-ddcfe2f4cca1_s0.mp4"

CAP_DIR=/tmp/libero_pilot_captions
mkdir -p "$CAP_DIR"

# Per-panel top + bottom captions
write_cap () { printf '%s' "$2" > "$CAP_DIR/$1"; }

write_cap baseline_top  'BASELINE — no NLA steer'
write_cap baseline_bot  'Result: 3/3 success — solved in ~80 sim steps'

write_cap a_top  'STEER A — AR(bowl on plate) at one image patch (blend=1.0)'
write_cap a_bot  'Result: 1/1 success — solved in ~80 sim steps'

write_cap b_top  'STEER B — AR(wine bottle) at one image patch (blend=1.0)'
write_cap b_bot  'Result: 1/1 success — light steer is below threshold'

write_cap bagg_top  'AGGRESSIVE STEER — AR(wine bottle) at every image patch (blend=1.5)'
write_cap bagg_bot  'Result: 0/3 success — timed out at 200 sim steps'

write_cap pair1_header  'Pair 1 — Wrapper sanity: no steer vs matching prompt (both still solve)'
write_cap pair2_header  'Pair 2 — Prompt sensitivity at image_patch + blend=1.0 (steer too weak to flip)'
write_cap pair3_header  'Pair 3 — Strong steer flips behavior: no steer vs AR(wine) at every patch + blend=1.5'
write_cap master_header 'NLA backbone steering on libero_sim/put_the_bowl_on_the_plate · GR00T-N1.7-LIBERO/libero_goal · seed=0'

# Each panel: scale to 1024x512, pad video to 5.0s by holding last frame, then
# add 80px black band on top and 60px on bottom for captions.
mk_panel_filter () {
  local idx=$1 top_cap=$2 bot_cap=$3 top_color=${4:-white} bot_color=${5:-0xFFFFFF}
  echo "[${idx}:v]scale=1024:512:flags=lanczos,tpad=stop_mode=clone:stop_duration=max(0\\,5.0-(N+1)/20),pad=1024:632:0:80:black,drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${top_cap}:x=(w-tw)/2:y=22:fontsize=22:fontcolor=${top_color},drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${bot_cap}:x=(w-tw)/2:y=h-32:fontsize=20:fontcolor=${bot_color}[P${idx}]"
}

# Simpler: pad each input to exactly 5s using -t and -shortest tricks; just use
# tpad=stop_mode=clone:stop_duration=10 then trim.
mk_pair () {
  local left=$1 right=$2 ltop=$3 lbot=$4 lcol=$5 rtop=$6 rbot=$7 rcol=$8 header=$9 outfile=${10}
  ffmpeg -y -i "$left" -i "$right" -filter_complex "
[0:v]scale=1024:512:flags=lanczos,tpad=stop_mode=clone:stop_duration=10,trim=duration=5,setpts=PTS-STARTPTS,pad=1024:632:0:80:black,
drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${ltop}:x=(w-tw)/2:y=22:fontsize=22:fontcolor=white,
drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${lbot}:x=(w-tw)/2:y=h-32:fontsize=20:fontcolor=${lcol}[L];
[1:v]scale=1024:512:flags=lanczos,tpad=stop_mode=clone:stop_duration=10,trim=duration=5,setpts=PTS-STARTPTS,pad=1024:632:0:80:black,
drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${rtop}:x=(w-tw)/2:y=22:fontsize=22:fontcolor=white,
drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${rbot}:x=(w-tw)/2:y=h-32:fontsize=20:fontcolor=${rcol}[R];
[L][R]hstack=inputs=2[stacked];
[stacked]pad=2048:692:0:60:black,
drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/${header}:x=(w-tw)/2:y=20:fontsize=22:fontcolor=0xFFD700[final]
" -map "[final]" -r 20 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$outfile" 2>&1 | tail -3
  echo "wrote: $outfile"
}

PAIR1="$OUT_DIR/pair1_baseline_vs_steerA.mp4"
PAIR2="$OUT_DIR/pair2_steerA_vs_steerB.mp4"
PAIR3="$OUT_DIR/pair3_baseline_vs_steerB_agg.mp4"

mk_pair "$BASELINE" "$STEER_A"  baseline_top baseline_bot 0x7CFC00 a_top a_bot 0x7CFC00 pair1_header "$PAIR1"
mk_pair "$STEER_A"  "$STEER_B"  a_top a_bot 0x7CFC00 b_top b_bot 0x7CFC00 pair2_header "$PAIR2"
mk_pair "$BASELINE" "$STEER_BAGG" baseline_top baseline_bot 0x7CFC00 bagg_top bagg_bot 0xFF6347 pair3_header "$PAIR3"

# Vertical 3-up stack with master title (60px) on top.
THREE="$OUT_DIR/three_pairs_stack.mp4"
ffmpeg -y -i "$PAIR1" -i "$PAIR2" -i "$PAIR3" -filter_complex "
[0:v][1:v][2:v]vstack=inputs=3[v3];
[v3]pad=2048:ih+60:0:60:black,
drawtext=fontfile=${FONT}:textfile=${CAP_DIR}/master_header:x=(w-tw)/2:y=20:fontsize=24:fontcolor=0xFFD700[final]
" -map "[final]" -r 20 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "$THREE" 2>&1 | tail -3
echo "wrote: $THREE"

ls -lh "$PAIR1" "$PAIR2" "$PAIR3" "$THREE"

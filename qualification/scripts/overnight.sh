#!/usr/bin/env bash
# Overnight round 2 on the pod: wait for the running bake-off, then frames -> round-2 candidates
# -> leaderboard (stacker) -> final multi-seed retrain -> assemble.
#   nohup bash scripts/overnight.sh > /workspace/overnight.log 2>&1 &
# Edit the four paths below to match the round-1 command (see /workspace/round1.log).
set -uo pipefail
cd "$(dirname "$0")/.."

IMAGES=/workspace/img2048                     # photos used in round 1
LABELS=/workspace/prequal/train/train_dataset.csv
VIDEO_ROOT=/workspace/prequal/train           # folder the CSV video paths are relative to
WORK=/workspace/day_one
GROUPS="--groups-file $WORK/groups.csv"
FRAMES=/workspace/frames
ROUND2=vlm_qwen3vl_2b_crops,vlm_qwen3vl_4b_crops

stamp() { echo "=== $(date '+%F %T') $*"; }

stamp "waiting for the running bake-off"
while pgrep -f "bakeoff.py run|day_one.py run" >/dev/null; do sleep 60; done

stamp "free space"; df -h /workspace /

stamp "video frames"
python -c "import imageio_ffmpeg" 2>/dev/null || command -v ffmpeg >/dev/null || pip install -q imageio-ffmpeg
python scripts/extract_video_frames.py --csv "$LABELS" --root "$VIDEO_ROOT" --out "$FRAMES" \
  || { stamp "frame extraction failed: continuing without frames"; FRAMES=""; }
EXTRA=(); [ -n "$FRAMES" ] && EXTRA=(--extra-images "$FRAMES/images" --extra-labels "$FRAMES/labels.csv")

stamp "round-2 candidates: $ROUND2"
python scripts/bakeoff.py run --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" \
  --candidates "$ROUND2" $GROUPS "${EXTRA[@]}" || stamp "a round-2 candidate failed: leaderboard uses what finished"

stamp "leaderboard (greedy + stacker)"
python scripts/bakeoff.py leaderboard --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" $GROUPS \
  | tee "$WORK/leaderboard_round2.txt"

stamp "final retrain on all data, seeds 0,1,2"
python scripts/bakeoff.py final --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" \
  --final "$WORK/runs_final" --out "$WORK/weights_final" "${EXTRA[@]}"

stamp "done"; df -h /workspace

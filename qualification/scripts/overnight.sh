#!/usr/bin/env bash
# Round 2 on the pod(s). Revised 2026-11-22 after the round-1 error review:
#  - main errors are wound/amputation predicted as no_injury on unseen casualties (40% / 42%);
#    left/right swaps are rare (21 of 853 erroneous pairs); post-hoc class biases do NOT help
#    (nested 0.708 vs 0.714), so the levers are more injury examples (video frames) and close-ups.
#  - each change is measured on its own: "4B + frames" vs round 1 "4B", and "4B crops + frames".
#  - weak DINOv2 probe checkpoints are pruned first (their OOF files stay for the leaderboard).
#  - the final retrain uses frames only if a frames-trained candidate won, then is TIMED.
#
#   Two pods on the same volume (recommended, halves the wait):
#     pod A:  ROUND2=vlm_qwen3vl_4b_frames WAIT_FOR=vlm_qwen3vl_4b_crops nohup bash scripts/overnight.sh > /workspace/overnight_a.log 2>&1 &
#     pod B:  ROUND2=vlm_qwen3vl_4b_crops ONLY_TRAIN=1 nohup bash scripts/overnight.sh > /workspace/overnight_b.log 2>&1 &
#   One pod: ROUND2=vlm_qwen3vl_4b_frames,vlm_qwen3vl_4b_crops nohup bash scripts/overnight.sh > /workspace/overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

IMAGES=${IMAGES:-/workspace/img2048}                  # photos used in round 1
LABELS=${LABELS:-/workspace/prequal/train/train_dataset.csv}
VIDEO_ROOT=${VIDEO_ROOT:-/workspace/prequal/train}    # folder the CSV video paths are relative to
WORK=${WORK:-/workspace/day_one}
GROUPS="--groups-file $WORK/groups.csv"
FRAMES=${FRAMES:-/workspace/frames}
PER_VIDEO=${PER_VIDEO:-3}
ROUND2=${ROUND2:-vlm_qwen3vl_4b_frames,vlm_qwen3vl_4b_crops}
WAIT_FOR=${WAIT_FOR:-}          # comma list of candidates another pod is training; wait for them
ONLY_TRAIN=${ONLY_TRAIN:-0}     # 1 = train ROUND2 and stop (the other pod does leaderboard/final)
export HF_HOME=${HF_HOME:-/root/hf} TORCH_HOME=${TORCH_HOME:-/root/torch}

stamp() { echo "=== $(date '+%F %T') $*"; }
space() { df -h /workspace | tail -1; }
guard() {  # stop before the volume fills (round 1 lost a run that way)
  use=$(df --output=pcent /workspace | tail -1 | tr -dc 0-9)
  [ "$use" -lt 95 ] || { stamp "VOLUME ${use}% FULL: stopping"; exit 3; }
}

stamp "waiting for any running bake-off"
while pgrep -f "bakeoff.py (run|final)|day_one.py run" >/dev/null; do sleep 60; done

stamp "prune weak checkpoints (keep OOF predictions)"; space
for d in dinov2_l_reg_probe dinov2_l_reg_limb_probe dinov2_b_reg_limb_probe; do
  rm -f "$WORK/runs/$d"/fold*.pt
done
space; guard

if [ ! -s "$FRAMES/labels.csv" ]; then
  stamp "video frames ($PER_VIDEO per video)"
  if mkdir "$FRAMES.lock" 2>/dev/null; then   # only one pod extracts
    python -c "import imageio_ffmpeg" 2>/dev/null || command -v ffmpeg >/dev/null || pip install -q imageio-ffmpeg
    python scripts/extract_video_frames.py --csv "$LABELS" --root "$VIDEO_ROOT" --out "$FRAMES" --per-video "$PER_VIDEO" \
      || stamp "frame extraction failed"
    rmdir "$FRAMES.lock"
  else
    while [ -d "$FRAMES.lock" ]; do sleep 20; done
  fi
fi
EXTRA=(); [ -s "$FRAMES/labels.csv" ] && EXTRA=(--extra-images "$FRAMES/images" --extra-labels "$FRAMES/labels.csv")
[ ${#EXTRA[@]} -gt 0 ] || stamp "WARNING: no frames; round-2 runs train on photos only"
space; guard

stamp "round-2 candidates: $ROUND2"
python scripts/bakeoff.py run --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" \
  --candidates "$ROUND2" --epochs 15 $GROUPS "${EXTRA[@]}" -- --data-source darpa \
  || stamp "a round-2 candidate failed: leaderboard uses what finished"
space; guard
[ "$ONLY_TRAIN" = 1 ] && { stamp "ONLY_TRAIN: done"; exit 0; }

for c in $(echo "$WAIT_FOR" | tr ',' ' '); do
  stamp "waiting for $c (other pod)"
  while [ ! -s "$WORK/runs/$c/oof_probs.json" ] && [ ! -s "$WORK/runs/$c/run_info.json" ]; do sleep 120; done
done

stamp "leaderboard (greedy + stacker)"
python scripts/bakeoff.py leaderboard --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" $GROUPS \
  | tee "$WORK/leaderboard_round2.txt"

# Frames go into the final retrain only if a frames-trained candidate is in the chosen ensemble.
USE_FRAMES=$(python -c "import json;w=json.load(open('$WORK/runs/ensemble_choice.json'))['weights'];print(int(any(k.endswith(('_frames','_crops')) for k in w)))")
FINAL_EXTRA=(); [ "$USE_FRAMES" = 1 ] && FINAL_EXTRA=("${EXTRA[@]}")
stamp "final retrain on all data, seeds 0,1,2 (frames: $USE_FRAMES)"
python scripts/bakeoff.py final --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" \
  --final "$WORK/runs_final" --out "$WORK/weights_final" --epochs 15 "${FINAL_EXTRA[@]}"
space; guard

stamp "timing check of the final weights (200 photos, offline)"
mkdir -p "$WORK/timing_in"
[ "$(ls "$WORK/timing_in" | wc -l)" -ge 200 ] || ls "$IMAGES" | head -200 | xargs -I{} cp "$IMAGES/{}" "$WORK/timing_in/"
T0=$(date +%s)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python inference.py --input "$WORK/timing_in" \
  --output "$WORK/timing_final.json" --weights "$WORK/weights_final" --time-budget 3000 > "$WORK/timing_final.log" 2>&1
T1=$(date +%s)
python validate_predictions.py "$WORK/timing_final.json" --images "$WORK/timing_in" | tail -1
python - "$((T1 - T0))" <<'PY'
import sys; dt=int(sys.argv[1]); per=dt/200; proj=per*190
print(f"final timing: 200 photos in {dt}s ({per:.2f} s/photo incl. load) -> {proj/60:.1f} min for ~190 vs 50 min: "
      + ("OK" if proj < 0.8*3000 else "TIGHT" if proj < 3000 else "TOO SLOW"))
PY
nvidia-smi --query-gpu=memory.used --format=csv,noheader
stamp "done"; space

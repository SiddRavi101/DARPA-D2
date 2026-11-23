#!/usr/bin/env bash
# Gemma push (Nov 23): after the running Gemma E4B bake-off, train vlm_gemma4_e4b_hr
# (768 px, wound/amputation oversampled x2), then the leaderboard (greedy + stacker over all candidates).
#   nohup bash scripts/gemma_hr.sh > /workspace/gemma_hr.log 2>&1 &
# No final retrain here: the 5 fold adapters of the winner already form a usable ensemble.
set -uo pipefail
cd "$(dirname "$0")/.."

IMAGES=/root/img2048
LABELS=/workspace/prequal/train/train_dataset.csv
WORK=/workspace/day_one
GRP="--groups-file $WORK/groups.csv"
stamp() { echo "=== $(date '+%F %T') $*"; }

stamp "waiting for the running bake-off"
while pgrep -f "bakeoff.py (run|final)|day_one.py run" >/dev/null; do sleep 60; done

stamp "train vlm_gemma4_e4b_hr"
python scripts/bakeoff.py run --images "$IMAGES" --labels "$LABELS" --runs "$WORK/runs" \
  --candidates vlm_gemma4_e4b_hr --folds 5 --epochs 15 $GRP -- --data-source darpa \
  || stamp "vlm_gemma4_e4b_hr failed (see $WORK/runs/vlm_gemma4_e4b_hr/train.log)"

stamp "leaderboard (greedy + stacker), Qwen excluded by team policy"
NQ="$WORK/runs_noqwen"; mkdir -p "$NQ"
for d in "$WORK"/runs/*/; do
  n=$(basename "$d"); case "$n" in *qwen*) continue ;; esac
  [ -e "$NQ/$n" ] || ln -s "$d" "$NQ/$n"
done
for f in keypoints.json keypoints_extra.json; do [ -e "$WORK/runs/$f" ] && ln -sf "$WORK/runs/$f" "$NQ/$f"; done
python scripts/bakeoff.py leaderboard --images "$IMAGES" --labels "$LABELS" --runs "$NQ" $GRP \
  | tee "$WORK/leaderboard_gemma.txt"
stamp "done"

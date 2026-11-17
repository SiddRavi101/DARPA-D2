#!/usr/bin/env bash
# Synthetic proof-of-concept on a GPU box (e.g. a Runpod PyTorch pod):
#   generate train/test sets → 5-fold train → ensemble inference on held-out test
#   → validate → score (Qualification Guide §4.2.2 metric).
# Results are printed between ===RESULTS_BEGIN=== / ===RESULTS_END=== so they can
# be read from container logs.
#
#   bash qualification/scripts/run_synthetic_poc.sh [repo_url] [branch]
set -uo pipefail
REPO=${1:-https://github.com/SiddRavi101/DARPA-D2}
BRANCH=${2:-main}
WORK=${WORK:-/workspace}
EPOCHS=${EPOCHS:-12}
FOLDS=${FOLDS:-5}
BACKBONE=${BACKBONE:-convnext_small.fb_in22k_ft_in1k_384}
SIZE=${SIZE:-384}

cd "$WORK"
if [ -d DARPA-D2 ]; then git -C DARPA-D2 pull -q --ff-only; else git clone --depth 1 -b "$BRANCH" "$REPO" DARPA-D2; fi
cd DARPA-D2/qualification
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# huggingface_hub + safetensors are needed to download pretrained backbone weights
pip install -q -r requirements-train.txt 2>&1 | tail -1
python -c "import timm, huggingface_hub, timm.models._hub as th; assert th._has_hf_hub, \"timm cannot use huggingface_hub\"; print(\"timm\", timm.__version__, \"hf_hub\", huggingface_hub.__version__)" \
  || { echo "DEPENDENCY_INSTALL_FAILED"; exit 1; }

echo "== generating synthetic data $(date -u +%T)"
[ -f data/synth/train/labels.json ] || python synthetic/generate.py --out data/synth/train --n 3000 --seed 1 &
[ -f data/synth/test/labels.json ] || python synthetic/generate.py --out data/synth/test --n 600 --seed 424242 --prefix test_ &
wait
echo "== training $(date -u +%T)"
rm -f model/weights/*.pt
python -m d2qual.train --images data/synth/train --labels data/synth/train/labels.json \
  --out model/weights --backbone "$BACKBONE" --size "$SIZE" --epochs "$EPOCHS" \
  --batch 32 --lr 1e-3 --backbone-lr-mult 0.1 --folds "$FOLDS" --workers 8 \
  || { echo "TRAINING_FAILED - not scoring the prior fallback"; exit 1; }
ls model/weights/*.pt >/dev/null 2>&1 || { echo "NO_CHECKPOINTS - not scoring"; exit 1; }

echo "== held-out inference $(date -u +%T)"
mkdir -p /tmp/test_in && cp data/synth/test/*.jpg data/synth/test/*.png /tmp/test_in/ 2>/dev/null
T0=$(date +%s)
python inference.py --input /tmp/test_in --output /tmp/out --weights model/weights
T1=$(date +%s)
python validate_predictions.py /tmp/out/predictions.json --images /tmp/test_in

echo "===RESULTS_BEGIN==="
echo "inference_seconds_for_600_images=$((T1 - T0))"
python - <<'PY'
import json; r = json.load(open("model/weights/train_report.json"))
print("cv_fold_accuracy:", {k: round(v["overall_accuracy"], 4) for k, v in r.items()})
PY
python score.py /tmp/out/predictions.json data/synth/test/labels.json --json /tmp/out/score.json
echo "===RESULTS_END==="
echo "== done $(date -u +%T)"

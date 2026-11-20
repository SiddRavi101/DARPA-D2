#!/usr/bin/env bash
# COCO real-photo FILTER bake-off (before DARPA data). Runs in the submission base image
# pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime. Results -> $WORK/out (serve on :8000).
#   bash scripts/run_coco_bakeoff.sh [candidates] [folds] [epochs]
set -uo pipefail
CANDS=${1:-convnext_s,dinov2_b,eva02_b,siglip_b,convnext_s_limb,vlm_qwen2vl_2b}
FOLDS=${2:-3}
EPOCHS=${3:-8}
WORK=${WORK:-/root/work}
mkdir -p "$WORK/out"
cd "$WORK"
export HF_HOME=$WORK/hf TORCH_HOME=$WORK/torch
(apt-get update -qq && apt-get install -y -qq git curl libgl1 libglib2.0-0) >/dev/null 2>&1
if [ -d DARPA-D2 ]; then git -C DARPA-D2 pull -q --ff-only; else git clone -q --depth 1 https://github.com/SiddRavi101/DARPA-D2; fi
cd DARPA-D2/qualification
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q -r requirements-train.txt -r requirements-vlm.txt 2>&1 | grep -v "Running pip as" | tail -2
python -c "import torch, timm, transformers, peft, huggingface_hub, timm.models._hub as th; assert th._has_hf_hub; print('torch', torch.__version__, 'timm', timm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'hub', huggingface_hub.__version__)" \
  || { echo DEPENDENCY_FAILED; exit 1; }

echo "=== COCO PROXY $(date -u +%T)"
python scripts/build_coco_proxy.py --coco-dir "$WORK/coco" --out data/coco_proxy | tee "$WORK/out/proxy_stats.txt"

echo "=== BAKE-OFF $(date -u +%T)"
python scripts/bakeoff.py run --images data/coco_proxy --labels data/coco_proxy/labels.json --runs "$WORK/runs" \
  --candidates "$CANDS" --folds "$FOLDS" --epochs "$EPOCHS" --batch 16 --workers 8
python scripts/bakeoff.py leaderboard --images data/coco_proxy --labels data/coco_proxy/labels.json --runs "$WORK/runs"
cp "$WORK/runs/leaderboard.md" "$WORK/runs/ensemble_choice.json" "$WORK/out/" 2>/dev/null
for d in "$WORK"/runs/*/; do n=$(basename "$d"); grep -E "val_acc|CV overall|Error|error|Traceback" "$d/train.log" | tail -40 > "$WORK/out/log_$n.txt"; done

echo "=== ASSEMBLE + INFERENCE WITH SUBMISSION PINS $(date -u +%T)"
python scripts/bakeoff.py assemble --runs "$WORK/runs" --out model/weights_bake
if ls model/weights_bake/*/vlm_meta.json >/dev/null 2>&1; then
  python -c "import glob,json;from pathlib import Path;from huggingface_hub import snapshot_download as s;from d2qual.vlm_ft import base_dir_name as b;[s(x, local_dir=str(Path('model/vlm_base')/b(x))) for x in sorted({json.load(open(m))['base'] for m in glob.glob('model/weights_bake/*/vlm_meta.json')})]"
fi
[ -f model/pose/keypointrcnn.pth ] || python -m d2qual.pose --save-weights model/pose/keypointrcnn.pth
pip install -q -r requirements.txt 2>&1 | grep -v "Running pip as" | tail -1
ls model/weights_bake/*/vlm_meta.json >/dev/null 2>&1 && pip install -q -r requirements-vlm.txt 2>&1 | grep -v "Running pip as" | tail -1
mkdir -p /tmp/inf_in && ls data/coco_proxy/*.jpg | head -300 | xargs -I{} cp {} /tmp/inf_in/
T0=$(date +%s)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python inference.py --input /tmp/inf_in --output /tmp/inf_out --weights model/weights_bake 2>&1 \
  | grep -E "loaded|vlm members|pose model|backend|wrote|ERROR|Error" | head -30 | tee "$WORK/out/inference.txt"
T1=$(date +%s)
python validate_predictions.py /tmp/inf_out/predictions.json --images /tmp/inf_in | tee -a "$WORK/out/inference.txt"
echo "inference_seconds_for_300_images=$((T1 - T0)) (offline, ensemble incl. any VLM members)" | tee -a "$WORK/out/inference.txt"
nvidia-smi --query-gpu=memory.used --format=csv | tee -a "$WORK/out/inference.txt"
echo "=== COCO_BAKEOFF_DONE $(date -u +%T)"

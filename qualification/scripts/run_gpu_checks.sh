#!/usr/bin/env bash
# GPU checks in the SUBMISSION base image (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime):
#   1. backbone fit check (memory / speed vs the 16 GB, 1 h limits)
#   2. pose model on real photos of people (COCO val, CC-licensed) -> keypoint sanity
#   3. synthetic train: baseline vs limb crops (2 folds), out-of-fold review
#   4. inference with the pinned submission requirements: GPU path, timing, validator
# Results go to $WORK/out (served on :8000 if the pod exposes it) and $WORK/gpu_checks.log.
set -uo pipefail
WORK=${WORK:-/root/work}
mkdir -p "$WORK/out"
cd "$WORK"
export HF_HOME=$WORK/hf TORCH_HOME=$WORK/torch
(apt-get update -qq && apt-get install -y -qq git curl libgl1 libglib2.0-0) >/dev/null 2>&1
if [ -d DARPA-D2 ]; then git -C DARPA-D2 pull -q --ff-only; else git clone -q --depth 1 https://github.com/SiddRavi101/DARPA-D2; fi
cd DARPA-D2/qualification
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
pip install -q -r requirements.txt 2>&1 | tail -1          # the exact submission pins
pip install -q numpy 2>&1 | tail -1
python -c "import torch, torchvision, timm, huggingface_hub; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__, 'timm', timm.__version__, 'hub', huggingface_hub.__version__)"

echo "=== 1. FIT CHECK $(date -u +%T)"
python scripts/fit_check.py --test-images 2000 --folds 5 --json "$WORK/out/fit_check.json"

echo "=== 2. POSE ON REAL PHOTOS $(date -u +%T)"
python -m d2qual.pose --save-weights model/pose/keypointrcnn.pth
mkdir -p "$WORK/coco"
for id in 000000000785 000000001000 000000002153 000000005529 000000007386 000000008690 000000013291 000000017899; do
  curl -sf -o "$WORK/coco/$id.jpg" "http://images.cocodataset.org/val2017/$id.jpg" || true
done
python - "$WORK/coco" "$WORK/out" <<'PY'
import sys, json, time, torch
from pathlib import Path
from PIL import Image, ImageDraw
from d2qual.pose import PoseExtractor, LIMB_CHAINS, KP
from d2qual.data import load_image, limb_boxes
src, out = Path(sys.argv[1]), Path(sys.argv[2])
ims = [(p.name, load_image(p)) for p in sorted(src.glob("*.jpg"))]
ext = PoseExtractor(torch.device("cuda"), "model/pose/keypointrcnn.pth")
# lying-down test: also feed each photo rotated 90 degrees (the rotation search should recover it)
batch = [im for _, im in ims] + [im.rotate(90, expand=True) for _, im in ims]
t0 = time.time(); kps = ext.extract(batch); dt = time.time() - t0
print(f"pose: {len(batch)} images in {dt:.1f}s ({dt/len(batch)*1000:.0f} ms/img incl. 3 rotations)")
colors = [(255, 60, 60), (60, 160, 255), (255, 200, 0), (0, 230, 120)]  # L arm, R arm, L leg, R leg
for k, ((name, _), im, kp) in enumerate(zip(ims * 2, batch, kps)):
    d = ImageDraw.Draw(im)
    boxes, feats = limb_boxes(kp, im.size)
    for s, b in enumerate(boxes):
        if b: d.rectangle(b, outline=colors[s], width=4)
    for i in range(17):
        if kp[i, 2] > 0.35: d.ellipse([kp[i,0]-5, kp[i,1]-5, kp[i,0]+5, kp[i,1]+5], fill=(255,255,255))
    im.save(out / f"pose_{'rot90_' if k >= len(ims) else ''}{name}", quality=85)
    print(name, "rot90" if k >= len(ims) else "upright", "limbs found:", feats[:, 3].astype(int).tolist(),
          "mean conf", round(float(kp[:, 2].mean()), 2))
PY

echo "=== 3. SYNTHETIC TRAIN: BASELINE vs LIMB CROPS $(date -u +%T)"
pip install -q -r requirements-train.txt 2>&1 | tail -1
python synthetic/generate.py --out data/synth/train --n 1200 --seed 1 >/dev/null &
python synthetic/generate.py --out data/synth/test --n 300 --seed 424242 --prefix test_ >/dev/null &
wait
for mode in base limb; do
  extra=""; [ "$mode" = limb ] && extra="--limb-crops"
  python -m d2qual.train --images data/synth/train --labels data/synth/train/labels.json \
    --out "$WORK/w_$mode" --size 384 --epochs 6 --batch 16 --folds 2 --workers 8 $extra 2>&1 \
    | grep -E "labelled|limb crops|fold|CV overall|out-of-fold|Error|error" | grep -v "^epoch"
done
python -m d2qual.review --images data/synth/train --labels data/synth/train/labels.json \
  --pred "$WORK/w_limb/oof_predictions.json" --probs "$WORK/w_limb/oof_probs.json" \
  --keypoints "$WORK/w_limb/keypoints.json" --out "$WORK/out/review_limb" | tail -3

echo "=== 4. INFERENCE WITH SUBMISSION PINS, LIMB MODEL ON GPU $(date -u +%T)"
pip install -q -r requirements.txt 2>&1 | tail -1
mkdir -p /tmp/test_in && cp data/synth/test/*.jpg /tmp/test_in/ 2>/dev/null; cp data/synth/test/*.png /tmp/test_in/ 2>/dev/null
cp -r "$WORK/w_limb" model/weights_limb && mkdir -p model/weights_limb/pose && cp model/pose/keypointrcnn.pth model/weights_limb/pose/
T0=$(date +%s)
python inference.py --input /tmp/test_in --output /tmp/out --weights model/weights_limb 2>&1 | grep -E "loaded|backend|pose|wrote|error" | head
T1=$(date +%s)
python validate_predictions.py /tmp/out/predictions.json --images /tmp/test_in
echo "inference_seconds_for_$(ls /tmp/test_in | wc -l)_images=$((T1 - T0))"
python score.py /tmp/out/predictions.json data/synth/test/labels.json | head -3
echo "=== GPU_CHECKS_DONE $(date -u +%T)"

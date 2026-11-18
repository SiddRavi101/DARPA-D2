#!/usr/bin/env bash
# Build → local smoke test → ClamAV scan → push to GitLab registry → print digest.
# Mirrors ICD v1.1 §3.5 and §4.1. Usage:
#   TEAM=Pinpoint PROJECT=gitlab-user/d2 TAG=v1 EMAIL=me@x.com ./scripts/submit.sh
set -euo pipefail
: "${TEAM:?}" "${PROJECT:?}" "${TAG:?}" "${EMAIL:?}"
cd "$(dirname "$0")/.."

TEAM_SLUG=$(echo "$TEAM" | tr "[:upper:] " "[:lower:]-")
IMAGE="${TEAM_SLUG}-d2-qualification-task:${TAG}"
REMOTE="registry.gitlab.com/${PROJECT}/${IMAGE}"

ls model/weights/*.pt >/dev/null 2>&1 || { echo "No trained checkpoints in model/weights — train first"; exit 1; }
# Limb-crop checkpoints need the pose model baked in (the container has no network).
if python -c "import glob,sys,torch; sys.exit(0 if any(torch.load(p,map_location='cpu',weights_only=False)['config'].get('limb_crops') for p in glob.glob('model/weights/*.pt')) else 1)"; then
  [ -f model/pose/keypointrcnn.pth ] || python -m d2qual.pose --save-weights model/pose/keypointrcnn.pth
fi
# VLM members need their base model baked in (Apache-2.0 Qwen2-VL-2B by default).
if ls model/weights/*/vlm_meta.json >/dev/null 2>&1 && [ ! -f model/vlm_base/config.json ]; then
  BASE=$(python -c "import glob,json;print(json.load(open(glob.glob('model/weights/*/vlm_meta.json')[0]))['base'])")
  python -c "import sys;from huggingface_hub import snapshot_download;snapshot_download(sys.argv[1],local_dir='model/vlm_base')" "$BASE"
fi

docker build -t "$IMAGE" --build-arg TEAM_NAME="$TEAM" --build-arg CONTACT_EMAIL="$EMAIL" \
  --build-arg SUBMISSION_VERSION="$TAG" .

# Smoke test exactly as JHU/APL will run it (network off).
if [ -z "${SMOKE_IN:-}" ]; then
  SMOKE_IN=$(mktemp -d)
  python -c "import sys;from PIL import Image;[Image.new('RGB',(640,480),(i*60,90,120)).save(f'{sys.argv[1]}/smoke_{i}.jpg') for i in range(3)]" "$SMOKE_IN"
fi
OUT=$(mktemp -d)
docker run --rm ${GPU_FLAG:---gpus all} --network none \
  -v "$(realpath "$SMOKE_IN")":/app/input:ro -v "$OUT":/app/output:rw "$IMAGE"
python validate_predictions.py "$OUT/predictions.json" --images "$SMOKE_IN"

# ClamAV (ICD §3.5)
SCAN=$(mktemp -d)
docker save "$IMAGE" | tar -xC "$SCAN"
docker run --rm -v "$SCAN":/scandir clamav/clamav:latest clamscan -r -i /scandir
rm -rf "$SCAN"

docker tag "$IMAGE" "$REMOTE"
docker push "$REMOTE"
echo "Digest:"; docker inspect --format='{{index .RepoDigests 0}}' "$REMOTE"
echo "Email Team Name, GitLab user, Project URL, image, tag, digest, contact to APL-D2C-Help@jhuapl.edu"

#!/usr/bin/env bash
# Feasibility test for photorealistic, label-correct synthetic casualties
# (synthetic/flux_inpaint.py) on a >=40 GB GPU pod. Logs to $WORK/flux.log.
#
#   bash qualification/scripts/run_flux_feasibility.sh [n_images]
set -uo pipefail
N=${1:-20}
WORK=${WORK:-/workspace}
export HF_HOME=$WORK/hf
cd "$WORK"
if [ -d DARPA-D2 ]; then git -C DARPA-D2 pull -q --ff-only; else git clone --depth 1 https://github.com/SiddRavi101/DARPA-D2; fi
cd DARPA-D2/qualification
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
(apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0) >/dev/null 2>&1
pip install -q "diffusers==0.32.2" "transformers==4.47.1" "accelerate==1.2.1" sentencepiece protobuf \
    "ultralytics==8.3.40" "huggingface_hub>=0.26,<1" pillow 2>&1 | tail -1
python -c "import diffusers, ultralytics; print('diffusers', diffusers.__version__, 'ultralytics', ultralytics.__version__)" \
  || { echo "DEPENDENCY_INSTALL_FAILED"; exit 1; }
echo "== generating $(date -u +%T)"
python synthetic/flux_inpaint.py --out "$WORK/flux_out" --n "$N" --seed 7 && echo "FLUX_DONE" || echo "FLUX_FAILED"

#!/usr/bin/env bash
# COCO re-test of the newer candidates (SigLIP 2, Qwen3-VL 2B/4B) against the current ones, after the
# library upgrade (timm 1.0.20, transformers 4.57.6). Same proxy, folds and epochs as the 2026-11-18 run,
# but every fold now keeps its FINAL epoch. Results -> $WORK/out (serve on :8000).
#   bash scripts/run_coco_retest.sh [candidates]
set -uo pipefail
CANDS=${1:-siglip_b,siglip2_b,eva02_b,convnext_s,vlm_qwen2vl_2b,vlm_qwen3vl_2b,vlm_qwen3vl_4b}
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
python -c "import torch, timm, transformers, peft, huggingface_hub as h; print('torch', torch.__version__, 'timm', timm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'hub', h.__version__); assert timm.list_models('vit_base_patch16_siglip_384', pretrained=True)" \
  || { echo DEPENDENCY_FAILED; exit 1; }
python -c "import timm; assert 'vit_base_patch16_siglip_384.v2_webli' in timm.list_models(pretrained=True), 'no SigLIP2 in timm'; print('siglip2 ok')" || { echo SIGLIP2_MISSING; exit 1; }

echo "=== VLM FAST-PATH CHECKS $(date -u +%T)"
for B in Qwen/Qwen3-VL-2B-Instruct Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen3-VL-4B-Instruct; do
  n=$(echo "$B" | tr '/' '_')
  python scripts/vlm_speed.py --base "$B" --n 48 --out "$WORK/out/speed_$n.json" > "$WORK/out/speed_$n.log" 2>&1
  echo "$B: $(grep -E 'equivalence|3_adapters_tta|train ' "$WORK/out/speed_$n.log" | tr '\n' ' ')"
  grep -q "Traceback" "$WORK/out/speed_$n.log" && tail -5 "$WORK/out/speed_$n.log"
done
# Drop a VLM whose fast path disagrees with the reference path (it would be scored on wrong numbers).
KEEP=""
for c in $(echo "$CANDS" | tr ',' ' '); do
  case $c in
    vlm_qwen3vl_2b) f=Qwen_Qwen3-VL-2B-Instruct ;; vlm_qwen3vl_4b) f=Qwen_Qwen3-VL-4B-Instruct ;;
    vlm_qwen2vl_2b) f=Qwen_Qwen2-VL-2B-Instruct ;; *) f="" ;;
  esac
  if [ -n "$f" ] && ! python -c "import json,sys;r=json.load(open(sys.argv[1]));sys.exit(0 if r['argmax_agreement']>=0.9 and r['max_abs_prob_diff_fast_vs_reference']<0.1 else 1)" "$WORK/out/speed_$f.json" 2>/dev/null; then
    echo "DROPPING $c: fast path failed or disagrees with reference"; continue
  fi
  KEEP="$KEEP${KEEP:+,}$c"
done
echo "candidates: $KEEP"

echo "=== COCO PROXY $(date -u +%T)"
python scripts/build_coco_proxy.py --coco-dir "$WORK/coco" --out data/coco_proxy | tail -3 | tee "$WORK/out/proxy_stats.txt"

echo "=== BAKE-OFF $(date -u +%T)"
python scripts/bakeoff.py run --images data/coco_proxy --labels data/coco_proxy/labels.json --runs "$WORK/runs" \
  --candidates "$KEEP" --folds 3 --epochs 8 --batch 16 --workers 8
python scripts/bakeoff.py leaderboard --images data/coco_proxy --labels data/coco_proxy/labels.json --runs "$WORK/runs"
cp "$WORK/runs/leaderboard.md" "$WORK/runs/ensemble_choice.json" "$WORK/out/" 2>/dev/null
for d in "$WORK"/runs/*/; do n=$(basename "$d"); cp "$d/oof_probs.json" "$WORK/out/oof_$n.json" 2>/dev/null; grep -E "val_acc|CV overall|Error|Traceback" "$d/train.log" | tail -20 > "$WORK/out/log_$n.txt"; done
echo "=== COCO_RETEST_DONE $(date -u +%T)"

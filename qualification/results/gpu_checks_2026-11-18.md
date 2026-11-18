# GPU checks: limb crops, backbones, submission image (2026-11-18)

Run on Runpod Secure Cloud, 1x RTX 4090 24 GB (the A40 was sold out), **inside the submission's base image** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with the pinned `requirements.txt`. That gives torch 2.5.1+cu124, torchvision 0.20.1, timm 1.0.12 and huggingface_hub 0.26.2. Script: `scripts/run_gpu_checks.sh` at commit `2e7f991`. About 25 min of pod time at $0.89/hr ≈ $0.40.

## 1. Backbones vs DARPA's limits (16 GB GPU, 1 h)
Projected container time is for 2,000 test images × a 5-model ensemble with mirror TTA. Limb-crop rows include the pose model (≈0.15 s per image).

| Backbone | Size | Limb crops | Train peak (batch 16) | Inference peak | Images/s per model | Projected time |
|---|---|---|---|---|---|---|
| ConvNeXt-S (current default) | 512 | no | 9.2 GB | 0.8 GB | 339 | 0.5 min |
| ConvNeXt-S | 512 | **yes** | 15.3 GB | 0.9 GB | 199 | 5.8 min |
| ConvNeXt-B | 512 | no | 12.5 GB | 1.2 GB | 250 | 0.7 min |
| EVA02-B | 448 | no | 11.0 GB | 0.8 GB | 180 | 0.9 min |
| DINOv2-B | 518 | no | 9.9 GB | 0.8 GB | 198 | 0.8 min |
| DINOv2-B | 518 | **yes** | 16.1 GB | 0.9 GB | 118 | 6.4 min |
| DINOv2-L | 518 | no | out of memory at batch 16 on 24 GB (training only) | — | — | — |
| SigLIP-B | 384 | no | 4.9 GB | 0.6 GB | 413 | 0.4 min |

**Every candidate fits the evaluation box by a wide margin.** Inference is at most 1.2 GB of 16 GB, and at most about 6 min of 60. DINOv2-L would need batch 4–8 (or a 48 GB GPU) for training. All pretrained tags exist in the pinned timm 1.0.12. DINOv3 would need a timm upgrade.

## 2. Pose model on real photos
torchvision Keypoint R-CNN with the 0/90/270° search, on 8 COCO photos of people plus the same 8 rotated 90° to simulate lying casualties:

* Upright: all 4 limbs found in 8/8 photos, on the correct anatomical sides (checked visually).
* Rotated 90°: all 4 limbs found in 7/8 photos (1 missed an arm); sides still correct.
* Speed: 164 ms per image including the 3 rotations.

## 3. Synthetic training: baseline vs limb crops (small run, NOT decisive)
1,200 cartoon-synthetic images, 2 folds × 6 epochs, 384 px:

| Model | Grouped-CV accuracy |
|---|---|
| Baseline (global image only) | **0.662** |
| Limb crops | 0.644 |

On the cartoon images the pose model found wrists/ankles in only 49–65% of images, and the error review showed it sometimes confusing front and back (face-up vs face-down) on the drawn figures, which swaps the limb boxes. So this run shows the limb-crop path **works end to end on GPU**. It does **not** show whether it helps on real manikin photos. Decide on DARPA's data by grouped CV:
```bash
python -m d2qual.train ... --out runs/base
python -m d2qual.train ... --out runs/limb --limb-crops
```
Keep limb crops only if they beat the baseline.

## 4. Submission path on GPU
In the submission base image with the exact pins: limb-crop checkpoints loaded, pose weights loaded from the baked file (no download), backend `ensemble on cuda`. **300 images in 54 s** including model loading. The validator passed 300/300.

Still not tested: the literal `docker run --gpus all --network none` of our built image on a GPU host. That needs the image in a registry (e.g. the GitLab one DARPA provides).

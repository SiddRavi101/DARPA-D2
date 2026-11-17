# Synthetic proof-of-concept: 2026-11-17

**What this shows:** the full Part 2 pipeline works end to end on a GPU. It covers data → 5-fold training → ensemble `inference.py` → ICD validator → Qualification Guide §4.2.2 scoring.

**What it does not show:** what the model will score on DARPA's real manikin photos. The images are procedurally generated (`synthetic/generate.py`), and the test set comes from the same generator with a different seed. It is held out, but it has the same appearance as the training data. Expect real-photo accuracy to be lower until the model is fine-tuned on DARPA's training set.

## Setup
| | |
|---|---|
| Hardware | Runpod Secure Cloud, 1× RTX 4090 24 GB, driver 595.112.07 (A40/L40/L40S had no stock) |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, timm 1.0.12, huggingface_hub 1.33.0 |
| Data | 3,000 train images (seed 1), 600 held-out test images (seed 424242) |
| Model | `convnext_small.fb_in22k_ft_in1k_384` at 384 px, joint 4-site × 4-class head |
| Training | 5-fold stratified CV, 12 epochs, batch 32, AdamW lr 1e-3 (backbone ×0.1), mirror-with-side-swap augmentation |
| Inference | 5-fold ensemble + mirror test-time augmentation, fp16 |
| Script | `scripts/run_synthetic_poc.sh` at commit `d00aaaf` |
| Wall time | ~6 min data, 14 min training, 39 s inference for 600 images |
| Cost | ~38 min pod time at $0.89/hr ≈ $0.56 (including two failed dependency attempts) |

## Results on the 600 held-out images
**Overall accuracy (ICD metric): 0.9650**. Macro-F1: 0.9638. The validator passed on 600/600 predictions.

5-fold cross-validation accuracy: 0.9688 ± 0.0084 (folds: 0.9629, 0.9587, 0.9712, 0.9833, 0.9675).

| Site | Accuracy |
|---|---|
| upper_extremity:left | 0.9667 |
| upper_extremity:right | 0.9767 |
| lower_extremity:left | 0.9533 |
| lower_extremity:right | 0.9633 |

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| no_injury | 0.967 | 0.972 | 0.970 | 1101 |
| wound | 0.958 | 0.956 | 0.957 | 642 |
| amputation | 0.971 | 0.950 | 0.960 | 278 |
| not_testable | 0.966 | 0.971 | 0.968 | 379 |

Confusion matrix (rows = truth, columns = prediction):

| | no_injury | wound | amputation | not_testable |
|---|---|---|---|---|
| **no_injury** | 1070 | 18 | 3 | 10 |
| **wound** | 21 | 614 | 5 | 2 |
| **amputation** | 6 | 7 | 264 | 1 |
| **not_testable** | 9 | 2 | 0 | 368 |

For comparison, the prior-only fallback (no trained model, every limb `no_injury`) scored **0.4587** on the same test set.

## Reading the errors
* The largest error bucket is **no_injury ↔ wound (39 of 84 errors)**. Small or low-contrast synthetic wounds are hard to see at 384 px. Real photos will make this harder, which argues for 448–512 px on the real data.
* Amputations are rarely confused with no_injury (6). The ragged stump and blood pool are strong cues.
* Per-site accuracy is balanced between left and right (0.953–0.977). This means the anatomical-laterality handling (supine vs prone, ±180° rotation) is working rather than the model guessing sides.

## Next steps once DARPA's training set arrives
1. Fine-tune the same pipeline on the real images, optionally starting from these synthetic weights.
2. Compare 384 vs 512 px and ConvNeXt-S vs EVA02-B by cross-validation accuracy.
3. Run `scripts/submit.sh`: build, smoke test with `--network none`, ClamAV scan, push to GitLab.

## Container verification (same day, local Docker 27.1.1)
| Check | Result |
|---|---|
| `docker build` (pinned `requirements.txt`, no transformers) | OK. `pinpoint-d2-qualification-task:dev`, **6.17 GB** (limit 20 GB) |
| Run like the evaluator: `--network none`, input `:ro`, output `:rw` | OK. 26/26 predictions; the validator passed inside and outside the container |
| Corrupt input file | Labelled `not_testable` × 4. The run did not crash |
| Submission metadata | `team_name: Pinpoint` |
| GPU path inside the container | **Not tested locally**: the laptop has no NVIDIA GPU, so the run used the CPU fallback. The GPU code path was verified on the RTX 4090 pod outside Docker |
| ClamAV (ICD §3.5 command, `clamav/clamav:latest`, engine 1.5.4) | **Infected files: 0**. 5.80 GiB read, 330.86 MiB scanned under the default size limits |

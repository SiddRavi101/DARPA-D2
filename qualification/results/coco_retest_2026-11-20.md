# COCO re-test with newer candidates — 2026-11-20

**Why:** check the newer models found in the Nov 19 research (SigLIP 2; Qwen3-VL 2B and 4B, Apache-2.0) against the current candidates, after upgrading to timm 1.0.20 and transformers 4.57.6.

**Setup:**
- Runpod Secure Cloud A40 (pod `x4waywhtlbhwqa`, deleted afterwards), commit 8888a46, `scripts/run_coco_retest.sh`.
- The same 819-image COCO proxy, 3 grouped folds and 8 epochs (VLMs: 3) as the [Nov 18 run](coco_bakeoff_2026-11-18.md).
- **Every fold now keeps its final epoch** (`--select last`), so these scores are honest and slightly lower than Nov 18's best-epoch figures.

## VLM fast-path check (before training)

The fast inference path and the original slow path gave **identical answers** (argmax agreement 1.00) for all three VLMs. The largest probability gaps were 0.068, 0.027 and 0.059 (bf16 rounding).

| VLM | s / image (3 adapters + mirror TTA) | min per 2,000 images | peak GPU | train step (batch 4) |
|---|---|---|---|---|
| Qwen2-VL-2B | 0.84 | 27.8 | 7.8 GB | 1.12 s |
| Qwen3-VL-2B | 0.71 | 23.8 | 6.7 GB | 1.12 s |
| Qwen3-VL-4B | 1.37 | 45.7 | 12.6 GB | 1.94 s |

## Leaderboard (out-of-fold, DARPA metric)

| Candidate | OOF accuracy | recall no_injury | recall not_testable | train time |
|---|---|---|---|---|
| vlm_qwen3vl_4b | **0.8941** | 0.949 | 0.792 | 69 min |
| vlm_qwen3vl_2b | **0.8907** | 0.940 | 0.798 | 37 min |
| vlm_qwen2vl_2b | **0.8868** | 0.946 | 0.777 | 40 min |
| eva02_b | **0.8697** | 0.935 | 0.748 | 6 min |
| siglip_b | **0.8687** | 0.936 | 0.743 | 2 min |
| convnext_s | **0.8596** | 0.925 | 0.737 | 4 min |
| siglip2_b | **0.8013** | 0.889 | 0.638 | 2 min |

**Best ensemble (greedy on OOF): 0.8980**, half Qwen3-VL-4B and half Qwen3-VL-2B.

## Findings

- **Qwen3-VL is the best family.**
  - The 4B scored 0.894 and the 2B 0.891, against 0.887 for Qwen2-VL-2B.
  - The gaps are within noise for 819 images, but Qwen3-VL-2B is also faster and lighter than Qwen2-VL-2B, so it replaces it as the main VLM.
- **VLMs beat the image classifiers by about 2 points.** The gain is mostly on `not_testable` (0.78–0.80 recall vs 0.74–0.75).
- **The VLM speed fix holds:**
  - Qwen2-VL-2B trained in 40 min vs 133 min on Nov 18, with the same accuracy (0.887 both times).
- **SigLIP 2 underperformed** (0.801):
  - Its training loss stayed around twice that of SigLIP-B.
  - It probably needs its own learning rate rather than the shared `--backbone-lr-mult 0.05`.
  - Dropped unless re-tuned.
- **The best ensemble doesn't fit the time limit as-is:**
  - 4B + 2B with 3 adapters each and mirror TTA ≈ 2.1 s/image, about 69 min per 2,000 images.
  - Options to fit 50 min: turn TTA off (`D2_VLM_TTA=0`, about 35 min); use 1 adapter per model; or use the 2B alone (about 24 min).
  - The DARPA timing check decides.

## Candidates for the DARPA bake-off

| Status | Candidates |
|---|---|
| Main | vlm_qwen3vl_2b, vlm_qwen3vl_4b (watch time), eva02_b, siglip_b |
| Backup | vlm_qwen2vl_2b, convnext_s |
| Dropped | siglip2_b (unless re-tuned), dinov2_b (Nov 18) |

**Caveat:** COCO has no wound or amputation classes, and its photos are of real people, not manikins. This run filters candidates; it doesn't rank them on the real task.

## Cost

This pod billed $1.33 through Nov 20, plus about 12 min on Nov 21 not yet in the billing read, so about $1.45 in total.

Runpod pod billing read on 2026-11-21: **$4.67** across all pods so far. The pre-DARPA total is about **$4.80** of the $5 cap.

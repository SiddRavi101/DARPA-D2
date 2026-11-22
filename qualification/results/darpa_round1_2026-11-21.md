# DARPA round 1 bake-off — 2026-11-21

**Setup:**
- Runpod A40 pod `u19dwmm0jy0134` (CA-MTL-1), with ConvNeXt-B trained on a second A40 on the same volume.
- 952 labelled DARPA photos, 5 folds grouped by `casualty_id` (62 casualties), each fold keeping its final epoch.
- Photos were pre-resized to 2048 px on the pod's local disk.
- Majority-class floor: **0.570**.

## Leaderboard (out-of-fold, all 952 photos, DARPA metric)

| Candidate | OOF accuracy | recall no_injury | recall wound | recall amputation | recall not_testable | train time |
|---|---|---|---|---|---|---|
| **vlm_qwen3vl_4b** | **0.7138** | 0.867 | 0.465 | 0.326 | 0.713 | 157 min |
| vlm_qwen3vl_2b | 0.6499 | 0.838 | 0.417 | 0.258 | 0.514 | 86 min |
| siglip_b | 0.6439 | 0.856 | 0.232 | 0.140 | 0.676 | 20 min |
| eva02_b | 0.6397 | 0.851 | 0.226 | 0.129 | 0.682 | 26 min |
| convnext_b (Shravan) | 0.6226 | 0.846 | 0.174 | 0.095 | 0.666 | 24 min |
| dinov2_l_reg_limb_probe (Rohan B) | 0.6226 | 0.844 | 0.180 | 0.201 | 0.574 | 65 min |
| dinov2_b_reg_limb_probe (Rohan B) | 0.6108 | 0.827 | 0.201 | 0.127 | 0.606 | 35 min |
| dinov2_l_reg_probe (Rohan A) | 0.5523 | 0.788 | 0.135 | 0.046 | 0.503 | 23 min |

**Greedy ensemble: Qwen3-VL-4B alone, 0.7138.** No other member improved the out-of-fold score.

**Timing** (assembled 5-fold Qwen3-VL-4B, offline, A40):
- 200 images in 491 s, which is 2.45 s per image including model load.
- That projects to **7.8 min for about 190 test images**, against the 50-minute budget: OK.

## Findings

- **Qwen3-VL-4B wins clearly:** +6.4 points over the next model, and the best score on every fold.
- It roughly **doubles wound and amputation recall** compared with the image classifiers.
- It's also the best at `not_testable` (0.713).
- **Amputation is still the weakest class** (recall 0.33).
- **Fold spread is large:** one fold is a single amputee casualty with 86 photos, and every model scores 0.23–0.38 there.
- **Rohan's frozen DINOv2 probes underperformed.**
  - The whole-image probe scored below the floor.
  - The limb crops didn't fix amputations: the pose model "found" a wrist or ankle on 96–99% of limbs, including amputated ones, so the anchor-only stump signal rarely fired.
- **Data fixes made during the run:**
  - 20 HEIC files named `.png` (pillow-heif added, container asserts it).
  - Labels were matched to a `.png` twin instead of the labelled `.jpg` (exact-name matching now).
  - The volume filled up: the 14 GB zip copy was deleted (an identical copy stays in S3) and 4B was rerun.

## Next (round 2 candidates)

1. Run the error review on the Qwen3-VL-4B out-of-fold predictions: left/right swaps, amputation vs not_testable.
2. Retrain 4B with more data per model (10 folds, or all data) and/or add frames from the 235 labelled videos.
3. Try class-balanced sampling for amputation and wound.
4. Container: 5 × 4B adapters plus the 4B base. Check image size and memory on the dry-run build.

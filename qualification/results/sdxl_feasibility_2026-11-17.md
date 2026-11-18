# Photoreal synthetic casualties: feasibility test, 2026-11-17

**Question:** can a diffusion model produce photorealistic manikin images whose per-limb labels are correct, so they could be used as training data before DARPA's set arrives?

**Answer: not yet.** Sides and wounds are mostly right, but amputations, the hardest class, are unreliable and often have visible artifacts. Do not scale up. Put the effort into DARPA's real training set.

## What was tried
| Step | Result |
|---|---|
| Text-to-image only (FLUX.1-Krea-dev, public HF Space), 3 images | Looks photographic, but 2 of 2 amputations came out as wounds and 1 of 2 wounds was on the wrong side |
| FLUX.1-schnell on Runpod | Blocked: the repo is gated (needs an HF token and accepted terms) |
| RealVisXL V5.0 base + plain masked inpainting, 12 images | Base manikins are excellent. Inpainting redrew intact limbs: 0 working amputations, and wounds were barely visible |
| **Paint-then-refine** (`synthetic/flux_inpaint.py`): patch-copy limb removal, crude stump / wound / blanket painted in, then SDXL inpaint at 0.45–0.7 strength, 20 images | See below |

Run: Runpod Secure Cloud A40, about 1 h of pod time at $0.59/hr, including restarts.

## Paint-then-refine results (visual grading of 8 of 20 images, chosen to cover the hard cases)
| Class | Correct limb and side? | Looks real? |
|---|---|---|
| no_injury | Yes, almost always | Yes. The base manikins look like real photos |
| wound | Yes, almost always on the right limb and side | About half look like flat painted red blobs, not moulage |
| not_testable (blanket) | About 2 of 4 fully cover the limb; the others leave a boot or knee visible | Blankets look fairly real, with occasional halos |
| amputation | About 40–50% convincing (stump present, hand or foot gone) | Frequent artifacts: ghost limbs, floating blood blobs. Images with 3–4 amputations break down |

Overall about 64% of the inspected limb labels would be correct as written. The errors fall mostly on amputation and not_testable, which is exactly where training data matters most. The pose-model auto-check (amputation 3/15, not_testable 0/7 "consistent") is **not** a reliable filter, because pose models guess hidden joints under blankets and at stumps.

## If this is revisited
* Use an inpainting model with stronger edit control (e.g. FLUX.1-Fill-dev with an HF token, or ControlNet with a pose/depth map edited to remove the limb). Alternatively, generate manikins already missing a limb, then verify.
* Keep at most one amputation per image.
* Every image needs a human or VLM label check before training. Do not trust the pose auto-check.
* Generated files from this test: `data/sdxl_feasibility/` (gitignored).

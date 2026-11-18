# COCO real-photo filter bake-off — 2026-11-18

**Purpose:** a cheap pre-screen before DARPA data arrives. This is a filter, not a judge.
- COCO val2017 has real photos of people, but no wounds and no amputations.
- So the proxy only tests two things: limb visibility (no_injury vs not_testable) and left/right handling on real photographs.

**Setup**
- Pod: Runpod Secure Cloud A40 (pod `50tiwivg1957yp`, deleted afterwards), base image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, commit 62fcd4c.
- Script: `scripts/run_coco_bakeoff.sh` defaults.
- Data: 819 proxy images.
- Training: 3 grouped folds, seed 42, 8 epochs (VLM: 3 epochs).

## Leaderboard (out-of-fold, DARPA metric)

| Candidate | OOF accuracy | recall no_injury | recall not_testable | train time |
|---|---|---|---|---|
| vlm_qwen2vl_2b | **0.8871** | 0.947 | 0.776 | 133 min |
| eva02_b | **0.8770** | 0.949 | 0.743 | 8 min |
| siglip_b | **0.8700** | 0.934 | 0.750 | 3 min |
| convnext_s | **0.8651** | 0.925 | 0.754 | 5 min |
| convnext_s_limb | **0.8269** | 0.885 | 0.720 | 10 min |
| dinov2_b | **0.7701** | 0.874 | 0.576 | 6 min |

**Greedy ensemble (OOF): 0.8944**, with weights vlm 0.40, eva02_b 0.20, siglip_b 0.20, convnext_s_limb 0.20.

## Inference with the submission pins (offline, A40)

- The assembled 12-member ensemble (9 CNN/ViT folds + 3 VLM adapters + pose model) produced valid predictions.json for 300 images.
- It hit the time budget at **256/300 images after 50 min** (~11.7 s/image).
- The remaining 44 images fell back to the prior.

**The VLM, as built, is not submittable.** It must get roughly 10× faster before it can go in the container. Fixes to try first:
- 1 adapter instead of 3 folds
- drop mirror TTA, or batch the 4 site questions together
- a smaller pixel budget

Each fix has to be re-checked against the 1 h limit.

## Verdict (filter)

| Candidate | Status | Reason |
|---|---|---|
| eva02_b | **keep** | Best non-VLM; 8 min training. |
| siglip_b | **keep** | Ties eva02 within noise; fastest (3 min). |
| convnext_s | **keep** | Within noise of siglip; a different architecture adds ensemble diversity. |
| vlm_qwen2vl_2b | **keep, conditional** | Best single model (+0.010 over eva02, near noise) and the PRONTO-style approach. Inference must be made ~10× faster. |
| convnext_s_limb | **hold** | Below its own baseline on real photos. Still picked by the ensemble for diversity. Wounds and amputations are local detail the proxy can't test, so re-test it on DARPA data only. |
| dinov2_b | **drop** | Clearly last; worst not_testable recall (0.576). |

## Caveats

- 819 images: differences under ~0.01–0.02 are noise.
- The proxy has no wound or amputation classes, which are the hard part of the real task.
- COCO photos are not manikins.
- Final ranking happens on the DARPA training set with the same `bakeoff.py`.

## Cost

Runpod pod billing read on 2026-11-18:
- $2.67 total across the four feasibility/check pods, with today's figure not yet final.
- This run billed $1.60 so far; its final total will be about $2.2.
- Cumulative pre-DARPA spend: about $3.2 of the $5 cap.

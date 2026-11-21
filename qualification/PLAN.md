# Part 2 plan: from DARPA training data to a submitted image

As of 2026-11-20.

**How we qualify:** rank candidate models on honest, grouped out-of-fold accuracy, ship the most accurate combination that finishes inside the time limit, and submit by Nov 30 (deadline Dec 1; DARPA names up to 20 teams by Dec 10).

## Target and constraints

For each manikin photo we label four limb sites as `no_injury`, `wound`, `amputation` or `not_testable`.
- The sites are left/right × upper/lower extremity.
- Left and right are the **casualty's own** sides, not the viewer's.
- The score is overall class-at-site accuracy.

| ICD limit | Value | How we meet it |
|---|---|---|
| Runtime | 1 h for the test set, which is about 20% the size of the training set | `predict.py` stops at 50 min; any remaining images get the class prior |
| GPU | 1 × A40 / L40 / H100, < 16 GB | Peak measured 7.8 GB with the VLM ([vlm_speed](results/vlm_speed_2026-11-19.md)) |
| CPU / RAM / disk | 8 cores, < 32 GB RAM, < 20 GB disk and image | Image size measured at the dry-run build |
| Network | `--network none` | All weights baked in; offline flags set |
| Delivery | GitLab registry, `@D2-evaluator` as Reporter, ClamAV clean | [scripts/submit.sh](scripts/submit.sh) |

## Pipeline

1. **Inspect** (no GPU):

   ```
   python scripts/day_one.py inspect --images I --labels L
   ```

   It checks that labels match files and prints:
   - class counts per site
   - the majority-class floor
   - image sizes
   - candidate scene/manikin groupings guessed from filenames
2. **Choose the grouping** (a person decides). Photos of the same manikin or scene must stay in one fold. Otherwise near-duplicates leak across folds and inflate every score. `run` refuses to start without `--group-regex`, `--groups-file` or `--no-groups`.
3. **Bake-off.**

   ```
   python scripts/day_one.py run --images I --labels L --group-regex '...'
   ```

   - All candidates train on **identical** grouped folds: 5 folds × 15 epochs.
   - Each fold keeps its **final epoch** (`--select last`), so out-of-fold scores are honest.
   - Candidates can run on parallel pods, one `--candidates` subset each. Copy `runs/<name>/` together before the leaderboard.
4. **Leaderboard and ensemble** (`scripts/bakeoff.py leaderboard`).
   - Candidates are ranked by out-of-fold accuracy.
   - Greedy forward selection (Caruana) picks the weighted combination.
   - Differences under about 0.01–0.02 count as ties; prefer the faster model.
5. **Error review** (`python -m d2qual.review`).
   - Produces contact sheets of the most confident mistakes, per-class recall and a left/right swap count.
   - Fix systematic errors, then run round 2 on the top candidates only.
6. **Timing check** (inside `day_one.py run`). The assembled ensemble runs offline on 200 images, and the result is projected to 20% of the training set against the 50-minute budget (OK / TIGHT / TOO SLOW).
7. **Blind test** on the Mannequin In-Bed set: one run of the locked ensemble, scored once (see the next section).
8. **Submit** (a person runs it).
   - `scripts/submit.sh` builds the image, runs a CPU smoke test and the validator, scans with ClamAV, pushes, and records the digest.
   - Then send the submission details to DARPA.

## Blind test: Mannequin In-Bed

This is an outside check that the locked ensemble works on a **manikin** it has never seen: lying down, in different lighting, from a different camera. It exists to catch failures that DARPA-only cross-validation can't, such as left/right swaps on a lying body or a model that only works on DARPA's camera setup.

- **Data:** Northeastern ACLab "Mannequin In-Bed" set (about 419 images, RGB webcam + infrared, 2D joint labels). Page: <https://ostadabbas.sites.northeastern.edu/?p=5153>; zips:
  - <http://www.coe.neu.edu/Research/AClab/pose/IRS_MANNE_rawN.zip>
  - <http://www.coe.neu.edu/Research/AClab/pose/colorManneNumbered.zip>
- **Labels:** derived from the joint labels the same way as the COCO proxy. A limb is `no_injury` when at least 2 of its joints are visible in the frame, and `not_testable` otherwise. Tight crops and 0/90/180/270° rotations add hidden-limb and orientation cases. There are no wounds or amputations, so it tests **limb finding, side and visibility** only.
- **Blind rules:**
  - Never used for training, model selection, ensemble weights or threshold tuning.
  - Run **once**, on the locked ensemble, after the leaderboard is final (Nov 25).
  - Report overall accuracy, accuracy per site, and the **left/right swap rate**.
  - If it exposes a problem (e.g. a high swap rate), fix the cause on DARPA data, never by tuning to this set.
- **Pass bar:** no hard bar. Treat it as a warning light: a large gap from the DARPA cross-validation score, or a swap rate well above the COCO screen's, means the ensemble is leaning on DARPA-specific cues.
- **Before use:**
  - confirm the links still work and the license; no terms are posted for the mannequin data, so email the ACLab;
  - include it in the DARPA data question below;
- **Built** (2026-11-20) with `scripts/build_mannequin_test.py` from the IRS zip: **838 images** from 419 photos (one loose crop plus one tight or half-body crop each, random 0/90/180/270° rotation). Each site is 73–82% `no_injury` and 18–27% `not_testable`. Stored in `qualification/data/mannequin_test/` (git-ignored). The joint labels were checked visually: they use the manikin's own sides.
- **Run (once, Nov 25):**

  ```
  python inference.py --input data/mannequin_test --output /tmp/blind.json --weights model/weights
  python score.py /tmp/blind.json data/mannequin_test/labels.json
  python -m d2qual.review --images data/mannequin_test --labels data/mannequin_test/labels.json --pred /tmp/blind.json --out review_blind
  ```

## Candidates

The main four run in the bake-off; the DARPA data decides what ships. Every candidate uses:
- mirror-flip augmentation and test-time augmentation that **swap the left/right labels** with the image;
- letterboxing, never random crops, because a crop can cut a limb out of frame.

Scores are from the Nov 20 COCO re-test, where every fold keeps its final epoch.

| Candidate | `bakeoff.py` name | Type | COCO screen accuracy | Notes |
|---|---|---|---|---|
| Qwen3-VL-2B + LoRA | `vlm_qwen3vl_2b` | VLM (Apache-2.0) | 0.891 | **Main VLM.** One A/B/C/D question per limb, reading the letter scores; this follows PRONTO (UPenn), whose small fine-tuned VLM beat their other approaches. 37 min to train on COCO; 0.71 s per image with 3 adapters + mirror TTA; 6.7 GB peak. |
| Qwen3-VL-4B + LoRA | `vlm_qwen3vl_4b` | VLM (Apache-2.0) | 0.894 | Best single model, within noise of the 2B. 1.37 s per image and 12.6 GB peak, so it fits only with TTA off or 1 adapter. |
| EVA02-B | `eva02_b` | Image classifier | 0.870 | Best plain image model |
| SigLIP-B | `siglip_b` | Image classifier | 0.869 | Fastest (2 min on COCO) |

- **Backup:** Qwen2-VL-2B (`vlm_qwen2vl_2b`, 0.887) and ConvNeXt-S (`convnext_s`, 0.860).
- **Dropped:** SigLIP 2-B (0.801; would need its own learning rate) and DINOv2-B (0.770).
- **On hold:** pose-guided limb crops (`convnext_s_limb`, 0.827). Retest only if errors show that wounds need close-up detail, or that left/right is the main error.
- The best COCO ensemble was half Qwen3-VL-4B, half Qwen3-VL-2B: 0.898. It needs about 69 min per 2,000 images with TTA, so TTA off or fewer adapters is required.
- COCO has no wound or amputation classes, so these numbers only screen candidates; they don't predict the DARPA score. See [coco_retest](results/coco_retest_2026-11-20.md) and [coco_bakeoff](results/coco_bakeoff_2026-11-18.md).

**What gets picked:**
- **Accuracy** on the held-out (out-of-fold) predictions decides.
- **Training time** only breaks ties.
- **Container speed** is a hard limit: if the chosen ensemble is too slow, set `D2_VLM_TTA=0`, keep fewer VLM folds, or drop the slowest member.

## Compute and cost

Training runs on Runpod Secure Cloud A40s (48 GB, $0.59/hr). That's the card our speed figures come from, and one of DARPA's evaluation GPUs. Cost grows in proportion to the number of labelled photos:

| Labelled photos | 3 image models | + VLM | Total A40 time | Cost |
|---|---|---|---|---|
| 2,000 | ~2.3 h | ~2 h | ~4.5 h | ~$2.70 |
| 5,000 | ~5.8 h | ~5 h | ~11 h | ~$6.50 |
| 10,000 | ~11.5 h | ~10 h | ~22 h | ~$13 |

- Add 20–30% for reruns and the final container test.
- Running candidates on parallel pods cuts the wait to the slowest model, at the same total cost.
- Budget: up to $30 for Part 2, which covers two full rounds.
- If DARPA rules that a rented GPU is not "our own systems" (Rules §6.1), train on AWS **spot** GPUs on our own server instead, at about the same cost per run.

## Timeline

1. **Nov 21:** get server access, run `inspect`, pick the grouping, and email DARPA the cloud-GPU question.
2. **Nov 22:** bake-off round 1 (image models and VLM in parallel). GitLab dry-run push with a placeholder image.
3. **Nov 23–24:** error review, targeted fixes, round 2 on the top candidates.
4. **Nov 25:** lock the ensemble, run the timing check, then the one-time Mannequin In-Bed blind test.
5. **Nov 27–28:** final build, smoke test, ClamAV, push, digest, submit the details.
6. **Nov 29–30:** buffer. **Deadline Dec 1.**

## Risks

| Risk | Mitigation |
|---|---|
| Wrong or missing grouping inflates scores | `run` requires an explicit grouping; check it in the inspect report |
| Label format our reader doesn't parse | `inspect` fails on day one; add a reader in `d2qual/data.py` |
| Ensemble too slow | Timing check before submitting; turn off VLM TTA or drop members |
| Left/right confusion | Side-swapping mirror augmentation; review counts left/right swaps |
| GitLab 5-member cap | Remove the Duo Code Review bot; 3 of us + `@D2-evaluator` = 4 |

## Open questions for DARPA (D2Sprint@darpa.mil)

- [ ] May we process the dataset on a rented cloud GPU instance we control?
- [ ] May we use public datasets (COCO, the Mannequin In-Bed manikin set) to select and sanity-check models? Rules §6 bars "any other human subject data".
- [ ] Do submission details go by email (ICD v1.1) or through a Submission Portal (v1.0)?
- [ ] Is `not_testable` valid in the v1.1 schema appendix? The v1.0 appendix omits it.

## Data handling

- DARPA data never goes into git (`.gitignore` covers it).
- Never upload it to third-party services.
- No re-identification and no facial recognition (Rules §6.1).

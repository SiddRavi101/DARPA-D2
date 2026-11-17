# DARPA D2 Sprint — SecondLife team

Refactor of [SecondLife](https://github.com/rmagesh-quml/SecondLife_GoogleHackathon) (offline Gemma 4 E4B emergency assistant with TCCC/MARCH mode, FAISS + BM25 protocol RAG and a hash-chained audit log) into the DARPA **Documentation and Decision Support (D2) Sprint**.

| Directory | What it is for | Deadline |
|---|---|---|
| [`qualification/`](qualification/) | **Part 2 Qualification Task**: a Docker container that classifies limb injuries in manikin photos (ICD v1.1) | **1 Dec 2026** |
| [`sprint/`](sprint/) | Final-event pipeline: multi-sensor scenario → DD Form 1380 (Lane 1) + MARCH decision support (Lane 2) | 1 Mar 2027 |

> **Important:** the official Qualification Guide (v1.0) and ICD (v1.1) define Part 2 as an **image** task. The container reads `.jpg`/`.jpeg`/`.png` files from `/app/input` and writes `/app/output/predictions.json`. For each image it labels four limb sites (left/right × upper/lower extremity) as `no_injury`, `wound`, `amputation` or `not_testable`. Left and right are the **casualty's anatomical** sides. Scoring is overall class-at-site accuracy. The audio/video/accelerometry pipeline targets the final sprint, not qualification.

---

## 1. Part 2 qualification container

### How it is designed to score well
* **Joint 4-site classifier:** one pretrained timm backbone (default ConvNeXt-S, 512 px) with a 4×4-logit head. All four limbs share visual context, for example a casualty lying on one side hides a limb.
* **Mirror augmentation that swaps sides:** flipping an image left-right swaps the casualty's anatomical left and right. Most teams either skip flipping or corrupt their labels with it. Here the left/right labels are swapped with the image, which doubles the effective data. The same trick gives free test-time augmentation by averaging the original and mirrored predictions.
* **No random crops:** cropping a limb out of frame changes its true label to `not_testable`. Images are letterboxed instead.
* **5-fold stratified CV and an ensemble:** each fold is checkpointed, and inference averages all `model/weights/*.pt` files. The metric reported is exactly the ICD's overall accuracy.
* **Cannot emit an invalid file:**
  * A valid fallback `predictions.json` is written before inference starts.
  * Every image gets exactly four sites, and broken images become `not_testable`.
  * There is a 50-minute wall-clock budget (the limit is 1 h).
  * If CUDA fails, inference falls back to the CPU, and an out-of-memory error triggers a per-image retry.
  * The output is checked by a local copy of the JHU/APL validator.
* **Optional Gemma 4 vision fallback** (`D2_ENABLE_VLM=1`): this reuses SecondLife's 768×768 frame contract and JSON-only prompt parsing for zero-shot answers before any training.

### Train (on the DARPA training set, GPU machine)
```bash
cd qualification
pip install torch torchvision timm==1.0.12 pillow
python -m d2qual.train --images /data/d2/train/images --labels /data/d2/train/labels.json \
    --out model/weights --folds 5 --epochs 25 --batch 16
```
The label loader accepts several layouts: an ICD-style `predictions.json`, a bare prediction list, JSONL, a long CSV (`image_id,body_region,laterality,injury_type`), a wide CSV (`LUE,RUE,LLE,RLE` …), or a `{image: {site: class}}` dict. Things to try for higher CV accuracy:
* `--backbone eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --size 448`
* `--backbone convnext_base.fb_in22k_ft_in1k_384`

### Build, test, scan, push
```bash
cd qualification
TEAM=<team> PROJECT=<gitlab-user>/<project> TAG=v1 EMAIL=<contact> ./scripts/submit.sh
```
The script does the following, in order:
1. Builds the image.
2. Runs it exactly like JHU/APL (`--gpus all --network none`, read-only input).
3. Validates the output.
4. Runs ClamAV over the exported filesystem (ICD §3.5).
5. Pushes to the GitLab registry and prints the digest to email to `APL-D2C-Help@jhuapl.edu`.

Remember to invite `@D2-evaluator` as **Reporter** on the GitLab project.

Local run without Docker:
```bash
python qualification/inference.py --input some/images --output /tmp/out --weights qualification/model/weights
python qualification/validate_predictions.py /tmp/out/predictions.json --images some/images
```

---

## 2. Final-event pipeline (`sprint/`)

```
audio/video ─► faster-whisper (TCCC-biased prompt, VAD) ─► diarization ─► transcript ─► extract ─┐
accelerometry ─► IMU classifier (CPR @100-120/min, movement, stationary) ──────────────────────────┤
                                                                                                    ▼
                        time-ordered events ─► stateful DD1380 card (dedup + provenance)  [Lane 1]
                                                         │ on change
                                                         ▼
                        MARCH-PAWS rules engine + hybrid BM25/FAISS RAG citations           [Lane 2]
                        (optional Gemma 4 E4B "Military Medic" rewording; never changes content)
```

| SecondLife piece | Where it went |
|---|---|
| `rag/pipeline.py` (FAISS + bge-small) and Android `BM25Retriever` | `d2sprint/rag.py`: Okapi BM25 + FAISS fused with reciprocal-rank fusion; MARCH-tagged TCCC corpus in `data/tccc_protocols.json` |
| `function_calling/roles.py` MILITARY_MEDIC + panic-mode JSON | `d2sprint/llm.py` (optional advisor) |
| `inference/session.py` Gemma 4 E4B | `transformers` backend on the eval GPU (LiteRT-LM is for phones) |
| 16 kHz PCM / 768×768 contracts | `d2sprint/ingest.py` (`load_audio`, `sample_frames`) |
| `audit_log/logger.py` hash chain | `d2sprint/audit.py`: every card change and decision point is chained |
| *(new)* | `asr.py` (faster-whisper + diarization), `imu.py`, `extract.py`, `dd1380.py`, `decision.py` |

```bash
cd sprint
python examples/make_demo.py examples/demo_scenario      # synthetic scenario, no human-subjects data
python -m d2sprint both --input examples/demo_scenario --output out/demo
# containers (bake models first: python scripts/fetch_models.py)
docker build --build-arg LANE=documentation    -t d2-documentation:dev .
docker build --build-arg LANE=decision_support -t d2-decision-support:dev .
```
Each run writes `dd1380.json`, `decision_support.json`, `transcript.json`, `imu_events.json` and `audit_log.jsonl`.

**Open items:**
* DARPA has not yet published the final-event input layout or output schema. Ingestion discovers files by extension, and the card is mapped in a single function (`DD1380Card.to_output`), so adapting should be one small change each.
* `data/tccc_protocols.json` paraphrases CoTCCC guidance. Have a clinician check it against the current guideline edition before the final event.

---

## Tests
```bash
python -m venv .venv && .venv/Scripts/pip install timm==1.0.12 pytest torch pillow numpy scipy
.venv/Scripts/python -m pytest
```
The tests cover:
* ICD schema and validator, label formats, and the side-swapping flip.
* End-to-end train → predict → validate with a tiny model.
* Extraction, negation and spoken numbers.
* Card de-duplication, IMU CPR rate detection and MARCH ordering.
* A full demo scenario run.

## Data handling
DARPA data must never be committed (Qualification Guide §4.1: no redistribution, no re-identification, no facial recognition). `.gitignore` blocks the usual data and weight paths. Keep the training set outside the repo.

License: MIT (inherited from SecondLife).

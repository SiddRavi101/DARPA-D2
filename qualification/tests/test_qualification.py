import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from d2qual import schema  # noqa: E402
from d2qual.data import flip_labels, load_labels, match_images, normalize_class  # noqa: E402
from d2qual.vlm import parse_answer  # noqa: E402

LABELS = {
    "a.jpg": ["wound", "no_injury", "amputation", "not_testable"],
    "b.PNG": ["no_injury", "no_injury", "no_injury", "wound"],
    "c.jpeg": ["amputation", "wound", "no_injury", "no_injury"],
    "d.jpg": ["no_injury", "amputation", "wound", "no_injury"],
}


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    rng = random.Random(0)
    for name in LABELS:
        img = Image.new("RGB", (rng.randint(80, 200), rng.randint(80, 200)),
                        tuple(rng.randint(0, 255) for _ in range(3)))
        fmt = "PNG" if name.lower().endswith("png") else "JPEG"
        img.save(d / name, fmt)
    (d / "notes.txt").write_text("ignore me")
    (d / "broken.jpg").write_bytes(b"not an image")
    return d


def _doc(labels):
    return schema.build_document([schema.make_prediction(k, v) for k, v in labels.items()], "t", "1", "e@x")


def test_schema_example_from_icd_is_valid():
    assert schema.validate_document(_doc(LABELS), list(LABELS)) == []


def test_validator_catches_violations():
    doc = _doc(LABELS)
    doc["predictions"][0]["sites"][2]["injury_type"] = "bruise"
    doc["predictions"][1]["sites"].pop()
    doc["submission"].pop("email")
    errs = schema.validate_document(doc, list(LABELS) + ["missing.jpg"])
    joined = "\n".join(errs)
    assert "$.predictions[0].sites[2].injury_type" in joined
    assert "$.predictions[1].sites" in joined
    assert "$.submission.email" in joined
    assert "missing.jpg" in joined


def test_write_rejects_nan(tmp_path):
    with pytest.raises(ValueError):
        schema.write_document({"x": float("nan")}, tmp_path / "p.json")


def test_flip_swaps_anatomical_sides():
    lab = ["UE_L", "UE_R", "LE_L", "LE_R"]
    assert flip_labels(lab) == ["UE_R", "UE_L", "LE_R", "LE_L"]
    assert flip_labels(flip_labels(lab)) == lab


@pytest.mark.parametrize("raw,want", [("No Injury", "no_injury"), ("not-testable", "not_testable"),
                                      ("AMPUTATION", "amputation"), ("Wound", "wound")])
def test_normalize_class(raw, want):
    assert normalize_class(raw) == want


def test_label_formats_round_trip(tmp_path):
    # 1) predictions.json style
    p1 = tmp_path / "l.json"
    p1.write_text(json.dumps(_doc(LABELS)))
    assert load_labels(p1) == LABELS
    # 2) long CSV
    p2 = tmp_path / "long.csv"
    rows = ["image_id,body_region,laterality,injury_type"]
    for k, v in LABELS.items():
        for (r, l), c in zip(schema.SITES, v):
            rows.append(f"{k},{r},{l},{c}")
    p2.write_text("\n".join(rows))
    assert load_labels(p2) == LABELS
    # 3) wide CSV with abbreviations
    p3 = tmp_path / "wide.csv"
    p3.write_text("filename,LUE,RUE,LLE,RLE\n" + "\n".join(f"{k},{','.join(v)}" for k, v in LABELS.items()))
    assert load_labels(p3) == LABELS


def test_match_images_case_and_extension(image_dir):
    lab = {"A.JPG": LABELS["a.jpg"], "b": LABELS["b.PNG"], "zzz.jpg": LABELS["a.jpg"]}
    m = match_images(image_dir, lab)
    assert set(m) == {"a.jpg", "b.PNG"}


def test_vlm_parser():
    raw = 'Sure!\n```json\n{"left_arm":"wound","right_arm":"no injury","left_leg":"amputation","right_leg":"not testable"}\n```'
    assert parse_answer(raw) == ["wound", "no_injury", "amputation", "not_testable"]
    assert parse_answer("no json here") is None


def test_inference_without_weights_is_still_valid(image_dir, tmp_path):
    from d2qual.predict import run

    out = tmp_path / "out" / "predictions.json"
    doc = run(image_dir, out, tmp_path / "no_weights", "t", "1", "e@x")
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    names = [p.name for p in schema.list_images(image_dir)]
    assert schema.validate_document(on_disk, names) == []
    assert len(doc["predictions"]) == 5  # 4 good + broken.jpg, txt ignored
    broken = next(p for p in on_disk["predictions"] if p["image_id"] == "broken.jpg")
    assert {s["injury_type"] for s in broken["sites"]} == {"not_testable"}


def test_train_then_predict_end_to_end(image_dir, tmp_path):
    pytest.importorskip("timm")
    from d2qual import train
    from d2qual.predict import run

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_doc(LABELS)))
    wdir = tmp_path / "weights"
    train.main(["--images", str(image_dir), "--labels", str(labels_path), "--out", str(wdir),
                "--backbone", "resnet18", "--size", "64", "--epochs", "1", "--batch", "2",
                "--folds", "2", "--workers", "0", "--no-pretrained"])
    assert sorted(p.name for p in wdir.glob("*.pt")) == ["fold0.pt", "fold1.pt"]
    out = tmp_path / "predictions.json"
    run(image_dir, out, wdir, "t", "1", "e@x", batch_size=3)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_document(doc, [p.name for p in schema.list_images(image_dir)]) == []


def test_cli_and_validator_exit_codes(image_dir, tmp_path):
    import inference
    import validate_predictions

    out = tmp_path / "o"
    assert inference.main(["--input", str(image_dir), "--output", str(out), "--weights", str(tmp_path)]) == 0
    assert validate_predictions.main([str(out / "predictions.json"), "--images", str(image_dir)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert validate_predictions.main([str(bad)]) == 2
    bad.write_text('{"schema_version": NaN}')
    assert validate_predictions.main([str(bad)]) == 2

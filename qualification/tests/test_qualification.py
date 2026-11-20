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


def test_score_metric(tmp_path):
    from score import score

    gt = dict(LABELS)
    assert score(gt, gt)["overall_accuracy"] == 1.0
    pred = {k: list(v) for k, v in gt.items()}
    pred["a.jpg"][0] = "no_injury"          # 1 wrong site out of 16
    r = score(pred, gt)
    assert r["overall_accuracy"] == 15 / 16 and r["per_site_accuracy"]["upper_extremity:left"] == 0.75


def test_synthetic_generator_labels_are_valid(tmp_path):
    sys.path.insert(0, str(ROOT / "synthetic"))
    import generate

    generate.main(["--out", str(tmp_path / "s"), "--n", "6", "--seed", "3"])
    doc = json.loads((tmp_path / "s" / "labels.json").read_text())
    assert schema.validate_document(doc, [p.name for p in schema.list_images(tmp_path / "s")]) == []


def test_grouped_folds_never_split_a_group():
    from d2qual.train import grouped_folds, load_groups

    ids, labels = [], {}
    rng = random.Random(0)
    for scene in range(12):
        for shot in range(rng.randint(1, 6)):
            iid = f"scene{scene:02d}_shot{shot}.jpg"
            ids.append(iid)
            labels[iid] = [rng.choice(schema.CLASSES) for _ in range(4)]
    groups = load_groups(ids, group_regex=r"^(scene\d+)_")
    assert len(set(groups.values())) == 12
    folds = grouped_folds(ids, labels, groups, 4, seed=1)
    assert sorted(x for f in folds for x in f) == sorted(ids)
    fold_of_group = {}
    for fi, f in enumerate(folds):
        for iid in f:
            assert fold_of_group.setdefault(groups[iid], fi) == fi  # one fold per group
    assert all(folds)


def test_groups_file(tmp_path):
    from d2qual.train import load_groups

    p = tmp_path / "g.csv"
    p.write_text("image_id,group\na.jpg,m1\nb,m1\n")
    g = load_groups(["a.jpg", "b.PNG", "c.jpg"], groups_file=str(p))
    assert g["a.jpg"] == g["b.PNG"] == "m1" and g["c.jpg"].startswith("__single__")


# ── pose-guided limb crops ────────────────────────────────────────────────────

def _fake_kps(w=200, h=160):
    """A casualty lying face-up, head at the top: its LEFT side is on the image RIGHT."""
    import numpy as np

    k = np.zeros((17, 3), np.float32)
    pts = {5: (120, 40), 6: (80, 40), 7: (150, 60), 8: (50, 60), 9: (175, 80), 10: (25, 80),
           11: (112, 90), 12: (88, 90), 13: (125, 120), 14: (75, 120), 15: (130, 150), 16: (70, 150)}
    for i, (x, y) in pts.items():
        k[i] = (x * w / 200, y * h / 160, 0.9)
    return k


def test_unrotate_maps_points_back():
    import numpy as np

    from d2qual.pose import _unrotate

    img = Image.new("RGB", (120, 80))
    pt = np.array([[30.0, 20.0]])
    img.putpixel((30, 20), (255, 255, 255))
    for rot in (90, 270):
        r = img.rotate(rot, expand=True)
        a = np.asarray(r).sum(-1)
        ys, xs = np.nonzero(a)
        back = _unrotate(np.array([[xs.mean(), ys.mean()]]), rot, img.size, r.size)
        assert np.abs(back - pt).max() < 1.5, (rot, back)


def test_limb_crops_follow_anatomical_side_and_mirror():
    import numpy as np

    from d2qual.data import limb_crops, mirror_limb_inputs

    img = Image.new("RGB", (200, 160), (0, 0, 0))
    for x in range(100, 200):  # paint the casualty's LEFT half (image right) red
        for y in range(160):
            img.putpixel((x, y), (255, 0, 0))
    crops, feats = limb_crops(img, _fake_kps(), 32)
    assert len(crops) == 4 and feats.shape == (4, 4) and feats[:, 3].tolist() == [1, 1, 1, 1]
    red = [np.asarray(c)[..., 0].mean() for c in crops]
    assert red[0] > red[1] and red[2] > red[3]  # left limbs are on the red side
    mc, mf = mirror_limb_inputs(crops, feats)
    # after mirroring, the "left arm" crop is the mirrored right-arm crop
    assert np.array_equal(np.asarray(mc[0]), np.asarray(crops[1])[:, ::-1])
    assert np.array_equal(mf, feats[[1, 0, 3, 2]])
    blank, f0 = limb_crops(img, None, 32)
    assert f0.sum() == 0 and len(blank) == 4


def test_limb_crop_model_trains_and_predicts(image_dir, tmp_path):
    pytest.importorskip("timm")
    from d2qual import train
    from d2qual.predict import run
    from d2qual.review import review

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_doc(LABELS)))
    kp_path = tmp_path / "kp.json"
    kp_path.write_text(json.dumps({k: _fake_kps().tolist() for k in LABELS}))
    wdir = tmp_path / "weights"
    train.main(["--images", str(image_dir), "--labels", str(labels_path), "--out", str(wdir),
                "--backbone", "resnet18", "--size", "64", "--crop-size", "32", "--epochs", "1",
                "--batch", "2", "--folds", "2", "--workers", "0", "--no-pretrained",
                "--limb-crops", "--keypoints", str(kp_path)])
    assert (wdir / "oof_predictions.json").exists() and (wdir / "oof_probs.json").exists()
    oof = load_labels(wdir / "oof_predictions.json")
    assert set(oof) == set(LABELS)
    # no pose weights in this test: inference must still produce a valid file (blank crops)
    out = tmp_path / "predictions.json"
    run(image_dir, out, wdir, "t", "1", "e@x", batch_size=3)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_document(doc, [p.name for p in schema.list_images(image_dir)]) == []
    acc = review(image_dir, labels_path, wdir / "oof_predictions.json", tmp_path / "rev",
                 wdir / "oof_probs.json", kp_path)
    assert 0 <= acc <= 1 and (tmp_path / "rev" / "index.html").exists()


# ── bake-off, COCO proxy, VLM plumbing ─────────────────────────────────────────

def test_bakeoff_run_leaderboard_assemble_predict(image_dir, tmp_path, monkeypatch):
    pytest.importorskip("timm")
    sys.path.insert(0, str(ROOT / "scripts"))
    import bakeoff
    from d2qual.predict import run

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_doc(LABELS)))
    tiny = ["--backbone", "resnet18", "--size", "64", "--no-pretrained"]
    monkeypatch.setattr(bakeoff, "CANDIDATES", {"a": tiny, "b": tiny + ["--lr", "3e-3"]})
    runs = tmp_path / "runs"
    bakeoff.main(["run", "--images", str(image_dir), "--labels", str(labels_path), "--runs", str(runs),
                  "--candidates", "a,b", "--folds", "2", "--epochs", "1", "--batch", "2", "--workers", "0"])
    assert (runs / "a" / "oof_probs.json").exists() and (runs / "b" / "oof_probs.json").exists()
    bakeoff.main(["leaderboard", "--images", str(image_dir), "--labels", str(labels_path), "--runs", str(runs)])
    choice = json.loads((runs / "ensemble_choice.json").read_text())
    assert abs(sum(choice["weights"].values()) - 1) < 1e-6
    wdir = tmp_path / "weights"
    bakeoff.main(["assemble", "--runs", str(runs), "--out", str(wdir)])
    ens = json.loads((wdir / "ensemble.json").read_text())["weights"]
    assert ens and all((wdir / m).exists() for m in ens)
    out = tmp_path / "predictions.json"
    run(image_dir, out, wdir, "t", "1", "e@x", batch_size=3)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_document(doc, [p.name for p in schema.list_images(image_dir)]) == []


def test_coco_proxy_limb_rule():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_coco_proxy import limb_label

    crop = (0, 0, 100, 100)
    assert limb_label([(10, 10, 2), (20, 20, 2), (30, 30, 0)], crop) == "no_injury"
    assert limb_label([(10, 10, 2), (20, 20, 1), (30, 30, 0)], crop) == "not_testable"     # occluded
    assert limb_label([(10, 10, 2), (150, 20, 2), (160, 30, 2)], crop) == "not_testable"   # cut out of frame


def test_vlm_questions_name_anatomical_sides():
    from d2qual.vlm_ft import LETTERS, _question

    assert len(LETTERS) == len(schema.CLASSES)
    q = [_question(s) for s in range(4)]
    assert "LEFT ARM" in q[0] and "casualty's own left" in q[0]
    assert "RIGHT LEG" in q[3] and "not the viewer's right" in q[3]


def test_vlm_members_skipped_on_cpu_but_output_valid(image_dir, tmp_path):
    from d2qual.predict import run

    wdir = tmp_path / "weights"
    (wdir / "x__vlm_fold0").mkdir(parents=True)
    (wdir / "x__vlm_fold0" / "vlm_meta.json").write_text(json.dumps({"format": "d2qual-vlm-v1", "base": "none"}))
    out = tmp_path / "predictions.json"
    run(image_dir, out, wdir, "t", "1", "e@x", batch_size=3)  # CPU in tests: VLM skipped, prior used
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert schema.validate_document(doc, [p.name for p in schema.list_images(image_dir)]) == []


def test_fixed_final_epoch_is_kept_by_default(image_dir, tmp_path):
    pytest.importorskip("timm")
    import torch

    from d2qual import train

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(_doc(LABELS)))
    wdir = tmp_path / "w"
    train.main(["--images", str(image_dir), "--labels", str(labels_path), "--out", str(wdir),
                "--backbone", "resnet18", "--size", "64", "--epochs", "3", "--batch", "2",
                "--folds", "2", "--workers", "0", "--no-pretrained"])
    for p in wdir.glob("fold*.pt"):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert ck["metrics"]["epoch"] == 3, ck["metrics"]
    assert json.loads((wdir / "oof_probs.json").read_text())


def test_day_one_inspect_and_run_requires_grouping(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import day_one

    d = tmp_path / "imgs"
    d.mkdir()
    labels = {}
    for s in range(3):
        for k in range(4):
            name = f"scene{s}_{k:03d}.jpg"
            Image.new("RGB", (120, 90), (s * 60, k * 40, 90)).save(d / name)
            labels[name] = ["no_injury", "wound", "amputation", "not_testable"]
    lp = tmp_path / "labels.json"
    lp.write_text(json.dumps(_doc(labels)))
    work = tmp_path / "work"
    day_one.main(["inspect", "--images", str(d), "--labels", str(lp), "--work", str(work)])
    rep = json.loads((work / "inspect.json").read_text())
    assert rep["n"] == 12
    assert any(g["n_groups"] == 3 for g in rep["groupings"])
    assert "Majority-class accuracy" in (work / "inspect.md").read_text()
    with pytest.raises(SystemExit):
        day_one.main(["run", "--images", str(d), "--labels", str(lp), "--work", str(work)])


def test_vlm_base_dirs_resolve_per_model(tmp_path):
    from d2qual.vlm_ft import base_dir_name, resolve_base

    assert base_dir_name("Qwen/Qwen3-VL-2B-Instruct") == "Qwen__Qwen3-VL-2B-Instruct"
    root = tmp_path / "vlm_base"
    assert resolve_base("Qwen/Qwen3-VL-2B-Instruct", root) == "Qwen/Qwen3-VL-2B-Instruct"  # not baked: hub id
    (root / "Qwen__Qwen3-VL-2B-Instruct").mkdir(parents=True)
    (root / "Qwen__Qwen3-VL-2B-Instruct" / "config.json").write_text("{}")
    assert resolve_base("Qwen/Qwen3-VL-2B-Instruct", root).endswith("Qwen__Qwen3-VL-2B-Instruct")


def test_new_candidates_registered():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bakeoff

    for name in ("siglip2_b", "vlm_qwen3vl_2b", "vlm_qwen3vl_4b"):
        assert name in bakeoff.CANDIDATES

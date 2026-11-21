"""
Day one with the DARPA training set: two commands, and a human checkpoint in between.

  1. inspect   read the labels, report class counts per site, image sizes, and candidate
               groupings (scene / manikin / session) guessed from the filenames.
               Writes <work>/inspect.md. Nothing is trained.
       python scripts/day_one.py inspect --images DATA/images --labels DATA/labels.json

  2. run       YOU pick the grouping from the inspect report (or --no-groups if every photo
               really is independent). Then: grouped bake-off -> leaderboard -> ensemble ->
               timed offline inference projected to the test-set size. It STOPS there and prints
               the submit command; it never pushes anything.
       python scripts/day_one.py run --images DATA/images --labels DATA/labels.json \\
           --group-regex '^(scene\\d+)_' [--candidates eva02_b,siglip_b,convnext_s,vlm_qwen2vl_2b] [--test-size N, default 20% of train]

TODO once the data is seen: if the label format is not one d2qual.data.load_labels already
reads (ICD predictions JSON, JSONL, CSV), add a reader there - inspect will fail loudly first.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d2qual.data import load_label_groups, load_labels, match_images  # noqa: E402
from d2qual.schema import CLASSES, SITE_KEYS  # noqa: E402

DEFAULT_CANDIDATES = (
    # fast first, so leaderboard rows arrive early; VLMs (slowest) last
    "eva02_b,siglip_b,convnext_b,"  # COCO re-test 2026-11-20 (PLAN.md); ConvNeXt-B = CNN for ensemble diversity
    "dinov2_l_reg_probe,dinov2_b_reg_limb_probe,dinov2_l_reg_limb_probe,"  # Rohan's approaches A and B
    "vlm_qwen3vl_2b,vlm_qwen3vl_4b"
)
BUDGET_S = 50 * 60  # predict.py's own wall-clock budget inside DARPA's 1 h

# Candidate grouping patterns, most specific first. Each must leave >= 2 images per group on average.
GROUP_PATTERNS = [
    (r"((?:scene|manikin|mannequin|subject|casualty|victim|session|run|trial)[-_ ]?\d+)", "named id"),
    (r"^(.+?)[-_ ]\d+\.\w+$", "name minus trailing frame number"),
    (r"^([^-_ .]+)[-_ ]", "first token"),
    (r"^([^-_ .]+[-_ ][^-_ .]+)[-_ ]", "first two tokens"),
]


def propose_groupings(ids: list[str]) -> list[dict]:
    out = []
    for pat, why in GROUP_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        groups: dict[str, list[str]] = defaultdict(list)
        unmatched = 0
        for i in ids:
            m = rx.search(Path(i).name)
            if m:
                groups[m.group(1).lower()].append(i)
            else:
                unmatched += 1
        if not groups:
            continue
        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        out.append({"regex": pat, "why": why, "n_groups": len(groups), "unmatched": unmatched,
                    "largest": sizes[:5], "median": sizes[len(sizes) // 2],
                    "examples": [f"{k}: {v[:3]}" for k, v in list(groups.items())[:3]]})
    # useful = groups that actually merge images and cover most of the data
    return [g for g in out if g["n_groups"] < len(ids) * 0.9 and g["unmatched"] < len(ids) * 0.1]


def cmd_inspect(a) -> None:
    from PIL import Image

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    raw = load_labels(a.labels)
    labels = match_images(a.images, raw)
    on_disk = sorted(p.name for p in Path(a.images).iterdir() if p.is_file())
    if not labels:
        raise SystemExit(f"0 of {len(raw)} labels matched the {len(on_disk)} files in {a.images}: "
                         "check the label format / id naming before anything else")
    ids = sorted(labels)
    lines = [f"# Day-one inspect: {a.images}", "",
             f"- labels read: {len(raw)}; matched to files: {len(ids)}; files on disk: {len(on_disk)}",
             f"- unlabelled files: {len(set(on_disk) - set(ids))}", "",
             "## Class counts per site", "",
             "| Site | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for s, key in enumerate(SITE_KEYS):
        c = Counter(labels[i][s] for i in ids)
        lines.append(f"| {key} | " + " | ".join(str(c.get(k, 0)) for k in CLASSES) + " |")
    total = Counter(c for i in ids for c in labels[i])
    lines += ["", "Overall: " + ", ".join(f"{k} {total.get(k, 0)}" for k in CLASSES),
              f"Majority-class accuracy (the floor to beat): {max(total.values()) / sum(total.values()):.3f}", ""]

    sizes = Counter()
    for i in ids[: a.size_sample]:
        with Image.open(Path(a.images) / i) as im:
            sizes[(im.width, im.height, "landscape" if im.width >= im.height else "portrait")] += 1
    lines += [f"## Image sizes (first {min(len(ids), a.size_sample)})", ""]
    lines += [f"- {w}x{h} ({o}): {n}" for (w, h, o), n in sizes.most_common(8)]

    props = propose_groupings(ids)
    lines += ["", "## Candidate groupings (pick one - or none - and pass it to `run`)", ""]
    lg = load_label_groups(a.labels)
    if lg:
        by_name = {i.lower(): i for i in ids}
        gmap = {by_name[Path(k).name.lower()]: g for k, g in lg.items() if Path(k).name.lower() in by_name}
        gfile = work / "groups.csv"
        gfile.write_text("image_id,group" + chr(10) + "".join(f"{k},{v}" + chr(10) for k, v in sorted(gmap.items())), encoding="utf-8")
        sizes = sorted(Counter(gmap.values()).values(), reverse=True)
        lines += [f"- **Label file has a casualty/subject id column**: {len(set(gmap.values()))} groups for "
                  f"{len(gmap)} images (median {sizes[len(sizes) // 2]} images/group, largest {sizes[:5]}). "
                  f"Use `--groups-file {gfile}` (recommended over filename guesses).", ""]
    if not props:
        lines.append("No filename pattern groups the images. Ask DARPA / read the data README for "
                     "scene or manikin ids; otherwise use --no-groups and treat CV as optimistic.")
    for g in props:
        lines += [f"- `--group-regex '{g['regex']}'` ({g['why']}): {g['n_groups']} groups, "
                  f"median {g['median']} images/group, largest {g['largest']}, unmatched {g['unmatched']}",
                  f"  - e.g. {'; '.join(g['examples'])}"]
    lines += ["", "Check: photos of the SAME manikin/scene must share a group, or CV leaks near-duplicates.",
              f"Sample filenames: {ids[:8]}"]
    (work / "inspect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (work / "inspect.json").write_text(json.dumps({"n": len(ids), "groupings": props}, indent=1))
    print("\n".join(lines))


def _sh(cmd: list[str], log: Path | None = None) -> int:
    print("$ " + " ".join(cmd), flush=True)
    if log is None:
        return subprocess.call(cmd, cwd=ROOT)
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def cmd_run(a) -> None:
    if not (a.group_regex or a.groups_file or a.no_groups):
        raise SystemExit("choose a grouping from the inspect report: --group-regex / --groups-file, "
                         "or --no-groups to accept optimistic CV")
    work = Path(a.work)
    runs, weights = work / "runs", work / "weights"
    work.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    grp = (["--group-regex", a.group_regex] if a.group_regex else []) + \
          (["--groups-file", a.groups_file] if a.groups_file else [])
    common = ["--images", a.images, "--labels", a.labels]

    bake = [py, "scripts/bakeoff.py", "run", *common, "--runs", str(runs), "--candidates", a.candidates,
            "--folds", str(a.folds), "--epochs", str(a.epochs), *grp, "--", "--data-source", "darpa"]
    if _sh(bake) != 0:
        raise SystemExit("bake-off failed; see runs/<candidate>/train.log")
    _sh([py, "scripts/bakeoff.py", "leaderboard", *common, "--runs", str(runs)])
    _sh([py, "scripts/bakeoff.py", "assemble", "--runs", str(runs), "--out", str(weights)])

    # Timing: run the assembled ensemble offline on a sample, project to the test-set size.
    if a.test_size is None:  # ICD: test set ~20% the size of the training set
        a.test_size = max(1, round(0.2 * len(match_images(a.images, load_labels(a.labels)))))
    sample = sorted(p for p in Path(a.images).iterdir() if p.is_file())[: a.time_sample]
    tin = work / "timing_in"
    tin.mkdir(exist_ok=True)
    for p in sample:
        dst = tin / p.name
        if not dst.exists():
            dst.write_bytes(p.read_bytes())
    t0 = time.time()
    rc = _sh([py, "inference.py", "--input", str(tin), "--output", str(work / "timing_predictions.json"),
              "--weights", str(weights), "--time-budget", str(BUDGET_S)], work / "timing.log")
    dt = time.time() - t0
    per = dt / max(1, len(sample))
    proj = per * a.test_size
    verdict = "OK" if proj < 0.8 * BUDGET_S else ("TIGHT" if proj < BUDGET_S else "TOO SLOW")
    msg = [f"timing: {len(sample)} images in {dt:.0f}s ({per:.2f} s/img incl. model load) -> "
           f"{proj / 60:.1f} min projected for {a.test_size} images vs {BUDGET_S / 60:.0f} min budget: {verdict}",
           f"inference rc={rc}; log {work / 'timing.log'}"]
    if verdict != "OK":
        msg.append("Speed-ups: D2_VLM_TTA=0, fewer folds per candidate in ensemble.json, or drop the slowest member.")
    msg += ["", "STOPPED before submitting. Review runs/leaderboard.md, then:",
            f"  python -m d2qual.review --images {a.images} --labels {a.labels} "
            f"--pred {runs}/<best>/oof_predictions.json --out {work}/review",
            f"  copy {weights}/* to model/weights/, then TEAM=Pinpoint PROJECT=<group/project> TAG=v1 "
            "EMAIL=<contact> ./scripts/submit.sh"]
    (work / "run_summary.txt").write_text("\n".join(msg) + "\n", encoding="utf-8")
    print("\n".join(msg))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "run"):
        p = sub.add_parser(name)
        p.add_argument("--images", required=True)
        p.add_argument("--labels", required=True)
        p.add_argument("--work", default="day_one")
    i = sub.choices["inspect"]
    i.add_argument("--size-sample", type=int, default=300)
    r = sub.choices["run"]
    r.add_argument("--group-regex")
    r.add_argument("--groups-file")
    r.add_argument("--no-groups", action="store_true")
    r.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--test-size", type=int, default=None,
                   help="expected number of test images (default: 20%% of the training set, per ICD v1.0 3.1)")
    r.add_argument("--time-sample", type=int, default=200)
    a = ap.parse_args(argv)
    (cmd_inspect if a.cmd == "inspect" else cmd_run)(a)


if __name__ == "__main__":
    main()

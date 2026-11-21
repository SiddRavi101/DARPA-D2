"""
Turn DARPA's labelled training videos into extra training photos.

    python scripts/extract_video_frames.py --csv DATA/train_dataset.csv --root DATA \
        --out work/frames [--per-video 4]

Writes work/frames/images/<video stem>_f<k>.jpg (k evenly spaced frames, skipping the first and
last 10%) and work/frames/labels.csv with the video's casualty_id and four limb labels.
Pass them to training with
    --extra-images work/frames/images --extra-labels work/frames/labels.csv
Training then: ignores a frame's label for any limb the pose model cannot find (videos carry no
not_testable label, "not annotated for visibility"), never puts a frame of a validation casualty
into training, and never scores on frames - CV accuracy stays photo-only.

Needs an ffmpeg binary (system ffmpeg, or `pip install imageio-ffmpeg`, which bundles one).
DARPA data: frames stay on the training box like the photos; never commit or upload them.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("no ffmpeg: install it or `pip install imageio-ffmpeg`")


def duration_s(ff: str, path: Path) -> float:
    err = subprocess.run([ff, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        raise ValueError(f"{path.name}: no duration in ffmpeg output")
    h, mi, se = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(se)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="DARPA label CSV (rows with media_format=video are used)")
    ap.add_argument("--root", required=True, help="folder the CSV filepaths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=4)
    a = ap.parse_args(argv)
    ff = ffmpeg_exe()
    out_img = Path(a.out) / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cols = {c.strip().lower(): c for c in rows[0]}
    vids = [r for r in rows if r[cols["media_format"]].strip().lower() == "video"]
    sites = ["arm_left", "arm_right", "leg_left", "leg_right"]
    out_rows, failed = [], 0
    for r in vids:
        src = Path(a.root) / r[cols["filepath"]]
        if not src.exists():  # tolerate CSV paths relative to a different root
            hits = list(Path(a.root).rglob(Path(r[cols["filepath"]]).name))
            if not hits:
                failed += 1
                print(f"missing {src}")
                continue
            src = hits[0]
        try:
            dur = duration_s(ff, src)
        except ValueError as e:
            failed += 1
            print(e)
            continue
        k = a.per_video
        for j in range(k):
            t = dur * (0.1 + 0.8 * (j + 0.5) / k)
            name = f"{src.stem}_f{j}.jpg"
            dst = out_img / name
            if not dst.exists():
                subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(src),
                                "-frames:v", "1", "-q:v", "2", str(dst)], check=False)
            if dst.exists():
                out_rows.append({"filepath": name, "casualty_id": r[cols["casualty_id"]], "media_format": "image",
                                 **{s: r[cols[s]] for s in sites}})
    with open(Path(a.out) / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "casualty_id", "media_format", *sites])
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(vids)} videos -> {len(out_rows)} frames in {out_img} ({failed} videos failed)")


if __name__ == "__main__":
    main()

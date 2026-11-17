"""
Generate a synthetic demo scenario (no human-subjects data):
  transcript.srt  — scripted medic narration of a blast casualty
  accel_chest.csv — 50 Hz accelerometer: rest → litter carry → CPR at 110/min → rest

    python examples/make_demo.py examples/demo_scenario
    python -m d2sprint both --input examples/demo_scenario --output out/demo
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT = [
    (5, "Contact, IED blast, one casualty down."),
    (18, "He's got a right leg amputation, arterial bleeding from the right thigh."),
    (31, "Tourniquet going on right leg high and tight, time 1432."),
    (52, "Tourniquet is on the right leg, bleeding controlled."),
    (70, "Shrapnel wound to the left chest, it's sucking air."),
    (84, "Chest seal applied left chest."),
    (101, "Vitals: pulse one thirty radial weak, resps twenty eight, sats eighty eight."),
    (118, "He responds to voice. Pain is eight out of ten."),
    (140, "IO access established, sternal."),
    (160, "Giving TXA two grams IV slow push."),
    (190, "Tourniquet is on the right leg, still holding."),
    (230, "Hypothermia kit on, blizzard blanket wrapped."),
    (260, "Ketamine fifty milligrams IM given for pain."),
    (300, "Moving him to the litter now."),
    (380, "He's unresponsive, no pulse, starting CPR."),
    (445, "Pulse is back, carotid, rate one ten. Evac category urgent surgical, nine line sent."),
]


def srt_time(t: float) -> str:
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(t % 1 * 1000):03d}"


def main(out: str) -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "transcript.srt", "w", encoding="utf-8") as f:
        for i, (t, text) in enumerate(SCRIPT, 1):
            f.write(f"{i}\n{srt_time(t)} --> {srt_time(t + 6)}\nMedic: {text}\n\n")

    fs, rng = 50, np.random.default_rng(0)
    t = np.arange(0, 480, 1 / fs)
    xyz = np.zeros((len(t), 3))
    xyz[:, 2] = 9.81
    xyz += rng.normal(0, 0.05, xyz.shape)
    carry = (t >= 300) & (t < 375)
    xyz[carry] += rng.normal(0, 2.0, (carry.sum(), 3))
    cpr = (t >= 382) & (t < 440)
    xyz[cpr, 2] += 6.0 * np.sin(2 * np.pi * (110 / 60) * t[cpr])
    with open(d / "accel_chest.csv", "w", encoding="utf-8") as f:
        f.write("time,x,y,z\n")
        for ti, (x, y, z) in zip(t, xyz):
            f.write(f"{ti:.3f},{x:.4f},{y:.4f},{z:.4f}\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "demo_scenario"))

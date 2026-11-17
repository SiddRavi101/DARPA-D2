"""
Container entry point (ICD v1.1 §3):

    docker run --rm --gpus all --network none \
        -v /input:/app/input:ro -v /output:/app/output:rw team_image@<digest>

Submission metadata comes from env vars baked in at build time
(see Dockerfile ARGs) so the email never lives in the source tree.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d2qual import __version__  # noqa: E402
from d2qual.predict import run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("D2_INPUT", "/app/input"))
    ap.add_argument("--output", default=os.environ.get("D2_OUTPUT", "/app/output/predictions.json"))
    ap.add_argument("--weights", default=os.environ.get("D2_WEIGHTS", str(Path(__file__).parent / "model" / "weights")))
    ap.add_argument("--time-budget", type=float, default=float(os.environ.get("D2_TIME_BUDGET_S", 50 * 60)))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("D2_BATCH", 16)))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.output)
    if out.is_dir() or not out.suffix:
        out = out / "predictions.json"

    run(
        input_dir=args.input,
        output_path=out,
        weights_dir=args.weights,
        team_name=os.environ.get("D2_TEAM_NAME", "secondlife"),
        version=os.environ.get("D2_SUBMISSION_VERSION", __version__),
        email=os.environ.get("D2_CONTACT_EMAIL", "set-D2_CONTACT_EMAIL-at-build@example.com"),
        time_budget_s=args.time_budget,
        batch_size=args.batch_size,
    )
    return 0  # always 0: a valid predictions.json is on disk even on internal failure


if __name__ == "__main__":
    sys.exit(main())

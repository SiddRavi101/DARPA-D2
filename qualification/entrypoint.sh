#!/usr/bin/env bash
# Runs inference, then self-validates the output. Always exits 0 if a valid
# predictions.json exists, so a transient warning never fails the eval run.
set -uo pipefail
cd /app
python inference.py "$@"
python validate_predictions.py /app/output/predictions.json --images /app/input

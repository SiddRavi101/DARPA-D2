# D2 Sprint Part 2 qualification container (ICD v1.1)
#   build:  docker build -t <team>-d2-qualification-task:<tag> \
#             --build-arg TEAM_NAME=<team> --build-arg CONTACT_EMAIL=<email> qualification/
#   run:    docker run --rm --gpus all --network none \
#             -v /input:/app/input:ro -v /output:/app/output:rw <image>
#
# CUDA 12.4 runtime works with the eval hosts' driver >= 580 (A40 / L40 / H100).
# No NVIDIA kernel driver is installed in the image (ICD §5).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG TEAM_NAME=secondlife
ARG CONTACT_EMAIL=set-at-build@example.com
ARG SUBMISSION_VERSION=1.0.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    D2_TEAM_NAME=${TEAM_NAME} \
    D2_CONTACT_EMAIL=${CONTACT_EMAIL} \
    D2_SUBMISSION_VERSION=${SUBMISSION_VERSION}

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache

COPY d2qual/ d2qual/
COPY inference.py validate_predictions.py entrypoint.sh ./
# Trained checkpoints (fold*.pt / full.pt) — produced by `python -m d2qual.train`.
# Optional: Gemma 4 weights under model/gemma for the zero-shot fallback.
COPY model/ model/
RUN chmod +x entrypoint.sh && mkdir -p /app/input /app/output \
    && python -c "import d2qual.predict, timm, PIL; print('import ok')"

ENTRYPOINT ["/app/entrypoint.sh"]

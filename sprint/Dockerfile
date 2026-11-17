# D2 Sprint final-event container. One Dockerfile, two images (one per lane):
#   docker build --build-arg LANE=documentation    -t d2-documentation:dev    sprint/
#   docker build --build-arg LANE=decision_support -t d2-decision-support:dev sprint/
#   docker run --rm --gpus all --network none -v $IN:/app/input:ro -v $OUT:/app/output:rw <image>
#
# Bake offline models first (needs internet once):  python scripts/fetch_models.py
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG LANE=both
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    D2_LANE=${LANE} D2_WHISPER_MODEL=/app/models/whisper D2_EMBED_CACHE=/app/models/fastembed \
    D2_LLM_PATH=/app/models/gemma

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && rm -rf /root/.cache

COPY d2sprint/ d2sprint/
COPY data/ data/
COPY models/ models/
RUN python3 -c "import d2sprint.pipeline; print('import ok')"

ENTRYPOINT ["python3", "-m", "d2sprint"]
CMD []

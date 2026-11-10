# CeduSync backend image for Railway (CPU only).
# Python 3.12 as pinned in .python-version / runtime.txt.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Whisper (~/.cache/whisper) and Hugging Face (faster-whisper, DistilBERT) both cache under
    # XDG_CACHE_HOME. Mount a Railway volume at /data so models are not downloaded on every restart.
    XDG_CACHE_HOME=/data/cache

# ffmpeg: openai-whisper decodes non-WAV uploads with it. libzbar0: pyzbar (QR decode).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel adds several GB of CUDA packages),
# same versions as the laptop venv; then the rest of requirements.txt.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchaudio==2.11.0 \
    && pip install -r requirements.txt

COPY . .

# One worker: each extra worker loads the models again.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

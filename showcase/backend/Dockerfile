# RaceMind AI Research Lab — backend image (CPU inference).
FROM python:3.12-slim

# System deps:
#   swig + build-essential -> compile box2d-py (no Linux wheel)
#   ffmpeg                 -> MP4 encoding
#   libgl1 + libglib2.0-0  -> rendering runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch FIRST. The default index pulls the multi-GB CUDA build,
# which can exhaust disk/memory on the host and slow the build to a crawl.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The model is git-ignored: either bake best.zip in here, or set MODEL_URL to
# download it on startup (see .env.example).
ENV MODEL_PATH=/app/best.zip \
    VIDEO_DIR=/app/videos \
    MAX_STEPS=600 \
    PORT=8000 \
    SDL_VIDEODRIVER=dummy \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# Render/Fly inject $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

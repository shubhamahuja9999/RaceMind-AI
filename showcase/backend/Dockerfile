# RaceMind AI Research Lab — backend image (CPU inference)
FROM python:3.12-slim

# System deps for Box2D / rendering / ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The model must be present at build/deploy time. Either bake it into the image
# by copying best.zip into this directory, or mount it and set MODEL_PATH.
ENV MODEL_PATH=/app/best.zip \
    VIDEO_DIR=/app/videos \
    MAX_STEPS=600 \
    PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

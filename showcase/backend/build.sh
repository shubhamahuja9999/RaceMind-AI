#!/usr/bin/env bash
# Render build script for the RaceMind AI backend (native Python, no Docker).
# Set the Render "Build Command" to:  bash build.sh
set -e

pip install --upgrade pip

# Install SWIG first so box2d compiles (it has no wheel for Python 3.12).
pip install swig

# CPU-only PyTorch (avoids the multi-GB CUDA build).
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else.
pip install -r requirements.txt

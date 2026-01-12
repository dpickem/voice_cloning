# Voice Cloning TTS Server
# Base image with CUDA support for GPU acceleration

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including Python 3.11
# Python 3.11+ avoids TTS's strict numpy==1.22.0 requirement that conflicts with librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install torch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
# --ignore-installed handles distutils packages (like blinker) that can't be cleanly uninstalled
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application source code
COPY src/ .

# Copy scripts (for preprocessing inside container)
COPY scripts/preprocess_audio.py ./scripts/

# Create non-root user to avoid root-owned files in bind mounts
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} appuser

# Create directories for volumes with proper ownership
RUN mkdir -p /app/voice_references /app/audio_output /app/logs /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TTS_HOME=/app/models
ENV COQUI_TOS_AGREED=1

# Pre-download the XTTS-v2 model during build (optional, makes first start faster)
# Uncomment the following lines if you want to bake the model into the image
# RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["python", "server.py"]

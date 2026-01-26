# Voice Cloning TTS Server — Qwen3-TTS (SOTA) + XTTS-v2
# Base image with CUDA support for GPU acceleration

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including Python 3.12 (required for qwen-tts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install torch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention 2 for Qwen3-TTS (requires devel image for CUDA headers)
# Limit jobs to avoid OOM during compilation
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# Install remaining dependencies
# --ignore-installed handles distutils packages (like blinker) that can't be cleanly uninstalled
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy application source code
COPY src/ .

# Copy scripts (for preprocessing and fine-tuning inside container)
COPY scripts/ ./scripts/

# Create non-root user to avoid root-owned files in bind mounts
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} appuser

# Create directories for volumes with proper ownership
RUN mkdir -p /app/voice_references /app/audio_output /app/logs /app/models /app/finetuned_model && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TTS_HOME=/app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV COQUI_TOS_AGREED=1

# Default to Qwen3-TTS backend (can be overridden via TTS_BACKEND env var)
ENV TTS_BACKEND=qwen3-tts

# Pre-download the Qwen3-TTS model during build (optional, makes first start faster)
# Uncomment the following lines if you want to bake the model into the image
# RUN python -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Expose API port
EXPOSE 8080

# Health check (longer start period for model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["python", "server.py"]

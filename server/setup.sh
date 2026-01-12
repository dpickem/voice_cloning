#!/bin/bash
# Setup script for Voice Cloning TTS Server (Linux/Docker)
# Run this script on the Linux server to prepare the environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "Voice Cloning TTS Server Setup"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
echo "Checking prerequisites..."

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Install Docker: https://docs.docker.com/engine/install/"
    exit 1
fi
echo "  ✓ Docker installed"

# Check for Docker Compose
if ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose is not available"
    echo "Docker Compose should be included with Docker Desktop or docker-compose-plugin"
    exit 1
fi
echo "  ✓ Docker Compose available"

# Check for NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit not detected"
    echo "GPU acceleration may not work without it."
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ✓ NVIDIA Container Toolkit available"
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ GPU detected: $GPU_NAME"
else
    echo "WARNING: nvidia-smi not found, cannot verify GPU"
fi

echo ""

# -----------------------------------------------------------------------------
# Create required directories
# -----------------------------------------------------------------------------
echo "Creating directories..."

cd "$SCRIPT_DIR"

# Local directories for Docker volume mounts
mkdir -p audio_output
mkdir -p logs
mkdir -p models
mkdir -p finetuned_model

# Data directories at project root (shared across components)
mkdir -p "$PROJECT_ROOT/data/voice_references"
mkdir -p "$PROJECT_ROOT/data/training"

# Create symlink for voice_references if it doesn't exist
if [ ! -e "voice_references" ]; then
    ln -sf "$PROJECT_ROOT/data/voice_references" voice_references
    echo "  ✓ Created symlink: voice_references -> ../data/voice_references"
else
    echo "  ✓ voice_references already exists"
fi

# Create symlink for training_data if it doesn't exist
if [ ! -e "training_data" ]; then
    ln -sf "$PROJECT_ROOT/data/training" training_data
    echo "  ✓ Created symlink: training_data -> ../data/training"
else
    echo "  ✓ training_data already exists"
fi

echo "  ✓ All directories ready"
echo ""

# -----------------------------------------------------------------------------
# Setup environment file
# -----------------------------------------------------------------------------
if [ ! -f "config/.env" ]; then
    echo "Creating .env file from template..."
    cp config/env.example config/.env
    echo "  ✓ Created config/.env (edit to customize settings)"
else
    echo "  ✓ config/.env file already exists"
fi

echo ""

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
echo "Building Docker image..."
echo "(This may take several minutes on first run)"
echo ""

docker compose -f docker/docker-compose.yml build

echo ""
echo "  ✓ Docker image built successfully"
echo ""

# -----------------------------------------------------------------------------
# Summary and next steps
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
echo "  $SCRIPT_DIR/"
echo "  ├── src/                 -> Application source code"
echo "  ├── scripts/             -> Processing & training scripts"
echo "  ├── config/              -> Configuration files"
echo "  ├── docker/              -> Docker files"
echo "  ├── tests/               -> Test files"
echo "  ├── voice_references     -> symlink to ../data/voice_references"
echo "  ├── training_data        -> symlink to ../data/training"
echo "  ├── audio_output/        -> Generated audio files"
echo "  ├── logs/                -> Server logs"
echo "  ├── models/              -> Cached TTS models (~2-3GB)"
echo "  └── finetuned_model/     -> Fine-tuned model weights"
echo ""
echo "Next steps:"
echo ""
echo "  1. Add voice reference files:"
echo "     cp your_voice.wav ../data/voice_references/"
echo ""
echo "  2. Start the server:"
echo "     docker compose -f docker/docker-compose.yml up -d"
echo ""
echo "  3. Check server status:"
echo "     docker compose -f docker/docker-compose.yml logs -f voice-tts"
echo "     curl http://localhost:8080/health"
echo ""
echo "  4. Test synthesis:"
echo "     curl -X POST http://localhost:8080/synthesize \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"text\": \"Hello world\", \"voice\": \"your_voice.wav\"}'"
echo ""
echo "API will be available at: http://$(hostname -I | awk '{print $1}'):8080"
echo ""

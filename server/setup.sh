#!/bin/bash
# Setup script for Voice Cloning TTS Server
# Supports both Docker deployment and local conda environment for development
#
# Usage:
#   ./setup.sh              # Interactive mode (asks what to set up)
#   ./setup.sh --docker     # Docker setup only
#   ./setup.sh --conda      # Local conda environment only
#   ./setup.sh --all        # Both Docker and conda

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default: interactive mode
SETUP_DOCKER=false
SETUP_CONDA=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            SETUP_DOCKER=true
            shift
            ;;
        --conda)
            SETUP_CONDA=true
            shift
            ;;
        --all)
            SETUP_DOCKER=true
            SETUP_CONDA=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--docker] [--conda] [--all]"
            echo "  --docker  Set up Docker environment only"
            echo "  --conda   Set up local conda environment only"
            echo "  --all     Set up both Docker and conda"
            echo "  (no args) Interactive mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Interactive mode if no flags specified
if [[ "$SETUP_DOCKER" == "false" && "$SETUP_CONDA" == "false" ]]; then
    echo "=============================================="
    echo "Voice Cloning TTS Server Setup"
    echo "=============================================="
    echo ""
    echo "What would you like to set up?"
    echo "  1) Docker environment (for production deployment)"
    echo "  2) Local conda environment (for development/fine-tuning)"
    echo "  3) Both"
    echo ""
    read -p "Enter choice [1-3]: " choice
    case $choice in
        1) SETUP_DOCKER=true ;;
        2) SETUP_CONDA=true ;;
        3) SETUP_DOCKER=true; SETUP_CONDA=true ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
    echo ""
fi

echo "=============================================="
echo "Voice Cloning TTS Server Setup"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
echo "Checking prerequisites..."

# Check for NVIDIA GPU (needed for both Docker and local)
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ GPU detected: $GPU_NAME"
else
    echo "WARNING: nvidia-smi not found, cannot verify GPU"
fi

# Docker-specific prerequisites
if [[ "$SETUP_DOCKER" == "true" ]]; then
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
fi

# Conda-specific prerequisites
if [[ "$SETUP_CONDA" == "true" ]]; then
    if ! command -v conda &> /dev/null; then
        echo "ERROR: Conda is not installed"
        echo "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    echo "  ✓ Conda installed"
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
# Build Docker image (if requested)
# -----------------------------------------------------------------------------
if [[ "$SETUP_DOCKER" == "true" ]]; then
    echo "Building Docker image..."
    echo "(This may take several minutes on first run)"
    echo ""

    docker compose -f docker/docker-compose.yml build

    echo ""
    echo "  ✓ Docker image built successfully"
    echo ""
fi

# -----------------------------------------------------------------------------
# Setup local conda environment (if requested)
# -----------------------------------------------------------------------------
if [[ "$SETUP_CONDA" == "true" ]]; then
    CONDA_ENV_NAME="voice-tts"
    PYTHON_VERSION="3.11"

    echo "Setting up local conda environment..."
    echo ""

    # Check if environment already exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        echo "Conda environment '${CONDA_ENV_NAME}' already exists."
        read -p "Recreate it? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            conda env remove -n "${CONDA_ENV_NAME}" -y
        else
            echo "Skipping conda environment creation."
            echo "To activate: conda activate ${CONDA_ENV_NAME}"
            SKIP_CONDA_CREATE=true
        fi
    fi

    if [[ "${SKIP_CONDA_CREATE}" != "true" ]]; then
        echo "Creating conda environment '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
        conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y

        echo ""
        echo "Installing PyTorch with CUDA support..."
        # Use conda run to execute in the new environment
        conda run -n "${CONDA_ENV_NAME}" pip install torch==2.1.0 torchaudio==2.1.0 \
            --index-url https://download.pytorch.org/whl/cu121

        echo ""
        echo "Installing project dependencies..."
        conda run -n "${CONDA_ENV_NAME}" pip install -r requirements.txt

        echo ""
        echo "  ✓ Conda environment '${CONDA_ENV_NAME}' created successfully"
    fi

    echo ""
fi

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

if [[ "$SETUP_DOCKER" == "true" ]]; then
    echo "  Docker deployment:"
    echo "  ------------------"
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
    echo "  API will be available at: http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):8080"
    echo ""
fi

if [[ "$SETUP_CONDA" == "true" ]]; then
    echo "  Local development (conda):"
    echo "  --------------------------"
    echo "  Activate the environment:"
    echo "     conda activate voice-tts"
    echo ""
    echo "  Run fine-tuning:"
    echo "     cd $SCRIPT_DIR"
    echo "     python scripts/finetune_xtts.py --config config/finetune_config.json"
    echo ""
    echo "  Run tests:"
    echo "     python -m pytest tests/"
    echo ""
    echo "  Run the server locally (without Docker):"
    echo "     python src/server.py"
    echo ""
fi

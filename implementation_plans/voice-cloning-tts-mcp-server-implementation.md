# Voice Cloning TTS with MCP Server — Implementation Plan

**Created:** 2026-01-08  
**Updated:** 2026-01-26  
**Status:** 🟢 Ready for Implementation  
**Tags:** #implementation-plan #tts #voice-cloning #mcp #cursor #docker #qwen3-tts  
**Based on:** [[voice-cloning-tts-mcp-server]] (Design Proposal)

> **Note:** Updated January 2026 to use **Qwen3-TTS** (SOTA open-source model) instead of XTTS-v2.

---

## Quick Reference

| Component | Location | Access |
|-----------|----------|--------|
| **TTS Server** | `10.111.79.180` (alias: `desktop`) | SSH |
| **TTS API Port** | `8080` | HTTP |
| **Container Runtime** | Docker with NVIDIA GPU support | |
| **MCP Server** | Local Mac | Cursor integration |

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Phase 1: Docker Setup on Linux Server](#2-phase-1-docker-setup-on-linux-server)
3. [Phase 2: Voice Reference Recording](#3-phase-2-voice-reference-recording)
4. [Phase 2A: Voice Model Fine-Tuning (Optional)](#4-phase-2a-voice-model-fine-tuning-optional)
5. [Phase 3: TTS Docker Container](#5-phase-3-tts-docker-container)
6. [Phase 4: MCP Server for Cursor](#6-phase-4-mcp-server-for-cursor)
7. [Phase 5: Integration & Testing](#7-phase-5-integration--testing)
8. [Maintenance & Operations](#8-maintenance--operations)
9. [Troubleshooting Guide](#9-troubleshooting-guide)

---

## 1. Prerequisites

### 1.1 Linux Server (desktop @ 10.111.79.180)

- [ ] **GPU:** NVIDIA GPU with 8GB+ VRAM
- [ ] **NVIDIA Driver:** Version 525.60.13 or higher
- [ ] **Docker:** Version 24.0 or higher
- [ ] **NVIDIA Container Toolkit:** For GPU passthrough to containers
- [ ] **RAM:** 16GB minimum
- [ ] **Storage:** 20GB free space for Docker images, model weights, and audio files

#### Verify GPU and NVIDIA Driver

```bash
ssh desktop

# Check NVIDIA GPU and driver version
nvidia-smi

# Expected output should show driver version >= 525.60.13
```

#### Install Docker (if not installed)

```bash
ssh desktop

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
docker run hello-world
```

#### Install NVIDIA Container Toolkit

```bash
ssh desktop

# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 1.2 Mac Client (Local Machine)

- [ ] **Python:** 3.9 or higher
- [ ] **Cursor:** Latest version with MCP support
- [ ] **Audio output:** Working speakers or headphones
- [ ] **Network:** SSH access to `desktop` (10.111.79.180)

#### Verify SSH Access

```bash
# Test SSH connection
ssh desktop "echo 'SSH connection successful'"

# Add SSH config if not already present
cat >> ~/.ssh/config << 'EOF'
Host desktop
    HostName 10.111.79.180
    User YOUR_USERNAME
    IdentityFile ~/.ssh/id_rsa
EOF
```

---

## 2. Phase 1: Docker Setup on Linux Server

### 2.1 Create Project Directory Structure

```bash
ssh desktop

# Create project directory
mkdir -p ~/voice-tts-server
cd ~/voice-tts-server

# Create directory structure
mkdir -p {voice_references,audio_output,logs,models}

# Directory structure:
# ~/voice-tts-server/
# ├── Dockerfile               # Container build instructions
# ├── docker-compose.yml       # Container orchestration
# ├── requirements.txt         # Python dependencies
# ├── tts_server.py           # FastAPI TTS server
# ├── preprocess_audio.py     # Audio preprocessing utility
# ├── voice_references/       # Voice clone reference audio (mounted volume)
# ├── audio_output/           # Generated audio files (mounted volume)
# ├── logs/                   # Server logs (mounted volume)
# └── models/                 # Cached model weights (mounted volume)
```

### 2.2 Create Requirements File

```bash
cd ~/voice-tts-server

cat > requirements.txt << 'EOF'
# Core TTS - Qwen3-TTS (SOTA voice cloning)
qwen-tts>=1.0.0
torch>=2.1.0
torchaudio>=2.1.0

# FlashAttention 2 for memory efficiency (installed separately)
# pip install flash-attn --no-build-isolation

# API Server
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6

# Audio Processing
soundfile>=0.12.1
numpy>=1.24.0
librosa>=0.10.1
noisereduce>=3.0.0

# Utilities
pydantic>=2.5.0
python-dotenv>=1.0.0
EOF
```

### 2.3 Create Dockerfile

```bash
cd ~/voice-tts-server

cat > Dockerfile << 'EOF'
# Voice Cloning TTS Server using Qwen3-TTS (SOTA)
# Base image with CUDA support for GPU acceleration

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Install torch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention 2 (requires CUDA dev tools, hence devel base image)
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# Install remaining dependencies including qwen-tts
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY tts_server.py .
COPY preprocess_audio.py .

# Create directories for volumes
RUN mkdir -p /app/voice_references /app/audio_output /app/logs /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Pre-download the Qwen3-TTS model during build (optional, makes first start faster)
# Uncomment the following lines if you want to bake the model into the image
# RUN python -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["python", "tts_server.py"]
EOF
```

### 2.4 Create Docker Compose Configuration

```bash
cd ~/voice-tts-server

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  voice-tts:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: voice-tts-server
    restart: unless-stopped
    
    # GPU access
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # Port mapping
    ports:
      - "8080:8080"
    
    # Volume mounts for persistent data
    volumes:
      # Voice reference audio files
      - ./voice_references:/app/voice_references
      # Generated audio output
      - ./audio_output:/app/audio_output
      # Server logs
      - ./logs:/app/logs
      # Model cache (persists downloaded models)
      - ./models:/app/models
    
    # Environment variables
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_HOME=/app/models
      - TRANSFORMERS_CACHE=/app/models
      - PYTHONUNBUFFERED=1
      - TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
    
    # Network settings
    networks:
      - tts-network

networks:
  tts-network:
    driver: bridge
EOF
```

### 2.5 Create TTS Server Application

```bash
cd ~/voice-tts-server

cat > tts_server.py << 'EOF'
#!/usr/bin/env python3
"""
Voice Cloning TTS API Server (Docker Edition) — Qwen3-TTS

FastAPI server that provides text-to-speech synthesis using Qwen3-TTS (SOTA)
with voice cloning capabilities. Designed to run in a Docker container
with NVIDIA GPU support.

Features:
    - 3-second rapid voice cloning (SOTA)
    - 10 language support
    - Ultra-low latency streaming (97ms first packet)
    - Instruction-controlled speech generation

Endpoints:
    POST /synthesize - Convert text to speech with voice cloning
    GET /health - Health check
    GET /voices - List available voice references
"""

import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Configuration (can be overridden via environment variables)
VOICE_REFERENCES_DIR = Path(os.getenv("VOICE_REFERENCES_DIR", "/app/voice_references"))
AUDIO_OUTPUT_DIR = Path(os.getenv("AUDIO_OUTPUT_DIR", "/app/audio_output"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "voice_reference.wav")
DEFAULT_VOICE_TEXT = os.getenv("DEFAULT_VOICE_TEXT", "")  # Transcript of reference audio
MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

# Global model instance
tts_model = None
voice_clone_prompt = None  # Cached voice prompt for faster inference

# Language mapping for Qwen3-TTS
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese", 
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field(default="en", description="Language code (en, zh, ja, ko, de, fr, ru, pt, es, it)")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice reference filename")
    voice_text: str = Field(default="", description="Transcript of voice reference (improves quality)")


class TTSResponse(BaseModel):
    """Response model for synthesis results."""
    success: bool
    audio_base64: Optional[str] = None
    duration_seconds: float
    sample_rate: int
    processing_time_ms: float
    text_length: int
    model: str = "Qwen3-TTS"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    gpu_available: bool
    gpu_name: Optional[str]
    cuda_version: Optional[str]
    container: bool
    timestamp: str


class VoiceInfo(BaseModel):
    """Voice reference information."""
    filename: str
    size_bytes: int
    has_transcript: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup."""
    global tts_model, voice_clone_prompt
    
    print("=" * 60)
    print("Voice Cloning TTS Server (Qwen3-TTS)")
    print("=" * 60)
    
    print(f"Loading model: {MODEL_NAME}")
    start_time = time.time()
    
    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")
    if gpu_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load Qwen3-TTS model
    from qwen_tts import Qwen3TTSModel
    
    tts_model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0" if gpu_available else "cpu",
        dtype=torch.bfloat16 if gpu_available else torch.float32,
        attn_implementation="flash_attention_2" if gpu_available else "eager",
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Ensure directories exist
    VOICE_REFERENCES_DIR.mkdir(exist_ok=True)
    AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # List available voice references
    voices = list(VOICE_REFERENCES_DIR.glob("*.wav"))
    print(f"Voice references available: {len(voices)}")
    for v in voices:
        print(f"  - {v.name}")
    
    # Pre-cache default voice prompt if it exists
    default_voice_path = VOICE_REFERENCES_DIR / DEFAULT_VOICE
    if default_voice_path.exists() and DEFAULT_VOICE_TEXT:
        print(f"Pre-caching voice prompt for: {DEFAULT_VOICE}")
        voice_clone_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=str(default_voice_path),
            ref_text=DEFAULT_VOICE_TEXT,
        )
        print("Voice prompt cached ✓")
    
    print("=" * 60)
    print(f"Server ready at http://{HOST}:{PORT}")
    print("=" * 60)
    
    yield
    
    # Cleanup
    print("Shutting down TTS server...")


app = FastAPI(
    title="Voice Cloning TTS API (Qwen3-TTS)",
    description="SOTA text-to-speech synthesis with 3-second voice cloning using Qwen3-TTS",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    gpu_available = torch.cuda.is_available()
    return HealthResponse(
        status="healthy" if tts_model is not None else "unhealthy",
        model_loaded=tts_model is not None,
        model_name=MODEL_NAME,
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
        cuda_version=torch.version.cuda if gpu_available else None,
        container=True,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    """List available voice reference files."""
    voices = []
    for file in VOICE_REFERENCES_DIR.glob("*.wav"):
        # Check if transcript file exists
        transcript_file = file.with_suffix(".txt")
        voices.append(VoiceInfo(
            filename=file.name,
            size_bytes=file.stat().st_size,
            has_transcript=transcript_file.exists()
        ))
    return voices


@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text using cloned voice.
    
    Qwen3-TTS achieves SOTA voice cloning with just 3 seconds of reference audio.
    Providing a transcript of the reference audio improves cloning quality.
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    # Validate voice reference exists
    voice_path = VOICE_REFERENCES_DIR / request.voice
    if not voice_path.exists():
        available = [f.name for f in VOICE_REFERENCES_DIR.glob("*.wav")]
        raise HTTPException(
            status_code=404,
            detail=f"Voice reference '{request.voice}' not found. Available: {available}"
        )
    
    # Get voice transcript if available
    voice_text = request.voice_text
    if not voice_text:
        transcript_file = voice_path.with_suffix(".txt")
        if transcript_file.exists():
            voice_text = transcript_file.read_text().strip()
    
    # Map language code to Qwen3-TTS language name
    language = LANGUAGE_MAP.get(request.language.lower(), "English")
    
    start_time = time.time()
    
    try:
        # Use cached prompt if available and voice matches default
        global voice_clone_prompt
        
        loop = asyncio.get_event_loop()
        
        if voice_clone_prompt and request.voice == DEFAULT_VOICE:
            # Use cached voice prompt for faster inference
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: tts_model.generate_voice_clone(
                    text=request.text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                )
            )
        else:
            # Generate with voice reference
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: tts_model.generate_voice_clone(
                    text=request.text,
                    language=language,
                    ref_audio=str(voice_path),
                    ref_text=voice_text if voice_text else None,
                )
            )
        
        wav = wavs[0]
        
        # Convert to numpy array if needed
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        
        # Write to buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV', subtype='PCM_16')
        audio_bytes = buffer.getvalue()
        
        processing_time = (time.time() - start_time) * 1000
        duration = len(wav) / sr
        
        # Encode as base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return TTSResponse(
            success=True,
            audio_base64=audio_base64,
            duration_seconds=duration,
            sample_rate=sr,
            processing_time_ms=processing_time,
            text_length=len(request.text),
            model="Qwen3-TTS"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/synthesize/raw")
async def synthesize_raw(request: TTSRequest):
    """
    Synthesize speech and return raw WAV bytes.
    
    Use this endpoint for direct audio playback without base64 encoding overhead.
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    
    voice_path = VOICE_REFERENCES_DIR / request.voice
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice reference '{request.voice}' not found")
    
    # Get voice transcript if available
    voice_text = request.voice_text
    if not voice_text:
        transcript_file = voice_path.with_suffix(".txt")
        if transcript_file.exists():
            voice_text = transcript_file.read_text().strip()
    
    language = LANGUAGE_MAP.get(request.language.lower(), "English")
    
    try:
        global voice_clone_prompt
        loop = asyncio.get_event_loop()
        
        if voice_clone_prompt and request.voice == DEFAULT_VOICE:
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: tts_model.generate_voice_clone(
                    text=request.text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                )
            )
        else:
            wavs, sr = await loop.run_in_executor(
                None,
                lambda: tts_model.generate_voice_clone(
                    text=request.text,
                    language=language,
                    ref_audio=str(voice_path),
                    ref_text=voice_text if voice_text else None,
                )
            )
        
        wav = wavs[0]
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV', subtype='PCM_16')
        
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "X-Duration-Seconds": str(len(wav) / sr),
                "X-Sample-Rate": str(sr),
                "X-Model": "Qwen3-TTS"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tts_server:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=1,  # Single worker for GPU model
        log_level="info"
    )
EOF
```

### 2.6 Create Audio Preprocessing Script

```bash
cd ~/voice-tts-server

cat > preprocess_audio.py << 'EOF'
#!/usr/bin/env python3
"""
Audio preprocessing script for voice cloning reference files.
Normalizes, denoises, and resamples audio to optimal format.

Can be run inside the Docker container or standalone with dependencies installed.
"""

import argparse
import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

def preprocess_audio(
    input_file: str,
    output_file: str,
    target_sr: int = 24000,
    normalize: bool = True,
    denoise: bool = True
) -> str:
    """
    Preprocess audio file for voice cloning.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to save processed audio
        target_sr: Target sample rate (24000 for Qwen3-TTS)
        normalize: Whether to normalize volume
        denoise: Whether to apply noise reduction
    
    Returns:
        Path to processed audio file
    """
    print(f"Loading: {input_file}")
    
    # Load audio
    audio, sr = librosa.load(input_file, sr=None, mono=True)
    print(f"Original: {sr}Hz, {len(audio)/sr:.2f}s, {len(audio)} samples")
    
    # Resample if needed
    if sr != target_sr:
        print(f"Resampling: {sr}Hz → {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Apply noise reduction
    if denoise:
        print("Applying noise reduction...")
        audio = nr.reduce_noise(y=audio, sr=target_sr, prop_decrease=0.8)
    
    # Trim silence from beginning and end
    print("Trimming silence...")
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    # Normalize volume
    if normalize:
        print("Normalizing volume...")
        audio = librosa.util.normalize(audio)
    
    # Save processed audio
    sf.write(output_file, audio, target_sr, subtype='PCM_16')
    
    duration = len(audio) / target_sr
    print(f"Saved: {output_file}")
    print(f"Final: {target_sr}Hz, {duration:.2f}s, {len(audio)} samples")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio for voice cloning")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output file (default: input_processed.wav)")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate (24000 for Qwen3-TTS)")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalization")
    parser.add_argument("--no-denoise", action="store_true", help="Skip noise reduction")
    
    args = parser.parse_args()
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_processed.wav"
    
    preprocess_audio(
        args.input,
        args.output,
        target_sr=args.sr,
        normalize=not args.no_normalize,
        denoise=not args.no_denoise
    )


if __name__ == "__main__":
    main()
EOF

chmod +x preprocess_audio.py
```

### 2.7 Build Docker Image

```bash
ssh desktop
cd ~/voice-tts-server

# Build the Docker image
docker compose build

# This will take 5-10 minutes on first build
# The image will be cached for subsequent builds
```

---

## 3. Phase 2: Voice Reference Recording

### 3.1 Recording Requirements

> **Qwen3-TTS Advantage:** Only **3 seconds** of reference audio needed for SOTA voice cloning!

| Requirement | Specification |
|-------------|---------------|
| **Format** | WAV (16-bit PCM) |
| **Duration** | **3-10 seconds** (Qwen3-TTS needs only 3s minimum!) |
| **Sample Rate** | 24000 Hz or higher |
| **Channels** | Mono |
| **Quality** | Clean, no background noise |
| **Transcript** | Recommended — improves cloning quality |

### 3.2 Recording Script

Read this script naturally for best phoneme coverage:

```text
Hello, my name is [Your Name]. I'm recording this sample to create a voice clone.

The quick brown fox jumps over the lazy dog. Pack my box with five dozen 
liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards 
jump quickly.

I enjoy working on interesting technical projects. Today's weather seems 
quite pleasant, though perhaps a bit chilly. Would you like to join me 
for a coffee later? That sounds like a wonderful idea!

Sphinx of black quartz, judge my vow. The job requires extra pluck and 
zeal from every young wage earner.
```

### 3.3 Upload Voice Reference

```bash
# From Mac: Upload recorded audio to server
scp ~/path/to/your_voice_recording.wav desktop:~/voice-tts-server/voice_references/raw_reference.wav
```

### 3.4 Preprocess Voice Reference (Using Docker)

```bash
ssh desktop
cd ~/voice-tts-server

# Run preprocessing inside the Docker container
docker compose run --rm voice-tts python preprocess_audio.py \
    /app/voice_references/raw_reference.wav \
    -o /app/voice_references/voice_reference.wav

# Verify the processed file exists
ls -la voice_references/
```

### 3.5 Create Voice Transcript (Recommended)

For best cloning quality with Qwen3-TTS, create a transcript file for your voice reference:

```bash
ssh desktop
cd ~/voice-tts-server/voice_references

# Create transcript file with the exact text spoken in your reference audio
cat > voice_reference.txt << 'EOF'
Hello, my name is [Your Name]. I'm recording this sample to create a voice clone. The quick brown fox jumps over the lazy dog.
EOF

# The server will automatically load this transcript when using voice_reference.wav
```

### 3.6 Validate Voice Reference (Using Docker)

```bash
ssh desktop
cd ~/voice-tts-server

# Start the container temporarily to test voice cloning
docker compose run --rm voice-tts python << 'EOF'
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load model
print("Loading Qwen3-TTS model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
)

# Load transcript if available
ref_text = ""
try:
    with open("/app/voice_references/voice_reference.txt", "r") as f:
        ref_text = f.read().strip()
    print(f"Using transcript: {ref_text[:50]}...")
except FileNotFoundError:
    print("No transcript found (optional but recommended)")

# Test synthesis with your voice
print("Testing voice cloning...")
wavs, sr = model.generate_voice_clone(
    text="Hello! This is a test of my cloned voice. The quality should sound natural and clear.",
    language="English",
    ref_audio="/app/voice_references/voice_reference.wav",
    ref_text=ref_text if ref_text else None,
)

sf.write("/app/audio_output/test_output.wav", wavs[0], sr)
print("Test audio saved to: /app/audio_output/test_output.wav")
EOF

# Download and listen to test output
# From Mac:
scp desktop:~/voice-tts-server/audio_output/test_output.wav ~/Desktop/
afplay ~/Desktop/test_output.wav
```

---

## 4. Phase 2A: Voice Model Fine-Tuning (Optional)

> **Note:** This phase is optional. Qwen3-TTS zero-shot cloning with just **3 seconds** of audio produces excellent results for most use cases. Fine-tuning is only needed for production-critical applications requiring near-perfect voice reproduction.

### 4.1 When to Fine-Tune vs. Zero-Shot

| Approach | Audio Needed | Training Time | Quality | Use Case |
|----------|--------------|---------------|---------|----------|
| **Zero-Shot (Qwen3-TTS)** | **3-10 seconds** | None | **Excellent** | Most use cases |
| **Fine-Tuning** | 5-30 minutes | 2-8 hours | Near-perfect | Production-critical, high fidelity |

**Choose fine-tuning when:**
- Zero-shot results don't fully capture subtle voice characteristics
- You need perfect consistency across thousands of generations
- Voice nuances (specific accent patterns, unique cadence) are critical
- You're building a commercial product with high quality bar

> **Qwen3-TTS Fine-Tuning:** See the [official fine-tuning guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) for detailed instructions.

### 4.2 Hardware Requirements for Fine-Tuning

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU VRAM** | 12GB | 16GB+ |
| **System RAM** | 32GB | 64GB |
| **Storage** | 50GB free | 100GB free |
| **Training Time** | 2-4 hours | 4-8 hours |

### 4.3 Recording Training Data

For fine-tuning, you need more audio than zero-shot cloning:

| Quality Level | Audio Duration | Number of Clips | Expected Quality |
|---------------|----------------|-----------------|------------------|
| **Minimum** | 5 minutes | 50+ clips | Noticeable improvement |
| **Good** | 15 minutes | 150+ clips | Significant improvement |
| **Excellent** | 30+ minutes | 300+ clips | Near-perfect reproduction |

#### Recording Guidelines

**Content to Record:**
- Read diverse text: news articles, book passages, technical documentation
- Include various sentence types: questions, exclamations, statements
- Vary emotional tones: neutral, happy, serious, explanatory
- Cover all common phonemes and phoneme combinations

**Sample Recording Script (Extended):**

```text
# Section 1: Pangrams and Phoneme Coverage
The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump!
Sphinx of black quartz, judge my vow.
The five boxing wizards jump quickly.
Jackdaws love my big sphinx of quartz.

# Section 2: Numbers and Technical Content
The server runs on port 8080 with 16 gigabytes of RAM.
Version 2.5.3 was released on January 15th, 2026.
The API endpoint returns a 200 status code on success.
Please enter your 4-digit PIN followed by the pound sign.

# Section 3: Questions and Exclamations
Would you like to proceed with the installation?
How does this feature compare to the previous version?
That's absolutely incredible! I can't believe it worked!
What time should we schedule the meeting for tomorrow?

# Section 4: Conversational Content
Hello, my name is [Your Name]. Nice to meet you.
I've been working on this project for about three months now.
The main challenge was getting the audio quality just right.
Let me explain how this system works step by step.

# Section 5: Longer Passages (for prosody training)
When implementing a voice cloning system, there are several important
considerations to keep in mind. First, the quality of your reference
audio directly impacts the quality of the synthesized output. Second,
the model needs diverse training data to capture the full range of
your vocal characteristics. Finally, proper preprocessing ensures
consistent audio quality across all samples.
```

**Recording Session Tips:**
- Record in multiple sessions to avoid fatigue
- Take breaks every 10-15 minutes
- Keep a glass of water nearby
- Maintain consistent microphone distance
- Re-record any clips with errors or background noise

### 4.4 Create Training Dataset Directory

```bash
ssh desktop
cd ~/voice-tts-server

# Create training dataset structure
mkdir -p training_data/{wavs,processed}

# Directory structure:
# ~/voice-tts-server/training_data/
# ├── wavs/                  # Raw recorded audio clips
# │   ├── clip_001.wav
# │   ├── clip_002.wav
# │   └── ...
# ├── processed/             # Preprocessed audio clips
# │   ├── clip_001.wav
# │   └── ...
# ├── metadata.csv           # Transcriptions
# └── metadata_train.csv     # Training split
# └── metadata_eval.csv      # Evaluation split
```

### 4.5 Upload and Organize Audio Files

```bash
# From Mac: Upload all recorded clips
scp ~/recordings/*.wav desktop:~/voice-tts-server/training_data/wavs/

# SSH to server
ssh desktop
cd ~/voice-tts-server/training_data

# Rename files to sequential format (optional but recommended)
i=1; for f in wavs/*.wav; do
    mv "$f" "wavs/clip_$(printf '%03d' $i).wav"
    ((i++))
done

# Verify files
ls -la wavs/ | head -20
echo "Total clips: $(ls wavs/*.wav | wc -l)"
```

### 4.6 Create Metadata (Transcriptions)

Create a `metadata.csv` file with transcriptions for each audio clip:

```bash
cd ~/voice-tts-server/training_data

# Create metadata.csv with format: filename|transcription
cat > metadata.csv << 'EOF'
clip_001|The quick brown fox jumps over the lazy dog.
clip_002|Pack my box with five dozen liquor jugs.
clip_003|How vexingly quick daft zebras jump!
clip_004|Sphinx of black quartz, judge my vow.
clip_005|The five boxing wizards jump quickly.
EOF

# Add more entries for all your clips...
# Each line: filename_without_extension|exact_transcription
```

**Transcription Tips:**
- Transcriptions must match audio exactly (punctuation matters)
- Use lowercase for consistency (or match your speaking style)
- Include punctuation: periods, commas, question marks
- Don't include filler words unless actually spoken ("um", "uh")

### 4.7 Preprocess Training Audio

```bash
ssh desktop
cd ~/voice-tts-server

# Create preprocessing script for batch processing
cat > preprocess_training_data.py << 'EOF'
#!/usr/bin/env python3
"""
Batch preprocess training audio files for fine-tuning.
Normalizes, denoises, and resamples all clips to consistent format.
"""

import os
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import csv

def preprocess_single(args):
    """Preprocess a single audio file."""
    input_file, output_file, target_sr = args
    
    try:
        # Load audio
        audio, sr = librosa.load(input_file, sr=None, mono=True)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Apply noise reduction
        audio = nr.reduce_noise(y=audio, sr=target_sr, prop_decrease=0.7)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        # Normalize volume
        audio = librosa.util.normalize(audio)
        
        # Save processed audio
        sf.write(output_file, audio, target_sr, subtype='PCM_16')
        
        duration = len(audio) / target_sr
        return (input_file, output_file, duration, None)
        
    except Exception as e:
        return (input_file, None, 0, str(e))


def validate_metadata(metadata_file, wavs_dir):
    """Validate metadata.csv against audio files."""
    errors = []
    valid_entries = []
    
    with open(metadata_file, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) != 2:
                errors.append(f"Invalid row format: {row}")
                continue
            
            filename, transcription = row
            wav_path = Path(wavs_dir) / f"{filename}.wav"
            
            if not wav_path.exists():
                errors.append(f"Audio file not found: {wav_path}")
                continue
            
            if len(transcription.strip()) < 3:
                errors.append(f"Transcription too short for {filename}")
                continue
            
            valid_entries.append((filename, transcription))
    
    return valid_entries, errors


def main():
    parser = argparse.ArgumentParser(description="Preprocess training audio")
    parser.add_argument("--input-dir", default="training_data/wavs", help="Input directory")
    parser.add_argument("--output-dir", default="training_data/processed", help="Output directory")
    parser.add_argument("--metadata", default="training_data/metadata.csv", help="Metadata file")
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate (24000 for Qwen3-TTS)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate metadata
    print(f"Validating metadata: {args.metadata}")
    valid_entries, errors = validate_metadata(args.metadata, input_dir)
    
    if errors:
        print(f"\n⚠️  Found {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"\n✓ Valid entries: {len(valid_entries)}")
    
    # Prepare processing tasks
    tasks = []
    for filename, _ in valid_entries:
        input_file = input_dir / f"{filename}.wav"
        output_file = output_dir / f"{filename}.wav"
        tasks.append((str(input_file), str(output_file), args.sr))
    
    # Process in parallel
    print(f"\nProcessing {len(tasks)} files with {args.workers} workers...")
    
    total_duration = 0
    processed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(preprocess_single, tasks):
            input_file, output_file, duration, error = result
            if error:
                print(f"  ✗ {Path(input_file).name}: {error}")
                failed += 1
            else:
                total_duration += duration
                processed += 1
                if processed % 50 == 0:
                    print(f"  Processed {processed}/{len(tasks)}...")
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"  Processed: {processed} files")
    print(f"  Failed: {failed} files")
    print(f"  Total duration: {total_duration/60:.1f} minutes")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
EOF

chmod +x preprocess_training_data.py

# Run preprocessing inside Docker container
docker compose run --rm voice-tts python preprocess_training_data.py \
    --input-dir /app/training_data/wavs \
    --output-dir /app/training_data/processed \
    --metadata /app/training_data/metadata.csv \
    --workers 4
```

### 4.8 Split Dataset for Training/Evaluation

```bash
ssh desktop
cd ~/voice-tts-server

# Create train/eval split script
cat > split_dataset.py << 'EOF'
#!/usr/bin/env python3
"""Split metadata into training and evaluation sets."""

import csv
import random
import argparse
from pathlib import Path

def split_dataset(metadata_file, train_ratio=0.9, seed=42):
    """Split metadata into train and eval sets."""
    random.seed(seed)
    
    # Read all entries
    entries = []
    with open(metadata_file, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        entries = list(reader)
    
    # Shuffle
    random.shuffle(entries)
    
    # Split
    split_idx = int(len(entries) * train_ratio)
    train_entries = entries[:split_idx]
    eval_entries = entries[split_idx:]
    
    # Write train set
    train_file = metadata_file.replace('.csv', '_train.csv')
    with open(train_file, 'w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(train_entries)
    
    # Write eval set
    eval_file = metadata_file.replace('.csv', '_eval.csv')
    with open(eval_file, 'w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(eval_entries)
    
    print(f"Total entries: {len(entries)}")
    print(f"Training set: {len(train_entries)} ({train_file})")
    print(f"Evaluation set: {len(eval_entries)} ({eval_file})")
    
    return train_file, eval_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="training_data/metadata.csv")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()
    
    split_dataset(args.metadata, args.train_ratio)
EOF

# Run split
python3 split_dataset.py --metadata training_data/metadata.csv --train-ratio 0.9
```

### 4.9 Fine-Tuning Configuration

> **Important:** For Qwen3-TTS fine-tuning, refer to the [official fine-tuning guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) which provides comprehensive instructions and optimized configurations.

```bash
ssh desktop
cd ~/voice-tts-server

# Create fine-tuning configuration for Qwen3-TTS
cat > finetune_config.json << 'EOF'
{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "output_path": "finetuned_model/",
    "training": {
        "batch_size": 1,
        "eval_batch_size": 1,
        "num_epochs": 50,
        "learning_rate": 1e-5,
        "learning_rate_scheduler": "cosine",
        "warmup_steps": 100,
        "gradient_accumulation_steps": 8,
        "max_audio_length": 15,
        "save_checkpoint_every_n_epochs": 10,
        "keep_last_n_checkpoints": 3,
        "use_flash_attention": true,
        "dtype": "bfloat16"
    },
    "data": {
        "train_csv": "training_data/metadata_train.csv",
        "eval_csv": "training_data/metadata_eval.csv",
        "audio_dir": "training_data/processed/",
        "language": "English",
        "sample_rate": 24000
    },
    "logging": {
        "log_every_n_steps": 50,
        "eval_every_n_epochs": 5
    }
}
EOF
```

### 4.10 Create Fine-Tuning Script

```bash
cd ~/voice-tts-server

cat > finetune_xtts.py << 'EOF'
#!/usr/bin/env python3
"""
Fine-tune XTTS-v2 on custom voice dataset.

This script handles the complete fine-tuning process:
1. Load pre-trained XTTS-v2 model
2. Prepare dataset from metadata CSV
3. Run training loop
4. Save fine-tuned model
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime

# Set environment variables before importing TTS
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs


def load_config(config_path: str) -> dict:
    """Load fine-tuning configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def prepare_dataset(config: dict):
    """Prepare training and evaluation datasets."""
    train_samples = []
    eval_samples = []
    
    audio_dir = Path(config['data']['audio_dir'])
    
    # Load training samples
    with open(config['data']['train_csv'], 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                filename, text = parts
                audio_path = audio_dir / f"{filename}.wav"
                if audio_path.exists():
                    train_samples.append({
                        "audio_file": str(audio_path),
                        "text": text,
                        "speaker_name": "custom_voice",
                        "language": config['data']['language']
                    })
    
    # Load eval samples
    with open(config['data']['eval_csv'], 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                filename, text = parts
                audio_path = audio_dir / f"{filename}.wav"
                if audio_path.exists():
                    eval_samples.append({
                        "audio_file": str(audio_path),
                        "text": text,
                        "speaker_name": "custom_voice",
                        "language": config['data']['language']
                    })
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    
    return train_samples, eval_samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune XTTS-v2")
    parser.add_argument("--config", default="finetune_config.json", help="Config file")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 60)
    print("XTTS-v2 Fine-Tuning")
    print("=" * 60)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load XTTS config and model
    print("\nLoading pre-trained XTTS-v2 model...")
    model_config = XttsConfig()
    model_config.load_json(os.path.join(
        os.path.dirname(__file__),
        "models/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
    ))
    
    # Update config with fine-tuning parameters
    model_config.batch_size = config['training']['batch_size']
    model_config.eval_batch_size = config['training']['eval_batch_size']
    model_config.num_epochs = config['training']['num_epochs']
    model_config.lr = config['training']['learning_rate']
    model_config.output_path = str(output_path)
    
    # Initialize model
    model = Xtts.init_from_config(model_config)
    model.load_checkpoint(
        model_config,
        checkpoint_dir="models/tts_models--multilingual--multi-dataset--xtts_v2/",
        use_deepspeed=False
    )
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_samples, eval_samples = prepare_dataset(config)
    
    if len(train_samples) < 10:
        print("⚠️  Warning: Very few training samples. Results may be poor.")
    
    # Setup trainer
    trainer_args = TrainerArgs(
        restore_path=args.resume,
        skip_train_epoch=False,
        start_with_eval=True,
    )
    
    trainer = Trainer(
        trainer_args,
        model_config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Start training
    print("\nStarting fine-tuning...")
    print(f"Output directory: {output_path}")
    print(f"Training for {config['training']['num_epochs']} epochs")
    print("-" * 60)
    
    trainer.fit()
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
EOF

chmod +x finetune_xtts.py
```

### 4.11 Run Fine-Tuning

```bash
ssh desktop
cd ~/voice-tts-server

# Ensure model is downloaded first (if not already cached)
docker compose run --rm voice-tts python -c "
from TTS.api import TTS
TTS('tts_models/multilingual/multi-dataset/xtts_v2')
print('Model ready')
"

# Start fine-tuning (this will take several hours)
docker compose run --rm \
    -v $(pwd)/training_data:/app/training_data \
    -v $(pwd)/finetuned_model:/app/finetuned_model \
    voice-tts python finetune_xtts.py --config finetune_config.json

# Monitor GPU usage in another terminal
ssh desktop "watch -n 1 nvidia-smi"
```

### 4.12 Monitor Training Progress

```bash
# View training logs
ssh desktop
cd ~/voice-tts-server

# Check latest log files
ls -la finetuned_model/

# View training metrics (if using tensorboard)
docker compose run --rm -p 6006:6006 voice-tts tensorboard --logdir /app/finetuned_model/
# Access at http://10.111.79.180:6006

# Check for checkpoints
ls -la finetuned_model/*.pth
```

### 4.13 Test Fine-Tuned Model

```bash
ssh desktop
cd ~/voice-tts-server

# Test the fine-tuned model
docker compose run --rm voice-tts python << 'EOF'
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Load fine-tuned model
print("Loading fine-tuned model...")
config = XttsConfig()
config.load_json("finetuned_model/config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="finetuned_model/")
model.cuda()

# Synthesize test audio
print("Generating test audio...")
outputs = model.synthesize(
    text="Hello! This is a test of the fine-tuned voice model. The quality should be noticeably better than zero-shot cloning.",
    config=config,
    speaker_wav="/app/voice_references/voice_reference.wav",
    language="en"
)

# Save output
import soundfile as sf
sf.write("/app/audio_output/finetuned_test.wav", outputs["wav"], 22050)
print("Saved to: /app/audio_output/finetuned_test.wav")
EOF

# Download and compare
scp desktop:~/voice-tts-server/audio_output/finetuned_test.wav ~/Desktop/
afplay ~/Desktop/finetuned_test.wav
```

### 4.14 Compare Zero-Shot vs Fine-Tuned

```bash
ssh desktop
cd ~/voice-tts-server

# Generate comparison samples
docker compose run --rm voice-tts python << 'EOF'
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf

test_texts = [
    "The voice cloning system is now complete.",
    "How does this compare to the zero-shot approach?",
    "Fine-tuning provides better voice reproduction.",
]

# Zero-shot model
print("Generating zero-shot samples...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

for i, text in enumerate(test_texts):
    wav = tts.tts(
        text=text,
        speaker_wav="/app/voice_references/voice_reference.wav",
        language="en"
    )
    sf.write(f"/app/audio_output/compare_zeroshot_{i+1}.wav", wav, 22050)

# Fine-tuned model
print("Generating fine-tuned samples...")
config = XttsConfig()
config.load_json("finetuned_model/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="finetuned_model/")
model.cuda()

for i, text in enumerate(test_texts):
    outputs = model.synthesize(
        text=text,
        config=config,
        speaker_wav="/app/voice_references/voice_reference.wav",
        language="en"
    )
    sf.write(f"/app/audio_output/compare_finetuned_{i+1}.wav", outputs["wav"], 22050)

print("Comparison samples saved to /app/audio_output/")
EOF

# Download all comparison files
scp desktop:~/voice-tts-server/audio_output/compare_*.wav ~/Desktop/
```

### 4.15 Deploy Fine-Tuned Model

To use the fine-tuned model instead of the base model, update the TTS server:

```bash
ssh desktop
cd ~/voice-tts-server

# Create updated server script that uses fine-tuned model
cat > tts_server_finetuned.py << 'EOF'
#!/usr/bin/env python3
"""
TTS Server using fine-tuned XTTS-v2 model.
Drop-in replacement for tts_server.py with fine-tuned model support.
"""

import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Configuration
VOICE_REFERENCES_DIR = Path(os.getenv("VOICE_REFERENCES_DIR", "/app/voice_references"))
AUDIO_OUTPUT_DIR = Path(os.getenv("AUDIO_OUTPUT_DIR", "/app/audio_output"))
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "voice_reference.wav")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

# Model paths - check for fine-tuned model first
FINETUNED_MODEL_DIR = Path("/app/finetuned_model")
USE_FINETUNED = FINETUNED_MODEL_DIR.exists() and (FINETUNED_MODEL_DIR / "config.json").exists()

# Global model instance
model = None
model_config = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="en")
    voice: str = Field(default=DEFAULT_VOICE)


class TTSResponse(BaseModel):
    success: bool
    audio_base64: Optional[str] = None
    duration_seconds: float
    sample_rate: int
    processing_time_ms: float
    text_length: int
    model_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_config
    
    print("=" * 60)
    print("Voice Cloning TTS Server")
    print("=" * 60)
    
    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")
    
    if USE_FINETUNED:
        print("Loading FINE-TUNED model...")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        model_config = XttsConfig()
        model_config.load_json(str(FINETUNED_MODEL_DIR / "config.json"))
        
        model = Xtts.init_from_config(model_config)
        model.load_checkpoint(model_config, checkpoint_dir=str(FINETUNED_MODEL_DIR))
        
        if gpu_available:
            model.cuda()
        
        print(f"✓ Fine-tuned model loaded from {FINETUNED_MODEL_DIR}")
    else:
        print("Loading ZERO-SHOT model (no fine-tuned model found)...")
        from TTS.api import TTS
        
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu_available)
        print("✓ Zero-shot model loaded")
    
    print("=" * 60)
    print(f"Server ready at http://{HOST}:{PORT}")
    print("=" * 60)
    
    yield
    
    print("Shutting down...")


app = FastAPI(
    title="Voice Cloning TTS API (Fine-tuned)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "fine-tuned" if USE_FINETUNED else "zero-shot",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    voice_path = VOICE_REFERENCES_DIR / request.voice
    if not voice_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice reference not found: {request.voice}")
    
    start_time = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        
        if USE_FINETUNED:
            # Fine-tuned model synthesis
            outputs = await loop.run_in_executor(
                None,
                lambda: model.synthesize(
                    text=request.text,
                    config=model_config,
                    speaker_wav=str(voice_path),
                    language=request.language
                )
            )
            wav = outputs["wav"]
        else:
            # Zero-shot synthesis
            wav = await loop.run_in_executor(
                None,
                lambda: model.tts(
                    text=request.text,
                    speaker_wav=str(voice_path),
                    language=request.language
                )
            )
        
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        
        sample_rate = 22050
        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format='WAV', subtype='PCM_16')
        audio_bytes = buffer.getvalue()
        
        processing_time = (time.time() - start_time) * 1000
        duration = len(wav) / sample_rate
        
        return TTSResponse(
            success=True,
            audio_base64=base64.b64encode(audio_bytes).decode('utf-8'),
            duration_seconds=duration,
            sample_rate=sample_rate,
            processing_time_ms=processing_time,
            text_length=len(request.text),
            model_type="fine-tuned" if USE_FINETUNED else "zero-shot"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/raw")
async def synthesize_raw(request: TTSRequest):
    """Return raw WAV bytes."""
    response = await synthesize(request)
    audio_bytes = base64.b64decode(response.audio_base64)
    
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Duration-Seconds": str(response.duration_seconds),
            "X-Sample-Rate": str(response.sample_rate),
            "X-Model-Type": response.model_type
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server_finetuned:app", host=HOST, port=PORT, reload=False, workers=1)
EOF

# Update Dockerfile to use fine-tuned server
# (or simply copy the file over the original)
cp tts_server_finetuned.py tts_server.py

# Rebuild and restart container
docker compose build
docker compose up -d

# Verify fine-tuned model is being used
curl http://localhost:8080/health | jq '.model_type'
# Should return: "fine-tuned"
```

### 4.16 Fine-Tuning Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of memory** | Batch size too large | Reduce `batch_size` to 1, increase `gradient_accumulation_steps` |
| **Training loss not decreasing** | Learning rate too high/low | Try `learning_rate: 1e-6` or `1e-5` |
| **Overfitting** | Too few samples or too many epochs | Add more training data or reduce epochs |
| **Poor quality output** | Inconsistent training data | Re-record problematic clips, verify transcriptions |
| **Model won't load** | Checkpoint corruption | Resume from earlier checkpoint |

```bash
# Resume training from checkpoint if interrupted
docker compose run --rm voice-tts python finetune_xtts.py \
    --config finetune_config.json \
    --resume finetuned_model/checkpoint_epoch_20.pth
```

---

## 5. Phase 3: TTS Docker Container

### 5.1 Start the TTS Server Container

```bash
ssh desktop
cd ~/voice-tts-server

# Start the container in detached mode
docker compose up -d

# View logs to confirm startup
docker compose logs -f voice-tts

# Wait for "Server ready at http://0.0.0.0:8080" message
# Press Ctrl+C to exit logs (container keeps running)
```

### 5.2 Verify Container Status

```bash
# Check container is running
docker compose ps

# Check GPU is accessible inside container
docker compose exec voice-tts nvidia-smi

# Check health endpoint
curl http://localhost:8080/health | jq
```

### 5.3 Test API Server

```bash
# From Mac: Test health endpoint
curl http://10.111.79.180:8080/health | jq

# Test synthesis
curl -X POST http://10.111.79.180:8080/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the voice cloning API running in Docker.", "language": "en"}' \
  | jq '.success, .duration_seconds, .processing_time_ms'

# Test raw audio endpoint and save to file
curl -X POST http://10.111.79.180:8080/synthesize/raw \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test of the raw audio endpoint from the Docker container."}' \
  -o ~/Desktop/test_api.wav

# Play the audio
afplay ~/Desktop/test_api.wav
```

### 5.4 Container Auto-Start on Boot

The `docker-compose.yml` includes `restart: unless-stopped`, which means:
- Container automatically restarts after system reboot
- Container restarts on crash
- Container stays stopped if manually stopped

To ensure Docker itself starts on boot:

```bash
ssh desktop

# Enable Docker service on boot
sudo systemctl enable docker
```

---

## 6. Phase 4: MCP Server for Cursor

> **Note:** The MCP server runs on your Mac client, not in Docker.

### 6.1 Create MCP Server Directory

```bash
# On Mac (local)
mkdir -p ~/mcp-servers/voice-tts
cd ~/mcp-servers/voice-tts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 6.2 Install MCP Dependencies

```bash
cd ~/mcp-servers/voice-tts
source venv/bin/activate

cat > requirements.txt << 'EOF'
mcp>=1.0.0
httpx>=0.25.0
sounddevice>=0.4.6
soundfile>=0.12.1
numpy>=1.24.0
python-dotenv>=1.0.0
EOF

pip install -r requirements.txt
```

### 6.3 Create MCP Server

```bash
cd ~/mcp-servers/voice-tts

cat > mcp_voice_tts.py << 'EOF'
#!/usr/bin/env python3
"""
MCP Server for Voice Cloning TTS (Qwen3-TTS)

Provides Cursor with text-to-speech capabilities using a remote
voice cloning server (Docker container) running Qwen3-TTS (SOTA).
Plays synthesized audio through local speakers.

Features:
    - SOTA voice cloning with 3-second reference audio
    - 10 language support
    - Ultra-low latency (97ms first packet)

Tools:
    - speak: Convert text to speech and play through speakers
    - speak_summary: Speak a summary aloud
    - list_voices: List available voice references
    - check_tts_server: Check TTS server status
"""

import asyncio
import base64
import io
import os
from typing import Any

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Load environment variables
load_dotenv()

# Configuration
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://10.111.79.180:8080")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "voice_reference.wav")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

# Create MCP server
server = Server("voice-tts")


async def play_audio(audio_bytes: bytes) -> None:
    """Play audio bytes through local speakers."""
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Play audio (blocking)
    sd.play(audio_data, sample_rate)
    sd.wait()


async def synthesize_and_play(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    voice: str = DEFAULT_VOICE
) -> dict[str, Any]:
    """Call TTS server and play resulting audio."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{TTS_SERVER_URL}/synthesize/raw",
            json={
                "text": text,
                "language": language,
                "voice": voice
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"TTS server error: {response.text}")
        
        audio_bytes = response.content
        duration = float(response.headers.get("X-Duration-Seconds", 0))
        sample_rate = int(response.headers.get("X-Sample-Rate", 22050))
        
        # Play audio
        await play_audio(audio_bytes)
        
        return {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "text_length": len(text)
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="speak",
            description="Convert text to speech using your cloned voice (Qwen3-TTS SOTA) and play through speakers. Only needs 3 seconds of reference audio!",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak aloud"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (default: en). Supported: en, zh, ja, ko, de, fr, ru, pt, es, it",
                        "default": "en"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice reference filename (default: voice_reference.wav)",
                        "default": "voice_reference.wav"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="speak_summary",
            description="Summarize content and speak it aloud. Good for giving audio summaries of code, documents, or explanations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The summary text to speak aloud"
                    }
                },
                "required": ["summary"]
            }
        ),
        Tool(
            name="list_voices",
            description="List available voice reference files on the TTS server",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="check_tts_server",
            description="Check the status of the TTS server (Docker container) and its GPU availability",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "speak":
        text = arguments.get("text", "")
        language = arguments.get("language", DEFAULT_LANGUAGE)
        voice = arguments.get("voice", DEFAULT_VOICE)
        
        if not text:
            return [TextContent(type="text", text="Error: No text provided")]
        
        try:
            result = await synthesize_and_play(text, language, voice)
            return [TextContent(
                type="text",
                text=f"✓ Spoke {len(text)} characters ({result['duration_seconds']:.1f}s audio)"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "speak_summary":
        summary = arguments.get("summary", "")
        
        if not summary:
            return [TextContent(type="text", text="Error: No summary provided")]
        
        try:
            result = await synthesize_and_play(summary)
            return [TextContent(
                type="text",
                text=f"✓ Spoke summary ({result['duration_seconds']:.1f}s audio)"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "list_voices":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{TTS_SERVER_URL}/voices")
                
                if response.status_code == 200:
                    voices = response.json()
                    if voices:
                        voice_list = "\n".join([
                            f"  - {v['filename']} ({v['size_bytes'] / 1024:.1f} KB)"
                            for v in voices
                        ])
                        return [TextContent(
                            type="text",
                            text=f"Available voices:\n{voice_list}"
                        )]
                    else:
                        return [TextContent(
                            type="text",
                            text="No voice references found on server"
                        )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Error fetching voices: {response.status_code}"
                    )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "check_tts_server":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{TTS_SERVER_URL}/health")
                
                if response.status_code == 200:
                    health = response.json()
                    status = "✓" if health["status"] == "healthy" else "✗"
                    model_name = health.get("model_name", "Unknown")
                    gpu = health.get("gpu_name", "None")
                    cuda = health.get("cuda_version", "N/A")
                    container = "Docker" if health.get("container") else "Native"
                    return [TextContent(
                        type="text",
                        text=f"TTS Server Status:\n"
                             f"  Status: {status} {health['status']}\n"
                             f"  Model: {model_name}\n"
                             f"  Model loaded: {health['model_loaded']}\n"
                             f"  GPU: {gpu}\n"
                             f"  CUDA: {cuda}\n"
                             f"  Runtime: {container}\n"
                             f"  URL: {TTS_SERVER_URL}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Server returned error: {response.status_code}"
                    )]
        except httpx.ConnectError:
            return [TextContent(
                type="text",
                text=f"✗ Cannot connect to TTS server at {TTS_SERVER_URL}\n"
                     f"  Make sure the Docker container is running on desktop (10.111.79.180)\n"
                     f"  Try: ssh desktop 'cd ~/voice-tts-server && docker compose up -d'"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x mcp_voice_tts.py
```

### 6.4 Create Environment Configuration

```bash
cd ~/mcp-servers/voice-tts

cat > .env << 'EOF'
# TTS Server Configuration (Docker container on desktop)
TTS_SERVER_URL=http://10.111.79.180:8080
DEFAULT_VOICE=voice_reference.wav
DEFAULT_LANGUAGE=en
REQUEST_TIMEOUT=60
EOF
```

### 6.5 Configure Cursor MCP Integration

Edit Cursor's MCP configuration:

```bash
# Create/edit MCP config
mkdir -p ~/.cursor
cat > ~/.cursor/mcp.json << 'EOF'
{
  "mcpServers": {
    "voice-tts": {
      "command": "/Users/YOUR_USERNAME/mcp-servers/voice-tts/venv/bin/python",
      "args": ["/Users/YOUR_USERNAME/mcp-servers/voice-tts/mcp_voice_tts.py"],
      "env": {
        "TTS_SERVER_URL": "http://10.111.79.180:8080",
        "DEFAULT_VOICE": "voice_reference.wav",
        "DEFAULT_LANGUAGE": "en"
      }
    }
  }
}
EOF

# Replace YOUR_USERNAME with actual username
sed -i '' "s/YOUR_USERNAME/$USER/g" ~/.cursor/mcp.json

# Verify configuration
cat ~/.cursor/mcp.json
```

### 6.6 Test MCP Server Locally

```bash
cd ~/mcp-servers/voice-tts
source venv/bin/activate

# Test the server starts without errors
python mcp_voice_tts.py &
sleep 2
kill %1

echo "MCP server test complete"
```

---

## 7. Phase 5: Integration & Testing

### 7.1 Pre-flight Checklist

- [ ] **Linux Server (Docker)**
  - [ ] Docker container running (`docker compose ps`)
  - [ ] GPU accessible in container (`docker compose exec voice-tts nvidia-smi`)
  - [ ] Voice reference file in place (`ls ~/voice-tts-server/voice_references/`)
  - [ ] Health endpoint responding (`curl localhost:8080/health`)

- [ ] **Mac Client**
  - [ ] MCP server configured in `~/.cursor/mcp.json`
  - [ ] Network connectivity to 10.111.79.180:8080
  - [ ] Audio output working

### 7.2 End-to-End Test

```bash
# 1. Verify Docker container is running
ssh desktop "cd ~/voice-tts-server && docker compose ps"

# 2. Verify TTS server is accessible from Mac
curl http://10.111.79.180:8080/health | jq

# 3. Test full synthesis pipeline
curl -X POST http://10.111.79.180:8080/synthesize/raw \
  -H "Content-Type: application/json" \
  -d '{"text": "Integration test complete. The voice cloning system is running in Docker and working correctly."}' \
  -o ~/Desktop/integration_test.wav && afplay ~/Desktop/integration_test.wav

# 4. Restart Cursor to load MCP server
# Then in Cursor, try: "speak hello world"
```

### 7.3 Cursor Integration Test

After restarting Cursor:

1. Open Cursor chat
2. Type: `@voice-tts check the TTS server status`
3. Verify server status shows healthy with "Runtime: Docker"
4. Type: `@voice-tts speak "Hello, the voice cloning integration is working!"`
5. Verify audio plays through speakers

---

## 8. Maintenance & Operations

### 8.1 Container Management

```bash
# SSH to desktop first
ssh desktop
cd ~/voice-tts-server

# Check container status
docker compose ps

# View live logs
docker compose logs -f voice-tts

# View last 100 log lines
docker compose logs --tail=100 voice-tts

# Restart container
docker compose restart voice-tts

# Stop container
docker compose down

# Start container
docker compose up -d

# Rebuild and restart (after code changes)
docker compose up -d --build
```

### 8.2 Quick Commands from Mac

```bash
# Check container status
ssh desktop "cd ~/voice-tts-server && docker compose ps"

# View logs
ssh desktop "cd ~/voice-tts-server && docker compose logs --tail=50"

# Restart container
ssh desktop "cd ~/voice-tts-server && docker compose restart"

# Check health
curl http://10.111.79.180:8080/health | jq
```

### 8.3 Add New Voice References

```bash
# Upload new voice file from Mac
scp new_voice.wav desktop:~/voice-tts-server/voice_references/

# Preprocess using Docker container
ssh desktop
cd ~/voice-tts-server
docker compose run --rm voice-tts python preprocess_audio.py \
    /app/voice_references/new_voice.wav \
    -o /app/voice_references/new_voice_processed.wav

# Verify
docker compose exec voice-tts ls -la /app/voice_references/

# Use in Cursor: @voice-tts speak "Hello" using voice new_voice_processed.wav
```

### 8.4 Update Container

```bash
ssh desktop
cd ~/voice-tts-server

# Pull latest base image (if using newer CUDA version)
docker compose pull

# Rebuild with updated dependencies
docker compose build --no-cache

# Restart with new image
docker compose up -d
```

### 8.5 Backup Voice References

```bash
# Backup all voice references to Mac
mkdir -p ~/voice-tts-backups
scp -r desktop:~/voice-tts-server/voice_references/ ~/voice-tts-backups/$(date +%Y%m%d)/
```

### 8.6 Monitor Resource Usage

```bash
ssh desktop

# Container resource usage
docker stats voice-tts-server

# GPU usage inside container
docker compose exec voice-tts nvidia-smi

# Disk usage
docker system df
```

---

## 9. Troubleshooting Guide

### 9.1 Container Won't Start

**Symptom:** Container fails to start or exits immediately

```bash
ssh desktop
cd ~/voice-tts-server

# Check container status
docker compose ps -a

# View startup logs
docker compose logs voice-tts

# Check if GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Common fixes:**
- NVIDIA Container Toolkit not installed: `sudo apt install nvidia-container-toolkit`
- Docker not configured for GPU: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
- Port already in use: Check `lsof -i :8080` and kill conflicting process

### 9.2 GPU Not Available in Container

**Symptom:** Container starts but `gpu_available: false` in health check

```bash
# Verify host GPU works
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If above fails, reinstall NVIDIA Container Toolkit
sudo apt-get purge nvidia-container-toolkit
sudo apt-get install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 9.3 Cannot Connect to TTS Server

**Symptom:** Connection refused or timeout from Mac

```bash
# Verify container is running
ssh desktop "cd ~/voice-tts-server && docker compose ps"

# Check container can respond locally
ssh desktop "curl localhost:8080/health"

# Check firewall
ssh desktop "sudo ufw status"
ssh desktop "sudo ufw allow 8080/tcp"

# Test network connectivity
ping 10.111.79.180
nc -zv 10.111.79.180 8080
```

### 9.4 Model Download Fails

**Symptom:** Container hangs on first start while downloading model

```bash
ssh desktop
cd ~/voice-tts-server

# Check model cache volume
ls -la models/

# Manually trigger model download
docker compose run --rm voice-tts python -c "
from TTS.api import TTS
TTS('tts_models/multilingual/multi-dataset/xtts_v2')
print('Model downloaded successfully')
"

# Restart main container
docker compose up -d
```

### 9.5 Out of GPU Memory

**Symptom:** CUDA out of memory errors in logs

```bash
# Check GPU memory usage
ssh desktop "nvidia-smi"

# Kill other GPU processes if needed
ssh desktop "sudo fuser -v /dev/nvidia*"

# Restart container to clear GPU memory
ssh desktop "cd ~/voice-tts-server && docker compose restart"
```

### 9.6 Poor Audio Quality

**Symptom:** Synthesized speech sounds robotic or unclear

**Possible causes:**
1. Voice reference audio quality is poor
2. Reference audio too short (< 10 seconds)
3. Background noise in reference

**Fix:**
```bash
# Re-preprocess with aggressive settings
ssh desktop
cd ~/voice-tts-server
docker compose run --rm voice-tts python preprocess_audio.py \
    /app/voice_references/raw_reference.wav \
    -o /app/voice_references/voice_reference.wav
```

### 9.7 MCP Server Not Appearing in Cursor

**Symptom:** voice-tts tools not available in Cursor

```bash
# Verify MCP config
cat ~/.cursor/mcp.json | jq

# Check Python path is correct
ls -la ~/mcp-servers/voice-tts/venv/bin/python

# Test MCP server directly
cd ~/mcp-servers/voice-tts
source venv/bin/activate
python -c "import mcp; print('MCP installed correctly')"
```

**Fix:** Restart Cursor completely (Cmd+Q, then reopen)

### 9.8 Container Logs Full / Disk Space

```bash
ssh desktop

# Check Docker disk usage
docker system df

# Clean up unused images and containers
docker system prune -f

# Clean up unused volumes (careful - may delete model cache)
docker volume prune -f

# Rotate container logs manually
truncate -s 0 /var/lib/docker/containers/*/\*-json.log
```

---

## Appendix A: Complete File Listing

### Linux Server (desktop @ 10.111.79.180)

```
~/voice-tts-server/
├── Dockerfile                 # Container build instructions
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
├── tts_server.py             # FastAPI TTS server
├── preprocess_audio.py       # Audio preprocessing utility
├── voice_references/         # Voice reference audio (Docker volume)
│   └── voice_reference.wav   # Primary voice clone reference
├── audio_output/             # Generated audio files (Docker volume)
├── logs/                     # Server logs (Docker volume)
└── models/                   # Cached TTS model weights (Docker volume)
```

### Mac Client (Local)

```
~/mcp-servers/voice-tts/
├── venv/                      # Python virtual environment
├── requirements.txt           # Python dependencies
├── mcp_voice_tts.py          # MCP server for Cursor
└── .env                       # Environment configuration

~/.cursor/
└── mcp.json                   # Cursor MCP configuration
```

---

## Appendix B: API Reference

### TTS Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, GPU status, container info |
| `/voices` | GET | List available voice references |
| `/synthesize` | POST | Synthesize speech (base64 response) |
| `/synthesize/raw` | POST | Synthesize speech (raw WAV bytes) |

### MCP Tools

| Tool | Description |
|------|-------------|
| `speak` | Convert text to speech and play |
| `speak_summary` | Speak a summary aloud |
| `list_voices` | List available voice references |
| `check_tts_server` | Check TTS server status (shows Docker info) |

---

## Appendix C: Docker Commands Reference

| Task | Command |
|------|---------|
| Start container | `docker compose up -d` |
| Stop container | `docker compose down` |
| Restart container | `docker compose restart` |
| View logs | `docker compose logs -f voice-tts` |
| Rebuild image | `docker compose build --no-cache` |
| Shell into container | `docker compose exec voice-tts bash` |
| Run one-off command | `docker compose run --rm voice-tts <command>` |
| Check GPU in container | `docker compose exec voice-tts nvidia-smi` |
| Container stats | `docker stats voice-tts-server` |

---

## Related Documents

- [[voice-cloning-tts-mcp-server]] — Original design proposal
- [[2026-01-08]] — Implementation plan created
- [[2026-01-26]] — Updated to use Qwen3-TTS (SOTA open-source model)

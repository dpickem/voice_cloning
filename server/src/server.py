#!/usr/bin/env python3
"""
Voice Cloning TTS API Server (Multi-Model).

FastAPI server that provides text-to-speech synthesis with zero-shot
voice cloning. Supports multiple TTS backends:
- XTTS v2 (Coqui TTS) - default
- F5-TTS

Requires NVIDIA GPU with CUDA support.

Endpoints:
    POST /synthesize     - Convert text to speech (returns base64 JSON)
    POST /synthesize/raw - Convert text to speech (returns WAV bytes)
    GET  /health         - Health check with GPU and model status
    GET  /voices         - List available voice references
    GET  /models         - List available TTS models
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config import settings
from models import HealthResponse, ModelInfo, TTSRequest, TTSResponse, VoiceInfo
from tts_backends import ModelType, TTSBackend, get_available_backends, get_backend
from utils import (
    calculate_duration,
    ensure_numpy_array,
    list_voice_files,
    validate_voice_path,
    wav_to_base64,
    wav_to_bytes,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

PRINT_SEPARATOR_WIDTH = 60
BYTES_TO_GB = 1e9

# =============================================================================
# Global State
# =============================================================================

# Currently loaded backend (initialized on startup)
_current_backend: Optional[TTSBackend] = None

# Cache of loaded backends (for model switching)
_backend_cache: dict[str, TTSBackend] = {}


# =============================================================================
# Backend Management
# =============================================================================


def _get_or_load_backend(model_type: str) -> TTSBackend:
    """
    Get a backend, loading it if necessary.

    Uses caching to avoid reloading models when switching.

    Args:
        model_type: Model type string (e.g., 'xtts', 'f5-tts').

    Returns:
        Loaded backend instance.
    """
    global _backend_cache

    if model_type in _backend_cache:
        backend = _backend_cache[model_type]
        if backend.is_loaded:
            return backend

    # Create and load new backend
    backend = get_backend(model_type)
    backend.load()
    _backend_cache[model_type] = backend

    return backend


def _is_model_loaded() -> bool:
    """Check if any TTS model is loaded."""
    return _current_backend is not None and _current_backend.is_loaded


# =============================================================================
# Synthesis
# =============================================================================


async def _synthesize_audio(
    text: str,
    voice_path: str,
    language: str,
    model_type: Optional[str] = None,
    reference_text: Optional[str] = None,
) -> tuple[NDArray[np.floating], int]:
    """
    Run TTS synthesis in a thread pool to avoid blocking.

    Args:
        text: Text content to synthesize.
        voice_path: Path to the voice reference WAV file.
        language: Language code for synthesis.
        model_type: Model to use (None = default).
        reference_text: Reference audio transcript (for F5-TTS).

    Returns:
        Tuple of (waveform array, sample rate).
    """
    global _current_backend

    # Determine which backend to use
    if model_type and settings.ALLOW_MODEL_SWITCHING:
        backend = _get_or_load_backend(model_type)
    elif _current_backend is None:
        raise RuntimeError("No TTS model loaded")
    else:
        backend = _current_backend

    # Validate reference text requirement
    if backend.requires_reference_text and not reference_text:
        raise ValueError(
            f"{backend.display_name} requires reference_text (transcript of reference audio)"
        )

    # Run synthesis in thread pool
    loop = asyncio.get_event_loop()
    wav, sample_rate = await loop.run_in_executor(
        None,
        lambda: backend.synthesize(
            text=text,
            reference_audio=voice_path,
            language=language,
            reference_text=reference_text,
        ),
    )

    wav = ensure_numpy_array(wav)
    return wav, sample_rate


# =============================================================================
# Server Lifecycle
# =============================================================================


def _print_gpu_info() -> None:
    """Print GPU information to stdout."""
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / BYTES_TO_GB:.1f} GB")


def _validate_cuda() -> None:
    """Validate CUDA availability."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        print("This server requires an NVIDIA GPU with CUDA support.")
        raise SystemExit(1)


def _setup_directories() -> None:
    """Ensure required directories exist."""
    settings.VOICE_REFERENCES_DIR.mkdir(exist_ok=True, parents=True)
    settings.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def _print_voice_references() -> None:
    """Print available voice references."""
    voices = list_voice_files(settings.VOICE_REFERENCES_DIR)
    print(f"\nVoice references available: {len(voices)}")
    for v in voices:
        print(f"  - {v.filename}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle - load model on startup.

    Initializes the default TTS backend with GPU acceleration.
    """
    global _current_backend

    print("=" * PRINT_SEPARATOR_WIDTH)
    print("Voice Cloning TTS Server (Multi-Model)")
    print("=" * PRINT_SEPARATOR_WIDTH)

    _validate_cuda()
    _print_gpu_info()

    # Load default backend
    default_model = settings.TTS_BACKEND
    print(f"\nLoading default TTS backend: {default_model}")
    start_time = time.time()

    try:
        _current_backend = _get_or_load_backend(default_model)
        load_time = time.time() - start_time
        print(f"✓ {_current_backend.display_name} loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"✗ Failed to load {default_model}: {e}")
        raise SystemExit(1)

    _setup_directories()
    _print_voice_references()

    # Print available models
    print(f"\nAvailable models: {', '.join(m.value for m in ModelType)}")
    if settings.ALLOW_MODEL_SWITCHING:
        print("Model switching via API: enabled")
    else:
        print("Model switching via API: disabled")

    print("=" * PRINT_SEPARATOR_WIDTH)
    print(f"Server ready at {settings.server_url}")
    if settings.EXTERNAL_HOST != "localhost":
        print(f"Remote access: {settings.external_url}")
    print("=" * PRINT_SEPARATOR_WIDTH)

    yield

    # Cleanup
    print("Shutting down TTS server...")
    for backend in _backend_cache.values():
        backend.unload()
    _backend_cache.clear()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Voice Cloning TTS API",
    description="Multi-model text-to-speech synthesis with zero-shot voice cloning",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health, GPU status, and model status."""
    model_loaded = _is_model_loaded()
    gpu_available = torch.cuda.is_available()

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_type=_current_backend.model_type.value if _current_backend else None,
        model_name=_current_backend.display_name if _current_backend else None,
        default_model=settings.TTS_BACKEND,
        available_models=[m.value for m in ModelType],
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
        cuda_version=torch.version.cuda if gpu_available else None,
        container=True,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices() -> list[VoiceInfo]:
    """List available voice reference files."""
    return list_voice_files(settings.VOICE_REFERENCES_DIR)


@app.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List available TTS models and their status."""
    models = []
    for info in get_available_backends():
        is_loaded = info["model_type"] in _backend_cache and _backend_cache[info["model_type"]].is_loaded
        models.append(ModelInfo(
            model_type=info["model_type"],
            display_name=info.get("display_name", info["model_type"]),
            loaded=is_loaded,
            supports_languages=info.get("supports_languages", []),
            requires_reference_text=info.get("requires_reference_text", False),
            available=info.get("available", True),
            error=info.get("error"),
        ))
    return models


@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest) -> TTSResponse:
    """
    Synthesize speech from text.

    Returns JSON with base64-encoded audio.
    """
    if not _is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice_path = validate_voice_path(request.voice, settings.VOICE_REFERENCES_DIR)
    start_time = time.time()

    try:
        wav, sample_rate = await _synthesize_audio(
            text=request.text,
            voice_path=str(voice_path),
            language=request.language,
            model_type=request.model,
            reference_text=request.reference_text,
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Determine model type used
        model_used = request.model if request.model else settings.TTS_BACKEND

        return TTSResponse(
            success=True,
            audio_base64=wav_to_base64(wav, sample_rate),
            duration_seconds=calculate_duration(wav, sample_rate),
            sample_rate=sample_rate,
            processing_time_ms=processing_time_ms,
            text_length=len(request.text),
            model_type=model_used,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e!s}")


@app.post("/synthesize/raw")
async def synthesize_raw(request: TTSRequest) -> Response:
    """
    Synthesize speech from text.

    Returns raw WAV audio bytes.
    """
    if not _is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    voice_path = validate_voice_path(request.voice, settings.VOICE_REFERENCES_DIR)

    try:
        wav, sample_rate = await _synthesize_audio(
            text=request.text,
            voice_path=str(voice_path),
            language=request.language,
            model_type=request.model,
            reference_text=request.reference_text,
        )

        # Determine model type used
        model_used = request.model if request.model else settings.TTS_BACKEND

        return Response(
            content=wav_to_bytes(wav, sample_rate),
            media_type="audio/wav",
            headers={
                "X-Duration-Seconds": str(calculate_duration(wav, sample_rate)),
                "X-Sample-Rate": str(sample_rate),
                "X-Model-Type": model_used,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e!s}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
        log_level="info",
    )

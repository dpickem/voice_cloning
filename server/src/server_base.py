#!/usr/bin/env python3
"""
Shared base functionality for TTS servers.

Provides common validation, startup routines, and endpoint factories
used by both zero-shot and fine-tuned TTS server implementations.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Coroutine

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config import settings
from models import HealthResponse, TTSRequest, TTSResponse, VoiceInfo
from utils import (
    calculate_duration,
    list_voice_files,
    validate_voice_path,
    wav_to_base64,
    wav_to_bytes,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Type alias for synthesis function
SynthesizeFunc = Callable[
    [str, str, str], Coroutine[None, None, tuple["NDArray[np.floating]", int]]
]


def validate_cuda() -> None:
    """
    Validate that CUDA is available.

    Raises:
        SystemExit: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        print("This server requires an NVIDIA GPU with CUDA support.")
        print("Please ensure CUDA drivers are installed and a GPU is accessible.")
        raise SystemExit(1)


def print_gpu_info() -> None:
    """Print GPU information to stdout."""
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def setup_directories() -> None:
    """Ensure required directories exist."""
    settings.VOICE_REFERENCES_DIR.mkdir(exist_ok=True, parents=True)
    settings.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def print_voice_references() -> None:
    """Print available voice references to stdout."""
    voices = list_voice_files(settings.VOICE_REFERENCES_DIR)
    print(f"\nVoice references available: {len(voices)}")
    for v in voices:
        print(f"  - {v.filename}")


def print_server_ready() -> None:
    """Print server ready message."""
    print("=" * 60)
    print(f"Server ready at {settings.server_url}")
    print("=" * 60)


def create_app(title: str, description: str, version: str, lifespan: Callable) -> FastAPI:
    """
    Create a FastAPI application with standard middleware.

    Args:
        title: API title.
        description: API description.
        version: API version string.
        lifespan: Lifespan context manager for the app.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def create_health_check(
    model_loaded_check: Callable[[], bool],
    model_type: str,
) -> Callable[[], Coroutine[None, None, HealthResponse]]:
    """
    Create a health check endpoint handler.

    Args:
        model_loaded_check: Function that returns True if model is loaded.
        model_type: Model type string ('zero-shot' or 'fine-tuned').

    Returns:
        Async function for health check endpoint.
    """

    async def health_check() -> HealthResponse:
        """Check server health and model status."""
        model_loaded = model_loaded_check()
        gpu_available = torch.cuda.is_available()
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            model_type=model_type,
            gpu_available=gpu_available,
            gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
            cuda_version=torch.version.cuda if gpu_available else None,
            container=True,
            timestamp=datetime.utcnow().isoformat(),
        )

    return health_check


async def list_voices() -> list[VoiceInfo]:
    """
    List available voice reference files.

    Returns:
        List of VoiceInfo objects with filename and size for each
        WAV file in the voice references directory.
    """
    return list_voice_files(settings.VOICE_REFERENCES_DIR)


def create_synthesize_endpoint(
    model_loaded_check: Callable[[], bool],
    synthesize_func: SynthesizeFunc,
    model_type: str,
) -> Callable[[TTSRequest], Coroutine[None, None, TTSResponse]]:
    """
    Create a synthesize endpoint handler.

    Args:
        model_loaded_check: Function that returns True if model is loaded.
        synthesize_func: Async function to perform synthesis.
        model_type: Model type string for response.

    Returns:
        Async function for synthesize endpoint.
    """

    async def synthesize(request: TTSRequest) -> TTSResponse:
        """Synthesize speech from text."""
        if not model_loaded_check():
            raise HTTPException(status_code=503, detail="Model not loaded")

        voice_path = validate_voice_path(request.voice, settings.VOICE_REFERENCES_DIR)

        start_time = time.time()

        try:
            wav, sample_rate = await synthesize_func(
                request.text,
                str(voice_path),
                request.language,
            )

            processing_time_ms = (time.time() - start_time) * 1000

            return TTSResponse(
                success=True,
                audio_base64=wav_to_base64(wav, sample_rate),
                duration_seconds=calculate_duration(wav, sample_rate),
                sample_rate=sample_rate,
                processing_time_ms=processing_time_ms,
                text_length=len(request.text),
                model_type=model_type,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {e!s}")

    return synthesize


def create_synthesize_raw_endpoint(
    model_loaded_check: Callable[[], bool],
    synthesize_func: SynthesizeFunc,
    model_type: str,
) -> Callable[[TTSRequest], Coroutine[None, None, Response]]:
    """
    Create a synthesize/raw endpoint handler.

    Args:
        model_loaded_check: Function that returns True if model is loaded.
        synthesize_func: Async function to perform synthesis.
        model_type: Model type string for response headers.

    Returns:
        Async function for synthesize/raw endpoint.
    """

    async def synthesize_raw(request: TTSRequest) -> Response:
        """Synthesize speech and return raw WAV bytes."""
        if not model_loaded_check():
            raise HTTPException(status_code=503, detail="Model not loaded")

        voice_path = validate_voice_path(request.voice, settings.VOICE_REFERENCES_DIR)

        try:
            wav, sample_rate = await synthesize_func(
                request.text,
                str(voice_path),
                request.language,
            )

            return Response(
                content=wav_to_bytes(wav, sample_rate),
                media_type="audio/wav",
                headers={
                    "X-Duration-Seconds": str(calculate_duration(wav, sample_rate)),
                    "X-Sample-Rate": str(sample_rate),
                    "X-Model-Type": model_type,
                },
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {e!s}")

    return synthesize_raw

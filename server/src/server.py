#!/usr/bin/env python3
"""
Voice Cloning TTS API Server (Zero-Shot).

FastAPI server that provides text-to-speech synthesis using XTTS-v2
with zero-shot voice cloning. Requires NVIDIA GPU with CUDA support.

Endpoints:
    POST /synthesize     - Convert text to speech (returns base64 JSON)
    POST /synthesize/raw - Convert text to speech (returns WAV bytes)
    GET  /health         - Health check with GPU status
    GET  /voices         - List available voice references
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI
from TTS.api import TTS

from config import settings
from models import HealthResponse, TTSResponse, VoiceInfo
from server_base import (
    create_app,
    create_health_check,
    create_synthesize_endpoint,
    create_synthesize_raw_endpoint,
    list_voices,
    print_gpu_info,
    print_server_ready,
    print_voice_references,
    setup_directories,
    validate_cuda,
)
from utils import ensure_numpy_array

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Global TTS instance (initialized on startup)
_model: Optional[TTS] = None

MODEL_TYPE = "zero-shot"


def _is_model_loaded() -> bool:
    """Check if the TTS model is loaded."""
    return _model is not None


async def _synthesize_audio(
    text: str, voice_path: str, language: str
) -> tuple[NDArray[np.floating], int]:
    """
    Run TTS synthesis in a thread pool to avoid blocking.

    Args:
        text: Text content to synthesize.
        voice_path: Path to the voice reference WAV file.
        language: Language code for synthesis.

    Returns:
        Tuple of (waveform array, sample rate).

    Raises:
        RuntimeError: If the TTS model is not loaded.
    """
    if _model is None:
        raise RuntimeError("TTS model not loaded")

    loop = asyncio.get_event_loop()
    wav = await loop.run_in_executor(
        None,
        lambda: _model.tts(
            text=text,
            speaker_wav=voice_path,
            language=language,
        ),
    )

    wav = ensure_numpy_array(wav)
    sample_rate: int = _model.synthesizer.output_sample_rate

    return wav, sample_rate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle - load model on startup.

    Initializes the XTTS-v2 model with GPU acceleration.
    Requires CUDA to be available.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application after model initialization.

    Raises:
        SystemExit: If CUDA is not available.
    """
    global _model

    print("=" * 60)
    print("Voice Cloning TTS Server (Zero-Shot)")
    print("=" * 60)

    validate_cuda()
    print_gpu_info()

    print("\nLoading TTS model...")
    start_time = time.time()

    _model = TTS(settings.TTS_MODEL, gpu=True)

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")

    setup_directories()
    print_voice_references()
    print_server_ready()

    yield

    print("Shutting down TTS server...")


# Create the FastAPI application
app = create_app(
    title="Voice Cloning TTS API",
    description="Text-to-speech synthesis with zero-shot voice cloning using XTTS-v2",
    version="1.0.0",
    lifespan=lifespan,
)

# Register endpoints
app.get("/health", response_model=HealthResponse)(
    create_health_check(_is_model_loaded, MODEL_TYPE)
)

app.get("/voices", response_model=list[VoiceInfo])(list_voices)

app.post("/synthesize", response_model=TTSResponse)(
    create_synthesize_endpoint(_is_model_loaded, _synthesize_audio, MODEL_TYPE)
)

app.post("/synthesize/raw")(
    create_synthesize_raw_endpoint(_is_model_loaded, _synthesize_audio, MODEL_TYPE)
)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
        log_level="info",
    )

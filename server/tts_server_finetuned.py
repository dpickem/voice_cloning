#!/usr/bin/env python3
"""
TTS Server using fine-tuned XTTS-v2 model.

Requires a fine-tuned model to be present in FINETUNED_MODEL_DIR.
Use tts_server.py for zero-shot voice cloning instead.

Endpoints:
    POST /synthesize     - Convert text to speech (returns base64 JSON)
    POST /synthesize/raw - Convert text to speech (returns WAV bytes)
    GET  /health         - Health check with model status
    GET  /voices         - List available voice references
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from tts_config import settings
from tts_models import HealthResponse, TTSResponse, VoiceInfo
from tts_server_base import (
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
from tts_utils import ensure_numpy_array

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Global model instances
_model: Any = None
_model_config: Any = None

MODEL_TYPE = "fine-tuned"


def _is_model_loaded() -> bool:
    """Check if the TTS model is loaded."""
    return _model is not None


def _validate_finetuned_model() -> None:
    """
    Validate that a fine-tuned model is available.

    Raises:
        SystemExit: If the fine-tuned model directory or config.json is missing.
    """
    if not settings.FINETUNED_MODEL_DIR.exists():
        print(f"ERROR: Fine-tuned model directory not found: {settings.FINETUNED_MODEL_DIR}")
        print("Please train a model first or use tts_server.py for zero-shot cloning.")
        raise SystemExit(1)

    config_path = settings.FINETUNED_MODEL_DIR / "config.json"
    if not config_path.exists():
        print(f"ERROR: Model config not found: {config_path}")
        print("The fine-tuned model directory exists but is missing config.json.")
        raise SystemExit(1)


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
        RuntimeError: If no model is loaded.
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    loop = asyncio.get_event_loop()

    outputs: dict[str, Any] = await loop.run_in_executor(
        None,
        lambda: _model.synthesize(
            text=text,
            config=_model_config,
            speaker_wav=voice_path,
            language=language,
        ),
    )
    wav = ensure_numpy_array(outputs["wav"])
    sample_rate = settings.DEFAULT_SAMPLE_RATE

    return wav, sample_rate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle.

    Loads the fine-tuned TTS model on startup. Requires CUDA and
    a fine-tuned model to be available.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application after model initialization.

    Raises:
        SystemExit: If CUDA is not available or no fine-tuned model exists.
    """
    global _model, _model_config

    print("=" * 60)
    print("Voice Cloning TTS Server (Fine-Tuned)")
    print("=" * 60)

    validate_cuda()
    _validate_finetuned_model()
    print_gpu_info()

    start_time = time.time()

    print(f"\nLoading fine-tuned model from {settings.FINETUNED_MODEL_DIR}...")

    _model_config = XttsConfig()
    _model_config.load_json(str(settings.FINETUNED_MODEL_DIR / "config.json"))

    _model = Xtts.init_from_config(_model_config)
    _model.load_checkpoint(_model_config, checkpoint_dir=str(settings.FINETUNED_MODEL_DIR))
    _model.cuda()

    load_time = time.time() - start_time
    print(f"✓ Fine-tuned model loaded in {load_time:.2f}s")

    setup_directories()
    print_voice_references()
    print_server_ready()

    yield

    print("Shutting down TTS server...")


# Create the FastAPI application
app = create_app(
    title="Voice Cloning TTS API",
    description="Text-to-speech synthesis using fine-tuned XTTS-v2 model",
    version="2.0.0",
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
        "tts_server_finetuned:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        workers=1,
        log_level="info",
    )

#!/usr/bin/env python3
"""
Shared Pydantic models for TTS servers and training.

Provides request/response models used by both the zero-shot and fine-tuned
TTS server implementations, as well as training configuration models.
All models include full type annotations and validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config import settings


class ModelType(str, Enum):
    """Supported TTS model types."""

    XTTS = "xtts"
    F5_TTS = "f5-tts"
    CHATTERBOX = "chatterbox"
    OPENVOICE = "openvoice"


class TTSRequest(BaseModel):
    """
    Request model for text-to-speech synthesis.

    Attributes:
        text: The text content to synthesize into speech.
              Must be between 1 and 5000 characters.
        language: ISO language code for synthesis (e.g., 'en', 'es', 'fr').
                  Defaults to English.
        voice: Filename of the voice reference WAV file to use for cloning.
               Must exist in the voice references directory.
        model: TTS model backend to use (xtts, f5-tts, chatterbox, openvoice).
               If not specified, uses the server's default model.
        reference_text: Transcript of reference audio (required for F5-TTS,
                       optional for others).
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize (1-5000 characters)",
    )
    language: str = Field(
        default="en",
        description="Language code (e.g., 'en', 'es', 'fr', 'de', 'it', 'pt', 'pl')",
    )
    voice: str = Field(
        default=settings.DEFAULT_VOICE,
        description="Voice reference filename (must be a WAV file)",
    )
    model: Optional[str] = Field(
        default=None,
        description="TTS model to use: xtts, f5-tts, chatterbox, openvoice (default: server config)",
    )
    reference_text: Optional[str] = Field(
        default=None,
        description="Transcript of reference audio (required for F5-TTS)",
    )


class TTSResponse(BaseModel):
    """
    Response model for synthesis results with base64-encoded audio.

    Attributes:
        success: Whether the synthesis completed successfully.
        audio_base64: Base64-encoded WAV audio data, or None on failure.
        duration_seconds: Duration of the generated audio in seconds.
        sample_rate: Sample rate of the generated audio in Hz.
        processing_time_ms: Time taken to synthesize the audio in milliseconds.
        text_length: Character count of the input text.
        model_type: Type of model used ('zero-shot' or 'fine-tuned').
                    Optional for backward compatibility.
    """

    success: bool
    audio_base64: Optional[str] = None
    duration_seconds: float
    sample_rate: int
    processing_time_ms: float
    text_length: int
    model_type: Optional[str] = None


class HealthResponse(BaseModel):
    """
    Health check response with server and GPU status.

    Attributes:
        status: Current server health status ('healthy' or 'unhealthy').
        model_loaded: Whether the TTS model is loaded and ready.
        model_type: Type of loaded model (e.g., 'xtts', 'f5-tts').
        model_name: Display name of the loaded model.
        default_model: The server's default model type.
        available_models: List of available model types.
        gpu_available: Whether CUDA GPU acceleration is available.
        gpu_name: Name of the GPU device, or None if unavailable.
        cuda_version: CUDA toolkit version, or None if unavailable.
        container: Whether the server is running in a container.
        timestamp: ISO 8601 timestamp of the health check.
    """

    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    default_model: Optional[str] = None
    available_models: Optional[list[str]] = None
    gpu_available: bool
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    container: bool
    timestamp: str


class ModelInfo(BaseModel):
    """
    Information about a TTS model backend.

    Attributes:
        model_type: Model identifier (e.g., 'xtts', 'f5-tts').
        display_name: Human-readable model name.
        loaded: Whether this model is currently loaded.
        supports_languages: List of supported language codes.
        requires_reference_text: Whether reference transcript is needed.
        available: Whether the model dependencies are installed.
    """

    model_type: str
    display_name: str
    loaded: bool = False
    supports_languages: list[str] = []
    requires_reference_text: bool = False
    available: bool = True
    error: Optional[str] = None


class VoiceInfo(BaseModel):
    """
    Voice reference file information.

    Attributes:
        filename: Name of the voice reference WAV file.
        size_bytes: File size in bytes.
    """

    filename: str
    size_bytes: int


# =============================================================================
# Training/Fine-tuning Models
# =============================================================================


class Sample(BaseModel):
    """A single training or evaluation sample."""

    audio_file: str
    text: str
    speaker_name: str
    language: str


class DataConfig(BaseModel):
    """Dataset paths and language configuration."""

    audio_dir: str
    metadata_csv: str
    language: str
    eval_split_ratio: float = 0.1
    random_seed: int = 42


class TrainingConfig(BaseModel):
    """Training hyperparameters for fine-tuning."""

    batch_size: int
    eval_batch_size: int
    num_epochs: int
    learning_rate: float


class FinetuneConfig(BaseModel):
    """Top-level configuration for XTTS fine-tuning."""

    data: DataConfig
    training: TrainingConfig
    output_path: str

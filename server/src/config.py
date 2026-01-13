#!/usr/bin/env python3
"""
TTS Server Configuration

This module provides type-safe configuration loading using Pydantic settings.
Environment variables are loaded from .env file and validated.

CONFIGURATION HIERARCHY:
    1. Environment variables (highest priority)
    2. .env file in config/ directory
    3. Defaults defined in this file (lowest priority)

WHAT GOES WHERE:
    - .env / Environment variables: Deployment-specific paths, model settings
    - This file (config.py): Environment variable definitions with types and defaults

Usage:
    from config import settings

    # Access settings via the settings object
    voice_dir = settings.VOICE_REFERENCES_DIR
    output_dir = settings.AUDIO_OUTPUT_DIR
    port = settings.PORT
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Directory paths for default resolution
_SRC_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _SRC_DIR.parent
_PROJECT_ROOT = _SERVER_DIR.parent


class TTSSettings(BaseSettings):
    """
    TTS Server settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    # =========================================================================
    # DIRECTORY PATHS
    # =========================================================================
    # Directory containing voice reference WAV files for voice cloning
    VOICE_REFERENCES_DIR: Path = _PROJECT_ROOT / "data" / "voice_references"

    # Directory for saving generated audio files
    AUDIO_OUTPUT_DIR: Path = _SERVER_DIR / "audio_output"

    # Directory containing fine-tuned model files
    FINETUNED_MODEL_DIR: Path = _SERVER_DIR / "finetuned_model"

    # Directory containing training data
    TRAINING_DATA_DIR: Path = _PROJECT_ROOT / "data" / "training"

    @field_validator("VOICE_REFERENCES_DIR", "AUDIO_OUTPUT_DIR", "FINETUNED_MODEL_DIR", "TRAINING_DATA_DIR", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects and expand user (~)."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v.expanduser()

    # =========================================================================
    # VOICE SETTINGS
    # =========================================================================
    # Default voice reference filename (should exist in VOICE_REFERENCES_DIR)
    DEFAULT_VOICE: str = "voice_reference.wav"

    @property
    def default_voice_path(self) -> Path:
        """Full path to the default voice reference file."""
        return self.VOICE_REFERENCES_DIR / self.DEFAULT_VOICE

    # =========================================================================
    # SERVER SETTINGS
    # =========================================================================
    # Server bind address (0.0.0.0 for all interfaces, 127.0.0.1 for localhost only)
    HOST: str = "0.0.0.0"

    # Server port number
    PORT: int = 8080

    # External hostname/IP for remote access (used in logs and URLs)
    # Set this to your server's IP address for remote access
    EXTERNAL_HOST: str = "localhost"

    @property
    def server_url(self) -> str:
        """Constructed server URL for local access."""
        return f"http://{self.HOST}:{self.PORT}"

    @property
    def external_url(self) -> str:
        """Constructed server URL for remote access."""
        return f"http://{self.EXTERNAL_HOST}:{self.PORT}"

    # =========================================================================
    # MODEL SETTINGS
    # =========================================================================
    # TTS model backend to use: xtts, f5-tts, chatterbox, openvoice
    TTS_BACKEND: str = "xtts"

    # Model identifier for XTTS (only used when TTS_BACKEND=xtts)
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Default audio sample rate in Hz
    DEFAULT_SAMPLE_RATE: int = 22050

    # Whether to allow model switching via API (if False, only TTS_BACKEND is used)
    ALLOW_MODEL_SWITCHING: bool = True

    # =========================================================================
    # PYDANTIC SETTINGS CONFIG
    # =========================================================================
    model_config = {
        "env_file": str(_SERVER_DIR / "config" / ".env"),
        "env_file_encoding": "utf-8",
        # Allow extra fields to be ignored (forward compatibility)
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> TTSSettings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once and reused.
    """
    return TTSSettings()


def clear_settings_cache() -> None:
    """
    Clear the settings cache.

    Useful for testing when environment variables change and settings
    need to be reloaded. After calling this, the next call to get_settings()
    will create a new TTSSettings instance.
    """
    get_settings.cache_clear()


# Clear cache on module reload to support testing with different env vars
# This ensures that if the module is reloaded (e.g., importlib.reload),
# fresh settings are loaded from the current environment
clear_settings_cache()

# Primary settings object
settings = get_settings()

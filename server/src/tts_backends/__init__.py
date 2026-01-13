#!/usr/bin/env python3
"""
TTS Backend implementations for voice cloning models.

Currently available:
- XTTS v2 (Coqui TTS)

Usage:
    from tts_backends import get_backend, TTSBackend, ModelType

    backend = get_backend(ModelType.XTTS)
    backend.load()
    wav, sr = backend.synthesize("Hello world", "reference.wav", "en")
"""

from tts_backends.base import TTSBackend, ModelType
from tts_backends.registry import get_backend, get_available_backends

__all__ = [
    "TTSBackend",
    "ModelType",
    "get_backend",
    "get_available_backends",
]

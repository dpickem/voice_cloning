#!/usr/bin/env python3
"""
Base class and types for TTS backends.

Defines the interface that all TTS model backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class ModelType(str, Enum):
    """Supported TTS model types."""

    XTTS = "xtts"
    F5_TTS = "f5-tts"
    CHATTERBOX = "chatterbox"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """Convert string to ModelType, case-insensitive."""
        value_lower = value.lower().replace("_", "-")
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown model type: {value}. Available: {[m.value for m in cls]}")


class TTSBackend(ABC):
    """
    Abstract base class for TTS backends.

    All TTS model implementations must inherit from this class and
    implement the required methods.
    """

    # Class-level metadata (override in subclasses)
    model_type: ModelType
    display_name: str = "Unknown TTS"
    supports_languages: list[str] = ["en"]
    requires_reference_text: bool = False  # F5-TTS needs reference transcript

    def __init__(self) -> None:
        """Initialize the backend (model not loaded yet)."""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """
        Load the TTS model into memory.

        Should set self._loaded = True on success.

        Raises:
            RuntimeError: If model loading fails.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload the model from memory.

        Should set self._loaded = False.
        """
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        reference_text: str | None = None,
    ) -> tuple[NDArray[np.floating], int]:
        """
        Synthesize speech from text using voice cloning.

        Args:
            text: Text to synthesize into speech.
            reference_audio: Path to reference audio file for voice cloning.
            language: Language code (e.g., 'en', 'es').
            reference_text: Transcript of reference audio (required by some models).

        Returns:
            Tuple of (waveform as numpy array, sample rate in Hz).

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If required parameters are missing.
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate of the model in Hz."""
        pass

    def get_info(self) -> dict:
        """Get backend information for health checks."""
        return {
            "model_type": self.model_type.value,
            "display_name": self.display_name,
            "loaded": self.is_loaded,
            "supports_languages": self.supports_languages,
            "requires_reference_text": self.requires_reference_text,
            "sample_rate": self.sample_rate if self.is_loaded else None,
        }

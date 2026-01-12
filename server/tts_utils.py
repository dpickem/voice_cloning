#!/usr/bin/env python3
"""
Shared utility functions for TTS servers.

Provides common functionality for audio processing, voice validation,
and response building used by both TTS server implementations.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from tts_models import VoiceInfo

if TYPE_CHECKING:
    from numpy.typing import NDArray


def wav_to_bytes(wav: NDArray[np.floating], sample_rate: int) -> bytes:
    """
    Convert a waveform numpy array to WAV file bytes.

    Args:
        wav: Audio waveform as a numpy array of floating-point samples.
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Raw bytes of a valid WAV file with 16-bit PCM encoding.
    """
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def wav_to_base64(wav: NDArray[np.floating], sample_rate: int) -> str:
    """
    Convert a waveform numpy array to a base64-encoded WAV string.

    Args:
        wav: Audio waveform as a numpy array of floating-point samples.
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Base64-encoded string of the WAV file data.
    """
    audio_bytes = wav_to_bytes(wav, sample_rate)
    return base64.b64encode(audio_bytes).decode("utf-8")


def validate_voice_path(voice: str, voices_dir: Path) -> Path:
    """
    Validate that a voice reference file exists.

    Args:
        voice: Filename of the voice reference (e.g., 'my_voice.wav').
        voices_dir: Directory containing voice reference files.

    Returns:
        Full Path to the validated voice file.

    Raises:
        HTTPException: 404 error if the voice file does not exist,
                       including a list of available voices in the error detail.
    """
    voice_path = voices_dir / voice
    if not voice_path.exists():
        available = [f.name for f in voices_dir.glob("*.wav")]
        raise HTTPException(
            status_code=404,
            detail=f"Voice reference '{voice}' not found. Available: {available}",
        )
    return voice_path


def list_voice_files(voices_dir: Path) -> list[VoiceInfo]:
    """
    List all available voice reference files in a directory.

    Args:
        voices_dir: Directory to scan for WAV files.

    Returns:
        List of VoiceInfo objects containing filename and size
        for each WAV file found.
    """
    voices: list[VoiceInfo] = []
    for file in voices_dir.glob("*.wav"):
        voices.append(VoiceInfo(filename=file.name, size_bytes=file.stat().st_size))
    return voices


def ensure_numpy_array(wav: NDArray[np.floating] | list[float]) -> NDArray[np.floating]:
    """
    Ensure the waveform is a numpy array.

    Some TTS model outputs may return a list instead of a numpy array.
    This function normalizes the output to always be a numpy array.

    Args:
        wav: Audio waveform as either a numpy array or Python list.

    Returns:
        Audio waveform as a numpy array.
    """
    if not isinstance(wav, np.ndarray):
        return np.array(wav)
    return wav


def calculate_duration(wav: NDArray[np.floating], sample_rate: int) -> float:
    """
    Calculate the duration of an audio waveform in seconds.

    Args:
        wav: Audio waveform as a numpy array.
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Duration of the audio in seconds.
    """
    return len(wav) / sample_rate

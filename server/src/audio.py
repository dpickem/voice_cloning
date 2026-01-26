#!/usr/bin/env python3
"""
Shared audio processing utilities.

Provides a single processing pipeline used by both single-file and
batch preprocessing scripts to keep behavior consistent.
"""

from __future__ import annotations

from typing import Final, Tuple

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

# Type aliases
AudioArray = NDArray[np.float32]

# Defaults
# 24000 Hz is optimal for Qwen3-TTS (SOTA) and also works well with XTTS-v2
DEFAULT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_TRIM_TOP_DB: Final[int] = 25
DEFAULT_NOISE_REDUCTION: Final[float] = 0.8


def process_audio(
    input_file: str,
    output_file: str,
    *,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True,
    denoise: bool = True,
    trim_top_db: int = DEFAULT_TRIM_TOP_DB,
    noise_reduction: float = DEFAULT_NOISE_REDUCTION,
) -> Tuple[str, float]:
    """
    Run the audio preprocessing pipeline and write the output WAV.

    Args:
        input_file: Path to the input audio file.
        output_file: Path to write the processed WAV.
        target_sr: Target sample rate for resampling.
        normalize: Whether to normalize volume.
        denoise: Whether to apply noise reduction.
        trim_top_db: dB threshold for silence trimming.
        noise_reduction: Noise reduction aggressiveness (prop_decrease).

    Returns:
        Tuple of (output_file, duration_seconds).
    """
    print(f"Loading: {input_file}")

    # Load audio as mono
    raw_audio: NDArray[np.floating]
    sr: int
    raw_audio, sr = librosa.load(input_file, sr=None, mono=True)
    audio: AudioArray = np.asarray(raw_audio, dtype=np.float32)
    print(f"Original: {sr}Hz, {len(audio)/sr:.2f}s, {len(audio)} samples")

    # Resample if needed
    if sr != target_sr:
        print(f"Resampling: {sr}Hz -> {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Apply noise reduction
    if denoise:
        print("Applying noise reduction...")
        audio = nr.reduce_noise(y=audio, sr=target_sr, prop_decrease=noise_reduction)

    # Trim silence from beginning and end
    print("Trimming silence...")
    audio, _ = librosa.effects.trim(audio, top_db=trim_top_db)

    # Normalize volume
    if normalize:
        print("Normalizing volume...")
        audio = librosa.util.normalize(audio)

    # Save processed audio
    sf.write(output_file, audio, target_sr, subtype="PCM_16")

    duration_seconds: float = len(audio) / target_sr
    print(f"Saved: {output_file}")
    print(f"Final: {target_sr}Hz, {duration_seconds:.2f}s, {len(audio)} samples")

    return output_file, duration_seconds

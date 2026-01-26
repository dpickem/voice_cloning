#!/usr/bin/env python3
"""
Audio preprocessing script for voice cloning reference files.

Normalizes, denoises, and resamples audio to optimal format for
Qwen3-TTS (SOTA) and XTTS-v2 voice cloning. Can be run inside the
Docker container or standalone with dependencies installed.

Usage (from server/ directory):
    python scripts/preprocess_audio.py input.wav -o output.wav
    python scripts/preprocess_audio.py voice_references/raw.mp3 -o voice_references/processed.wav
    python scripts/preprocess_audio.py input.mp3 --sr 24000 --no-denoise

File structure:
    server/
    ├── scripts/
    │   └── preprocess_audio.py     # This script
    ├── src/
    │   └── audio.py                # Audio processing utilities
    └── voice_references/           # Store processed reference files here
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio import (
    DEFAULT_NOISE_REDUCTION,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TRIM_TOP_DB,
    process_audio,
)


def preprocess_audio(
    input_file: str,
    output_file: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True,
    denoise: bool = True,
) -> str:
    """
    Preprocess audio file for voice cloning.

    Applies a pipeline of audio processing steps to prepare
    voice samples for use with XTTS-v2:
    - Resampling to target sample rate
    - Noise reduction using spectral gating
    - Silence trimming from start and end
    - Volume normalization

    Args:
        input_file: Path to input audio file (WAV, MP3, etc.).
        output_file: Path to save processed audio (WAV format).
        target_sr: Target sample rate in Hz (22050 for XTTS-v2).
        normalize: Whether to normalize volume to peak amplitude.
        denoise: Whether to apply spectral noise reduction.

    Returns:
        Path to the processed audio file.
    """
    _, duration = process_audio(
        input_file=input_file,
        output_file=output_file,
        target_sr=target_sr,
        normalize=normalize,
        denoise=denoise,
        trim_top_db=DEFAULT_TRIM_TOP_DB,
        noise_reduction=DEFAULT_NOISE_REDUCTION,
    )
    return output_file


def main() -> None:
    """
    Command-line interface for audio preprocessing.

    Parses arguments and runs preprocessing on the specified input file.
    """
    parser = argparse.ArgumentParser(description="Preprocess audio for voice cloning")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument(
        "-o", "--output", help="Output file (default: input_processed.wav)"
    )
    parser.add_argument("--sr", type=int, default=24000, help="Target sample rate (24000 for Qwen3-TTS)")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip normalization"
    )
    parser.add_argument(
        "--no-denoise", action="store_true", help="Skip noise reduction"
    )

    args = parser.parse_args()

    if not args.output:
        base, _ = os.path.splitext(args.input)
        args.output = f"{base}_processed.wav"

    preprocess_audio(
        args.input,
        args.output,
        target_sr=args.sr,
        normalize=not args.no_normalize,
        denoise=not args.no_denoise,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch preprocess training audio files for fine-tuning.

Processes all audio files referenced in a metadata CSV, applying
normalization, denoising, and resampling to ensure consistent
format across the training dataset.

Usage:
    python preprocess_training_data.py
    python preprocess_training_data.py --input-dir wavs --output-dir processed --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeAlias

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio import (
    DEFAULT_NOISE_REDUCTION,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TRIM_TOP_DB,
    process_audio,
)

# Type alias for preprocessing task arguments
PreprocessArgs: TypeAlias = tuple[str, str, int]

# Type alias for preprocessing result
PreprocessResult: TypeAlias = tuple[str, str | None, float, str | None]


def preprocess_single(args: PreprocessArgs) -> PreprocessResult:
    """
    Preprocess a single audio file.

    Applies the standard preprocessing pipeline: resampling, noise
    reduction, silence trimming, and volume normalization.

    Args:
        args: Tuple of (input_file, output_file, target_sr).

    Returns:
        Tuple of (input_file, output_file, duration, error) where
        error is None on success or an error message on failure.
    """
    input_file, output_file, target_sr = args

    try:
        _, duration = process_audio(
            input_file=input_file,
            output_file=output_file,
            target_sr=target_sr,
            normalize=True,
            denoise=True,
            trim_top_db=DEFAULT_TRIM_TOP_DB,
            noise_reduction=DEFAULT_NOISE_REDUCTION,
        )
        return (input_file, output_file, duration, None)

    except Exception as e:
        return (input_file, None, 0.0, str(e))


def validate_metadata(
    metadata_file: str | Path,
    wavs_dir: str | Path
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Validate metadata.csv against audio files.

    Checks that each entry in the metadata file has a corresponding
    audio file and a valid transcription.

    Args:
        metadata_file: Path to the metadata CSV file.
        wavs_dir: Directory containing the audio files.

    Returns:
        Tuple of (valid_entries, errors) where valid_entries is a
        list of (filename, transcription) tuples and errors is a
        list of error messages for invalid entries.
    """
    errors: list[str] = []
    valid_entries: list[tuple[str, str]] = []

    with open(metadata_file, "r") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) != 2:
                errors.append(f"Invalid row format: {row}")
                continue

            filename, transcription = row
            wav_path = Path(wavs_dir) / f"{filename}.wav"

            if not wav_path.exists():
                errors.append(f"Audio file not found: {wav_path}")
                continue

            if len(transcription.strip()) < 3:
                errors.append(f"Transcription too short for {filename}")
                continue

            valid_entries.append((filename, transcription))

    return valid_entries, errors


def main() -> None:
    """
    Main entry point for batch audio preprocessing.

    Validates metadata, processes audio files in parallel, and
    reports results.
    """
    parser = argparse.ArgumentParser(description="Preprocess training audio")
    parser.add_argument("--input-dir", default="training_data/wavs", help="Input directory")
    parser.add_argument("--output-dir", default="training_data/processed", help="Output directory")
    parser.add_argument("--metadata", default="training_data/metadata.csv", help="Metadata file")
    parser.add_argument("--sr", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate metadata
    print(f"Validating metadata: {args.metadata}")
    valid_entries, errors = validate_metadata(args.metadata, input_dir)

    if errors:
        print(f"\n⚠️  Found {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\n✓ Valid entries: {len(valid_entries)}")

    # Prepare processing tasks
    tasks: list[PreprocessArgs] = []
    for filename, _ in valid_entries:
        input_file = input_dir / f"{filename}.wav"
        output_file = output_dir / f"{filename}.wav"
        tasks.append((str(input_file), str(output_file), args.sr))

    # Process in parallel
    print(f"\nProcessing {len(tasks)} files with {args.workers} workers...")

    total_duration: float = 0.0
    processed: int = 0
    failed: int = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(preprocess_single, tasks):
            input_file, output_file, duration, error = result
            if error:
                print(f"  ✗ {Path(input_file).name}: {error}")
                failed += 1
            else:
                total_duration += duration
                processed += 1
                if processed % 50 == 0:
                    print(f"  Processed {processed}/{len(tasks)}...")

    print(f"\n{'='*50}")
    print("Preprocessing complete!")
    print(f"  Processed: {processed} files")
    print(f"  Failed: {failed} files")
    print(f"  Total duration: {total_duration/60:.1f} minutes")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

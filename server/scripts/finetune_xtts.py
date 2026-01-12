#!/usr/bin/env python3
"""
Fine-tune XTTS-v2 on custom voice dataset.

This script handles the complete fine-tuning process:
1. Load pre-trained XTTS-v2 model
2. Split and prepare dataset from metadata CSV
3. Run training loop
4. Save fine-tuned model

Usage (from server/ directory):
    python scripts/finetune_xtts.py --config config/finetune_config.json
    python scripts/finetune_xtts.py --config config/finetune_config.json --resume /path/to/checkpoint

File structure:
    server/
    ├── config/
    │   └── finetune_config.json    # Training configuration
    ├── scripts/
    │   └── finetune_xtts.py        # This script
    ├── training_data/
    │   ├── metadata.csv            # Audio file list (filename|text)
    │   └── wavs/                   # Audio files referenced in metadata.csv
    └── finetuned_model/            # Output directory for trained model
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Set environment variables before importing TTS
os.environ["COQUI_TOS_AGREED"] = "1"

from models import FinetuneConfig, Sample

# Type alias for metadata entries: (filename, text, audio_path)
MetadataEntry = tuple[str, str, Path]

# Default speaker name for fine-tuned voice
DEFAULT_SPEAKER_NAME = "custom_voice"


# =============================================================================
# Configuration
# =============================================================================


def load_config(config_path: str) -> FinetuneConfig:
    """
    Load fine-tuning configuration from JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        FinetuneConfig instance with validated data paths, training
        hyperparameters, and output settings.
    """
    with open(config_path, "r") as f:
        config_dict: dict[str, Any] = json.load(f)
    return FinetuneConfig.model_validate(config_dict)


# =============================================================================
# Dataset Preparation
# =============================================================================


def load_valid_entries(
    metadata_csv: Path | str,
    audio_dir: Path | str,
) -> tuple[list[MetadataEntry], int]:
    """
    Load metadata CSV and filter to entries with existing audio files.

    Reads a pipe-delimited CSV file with 'filename|text' format and returns
    only entries where the corresponding .wav file exists in audio_dir.

    Args:
        metadata_csv: Path to the metadata CSV file.
        audio_dir: Directory containing the audio .wav files.

    Returns:
        Tuple of (valid_entries, skipped_count) where valid_entries is a list
        of (filename, text, audio_path) tuples.
    """
    audio_dir = Path(audio_dir)
    entries: list[MetadataEntry] = []
    skipped = 0

    with open(metadata_csv, "r") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) != 2:
                continue
            filename, text = row
            audio_path = audio_dir / f"{filename}.wav"
            if audio_path.exists():
                entries.append((filename, text, audio_path))
            else:
                skipped += 1

    return entries, skipped


def split_entries(
    entries: list[MetadataEntry],
    eval_ratio: float,
    seed: int,
) -> tuple[list[MetadataEntry], list[MetadataEntry]]:
    """
    Split entries into training and evaluation sets.

    Uses PyTorch's random_split for reproducible splitting. Ensures at least
    one sample in the eval set when there are 2+ entries total.

    Args:
        entries: List of (filename, text, audio_path) tuples.
        eval_ratio: Fraction of data for evaluation (0.0-1.0).
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_entries, eval_entries).
    """
    # Import torch here to allow testing data loading functions without torch
    import torch
    from torch.utils.data import random_split

    total = len(entries)
    eval_size = int(total * eval_ratio)
    train_size = total - eval_size

    # Ensure at least 1 eval sample if we have enough data
    if eval_size == 0 and total > 1:
        eval_size = 1
        train_size = total - 1

    generator = torch.Generator().manual_seed(seed)
    train_subset, eval_subset = random_split(
        entries, [train_size, eval_size], generator=generator
    )

    train_entries = [entries[i] for i in train_subset.indices]
    eval_entries = [entries[i] for i in eval_subset.indices]

    return train_entries, eval_entries


def save_split_csvs(
    metadata_csv: Path | str,
    train_entries: list[MetadataEntry],
    eval_entries: list[MetadataEntry],
) -> tuple[Path, Path]:
    """
    Save train and eval entries to separate CSV files.

    Creates files named {metadata_csv}_train.csv and {metadata_csv}_eval.csv
    in the same directory as the source metadata file.

    Args:
        metadata_csv: Path to the original metadata CSV (used to derive output paths).
        train_entries: Training set entries.
        eval_entries: Evaluation set entries.

    Returns:
        Tuple of (train_csv_path, eval_csv_path).
    """
    base_path = str(metadata_csv).replace(".csv", "")
    train_csv = Path(f"{base_path}_train.csv")
    eval_csv = Path(f"{base_path}_eval.csv")

    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows((fn, txt) for fn, txt, _ in train_entries)

    with open(eval_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows((fn, txt) for fn, txt, _ in eval_entries)

    return train_csv, eval_csv


def entries_to_samples(
    entries: list[MetadataEntry],
    language: str,
    speaker_name: str = DEFAULT_SPEAKER_NAME,
) -> list[dict[str, Any]]:
    """
    Convert metadata entries to Sample dictionaries for training.

    Args:
        entries: List of (filename, text, audio_path) tuples.
        language: Language code (e.g., 'en').
        speaker_name: Speaker identifier for the samples.

    Returns:
        List of Sample model dictionaries.
    """
    return [
        Sample(
            audio_file=str(audio_path),
            text=text,
            speaker_name=speaker_name,
            language=language,
        ).model_dump()
        for _, text, audio_path in entries
    ]


def prepare_dataset(
    config: FinetuneConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Prepare training and evaluation datasets from metadata CSV.

    Loads valid entries, splits them into train/eval sets, saves the split
    CSV files for reference, and converts to Sample dictionaries.

    Args:
        config: Fine-tuning configuration containing data paths.

    Returns:
        Tuple of (train_samples, eval_samples) for the Trainer.

    Raises:
        ValueError: If no valid audio files are found.
    """
    # Load and filter entries
    entries, skipped = load_valid_entries(
        config.data.metadata_csv, config.data.audio_dir
    )

    if skipped > 0:
        print(f"Note: Skipped {skipped} entries without audio files")

    if not entries:
        raise ValueError(f"No valid audio files found in {config.data.audio_dir}")

    # Split into train/eval
    train_entries, eval_entries = split_entries(
        entries, config.data.eval_split_ratio, config.data.random_seed
    )

    # Save split CSVs for reference
    train_csv, eval_csv = save_split_csvs(
        config.data.metadata_csv, train_entries, eval_entries
    )

    print(f"Dataset prepared (seed={config.data.random_seed}, "
          f"eval_ratio={config.data.eval_split_ratio}):")
    print(f"  Entries with audio: {len(entries)}")
    print(f"  Training: {len(train_entries)} -> {train_csv}")
    print(f"  Evaluation: {len(eval_entries)} -> {eval_csv}")

    # Convert to Sample dictionaries
    train_samples = entries_to_samples(train_entries, config.data.language)
    eval_samples = entries_to_samples(eval_entries, config.data.language)

    return train_samples, eval_samples


# =============================================================================
# Model Loading
# =============================================================================


def print_gpu_info() -> None:
    """Print GPU availability and specifications."""
    import torch

    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")


def get_model_directory() -> Path:
    """
    Get the XTTS-v2 model cache directory.

    Checks locations in order:
    1. TTS_HOME environment variable (Docker: /app/models)
    2. Default TTS cache (~/.local/share/tts)

    Returns:
        Path to the model directory.
    """
    model_subdir = "tts_models--multilingual--multi-dataset--xtts_v2"

    # Check TTS_HOME first (used in Docker)
    tts_home = os.getenv("TTS_HOME")
    if tts_home:
        model_dir = Path(tts_home) / model_subdir
        if (model_dir / "config.json").exists():
            return model_dir

    # Check default TTS cache location (local development)
    default_cache = Path.home() / ".local" / "share" / "tts" / model_subdir
    if (default_cache / "config.json").exists():
        return default_cache

    # Return TTS_HOME path if set, otherwise default cache (for downloading)
    if tts_home:
        return Path(tts_home) / model_subdir
    return default_cache


def ensure_model_downloaded(model_dir: Path) -> Path:
    """
    Download the XTTS-v2 model if not already cached.

    Args:
        model_dir: Expected model cache directory.

    Returns:
        Path to the actual model directory (may differ if TTS downloads elsewhere).
    """
    model_config_path = model_dir / "config.json"
    if not model_config_path.exists():
        print(f"Model config not found at {model_config_path}")
        print("Downloading model via TTS...")
        # Import here to avoid loading TTS unless needed
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        # Get actual model path from TTS manager
        actual_path = Path(tts.synthesizer.tts_model.config.model_dir)
        print(f"Model downloaded to: {actual_path}")
        return actual_path
    return model_dir


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Main entry point for XTTS-v2 fine-tuning.

    Parses command-line arguments, loads configuration, initializes
    the model, prepares datasets, and runs the training loop.
    """
    # Delayed imports for TTS components (heavy dependencies)
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from trainer import Trainer, TrainerArgs

    parser = argparse.ArgumentParser(description="Fine-tune XTTS-v2")
    parser.add_argument("--config", default="finetune_config.json", help="Config file")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("=" * 60)
    print("XTTS-v2 Fine-Tuning")
    print("=" * 60)
    print_gpu_info()
    print("=" * 60)

    # Create output directory
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get and verify model
    model_dir = get_model_directory()
    model_dir = ensure_model_downloaded(model_dir)
    print(f"Using model from: {model_dir}")

    # Load XTTS config and model
    print("\nLoading pre-trained XTTS-v2 model...")
    model_config = XttsConfig()
    model_config.load_json(str(model_dir / "config.json"))

    # Apply fine-tuning parameters
    model_config.batch_size = config.training.batch_size
    model_config.eval_batch_size = config.training.eval_batch_size
    model_config.num_epochs = config.training.num_epochs
    model_config.lr = config.training.learning_rate
    model_config.output_path = str(output_path)

    # Initialize model
    model: Xtts = Xtts.init_from_config(model_config)
    model.load_checkpoint(model_config, checkpoint_dir=str(model_dir), use_deepspeed=False)

    # Prepare datasets
    print("\nPreparing datasets...")
    train_samples, eval_samples = prepare_dataset(config)

    if len(train_samples) < 10:
        print("⚠️  Warning: Very few training samples. Results may be poor.")

    # Setup and run trainer
    trainer_args = TrainerArgs(
        restore_path=args.resume,
        skip_train_epoch=False,
        start_with_eval=True,
    )

    trainer = Trainer(
        trainer_args,
        model_config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("\nStarting fine-tuning...")
    print(f"Output directory: {output_path}")
    print(f"Training for {config.training.num_epochs} epochs")
    print("-" * 60)

    trainer.fit()

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

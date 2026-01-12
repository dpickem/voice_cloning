#!/usr/bin/env python3
"""
Fine-tune XTTS-v2 on custom voice dataset.

This script handles the complete fine-tuning process:
1. Load pre-trained XTTS-v2 model
2. Prepare dataset from metadata CSV
3. Run training loop
4. Save fine-tuned model

Usage:
    python finetune_xtts.py --config finetune_config.json
    python finetune_xtts.py --config finetune_config.json --resume /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

# Set environment variables before importing TTS
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from models import FinetuneConfig, Sample
from trainer import Trainer, TrainerArgs


def load_config(config_path: str) -> FinetuneConfig:
    """
    Load fine-tuning configuration from JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        FinetuneConfig instance containing data paths, training
        hyperparameters, and output settings.
    """
    with open(config_path, "r") as f:
        config_dict: dict[str, Any] = json.load(f)
    return FinetuneConfig.model_validate(config_dict)


def prepare_dataset(config: FinetuneConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Prepare training and evaluation datasets from CSV metadata files.

    Reads metadata CSV files in 'filename|text' format and creates
    sample dictionaries for each valid audio file found.

    Args:
        config: Fine-tuning configuration containing data paths.

    Returns:
        Tuple of (train_samples, eval_samples) where each is a list
        of sample dictionaries with audio_file, text, speaker_name,
        and language fields.
    """
    train_samples: list[dict[str, str]] = []
    eval_samples: list[dict[str, str]] = []

    audio_dir = Path(config.data.audio_dir)

    # Load training samples
    with open(config.data.train_csv, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 2:
                filename, text = parts
                audio_path = audio_dir / f"{filename}.wav"
                if audio_path.exists():
                    sample = Sample(
                        audio_file=str(audio_path),
                        text=text,
                        speaker_name="custom_voice",
                        language=config.data.language,
                    )
                    train_samples.append(sample.model_dump())

    # Load eval samples
    with open(config.data.eval_csv, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 2:
                filename, text = parts
                audio_path = audio_dir / f"{filename}.wav"
                if audio_path.exists():
                    sample = Sample(
                        audio_file=str(audio_path),
                        text=text,
                        speaker_name="custom_voice",
                        language=config.data.language,
                    )
                    eval_samples.append(sample.model_dump())

    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")

    return train_samples, eval_samples


def main() -> None:
    """
    Main entry point for XTTS-v2 fine-tuning.

    Parses command-line arguments, loads configuration, initializes
    the model, prepares datasets, and runs the training loop.
    """
    parser = argparse.ArgumentParser(description="Fine-tune XTTS-v2")
    parser.add_argument("--config", default="finetune_config.json", help="Config file")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("=" * 60)
    print("XTTS-v2 Fine-Tuning")
    print("=" * 60)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # Create output directory
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get TTS model cache directory
    tts_home: str = os.getenv("TTS_HOME", "/app/models")
    model_dir = Path(tts_home) / "tts_models--multilingual--multi-dataset--xtts_v2"

    # Load XTTS config and model
    print("\nLoading pre-trained XTTS-v2 model...")

    model_config_path = model_dir / "config.json"
    if not model_config_path.exists():
        print(f"Model config not found at {model_config_path}")
        print("Downloading model first...")
        from TTS.api import TTS
        TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("Model downloaded.")

    model_config = XttsConfig()
    model_config.load_json(str(model_config_path))

    # Update config with fine-tuning parameters
    model_config.batch_size = config.training.batch_size
    model_config.eval_batch_size = config.training.eval_batch_size
    model_config.num_epochs = config.training.num_epochs
    model_config.lr = config.training.learning_rate
    model_config.output_path = str(output_path)

    # Initialize model
    model: Xtts = Xtts.init_from_config(model_config)
    model.load_checkpoint(
        model_config,
        checkpoint_dir=str(model_dir),
        use_deepspeed=False
    )

    # Prepare datasets
    print("\nPreparing datasets...")
    train_samples, eval_samples = prepare_dataset(config)

    if len(train_samples) < 10:
        print("⚠️  Warning: Very few training samples. Results may be poor.")

    # Setup trainer
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

    # Start training
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

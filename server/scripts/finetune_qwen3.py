#!/usr/bin/env python3
"""
Fine-tune Qwen3-TTS on custom voice dataset.

This script handles the complete fine-tuning process for Qwen3-TTS:
1. Load pre-trained Qwen3-TTS model
2. Prepare dataset from metadata CSV
3. Run training loop with gradient checkpointing
4. Save fine-tuned model

Usage:
    python finetune_qwen3.py --config config/finetune_config.json

Reference:
    https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load fine-tuning configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def validate_dataset(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Validate training dataset exists and load metadata.

    Returns:
        Tuple of (train_samples, eval_samples)
    """
    data_config = config["data"]

    # Check for pre-split files first
    train_csv = Path(data_config.get("metadata_train_csv", ""))
    eval_csv = Path(data_config.get("metadata_eval_csv", ""))

    if train_csv.exists() and eval_csv.exists():
        print(f"Using pre-split datasets:")
        print(f"  Train: {train_csv}")
        print(f"  Eval: {eval_csv}")
        train_samples = load_metadata(train_csv, data_config)
        eval_samples = load_metadata(eval_csv, data_config)
    else:
        # Use main metadata and split
        metadata_csv = Path(data_config["metadata_csv"])
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

        all_samples = load_metadata(metadata_csv, data_config)

        # Split into train/eval
        random.seed(data_config.get("random_seed", 42))
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1 - data_config.get("eval_split_ratio", 0.1)))
        train_samples = all_samples[:split_idx]
        eval_samples = all_samples[split_idx:]

    print(f"\nDataset statistics:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Evaluation samples: {len(eval_samples)}")

    return train_samples, eval_samples


def load_metadata(csv_path: Path, data_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load samples from metadata CSV."""
    samples: list[dict[str, Any]] = []
    audio_dir = Path(data_config.get("processed_audio_dir", data_config["audio_dir"]))

    # Check if processed directory exists, fallback to raw
    if not audio_dir.exists():
        audio_dir = Path(data_config["audio_dir"])
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    with open(csv_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) != 2:
                print(f"  Warning: Invalid format at line {line_num}: {line}")
                continue

            filename, text = parts
            audio_path = audio_dir / f"{filename}.wav"

            if not audio_path.exists():
                print(f"  Warning: Audio file not found: {audio_path}")
                continue

            if len(text.strip()) < 3:
                print(f"  Warning: Text too short for {filename}")
                continue

            samples.append({
                "audio_file": str(audio_path.absolute()),
                "text": text.strip(),
                "speaker_name": "custom_voice",
                "language": data_config.get("language", "English"),
            })

    return samples


def check_gpu() -> bool:
    """Check GPU availability and print info."""
    # Lazy import: torch is heavy, only load when checking GPU
    import torch

    print("\n" + "=" * 60)
    print("GPU Information")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Training will be very slow on CPU.")
        return False

    print(f"CUDA available: True")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1e9
    print(f"GPU Memory: {total_memory:.1f} GB")

    if total_memory < 12:
        print("⚠️  Warning: < 12GB VRAM. Consider reducing batch_size or using gradient_checkpointing.")

    return True


def setup_training(
    config: dict[str, Any],
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
) -> tuple[Any, Path]:
    """
    Set up Qwen3-TTS fine-tuning.

    Note: This follows the official Qwen3-TTS fine-tuning guide.
    For the most up-to-date process, see:
    https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
    """
    # Lazy imports: heavy ML dependencies only loaded when actually training
    import torch
    from qwen_tts import Qwen3TTSModel

    training_config = config["training"]
    model_name = config["model"]
    output_path = Path(config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Loading Qwen3-TTS Model")
    print("=" * 60)
    print(f"Model: {model_name}")

    # Determine dtype
    dtype = torch.bfloat16 if training_config.get("dtype") == "bfloat16" else torch.float16

    # Check for FlashAttention
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2" if training_config.get("use_flash_attention", True) else "eager"
        print(f"Attention: {attn_impl}")
    except ImportError:
        attn_impl = "eager"
        print("FlashAttention not available, using eager attention")

    # Load model
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    print(f"✓ Model loaded")

    # Save config for reproducibility
    config_save_path = output_path / "finetune_config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {config_save_path}")

    return model, output_path


def run_finetuning(
    model: Any,
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Run the fine-tuning loop.

    Note: The actual fine-tuning implementation depends on the qwen-tts library's
    fine-tuning API. This is a template that should be adapted based on the
    official Qwen3-TTS fine-tuning guide.
    """
    training_config = config["training"]

    print("\n" + "=" * 60)
    print("Starting Fine-tuning")
    print("=" * 60)
    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"Effective batch: {training_config['batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"Epochs: {training_config['num_epochs']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Output: {output_path}")
    print("-" * 60)

    # Check if qwen_tts has a fine-tuning API
    # This is a placeholder - the actual implementation depends on the qwen-tts library
    try:
        # Try to use the official fine-tuning API if available
        from qwen_tts.finetuning import Qwen3TTSFineTuner

        finetuner = Qwen3TTSFineTuner(
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
            output_dir=str(output_path),
            batch_size=training_config["batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            num_epochs=training_config["num_epochs"],
            learning_rate=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            save_every_n_epochs=training_config["save_checkpoint_every_n_epochs"],
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        )

        finetuner.train()
        print("\n✓ Fine-tuning complete!")

    except ImportError:
        # If official fine-tuning API not available, provide instructions
        print("\n" + "=" * 60)
        print("MANUAL FINE-TUNING REQUIRED")
        print("=" * 60)
        print("""
The qwen-tts package doesn't include a built-in fine-tuning API.
Please follow the official Qwen3-TTS fine-tuning guide:

    https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning

Steps:
1. Clone the Qwen3-TTS repository:
   git clone https://github.com/QwenLM/Qwen3-TTS.git

2. Follow the fine-tuning instructions in the finetuning/ directory

3. Use the prepared dataset:
   - Training samples: {train_count}
   - Evaluation samples: {eval_count}
   - Audio directory: {audio_dir}

Your dataset has been validated and is ready for fine-tuning!
""".format(
            train_count=len(train_samples),
            eval_count=len(eval_samples),
            audio_dir=config["data"].get("processed_audio_dir", config["data"]["audio_dir"]),
        ))

        # Save samples for external fine-tuning script
        save_samples_for_external_finetuning(train_samples, eval_samples, output_path)


def save_samples_for_external_finetuning(
    train_samples: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save samples in a format usable by external fine-tuning scripts."""
    # Save as JSON for easy loading
    train_path = output_path / "train_samples.json"
    eval_path = output_path / "eval_samples.json"

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)

    with open(eval_path, "w") as f:
        json.dump(eval_samples, f, indent=2)

    print(f"\nDataset saved for external fine-tuning:")
    print(f"  Train: {train_path}")
    print(f"  Eval: {eval_path}")

    # Also save as simple text format
    train_txt = output_path / "train_manifest.txt"
    with open(train_txt, "w") as f:
        for sample in train_samples:
            f.write(f"{sample['audio_file']}|{sample['text']}\n")

    eval_txt = output_path / "eval_manifest.txt"
    with open(eval_txt, "w") as f:
        for sample in eval_samples:
            f.write(f"{sample['audio_file']}|{sample['text']}\n")

    print(f"  Train manifest: {train_txt}")
    print(f"  Eval manifest: {eval_txt}")


def main() -> None:
    """Main entry point for fine-tuning script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-TTS on custom voice dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fine-tune with default config
    python finetune_qwen3.py --config config/finetune_config.json

    # Validate dataset only (no training)
    python finetune_qwen3.py --config config/finetune_config.json --validate-only

Reference:
    https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/finetune_config.json",
        help="Path to fine-tuning config JSON",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate dataset, don't start training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-TTS Fine-Tuning")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"\nConfig: {config_path}")
    print(f"Model: {config['model']}")

    # Validate dataset
    try:
        train_samples, eval_samples = validate_dataset(config)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease record audio clips first. See:")
        print("  data/training/README.md")
        sys.exit(1)

    if len(train_samples) == 0:
        print("\nError: No valid training samples found!")
        print("Please record audio clips into: data/training/wavs/")
        sys.exit(1)

    if args.validate_only:
        print("\n✓ Dataset validation complete")
        print(f"  {len(train_samples)} training samples")
        print(f"  {len(eval_samples)} evaluation samples")
        sys.exit(0)

    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        response = input("\nContinue without GPU? (y/N): ")
        if response.lower() != "y":
            sys.exit(0)

    # Setup and run fine-tuning
    model, output_path = setup_training(config, train_samples, eval_samples)
    run_finetuning(model, train_samples, eval_samples, config, output_path)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

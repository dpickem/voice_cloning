#!/usr/bin/env python3
"""
Split metadata into training and evaluation sets.

Reads a metadata CSV file and randomly splits it into separate
training and evaluation files for model fine-tuning.

Usage:
    python split_dataset.py
    python split_dataset.py --metadata training_data/metadata.csv --train-ratio 0.85
"""

from __future__ import annotations

import argparse
import csv

import torch
from torch.utils.data import random_split


def split_dataset(
    metadata_file: str,
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[str, str]:
    """
    Split metadata into train and eval sets.

    Uses PyTorch's random_split for proper splitting with reproducible results.

    Args:
        metadata_file: Path to the metadata CSV file in 'filename|text' format.
        train_ratio: Fraction of data to use for training (0.0-1.0).
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_file, eval_file) paths to the generated files.
    """
    # Read all entries
    with open(metadata_file, "r") as f:
        reader = csv.reader(f, delimiter="|")
        entries = list(reader)

    # Use PyTorch's random_split for reproducible splitting
    train_size = int(len(entries) * train_ratio)
    eval_size = len(entries) - train_size
    generator = torch.Generator().manual_seed(seed)

    train_subset, eval_subset = random_split(entries, [train_size, eval_size], generator=generator)
    train_entries = [entries[i] for i in train_subset.indices]
    eval_entries = [entries[i] for i in eval_subset.indices]

    # Write train set
    train_file = metadata_file.replace(".csv", "_train.csv")
    with open(train_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(train_entries)

    # Write eval set
    eval_file = metadata_file.replace(".csv", "_eval.csv")
    with open(eval_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(eval_entries)

    print(f"Total entries: {len(entries)}")
    print(f"Training set: {len(train_entries)} ({train_file})")
    print(f"Evaluation set: {len(eval_entries)} ({eval_file})")

    return train_file, eval_file


def main() -> None:
    """
    Command-line interface for dataset splitting.

    Parses arguments and runs the split on the specified metadata file.
    """
    parser = argparse.ArgumentParser(description="Split metadata into train/eval sets")
    parser.add_argument("--metadata", default="training_data/metadata.csv", help="Metadata CSV file")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fraction for training (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_dataset(args.metadata, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()

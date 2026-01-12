#!/usr/bin/env python3
"""Unit tests for finetune_xtts.py dataset preparation functions."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase

import pytest

# Check if torch is available for tests that require it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Ensure the server module paths are available for imports
SERVER_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SERVER_DIR / "src"
SCRIPTS_DIR = SERVER_DIR / "scripts"
for path in [str(SERVER_DIR), str(SRC_DIR), str(SCRIPTS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from models import DataConfig, FinetuneConfig, TrainingConfig  # noqa: E402

# Skip marker for tests requiring torch
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


class LoadValidEntriesTests(TestCase):
    """Tests for loading and filtering metadata entries."""

    def setUp(self) -> None:
        """Create temporary directory structure for tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_dir = Path(self.temp_dir.name) / "audio"
        self.audio_dir.mkdir()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_metadata_csv(self, entries: list[tuple[str, str]]) -> Path:
        """Helper to create a metadata CSV file."""
        csv_path = Path(self.temp_dir.name) / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(entries)
        return csv_path

    def _create_audio_file(self, filename: str) -> Path:
        """Helper to create a dummy audio file."""
        audio_path = self.audio_dir / f"{filename}.wav"
        audio_path.touch()
        return audio_path

    def test_load_entries_with_existing_audio(self) -> None:
        """Should load entries when audio files exist."""
        # Import here to avoid issues with path setup
        from finetune_xtts import load_valid_entries

        # Create test data
        self._create_audio_file("clip_001")
        self._create_audio_file("clip_002")
        csv_path = self._create_metadata_csv([
            ("clip_001", "Hello world"),
            ("clip_002", "Test text"),
        ])

        entries, skipped = load_valid_entries(csv_path, self.audio_dir)

        self.assertEqual(len(entries), 2)
        self.assertEqual(skipped, 0)
        self.assertEqual(entries[0][0], "clip_001")
        self.assertEqual(entries[0][1], "Hello world")

    def test_load_entries_filters_missing_audio(self) -> None:
        """Should skip entries without corresponding audio files."""
        from finetune_xtts import load_valid_entries

        # Only create one audio file
        self._create_audio_file("clip_001")
        csv_path = self._create_metadata_csv([
            ("clip_001", "Hello world"),
            ("clip_002", "Missing audio"),
            ("clip_003", "Also missing"),
        ])

        entries, skipped = load_valid_entries(csv_path, self.audio_dir)

        self.assertEqual(len(entries), 1)
        self.assertEqual(skipped, 2)

    def test_load_entries_skips_malformed_rows(self) -> None:
        """Should skip rows that don't have exactly 2 columns."""
        from finetune_xtts import load_valid_entries

        self._create_audio_file("clip_001")
        csv_path = Path(self.temp_dir.name) / "metadata.csv"
        with open(csv_path, "w") as f:
            f.write("clip_001|Hello world\n")
            f.write("malformed_row\n")
            f.write("too|many|columns\n")

        entries, skipped = load_valid_entries(csv_path, self.audio_dir)

        self.assertEqual(len(entries), 1)


@requires_torch
class SplitEntriesTests(TestCase):
    """Tests for splitting entries into train/eval sets (requires torch)."""

    def test_split_entries_respects_ratio(self) -> None:
        """Should split entries according to eval_ratio."""
        from finetune_xtts import split_entries

        entries = [(f"clip_{i:03d}", f"text {i}", Path(f"/audio/clip_{i:03d}.wav"))
                   for i in range(100)]

        train, eval_ = split_entries(entries, eval_ratio=0.2, seed=42)

        self.assertEqual(len(train), 80)
        self.assertEqual(len(eval_), 20)

    def test_split_entries_is_reproducible(self) -> None:
        """Same seed should produce same split."""
        from finetune_xtts import split_entries

        entries = [(f"clip_{i:03d}", f"text {i}", Path(f"/audio/clip_{i:03d}.wav"))
                   for i in range(50)]

        train1, eval1 = split_entries(entries, eval_ratio=0.1, seed=42)
        train2, eval2 = split_entries(entries, eval_ratio=0.1, seed=42)

        self.assertEqual(train1, train2)
        self.assertEqual(eval1, eval2)

    def test_split_ensures_minimum_eval_samples(self) -> None:
        """Should ensure at least 1 eval sample when possible."""
        from finetune_xtts import split_entries

        entries = [(f"clip_{i:03d}", f"text {i}", Path(f"/audio/clip_{i:03d}.wav"))
                   for i in range(5)]

        # With 5 samples and 0.1 ratio, would normally get 0 eval samples
        train, eval_ = split_entries(entries, eval_ratio=0.1, seed=42)

        self.assertGreaterEqual(len(eval_), 1)
        self.assertEqual(len(train) + len(eval_), 5)


class EntriesToSamplesTests(TestCase):
    """Tests for converting entries to Sample dictionaries."""

    def test_converts_entries_to_sample_dicts(self) -> None:
        """Should convert entries to properly formatted Sample dicts."""
        from finetune_xtts import entries_to_samples

        entries = [
            ("clip_001", "Hello world", Path("/audio/clip_001.wav")),
            ("clip_002", "Test text", Path("/audio/clip_002.wav")),
        ]

        samples = entries_to_samples(entries, language="en")

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["audio_file"], "/audio/clip_001.wav")
        self.assertEqual(samples[0]["text"], "Hello world")
        self.assertEqual(samples[0]["speaker_name"], "custom_voice")
        self.assertEqual(samples[0]["language"], "en")


class SaveSplitCsvsTests(TestCase):
    """Tests for saving split metadata CSV files."""

    def setUp(self) -> None:
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_saves_train_and_eval_csvs(self) -> None:
        """Should save separate train and eval CSV files."""
        from finetune_xtts import save_split_csvs

        metadata_csv = Path(self.temp_dir.name) / "metadata.csv"
        train_entries = [("clip_001", "Train text", Path("/audio/clip_001.wav"))]
        eval_entries = [("clip_002", "Eval text", Path("/audio/clip_002.wav"))]

        train_csv, eval_csv = save_split_csvs(
            metadata_csv, train_entries, eval_entries
        )

        self.assertTrue(train_csv.exists())
        self.assertTrue(eval_csv.exists())

        # Verify contents
        with open(train_csv) as f:
            content = f.read()
            self.assertIn("clip_001", content)
            self.assertIn("Train text", content)


class LoadConfigTests(TestCase):
    """Tests for loading configuration from JSON."""

    def setUp(self) -> None:
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_loads_valid_config(self) -> None:
        """Should load and validate a proper config file."""
        from finetune_xtts import load_config
        import json

        config_path = Path(self.temp_dir.name) / "config.json"
        config_data: dict[str, Any] = {
            "output_path": "output/",
            "data": {
                "metadata_csv": "data/metadata.csv",
                "audio_dir": "data/audio/",
                "language": "en",
            },
            "training": {
                "batch_size": 2,
                "eval_batch_size": 2,
                "num_epochs": 10,
                "learning_rate": 1e-5,
            },
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config(str(config_path))

        self.assertIsInstance(config, FinetuneConfig)
        self.assertEqual(config.data.language, "en")
        self.assertEqual(config.training.num_epochs, 10)


@requires_torch
class PrepareDatasetIntegrationTests(TestCase):
    """Integration tests for the full prepare_dataset function (requires torch)."""

    def setUp(self) -> None:
        """Create temporary directory structure for tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_dir = Path(self.temp_dir.name) / "audio"
        self.audio_dir.mkdir()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def _create_test_config(self, csv_path: Path) -> FinetuneConfig:
        """Create a test configuration."""
        return FinetuneConfig(
            output_path=str(self.temp_dir.name),
            data=DataConfig(
                metadata_csv=str(csv_path),
                audio_dir=str(self.audio_dir),
                language="en",
                eval_split_ratio=0.2,
                random_seed=42,
            ),
            training=TrainingConfig(
                batch_size=2,
                eval_batch_size=2,
                num_epochs=10,
                learning_rate=1e-5,
            ),
        )

    def test_prepare_dataset_end_to_end(self) -> None:
        """Should prepare train/eval samples from metadata CSV."""
        from finetune_xtts import prepare_dataset

        # Create test data
        csv_path = Path(self.temp_dir.name) / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            for i in range(10):
                writer.writerow([f"clip_{i:03d}", f"Text number {i}"])
                (self.audio_dir / f"clip_{i:03d}.wav").touch()

        config = self._create_test_config(csv_path)
        train_samples, eval_samples = prepare_dataset(config)

        # With 10 samples and 0.2 ratio: 8 train, 2 eval
        self.assertEqual(len(train_samples), 8)
        self.assertEqual(len(eval_samples), 2)

    def test_prepare_dataset_raises_on_no_audio(self) -> None:
        """Should raise ValueError when no valid audio files exist."""
        from finetune_xtts import prepare_dataset

        csv_path = Path(self.temp_dir.name) / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["clip_001", "No audio file exists"])

        config = self._create_test_config(csv_path)

        with self.assertRaises(ValueError) as ctx:
            prepare_dataset(config)

        self.assertIn("No valid audio files found", str(ctx.exception))

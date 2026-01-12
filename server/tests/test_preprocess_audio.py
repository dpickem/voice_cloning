#!/usr/bin/env python3
"""Unit tests for preprocess_audio.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

# Ensure the server module paths are available for imports when running from repo root
SERVER_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SERVER_DIR / "src"
SCRIPTS_DIR = SERVER_DIR / "scripts"
for path in [str(SERVER_DIR), str(SRC_DIR), str(SCRIPTS_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from preprocess_audio import preprocess_audio  # noqa: E402


class PreprocessAudioTests(TestCase):
    """Tests for the audio preprocessing pipeline."""

    @mock.patch("audio.sf.write")
    @mock.patch("audio.librosa.util.normalize")
    @mock.patch("audio.librosa.effects.trim")
    @mock.patch("audio.nr.reduce_noise")
    @mock.patch("audio.librosa.resample")
    @mock.patch("audio.librosa.load")
    def test_preprocess_audio_pipeline(
        self,
        mock_load: mock.MagicMock,
        mock_resample: mock.MagicMock,
        mock_reduce_noise: mock.MagicMock,
        mock_trim: mock.MagicMock,
        mock_normalize: mock.MagicMock,
        mock_write: mock.MagicMock,
    ) -> None:
        """Ensure the preprocessing pipeline calls expected steps and writes output."""

        input_file = "input.wav"
        output_file = "output.wav"
        target_sr = 22050

        # Arrange mock behaviors
        mock_load.return_value = (np.array([0.0, 0.5, -0.5, 0.2], dtype=np.float32), 44100)
        mock_resample.return_value = np.array([0.0, 0.5], dtype=np.float32)
        mock_reduce_noise.return_value = np.array([0.0, 0.4], dtype=np.float32)
        mock_trim.return_value = (np.array([0.0, 0.4], dtype=np.float32), np.array([0, 0]))
        mock_normalize.return_value = np.array([0.0, 1.0], dtype=np.float32)

        # Act
        result = preprocess_audio(
            input_file,
            output_file,
            target_sr=target_sr,
            normalize=True,
            denoise=True,
        )

        # Assert return value
        self.assertEqual(result, output_file)

        # Assert pipeline calls
        mock_load.assert_called_once_with(input_file, sr=None, mono=True)
        mock_resample.assert_called_once_with(
            mock_load.return_value[0], orig_sr=mock_load.return_value[1], target_sr=target_sr
        )
        mock_reduce_noise.assert_called_once_with(y=mock_resample.return_value, sr=target_sr, prop_decrease=0.8)
        mock_trim.assert_called_once_with(mock_reduce_noise.return_value, top_db=25)
        mock_normalize.assert_called_once_with(mock_trim.return_value[0])

        # Assert write call includes expected audio and parameters
        mock_write.assert_called_once()
        write_args, write_kwargs = mock_write.call_args
        self.assertEqual(write_args[0], output_file)
        np.testing.assert_array_equal(write_args[1], mock_normalize.return_value)
        self.assertEqual(write_args[2], target_sr)
        self.assertEqual(write_kwargs.get("subtype"), "PCM_16")

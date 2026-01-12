#!/usr/bin/env python3
"""Unit tests for TTS server endpoints and shared utilities.

Tests cover:
- Health check endpoint
- Voice listing endpoint
- Synthesis validation logic
- Shared utility functions
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import TestCase, mock

import numpy as np

# Ensure the server module path is available for imports
SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestTTSUtils(TestCase):
    """Tests for shared TTS utility functions."""

    def test_wav_to_bytes_produces_valid_wav(self) -> None:
        """Ensure wav_to_bytes creates valid WAV data."""
        from tts_utils import wav_to_bytes

        # Create a simple sine wave
        sample_rate = 22050
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        wav: NDArray[np.floating] = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        result = wav_to_bytes(wav, sample_rate)

        # Check WAV header magic bytes (RIFF....WAVE)
        self.assertTrue(result.startswith(b"RIFF"))
        self.assertIn(b"WAVE", result[:12])
        self.assertGreater(len(result), 44)  # WAV header is 44 bytes minimum

    def test_wav_to_base64_produces_valid_encoding(self) -> None:
        """Ensure wav_to_base64 produces decodable base64."""
        from tts_utils import wav_to_base64

        sample_rate = 22050
        wav: NDArray[np.floating] = np.zeros(sample_rate, dtype=np.float32)

        result = wav_to_base64(wav, sample_rate)

        # Should be valid base64
        decoded = base64.b64decode(result)
        self.assertTrue(decoded.startswith(b"RIFF"))

    def test_validate_voice_path_returns_path_when_exists(self) -> None:
        """Ensure validate_voice_path returns Path when file exists."""
        from tts_utils import validate_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir)
            voice_file = voices_dir / "test_voice.wav"
            voice_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV-like file

            result = validate_voice_path("test_voice.wav", voices_dir)
            self.assertEqual(result, voice_file)

    def test_validate_voice_path_raises_on_missing_file(self) -> None:
        """Ensure validate_voice_path raises HTTPException for missing files."""
        from fastapi import HTTPException

        from tts_utils import validate_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir)

            with self.assertRaises(HTTPException) as ctx:
                validate_voice_path("nonexistent.wav", voices_dir)

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertIn("not found", ctx.exception.detail.lower())

    def test_list_voice_files_returns_wav_files(self) -> None:
        """Ensure list_voice_files returns only WAV files with correct info."""
        from tts_utils import list_voice_files

        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir)

            # Create test files
            wav1 = voices_dir / "voice1.wav"
            wav2 = voices_dir / "voice2.wav"
            txt = voices_dir / "readme.txt"

            wav1.write_bytes(b"RIFF" + b"\x00" * 100)
            wav2.write_bytes(b"RIFF" + b"\x00" * 200)
            txt.write_text("not a wav file")

            result = list_voice_files(voices_dir)

            self.assertEqual(len(result), 2)
            filenames = {v.filename for v in result}
            self.assertIn("voice1.wav", filenames)
            self.assertIn("voice2.wav", filenames)

    def test_list_voice_files_empty_directory(self) -> None:
        """Ensure list_voice_files returns empty list for empty directory."""
        from tts_utils import list_voice_files

        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_voice_files(Path(tmpdir))
            self.assertEqual(result, [])


class TestTTSModels(TestCase):
    """Tests for shared Pydantic models."""

    def test_tts_request_validation(self) -> None:
        """Ensure TTSRequest validates required fields."""
        from tts_models import TTSRequest

        # Valid request
        req = TTSRequest(text="Hello world")
        self.assertEqual(req.text, "Hello world")
        self.assertEqual(req.language, "en")  # Default

        # With custom language
        req = TTSRequest(text="Hola", language="es")
        self.assertEqual(req.language, "es")

    def test_tts_request_rejects_empty_text(self) -> None:
        """Ensure TTSRequest rejects empty text."""
        from pydantic import ValidationError

        from tts_models import TTSRequest

        with self.assertRaises(ValidationError):
            TTSRequest(text="")

    def test_tts_response_serialization(self) -> None:
        """Ensure TTSResponse serializes correctly."""
        from tts_models import TTSResponse

        resp = TTSResponse(
            success=True,
            audio_base64="dGVzdA==",
            duration_seconds=1.5,
            sample_rate=22050,
            processing_time_ms=150.0,
            text_length=10,
        )

        data = resp.model_dump()
        self.assertTrue(data["success"])
        self.assertEqual(data["sample_rate"], 22050)

    def test_health_response_with_model_type(self) -> None:
        """Ensure HealthResponse supports optional model_type."""
        from tts_models import HealthResponse

        # Without model_type
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            gpu_available=False,
            gpu_name=None,
            cuda_version=None,
            container=True,
            timestamp="2024-01-01T00:00:00",
        )
        self.assertIsNone(resp.model_type)

        # With model_type
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type="fine-tuned",
            gpu_available=True,
            gpu_name="RTX 4090",
            cuda_version="12.1",
            container=True,
            timestamp="2024-01-01T00:00:00",
        )
        self.assertEqual(resp.model_type, "fine-tuned")


class TestTTSConfig(TestCase):
    """Tests for configuration loading."""

    def test_config_has_required_attributes(self) -> None:
        """Ensure tts_config settings has required configuration."""
        from tts_config import settings

        self.assertIsInstance(settings.VOICE_REFERENCES_DIR, Path)
        self.assertIsInstance(settings.AUDIO_OUTPUT_DIR, Path)
        self.assertIsInstance(settings.DEFAULT_VOICE, str)
        self.assertIsInstance(settings.HOST, str)
        self.assertIsInstance(settings.PORT, int)

    @mock.patch.dict("os.environ", {"PORT": "9000", "HOST": "127.0.0.1"})
    def test_config_respects_environment_variables(self) -> None:
        """Ensure configuration can be overridden via environment."""
        # Force reimport to pick up new env vars
        import importlib

        import tts_config

        importlib.reload(tts_config)

        self.assertEqual(tts_config.settings.PORT, 9000)
        self.assertEqual(tts_config.settings.HOST, "127.0.0.1")


class TestTTSServerEndpoints(TestCase):
    """Integration tests for TTS server endpoints using TestClient.

    Note: These tests require torch and TTS packages to be installed.
    They are skipped if those dependencies are not available.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Check if heavy dependencies are available."""
        try:
            import torch  # noqa: F401

            cls.torch_available = True
        except ImportError:
            cls.torch_available = False

    def setUp(self) -> None:
        """Set up test fixtures."""
        if not self.torch_available:
            self.skipTest("torch not installed - skipping integration tests")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.voices_dir = Path(self.temp_dir.name) / "voices"
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.voices_dir.mkdir()
        self.output_dir.mkdir()

        # Create a test voice file
        self.test_voice = self.voices_dir / "test_voice.wav"
        self._create_minimal_wav(self.test_voice)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _create_minimal_wav(self, path: Path) -> None:
        """Create a minimal valid WAV file for testing."""
        import soundfile as sf

        # Create 0.1 second of silence
        wav = np.zeros(2205, dtype=np.float32)
        sf.write(str(path), wav, 22050, subtype="PCM_16")

    @mock.patch("tts_server.tts_model")
    def test_health_endpoint_returns_status(
        self, mock_model: mock.MagicMock
    ) -> None:
        """Ensure /health returns proper status."""
        from fastapi.testclient import TestClient

        from tts_server import app

        mock_model.__bool__ = lambda self: True
        mock_model.__ne__ = lambda self, other: other is None

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("gpu_available", data)

    @mock.patch("tts_server.settings")
    def test_voices_endpoint_lists_files(
        self, mock_settings: mock.MagicMock
    ) -> None:
        """Ensure /voices lists available voice files."""
        from fastapi.testclient import TestClient

        from tts_server import app

        # Configure mock settings to use test directory
        mock_settings.VOICE_REFERENCES_DIR = self.voices_dir

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/voices")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)


if __name__ == "__main__":
    import unittest

    unittest.main()

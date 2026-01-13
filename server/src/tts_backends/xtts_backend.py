#!/usr/bin/env python3
"""
XTTS v2 (Coqui TTS) backend implementation.

Uses the Coqui TTS library for zero-shot voice cloning with XTTS-v2 model.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from tts_backends.base import ModelType, TTSBackend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from TTS.api import TTS

# Agree to Coqui TOS
os.environ["COQUI_TOS_AGREED"] = "1"

# Model identifier
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Supported languages for XTTS-v2
XTTS_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"]


class XTTSBackend(TTSBackend):
    """
    XTTS v2 backend using Coqui TTS library.

    High-quality multilingual voice cloning with zero-shot capability.
    Requires NVIDIA GPU with CUDA support.
    """

    model_type = ModelType.XTTS
    display_name = "XTTS v2 (Coqui TTS)"
    supports_languages = XTTS_LANGUAGES
    requires_reference_text = False

    def __init__(self) -> None:
        """Initialize XTTS backend."""
        super().__init__()
        self._model: Optional[TTS] = None
        self._sample_rate: int = 24000  # XTTS output sample rate

    def load(self) -> None:
        """Load XTTS-v2 model with GPU acceleration."""
        if self._loaded:
            return

        # PyTorch 2.6+ changed weights_only default to True, breaking XTTS checkpoint loading
        # Workaround: Patch torch.load to use weights_only=False for TTS model loading
        import torch
        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load

        try:
            from TTS.api import TTS

            print(f"Loading {self.display_name}...")
            self._model = TTS(XTTS_MODEL_NAME, gpu=True)
            self._sample_rate = self._model.synthesizer.output_sample_rate
            self._loaded = True
            print(f"✓ {self.display_name} loaded")
        finally:
            # Restore original torch.load
            torch.load = _original_torch_load

    def unload(self) -> None:
        """Unload XTTS model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        reference_text: str | None = None,
    ) -> tuple[NDArray[np.floating], int]:
        """
        Synthesize speech using XTTS-v2.

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference WAV file.
            language: Language code.
            reference_text: Not used by XTTS.

        Returns:
            Tuple of (waveform array, sample rate).
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("XTTS model not loaded")

        import numpy as np

        wav = self._model.tts(
            text=text,
            speaker_wav=reference_audio,
            language=language,
        )

        # Ensure numpy array
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)

        return wav, self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Output sample rate (24000 Hz for XTTS)."""
        return self._sample_rate

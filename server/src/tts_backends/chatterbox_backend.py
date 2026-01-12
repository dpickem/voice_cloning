#!/usr/bin/env python3
"""
Chatterbox (Resemble AI) backend implementation.

Chatterbox is an MIT-licensed, multilingual TTS and voice-cloning model
with high audio quality and real-time synthesis capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tts_backends.base import ModelType, TTSBackend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Chatterbox supported languages
CHATTERBOX_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "nl", "ja", "ko", "zh"]


class ChatterboxBackend(TTSBackend):
    """
    Chatterbox backend for voice cloning.

    Chatterbox is a production-ready TTS model from Resemble AI with:
    - Zero-shot voice cloning from a few seconds of audio
    - Expressive speech with emotion control
    - Accent control
    - Real-time synthesis

    GitHub: https://github.com/resemble-ai/chatterbox
    """

    model_type = ModelType.CHATTERBOX
    display_name = "Chatterbox (Resemble AI)"
    supports_languages = CHATTERBOX_LANGUAGES
    requires_reference_text = False

    def __init__(self) -> None:
        """Initialize Chatterbox backend."""
        super().__init__()
        self._model = None
        self._sample_rate: int = 24000

    def load(self) -> None:
        """Load Chatterbox model."""
        if self._loaded:
            return

        try:
            from chatterbox.tts import ChatterboxTTS

            print(f"Loading {self.display_name}...")
            self._model = ChatterboxTTS.from_pretrained(device="cuda")
            self._loaded = True
            print(f"✓ {self.display_name} loaded")

        except ImportError as e:
            raise RuntimeError(
                f"Chatterbox not installed. Install with: pip install chatterbox-tts\n"
                f"Original error: {e}"
            )

    def unload(self) -> None:
        """Unload Chatterbox model from memory."""
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
        Synthesize speech using Chatterbox.

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference WAV file.
            language: Language code.
            reference_text: Not used by Chatterbox.

        Returns:
            Tuple of (waveform array, sample rate).
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Chatterbox model not loaded")

        import numpy as np
        import torchaudio

        # Load reference audio
        ref_wav, ref_sr = torchaudio.load(reference_audio)

        # Generate speech
        wav = self._model.generate(
            text=text,
            audio_prompt=ref_wav,
            audio_prompt_sr=ref_sr,
        )

        # Convert to numpy
        if hasattr(wav, 'cpu'):
            wav = wav.cpu().numpy()
        wav = np.array(wav, dtype=np.float32).squeeze()

        return wav, self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self._sample_rate

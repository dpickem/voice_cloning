#!/usr/bin/env python3
"""
F5-TTS backend implementation.

F5-TTS is a zero-shot voice cloning model that produces high-quality
natural speech. It requires reference text (transcript of reference audio).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tts_backends.base import ModelType, TTSBackend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# F5-TTS supported languages
F5_LANGUAGES = ["en", "zh"]


class F5TTSBackend(TTSBackend):
    """
    F5-TTS backend for zero-shot voice cloning.

    F5-TTS produces high-quality, natural-sounding speech with
    excellent voice cloning capabilities. Unlike XTTS, it requires
    a transcript of the reference audio for best results.

    GitHub: https://github.com/SWivid/F5-TTS
    """

    model_type = ModelType.F5_TTS
    display_name = "F5-TTS"
    supports_languages = F5_LANGUAGES
    requires_reference_text = True

    def __init__(self) -> None:
        """Initialize F5-TTS backend."""
        super().__init__()
        self._model = None
        self._sample_rate: int = 24000  # F5-TTS output sample rate

    def load(self) -> None:
        """Load F5-TTS model."""
        if self._loaded:
            return

        try:
            from f5_tts.api import F5TTS

            print(f"Loading {self.display_name}...")
            self._model = F5TTS()
            self._loaded = True
            print(f"✓ {self.display_name} loaded")

        except ImportError as e:
            raise RuntimeError(
                f"F5-TTS not installed. Install with: pip install f5-tts\n"
                f"Original error: {e}"
            )

    def unload(self) -> None:
        """Unload F5-TTS model from memory."""
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
        Synthesize speech using F5-TTS.

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference WAV file.
            language: Language code ('en' or 'zh').
            reference_text: Transcript of reference audio (recommended).

        Returns:
            Tuple of (waveform array, sample rate).

        Note:
            F5-TTS works best with reference_text provided. If not provided,
            it will attempt synthesis but quality may be reduced.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("F5-TTS model not loaded")

        import numpy as np
        import tempfile
        import soundfile as sf

        # F5-TTS infer method
        # Note: API may vary slightly between versions
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            self._model.infer(
                ref_file=reference_audio,
                ref_text=reference_text or "",  # Empty string if not provided
                gen_text=text,
                file_wave=output_path,
            )

            # Read the generated audio
            wav, sr = sf.read(output_path)
            wav = np.array(wav, dtype=np.float32)
            self._sample_rate = sr

            return wav, sr

        finally:
            # Clean up temp file
            import os
            if os.path.exists(output_path):
                os.unlink(output_path)

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self._sample_rate

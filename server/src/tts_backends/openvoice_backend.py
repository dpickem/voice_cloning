#!/usr/bin/env python3
"""
OpenVoice (MyShell) backend implementation.

OpenVoice provides instant voice cloning with fine-grained control
over voice style, including emotion, accent, rhythm, pauses, and intonation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tts_backends.base import ModelType, TTSBackend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# OpenVoice supported languages
OPENVOICE_LANGUAGES = ["en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"]


class OpenVoiceBackend(TTSBackend):
    """
    OpenVoice backend for instant voice cloning.

    OpenVoice V2 provides:
    - Zero-shot voice cloning
    - Fine-grained style control (emotion, accent, rhythm)
    - Cross-lingual voice cloning
    - Fast inference

    GitHub: https://github.com/myshell-ai/OpenVoice
    """

    model_type = ModelType.OPENVOICE
    display_name = "OpenVoice (MyShell)"
    supports_languages = OPENVOICE_LANGUAGES
    requires_reference_text = False

    def __init__(self) -> None:
        """Initialize OpenVoice backend."""
        super().__init__()
        self._base_speaker_tts = None
        self._tone_color_converter = None
        self._sample_rate: int = 24000

    def load(self) -> None:
        """Load OpenVoice models (base TTS + tone color converter)."""
        if self._loaded:
            return

        try:
            import torch
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter, BaseSpeakerTTS

            print(f"Loading {self.display_name}...")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load base speaker TTS (MeloTTS)
            self._base_speaker_tts = BaseSpeakerTTS(
                config_path="openvoice/checkpoints_v2/base_speakers/EN/config.json",
                device=device
            )
            self._base_speaker_tts.load_ckpt("openvoice/checkpoints_v2/base_speakers/EN/checkpoint.pth")

            # Load tone color converter
            self._tone_color_converter = ToneColorConverter(
                config_path="openvoice/checkpoints_v2/converter/config.json",
                device=device
            )
            self._tone_color_converter.load_ckpt("openvoice/checkpoints_v2/converter/checkpoint.pth")

            self._se_extractor = se_extractor
            self._device = device
            self._loaded = True
            print(f"✓ {self.display_name} loaded")

        except ImportError as e:
            raise RuntimeError(
                f"OpenVoice not installed. Install with: pip install openvoice-cli\n"
                f"Original error: {e}"
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"OpenVoice model checkpoints not found. Download from:\n"
                f"https://github.com/myshell-ai/OpenVoice\n"
                f"Original error: {e}"
            )

    def unload(self) -> None:
        """Unload OpenVoice models from memory."""
        if self._base_speaker_tts is not None:
            del self._base_speaker_tts
            self._base_speaker_tts = None
        if self._tone_color_converter is not None:
            del self._tone_color_converter
            self._tone_color_converter = None
        self._loaded = False

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        reference_text: str | None = None,
    ) -> tuple[NDArray[np.floating], int]:
        """
        Synthesize speech using OpenVoice.

        OpenVoice uses a two-stage process:
        1. Generate speech with base speaker
        2. Convert tone color to match reference voice

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference WAV file.
            language: Language code.
            reference_text: Not used by OpenVoice.

        Returns:
            Tuple of (waveform array, sample rate).
        """
        if not self._loaded:
            raise RuntimeError("OpenVoice model not loaded")

        import numpy as np
        import tempfile
        import soundfile as sf

        # Map language codes to OpenVoice format
        lang_map = {
            "en": "EN",
            "zh": "ZH",
            "ja": "JP",
            "ko": "KR",
            "fr": "FR",
            "de": "DE",
            "es": "ES",
            "it": "IT",
            "pt": "PT",
            "ru": "RU",
        }
        ov_lang = lang_map.get(language.lower(), "EN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = f"{tmp_dir}/base.wav"
            output_path = f"{tmp_dir}/output.wav"

            # Step 1: Generate base speech
            self._base_speaker_tts.tts(
                text=text,
                output_path=base_path,
                speaker="default",
                language=ov_lang,
                speed=1.0,
            )

            # Step 2: Extract speaker embedding from reference
            target_se, _ = self._se_extractor.get_se(
                reference_audio,
                self._tone_color_converter,
                vad=True,
            )

            # Step 3: Extract base speaker embedding
            source_se = self._base_speaker_tts.get_spk_embed(base_path)

            # Step 4: Convert tone color
            self._tone_color_converter.convert(
                audio_src_path=base_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
            )

            # Read output
            wav, sr = sf.read(output_path)
            wav = np.array(wav, dtype=np.float32)
            self._sample_rate = sr

            return wav, sr

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self._sample_rate

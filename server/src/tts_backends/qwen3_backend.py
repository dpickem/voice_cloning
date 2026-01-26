#!/usr/bin/env python3
"""
Qwen3-TTS backend implementation — SOTA voice cloning (January 2026).

Uses the qwen-tts library for state-of-the-art voice cloning with only 3 seconds
of reference audio. Supports 10 languages with ultra-low latency streaming.

Key Features:
    - 3-second rapid voice cloning (SOTA)
    - 97ms first packet latency
    - 10 language support
    - Voice prompt caching for faster inference
    - Optional reference text for improved cloning quality

Note:
    Heavy dependencies (torch, qwen_tts) are imported lazily in methods to avoid
    loading them until actually needed. This improves startup time when using
    other backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tts_backends.base import ModelType, TTSBackend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Model variants
QWEN3_TTS_MODELS = {
    "base-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "custom-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "custom-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "design-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Default model for voice cloning
DEFAULT_MODEL = QWEN3_TTS_MODELS["base-1.7b"]

# Supported languages for Qwen3-TTS
QWEN3_LANGUAGES = ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

# Language code to Qwen3-TTS language name mapping
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


class Qwen3TTSBackend(TTSBackend):
    """
    Qwen3-TTS backend for SOTA voice cloning.

    Released January 2026, this model achieves state-of-the-art performance
    on voice cloning benchmarks with only 3 seconds of reference audio.
    """

    model_type = ModelType.QWEN3_TTS
    display_name = "Qwen3-TTS (SOTA)"
    supports_languages = QWEN3_LANGUAGES
    requires_reference_text = False  # Optional but improves quality

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """
        Initialize Qwen3-TTS backend.

        Args:
            model_name: HuggingFace model ID or local path. Default is 1.7B Base model.
        """
        super().__init__()
        self._model = None
        self._model_name = model_name
        self._sample_rate: int = 24000  # Qwen3-TTS output sample rate
        self._voice_prompts: dict[str, object] = {}  # Cached voice prompts

    def load(self) -> None:
        """Load Qwen3-TTS model with GPU acceleration and FlashAttention 2."""
        if self._loaded:
            return

        import torch

        print(f"Loading {self.display_name} ({self._model_name})...")

        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Import qwen_tts
        from qwen_tts import Qwen3TTSModel

        # Determine attention implementation
        # FlashAttention 2 requires GPU and bfloat16
        if gpu_available:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                print("  Using FlashAttention 2")
            except ImportError:
                attn_impl = "eager"
                print("  FlashAttention not available, using eager attention")
        else:
            attn_impl = "eager"

        # Load model
        self._model = Qwen3TTSModel.from_pretrained(
            self._model_name,
            device_map="cuda:0" if gpu_available else "cpu",
            dtype=torch.bfloat16 if gpu_available else torch.float32,
            attn_implementation=attn_impl,
        )

        self._loaded = True
        print(f"✓ {self.display_name} loaded")

    def unload(self) -> None:
        """Unload Qwen3-TTS model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        # Clear cached voice prompts
        self._voice_prompts.clear()

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._loaded = False

    def create_voice_prompt(
        self,
        reference_audio: str,
        reference_text: str | None = None,
    ) -> object:
        """
        Create a reusable voice prompt from reference audio.

        Caches the prompt for faster subsequent synthesis with the same voice.

        Args:
            reference_audio: Path to reference audio file.
            reference_text: Transcript of reference audio (improves quality).

        Returns:
            Voice prompt object that can be reused for synthesis.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Qwen3-TTS model not loaded")

        # Check cache
        cache_key = f"{reference_audio}:{reference_text or ''}"
        if cache_key in self._voice_prompts:
            return self._voice_prompts[cache_key]

        # Create voice prompt
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=reference_text,
            x_vector_only_mode=reference_text is None,
        )

        # Cache for reuse
        self._voice_prompts[cache_key] = prompt
        return prompt

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "en",
        reference_text: str | None = None,
    ) -> tuple[NDArray[np.floating], int]:
        """
        Synthesize speech using Qwen3-TTS voice cloning.

        Args:
            text: Text to synthesize.
            reference_audio: Path to reference WAV file (only 3 seconds needed!).
            language: Language code (en, zh, ja, ko, de, fr, ru, pt, es, it).
            reference_text: Transcript of reference audio (optional, improves quality).

        Returns:
            Tuple of (waveform array, sample rate).
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Qwen3-TTS model not loaded")

        import numpy as np

        # Map language code to Qwen3-TTS language name
        qwen_language = LANGUAGE_MAP.get(language.lower(), "English")

        # Check if we have a cached voice prompt
        cache_key = f"{reference_audio}:{reference_text or ''}"
        
        if cache_key in self._voice_prompts:
            # Use cached voice prompt for faster inference
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=qwen_language,
                voice_clone_prompt=self._voice_prompts[cache_key],
            )
        else:
            # Try to load reference text from .txt file if not provided
            if reference_text is None:
                txt_path = Path(reference_audio).with_suffix(".txt")
                if txt_path.exists():
                    reference_text = txt_path.read_text().strip()

            # Generate with voice reference
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=qwen_language,
                ref_audio=reference_audio,
                ref_text=reference_text,
            )

        # Get first waveform (batch size 1)
        wav = wavs[0]

        # Ensure numpy array
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)

        return wav, sr

    def clear_voice_cache(self) -> None:
        """Clear all cached voice prompts."""
        self._voice_prompts.clear()

    @property
    def sample_rate(self) -> int:
        """Output sample rate (24000 Hz for Qwen3-TTS)."""
        return self._sample_rate

    def get_info(self) -> dict:
        """Get backend information for health checks."""
        info = super().get_info()
        info.update({
            "model_name": self._model_name,
            "cached_voices": len(self._voice_prompts),
            "min_reference_audio": "3 seconds",
        })
        return info

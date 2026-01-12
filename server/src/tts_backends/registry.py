#!/usr/bin/env python3
"""
TTS Backend registry for dynamic model loading.

Provides factory functions to instantiate the appropriate backend
based on configuration.
"""

from __future__ import annotations

from typing import Type

from tts_backends.base import ModelType, TTSBackend

# Backend class registry (lazy imports to avoid loading unused dependencies)
_BACKEND_REGISTRY: dict[ModelType, str] = {
    ModelType.XTTS: "tts_backends.xtts_backend.XTTSBackend",
    ModelType.F5_TTS: "tts_backends.f5tts_backend.F5TTSBackend",
}


def _import_backend_class(model_type: ModelType) -> Type[TTSBackend]:
    """
    Dynamically import a backend class.

    Args:
        model_type: The model type to import.

    Returns:
        The backend class.

    Raises:
        ValueError: If model type is not registered.
        ImportError: If backend module cannot be imported.
    """
    if model_type not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    module_path = _BACKEND_REGISTRY[model_type]
    module_name, class_name = module_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_backend(model_type: ModelType | str) -> TTSBackend:
    """
    Get a TTS backend instance for the specified model type.

    Args:
        model_type: Model type enum or string (e.g., 'xtts', 'f5-tts').

    Returns:
        Unloaded backend instance. Call .load() to initialize.

    Raises:
        ValueError: If model type is unknown.

    Example:
        >>> backend = get_backend(ModelType.F5_TTS)
        >>> backend.load()
        >>> wav, sr = backend.synthesize("Hello", "ref.wav")
    """
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)

    backend_class = _import_backend_class(model_type)
    return backend_class()


def get_available_backends() -> list[dict]:
    """
    Get information about all available backends.

    Returns:
        List of dicts with backend metadata.
    """
    backends = []
    for model_type in ModelType:
        try:
            backend_class = _import_backend_class(model_type)
            backends.append({
                "model_type": model_type.value,
                "display_name": backend_class.display_name,
                "supports_languages": backend_class.supports_languages,
                "requires_reference_text": backend_class.requires_reference_text,
                "available": True,
            })
        except ImportError:
            backends.append({
                "model_type": model_type.value,
                "display_name": model_type.value,
                "available": False,
                "error": "Dependencies not installed",
            })
    return backends

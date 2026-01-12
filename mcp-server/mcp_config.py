#!/usr/bin/env python3
"""
MCP Voice TTS Configuration

This module provides type-safe configuration loading using Pydantic settings.
Environment variables are loaded from .env file and validated.

CONFIGURATION HIERARCHY:
    1. Environment variables (highest priority)
    2. .env file in mcp-server directory
    3. Defaults defined in this file (lowest priority)

WHAT GOES WHERE:
    - .env / Environment variables: Deployment-specific settings (server URL, etc.)
    - This file (mcp_config.py): Environment variable definitions with types and defaults

Usage:
    from mcp_config import settings

    # Access settings via the settings object
    server_url = settings.TTS_SERVER_URL
    voice = settings.DEFAULT_VOICE
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Directory paths for default resolution
_MCP_DIR = Path(__file__).resolve().parent


class MCPSettings(BaseSettings):
    """
    MCP Voice TTS settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    """

    # =========================================================================
    # TTS SERVER CONNECTION
    # =========================================================================
    # URL of the TTS server (Docker container)
    TTS_SERVER_URL: str = "http://10.111.79.180:8080"

    # Request timeout in seconds for TTS synthesis
    REQUEST_TIMEOUT: float = 60.0

    # =========================================================================
    # VOICE SETTINGS
    # =========================================================================
    # Default voice reference filename (should exist on TTS server)
    DEFAULT_VOICE: str = "voice_reference.wav"

    # Default language code for synthesis
    DEFAULT_LANGUAGE: str = "en"

    # =========================================================================
    # PYDANTIC SETTINGS CONFIG
    # =========================================================================
    model_config = {
        "env_file": str(_MCP_DIR / ".env"),
        "env_file_encoding": "utf-8",
        # Allow extra fields to be ignored (forward compatibility)
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> MCPSettings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once and reused.
    """
    return MCPSettings()


def clear_settings_cache() -> None:
    """
    Clear the settings cache.

    Useful for testing when environment variables change and settings
    need to be reloaded. After calling this, the next call to get_settings()
    will create a new MCPSettings instance.
    """
    get_settings.cache_clear()


# Clear cache on module reload to support testing with different env vars
# This ensures that if the module is reloaded (e.g., importlib.reload),
# fresh settings are loaded from the current environment
clear_settings_cache()

# Primary settings object
settings = get_settings()

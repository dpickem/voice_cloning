#!/usr/bin/env python3
"""
MCP Server for Voice Cloning TTS.

Provides Cursor with text-to-speech capabilities using a remote
voice cloning server (Docker container). Plays synthesized audio
through local speakers.

Tools:
    - speak: Convert text to speech and play through speakers
    - list_voices: List available voice references
    - check_tts_server: Check TTS server status
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import httpx
import sounddevice as sd
import soundfile as sf
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_config import settings

# Create MCP server
server: Server = Server("voice-tts")


async def play_audio(audio_bytes: bytes) -> None:
    """
    Play audio bytes through local speakers.

    Reads WAV data from a byte buffer and plays it using the default
    audio output device. Blocks until playback completes.

    Args:
        audio_bytes: Raw WAV audio data as bytes.
    """
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Play audio (blocking)
    sd.play(audio_data, sample_rate)
    sd.wait()


async def synthesize_and_play(
    text: str,
    language: str = settings.DEFAULT_LANGUAGE,
    voice: str = settings.DEFAULT_VOICE,
) -> dict[str, Any]:
    """
    Synthesize text to speech and play the resulting audio.

    Sends a synthesis request to the remote TTS server, receives the
    audio response, and plays it through local speakers.

    Args:
        text: The text to convert to speech.
        language: Language code for synthesis (e.g., 'en', 'es', 'fr').
        voice: Filename of the voice reference on the server.

    Returns:
        Dictionary containing synthesis metadata:
            - duration_seconds: Length of generated audio
            - sample_rate: Audio sample rate in Hz
            - text_length: Number of characters in input text

    Raises:
        Exception: If the TTS server returns an error response.
    """
    async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{settings.TTS_SERVER_URL}/synthesize/raw",
            json={"text": text, "language": language, "voice": voice},
        )

        if response.status_code != 200:
            raise Exception(f"TTS server error: {response.text}")

        audio_bytes: bytes = response.content
        duration: float = float(response.headers.get("X-Duration-Seconds", 0))
        sample_rate: int = int(response.headers.get("X-Sample-Rate", 22050))

        # Play audio
        await play_audio(audio_bytes)

        return {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "text_length": len(text),
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available MCP tools for voice synthesis.

    Returns:
        List of Tool definitions with schemas for speak, speak_summary,
        list_voices, and check_tts_server operations.
    """
    return [
        Tool(
            name="speak",
            description="Convert text to speech using your cloned voice and play through speakers. Use this to read text aloud or provide audio feedback.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak aloud",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (default: en). Supported: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi",
                        "default": "en",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice reference filename (default: voice_reference.wav)",
                        "default": "voice_reference.wav",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="list_voices",
            description="List available voice reference files on the TTS server",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="check_tts_server",
            description="Check the status of the TTS server (Docker container) and its GPU availability",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle MCP tool invocations.

    Routes tool calls to appropriate handlers for speech synthesis,
    voice listing, and server health checks.

    Args:
        name: The name of the tool to invoke.
        arguments: Dictionary of arguments passed to the tool.

    Returns:
        List containing a single TextContent with the result message.
    """
    if name == "speak":
        text: str = arguments.get("text", "")
        language: str = arguments.get("language", settings.DEFAULT_LANGUAGE)
        voice: str = arguments.get("voice", settings.DEFAULT_VOICE)

        if not text:
            return [TextContent(type="text", text="Error: No text provided")]

        try:
            result = await synthesize_and_play(text, language, voice)
            return [
                TextContent(
                    type="text",
                    text=f"✓ Spoke {len(text)} characters ({result['duration_seconds']:.1f}s audio)",
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "list_voices":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{settings.TTS_SERVER_URL}/voices")

                if response.status_code == 200:
                    voices: list[dict[str, Any]] = response.json()
                    if voices:
                        voice_list: str = "\n".join(
                            [
                                f"  - {v['filename']} ({v['size_bytes'] / 1024:.1f} KB)"
                                for v in voices
                            ]
                        )
                        return [
                            TextContent(
                                type="text", text=f"Available voices:\n{voice_list}"
                            )
                        ]
                    else:
                        return [
                            TextContent(
                                type="text", text="No voice references found on server"
                            )
                        ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"Error fetching voices: {response.status_code}",
                        )
                    ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "check_tts_server":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{settings.TTS_SERVER_URL}/health")

                if response.status_code == 200:
                    health: dict[str, Any] = response.json()
                    status: str = "✓" if health["status"] == "healthy" else "✗"
                    gpu: str = health.get("gpu_name", "None")
                    cuda: str = health.get("cuda_version", "N/A")
                    container: str = "Docker" if health.get("container") else "Native"
                    return [
                        TextContent(
                            type="text",
                            text=f"TTS Server Status:\n"
                            f"  Status: {status} {health['status']}\n"
                            f"  Model loaded: {health['model_loaded']}\n"
                            f"  GPU: {gpu}\n"
                            f"  CUDA: {cuda}\n"
                            f"  Runtime: {container}\n"
                            f"  URL: {settings.TTS_SERVER_URL}",
                        )
                    ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"Server returned error: {response.status_code}",
                        )
                    ]
        except httpx.ConnectError:
            return [
                TextContent(
                    type="text",
                    text=f"✗ Cannot connect to TTS server at {settings.TTS_SERVER_URL}\n"
                    f"  Make sure the Docker container is running on desktop (10.111.79.180)\n"
                    f"  Try: ssh desktop 'cd ~/workspace/dpickem_voice_cloning/server && docker compose up -d'",
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main() -> None:
    """
    Run the MCP server.

    Initializes the stdio transport and starts the server event loop.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

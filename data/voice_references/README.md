# Voice References for Zero-Shot Voice Cloning

This directory contains reference audio samples and text scripts for zero-shot voice cloning.

## What is Zero-Shot Voice Cloning?

Zero-shot voice cloning allows you to clone a voice using just a short audio sample (typically 15-30 seconds) without any fine-tuning. The model extracts voice characteristics from the reference audio and applies them to synthesize new speech.

## Directory Structure

```
voice_references/
├── sample_text.txt      # Sample text to read for creating reference audio
├── README.md            # This file
└── *.wav                # Your recorded voice reference files (add here)
```

## Creating a Voice Reference

1. **Read the sample text** in `sample_text.txt` aloud
2. **Record your voice** using these guidelines:
   - Duration: 15-30 seconds
   - Format: WAV (16kHz sample rate or higher)
   - Environment: Quiet room with minimal background noise
   - Microphone: Keep consistent distance (6-12 inches)
   - Speaking style: Natural, clear, steady pace

3. **Save the recording** in this directory with a descriptive name:
   - Example: `john_doe_reference.wav`
   - Example: `speaker_01.wav`

## Usage

Reference audio files from this directory can be used with the TTS server's zero-shot voice cloning feature by specifying the path to the audio file in your API request.

## Supported Formats

- **WAV** (recommended)
- **MP3**
- **FLAC**
- **OGG**

For best results, use uncompressed WAV files at 16kHz or higher sample rate.

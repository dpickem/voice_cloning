# Voice Cloning TTS Server

A FastAPI-based Text-to-Speech server with zero-shot voice cloning capabilities using XTTS v2 (Coqui TTS).

## Architecture Overview

```
server/
├── src/
│   ├── server.py           # FastAPI application
│   ├── config.py           # Pydantic settings (env vars)
│   ├── models.py           # Request/response models
│   ├── utils.py            # Audio utilities
│   └── tts_backends/       # Pluggable TTS backend system
│       ├── base.py         # Abstract base class & ModelType enum
│       ├── registry.py     # Backend factory & discovery
│       └── xtts_backend.py # XTTS v2 implementation
├── docker/
│   ├── Dockerfile          # CUDA-enabled container
│   └── docker-compose.yml  # Production deployment
├── config/
│   └── env.example         # Environment variable template
└── scripts/
    ├── finetune_xtts.py    # Fine-tuning script
    └── preprocess_audio.py # Audio preprocessing
```

## Design Journey & Lessons Learned

### Backend Selection: Why XTTS Only?

We initially designed a multi-backend system supporting:
- **XTTS v2** (Coqui TTS) - Zero-shot voice cloning
- **F5-TTS** - High-quality voice cloning
- **Chatterbox** (Resemble AI) - Production TTS
- **OpenVoice** (MyShell) - Voice cloning

**However, dependency conflicts forced us to XTTS-only:**

| Backend | Issue | Conflict |
|---------|-------|----------|
| Chatterbox | `transformers==4.46.3` (exact) | XTTS needs `transformers<4.40.0` |
| F5-TTS | Pulls in `torchcodec` | Requires specific FFmpeg + PyTorch versions |
| OpenVoice | Removed by user request | - |

**Lesson:** TTS libraries have notoriously incompatible dependencies. Pick ONE backend and stick with it, or use separate containers/environments for each.

### PyTorch 2.6 Breaking Change

**Problem:** PyTorch 2.6 changed `torch.load()` default from `weights_only=False` to `weights_only=True`, breaking XTTS checkpoint loading.

**Error:**
```
WeightsUnpickler error: Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig
```

**Solution:** Temporarily patch `torch.load` during model loading:

```python
def load(self) -> None:
    import torch
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    try:
        # Load model...
    finally:
        torch.load = _original_torch_load
```

### Docker Configuration Pitfalls

#### 1. Root-Owned Files in Bind Mounts

**Problem:** Docker runs as root by default, creating root-owned files in mounted volumes.

**Solution:** Create non-root user in Dockerfile:
```dockerfile
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && \
    useradd -m -u ${UID} -g ${GID} appuser
USER appuser
```

#### 2. Path Resolution Inside Container

**Problem:** Python's `Path(__file__).parent` resolves differently inside Docker vs local development.

Local: `/Users/you/project/server/src/config.py` → parent chain works
Docker: `/app/config.py` → parent of `/app` is `/`, breaks path calculations

**Solution:** Set explicit paths via environment variables in docker-compose.yml:
```yaml
environment:
  - VOICE_REFERENCES_DIR=/app/voice_references
  - AUDIO_OUTPUT_DIR=/app/audio_output
```

#### 3. PyTorch Version Drift

**Problem:** Other packages (F5-TTS) upgrade PyTorch, breaking CUDA setup.

**Solution:** Pin exact versions in requirements.txt:
```
torch==2.1.0
torchaudio==2.1.0
```

## Quick Start

### Docker (Recommended)

```bash
cd server
docker compose -f docker/docker-compose.yml up -d --build

# Check logs
docker logs voice-tts-server -f

# Test health
curl http://localhost:8080/health | jq
```

### Local Development

```bash
cd server
python -m venv venv
source venv/bin/activate
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd src
python server.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health, GPU status, model info |
| `/voices` | GET | List available voice reference files |
| `/models` | GET | List available TTS backends |
| `/synthesize` | POST | Generate speech (JSON + base64 audio) |
| `/synthesize/raw` | POST | Generate speech (raw WAV bytes) |

### Synthesize Request

```bash
curl -X POST http://localhost:8080/synthesize/raw \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a voice cloning test.",
    "voice": "voice_reference.wav",
    "language": "en"
  }' \
  --output output.wav
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to synthesize (1-5000 chars) |
| `voice` | string | No | Voice reference filename (default: voice_reference.wav) |
| `language` | string | No | Language code (default: en) |

### Supported Languages (XTTS)

en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu

## Fine-Tuning

To fine-tune XTTS on your own voice:

```bash
# Prepare training data in data/training/
# - wavs/ directory with audio files
# - metadata.csv with "filename|transcript" format

cd server
python scripts/finetune_xtts.py --config config/finetune_config.json
```

See `scripts/finetune_xtts.py` for detailed documentation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server bind address |
| `PORT` | 8080 | Server port |
| `EXTERNAL_HOST` | localhost | External IP for remote access |
| `VOICE_REFERENCES_DIR` | ../data/voice_references | Voice files directory |
| `AUDIO_OUTPUT_DIR` | ./audio_output | Generated audio directory |
| `TTS_BACKEND` | xtts | TTS backend to use |
| `DEFAULT_VOICE` | voice_reference.wav | Default voice file |

## Troubleshooting

### "CUDA is not available"

Ensure NVIDIA drivers and CUDA toolkit are installed. For Docker, install nvidia-container-toolkit:
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### "Model not loaded" / Startup fails

Check Docker logs for specific errors:
```bash
docker logs voice-tts-server
```

Common issues:
- Missing voice reference files
- Insufficient GPU memory (~6GB required)
- Network issues downloading model on first run

### Slow first request

XTTS model loading takes ~10-15 seconds. First inference may also be slower due to CUDA kernel compilation.

## Future Considerations

If adding new TTS backends:

1. **Use separate Docker images** - Don't try to mix backends with conflicting deps
2. **Pin ALL versions** - TTS libraries are version-sensitive
3. **Test PyTorch compatibility** - New PyTorch versions often break things
4. **Consider API services** - OpenAI TTS, ElevenLabs, etc. avoid dependency hell

## License

MIT

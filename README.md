# Voice Cloning TTS with MCP Server

A voice cloning text-to-speech system using XTTS-v2, running as a Docker container with GPU acceleration. Includes an MCP (Model Context Protocol) server for integration with Cursor IDE.

## Architecture

```
┌─────────────────────┐      HTTP/REST      ┌─────────────────────┐
│   Mac Client        │ ──────────────────> │   Linux Server      │
│                     │                     │   (desktop)         │
│  ┌───────────────┐  │                     │  ┌───────────────┐  │
│  │ Cursor IDE    │  │                     │  │ Docker        │  │
│  │   + MCP       │  │                     │  │  Container    │  │
│  │   Server      │  │                     │  │   + XTTS-v2   │  │
│  └───────────────┘  │                     │  │   + GPU       │  │
│         │           │                     │  └───────────────┘  │
│         ▼           │                     │                     │
│  ┌───────────────┐  │                     │                     │
│  │ Audio Output  │  │                     │                     │
│  │  (Speakers)   │  │                     │                     │
│  └───────────────┘  │                     │                     │
└─────────────────────┘                     └─────────────────────┘
```

## Quick Start

### 1. Start TTS Server (on Linux server)

```bash
# SSH to server and navigate to repo
ssh desktop
cd ~/workspace/dpickem_voice_cloning/server

# Build and start Docker container
docker compose up -d

# Check logs
docker compose logs -f voice-tts
```

### 2. Setup MCP Server (on Mac)

```bash
cd mcp-server
chmod +x setup.sh
./setup.sh
```

### 3. Configure Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "voice-tts": {
      "command": "/path/to/mcp-server/venv/bin/python",
      "args": ["/path/to/mcp-server/mcp_voice_tts.py"],
      "env": {
        "TTS_SERVER_URL": "http://10.111.79.180:8080"
      }
    }
  }
}
```

Restart Cursor to load the MCP server.

### 4. Add Voice Reference

Record 15-30 seconds of clear speech, then:

```bash
# Copy to server
scp your_voice.wav desktop:~/workspace/dpickem_voice_cloning/server/voice_references/

# Preprocess (optional but recommended)
ssh desktop "cd ~/workspace/dpickem_voice_cloning/server && \
  docker compose run --rm voice-tts python preprocess_audio.py \
  /app/voice_references/your_voice.wav \
  -o /app/voice_references/voice_reference.wav"
```

## Project Structure

```
dpickem_voice_cloning/
├── server/                           # Docker-based TTS server
│   ├── Dockerfile                   # Container build instructions
│   ├── docker-compose.yml           # Container orchestration
│   ├── requirements.txt             # Python dependencies
│   ├── tts_server.py               # FastAPI TTS server (zero-shot)
│   ├── tts_server_finetuned.py     # FastAPI TTS server (auto-detects fine-tuned)
│   ├── preprocess_audio.py         # Single file audio preprocessing
│   ├── preprocess_training_data.py # Batch preprocessing for fine-tuning
│   ├── split_dataset.py            # Train/eval dataset splitter
│   ├── finetune_config.json        # Fine-tuning hyperparameters
│   ├── finetune_xtts.py            # Fine-tuning script
│   ├── voice_references/           # Voice reference audio files
│   ├── audio_output/               # Generated audio files
│   ├── logs/                       # Server logs
│   ├── models/                     # Cached TTS model weights
│   ├── training_data/              # Fine-tuning training data
│   │   ├── README.md               # Dataset documentation
│   │   ├── metadata.csv            # 100 pre-made text snippets
│   │   ├── metadata.csv.example    # Format documentation
│   │   ├── wavs/                   # Raw training audio clips
│   │   └── processed/              # Preprocessed training clips
│   └── finetuned_model/            # Fine-tuned model checkpoints
│
├── mcp-server/                      # Cursor MCP integration
│   ├── mcp_voice_tts.py            # MCP server script
│   ├── requirements.txt            # Python dependencies
│   ├── setup.sh                    # Setup script
│   └── env.example                 # Environment configuration template
│
├── design_docs/                    # Design documentation
└── implementation_plans/           # Implementation plans
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, GPU status, container info |
| `/voices` | GET | List available voice references |
| `/synthesize` | POST | Synthesize speech (base64 response) |
| `/synthesize/raw` | POST | Synthesize speech (raw WAV bytes) |

### Example API Usage

```bash
# Health check
curl http://10.111.79.180:8080/health | jq

# List voices
curl http://10.111.79.180:8080/voices | jq

# Synthesize speech
curl -X POST http://10.111.79.180:8080/synthesize/raw \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "language": "en"}' \
  -o test.wav
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `speak` | Convert text to speech and play through speakers |
| `speak_summary` | Speak a summary aloud |
| `list_voices` | List available voice references |
| `check_tts_server` | Check TTS server status |

### Usage in Cursor

```
@voice-tts speak "Hello, world!"
@voice-tts check the TTS server status
@voice-tts list available voices
```

## Fine-Tuning (Optional)

Fine-tuning improves voice quality but requires more audio data and training time.

| Approach | Audio Needed | Training Time | Quality |
|----------|--------------|---------------|---------|
| **Zero-Shot** | 15-30 seconds | None | Good |
| **Fine-Tuning** | 5-30 minutes | 2-8 hours | Excellent |

### Fine-Tuning Steps

#### 1. Record Training Data

A pre-made dataset of **100 diverse text snippets** is included in `server/training_data/metadata.csv`. The dataset covers:

- Pangrams (phoneme coverage)
- Technical content (numbers, versions, API terms)
- Questions, exclamations, conversational speech
- Emotional range (enthusiastic, apologetic, thoughtful)
- Longer passages for prosody training

Record each clip as `wavs/clip_001.wav`, `wavs/clip_002.wav`, etc. See `training_data/README.md` for detailed recording guidelines.

#### 2. Prepare Dataset
```bash
# Dataset already exists - just record the audio clips
# See server/training_data/metadata.csv for the 100 text snippets

# Preprocess audio
docker compose run --rm voice-tts python preprocess_training_data.py \
  --input-dir /app/training_data/wavs \
  --output-dir /app/training_data/processed \
  --metadata /app/training_data/metadata.csv

# Split into train/eval sets
docker compose run --rm voice-tts python split_dataset.py \
  --metadata /app/training_data/metadata.csv
```

#### 3. Run Fine-Tuning
```bash
docker compose run --rm voice-tts python finetune_xtts.py --config finetune_config.json
```

#### 4. Use Fine-Tuned Model
Replace `tts_server.py` with `tts_server_finetuned.py` in Dockerfile, or copy the fine-tuned server:
```bash
cp tts_server_finetuned.py tts_server.py
docker compose up -d --build
```

The server auto-detects fine-tuned models in `finetuned_model/`.

## Requirements

### Linux Server (TTS Server)
- NVIDIA GPU with 8GB+ VRAM (12GB+ for fine-tuning)
- NVIDIA Driver 525.60.13+
- Docker 24.0+
- NVIDIA Container Toolkit

### Mac Client (MCP Server)
- Python 3.9+
- Cursor IDE with MCP support
- Network access to Linux server

## Docker Commands

| Task | Command |
|------|---------|
| Start container | `docker compose up -d` |
| Stop container | `docker compose down` |
| View logs | `docker compose logs -f voice-tts` |
| Restart | `docker compose restart` |
| Rebuild | `docker compose build --no-cache` |
| Shell access | `docker compose exec voice-tts bash` |
| Check GPU | `docker compose exec voice-tts nvidia-smi` |

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs voice-tts

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Cannot connect from Mac
```bash
# Test connectivity
curl http://10.111.79.180:8080/health

# Check firewall
ssh desktop "sudo ufw allow 8080/tcp"
```

### MCP server not appearing in Cursor
1. Verify `~/.cursor/mcp.json` syntax
2. Check Python path is correct
3. Restart Cursor completely (Cmd+Q)

## License

MIT

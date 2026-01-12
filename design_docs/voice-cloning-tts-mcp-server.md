# Voice Cloning TTS with MCP Server Integration

**Created:** 2026-01-07  
**Updated:** 2026-01-09  
**Status:** 🟢 Ready for Implementation  
**Tags:** #design-proposal #tts #voice-cloning #mcp #cursor  
**Implementation:** [[voice-cloning-tts-mcp-server-implementation]]

---

## Overview

Build a personal voice cloning system using open-source TTS models that runs locally on a Linux desktop, accessible via API from other machines (e.g., Mac), with MCP server integration for Cursor.

## Goals

1. **Voice Cloning** — Clone my voice using an open-source, off-the-shelf TTS model
2. **Local Deployment** — Run the model locally on a Linux desktop
3. **Remote API Access** — Access the model from another machine (Mac) via HTTP/REST API
4. **MCP Integration** — Create an MCP server for Cursor integration that sends text and outputs audio via local speaker

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Mac (Client)                                │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Cursor    │───▶│   MCP Server     │───▶│  Local Speaker   │   │
│  │   (Text)    │    │   (Python)       │    │  (Audio Output)  │   │
│  └─────────────┘    └────────┬─────────┘    └──────────────────┘   │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ HTTP/REST API
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Linux Desktop (Server)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    TTS API Server (FastAPI)                   │   │
│  │  ┌────────────────┐    ┌────────────────────────────────┐    │   │
│  │  │  Voice Clone   │    │     TTS Model (e.g., XTTS-v2)  │    │   │
│  │  │  Reference     │───▶│     - Text → Speech            │    │   │
│  │  │  Audio Files   │    │     - Voice Embedding          │    │   │
│  │  └────────────────┘    └────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  GPU: NVIDIA RTX (for inference acceleration)                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Open Source TTS Models (as of January 2026)

### ⚠️ Qwen3-TTS-VC-VoiceDesign (API-Only)

| Attribute | Details |
|-----------|---------|
| **Model** | Qwen3-TTS-VC-VoiceDesign |
| **Voice Cloning** | ✅ High-fidelity voice cloning |
| **Features** | TTS + Voice Conversion (VC) + Voice Design |
| **Local Hosting** | ❌ **No** — API-only via Alibaba Cloud DashScope |
| **Blog** | [qwen.ai/blog](https://qwen.ai/blog?id=qwen3-tts-vc-voicedesign) |

**⚠️ Important:** This model is **NOT available for local self-hosting**. It requires calling Alibaba Cloud's DashScope API, which means:
- Requires internet connection
- Requires Alibaba Cloud account and API key
- Audio data is sent to cloud servers
- May have usage costs

**Pros:**
- High-quality voice cloning with subtle nuance capture
- Combined TTS + Voice Conversion capabilities
- Voice Design feature for creating new voices
- No local GPU required

**Cons:**
- ❌ Cannot be self-hosted locally
- Requires internet and cloud API
- Privacy concerns (audio sent to cloud)
- Potential API costs

---

### ⭐ Recommended for Local Hosting: XTTS-v2 (Coqui TTS)

| Attribute | Details |
|-----------|---------|
| **Model** | XTTS-v2 (from Coqui TTS) |
| **Voice Cloning** | ✅ Zero-shot with ~6 seconds of reference audio |
| **Languages** | 17 languages supported |
| **Features** | Emotional tone replication, speaking style preservation |
| **Local Hosting** | ✅ **Yes** — Fully self-hostable, weights on HuggingFace |
| **License** | Open source (Coqui shut down in 2024, but project continues) |
| **GitHub** | [coqui-ai/TTS](https://github.com/coqui-ai/TTS) |

**Pros:**
- ✅ **Fully self-hostable** — runs 100% locally
- ✅ Model weights freely downloadable from HuggingFace
- Mature codebase with extensive documentation
- Excellent voice cloning quality with minimal reference audio
- Multi-language support (17 languages)
- Active community despite company shutdown
- No cloud dependency, full privacy

**Cons:**
- No official commercial support (community-maintained)
- Requires local GPU (8GB+ VRAM recommended)

---

### Alternative: GLM-TTS

| Attribute | Details |
|-----------|---------|
| **Model** | GLM-TTS |
| **Voice Cloning** | ✅ Zero-shot with 3-10 seconds of reference audio |
| **Languages** | Chinese, English |
| **Features** | Real-time streaming, emotional expression control |
| **Local Hosting** | ✅ **Yes** — Open source, self-hostable |
| **Website** | [glm-tts.com](https://glm-tts.com/) |

**Pros:**
- ✅ Can be self-hosted locally
- Newer model with advanced emotional control
- Real-time streaming inference
- Minimal reference audio needed (3-10 sec)

**Cons:**
- Primarily focused on Chinese/English

---

### Alternative: OpenVoice

| Attribute | Details |
|-----------|---------|
| **Model** | OpenVoice (V2) |
| **Voice Cloning** | ✅ Instant voice cloning |
| **Languages** | Cross-lingual via IPA phoneme alignment |
| **Features** | Fine-grained control over emotion, accent, rhythm, pauses, intonation |
| **Local Hosting** | ✅ **Yes** — Open source, self-hostable |
| **Info** | [OpenVoice Framework](https://www.emergentmind.com/topics/openvoice-framework) |

**Pros:**
- ✅ Can be self-hosted locally
- Very fine-grained control over vocal attributes
- Cross-lingual capabilities
- Instant cloning

**Cons:**
- May require more tuning for best results

---

### Alternative: YourTTS

| Attribute | Details |
|-----------|---------|
| **Model** | YourTTS |
| **Voice Cloning** | ✅ Zero-shot multi-speaker |
| **Languages** | Multiple languages |
| **Local Hosting** | ✅ **Yes** — Open source, self-hostable |
| **GitHub** | Available in various implementations |

**Pros:**
- ✅ Can be self-hosted locally
- Multi-speaker support
- Zero-shot cloning

**Cons:**
- Less mature than XTTS-v2

---

## Voice Cloning Process

This section details how to actually clone your voice using reference audio samples.

### Audio Sample Requirements

For optimal voice cloning results, your input audio must meet these specifications:

| Requirement | Specification |
|-------------|---------------|
| **Format** | WAV (16-bit), MP3, or M4A |
| **Duration** | 10-30 seconds recommended (min 6 sec, max 60 sec) |
| **File Size** | Less than 10 MB |
| **Sample Rate** | 24 kHz or higher (44.1 kHz ideal) |
| **Channels** | Mono |
| **Bit Depth** | 16-bit or higher |

### Recording Best Practices

**Environment:**
- [ ] Quiet room with minimal echo/reverb
- [ ] No background noise (AC, fans, traffic)
- [ ] No background music or other voices
- [ ] Avoid rooms with hard surfaces (use soft furnishings to dampen)

**Equipment:**
- [ ] Use a good quality microphone (USB condenser recommended)
- [ ] Pop filter to reduce plosives (p, b, t sounds)
- [ ] Consistent distance from microphone (6-12 inches)
- [ ] Use a mic stand (avoid handling noise)

**Content:**
- [ ] Speak naturally at your normal pace
- [ ] Include diverse phonemes and sentence types
- [ ] Read varied content (not just one repeated phrase)
- [ ] Minimize pauses (keep gaps ≤ 2 seconds)
- [ ] At least 3 seconds of continuous, clear speech

**Sample Recording Script:**
```
"Hello, my name is [Name]. I'm recording this sample to clone my voice.
The quick brown fox jumps over the lazy dog. Pack my box with five 
dozen liquor jugs. How vexingly quick daft zebras jump! The five 
boxing wizards jump quickly. Sphinx of black quartz, judge my vow."
```
This script covers most English phonemes in natural sentences.

### Voice Cloning Approaches

#### Approach 1: Zero-Shot Cloning (Recommended)

Zero-shot cloning extracts voice characteristics from a short audio sample without any training. This is the fastest approach.

**For Qwen3-TTS-VC-VoiceDesign:**
```python
# Voice enrollment - creates a voice profile from reference audio
from dashscope.audio.tts_v2 import VoiceEnrollmentService

# Create voice profile from reference audio
enrollment = VoiceEnrollmentService()
voice_id = enrollment.create_voice(
    audio_file="my_voice_reference.wav",
    target_model="qwen3-tts-vc-realtime-2025-11-27"
)

# Use cloned voice for synthesis
from dashscope.audio.tts_v2 import SpeechSynthesizer

synthesizer = SpeechSynthesizer(
    model="qwen3-tts-vc-realtime-2025-11-27",
    voice=voice_id
)
audio = synthesizer.synthesize("Hello, this is my cloned voice!")
```

**For XTTS-v2 (Coqui TTS):**
```python
from TTS.api import TTS

# Load model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Zero-shot cloning - just pass reference audio
tts.tts_to_file(
    text="Hello, this is my cloned voice speaking.",
    speaker_wav="my_voice_reference.wav",  # Your recorded sample
    language="en",
    file_path="output.wav"
)
```

#### Approach 2: Fine-Tuning (Higher Quality)

For significantly better voice reproduction, you can fine-tune the model on your voice data. This requires more audio and training time but produces more accurate and consistent results.

##### When to Choose Fine-Tuning

| Criteria | Zero-Shot | Fine-Tuning |
|----------|-----------|-------------|
| **Setup Time** | Minutes | Hours to days |
| **Audio Required** | 15-30 seconds | 5-30 minutes |
| **Voice Accuracy** | Good | Excellent |
| **Consistency** | Variable | Highly consistent |
| **Best For** | Quick testing, general use | Production, high fidelity |

**Choose fine-tuning when:**
- Zero-shot results don't capture your voice characteristics accurately
- You need consistent, production-quality output
- Voice nuances (accent, cadence, intonation) are important
- You have time to invest in data preparation and training

##### Training Data Requirements

| Quality Level | Audio Duration | Number of Clips | Expected Results |
|---------------|----------------|-----------------|------------------|
| **Minimum** | 5 minutes | 50+ clips | Noticeable improvement |
| **Good** | 15 minutes | 150+ clips | Significant improvement |
| **Excellent** | 30+ minutes | 300+ clips | Near-perfect reproduction |

**Recording Guidelines:**
- Record diverse content: varied sentences, questions, exclamations
- Include different emotional tones: neutral, happy, serious
- Cover all common phonemes and phoneme combinations
- Maintain consistent microphone placement and room acoustics
- Keep each clip 3-15 seconds long
- Ensure accurate transcriptions for each clip

##### Hardware Requirements for Fine-Tuning

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU VRAM** | 12GB | 16GB+ |
| **System RAM** | 32GB | 64GB |
| **Storage** | 50GB free | 100GB free |
| **Training Time** | 2-4 hours | 4-8 hours |

##### Dataset Structure

```
voice_dataset/
├── metadata.csv          # text|audio_file mapping
├── metadata_train.csv    # 90% for training
├── metadata_eval.csv     # 10% for evaluation
├── wavs/                 # Raw audio clips
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
└── processed/            # Preprocessed clips (normalized, denoised)
    ├── clip_001.wav
    └── ...
```

**metadata.csv format:**
```csv
clip_001|Hello, this is the first sample sentence.
clip_002|The weather today is quite pleasant.
clip_003|I enjoy working on interesting projects.
clip_004|Would you like to hear more about this topic?
clip_005|That's absolutely incredible news!
```

##### Fine-Tuning Process Overview

1. **Data Collection** — Record 5-30 minutes of diverse speech
2. **Transcription** — Create accurate text transcriptions for each clip
3. **Preprocessing** — Normalize audio, reduce noise, trim silence
4. **Dataset Split** — 90% training, 10% evaluation
5. **Training** — Run fine-tuning (2-8 hours depending on data size)
6. **Evaluation** — Compare against zero-shot baseline
7. **Deployment** — Replace base model with fine-tuned version

**Fine-tuning with XTTS-v2:**
```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from trainer import Trainer, TrainerArgs

# Load pre-trained model config
config = XttsConfig()
config.load_json("path/to/xtts_v2/config.json")

# Configure training parameters
config.batch_size = 2
config.num_epochs = 50
config.lr = 5e-6

# Initialize model from pre-trained weights
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="path/to/xtts_v2/")

# Setup trainer
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path="my_voice_model/",
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Start fine-tuning
trainer.fit()
```

##### Fine-Tuning Trade-offs

| Pros | Cons |
|------|------|
| ✅ Much better voice accuracy | ❌ Requires significant audio recording |
| ✅ Consistent output quality | ❌ Training takes 2-8+ hours |
| ✅ Better emotional range | ❌ Needs accurate transcriptions |
| ✅ Production-ready quality | ❌ Higher GPU requirements (12GB+) |
| ✅ Personalized to your voice | ❌ Model versioning complexity |

> **Note:** See [[voice-cloning-tts-mcp-server-implementation#4-phase-2a-voice-model-fine-tuning-optional]] for detailed step-by-step implementation instructions.

### Voice Quality Optimization

**Tips for better cloning results:**

1. **Multiple Reference Samples:** Provide 3-5 different audio clips covering various emotions/tones
2. **Longer is Better:** 20-30 seconds produces better results than 6 seconds
3. **Clean Audio:** Remove any noise, clicks, or artifacts before cloning
4. **Consistent Recording:** Use same mic/room for all samples
5. **Natural Speech:** Avoid over-enunciation or unnatural pacing

**Audio Preprocessing Script:**
```python
import librosa
import soundfile as sf
import noisereduce as nr

def preprocess_audio(input_file, output_file, target_sr=24000):
    # Load audio
    audio, sr = librosa.load(input_file, sr=None)
    
    # Resample to target sample rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Reduce background noise
    audio = nr.reduce_noise(y=audio, sr=target_sr)
    
    # Normalize volume
    audio = librosa.util.normalize(audio)
    
    # Save processed audio
    sf.write(output_file, audio, target_sr)
    
    return output_file

# Usage
preprocess_audio("raw_recording.wav", "processed_reference.wav")
```

### Supported Languages

| Model | Languages |
|-------|-----------|
| **Qwen3-TTS-VC** | Chinese, English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, Russian |
| **XTTS-v2** | English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi |
| **GLM-TTS** | Chinese, English |
| **OpenVoice** | Cross-lingual (any via IPA phoneme alignment) |

---

## Implementation Plan

### Phase 1: Linux Server Setup

1. **Install TTS Model**
   ```bash
   pip install TTS
   # or clone from GitHub for latest version
   git clone https://github.com/coqui-ai/TTS.git
   cd TTS && pip install -e .
   ```

2. **Record Voice Reference Audio**
   - Record 10-30 seconds of clear speech
   - Diverse sentences covering different phonemes
   - High quality audio (44.1kHz, minimal background noise)

3. **Test Voice Cloning Locally**
   ```python
   from TTS.api import TTS
   
   tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
   tts.tts_to_file(
       text="Hello, this is my cloned voice speaking.",
       speaker_wav="my_voice_reference.wav",
       language="en",
       file_path="output.wav"
   )
   ```

### Phase 2: API Server (FastAPI)

Create a REST API server on the Linux desktop:

```python
# tts_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import base64
import io
import soundfile as sf

app = FastAPI()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    try:
        wav = tts.tts(
            text=request.text,
            speaker_wav="voice_reference.wav",
            language=request.language
        )
        # Convert to base64 for transmission
        buffer = io.BytesIO()
        sf.write(buffer, wav, 22050, format='WAV')
        audio_base64 = base64.b64encode(buffer.getvalue()).decode()
        return {"audio": audio_base64, "format": "wav"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run with:
```bash
uvicorn tts_server:app --host 0.0.0.0 --port 8080
```

### Phase 3: MCP Server for Cursor (Mac)

Create an MCP server that:
1. Receives text from Cursor
2. Calls the Linux TTS API
3. Plays audio through Mac's local speaker

```python
# mcp_tts_server.py
import asyncio
import base64
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
import sounddevice as sd
import numpy as np
import soundfile as sf
import io

TTS_SERVER_URL = "http://linux-desktop:8080"

server = Server("voice-tts")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="speak",
            description="Convert text to speech using cloned voice and play through speaker",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech"
                    }
                },
                "required": ["text"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "speak":
        text = arguments["text"]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TTS_SERVER_URL}/synthesize",
                json={"text": text},
                timeout=30.0
            )
            
        if response.status_code == 200:
            data = response.json()
            audio_bytes = base64.b64decode(data["audio"])
            
            # Play audio through local speaker
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            sd.play(audio_data, sample_rate)
            sd.wait()
            
            return [TextContent(type="text", text=f"Spoke: {text}")]
        else:
            return [TextContent(type="text", text=f"Error: {response.text}")]

if __name__ == "__main__":
    import mcp
    mcp.run(server)
```

### Phase 4: Cursor Integration

Add to Cursor's MCP config (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "voice-tts": {
      "command": "python",
      "args": ["/path/to/mcp_tts_server.py"],
      "env": {
        "TTS_SERVER_URL": "http://linux-desktop:8080"
      }
    }
  }
}
```

## Hardware Requirements

### Linux Desktop (Server)
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3070+ recommended)
- **RAM:** 16GB+
- **Storage:** 10GB+ for model weights
- **CUDA:** 11.7+

### Mac (Client)
- Audio output capability
- Python 3.9+
- Network access to Linux desktop

## Dependencies

### Linux Server
```txt
TTS>=0.22.0
torch>=2.0.0
fastapi
uvicorn
soundfile
numpy
```

### Mac Client
```txt
mcp
httpx
sounddevice
soundfile
numpy
```

## Security Considerations

- [ ] Run TTS server on private network only (not exposed to internet)
- [ ] Use SSH tunnel or VPN for remote access if needed
- [ ] Consider adding API key authentication to TTS server
- [ ] Firewall rules to restrict access to TTS server port

## Open Questions

1. **Latency** — What's the acceptable latency for real-time speech? May need to optimize or use streaming.
2. ~~**Voice Quality** — How much reference audio is optimal for best voice clone quality?~~ ✅ **Resolved:** Zero-shot needs 15-30s; fine-tuning needs 5-30 min for production quality.
3. **Model Selection** — Need to test XTTS-v2 vs GLM-TTS vs OpenVoice for quality comparison
4. **Streaming** — Should we implement streaming TTS for longer texts?

## Next Steps

### Core Implementation
- [ ] Set up Linux desktop with NVIDIA GPU and CUDA
- [ ] Install and test XTTS-v2 locally
- [ ] Record voice reference audio samples (15-30 seconds for zero-shot)
- [ ] Test voice cloning quality (zero-shot baseline)
- [ ] Build FastAPI server in Docker container
- [ ] Build MCP server for Cursor
- [ ] End-to-end integration test

### Optional: Fine-Tuning for Production Quality
- [ ] Record extended training data (5-30 minutes)
- [ ] Create transcriptions for all clips
- [ ] Preprocess and split dataset
- [ ] Run fine-tuning training (2-8 hours)
- [ ] Evaluate fine-tuned vs zero-shot quality
- [ ] Deploy fine-tuned model

## References

- [Qwen3-TTS-VC-VoiceDesign Blog](https://qwen.ai/blog?id=qwen3-tts-vc-voicedesign) ⭐ **Recommended**
- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [GLM-TTS](https://glm-tts.com/)
- [OpenVoice Framework](https://www.emergentmind.com/topics/openvoice-framework)
- [BentoML: Exploring Open Source TTS Models](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [TTS With Instant Voice Cloning: 5 Local Models Compared (YouTube)](https://www.youtube.com/watch?v=led0nCZHVkQ)
- [MCP Server Documentation](https://modelcontextprotocol.io/)

---

## Related

- [[voice-cloning-tts-mcp-server-implementation]] — Detailed implementation plan with step-by-step instructions
- [[2026-01-07]] — Initial design proposal created
- [[2026-01-09]] — Added comprehensive fine-tuning section


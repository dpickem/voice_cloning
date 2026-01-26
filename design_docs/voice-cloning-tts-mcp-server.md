# Voice Cloning TTS with MCP Server Integration

**Created:** 2026-01-07  
**Updated:** 2026-01-26  
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
│  │  │  Voice Clone   │    │  TTS Model (Qwen3-TTS/XTTS-v2) │    │   │
│  │  │  Reference     │───▶│     - Text → Speech            │    │   │
│  │  │  Audio Files   │    │     - Voice Embedding          │    │   │
│  │  └────────────────┘    └────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  GPU: NVIDIA RTX (for inference acceleration)                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Open Source TTS Models (as of January 2026)

### ⭐⭐ NEW: Qwen3-TTS (Open Source — SOTA Voice Cloning)

| Attribute | Details |
|-----------|---------|
| **Model** | Qwen3-TTS Series (0.6B / 1.7B) |
| **Voice Cloning** | ✅ SOTA 3-second rapid voice cloning |
| **Languages** | 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian) |
| **Features** | TTS + Voice Cloning + Voice Design + Instruction Control |
| **Streaming** | ✅ Ultra-low latency (97ms first packet) |
| **Local Hosting** | ✅ **Yes** — Fully open source under Apache 2.0 |
| **Training Data** | 5+ million hours of speech data |
| **GitHub** | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (4.8k+ stars) |
| **HuggingFace** | [Qwen/Qwen3-TTS Collection](https://huggingface.co/collections/Qwen/qwen3-tts) |
| **Paper** | [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) |
| **Blog** | [qwen.ai/blog](https://qwen.ai/blog?id=qwen3tts-0115) |

**🎉 Released January 22, 2026:** Qwen3-TTS is now **fully open source** and represents the current state-of-the-art in voice cloning TTS. Unlike the earlier Qwen3-TTS-VC-VoiceDesign (API-only), this release includes full model weights for self-hosting.

#### Available Models

| Model | Size | Features | Streaming | Instruction Control |
|-------|------|----------|-----------|---------------------|
| **Qwen3-TTS-12Hz-1.7B-Base** | 1.7B | 3-second voice clone, fine-tuning base | ✅ | — |
| **Qwen3-TTS-12Hz-1.7B-CustomVoice** | 1.7B | 9 premium voices, style control | ✅ | ✅ |
| **Qwen3-TTS-12Hz-1.7B-VoiceDesign** | 1.7B | Create voices from descriptions | ✅ | ✅ |
| **Qwen3-TTS-12Hz-0.6B-Base** | 0.6B | Lightweight voice clone | ✅ | — |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** | 0.6B | Lightweight with premium voices | ✅ | — |
| **Qwen3-TTS-Tokenizer-12Hz** | — | Speech tokenizer for encoding/decoding | — | — |

#### Key Technical Innovations

1. **Dual-Track LM Architecture** — Enables real-time streaming synthesis with a single model supporting both streaming and non-streaming generation
2. **Qwen3-TTS-Tokenizer-12Hz** — 12.5 Hz, 16-layer multi-codebook design achieving extreme bitrate reduction
3. **Lightweight Causal ConvNet** — Enables 97ms first-packet emission for ultra-low latency
4. **Intelligent Text Understanding** — Adaptive control of tone, speaking rate, and emotional expression based on instructions and text semantics

#### Benchmark Performance (SOTA)

| Benchmark | Qwen3-TTS-12Hz-1.7B | Best Competitor | Metric |
|-----------|---------------------|-----------------|--------|
| Seed-TTS (Chinese) | **0.77%** | CosyVoice 3: 0.71% | WER ↓ |
| Seed-TTS (English) | **1.24%** | CosyVoice 3: 1.45% | WER ↓ |
| Speaker Similarity (Avg) | **0.79+** | MiniMax: 0.75 | SIM ↑ |
| Long Speech (EN) | **1.22%** | Higgs-Audio-v2: 6.92% | WER ↓ |

**Pros:**
- ✅ **SOTA voice cloning** — Best-in-class on multiple benchmarks
- ✅ **Ultra-fast cloning** — Only 3 seconds of reference audio needed (vs 6s for XTTS-v2)
- ✅ **Ultra-low latency** — 97ms first packet for real-time applications
- ✅ **Voice Design** — Create entirely new voices from text descriptions
- ✅ **Instruction Control** — Natural language control over emotion, style, prosody
- ✅ **Fully self-hostable** — Open source under Apache 2.0
- ✅ **vLLM support** — Day-0 support for optimized inference
- ✅ **Fine-tuning support** — Full fine-tuning documentation included
- ✅ **10 languages** with strong multilingual performance

**Cons:**
- Newer model (released Jan 2026) — less community testing than XTTS-v2
- Larger model sizes (0.6B-1.7B) may require more VRAM than XTTS-v2
- Requires Python 3.12 and FlashAttention 2 for optimal performance

#### Quick Start

```bash
# Install
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
pip install -U flash-attn --no-build-isolation
```

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Voice cloning with just 3 seconds of reference audio
wavs, sr = model.generate_voice_clone(
    text="Hello, this is my cloned voice speaking.",
    language="English",
    ref_audio="my_voice_reference.wav",
    ref_text="The transcript of the reference audio.",
)
sf.write("output.wav", wavs[0], sr)
```

#### Voice Design (Create New Voices from Descriptions)

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Create a voice from a natural language description
wavs, sr = model.generate_voice_design(
    text="Welcome to our product demo!",
    language="English",
    instruct="Professional male voice, warm and confident, mid-30s, clear American accent",
)
sf.write("designed_voice.wav", wavs[0], sr)
```

---

### ⭐ Alternative: XTTS-v2 (Coqui TTS) — Mature & Well-Tested

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

## Speech-to-Text (ASR) Models

For complete voice agent pipelines, you may also need Speech-to-Text (ASR/STT) capabilities. These models transcribe spoken audio to text, which can then be processed by an LLM before generating a TTS response.

### ⭐ NVIDIA Nemotron Speech ASR (0.6B) — Low-Latency Streaming

| Attribute | Details |
|-----------|---------|
| **Model** | Nemotron Speech ASR |
| **Parameters** | 600M (0.6B) |
| **Architecture** | Cache-aware FastConformer encoder + RNNT decoder |
| **Audio Input** | 16 kHz mono, minimum 80ms chunks |
| **Streaming** | ✅ True streaming with cache-aware design (no overlapping windows) |
| **WER** | ~7.2–7.8% across standard benchmarks |
| **Local Hosting** | ✅ **Yes** — Open weights, self-hostable |
| **License** | NVIDIA Permissive Open Model License |
| **HuggingFace** | [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) |

**Key Innovation — Cache-Aware Streaming:**
Traditional streaming ASR uses overlapping windows that reprocess audio, wasting compute and causing latency drift. Nemotron Speech ASR keeps a cache of encoder states, processing each chunk only once:
- Non-overlapping frame processing (linear scaling with audio length)
- Predictable memory growth
- Stable latency under load (critical for voice agents)

**Configurable Chunk Sizes (Latency vs. Accuracy):**

| Chunk Size | Latency | WER | Best For |
|------------|---------|-----|----------|
| ~80ms | Ultra-low | ~7.8% | Aggressive interruption handling |
| ~160ms | Low | ~7.84% | Real-time voice agents |
| ~560ms | Medium | ~7.22% | Balanced transcription |
| ~1.12s | Higher | ~7.16% | Accuracy-focused workflows |

> **Note:** Chunk size is configurable at inference time via `att_context_size` — no retraining required.

**Concurrency & Throughput:**

| GPU | Concurrent Streams | Notes |
|-----|-------------------|-------|
| **H100** | ~560 streams | At 320ms chunk, ~3x baseline |
| **RTX A5000** | 5x+ baseline | Excellent for workstations |
| **DGX B200** | ~2x baseline | Data center scale |

**Voice Agent Latency (Full Pipeline):**
With Nemotron Speech ASR + Nemotron 3 Nano 30B + Magpie TTS on RTX 5090:
- Median time to final transcription: ~24ms
- Server-side voice-to-voice latency: ~500ms

**Pros:**
- ✅ **True streaming** — cache-aware design, no overlapping windows
- ✅ **Low latency** — sub-200ms end-to-end possible
- ✅ **High concurrency** — 3-5x more streams than baseline
- ✅ **Configurable** — trade latency vs. accuracy at runtime
- ✅ **Self-hostable** — open weights under permissive license
- ✅ **Well-documented** — trained on ~285k hours of audio

**Cons:**
- English-only (currently)
- Requires NVIDIA GPU for optimal performance
- Large training data requirement for fine-tuning

**Reference:** [NVIDIA Nemotron Speech ASR Release](https://marktechpost.com/2026/01/06/nvidia-ai-released-nemotron-speech-asr/)

---

## Fine-Tuning Models (Unsloth)

Unsloth provides optimized fine-tuning for TTS models with **1.5x faster training** and **50% less memory** using Flash Attention 2. Below are the models supported for fine-tuning via [Unsloth's TTS Fine-tuning](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning).

> **Why Fine-Tune?** Zero-shot voice cloning captures the general tone and timbre of a speaker's voice, but loses details like speaking speed, phrasing, vocal quirks, and the subtleties of prosody. Fine-tuning delivers far more accurate and realistic voice replication.

### ⭐ Orpheus-TTS (3B) — Recommended for Fine-Tuning

| Attribute | Details |
|-----------|---------|
| **Model** | Orpheus-TTS 3B |
| **Parameters** | 3B (Llama-based) |
| **Pre-training** | Fine-tuned on 8 professional voice actors |
| **Voice Consistency** | ✅ Built-in (no audio context required) |
| **Special Features** | Emotional cues (`<laugh>`, `<sigh>`, `<cough>`, etc.) |
| **Export** | llama.cpp compatible (GGUF) |
| **HuggingFace** | `unsloth/orpheus-3b-0.1-ft` |

**Pros:**
- ✅ Built-in voice consistency from pre-training
- ✅ Supports emotional expression tags in text
- ✅ Can export to GGUF via llama.cpp
- ✅ Better results out of the box (less compute needed)
- ✅ Realistic speech with natural prosody

**Cons:**
- Larger model (3B) = higher latency
- Requires more VRAM than smaller models

**Example Usage:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/orpheus-3b-0.1-ft",
    max_seq_length=2048,
    load_in_4bit=False,  # Use 4bit for less VRAM
)
```

---

### Sesame-CSM (1B) — Base Model

| Attribute | Details |
|-----------|---------|
| **Model** | Sesame-CSM 1B |
| **Parameters** | 1B |
| **Pre-training** | Base model (not fine-tuned) |
| **Voice Consistency** | ⚠️ Requires audio context per speaker |
| **Special Features** | Flexible, works with any voice via context |
| **HuggingFace** | Available via Unsloth |

**Pros:**
- ✅ Smaller model = lower latency
- ✅ More flexible (can adapt to any voice with context)
- ✅ Good for resource-constrained deployments

**Cons:**
- ⚠️ Requires audio context for each speaker for consistency
- ⚠️ More compute needed for fine-tuning (base model)
- Voice may vary across generations without context

**When to Use:** Choose CSM when you need a smaller model and can provide audio context, or when flexibility across different voices is more important than built-in consistency.

---

### Other Unsloth-Supported TTS Models

| Model | Parameters | Notes |
|-------|------------|-------|
| **Spark-TTS** | 0.5B | Ultra-lightweight, good for edge deployment |
| **Llasa-TTS** | 1B | Alternative 1B option |
| **Oute-TTS** | 1B | Alternative 1B option |
| **Dia-TTS** | Varies | Also supported (transformers-compatible) |
| **Moshi** | Varies | Also supported (transformers-compatible) |

> **Note:** Unsloth supports **any** `transformers`-compatible TTS model. Even without an official notebook, you can fine-tune other models.

---

### Unsloth Fine-Tuning Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU VRAM** | 8GB (4-bit) | 16GB+ (16-bit/FFT) |
| **Training** | LoRA 4-bit | LoRA 16-bit or Full Fine-tuning |
| **Dataset** | Audio clips + transcripts | ~3 hours recommended |
| **Audio Sample Rate** | 24 kHz | 24 kHz (Orpheus requirement) |

**Training Options:**
- `load_in_4bit = True` — 4-bit quantized (less VRAM)
- `load_in_8bit = True` — 8-bit quantized (balanced)
- `load_in_4bit = False` — 16-bit LoRA (higher quality)
- `full_finetuning = True` — Full fine-tuning (best quality, most VRAM)

**Example Training Config:**
```python
from transformers import TrainingArguments, Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Set num_train_epochs=1 for full run
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir="outputs",
    ),
)
trainer.fit()
```

---

### Emotional Expression Tags (Orpheus)

Orpheus supports special tags for emotional expressions in transcripts:

| Tag | Expression |
|-----|------------|
| `<laugh>` | Laughter |
| `<chuckle>` | Light chuckle |
| `<sigh>` | Sigh |
| `<cough>` | Cough |
| `<sniffle>` | Sniffle |
| `<groan>` | Groan |
| `<yawn>` | Yawn |
| `<gasp>` | Gasp |

**Example Transcript:**
```
"I missed you <laugh> so much! <sigh> It's been too long."
```

> **Tip:** Use the [MrDragonFox/Elise](https://huggingface.co/datasets/MrDragonFox/Elise) dataset as a reference — it includes emotion tags annotated in the transcripts.

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

**For Qwen3-TTS (Recommended — SOTA):**
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Load the base model for voice cloning
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Clone voice with just 3 seconds of reference audio
wavs, sr = model.generate_voice_clone(
    text="Hello, this is my cloned voice!",
    language="English",
    ref_audio="my_voice_reference.wav",
    ref_text="The transcript of my reference audio.",  # Optional but improves quality
)
sf.write("output.wav", wavs[0], sr)

# For multiple generations, create a reusable prompt to avoid recomputing
voice_prompt = model.create_voice_clone_prompt(
    ref_audio="my_voice_reference.wav",
    ref_text="The transcript of my reference audio.",
)

# Reuse the prompt for batch generation
wavs, sr = model.generate_voice_clone(
    text=["First sentence.", "Second sentence.", "Third sentence."],
    language=["English", "English", "English"],
    voice_clone_prompt=voice_prompt,
)
for i, wav in enumerate(wavs):
    sf.write(f"output_{i}.wav", wav, sr)
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
| **Qwen3-TTS** | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian (10 languages) |
| **XTTS-v2** | English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi (17 languages) |
| **GLM-TTS** | Chinese, English |
| **OpenVoice** | Cross-lingual (any via IPA phoneme alignment) |

---

## Implementation Plan

### Phase 1: Linux Server Setup

1. **Install TTS Model (Qwen3-TTS — Recommended)**
   ```bash
   # Create isolated environment
   conda create -n qwen3-tts python=3.12 -y
   conda activate qwen3-tts
   
   # Install qwen-tts package
   pip install -U qwen-tts
   
   # Install FlashAttention 2 for memory efficiency
   pip install -U flash-attn --no-build-isolation
   ```

   **Alternative: XTTS-v2 (Coqui TTS)**
   ```bash
   pip install TTS
   # or clone from GitHub for latest version
   git clone https://github.com/coqui-ai/TTS.git
   cd TTS && pip install -e .
   ```

2. **Record Voice Reference Audio**
   - Record 3-10 seconds of clear speech (Qwen3-TTS only needs 3 seconds!)
   - Diverse sentences covering different phonemes
   - High quality audio (24kHz+, minimal background noise)

3. **Test Voice Cloning Locally**

   **With Qwen3-TTS (Recommended):**
   ```python
   import torch
   import soundfile as sf
   from qwen_tts import Qwen3TTSModel
   
   model = Qwen3TTSModel.from_pretrained(
       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
       device_map="cuda:0",
       dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
   )
   
   wavs, sr = model.generate_voice_clone(
       text="Hello, this is my cloned voice speaking.",
       language="English",
       ref_audio="my_voice_reference.wav",
       ref_text="Transcript of my reference audio.",
   )
   sf.write("output.wav", wavs[0], sr)
   ```

   **With XTTS-v2:**
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

1. ~~**Latency** — What's the acceptable latency for real-time speech?~~ ✅ **Resolved:** Qwen3-TTS achieves 97ms first-packet latency with streaming support.
2. ~~**Voice Quality** — How much reference audio is optimal for best voice clone quality?~~ ✅ **Resolved:** Qwen3-TTS needs only 3 seconds; XTTS-v2 needs 6+ seconds; fine-tuning needs 5-30 min for production quality.
3. ~~**Model Selection** — Need to test XTTS-v2 vs GLM-TTS vs OpenVoice for quality comparison~~ ✅ **Resolved:** Qwen3-TTS is now SOTA on benchmarks; recommended as primary choice with XTTS-v2 as mature fallback.
4. ~~**Streaming** — Should we implement streaming TTS for longer texts?~~ ✅ **Resolved:** Qwen3-TTS has built-in streaming support with dual-track architecture.

## Next Steps

### Core Implementation
- [ ] Set up Linux desktop with NVIDIA GPU and CUDA
- [ ] Install and test Qwen3-TTS locally (recommended) or XTTS-v2 as fallback
- [ ] Record voice reference audio samples (3-10 seconds for Qwen3-TTS zero-shot)
- [ ] Test voice cloning quality (zero-shot baseline)
- [ ] Build FastAPI server in Docker container
- [ ] Build MCP server for Cursor
- [ ] End-to-end integration test

### Optional: Fine-Tuning for Production Quality
- [ ] Record extended training data (5-30 minutes)
- [ ] Create transcriptions for all clips
- [ ] Preprocess and split dataset
- [ ] Run fine-tuning training using [Qwen3-TTS fine-tuning guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [ ] Evaluate fine-tuned vs zero-shot quality
- [ ] Deploy fine-tuned model

## References

### Qwen3-TTS (SOTA — Recommended)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS) ⭐ **Open Source Repository**
- [Qwen3-TTS HuggingFace Collection](https://huggingface.co/collections/Qwen/qwen3-tts) ⭐ **Model Weights**
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621) — Academic paper with architecture details
- [Qwen3-TTS Blog](https://qwen.ai/blog?id=qwen3tts-0115) — Official release announcement
- [Qwen3-TTS Demo (HuggingFace Spaces)](https://huggingface.co/spaces/Qwen/Qwen3-TTS) — Interactive demo
- [vLLM-Omni Qwen3-TTS Support](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts) — Optimized inference examples

### Other TTS Models
- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS) — XTTS-v2 repository
- [GLM-TTS](https://glm-tts.com/)
- [OpenVoice Framework](https://www.emergentmind.com/topics/openvoice-framework)
- [BentoML: Exploring Open Source TTS Models](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [TTS With Instant Voice Cloning: 5 Local Models Compared (YouTube)](https://www.youtube.com/watch?v=led0nCZHVkQ)

### Fine-Tuning Resources
- [Qwen3-TTS Fine-tuning Guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning) ⭐ **Qwen3-TTS Fine-tuning**
- [Unsloth TTS Fine-tuning Guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) — Orpheus-TTS, Sesame-CSM fine-tuning
- [MrDragonFox/Elise Dataset](https://huggingface.co/datasets/MrDragonFox/Elise) — Example training dataset with emotion tags

### Speech Recognition (ASR)
- [NVIDIA Nemotron Speech ASR](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) — Low-latency streaming ASR model

### Integration
- [MCP Server Documentation](https://modelcontextprotocol.io/)

---

## Related

- [[voice-cloning-tts-mcp-server-implementation]] — Detailed implementation plan with step-by-step instructions
- [[2026-01-07]] — Initial design proposal created
- [[2026-01-09]] — Added comprehensive fine-tuning section
- [[2026-01-13]] — Added Unsloth fine-tuning models (Orpheus-TTS, Sesame-CSM, Spark-TTS, etc.)
- [[2026-01-15]] — Added NVIDIA Nemotron Speech ASR for low-latency streaming STT
- [[2026-01-26]] — **Major update:** Added Qwen3-TTS as new SOTA open-source model (released Jan 22, 2026)


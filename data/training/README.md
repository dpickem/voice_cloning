# Voice Training Dataset

This dataset contains 100 diverse text snippets designed for recording voice samples to fine-tune an XTTS-v2 voice cloning model.

## Purpose

Zero-shot voice cloning (using ~15-30 seconds of reference audio) produces good results, but fine-tuning on a larger, diverse dataset yields significantly better voice reproduction. This dataset is designed to capture the full range of vocal characteristics needed for high-fidelity voice synthesis.

## Dataset Composition

The 100 clips are strategically organized to cover different aspects of natural speech:

### Phoneme Coverage (Clips 1-10)

**Purpose:** Ensure all English phonemes are well-represented in the training data.

These clips use pangrams—sentences containing every letter of the alphabet—which naturally cover most phoneme combinations:

- "The quick brown fox jumps over the lazy dog"
- "Pack my box with five dozen liquor jugs"
- "Sphinx of black quartz, judge my vow"

**Why it matters:** Voice models need exposure to all sound combinations to accurately reproduce them. Missing phonemes result in unnatural pronunciation.

### Introductions & Greetings (Clips 11-15)

**Purpose:** Capture friendly, conversational tone for common interactions.

- Self-introductions
- Greetings and pleasantries
- Collaborative language

**Why it matters:** These are high-frequency speech patterns that should sound natural and warm.

### Technical Content (Clips 16-20)

**Purpose:** Train the model on numbers, version strings, and technical terminology.

- Port numbers, memory sizes, dates
- API terminology, status codes
- Precise numerical values

**Why it matters:** Technical speech has different prosody than conversational speech. Numbers and acronyms need special attention.

### Questions (Clips 21-25)

**Purpose:** Capture the rising intonation and phrasing of interrogative sentences.

- Yes/no questions
- Open-ended questions
- Rhetorical questions

**Why it matters:** Question intonation is distinctly different from statements. The model needs examples to reproduce this correctly.

### Exclamations & Enthusiasm (Clips 26-30)

**Purpose:** Train emotional expressiveness with high-energy speech.

- Excitement and celebration
- Surprise and amazement
- Positive reactions

**Why it matters:** Emotional range makes synthesized speech sound natural rather than monotone.

### Conversational Content (Clips 31-35)

**Purpose:** Capture natural, informal speech patterns.

- Project updates and status
- Suggestions and opinions
- Agreement and acknowledgment

**Why it matters:** Day-to-day conversation has a relaxed rhythm that differs from formal speech.

### Longer Passages (Clips 36-44)

**Purpose:** Train prosody and breath patterns over extended speech.

- Multi-sentence explanations
- Technical descriptions
- Connected reasoning

**Why it matters:** Longer utterances reveal natural pausing, emphasis, and rhythm patterns that short clips don't capture.

### Thoughtful & Hesitant Speech (Clips 45-49)

**Purpose:** Capture uncertainty and reflective tones.

- Hedging language ("I'm not entirely sure...")
- Thinking aloud ("Hmm, that's interesting...")
- Qualified agreement

**Why it matters:** Natural speech includes hesitation and reflection. Models trained only on confident speech sound artificially certain.

### Development & Code Context (Clips 50-54)

**Purpose:** Technical jargon and developer-specific language.

- Code review terminology
- Testing and deployment language
- Git and CI/CD references

**Why it matters:** Domain-specific vocabulary benefits from dedicated training examples.

### Polite Responses (Clips 55-59)

**Purpose:** Capture helpful, accommodating tone.

- Offers of assistance
- Acknowledgments
- Confirmations

**Why it matters:** Polite speech has subtle prosodic features (softer onset, warmer tone) worth capturing.

### General Topics (Clips 60-64)

**Purpose:** Casual, non-technical conversation.

- Weather and seasons
- Personal preferences
- Hobbies and interests

**Why it matters:** Variety in subject matter prevents the model from overfitting to technical content.

### Apologetic & Negative News (Clips 65-69)

**Purpose:** Train delivery of disappointing or regretful content.

- Deadline misses
- Feature limitations
- Service disruptions

**Why it matters:** Delivering bad news has distinct prosodic features—lower energy, measured pace, empathetic tone.

### Transition Phrases (Clips 70-79)

**Purpose:** Discourse markers and logical connectors.

- "On the other hand..."
- "In conclusion..."
- "For example..."
- "As a result..."

**Why it matters:** These phrases have predictable intonation patterns that signal structure in speech.

### Instructions & Commands (Clips 80-84)

**Purpose:** Directive speech patterns.

- Reminders and warnings
- Step-by-step guidance
- Cautionary advice

**Why it matters:** Instructional speech uses different emphasis patterns than conversational speech.

### Numbers & Metrics (Clips 85-89)

**Purpose:** Additional numerical content with units and measurements.

- Function parameters
- Array indices
- Performance metrics

**Why it matters:** Reinforces numerical pronunciation, which can be challenging for TTS models.

### Optimistic & Closing (Clips 90-100)

**Purpose:** Forward-looking, inspirational content.

- Technology enthusiasm
- Growth mindset
- Motivational statements
- Closing remarks

**Why it matters:** Ends the dataset on an uplifting note and provides examples of aspirational speech.

## Recording Guidelines

### Environment
- Quiet room with minimal echo
- No background noise (AC, fans, traffic)
- Soft furnishings to dampen reflections

### Equipment
- USB condenser microphone recommended
- Pop filter for plosive sounds (p, b, t)
- Consistent 6-12 inch distance from mic
- Microphone stand (avoid handling noise)

### Technique
- Speak naturally at your normal pace
- Don't over-enunciate or change your voice
- Take breaks every 10-15 minutes
- Keep a glass of water nearby
- Re-record clips with errors or background noise

### File Format
- Save as WAV (32-bit float or 16-bit PCM)
- Sample rate: 44100 Hz (or 48000 Hz) — higher is better for source recordings
- Mono channel
- Filename: `clip_XXX.wav` (matching metadata.csv)

> **Note:** Record at high quality (44.1 kHz, 32-bit). The preprocessing pipeline will automatically resample to 22050 Hz for XTTS-v2 training. Higher-quality source recordings produce better results after noise reduction and normalization.

## Recording with Audacity on macOS

[Audacity](https://www.audacityteam.org/) is a free, open-source audio editor that works well for recording voice training data.

### Installation

```bash
# Install via Homebrew
brew install --cask audacity

# Or download from https://www.audacityteam.org/download/mac/
```

### Initial Setup

1. **Open Audacity** and go to **Audacity → Settings** (or press `Cmd + ,`)

2. **Audio Settings** (under Devices):
   - **Host:** Core Audio
   - **Recording Device:** Your USB microphone (or built-in if no external mic)
   - **Recording Channels:** 1 (Mono)

3. **Quality Settings** (under Quality):
   - **Default Sample Rate:** 44100 Hz (high quality source; preprocessing will resample)
   - **Default Sample Format:** 32-bit float (provides editing headroom)

4. **Recording Settings** (under Recording):
   - ✅ **Sound Activated Recording:** OFF (record manually)
   - **Audio to buffer:** 100 ms

5. **Click OK** to save settings

### Project Setup

Before recording your first clip:

1. **Set Project Rate:** Look at bottom-left of Audacity window, ensure it shows **44100** Hz
2. **Set to Mono:** Click the track dropdown (after recording) → **Split Stereo to Mono** → delete one channel, or just record in mono from the start

### Recording Workflow

#### Option A: Record Each Clip Individually (Recommended for Beginners)

1. **Open `metadata.csv`** in a text editor or spreadsheet alongside Audacity
2. **Press R** (or click Record button) to start recording
3. **Read the text** for clip_001 naturally
4. **Press Space** to stop recording
5. **Listen back** with Space to check quality
6. **Export:**
   - `File → Export → Export as WAV`
   - Navigate to `training_data/wavs/`
   - Filename: `clip_001.wav`
   - Encoding: **Signed 16-bit PCM**
7. **Clear the track:** `Cmd + A` (select all) → `Delete`
8. **Repeat** for clips 002-100

#### Option B: Batch Recording (Faster, More Efficient)

Record multiple clips in one session, then split:

1. **Record continuously:**
   - Press **R** to start
   - Read clips 001-010, leaving **2-3 seconds of silence** between each
   - Press **Space** to stop

2. **Add labels for splitting:**
   - Click at the start of clip_001
   - Press `Cmd + B` to add a label, type `clip_001`
   - Click at the start of clip_002
   - Press `Cmd + B`, type `clip_002`
   - Repeat for all clips in the recording

3. **Export multiple files at once:**
   - `File → Export → Export Multiple...`
   - **Format:** WAV
   - **Split files based on:** Labels
   - **Name files:** Using label/track name
   - **Export location:** `training_data/wavs/`
   - Click **Export**

### Export Settings

When exporting WAV files, use these settings:

| Setting | Value |
|---------|-------|
| **Format** | WAV (Microsoft) |
| **Encoding** | 32-bit float (or Signed 16-bit PCM) |
| **Sample Rate** | 44100 Hz |
| **Channels** | Mono |

> **Why high quality?** The preprocessing pipeline (`preprocess_training_data.py`) will resample to 22050 Hz and normalize for XTTS-v2. Recording at higher quality preserves more detail for noise reduction and ensures the best possible source material.

### Audacity Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Record | `R` |
| Stop | `Space` |
| Play/Pause | `Space` |
| Play from start | `Shift + Space` |
| Select all | `Cmd + A` |
| Delete selection | `Delete` |
| Undo | `Cmd + Z` |
| Add label | `Cmd + B` |
| Zoom in | `Cmd + 1` |
| Zoom out | `Cmd + 3` |
| Fit to window | `Cmd + F` |
| Export audio | `Cmd + Shift + E` |

### Quality Check Before Export

Before exporting each clip:

1. **Visual inspection:**
   - Waveform should show clear speech with minimal background noise
   - Amplitude should be consistent (not too quiet or clipping)
   - No long silences at start/end (trim if needed)

2. **Normalize volume** (optional but recommended):
   - Select the audio (`Cmd + A`)
   - `Effect → Volume and Compression → Normalize...`
   - Set to **-1.0 dB**
   - Click **Apply**

3. **Trim silence:**
   - Select silent portions at the start/end
   - Press `Delete`
   - Leave ~0.1-0.2 seconds of silence at each end

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **No audio input** | Check System Preferences → Sound → Input; ensure mic is selected |
| **Audio too quiet** | Increase input volume in System Preferences or Audacity's input slider |
| **Clipping (distortion)** | Reduce input volume; re-record affected clips |
| **Background noise** | Use `Effect → Noise Reduction` (record 2s of silence first as noise profile) |
| **Echo/reverb** | Record in a smaller room with soft furnishings; hang blankets if needed |
| **Plosives (p/b pops)** | Use a pop filter; position mic slightly off-axis from mouth |

### Noise Reduction (Optional)

If your recordings have consistent background noise:

1. **Record 2-3 seconds of silence** (just room noise, no speaking)
2. Select the silent portion
3. `Effect → Noise Reduction and Repair → Noise Reduction...`
4. Click **Get Noise Profile**
5. Select your entire recording (`Cmd + A`)
6. `Effect → Noise Reduction and Repair → Noise Reduction...`
7. Set:
   - **Noise reduction (dB):** 12
   - **Sensitivity:** 6
   - **Frequency smoothing:** 3
8. Click **OK**

> **Note:** The preprocessing pipeline will also apply noise reduction, so this step is optional.

### Session Tips

- **Batch your recording:** Record 10-20 clips per session
- **Take breaks:** Rest your voice every 15-20 minutes
- **Stay hydrated:** Keep water nearby
- **Consistent positioning:** Mark your mic position to maintain consistent distance
- **Monitor levels:** Watch the recording meter—aim for peaks around -6 to -12 dB
- **Save the project:** `File → Save Project` to keep undo history in case you need to re-export

## Directory Structure

```
training_data/
├── README.md              # This file
├── metadata.csv           # 100 clips: filename|transcription
├── metadata.csv.example   # Format documentation
├── wavs/                  # Raw recorded audio clips
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
└── processed/             # Preprocessed audio (after running preprocessing)
    ├── clip_001.wav
    └── ...
```

## Processing Pipeline

1. **Record** all 100 clips into `wavs/` directory
2. **Preprocess** using Docker container:
   ```bash
   docker compose run --rm voice-tts python preprocess_training_data.py \
       --input-dir /app/training_data/wavs \
       --output-dir /app/training_data/processed \
       --metadata /app/training_data/metadata.csv
   ```
3. **Split** into train/eval sets:
   ```bash
   python split_dataset.py --metadata training_data/metadata.csv --train-ratio 0.9
   ```
4. **Fine-tune** the model:
   ```bash
   docker compose run --rm voice-tts python finetune_xtts.py --config finetune_config.json
   ```

## Expected Results

| Dataset Size | Training Time | Expected Quality |
|--------------|---------------|------------------|
| 50 clips (~5 min audio) | 2-3 hours | Noticeable improvement |
| 100 clips (~10 min audio) | 4-6 hours | Significant improvement |
| 150+ clips (~15+ min audio) | 6-8+ hours | Near-perfect reproduction |

## References

- [Implementation Plan](../../implementation_plans/voice-cloning-tts-mcp-server-implementation.md)
- [Design Document](../../design_docs/voice-cloning-tts-mcp-server.md)
- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)

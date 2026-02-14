# EdgeTutor AI — Architecture

## System Overview

EdgeTutor AI is a modular, offline AI tutoring system designed for NVIDIA Jetson
devices. All processing happens locally — no cloud calls, no telemetry.

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web UI                           │
│  (Chat, Push-to-Talk, Camera, Settings, Quiz Mode)          │
│                    localhost:7860                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Tutor Orchestrator                          │
│  (Routes inputs → modules → assembles response)             │
└──┬──────────┬──────────┬──────────┬──────────┬──────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│ STT  │ │ TTS  │ │ OCR  │ │ LLM  │ │   RAG    │
│      │ │      │ │      │ │      │ │          │
│faster│ │Piper │ │Tess- │ │llama │ │FAISS +   │
│whisp │ │ TTS  │ │eract │ │.cpp  │ │sentence- │
│er    │ │      │ │      │ │(GGUF)│ │transform │
└──────┘ └──────┘ └──────┘ └──────┘ └──────────┘
   ▲          ▲          ▲          ▲          ▲
   │          │          │          │          │
  CUDA      ONNX       CPU       CUDA      CPU
```

## Module Details

### 1. Frontend (Gradio UI)
- **File**: `edgetutor/app/ui.py`
- **Tech**: Gradio 4.x Blocks
- **Serves**: `http://0.0.0.0:7860` (LAN accessible)
- **Features**:
  - Chat interface with message history
  - Push-to-talk microphone input
  - Camera capture / image upload
  - Audio playback of TTS responses
  - Settings panel (age, subject, parent mode, quiz mode)
  - Module health status bar
  - Latency metrics per response

### 2. Tutor Orchestrator
- **File**: `edgetutor/app/orchestrator.py`
- **Role**: Central coordinator
- **Pipeline**:
  1. Receive `TutorRequest` (text, audio, image, settings)
  2. STT: Transcribe audio → text (if audio provided)
  3. OCR: Extract text from image (if image provided)
  4. RAG: Retrieve relevant content chunks
  5. Prompt: Build system + user prompt from templates
  6. LLM: Generate response (streaming or batch)
  7. TTS: Synthesize response audio
  8. Return `TutorResponse`
- **Graceful degradation**: Any module can fail without crashing the app

### 3. LLM Backend
- **File**: `edgetutor/core/llm.py`
- **Tech**: llama-cpp-python (GGUF format)
- **Default model**: Phi-3 Mini 4K Instruct Q4 (3.8B params, ~2.3 GB)
- **Features**:
  - GPU offloading (configurable layers)
  - Streaming token output
  - System prompt injection
  - Pre/post safety filtering
- **Config**: `LLM_MODEL_PATH`, `LLM_N_GPU_LAYERS`, `LLM_CONTEXT_SIZE`

### 4. Prompt System
- **File**: `edgetutor/core/prompts.py`
- **Design**: Composable templates
- **Components**:
  - Age tone (7/10/16) — controls vocabulary and depth
  - Subject instruction (math/reading/science/general)
  - Socratic mode (on by default, off in parent mode)
  - Safety rules (always included unless explicitly disabled)
  - Vision/OCR template (wraps extracted text)
  - Quiz generation template
  - RAG context template

### 5. Safety Layer
- **File**: `edgetutor/core/safety.py`
- **Input filtering**: Regex patterns for violence, weapons, self-harm,
  drugs, explicit content, hate speech
- **Output scrubbing**: Pattern matching on LLM output
- **Redirect**: Calm, encouraging message offering safe alternatives
- **Design**: Lightweight and fast; not meant to be bulletproof but catches
  common unsafe queries

### 6. Speech-to-Text (STT)
- **File**: `edgetutor/audio/stt.py`
- **Tech**: faster-whisper (CTranslate2)
- **Default model**: Whisper Small (with CUDA)
- **Features**:
  - Voice Activity Detection (VAD) to skip silence
  - File and numpy array input
  - CPU fallback if CUDA fails
- **Config**: `STT_MODEL_SIZE`, `STT_DEVICE`, `STT_COMPUTE_TYPE`

### 7. Text-to-Speech (TTS)
- **File**: `edgetutor/audio/tts.py`
- **Tech**: Piper TTS (ONNX-based VITS)
- **Default voice**: en_US-lessac-medium
- **Features**:
  - WAV synthesis to bytes or file
  - Gradio-compatible `(sample_rate, numpy_array)` output
- **Config**: `TTS_VOICE_MODEL`, `TTS_ENABLED`

### 8. Vision / OCR
- **File**: `edgetutor/vision/ocr.py`
- **Tech**: Tesseract OCR
- **Features**:
  - Text extraction from images (PIL, numpy, file path)
  - Image preprocessing (contrast, sharpening)
  - Confidence scoring
  - Math detection heuristics:
    - Equations, fractions, exponents
    - Variables, sqrt, comparison operators
    - Math keywords (solve, calculate, etc.)
- **Config**: `OCR_ENGINE`, `OCR_LANGUAGE`

### 9. RAG (Retrieval-Augmented Generation)
- **File**: `edgetutor/core/rag.py`
- **Tech**: FAISS (CPU) + sentence-transformers
- **Default embeddings**: all-MiniLM-L6-v2
- **Features**:
  - Document ingestion (TXT, MD, PDF)
  - Chunking with configurable size
  - Cosine similarity search
  - Index persistence to disk
- **Config**: `RAG_CONTENT_DIR`, `RAG_EMBEDDING_MODEL`, `RAG_TOP_K`

### 10. Jetson Hardware Detection
- **File**: `edgetutor/core/jetson.py`
- **Features**:
  - Auto-detect board via `/proc/device-tree/model`
  - RAM and GPU memory detection
  - Profile-based recommendations for each board
  - Tiered model fallback based on available memory

## Configuration

All configuration flows through `edgetutor/core/settings.py`:
- **Source**: `.env` file + environment variables
- **Library**: pydantic-settings
- **Runtime overrides**: Age mode, subject mode, parent mode, quiz mode
  can be changed from the UI without restart

## Data Flow

```
User Input (text / voice / image)
    │
    ├─ Voice ──→ STT (faster-whisper) ──→ text
    │
    ├─ Image ──→ OCR (Tesseract) ──→ extracted text + math detection
    │
    ├─ Text ──→ Safety Input Check
    │              │
    │              ├─ BLOCKED → return redirect message
    │              │
    │              └─ SAFE → continue
    │
    ├─ [Query] ──→ RAG retrieval (FAISS) ──→ context chunks
    │
    ├─ Build prompt (system + user + context)
    │
    ├─ LLM inference (llama.cpp) ──→ response text
    │
    ├─ Safety Output Check ──→ scrubbed response
    │
    └─ TTS (Piper) ──→ audio response

Response (text + audio + metadata)
```

## Performance Considerations

- **Jetson Orin Nano 8GB**: ~2-5 seconds for LLM response (Phi-3 Mini Q4)
- **STT**: ~1-3 seconds for short utterances (Whisper Small)
- **TTS**: ~0.5-2 seconds per sentence
- **OCR**: ~0.5-1 second per image
- **Total pipeline**: ~4-10 seconds end-to-end (varies with input)

Tips:
- Use `LLM_N_GPU_LAYERS=20` to offload model layers to GPU
- Use `STT_MODEL_SIZE=tiny` for faster (lower quality) transcription
- Use `LLM_CONTEXT_SIZE=1024` to reduce memory usage
- Streaming mode shows tokens as they generate

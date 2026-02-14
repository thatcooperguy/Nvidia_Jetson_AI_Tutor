# 🎓 EdgeTutor AI

**An offline AI tutor for NVIDIA Jetson — voice, vision, and chat for students.**

[![CI](https://github.com/thatcooperguy/edgetutor-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/thatcooperguy/edgetutor-ai/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)

EdgeTutor AI is a **fully offline**, **kid-safe** AI tutoring system that runs
on an NVIDIA Jetson device. Students can ask questions by typing, speaking, or
holding up a worksheet — and the tutor responds with age-appropriate,
Socratic-style guidance.

**No cloud. No telemetry. No internet required at runtime.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Chat Tutor** | Ask any school subject question via text |
| 🎤 **Voice Input** | Push-to-talk speech recognition (faster-whisper) |
| 🔊 **Voice Output** | Tutor speaks responses aloud (Piper TTS) |
| 📷 **Worksheet Scanner** | Hold up a page → OCR → explain step-by-step |
| 🔢 **Math Detection** | Auto-detects equations and walks through solutions |
| 🧒 **Age Modes** | 7 / 10 / 16 — adjusts tone and complexity |
| 📚 **Subject Modes** | Math, Reading, Science, General |
| 📝 **Quiz Mode** | Generates quizzes offline from content packs |
| 🔓 **Parent Mode** | Direct answers + advanced settings |
| 📖 **Content Packs** | RAG with your own curriculum (PDF/TXT/MD) |
| 🛡️ **Kid-Safe** | Content filtering — refuses unsafe requests |
| 🖥️ **Web UI** | Gradio interface accessible from any device on LAN |
| ⚡ **Jetson Optimized** | CUDA/GPU acceleration, adaptive model selection |

---

## 🖥️ Hardware Requirements

### Recommended
- **NVIDIA Jetson Orin Nano 8GB** Developer Kit (~$249)
- MicroSD card (128GB+) or NVMe SSD (recommended)
- USB-C power supply (5V/3A+)

### Optional (for full experience)
- USB webcam (for worksheet scanning)
- USB microphone (for voice input)
- Speaker (for TTS output)

### Also Works On
- Jetson Orin NX 8/16GB
- Jetson AGX Orin 32/64GB
- Jetson Xavier NX (limited)
- Any Linux system with NVIDIA GPU (for development)

> **📋 Full hardware guide with Amazon links**: [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md)

---

## 🚀 Quickstart

### 1. Clone & Setup (one command)

```bash
git clone https://github.com/thatcooperguy/edgetutor-ai.git
cd edgetutor-ai
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

This installs all dependencies, creates a Python venv, and offers to download
the default models (~2.5 GB total).

### 2. Start EdgeTutor (one command)

```bash
./scripts/run.sh
```

### 3. Open in Browser

On the Jetson:
```
http://localhost:7860
```

From another device on the same network:
```
http://<jetson-ip>:7860
```

---

## 📸 How It Works

### Text Chat
Type a question → EdgeTutor responds with age-appropriate guidance.

### Voice Chat
Click the microphone → speak → release → EdgeTutor transcribes and responds.

### Worksheet Scanning
1. Point your webcam at a worksheet (or upload an image)
2. Click "📄 Scan Worksheet"
3. EdgeTutor extracts text, detects math, and explains step-by-step

### Quiz Mode
Enable "Quiz Mode" in settings → ask a topic → get a 3-5 question quiz.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL_PATH` | `models/default.gguf` | Path to GGUF model |
| `LLM_N_GPU_LAYERS` | `20` | GPU layers for acceleration |
| `STT_MODEL_SIZE` | `small` | Whisper model (tiny/base/small/medium) |
| `TTS_ENABLED` | `true` | Enable/disable voice output |
| `SAFETY_ENABLED` | `true` | Kid-safe content filtering |

> **Full model guide**: [docs/MODELS.md](docs/MODELS.md)

---

## 📚 Adding Curriculum Content

Place PDF, TXT, or Markdown files in `content/` and run:

```bash
./scripts/ingest_content.sh
```

EdgeTutor will use your content to provide curriculum-grounded answers.

> **Full guide**: [docs/CONTENT_PACKS.md](docs/CONTENT_PACKS.md)

---

## 🏗️ Architecture

```
User → Gradio UI → Orchestrator → STT/OCR/RAG → LLM → TTS → Response
```

| Module | Tech | Purpose |
|--------|------|---------|
| UI | Gradio 4.x | Web interface |
| LLM | llama-cpp-python (GGUF) | Local language model |
| STT | faster-whisper | Speech-to-text |
| TTS | Piper TTS | Text-to-speech |
| OCR | Tesseract | Image text extraction |
| RAG | FAISS + sentence-transformers | Content retrieval |
| Safety | Regex filters + prompts | Kid-safe guardrails |

> **Full architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🔒 Safety & Privacy

- **Fully offline**: No data ever leaves the device
- **No telemetry**: Zero tracking, analytics, or phone-home
- **No accounts**: No login required
- **Kid-safe filters**: Input/output content filtering
- **Socratic by default**: Guides learning instead of giving answers
- **Open source**: Inspect every line of code

---

## 📂 Project Structure

```
edgetutor-ai/
├── edgetutor/
│   ├── app/            # UI (Gradio) + Orchestrator
│   ├── core/           # LLM, RAG, safety, settings, Jetson detection
│   ├── vision/         # OCR pipeline
│   ├── audio/          # STT + TTS
│   └── tests/          # Unit tests
├── scripts/            # Setup, run, ingest, model download
├── content/            # Curriculum packs (starter pack included)
├── docs/               # Architecture, models, hardware guide
├── models/             # GGUF model files (gitignored)
├── voices/             # Piper TTS voices (gitignored)
└── logs/               # Runtime logs (gitignored)
```

---

## 🧪 Development

### Run Tests
```bash
source .venv/bin/activate
pytest edgetutor/tests/ -v
```

### Lint
```bash
pip install ruff
ruff check edgetutor/
```

### Run in Debug Mode
```bash
./scripts/run.sh --debug
```

---

## 🗺️ Roadmap

See [GitHub Issues](https://github.com/thatcooperguy/edgetutor-ai/issues)
for planned features:

- [ ] Wake word detection ("Hey EdgeTutor")
- [ ] Improved math step parsing (LaTeX, handwriting)
- [ ] Multi-user profiles (student accounts)
- [ ] Spanish language mode
- [ ] Classroom fleet management (multi-device)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

---

## 📄 License

[MIT License](LICENSE) — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

Built with these excellent open-source projects:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Local LLM inference
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Speech recognition
- [Piper TTS](https://github.com/rhasspy/piper) — Text-to-speech
- [Gradio](https://gradio.app) — Web UI
- [Tesseract](https://github.com/tesseract-ocr/tesseract) — OCR
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [sentence-transformers](https://www.sbert.net) — Embeddings

---

*Made with ❤️ for students and educators everywhere.*

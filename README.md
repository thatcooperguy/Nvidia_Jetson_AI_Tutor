# 🎓 EdgeTutor AI

**Offline-first AI Tutor + AI Mentor for NVIDIA Tegra / Jetson platforms.**

[![CI](https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor/actions/workflows/ci.yml/badge.svg)](https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)

EdgeTutor AI is a **fully offline**, **kid-safe**, **hardware-aware** AI
platform that runs on NVIDIA Jetson devices (and compatible Tegra-based
systems). It's three things in one:

1. **📚 A Learning Tutor** — Students ask questions by typing, speaking, or
   holding up a worksheet. The tutor responds with age-appropriate,
   Socratic-style guidance.
2. **🧠 An AI Mentor** — Learn how GPUs, CUDA, LLMs, quantization, and edge
   AI actually work — using the real hardware sitting on your desk.
3. **🔬 A Local AI Lab** — An introduction to building and running AI systems
   on real hardware, with no cloud dependency.

**Setup online once. Inference stays local. No telemetry. No cloud calls at runtime.**

> **⚠️ Disclaimer:** This is an independent open-source project and is **not
> affiliated with or endorsed by NVIDIA**. NVIDIA, Jetson, Tegra, JetPack,
> and related marks are trademarks of NVIDIA Corporation.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Chat Tutor** | Ask any school subject question via text |
| 🧠 **AI Mentor** | Learn how GPUs, CUDA, LLMs, and AI work — on real hardware |
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
| 🖥️ **Dark Theme UI** | Clean, inspiring Gradio interface on LAN |
| ⚡ **Hardware-Aware** | Auto-detects Jetson model, scales models to fit RAM |
| 🔄 **Auto-Scaling** | Tiered fallback: picks the best model for your device |

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

### Supported Tegra / Jetson Platforms

| Platform | RAM | Status | Notes |
|----------|-----|--------|-------|
| **Orin Nano 8GB** | 8 GB | ⭐ Primary target | Best value |
| **Orin Nano 4GB** | 4 GB | ✅ Supported | Smaller models auto-selected |
| **Orin NX 8/16GB** | 8-16 GB | ✅ Supported | Great performance |
| **AGX Orin 32/64GB** | 32-64 GB | ✅ Supported | Premium, can run larger models |
| **Xavier NX** | 8 GB | ⚠️ Limited | Older GPU arch, use small models |
| **Jetson Nano (orig)** | 4 GB | ⚠️ Minimal | Very constrained |
| **Tegra / Spark (Linux)** | Varies | 🔬 Experimental | Where local inference is available |
| **Any Linux + NVIDIA GPU** | 4+ GB | ✅ Dev/Testing | For development on desktop |

The system **auto-detects your hardware** and selects appropriate models.
JetPack 5.x and 6.x (Ubuntu 20.04 / 22.04) are both supported.

> **📋 Full hardware guide with Amazon links**: [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md)

---

## 🚀 Quickstart

### 1. Clone & Setup (one command)

```bash
git clone https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor.git
cd Nvidia_Jetson_AI_Tutor
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

### Install for Development

```bash
# Lightweight (CI / testing only — no heavy AI deps)
pip install -e ".[dev]"

# Full stack (includes Gradio, LLM, STT, TTS, RAG)
pip install -e ".[all]"
```

> **On Jetson**: Use `./scripts/setup_jetson.sh` — it installs everything including
> CUDA-accelerated llama.cpp automatically.

### Dependency Extras

| Extra | What's included | Use case |
|-------|----------------|----------|
| _(base)_ | pydantic, numpy, Pillow, pytesseract | Always installed |
| `[dev]` | pytest, ruff, black | CI and linting |
| `[ai]` | gradio, llama-cpp, whisper, piper, faiss, sentence-transformers | On-device inference |
| `[all]` | `[ai]` + `[dev]` combined | Full local development |

### Run Tests
```bash
pytest edgetutor/tests/ -v
```

### Lint & Format
```bash
ruff check edgetutor/
ruff format --check edgetutor/
```

### Run in Debug Mode
```bash
./scripts/run.sh --debug
```

---

## 🗺️ Roadmap

See [GitHub Issues](https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor/issues)
for planned features:

- [ ] Wake word detection ("Hey EdgeTutor")
- [ ] Improved math step parsing (LaTeX, handwriting)
- [ ] Multi-user profiles (student accounts)
- [ ] Spanish language mode
- [ ] Classroom fleet management (multi-device)

---

## 🤝 Contributing

Contributions are welcome! See **[CONTRIBUTING.md](CONTRIBUTING.md)** for:

- Development setup and dependency tiers
- Testing, linting, and formatting instructions
- Pull request guidelines and code style

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

## ⚠️ Disclaimer

This is an independent open-source project and is **not affiliated with or
endorsed by NVIDIA**. NVIDIA, Jetson, Tegra, JetPack, and related marks are
trademarks of NVIDIA Corporation. This project is built by the community
for the community.

---

*Made with ❤️ for students and educators everywhere.*

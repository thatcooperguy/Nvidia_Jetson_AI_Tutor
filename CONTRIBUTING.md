# Contributing to EdgeTutor AI

Thanks for your interest in contributing to EdgeTutor AI! This project aims to bring offline AI tutoring to every student with an NVIDIA Jetson device.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor.git
cd Nvidia_Jetson_AI_Tutor
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

For **development / CI** (lightweight, no GPU dependencies):

```bash
pip install -e ".[dev]"
```

For **full local testing** with all AI modules:

```bash
pip install -e ".[all]"
```

> On Jetson devices, use `./scripts/setup_jetson.sh` which handles CUDA-accelerated builds automatically.

### 4. Install system dependencies

Tesseract OCR is needed for tests:

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

## Running Tests

```bash
pytest edgetutor/tests/ -v
```

Tests are designed to run **without** GPU hardware, LLM models, or TTS voices. All hardware-dependent features are auto-disabled in the test fixtures.

## Linting & Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for lint issues
ruff check edgetutor/

# Auto-fix lint issues
ruff check edgetutor/ --fix

# Check formatting
ruff format --check edgetutor/

# Auto-format
ruff format edgetutor/
```

CI enforces both `ruff check` and `ruff format --check` on every push.

## Pull Request Guidelines

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for new functionality
3. **Run the full check** before submitting:
   ```bash
   ruff check edgetutor/ && ruff format --check edgetutor/ && pytest edgetutor/tests/ -v
   ```
4. **Keep PRs focused** — one feature or fix per PR
5. **Write clear commit messages** describing _why_, not just _what_
6. **Update documentation** if your change affects user-facing behavior

## Code Style

- Python 3.10+ with `from __future__ import annotations`
- Type hints on all public functions
- Use `X | None` instead of `Optional[X]`
- Use `collections.abc.Generator` instead of `typing.Generator`
- Docstrings on all public classes and methods
- Imports sorted by ruff (isort-compatible)

## Architecture Notes

- **Modules are lazy-loaded** — heavy imports (llama-cpp, whisper, faiss) happen inside methods, not at module level
- **Graceful degradation** — if a module fails to load, others keep working
- **Singleton pattern** — `get_llm()`, `get_stt()`, `get_tts()`, `get_rag()` return cached instances
- **Settings via pydantic-settings** — all config flows through `edgetutor/core/settings.py`
- **Kid-safe by default** — safety filters are always on unless explicitly disabled

## Dependency Tiers

| Extra | What's included | When to use |
|-------|----------------|-------------|
| _(base)_ | pydantic, numpy, Pillow, pytesseract | Always installed |
| `[dev]` | pytest, ruff, black | CI and development |
| `[ai]` | gradio, llama-cpp, whisper, piper, faiss, sentence-transformers | On-device with full AI stack |
| `[all]` | Everything above | Full local development |

## Reporting Issues

- Use [GitHub Issues](https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor/issues)
- Include your Jetson model, JetPack version, and Python version
- For crashes, include the full traceback from `logs/edgetutor.log`

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

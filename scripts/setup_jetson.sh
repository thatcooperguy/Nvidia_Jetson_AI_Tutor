#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EdgeTutor AI — Jetson Setup Script
#
# Installs all dependencies, creates a Python venv, and optionally downloads
# default models. Run this ONCE after cloning the repo.
#
# Usage:
#   chmod +x scripts/setup_jetson.sh
#   ./scripts/setup_jetson.sh
#
# Requirements:
#   - NVIDIA Jetson with JetPack 6.x (Ubuntu 22.04)
#   - Internet connection (for initial setup only)
#   - sudo access
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON="python3"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          EdgeTutor AI — Jetson Setup Script                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Repo root: ${REPO_ROOT}"
echo ""

# ── Step 1: System dependencies ──────────────────────────────────────────────
echo "▶ [1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    tesseract-ocr tesseract-ocr-eng \
    libsndfile1 ffmpeg portaudio19-dev \
    cmake build-essential \
    libjpeg-dev libpng-dev \
    2>/dev/null

echo "  ✅ System dependencies installed."

# ── Step 2: Create Python virtual environment ────────────────────────────────
echo ""
echo "▶ [2/7] Setting up Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    ${PYTHON} -m venv "${VENV_DIR}"
    echo "  Created venv at ${VENV_DIR}"
else
    echo "  Venv already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

echo "  ✅ Python venv ready ($(python3 --version))"

# ── Step 3: Install Python dependencies ──────────────────────────────────────
echo ""
echo "▶ [3/7] Installing Python dependencies..."
pip install -e "${REPO_ROOT}[ai,dev]" -q 2>&1 | tail -5

echo "  ✅ Python dependencies installed."

# ── Step 4: Install llama-cpp-python with CUDA ───────────────────────────────
echo ""
echo "▶ [4/7] Installing llama-cpp-python with CUDA support..."
echo "  This may take a few minutes (compiles from source)..."

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "  Found CUDA: $(nvcc --version | head -4 | tail -1)"
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir -q 2>&1 | tail -3
    echo "  ✅ llama-cpp-python installed with CUDA support."
else
    echo "  ⚠️ CUDA not found. Installing CPU-only version..."
    pip install llama-cpp-python -q
    echo "  ✅ llama-cpp-python installed (CPU only)."
fi

# ── Step 5: Create directories ───────────────────────────────────────────────
echo ""
echo "▶ [5/7] Creating directories..."
mkdir -p "${REPO_ROOT}/models"
mkdir -p "${REPO_ROOT}/voices"
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/content/starter_pack"
echo "  ✅ Directories created."

# ── Step 6: Copy .env if needed ──────────────────────────────────────────────
echo ""
echo "▶ [6/7] Setting up configuration..."
if [ ! -f "${REPO_ROOT}/.env" ]; then
    cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
    echo "  Created .env from .env.example"
else
    echo "  .env already exists — keeping existing config."
fi

# ── Step 7: Download models (optional) ───────────────────────────────────────
echo ""
echo "▶ [7/7] Model downloads..."
echo ""
echo "  Would you like to download the default LLM model? (~2-4 GB)"
echo "  This is required to use the tutor. You can also download manually later."
echo ""
read -p "  Download default LLM? [y/N]: " -n 1 -r DOWNLOAD_LLM
echo ""

if [[ $DOWNLOAD_LLM =~ ^[Yy]$ ]]; then
    bash "${REPO_ROOT}/scripts/download_model.sh"
else
    echo "  Skipped. Run 'scripts/download_model.sh' later to download a model."
fi

echo ""
read -p "  Download default TTS voice? (~75 MB) [y/N]: " -n 1 -r DOWNLOAD_TTS
echo ""

if [[ $DOWNLOAD_TTS =~ ^[Yy]$ ]]; then
    bash "${REPO_ROOT}/scripts/download_voice.sh"
else
    echo "  Skipped. Run 'scripts/download_voice.sh' later."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! 🎉                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  To start EdgeTutor:                                       ║"
echo "║    ./scripts/run.sh                                        ║"
echo "║                                                            ║"
echo "║  To index curriculum content:                              ║"
echo "║    ./scripts/ingest_content.sh                             ║"
echo "║                                                            ║"
echo "║  UI will be available at: http://localhost:7860             ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EdgeTutor AI — Download Default LLM Model
#
# Downloads a quantized Phi-3 Mini (3.8B) model in GGUF format.
# This is a great default for Jetson Orin Nano 8GB:
#   - Small enough to fit in memory with GPU offloading
#   - Good instruction-following quality
#   - Quantized to Q4_K_M for speed
#
# You can replace this with any GGUF model. See docs/MODELS.md for options.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
mkdir -p "${MODELS_DIR}"

echo "📦 EdgeTutor AI — LLM Model Downloader"
echo ""

# Default: Phi-3 Mini Instruct Q4_K_M (~2.3 GB)
DEFAULT_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
DEFAULT_FILENAME="Phi-3-mini-4k-instruct-q4.gguf"

# Alternative options
echo "Available models (choose one):"
echo ""
echo "  1) Phi-3 Mini 4K Instruct Q4 (~2.3 GB) [RECOMMENDED for 8GB Jetson]"
echo "     Good quality, fast, fits comfortably in memory."
echo ""
echo "  2) TinyLlama 1.1B Chat Q8 (~1.1 GB) [FAST, lower quality]"
echo "     Very fast but less capable. Good for testing."
echo ""
echo "  3) Custom URL (provide your own GGUF)"
echo ""

read -p "Choose [1/2/3] (default: 1): " -n 1 -r CHOICE
echo ""

case "${CHOICE:-1}" in
    2)
        MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        MODEL_FILENAME="tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
        ;;
    3)
        read -p "Enter GGUF download URL: " MODEL_URL
        read -p "Enter filename: " MODEL_FILENAME
        ;;
    *)
        MODEL_URL="${DEFAULT_URL}"
        MODEL_FILENAME="${DEFAULT_FILENAME}"
        ;;
esac

DEST="${MODELS_DIR}/${MODEL_FILENAME}"

if [ -f "${DEST}" ]; then
    echo "  Model already exists: ${DEST}"
    echo "  Delete it first if you want to re-download."
    exit 0
fi

echo ""
echo "  Downloading: ${MODEL_FILENAME}"
echo "  From: ${MODEL_URL}"
echo "  To: ${DEST}"
echo ""

# Use wget or curl
if command -v wget &> /dev/null; then
    wget -O "${DEST}" "${MODEL_URL}" --show-progress
elif command -v curl &> /dev/null; then
    curl -L -o "${DEST}" "${MODEL_URL}" --progress-bar
else
    echo "❌ Neither wget nor curl found. Please install one."
    exit 1
fi

# Create symlink as default.gguf
ln -sf "${MODEL_FILENAME}" "${MODELS_DIR}/default.gguf"

echo ""
echo "✅ Model downloaded: ${DEST}"
echo "   Symlinked as: ${MODELS_DIR}/default.gguf"
echo ""
echo "   You can now run EdgeTutor with: ./scripts/run.sh"

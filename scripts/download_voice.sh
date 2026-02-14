#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EdgeTutor AI — Download Default TTS Voice
#
# Downloads a Piper TTS voice model for offline text-to-speech.
# Default: en_US-lessac-medium (clear, natural American English voice)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICES_DIR="${REPO_ROOT}/voices"
mkdir -p "${VOICES_DIR}"

echo "🔊 EdgeTutor AI — TTS Voice Downloader"
echo ""

VOICE_NAME="en_US-lessac-medium"
VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

VOICE_FILE="${VOICES_DIR}/${VOICE_NAME}.onnx"
CONFIG_FILE="${VOICES_DIR}/${VOICE_NAME}.onnx.json"

if [ -f "${VOICE_FILE}" ] && [ -f "${CONFIG_FILE}" ]; then
    echo "  Voice already downloaded: ${VOICE_FILE}"
    exit 0
fi

echo "  Downloading: ${VOICE_NAME}"
echo ""

# Download voice model
if command -v wget &> /dev/null; then
    wget -O "${VOICE_FILE}" "${VOICE_URL}" --show-progress
    wget -O "${CONFIG_FILE}" "${CONFIG_URL}" --show-progress
elif command -v curl &> /dev/null; then
    curl -L -o "${VOICE_FILE}" "${VOICE_URL}" --progress-bar
    curl -L -o "${CONFIG_FILE}" "${CONFIG_URL}" --progress-bar
else
    echo "❌ Neither wget nor curl found."
    exit 1
fi

echo ""
echo "✅ TTS voice downloaded: ${VOICE_FILE}"
echo ""
echo "   To use a different voice, see: docs/MODELS.md"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EdgeTutor AI — Run Script
#
# Starts the EdgeTutor application (Gradio UI + all modules).
# Assumes setup_jetson.sh has been run first.
#
# Usage:
#   ./scripts/run.sh
#   ./scripts/run.sh --port 8080
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# Activate venv
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "❌ Virtual environment not found at ${VENV_DIR}"
    echo "   Run './scripts/setup_jetson.sh' first."
    exit 1
fi

# Pass any extra args as env vars
for arg in "$@"; do
    case $arg in
        --port=*|--port)
            if [[ "$arg" == "--port" ]]; then
                shift
                export EDGETUTOR_PORT="${1:-7860}"
            else
                export EDGETUTOR_PORT="${arg#*=}"
            fi
            ;;
        --host=*)
            export EDGETUTOR_HOST="${arg#*=}"
            ;;
        --debug)
            export LOG_LEVEL="DEBUG"
            ;;
    esac
done

echo "🎓 Starting EdgeTutor AI..."
echo "   Access at: http://localhost:${EDGETUTOR_PORT:-7860}"
echo "   Press Ctrl+C to stop."
echo ""

cd "${REPO_ROOT}"
python -m edgetutor

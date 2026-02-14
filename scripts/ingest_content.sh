#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EdgeTutor AI — Content Ingestion Script
#
# Indexes curriculum packs (PDF/TXT/MD) from the content/ folder into the
# local FAISS vector store for RAG retrieval.
#
# Usage:
#   ./scripts/ingest_content.sh
#   ./scripts/ingest_content.sh /path/to/custom/content
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# Activate venv
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "❌ Virtual environment not found. Run setup_jetson.sh first."
    exit 1
fi

CONTENT_DIR="${1:-${REPO_ROOT}/content}"

echo "📚 EdgeTutor AI — Content Ingestion"
echo ""
echo "  Content directory: ${CONTENT_DIR}"
echo ""

cd "${REPO_ROOT}"
python -c "
from edgetutor.core.rag import get_rag

rag = get_rag()
print('Loading embedding model...')
ok = rag.load_embedder()
if not ok:
    print('Failed to load embedder.')
    exit(1)

from pathlib import Path
content_dir = Path('${CONTENT_DIR}')
print(f'Ingesting from: {content_dir}')

count = rag.ingest(content_dir)
print(f'Done! Indexed {count} chunks.')
"

echo ""
echo "✅ Content ingestion complete."
echo "   The RAG index is now available for EdgeTutor queries."

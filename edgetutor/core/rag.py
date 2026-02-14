"""
EdgeTutor AI — RAG (Retrieval-Augmented Generation) module.

Provides local document ingestion and retrieval using FAISS + sentence-transformers.
Documents are chunked, embedded, and stored in a local FAISS index.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)

_rag_instance = None


class RAGStore:
    """Local FAISS-backed vector store for curriculum content."""

    def __init__(self):
        self.index = None
        self.chunks: list[dict] = []  # {"text": ..., "source": ..., "idx": ...}
        self.embedder = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load_embedder(self) -> bool:
        """Load the sentence-transformer embedding model."""
        cfg = get_settings()
        try:
            from sentence_transformers import SentenceTransformer

            t0 = time.time()
            self.embedder = SentenceTransformer(cfg.rag_embedding_model)
            logger.info(
                "RAG embedder loaded in %.1fs — model=%s",
                time.time() - t0,
                cfg.rag_embedding_model,
            )
            return True
        except ImportError:
            logger.error("sentence-transformers not installed.")
            return False
        except Exception as e:
            logger.error("Failed to load embedder: %s", e)
            return False

    def load_index(self) -> bool:
        """Load an existing FAISS index from disk."""
        cfg = get_settings()
        index_dir = cfg.rag_index_resolved

        index_file = index_dir / "index.faiss"
        meta_file = index_dir / "chunks.json"

        if not index_file.exists() or not meta_file.exists():
            logger.info("No FAISS index found at %s — run ingest first.", index_dir)
            return False

        try:
            import faiss

            self.index = faiss.read_index(str(index_file))
            with open(meta_file, encoding="utf-8") as f:
                self.chunks = json.load(f)

            logger.info(
                "RAG index loaded: %d vectors, %d chunks", self.index.ntotal, len(self.chunks)
            )
            self._ready = self.embedder is not None
            return True
        except ImportError:
            logger.error("faiss-cpu not installed.")
            return False
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
            return False

    def ingest(self, content_dir: Path | None = None) -> int:
        """
        Ingest documents from content directory into FAISS index.

        Supports: .txt, .md, .pdf
        Returns the number of chunks indexed.
        """
        cfg = get_settings()
        if content_dir is None:
            content_dir = cfg.rag_content_resolved

        if not content_dir.exists():
            logger.warning("Content directory not found: %s", content_dir)
            return 0

        if self.embedder is None:
            logger.error("Embedder not loaded. Call load_embedder() first.")
            return 0

        # Gather documents
        documents: list[tuple[str, str]] = []  # (text, source_filename)

        for fpath in sorted(content_dir.rglob("*")):
            if fpath.suffix.lower() == ".txt" or fpath.suffix.lower() == ".md":
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    if text.strip():
                        documents.append((text, fpath.name))
                except Exception as e:
                    logger.warning("Could not read %s: %s", fpath, e)

            elif fpath.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(str(fpath))
                    pages = [page.extract_text() or "" for page in reader.pages]
                    text = "\n".join(pages)
                    if text.strip():
                        documents.append((text, fpath.name))
                except ImportError:
                    logger.warning("pypdf not installed — skipping %s", fpath.name)
                except Exception as e:
                    logger.warning("Could not read PDF %s: %s", fpath, e)

        if not documents:
            logger.info("No documents found in %s", content_dir)
            return 0

        # Chunk documents
        chunk_size = cfg.rag_chunk_size
        self.chunks = []
        for text, source in documents:
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i : i + chunk_size].strip()
                if chunk_text:
                    self.chunks.append(
                        {"text": chunk_text, "source": source, "idx": len(self.chunks)}
                    )

        logger.info("Chunked %d documents into %d chunks", len(documents), len(self.chunks))

        # Embed
        texts = [c["text"] for c in self.chunks]
        t0 = time.time()
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        logger.info("Embedded %d chunks in %.1fs", len(texts), time.time() - t0)

        # Build FAISS index
        import faiss

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)

        # L2-normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Save to disk
        index_dir = cfg.rag_index_resolved
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        self._ready = True
        logger.info("FAISS index saved to %s (%d vectors)", index_dir, self.index.ntotal)
        return len(self.chunks)

    def query(self, text: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Returns list of {"text": ..., "source": ..., "score": ...}.
        """
        if not self._ready or self.index is None or self.embedder is None:
            return []

        cfg = get_settings()
        k = top_k or cfg.rag_top_k

        import faiss

        query_vec = self.embedder.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(score),
                }
            )

        return results


def get_rag() -> RAGStore:
    """Return the singleton RAG store."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGStore()
    return _rag_instance

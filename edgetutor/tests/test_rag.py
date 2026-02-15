"""Tests for edgetutor.core.rag — RAG store (mock-based, no FAISS/embedder required)."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestRAGStoreInit:
    """Test RAGStore initialization and state."""

    def test_default_state(self):
        """RAG store should start unready with no index."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        assert store.is_ready is False
        assert store.index is None
        assert store.chunks == []
        assert store.embedder is None

    def test_singleton_returns_instance(self):
        """get_rag() should return a RAGStore."""
        from edgetutor.core.rag import RAGStore, get_rag

        store = get_rag()
        assert isinstance(store, RAGStore)

    def test_query_when_not_ready(self):
        """Querying an unready store should return empty list."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        results = store.query("What is photosynthesis?")
        assert results == []

    def test_query_when_no_embedder(self):
        """Querying with no embedder should return empty list."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        store._ready = True
        store.index = MagicMock()
        store.embedder = None
        results = store.query("test")
        assert results == []


class TestRAGLoadEmbedder:
    """Test embedder loading (mocked)."""

    def test_load_embedder_missing_package(self):
        """Should return False if sentence-transformers is not installed."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            patch(
                "edgetutor.core.rag.RAGStore.load_embedder",
                side_effect=ImportError("No module"),
            ),
            contextlib.suppress(ImportError),
        ):
            store.load_embedder()
        assert store.embedder is None


class TestRAGLoadIndex:
    """Test index loading (mocked filesystem)."""

    def test_load_index_no_files(self, tmp_path, monkeypatch):
        """Should return False when index files don't exist."""
        from edgetutor.core.rag import RAGStore
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        # Patch the underlying path field so the resolved property points to nonexistent dir
        monkeypatch.setattr(cfg, "rag_index_dir", str(tmp_path / "nonexistent"))

        store = RAGStore()
        result = store.load_index()
        assert result is False
        assert store.is_ready is False


class TestRAGIngest:
    """Test document ingestion logic (mocked)."""

    def test_ingest_empty_directory(self, tmp_path):
        """Ingesting from an empty directory should return 0 chunks."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        store.embedder = MagicMock()  # Pretend embedder is loaded
        result = store.ingest(content_dir=tmp_path)
        assert result == 0

    def test_ingest_no_embedder(self, tmp_path):
        """Ingesting without embedder should return 0."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        assert store.embedder is None
        result = store.ingest(content_dir=tmp_path)
        assert result == 0

    def test_ingest_missing_directory(self):
        """Ingesting from nonexistent directory should return 0."""
        from edgetutor.core.rag import RAGStore

        store = RAGStore()
        store.embedder = MagicMock()
        result = store.ingest(content_dir=Path("/nonexistent/path"))
        assert result == 0

    def test_ingest_reads_txt_files(self, tmp_path, monkeypatch):
        """Should read .txt files from content directory."""
        from edgetutor.core.rag import RAGStore

        # Create a sample text file
        (tmp_path / "test.txt").write_text("The sun is a star. It gives us light and heat.")

        import numpy as np

        # Mock embedder and faiss
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_faiss.IndexFlatIP.return_value = mock_index

        store = RAGStore()
        store.embedder = mock_embedder

        with (
            patch.dict("sys.modules", {"faiss": mock_faiss}),
            patch("edgetutor.core.rag.get_settings") as mock_cfg,
        ):
            cfg = MagicMock()
            cfg.rag_chunk_size = 500
            cfg.rag_index_resolved = tmp_path / "index"
            mock_cfg.return_value = cfg

            result = store.ingest(content_dir=tmp_path)

        assert result > 0
        assert len(store.chunks) > 0
        assert store.chunks[0]["source"] == "test.txt"

    def test_ingest_reads_md_files(self, tmp_path, monkeypatch):
        """Should read .md files from content directory."""
        from edgetutor.core.rag import RAGStore

        (tmp_path / "notes.md").write_text("# Chapter 1\n\nPhotosynthesis is the process...")

        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.random.rand(1, 384).astype(np.float32)

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 1
        mock_faiss.IndexFlatIP.return_value = mock_index

        store = RAGStore()
        store.embedder = mock_embedder

        with (
            patch.dict("sys.modules", {"faiss": mock_faiss}),
            patch("edgetutor.core.rag.get_settings") as mock_cfg,
        ):
            cfg = MagicMock()
            cfg.rag_chunk_size = 500
            cfg.rag_index_resolved = tmp_path / "index"
            mock_cfg.return_value = cfg

            result = store.ingest(content_dir=tmp_path)

        assert result > 0
        assert store.chunks[0]["source"] == "notes.md"

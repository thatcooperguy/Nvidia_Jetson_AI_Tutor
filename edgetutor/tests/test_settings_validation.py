"""Tests for edgetutor.core.settings — field validation constraints."""

import pytest
from pydantic import ValidationError


class TestSettingsValidation:
    """Test that Settings rejects out-of-range values."""

    def test_port_too_high(self):
        """Port above 65535 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="less than or equal to 65535"):
            Settings(_env_file="nonexistent.env", port=99999)

    def test_port_too_low(self):
        """Port below 1 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Settings(_env_file="nonexistent.env", port=0)

    def test_temperature_too_high(self):
        """Temperature above 2.0 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="less than or equal to 2"):
            Settings(_env_file="nonexistent.env", llm_temperature=5.0)

    def test_temperature_negative(self):
        """Negative temperature should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Settings(_env_file="nonexistent.env", llm_temperature=-0.1)

    def test_context_size_too_small(self):
        """Context size below 128 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 128"):
            Settings(_env_file="nonexistent.env", llm_context_size=64)

    def test_max_tokens_zero(self):
        """Max tokens of 0 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Settings(_env_file="nonexistent.env", llm_max_tokens=0)

    def test_rag_top_k_zero(self):
        """RAG top_k of 0 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Settings(_env_file="nonexistent.env", rag_top_k=0)

    def test_rag_chunk_size_too_small(self):
        """RAG chunk size below 50 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 50"):
            Settings(_env_file="nonexistent.env", rag_chunk_size=10)

    def test_tts_rate_too_low(self):
        """TTS rate below 0.1 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="greater than or equal to 0.1"):
            Settings(_env_file="nonexistent.env", tts_rate=0.0)

    def test_tts_rate_too_high(self):
        """TTS rate above 5.0 should be rejected."""
        from edgetutor.core.settings import Settings

        with pytest.raises(ValidationError, match="less than or equal to 5"):
            Settings(_env_file="nonexistent.env", tts_rate=10.0)

    def test_valid_values_accepted(self, monkeypatch):
        """Valid edge-case values should be accepted."""
        for key in ("STT_ENABLED", "TTS_ENABLED", "CAMERA_ENABLED", "LLM_MODEL_PATH"):
            monkeypatch.delenv(key, raising=False)

        from edgetutor.core.settings import Settings

        s = Settings(
            _env_file="nonexistent.env",
            port=8080,
            llm_temperature=0.0,
            llm_context_size=128,
            llm_max_tokens=1,
            rag_top_k=1,
            rag_chunk_size=50,
            tts_rate=0.1,
        )
        assert s.port == 8080
        assert s.llm_temperature == 0.0
        assert s.llm_context_size == 128
        assert s.rag_top_k == 1
        assert s.tts_rate == 0.1

    def test_gpu_layers_negative_one_allowed(self, monkeypatch):
        """GPU layers of -1 (all layers) should be accepted."""
        for key in ("STT_ENABLED", "TTS_ENABLED", "CAMERA_ENABLED", "LLM_MODEL_PATH"):
            monkeypatch.delenv(key, raising=False)

        from edgetutor.core.settings import Settings

        s = Settings(_env_file="nonexistent.env", llm_n_gpu_layers=-1)
        assert s.llm_n_gpu_layers == -1

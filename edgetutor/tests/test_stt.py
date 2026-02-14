"""Tests for edgetutor.audio.stt — Speech-to-Text engine (mock-based)."""

from unittest.mock import MagicMock, patch


class TestSTTEngineInit:
    """Test STTEngine initialization and state."""

    def test_default_state(self):
        """STT engine should start unready with no model."""
        from edgetutor.audio.stt import STTEngine

        engine = STTEngine()
        assert engine.is_ready is False
        assert engine.model is None

    def test_singleton_returns_instance(self):
        """get_stt() should return an STTEngine."""
        from edgetutor.audio.stt import STTEngine, get_stt

        engine = get_stt()
        assert isinstance(engine, STTEngine)


class TestSTTLoad:
    """Test model loading (mocked)."""

    def test_load_when_disabled(self, monkeypatch):
        """Should return False when STT is disabled in settings."""
        from edgetutor.audio.stt import STTEngine
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "stt_enabled", False)

        engine = STTEngine()
        result = engine.load()
        assert result is False
        assert engine.is_ready is False

    def test_load_missing_package(self, monkeypatch):
        """Should return False if faster-whisper is not installed."""
        from edgetutor.audio.stt import STTEngine
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "stt_enabled", True)

        engine = STTEngine()

        with patch.dict("sys.modules", {"faster_whisper": None}):
            result = engine.load()

        assert result is False
        assert engine.is_ready is False


class TestSTTTranscribe:
    """Test transcription methods (mocked)."""

    def test_transcribe_when_not_ready(self):
        """Transcribing without a loaded model should return error."""
        from edgetutor.audio.stt import STTEngine

        engine = STTEngine()
        result = engine.transcribe("/fake/audio.wav")
        assert result["text"] == ""
        assert "error" in result
        assert "not loaded" in result["error"].lower()

    def test_transcribe_numpy_when_not_ready(self):
        """Numpy transcription without a loaded model should return error."""
        from edgetutor.audio.stt import STTEngine

        engine = STTEngine()
        result = engine.transcribe_numpy([0.0, 0.1, 0.2])
        assert result["text"] == ""
        assert "error" in result

    def test_transcribe_numpy_normalizes_audio(self, monkeypatch):
        """Should normalize audio to float32 mono and call model."""
        import numpy as np

        from edgetutor.audio.stt import STTEngine
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "stt_language", "en")

        engine = STTEngine()
        engine._ready = True

        # Mock model
        mock_segment = MagicMock()
        mock_segment.text = " Hello world "
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 1.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)
        engine.model = mock_model

        # Stereo int16 input
        audio = np.array([[100, 200], [300, 400]], dtype=np.int16)
        result = engine.transcribe_numpy(audio, sample_rate=16000)

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        mock_model.transcribe.assert_called_once()

    def test_transcribe_handles_exception(self):
        """Model exception should return error dict, not raise."""
        from edgetutor.audio.stt import STTEngine

        engine = STTEngine()
        engine._ready = True
        engine.model = MagicMock()
        engine.model.transcribe.side_effect = RuntimeError("decode failed")

        result = engine.transcribe("/fake/audio.wav")
        assert result["text"] == ""
        assert "error" in result
        assert "decode failed" in result["error"]

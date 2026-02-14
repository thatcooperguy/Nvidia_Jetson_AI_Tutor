"""Tests for edgetutor.audio.tts — Text-to-Speech engine (mock-based)."""

from unittest.mock import MagicMock, patch


class TestTTSEngineInit:
    """Test TTSEngine initialization and state."""

    def test_default_state(self):
        """TTS engine should start unready with no voice."""
        from edgetutor.audio.tts import TTSEngine

        engine = TTSEngine()
        assert engine.is_ready is False
        assert engine.voice is None
        assert engine.synthesizer is None

    def test_singleton_returns_instance(self):
        """get_tts() should return a TTSEngine."""
        from edgetutor.audio.tts import TTSEngine, get_tts

        engine = get_tts()
        assert isinstance(engine, TTSEngine)


class TestTTSLoad:
    """Test voice loading (mocked)."""

    def test_load_when_disabled(self, monkeypatch):
        """Should return False when TTS is disabled in settings."""
        from edgetutor.audio.tts import TTSEngine
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "tts_enabled", False)

        engine = TTSEngine()
        result = engine.load()
        assert result is False
        assert engine.is_ready is False

    def test_load_missing_voice_file(self, monkeypatch, tmp_path):
        """Should return False when voice model file doesn't exist."""
        from edgetutor.audio.tts import TTSEngine
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "tts_enabled", True)
        monkeypatch.setattr(cfg, "tts_voice_model", str(tmp_path / "nonexistent.onnx"))

        engine = TTSEngine()

        with patch.object(
            type(cfg),
            "tts_voice_resolved",
            new_callable=lambda: property(lambda s: tmp_path / "nonexistent.onnx"),
        ):
            result = engine.load()

        assert result is False
        assert engine.is_ready is False

    def test_load_missing_package(self, monkeypatch, tmp_path):
        """Should return False if piper-tts is not installed."""
        from edgetutor.audio.tts import TTSEngine
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "tts_enabled", True)

        # Create a fake voice file so the file-exists check passes
        voice_file = tmp_path / "test_voice.onnx"
        voice_file.write_text("fake")

        engine = TTSEngine()

        with (
            patch.object(
                type(cfg),
                "tts_voice_resolved",
                new_callable=lambda: property(lambda s: voice_file),
            ),
            patch.dict("sys.modules", {"piper": None}),
        ):
            result = engine.load()

        assert result is False


class TestTTSSynthesize:
    """Test synthesis methods (mocked)."""

    def test_synthesize_when_not_ready(self):
        """Synthesis without a loaded voice should return None."""
        from edgetutor.audio.tts import TTSEngine

        engine = TTSEngine()
        result = engine.synthesize("Hello world")
        assert result is None

    def test_synthesize_empty_text(self):
        """Empty text should return None."""
        from edgetutor.audio.tts import TTSEngine

        engine = TTSEngine()
        engine._ready = True
        engine.voice = MagicMock()
        assert engine.synthesize("") is None
        assert engine.synthesize("   ") is None

    def test_get_audio_tuple_when_not_ready(self):
        """get_audio_tuple without a loaded voice should return None."""
        from edgetutor.audio.tts import TTSEngine

        engine = TTSEngine()
        result = engine.get_audio_tuple("Hello")
        assert result is None

    def test_synthesize_to_file_when_not_ready(self, tmp_path):
        """synthesize_to_file without a loaded voice should return False."""
        from edgetutor.audio.tts import TTSEngine

        engine = TTSEngine()
        result = engine.synthesize_to_file("Hello", tmp_path / "out.wav")
        assert result is False

    def test_synthesize_to_file_creates_directory(self, tmp_path):
        """synthesize_to_file should create parent directories."""
        from edgetutor.audio.tts import TTSEngine

        engine = TTSEngine()
        engine._ready = True
        engine._sample_rate = 22050

        # Create a real minimal WAV
        mock_voice = MagicMock()

        def fake_synth(text, wav_file):
            wav_file.writeframes(b"\x00\x00" * 100)

        mock_voice.synthesize = fake_synth
        engine.voice = mock_voice

        output = tmp_path / "subdir" / "output.wav"
        result = engine.synthesize_to_file("Hello", output)
        assert result is True
        assert output.exists()

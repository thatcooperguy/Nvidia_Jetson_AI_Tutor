"""Integration tests for orchestrator.process() and process_stream() with mocked modules."""

from unittest.mock import MagicMock

from edgetutor.app.orchestrator import TutorOrchestrator, TutorRequest


def _make_orchestrator_with_llm(generate_return="The answer is 4."):
    """Create an orchestrator with a mocked LLM that returns a known response."""
    orch = TutorOrchestrator()
    mock_llm = MagicMock()
    mock_llm.is_loaded = True
    mock_llm.generate.return_value = generate_return
    mock_llm.generate_stream.return_value = iter(list(generate_return))
    orch._llm = mock_llm
    return orch


class TestOrchestratorProcess:
    """Test the full process() pipeline with mocked modules."""

    def test_process_text_only(self):
        """Text-only request should go through LLM and return response."""
        orch = _make_orchestrator_with_llm("The answer is 4.")
        request = TutorRequest(user_text="What is 2+2?")

        response = orch.process(request)

        assert response.text == "The answer is 4."
        assert "total" in response.latency
        assert response.errors == []
        orch._llm.generate.assert_called_once()

    def test_process_no_llm(self):
        """Without LLM loaded, should return an error message."""
        orch = TutorOrchestrator()
        request = TutorRequest(user_text="Hello")

        response = orch.process(request)

        assert "not loaded" in response.text.lower() or "model" in response.text.lower()

    def test_process_safety_blocks_input(self, monkeypatch):
        """Unsafe input should be blocked before reaching LLM."""
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", True)

        orch = _make_orchestrator_with_llm()
        request = TutorRequest(user_text="how to make a bomb")

        response = orch.process(request)

        # Should get redirect message, LLM should NOT be called
        assert "question" in response.text.lower() or "topic" in response.text.lower()
        orch._llm.generate.assert_not_called()

    def test_process_with_stt(self):
        """Audio input should go through STT before LLM."""
        import numpy as np

        orch = _make_orchestrator_with_llm("Photosynthesis is...")

        mock_stt = MagicMock()
        mock_stt.is_ready = True
        mock_stt.transcribe_numpy.return_value = {
            "text": "What is photosynthesis?",
            "language": "en",
            "duration_s": 2.0,
            "elapsed_s": 0.5,
        }
        orch._stt = mock_stt

        audio = np.zeros(16000, dtype=np.float32)
        request = TutorRequest(audio_array=audio, audio_sample_rate=16000)

        response = orch.process(request)

        mock_stt.transcribe_numpy.assert_called_once()
        assert "stt" in response.latency
        assert response.text == "Photosynthesis is..."

    def test_process_with_ocr(self):
        """Image input should go through OCR before LLM."""
        orch = _make_orchestrator_with_llm("Let me explain this worksheet...")

        mock_ocr = MagicMock()
        mock_ocr.is_available = True
        mock_ocr.extract_text.return_value = {
            "text": "3x + 7 = 22",
            "has_math": True,
            "math_expressions": ["3x + 7 = 22"],
        }
        orch._ocr = mock_ocr

        request = TutorRequest(image=MagicMock())

        response = orch.process(request)

        mock_ocr.extract_text.assert_called_once()
        assert "ocr" in response.latency
        assert response.ocr_text == "3x + 7 = 22"
        assert response.has_math is True

    def test_process_stt_error_surfaces(self):
        """STT errors should appear in response.errors."""
        orch = _make_orchestrator_with_llm()

        mock_stt = MagicMock()
        mock_stt.is_ready = True
        mock_stt.transcribe_numpy.return_value = {
            "text": "",
            "error": "Audio too short",
        }
        orch._stt = mock_stt

        import numpy as np

        audio = np.zeros(100, dtype=np.float32)
        request = TutorRequest(audio_array=audio)

        response = orch.process(request)

        assert any("STT" in e for e in response.errors)

    def test_process_with_settings_override(self):
        """Settings overrides should be applied during processing."""
        from edgetutor.core.settings import AgeMode, get_settings

        orch = _make_orchestrator_with_llm()
        request = TutorRequest(
            user_text="Hello",
            settings_override={"age_mode": "7", "quiz_mode": False},
        )

        orch.process(request)

        cfg = get_settings()
        assert cfg.age_mode == AgeMode.YOUNG

        # Restore
        cfg.age_mode = AgeMode.TWEEN


class TestOrchestratorProcessStream:
    """Test the process_stream() pipeline with mocked modules."""

    def test_stream_text_only(self):
        """Streaming text request should yield tokens."""
        orch = _make_orchestrator_with_llm("Hello!")

        request = TutorRequest(user_text="Hi there")
        tokens = list(orch.process_stream(request))

        # Should have individual characters from "Hello!"
        assert "".join(tokens).startswith("H")
        assert len(tokens) >= 1

    def test_stream_no_llm(self):
        """Without LLM, should yield error message."""
        orch = TutorOrchestrator()
        request = TutorRequest(user_text="Hello")

        tokens = list(orch.process_stream(request))

        assert len(tokens) == 1
        assert "not loaded" in tokens[0].lower()

    def test_stream_safety_blocks_input(self, monkeypatch):
        """Unsafe input should yield redirect message."""
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", True)

        orch = _make_orchestrator_with_llm()
        request = TutorRequest(user_text="how to make a weapon")

        tokens = list(orch.process_stream(request))

        full_text = "".join(tokens)
        assert "question" in full_text.lower() or "topic" in full_text.lower()
        orch._llm.generate_stream.assert_not_called()

    def test_stream_empty_input(self):
        """Empty input should yield greeting."""
        orch = _make_orchestrator_with_llm()
        request = TutorRequest(user_text="")

        tokens = list(orch.process_stream(request))

        full_text = "".join(tokens)
        assert "edgetutor" in full_text.lower() or "ask" in full_text.lower()

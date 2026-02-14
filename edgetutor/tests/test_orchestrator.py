"""Tests for edgetutor.app.orchestrator — tutor orchestrator."""

from edgetutor.app.orchestrator import TutorOrchestrator, TutorRequest, TutorResponse


class TestTutorRequest:
    """Test TutorRequest dataclass."""

    def test_default_request(self):
        """Default request should have empty fields."""
        req = TutorRequest()
        assert req.user_text == ""
        assert req.audio_path is None
        assert req.audio_array is None
        assert req.image is None
        assert req.settings_override is None

    def test_request_with_text(self):
        """Request with text should store it."""
        req = TutorRequest(user_text="What is 2+2?")
        assert req.user_text == "What is 2+2?"


class TestTutorResponse:
    """Test TutorResponse dataclass."""

    def test_default_response(self):
        """Default response should have empty/default fields."""
        resp = TutorResponse()
        assert resp.text == ""
        assert resp.audio is None
        assert resp.ocr_text == ""
        assert resp.has_math is False
        assert resp.math_expressions == []
        assert resp.quiz_questions == ""
        assert resp.suggestions == []
        assert resp.latency == {}
        assert resp.errors == []


class TestOrchestratorInit:
    """Test orchestrator initialization (without loading heavy modules)."""

    def test_can_instantiate(self):
        """Orchestrator should instantiate without errors."""
        orch = TutorOrchestrator()
        assert orch._modules_loaded is False
        assert orch._llm is None

    def test_suggestions_no_context(self):
        """Suggestions without context should offer starting prompts."""
        orch = TutorOrchestrator()
        suggestions = orch._generate_suggestions("", "", "general")
        assert len(suggestions) > 0
        assert any("question" in s.lower() or "scan" in s.lower() for s in suggestions)

    def test_suggestions_with_ocr(self):
        """Suggestions with OCR context should offer follow-ups."""
        orch = TutorOrchestrator()
        suggestions = orch._generate_suggestions("", "some ocr text", "math")
        assert any("scan" in s.lower() or "explain" in s.lower() or "check" in s.lower() for s in suggestions)

    def test_apply_settings(self):
        """_apply_settings should update runtime config."""
        orch = TutorOrchestrator()
        from edgetutor.core.settings import AgeMode, SubjectMode, get_settings

        cfg = get_settings()
        original_age = cfg.age_mode

        orch._apply_settings({"age_mode": "7", "subject_mode": "math", "parent_mode": True})

        assert cfg.age_mode == AgeMode.YOUNG
        assert cfg.subject_mode == SubjectMode.MATH
        assert cfg.parent_mode is True

        # Restore
        cfg.age_mode = original_age
        cfg.subject_mode = SubjectMode.GENERAL
        cfg.parent_mode = False

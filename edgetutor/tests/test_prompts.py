"""Tests for edgetutor.core.prompts — prompt template builders."""

from edgetutor.core.prompts import (
    AGE_TONES,
    SUBJECT_INSTRUCTIONS,
    build_quiz_prompt,
    build_rag_context,
    build_system_prompt,
    build_vision_prompt,
)


class TestBuildSystemPrompt:
    """Test system prompt assembly."""

    def test_default_prompt_contains_edgetutor(self):
        """Default prompt should identify as EdgeTutor."""
        prompt = build_system_prompt()
        assert "EdgeTutor" in prompt

    def test_age_7_uses_simple_language(self):
        """Age 7 prompt should mention simple words."""
        prompt = build_system_prompt(age="7")
        assert "simple" in prompt.lower() or "7-year-old" in prompt.lower()

    def test_age_16_uses_advanced_language(self):
        """Age 16 prompt should mention advanced vocabulary."""
        prompt = build_system_prompt(age="16")
        assert "advanced" in prompt.lower() or "16-year-old" in prompt.lower()

    def test_math_subject(self):
        """Math subject should include step-by-step instructions."""
        prompt = build_system_prompt(subject="math")
        assert "step" in prompt.lower()

    def test_socratic_mode_on(self):
        """Non-parent mode should use Socratic method."""
        prompt = build_system_prompt(parent_mode=False)
        assert "Socratic" in prompt

    def test_socratic_mode_off_in_parent_mode(self):
        """Parent mode should disable Socratic method."""
        prompt = build_system_prompt(parent_mode=True)
        assert "direct" in prompt.lower()

    def test_safety_included_by_default(self):
        """Safety instruction should be included by default."""
        prompt = build_system_prompt(safety_enabled=True)
        assert "REFUSE" in prompt or "refuse" in prompt.lower()

    def test_safety_can_be_disabled(self):
        """Safety instruction should be removable."""
        prompt = build_system_prompt(safety_enabled=False)
        assert "REFUSE" not in prompt

    def test_all_age_tones_exist(self):
        """All age tones should be defined."""
        assert "7" in AGE_TONES
        assert "10" in AGE_TONES
        assert "16" in AGE_TONES

    def test_all_subjects_exist(self):
        """All subject instructions should be defined."""
        for subj in ["math", "reading", "science", "general"]:
            assert subj in SUBJECT_INSTRUCTIONS


class TestBuildVisionPrompt:
    """Test vision/OCR prompt building."""

    def test_includes_ocr_text(self):
        """Vision prompt should include the OCR text."""
        prompt = build_vision_prompt("Hello world from the worksheet")
        assert "Hello world from the worksheet" in prompt

    def test_math_mode(self):
        """Math mode should include step-by-step instructions."""
        prompt = build_vision_prompt("2 + 3 = ?", is_math=True)
        assert "step" in prompt.lower()

    def test_general_mode(self):
        """General mode should include summarize instructions."""
        prompt = build_vision_prompt("The cell is the basic unit of life.", is_math=False)
        assert "Summarize" in prompt or "key points" in prompt.lower()


class TestBuildQuizPrompt:
    """Test quiz prompt building."""

    def test_contains_topic(self):
        """Quiz prompt should include the requested topic."""
        prompt = build_quiz_prompt(topic="fractions", subject="math", age="10")
        assert "fractions" in prompt

    def test_contains_age(self):
        """Quiz prompt should mention the age level."""
        prompt = build_quiz_prompt(topic="planets", age="7")
        assert "7" in prompt

    def test_with_rag_context(self):
        """Quiz with RAG context should include it."""
        prompt = build_quiz_prompt(
            topic="photosynthesis",
            rag_context="Plants use sunlight to make food.",
        )
        assert "Plants use sunlight" in prompt


class TestBuildRagContext:
    """Test RAG context wrapping."""

    def test_wraps_context(self):
        """RAG context should be wrapped in template."""
        result = build_rag_context("Some reference material")
        assert "Some reference material" in result
        assert "curriculum" in result.lower() or "reference" in result.lower()

    def test_empty_context_returns_empty(self):
        """Empty context should return empty string."""
        assert build_rag_context("") == ""
        assert build_rag_context("   ") == ""

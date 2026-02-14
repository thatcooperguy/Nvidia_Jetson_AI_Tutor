"""Tests for edgetutor.core.settings."""

from pathlib import Path


class TestSettings:
    """Test the Settings configuration class."""

    def test_default_values(self):
        """Settings should have sensible defaults without any .env file."""
        from edgetutor.core.settings import Settings

        # Create with no env file
        s = Settings(_env_file="nonexistent.env")
        assert s.host == "0.0.0.0"
        assert s.port == 7860
        assert s.llm_context_size == 2048
        assert s.llm_max_tokens == 512
        assert s.stt_enabled is True
        assert s.tts_enabled is True
        assert s.safety_enabled is True
        assert s.log_level == "INFO"

    def test_age_mode_enum(self):
        """AgeMode enum should have correct values."""
        from edgetutor.core.settings import AgeMode

        assert AgeMode.YOUNG.value == "7"
        assert AgeMode.TWEEN.value == "10"
        assert AgeMode.TEEN.value == "16"

    def test_subject_mode_enum(self):
        """SubjectMode enum should have correct values."""
        from edgetutor.core.settings import SubjectMode

        assert SubjectMode.MATH.value == "math"
        assert SubjectMode.READING.value == "reading"
        assert SubjectMode.SCIENCE.value == "science"
        assert SubjectMode.GENERAL.value == "general"

    def test_resolve_path_relative(self):
        """resolve_path should resolve relative paths from repo root."""
        from edgetutor.core.settings import REPO_ROOT, Settings

        s = Settings(_env_file="nonexistent.env")
        resolved = s.resolve_path("models/test.gguf")
        assert resolved == REPO_ROOT / "models" / "test.gguf"

    def test_resolve_path_absolute(self):
        """resolve_path should return absolute paths unchanged."""
        from edgetutor.core.settings import Settings

        s = Settings(_env_file="nonexistent.env")
        resolved = s.resolve_path("/tmp/test.gguf")
        assert resolved == Path("/tmp/test.gguf")

    def test_env_override(self, monkeypatch):
        """Settings should pick up environment variable overrides."""
        from edgetutor.core.settings import Settings

        monkeypatch.setenv("EDGETUTOR_PORT", "9999")
        monkeypatch.setenv("LLM_CONTEXT_SIZE", "4096")

        s = Settings(_env_file="nonexistent.env")
        assert s.port == 9999
        assert s.llm_context_size == 4096

    def test_runtime_state_defaults(self):
        """Runtime state should default to safe values."""
        from edgetutor.core.settings import AgeMode, Settings, SubjectMode

        s = Settings(_env_file="nonexistent.env")
        assert s.age_mode == AgeMode.TWEEN
        assert s.subject_mode == SubjectMode.GENERAL
        assert s.parent_mode is False
        assert s.quiz_mode is False

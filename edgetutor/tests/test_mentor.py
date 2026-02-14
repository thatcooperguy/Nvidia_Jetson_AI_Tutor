"""Tests for edgetutor.core.mentor — AI Mentor mode."""

from edgetutor.core.mentor import (
    MENTOR_TOPICS,
    build_mentor_prompt,
    get_mentor_topic_list,
    get_mentor_topic_prompt,
    get_system_stats_text,
)


class TestMentorPrompt:
    """Test mentor system prompt building."""

    def test_prompt_contains_ai_mentor(self):
        """Mentor prompt should identify the AI Mentor role."""
        prompt = build_mentor_prompt()
        assert "Mentor" in prompt or "mentor" in prompt

    def test_prompt_includes_system_stats(self):
        """Mentor prompt should include system stats."""
        prompt = build_mentor_prompt()
        # Should have some system stat info (even if "unavailable")
        assert "Device" in prompt or "unavailable" in prompt.lower() or "Board" in prompt

    def test_prompt_respects_age(self):
        """Mentor prompt should adapt to age."""
        prompt_7 = build_mentor_prompt(age="7")
        prompt_16 = build_mentor_prompt(age="16")
        assert "7-year-old" in prompt_7 or "simple" in prompt_7.lower()
        assert "16-year-old" in prompt_16 or "advanced" in prompt_16.lower()


class TestMentorTopics:
    """Test mentor topic library."""

    def test_topics_exist(self):
        """Should have at least 8 topics."""
        assert len(MENTOR_TOPICS) >= 8

    def test_key_topics_present(self):
        """Key topics should be present."""
        expected = ["gpu", "cuda", "llm", "quantization", "this_device"]
        for key in expected:
            assert key in MENTOR_TOPICS, f"Missing topic: {key}"

    def test_all_topics_have_title_and_prompt(self):
        """Every topic should have title and prompt."""
        for key, topic in MENTOR_TOPICS.items():
            assert "title" in topic, f"Topic {key} missing title"
            assert "prompt" in topic, f"Topic {key} missing prompt"
            assert len(topic["title"]) > 0
            assert len(topic["prompt"]) > 10

    def test_topic_list(self):
        """get_mentor_topic_list should return formatted list."""
        topics = get_mentor_topic_list()
        assert len(topics) == len(MENTOR_TOPICS)
        for item in topics:
            assert "key" in item
            assert "title" in item

    def test_get_topic_prompt(self):
        """Should return the prompt for a known topic."""
        prompt = get_mentor_topic_prompt("gpu")
        assert "GPU" in prompt

    def test_get_unknown_topic_prompt(self):
        """Unknown topic should return a generic prompt."""
        prompt = get_mentor_topic_prompt("quantum_computing")
        assert "quantum_computing" in prompt


class TestSystemStats:
    """Test system stats text generation."""

    def test_returns_string(self):
        """Should return a non-empty string."""
        stats = get_system_stats_text()
        assert isinstance(stats, str)
        assert len(stats) > 0

    def test_includes_device_info(self):
        """Stats should mention device info."""
        stats = get_system_stats_text()
        # On non-Jetson systems, should still have some info
        assert "Device" in stats or "unavailable" in stats.lower()

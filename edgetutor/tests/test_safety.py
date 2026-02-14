"""Tests for edgetutor.core.safety — kid-safe guardrail layer."""

from edgetutor.core.safety import (
    REDIRECT_MESSAGE,
    check_input_safety,
    check_output_safety,
    get_redirect_message,
)


class TestInputSafety:
    """Test input (user query) safety filtering."""

    def test_safe_input(self):
        """Normal educational questions should be safe."""
        safe_inputs = [
            "What is 2 + 2?",
            "Explain photosynthesis",
            "Help me with my homework",
            "What are the planets in the solar system?",
            "Can you quiz me on fractions?",
            "I don't understand this math problem",
            "What does the word 'benevolent' mean?",
        ]
        for text in safe_inputs:
            is_safe, category = check_input_safety(text)
            assert is_safe, f"'{text}' should be safe but was blocked ({category})"

    def test_blocks_weapons(self):
        """Weapon-related queries should be blocked."""
        is_safe, category = check_input_safety("how to make a bomb")
        assert not is_safe
        assert category == "weapons"

    def test_blocks_violence(self):
        """Violence-related queries should be blocked."""
        is_safe, category = check_input_safety("how to kill someone")
        assert not is_safe
        assert category == "violence"

    def test_blocks_self_harm(self):
        """Self-harm queries should be blocked."""
        is_safe, category = check_input_safety("how to commit suicide")
        assert not is_safe
        assert category == "self-harm"

    def test_blocks_explicit(self):
        """Explicit content queries should be blocked."""
        is_safe, category = check_input_safety("show me porn")
        assert not is_safe
        assert category == "explicit"

    def test_blocks_drugs(self):
        """Drug-related queries should be blocked."""
        is_safe, category = check_input_safety("how to make meth")
        assert not is_safe
        assert category == "drugs"

    def test_empty_input_is_safe(self):
        """Empty or whitespace input should be considered safe."""
        is_safe, _ = check_input_safety("")
        assert is_safe
        is_safe, _ = check_input_safety("   ")
        assert is_safe

    def test_case_insensitive(self):
        """Safety checks should be case-insensitive."""
        is_safe, _ = check_input_safety("HOW TO MAKE A BOMB")
        assert not is_safe


class TestOutputSafety:
    """Test output (LLM response) safety scrubbing."""

    def test_safe_output_unchanged(self):
        """Safe output should pass through unchanged."""
        text = "Great question! 2 + 2 = 4. Now try 3 + 3!"
        assert check_output_safety(text) == text

    def test_blocks_harmful_instructions(self):
        """Output containing harmful instructions should be replaced."""
        text = "Here is how to harm someone: step 1..."
        result = check_output_safety(text)
        assert result == REDIRECT_MESSAGE

    def test_empty_output(self):
        """Empty output should return empty."""
        assert check_output_safety("") == ""
        assert check_output_safety(None) is None


class TestRedirectMessage:
    """Test the redirect message."""

    def test_redirect_message_is_encouraging(self):
        """Redirect message should be kid-friendly and encouraging."""
        msg = get_redirect_message()
        assert "learn" in msg.lower()
        assert "fun" in msg.lower() or "educational" in msg.lower()

    def test_redirect_message_offers_alternatives(self):
        """Redirect should suggest safe alternatives."""
        msg = get_redirect_message()
        assert "Math" in msg or "Reading" in msg or "Science" in msg

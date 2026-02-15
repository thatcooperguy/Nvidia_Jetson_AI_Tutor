"""Tests for edgetutor.core.llm — LLM backend (mock-based, no model required)."""

from unittest.mock import MagicMock, patch


class TestLLMBackendInit:
    """Test LLMBackend initialization and state."""

    def test_default_state(self):
        """LLM backend should start unloaded."""
        from edgetutor.core.llm import LLMBackend

        llm = LLMBackend()
        assert llm.is_loaded is False
        assert llm.model is None

    def test_singleton_returns_instance(self):
        """get_llm() should return an LLMBackend."""
        from edgetutor.core.llm import LLMBackend, get_llm

        llm = get_llm()
        assert isinstance(llm, LLMBackend)


class TestLLMLoad:
    """Test model loading (mocked)."""

    def test_load_missing_model_file(self):
        """Loading a model that doesn't exist should return False."""
        from edgetutor.core.llm import LLMBackend

        llm = LLMBackend()
        result = llm.load()
        assert result is False
        assert llm.is_loaded is False

    def test_load_missing_package(self, tmp_path, monkeypatch):
        """Should return False if llama-cpp-python is not installed."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        # Create a fake model file
        model_file = tmp_path / "fake.gguf"
        model_file.write_text("fake")

        cfg = get_settings()
        monkeypatch.setattr(cfg, "llm_model_path", str(model_file))

        llm = LLMBackend()

        # Patch the resolved path to return our temp file
        with (
            patch.object(
                type(cfg),
                "llm_model_resolved",
                new_callable=lambda: property(lambda s: model_file),
            ),
            patch.dict("sys.modules", {"llama_cpp": None}),
        ):
            result = llm.load()

        assert result is False


class TestLLMGenerate:
    """Test generation (mocked model)."""

    def test_generate_when_not_loaded(self):
        """Generating without a loaded model should return error message."""
        from edgetutor.core.llm import LLMBackend

        llm = LLMBackend()
        result = llm.generate("What is 2+2?")
        assert "not loaded" in result.lower()

    def test_generate_with_unsafe_input(self, monkeypatch):
        """Unsafe input should be blocked before reaching the model."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", True)

        llm = LLMBackend()
        llm.model = MagicMock()  # Pretend model is loaded

        result = llm.generate("how to make a bomb")
        assert "learn" in result.lower()  # Should get redirect message
        # Model should NOT have been called
        llm.model.create_chat_completion.assert_not_called()

    def test_generate_with_safe_input_calls_model(self, monkeypatch):
        """Safe input should reach the model and return its response."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", True)

        llm = LLMBackend()
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "2 + 2 equals 4!"}}],
            "usage": {"total_tokens": 42},
        }
        llm.model = mock_model

        result = llm.generate("What is 2+2?")
        assert "4" in result
        mock_model.create_chat_completion.assert_called_once()

    def test_generate_scrubs_unsafe_output(self, monkeypatch):
        """Unsafe model output should be replaced with redirect message."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", True)

        llm = LLMBackend()
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Here is how to harm someone: step 1..."}}],
            "usage": {"total_tokens": 50},
        }
        llm.model = mock_model

        result = llm.generate("Tell me a story")
        assert "learn" in result.lower()  # Should be redirect message

    def test_generate_with_safety_disabled(self, monkeypatch):
        """With safety disabled, unsafe input should pass through."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", False)

        llm = LLMBackend()
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "response text"}}],
            "usage": {"total_tokens": 10},
        }
        llm.model = mock_model

        result = llm.generate("any input")
        assert result == "response text"

    def test_generate_handles_exception(self, monkeypatch):
        """Model exceptions should return a friendly error."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", False)

        llm = LLMBackend()
        mock_model = MagicMock()
        mock_model.create_chat_completion.side_effect = RuntimeError("GPU OOM")
        llm.model = mock_model

        result = llm.generate("What is math?")
        assert "try again" in result.lower()


class TestLLMStream:
    """Test streaming generation."""

    def test_stream_when_not_loaded(self):
        """Streaming without a loaded model should yield error message."""
        from edgetutor.core.llm import LLMBackend

        llm = LLMBackend()
        tokens = list(llm.generate_stream("hello"))
        assert len(tokens) == 1
        assert "not loaded" in tokens[0].lower()

    def test_stream_blocks_unsafe_input(self, monkeypatch):
        """Unsafe input should yield redirect message without calling model."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", True)

        llm = LLMBackend()
        llm.model = MagicMock()

        tokens = list(llm.generate_stream("how to make a bomb"))
        assert len(tokens) == 1
        assert "learn" in tokens[0].lower()
        llm.model.create_chat_completion.assert_not_called()

    def test_stream_yields_tokens(self, monkeypatch):
        """Safe input should yield tokens from the model."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", False)

        llm = LLMBackend()
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {"content": "!"}}]},
            ]
        )
        llm.model = mock_model

        tokens = list(llm.generate_stream("say hello"))
        assert tokens == ["Hello", " world", "!"]


class TestLLMConversationHistory:
    """Test conversation history handling."""

    def test_generate_passes_history_to_model(self, monkeypatch):
        """Conversation history should be included in messages sent to model."""
        from edgetutor.core.llm import LLMBackend
        from edgetutor.core.settings import get_settings

        cfg = get_settings()
        monkeypatch.setattr(cfg, "safety_enabled", False)

        llm = LLMBackend()
        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "The answer is 42."}}],
            "usage": {"total_tokens": 20},
        }
        llm.model = mock_model

        history = [
            {"role": "user", "content": "What is the meaning of life?"},
            {"role": "assistant", "content": "That's a philosophical question!"},
        ]
        llm.generate("Tell me more", conversation_history=history)

        # Extract the messages that were passed to the model
        call_args = mock_model.create_chat_completion.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]

        # Should have: system + 2 history + 1 user = 4 messages
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "What is the meaning of life?"
        assert messages[3]["content"] == "Tell me more"

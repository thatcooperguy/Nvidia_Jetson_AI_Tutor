"""
EdgeTutor AI — LLM backend (llama.cpp via llama-cpp-python).

Provides a thin wrapper around a local GGUF model with streaming support,
system prompt injection, and safety filtering.
"""

from __future__ import annotations

import time
from collections.abc import Generator

from edgetutor.core.logging_config import get_logger
from edgetutor.core.prompts import build_system_prompt
from edgetutor.core.safety import check_input_safety, check_output_safety, get_redirect_message
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)

# Lazy-loaded model instance
_llm_instance = None


class LLMBackend:
    """Wrapper around llama-cpp-python for local inference."""

    def __init__(self):
        self.model = None
        self._load_time: float = 0.0

    def load(self) -> bool:
        """Load the GGUF model. Returns True on success."""
        cfg = get_settings()
        model_path = cfg.llm_model_resolved

        if not model_path.exists():
            logger.error("LLM model not found at %s", model_path)
            logger.error(
                "Run 'scripts/download_model.sh' or place a .gguf file at %s",
                model_path,
            )
            return False

        try:
            from llama_cpp import Llama

            t0 = time.time()
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=cfg.llm_context_size,
                n_gpu_layers=cfg.llm_n_gpu_layers,
                verbose=cfg.log_level.upper() == "DEBUG",
            )
            self._load_time = time.time() - t0
            logger.info(
                "LLM loaded in %.1fs — model=%s, ctx=%d, gpu_layers=%d",
                self._load_time,
                model_path.name,
                cfg.llm_context_size,
                cfg.llm_n_gpu_layers,
            )
            return True
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error("Failed to load LLM: %s", e)
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def generate(
        self,
        user_message: str,
        system_prompt: str | None = None,
        conversation_history: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Args:
            user_message: The user's input text.
            system_prompt: Override system prompt (otherwise built from settings).
            conversation_history: Prior messages as [{"role": ..., "content": ...}].
            max_tokens: Override max tokens.
            temperature: Override temperature.

        Returns:
            The assistant's response text.
        """
        if not self.is_loaded:
            return "[EdgeTutor] The language model is not loaded. Please check the setup."

        cfg = get_settings()

        # Safety check on input
        if cfg.safety_enabled:
            is_safe, category = check_input_safety(user_message)
            if not is_safe:
                logger.info("Input blocked by safety filter: category=%s", category)
                return get_redirect_message()

        # Build messages
        if system_prompt is None:
            system_prompt = build_system_prompt(
                age=cfg.age_mode.value,
                subject=cfg.subject_mode.value,
                parent_mode=cfg.parent_mode,
                safety_enabled=cfg.safety_enabled,
            )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            t0 = time.time()
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or cfg.llm_max_tokens,
                temperature=temperature or cfg.llm_temperature,
            )
            elapsed = time.time() - t0

            text = response["choices"][0]["message"]["content"].strip()
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            logger.info("LLM response: %.1fs, %d tokens", elapsed, tokens_used)

            # Safety check on output
            if cfg.safety_enabled:
                text = check_output_safety(text)

            return text

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return "[EdgeTutor] Sorry, I had trouble thinking about that. Could you try again?"

    def generate_stream(
        self,
        user_message: str,
        system_prompt: str | None = None,
        conversation_history: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens one-by-one (generator).

        Yields individual tokens as they are produced.
        """
        if not self.is_loaded:
            yield "[EdgeTutor] The language model is not loaded. Please check the setup."
            return

        cfg = get_settings()

        # Safety check on input
        if cfg.safety_enabled:
            is_safe, category = check_input_safety(user_message)
            if not is_safe:
                yield get_redirect_message()
                return

        if system_prompt is None:
            system_prompt = build_system_prompt(
                age=cfg.age_mode.value,
                subject=cfg.subject_mode.value,
                parent_mode=cfg.parent_mode,
                safety_enabled=cfg.safety_enabled,
            )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            t0 = time.time()
            full_text = ""
            for chunk in self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or cfg.llm_max_tokens,
                temperature=temperature or cfg.llm_temperature,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text += token
                    yield token

            elapsed = time.time() - t0
            logger.info("LLM stream complete: %.1fs, ~%d chars", elapsed, len(full_text))

        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            yield "\n[EdgeTutor] Sorry, I had trouble there. Could you try again?"


def get_llm() -> LLMBackend:
    """Return the singleton LLM backend."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMBackend()
    return _llm_instance

"""
EdgeTutor AI — Tutor Orchestrator.

Central coordinator that takes user input (text, audio, image) and settings,
routes through the appropriate modules, and produces a tutor response.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from dataclasses import dataclass, field

from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)


@dataclass
class TutorRequest:
    """Input to the orchestrator."""

    user_text: str = ""
    audio_path: str | None = None
    audio_array: object | None = None  # numpy array from Gradio
    audio_sample_rate: int = 16000
    image: object | None = None  # PIL Image or numpy array
    settings_override: dict | None = None


@dataclass
class TutorResponse:
    """Output from the orchestrator."""

    text: str = ""
    audio: tuple | None = None  # (sample_rate, numpy_array) for Gradio
    ocr_text: str = ""
    has_math: bool = False
    math_expressions: list[str] = field(default_factory=list)
    quiz_questions: str = ""
    suggestions: list[str] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class TutorOrchestrator:
    """
    Main orchestrator: takes a TutorRequest, calls modules, returns TutorResponse.
    """

    def __init__(self):
        self._modules_loaded = False
        self._llm = None
        self._stt = None
        self._tts = None
        self._ocr = None
        self._rag = None

    def load_modules(self) -> dict:
        """
        Load all available modules. Returns status dict.

        This is called once at startup. Individual modules failing
        does not prevent the app from running — graceful degradation.
        """
        status = {}

        # LLM (critical)
        try:
            from edgetutor.core.llm import get_llm

            self._llm = get_llm()
            loaded = self._llm.load()
            status["llm"] = "ready" if loaded else "not loaded (model missing?)"
        except Exception as e:
            status["llm"] = f"error: {e}"
            logger.error("LLM load failed: %s", e)

        # STT (optional)
        try:
            from edgetutor.audio.stt import get_stt

            self._stt = get_stt()
            loaded = self._stt.load()
            status["stt"] = "ready" if loaded else "disabled or failed"
        except Exception as e:
            status["stt"] = f"error: {e}"
            logger.error("STT load failed: %s", e)

        # TTS (optional)
        try:
            from edgetutor.audio.tts import get_tts

            self._tts = get_tts()
            loaded = self._tts.load()
            status["tts"] = "ready" if loaded else "disabled or failed"
        except Exception as e:
            status["tts"] = f"error: {e}"
            logger.error("TTS load failed: %s", e)

        # OCR (optional)
        try:
            from edgetutor.vision.ocr import get_ocr

            self._ocr = get_ocr()
            status["ocr"] = "ready" if self._ocr.is_available else "tesseract not found"
        except Exception as e:
            status["ocr"] = f"error: {e}"
            logger.error("OCR load failed: %s", e)

        # RAG (optional)
        try:
            from edgetutor.core.rag import get_rag

            self._rag = get_rag()
            emb_ok = self._rag.load_embedder()
            idx_ok = self._rag.load_index() if emb_ok else False
            if emb_ok and idx_ok:
                status["rag"] = "ready"
            elif emb_ok:
                status["rag"] = "embedder ready, no index (run ingest)"
            else:
                status["rag"] = "embedder failed"
        except Exception as e:
            status["rag"] = f"error: {e}"
            logger.error("RAG load failed: %s", e)

        self._modules_loaded = True
        logger.info("Module status: %s", status)
        return status

    def process(
        self,
        request: TutorRequest,
        conversation_history: list[dict] | None = None,
    ) -> TutorResponse:
        """
        Process a tutor request end-to-end (non-streaming).
        """
        response = TutorResponse()
        cfg = get_settings()
        total_t0 = time.time()

        # Apply settings overrides if any
        if request.settings_override:
            self._apply_settings(request.settings_override)

        # ─── Step 1: Speech-to-text (if audio provided) ──────────────────
        user_text = request.user_text
        if request.audio_array is not None and self._stt and self._stt.is_ready:
            t0 = time.time()
            result = self._stt.transcribe_numpy(
                request.audio_array, request.audio_sample_rate
            )
            response.latency["stt"] = time.time() - t0
            if result.get("text"):
                user_text = result["text"]
                logger.info("STT transcribed: '%s'", user_text[:100])
            elif result.get("error"):
                response.errors.append(f"STT: {result['error']}")
        elif request.audio_path and self._stt and self._stt.is_ready:
            t0 = time.time()
            result = self._stt.transcribe(request.audio_path)
            response.latency["stt"] = time.time() - t0
            if result.get("text"):
                user_text = result["text"]
            elif result.get("error"):
                response.errors.append(f"STT: {result['error']}")

        # ─── Step 2: OCR (if image provided) ─────────────────────────────
        ocr_text = ""
        if request.image is not None and self._ocr and self._ocr.is_available:
            t0 = time.time()
            ocr_result = self._ocr.extract_text(request.image)
            response.latency["ocr"] = time.time() - t0

            if ocr_result.get("error"):
                response.errors.append(f"OCR: {ocr_result['error']}")
            else:
                ocr_text = ocr_result["text"]
                response.ocr_text = ocr_text
                response.has_math = ocr_result["has_math"]
                response.math_expressions = ocr_result["math_expressions"]

        # ─── Step 3: RAG retrieval (if relevant) ─────────────────────────
        rag_context = ""
        query_for_rag = user_text or ocr_text
        if query_for_rag and self._rag and self._rag.is_ready:
            t0 = time.time()
            results = self._rag.query(query_for_rag)
            response.latency["rag"] = time.time() - t0
            if results:
                rag_context = "\n\n".join(r["text"] for r in results)

        # ─── Step 4: Build the prompt and call LLM ───────────────────────
        if self._llm and self._llm.is_loaded:
            from edgetutor.core.prompts import (
                build_rag_context,
                build_quiz_prompt,
                build_vision_prompt,
            )

            # Compose the user message
            final_message = ""

            if ocr_text:
                # Vision mode: wrap OCR text in a vision prompt
                final_message = build_vision_prompt(
                    ocr_text=ocr_text,
                    is_math=response.has_math,
                )
                if user_text:
                    final_message += f"\n\nThe student also asks: {user_text}"
            elif cfg.quiz_mode and user_text:
                # Quiz mode
                final_message = build_quiz_prompt(
                    topic=user_text,
                    subject=cfg.subject_mode.value,
                    age=cfg.age_mode.value,
                    rag_context=rag_context,
                )
            else:
                # Normal chat mode
                final_message = user_text
                if rag_context:
                    final_message = build_rag_context(rag_context) + "\n\n" + user_text

            if final_message:
                t0 = time.time()
                text_response = self._llm.generate(
                    user_message=final_message,
                    conversation_history=conversation_history,
                )
                response.latency["llm"] = time.time() - t0
                response.text = text_response
            else:
                response.text = (
                    "Hi there! I'm EdgeTutor, your study buddy. "
                    "Type a question, speak using the microphone, or scan a worksheet!"
                )
        else:
            response.text = (
                "[EdgeTutor] The language model isn't loaded yet. "
                "Check that a .gguf model file is in the models/ folder "
                "and restart the app."
            )

        # ─── Step 5: Text-to-speech (if enabled) ─────────────────────────
        if self._tts and self._tts.is_ready and response.text:
            t0 = time.time()
            audio_tuple = self._tts.get_audio_tuple(response.text)
            response.latency["tts"] = time.time() - t0
            response.audio = audio_tuple

        # ─── Step 6: Suggestions ──────────────────────────────────────────
        response.suggestions = self._generate_suggestions(
            user_text, ocr_text, cfg.subject_mode.value
        )

        response.latency["total"] = time.time() - total_t0
        logger.info("Orchestrator total: %.1fs — latency=%s", response.latency["total"], response.latency)

        return response

    def process_stream(
        self,
        request: TutorRequest,
        conversation_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Process a tutor request with streaming LLM output.

        Yields text tokens as they are generated.
        Note: STT and OCR still run synchronously before streaming starts.
        """
        cfg = get_settings()

        if request.settings_override:
            self._apply_settings(request.settings_override)

        # STT
        user_text = request.user_text
        if request.audio_array is not None and self._stt and self._stt.is_ready:
            result = self._stt.transcribe_numpy(request.audio_array, request.audio_sample_rate)
            if result.get("text"):
                user_text = result["text"]
        elif request.audio_path and self._stt and self._stt.is_ready:
            result = self._stt.transcribe(request.audio_path)
            if result.get("text"):
                user_text = result["text"]

        # OCR
        ocr_text = ""
        has_math = False
        if request.image is not None and self._ocr and self._ocr.is_available:
            ocr_result = self._ocr.extract_text(request.image)
            if not ocr_result.get("error"):
                ocr_text = ocr_result["text"]
                has_math = ocr_result["has_math"]

        # RAG
        rag_context = ""
        query_for_rag = user_text or ocr_text
        if query_for_rag and self._rag and self._rag.is_ready:
            results = self._rag.query(query_for_rag)
            if results:
                rag_context = "\n\n".join(r["text"] for r in results)

        # Build final message
        if not self._llm or not self._llm.is_loaded:
            yield "[EdgeTutor] Language model not loaded. Check models/ folder."
            return

        from edgetutor.core.prompts import build_rag_context, build_quiz_prompt, build_vision_prompt

        if ocr_text:
            final_message = build_vision_prompt(ocr_text=ocr_text, is_math=has_math)
            if user_text:
                final_message += f"\n\nThe student also asks: {user_text}"
        elif cfg.quiz_mode and user_text:
            final_message = build_quiz_prompt(
                topic=user_text,
                subject=cfg.subject_mode.value,
                age=cfg.age_mode.value,
                rag_context=rag_context,
            )
        else:
            final_message = user_text
            if rag_context:
                final_message = build_rag_context(rag_context) + "\n\n" + user_text

        if not final_message:
            yield "Hi! I'm EdgeTutor. Ask me anything or scan a worksheet!"
            return

        # Stream LLM
        yield from self._llm.generate_stream(
            user_message=final_message,
            conversation_history=conversation_history,
        )

    def _apply_settings(self, overrides: dict) -> None:
        """Apply runtime setting overrides."""
        cfg = get_settings()
        from edgetutor.core.settings import AgeMode, SubjectMode

        if "age_mode" in overrides:
            try:
                cfg.age_mode = AgeMode(str(overrides["age_mode"]))
            except ValueError:
                pass
        if "subject_mode" in overrides:
            try:
                cfg.subject_mode = SubjectMode(overrides["subject_mode"])
            except ValueError:
                pass
        if "parent_mode" in overrides:
            cfg.parent_mode = bool(overrides["parent_mode"])
        if "quiz_mode" in overrides:
            cfg.quiz_mode = bool(overrides["quiz_mode"])

    def _generate_suggestions(
        self, user_text: str, ocr_text: str, subject: str
    ) -> list[str]:
        """Generate 'next steps' suggestions based on context."""
        suggestions = []

        if ocr_text:
            suggestions.append("Scan another page")
            suggestions.append("Ask me to explain a specific part")
            if subject == "math":
                suggestions.append("Want me to check your work?")

        if not user_text and not ocr_text:
            suggestions = [
                "Ask me a question about any school subject",
                "Scan a worksheet or textbook page",
                "Try quiz mode to test your knowledge!",
            ]

        return suggestions


# Singleton
_orchestrator_instance = None


def get_orchestrator() -> TutorOrchestrator:
    """Return the singleton orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TutorOrchestrator()
    return _orchestrator_instance

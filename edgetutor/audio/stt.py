"""
EdgeTutor AI — Speech-to-Text module (faster-whisper).

Provides offline transcription using CTranslate2-accelerated Whisper models.
Supports CUDA on Jetson for fast inference.
"""

from __future__ import annotations

import time
from pathlib import Path

from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)

_stt_instance = None


class STTEngine:
    """Speech-to-text using faster-whisper."""

    def __init__(self):
        self.model = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> bool:
        """Load the Whisper model."""
        cfg = get_settings()

        if not cfg.stt_enabled:
            logger.info("STT is disabled in settings.")
            return False

        try:
            from faster_whisper import WhisperModel

            t0 = time.time()
            self.model = WhisperModel(
                cfg.stt_model_size,
                device=cfg.stt_device,
                compute_type=cfg.stt_compute_type,
            )
            elapsed = time.time() - t0
            logger.info(
                "STT loaded in %.1fs — model=%s, device=%s, compute=%s",
                elapsed,
                cfg.stt_model_size,
                cfg.stt_device,
                cfg.stt_compute_type,
            )
            self._ready = True
            return True

        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            return False
        except Exception as e:
            logger.error("Failed to load STT model: %s", e)
            # Try CPU fallback
            if cfg.stt_device == "cuda":
                logger.info("Retrying STT with CPU fallback...")
                try:
                    from faster_whisper import WhisperModel

                    self.model = WhisperModel(
                        cfg.stt_model_size,
                        device="cpu",
                        compute_type="int8",
                    )
                    self._ready = True
                    logger.info("STT loaded on CPU (fallback).")
                    return True
                except Exception as e2:
                    logger.error("CPU fallback also failed: %s", e2)
            return False

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.).
            language: Override language code (e.g., "en"). None = auto-detect.

        Returns:
            dict with keys: "text", "language", "duration_s", "elapsed_s"
        """
        if not self._ready or self.model is None:
            return {
                "text": "",
                "language": "",
                "duration_s": 0.0,
                "elapsed_s": 0.0,
                "error": "STT not loaded",
            }

        cfg = get_settings()
        lang = language or (cfg.stt_language if cfg.stt_language else None)

        try:
            t0 = time.time()
            segments, info = self.model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=5,
                vad_filter=True,  # Voice activity detection — skip silence
            )

            # Collect all segment texts
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = " ".join(text_parts)
            elapsed = time.time() - t0

            logger.info(
                "STT: %.1fs audio → '%s' (%.1fs, lang=%s)",
                info.duration,
                full_text[:80],
                elapsed,
                info.language,
            )

            return {
                "text": full_text,
                "language": info.language,
                "duration_s": info.duration,
                "elapsed_s": elapsed,
            }

        except Exception as e:
            logger.error("STT transcription failed: %s", e)
            return {
                "text": "",
                "language": "",
                "duration_s": 0.0,
                "elapsed_s": 0.0,
                "error": str(e),
            }

    def transcribe_numpy(
        self,
        audio_array,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe from a numpy array (e.g., from Gradio audio input).

        Args:
            audio_array: numpy array of audio samples.
            sample_rate: Sample rate (default 16kHz for Whisper).
            language: Override language code.

        Returns:
            Same dict as transcribe().
        """
        if not self._ready or self.model is None:
            return {
                "text": "",
                "language": "",
                "duration_s": 0.0,
                "elapsed_s": 0.0,
                "error": "STT not loaded",
            }

        import numpy as np

        cfg = get_settings()
        lang = language or (cfg.stt_language if cfg.stt_language else None)

        try:
            # Ensure float32 mono
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            # Normalize to [-1, 1]
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val

            t0 = time.time()
            segments, info = self.model.transcribe(
                audio_array,
                language=lang,
                beam_size=5,
                vad_filter=True,
            )

            text_parts = [seg.text.strip() for seg in segments]
            full_text = " ".join(text_parts)
            elapsed = time.time() - t0

            logger.info("STT (numpy): '%.80s' (%.1fs)", full_text, elapsed)

            return {
                "text": full_text,
                "language": info.language,
                "duration_s": info.duration,
                "elapsed_s": elapsed,
            }

        except Exception as e:
            logger.error("STT numpy transcription failed: %s", e)
            return {
                "text": "",
                "language": "",
                "duration_s": 0.0,
                "elapsed_s": 0.0,
                "error": str(e),
            }


def get_stt() -> STTEngine:
    """Return the singleton STT engine."""
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = STTEngine()
    return _stt_instance

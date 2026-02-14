"""
EdgeTutor AI — Text-to-Speech module (Piper TTS).

Provides fast offline speech synthesis using Piper's ONNX-based VITS models.
Outputs WAV audio that can be played in the Gradio UI or saved to file.
"""

from __future__ import annotations

import io
import struct
import time
import wave
from pathlib import Path
from typing import Optional

from edgetutor.core.logging_config import get_logger
from edgetutor.core.settings import get_settings

logger = get_logger(__name__)

_tts_instance = None


class TTSEngine:
    """Text-to-speech using Piper TTS."""

    def __init__(self):
        self.voice = None
        self.synthesizer = None
        self._ready = False
        self._sample_rate = 22050  # Default for most Piper voices

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> bool:
        """Load the Piper voice model."""
        cfg = get_settings()

        if not cfg.tts_enabled:
            logger.info("TTS is disabled in settings.")
            return False

        voice_path = cfg.tts_voice_resolved

        if not voice_path.exists():
            logger.warning(
                "TTS voice not found at %s — TTS disabled. "
                "Run scripts/download_voice.sh to get a voice.",
                voice_path,
            )
            return False

        try:
            from piper import PiperVoice

            t0 = time.time()
            self.voice = PiperVoice.load(str(voice_path))
            self._sample_rate = self.voice.config.sample_rate
            self._ready = True
            elapsed = time.time() - t0
            logger.info(
                "TTS loaded in %.1fs — voice=%s, rate=%dHz",
                elapsed,
                voice_path.name,
                self._sample_rate,
            )
            return True

        except ImportError:
            logger.error("piper-tts not installed. Install with: pip install piper-tts")
            return False
        except Exception as e:
            logger.error("Failed to load TTS voice: %s", e)
            return False

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to WAV audio bytes.

        Args:
            text: The text to speak.

        Returns:
            WAV file content as bytes, or None on failure.
        """
        if not self._ready or self.voice is None:
            logger.debug("TTS not ready — skipping synthesis.")
            return None

        if not text or not text.strip():
            return None

        try:
            t0 = time.time()

            # Synthesize to raw PCM
            audio_buffer = io.BytesIO()
            with wave.open(audio_buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self._sample_rate)
                self.voice.synthesize(text, wav)

            wav_bytes = audio_buffer.getvalue()
            elapsed = time.time() - t0

            # Calculate duration
            data_size = len(wav_bytes) - 44  # WAV header is ~44 bytes
            duration = data_size / (self._sample_rate * 2)  # 16-bit = 2 bytes/sample

            logger.info(
                "TTS: %d chars → %.1fs audio (%.1fs elapsed)",
                len(text),
                duration,
                elapsed,
            )

            return wav_bytes

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return None

    def synthesize_to_file(self, text: str, output_path: str | Path) -> bool:
        """
        Synthesize text and save to a WAV file.

        Returns True on success.
        """
        wav_bytes = self.synthesize(text)
        if wav_bytes is None:
            return False

        try:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(wav_bytes)
            logger.info("TTS saved to %s", output)
            return True
        except Exception as e:
            logger.error("Failed to save TTS audio: %s", e)
            return False

    def get_audio_tuple(self, text: str) -> Optional[tuple]:
        """
        Synthesize and return (sample_rate, numpy_array) for Gradio.

        Returns None if TTS is not available.
        """
        wav_bytes = self.synthesize(text)
        if wav_bytes is None:
            return None

        try:
            import numpy as np

            # Parse WAV bytes to numpy
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wav:
                n_frames = wav.getnframes()
                raw = wav.readframes(n_frames)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            return (self._sample_rate, audio)
        except Exception as e:
            logger.error("Failed to convert TTS to numpy: %s", e)
            return None


def get_tts() -> TTSEngine:
    """Return the singleton TTS engine."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance

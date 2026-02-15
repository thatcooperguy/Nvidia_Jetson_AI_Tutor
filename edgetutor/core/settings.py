"""
EdgeTutor AI — Central configuration.

All settings are loaded from environment variables / .env file with sensible
defaults so the app works out-of-the-box on a fresh Jetson.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# ── Repo root (two levels up from this file) ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class AgeMode(str, Enum):
    """Target audience age bracket — controls tone and depth."""

    YOUNG = "7"
    TWEEN = "10"
    TEEN = "16"


class SubjectMode(str, Enum):
    """Active subject focus."""

    MATH = "math"
    READING = "reading"
    SCIENCE = "science"
    GENERAL = "general"


class Settings(BaseSettings):
    """Application-wide configuration with env-var overrides."""

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="EDGETUTOR_HOST")
    port: int = Field(default=7860, alias="EDGETUTOR_PORT", ge=1, le=65535)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model_path: str = Field(default="models/default.gguf", alias="LLM_MODEL_PATH")
    llm_n_gpu_layers: int = Field(default=20, alias="LLM_N_GPU_LAYERS", ge=-1, le=200)
    llm_context_size: int = Field(default=2048, alias="LLM_CONTEXT_SIZE", ge=128, le=131072)
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS", ge=1, le=32768)
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE", ge=0.0, le=2.0)

    # ── STT ───────────────────────────────────────────────────────────────────
    stt_model_size: str = Field(default="small", alias="STT_MODEL_SIZE")
    stt_device: str = Field(default="cuda", alias="STT_DEVICE")
    stt_compute_type: str = Field(default="float16", alias="STT_COMPUTE_TYPE")
    stt_language: str = Field(default="en", alias="STT_LANGUAGE")
    stt_enabled: bool = Field(default=True, alias="STT_ENABLED")

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_voice_model: str = Field(default="voices/en_US-lessac-medium.onnx", alias="TTS_VOICE_MODEL")
    tts_enabled: bool = Field(default=True, alias="TTS_ENABLED")
    tts_rate: float = Field(default=1.0, alias="TTS_RATE", ge=0.1, le=5.0)

    # ── Vision / OCR ──────────────────────────────────────────────────────────
    ocr_engine: str = Field(default="tesseract", alias="OCR_ENGINE")
    ocr_language: str = Field(default="eng", alias="OCR_LANGUAGE")
    camera_enabled: bool = Field(default=True, alias="CAMERA_ENABLED")

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_content_dir: str = Field(default="content/", alias="RAG_CONTENT_DIR")
    rag_index_dir: str = Field(default="edgetutor/core/faiss_index/", alias="RAG_INDEX_DIR")
    rag_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="RAG_EMBEDDING_MODEL"
    )
    rag_top_k: int = Field(default=3, alias="RAG_TOP_K", ge=1, le=100)
    rag_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE", ge=50, le=10000)

    # ── Conversation ─────────────────────────────────────────────────────────
    conversation_history_limit: int = Field(
        default=20, alias="CONVERSATION_HISTORY_LIMIT", ge=1, le=200
    )

    # ── Safety ────────────────────────────────────────────────────────────────
    safety_enabled: bool = Field(default=True, alias="SAFETY_ENABLED")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/edgetutor.log", alias="LOG_FILE")

    # ── Runtime state (not from env) ──────────────────────────────────────────
    age_mode: AgeMode = AgeMode.TWEEN
    subject_mode: SubjectMode = SubjectMode.GENERAL
    parent_mode: bool = False
    quiz_mode: bool = False

    model_config = {
        "env_file": str(REPO_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    # ── Helpers ───────────────────────────────────────────────────────────────
    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to the repo root."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return REPO_ROOT / p

    @property
    def llm_model_resolved(self) -> Path:
        return self.resolve_path(self.llm_model_path)

    @property
    def tts_voice_resolved(self) -> Path:
        return self.resolve_path(self.tts_voice_model)

    @property
    def rag_content_resolved(self) -> Path:
        return self.resolve_path(self.rag_content_dir)

    @property
    def rag_index_resolved(self) -> Path:
        return self.resolve_path(self.rag_index_dir)

    @property
    def log_file_resolved(self) -> Path:
        return self.resolve_path(self.log_file)


def get_settings() -> Settings:
    """Return a Settings singleton (cached)."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()
    return get_settings._instance


def reset_settings() -> None:
    """Reset the cached settings singleton (for testing)."""
    if hasattr(get_settings, "_instance"):
        del get_settings._instance

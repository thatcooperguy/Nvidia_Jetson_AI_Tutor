"""
EdgeTutor AI — Conversation persistence.

Simple JSON-based save/load for chat histories. Conversations are stored
in a configurable directory (default: ~/.edgetutor/conversations/).

This keeps chat history available across page refreshes and app restarts
while staying fully offline — no database required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from edgetutor.core.logging_config import get_logger

logger = get_logger(__name__)

# Default storage location
_DEFAULT_DIR = Path.home() / ".edgetutor" / "conversations"


class ConversationStore:
    """Persist and retrieve chat conversations as JSON files."""

    def __init__(self, storage_dir: Path | str | None = None):
        self._dir = Path(storage_dir) if storage_dir else _DEFAULT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        return self._dir

    def save(self, session_id: str, history: list[dict]) -> Path:
        """
        Save a conversation history to disk.

        Args:
            session_id: Unique identifier for this conversation.
            history: List of message dicts [{"role": ..., "content": ...}].

        Returns:
            Path to the saved JSON file.
        """
        path = self._dir / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "updated_at": time.time(),
            "messages": history,
        }
        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug("Saved conversation %s (%d messages)", session_id, len(history))
        except Exception as e:
            logger.error("Failed to save conversation %s: %s", session_id, e)
        return path

    def load(self, session_id: str) -> list[dict]:
        """
        Load a conversation history from disk.

        Returns an empty list if the file doesn't exist or is corrupt.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages", [])
            logger.debug("Loaded conversation %s (%d messages)", session_id, len(messages))
            return messages
        except Exception as e:
            logger.error("Failed to load conversation %s: %s", session_id, e)
            return []

    def list_sessions(self) -> list[dict]:
        """
        List all saved sessions, sorted by most recent first.

        Returns:
            List of dicts with keys: session_id, updated_at, message_count.
        """
        sessions = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "session_id": data.get("session_id", path.stem),
                        "updated_at": data.get("updated_at", 0),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception:
                continue
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a conversation file. Returns True if deleted."""
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            logger.debug("Deleted conversation %s", session_id)
            return True
        return False


# Module-level singleton
_store_instance: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    """Return the singleton ConversationStore."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ConversationStore()
    return _store_instance

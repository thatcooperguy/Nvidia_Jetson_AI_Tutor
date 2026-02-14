"""Tests for edgetutor.core.conversations — conversation persistence."""

import json


class TestConversationStore:
    """Test ConversationStore save/load/list/delete."""

    def test_save_and_load(self, tmp_path):
        """Should save and reload a conversation."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        history = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2 + 2 = 4!"},
        ]

        store.save("session-1", history)
        loaded = store.load("session-1")

        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[1]["content"] == "2 + 2 = 4!"

    def test_load_nonexistent(self, tmp_path):
        """Loading a missing session should return empty list."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        assert store.load("does-not-exist") == []

    def test_load_corrupt_file(self, tmp_path):
        """Loading a corrupt file should return empty list, not crash."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        (tmp_path / "bad.json").write_text("not valid json {{{")
        assert store.load("bad") == []

    def test_list_sessions(self, tmp_path):
        """Should list all sessions sorted by most recent."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        store.save("older", [{"role": "user", "content": "hi"}])
        store.save(
            "newer",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hey"},
            ],
        )

        sessions = store.list_sessions()
        assert len(sessions) == 2
        # Most recent first
        assert sessions[0]["session_id"] == "newer"
        assert sessions[0]["message_count"] == 2
        assert sessions[1]["session_id"] == "older"
        assert sessions[1]["message_count"] == 1

    def test_delete_session(self, tmp_path):
        """Should delete a session file."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        store.save("to-delete", [{"role": "user", "content": "bye"}])

        assert store.delete("to-delete") is True
        assert store.load("to-delete") == []

    def test_delete_nonexistent(self, tmp_path):
        """Deleting a missing session should return False."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        assert store.delete("nope") is False

    def test_storage_dir_created(self, tmp_path):
        """Constructor should create the storage directory."""
        from edgetutor.core.conversations import ConversationStore

        new_dir = tmp_path / "nested" / "convos"
        store = ConversationStore(storage_dir=new_dir)
        assert store.storage_dir.exists()

    def test_save_creates_valid_json(self, tmp_path):
        """Saved file should be valid JSON with expected fields."""
        from edgetutor.core.conversations import ConversationStore

        store = ConversationStore(storage_dir=tmp_path)
        store.save("test", [{"role": "user", "content": "hello"}])

        data = json.loads((tmp_path / "test.json").read_text(encoding="utf-8"))
        assert data["session_id"] == "test"
        assert "updated_at" in data
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 1

    def test_singleton(self):
        """get_conversation_store() should return a ConversationStore."""
        from edgetutor.core.conversations import ConversationStore, get_conversation_store

        store = get_conversation_store()
        assert isinstance(store, ConversationStore)

"""Tests for stackoverflow_curator.memory module.

Only tests the in-memory fallback path (no Lakebase connection required in CI).
"""

from unittest.mock import MagicMock, patch

from stackoverflow_curator.memory import LakebaseMemory


class TestLakebaseMemoryUnavailable:
    """Tests for the graceful fallback when Lakebase is not available."""

    def _make_unavailable_memory(self) -> LakebaseMemory:
        """Create a LakebaseMemory instance that fails setup (no Lakebase in CI)."""
        with patch.object(LakebaseMemory, "_setup", lambda self: None):
            mem = LakebaseMemory.__new__(LakebaseMemory)
            mem.project_id = "test-project"
            mem._conn_string = None
            mem._available = False
        return mem

    def test_save_messages_no_op_when_unavailable(self, capsys) -> None:
        mem = self._make_unavailable_memory()
        mem.save_messages("s1", [{"role": "user", "content": "hello"}])
        # Should not raise

    def test_load_messages_returns_empty_when_unavailable(self) -> None:
        mem = self._make_unavailable_memory()
        result = mem.load_messages("s1")
        assert result == []

    def test_delete_session_no_op_when_unavailable(self) -> None:
        mem = self._make_unavailable_memory()
        mem.delete_session("s1")
        # Should not raise

    def test_close_no_op(self) -> None:
        mem = self._make_unavailable_memory()
        mem.close()
        # Should not raise


class TestLakebaseMemorySetup:
    """Tests for setup failure handling."""

    def test_setup_failure_marks_unavailable(self) -> None:
        with patch.object(
            LakebaseMemory,
            "_build_conn_string",
            side_effect=RuntimeError("no databricks"),
        ):
            mem = LakebaseMemory.__new__(LakebaseMemory)
            mem.project_id = "test-project"
            mem._conn_string = None
            mem._available = False
            mem._setup()

        assert mem._available is False

    def test_project_id_stored(self) -> None:
        with patch.object(LakebaseMemory, "_setup", lambda self: None):
            mem = LakebaseMemory.__new__(LakebaseMemory)
            mem.project_id = "my-project"
            mem._conn_string = None
            mem._available = False

        assert mem.project_id == "my-project"


class TestLakebaseMemoryAvailable:
    """Tests for the happy path using mocked psycopg."""

    def _make_available_memory(self) -> LakebaseMemory:
        with patch.object(LakebaseMemory, "_setup", lambda self: None):
            mem = LakebaseMemory.__new__(LakebaseMemory)
            mem.project_id = "test-project"
            mem._conn_string = "postgresql://fake"
            mem._available = True
        return mem

    def test_save_messages_executes_insert(self) -> None:
        mem = self._make_available_memory()
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_ctx) as mock_connect:
            mem.save_messages("s1", [{"role": "user", "content": "hello"}])
            mock_connect.assert_called_once_with("postgresql://fake")

    def test_load_messages_returns_parsed_rows(self) -> None:
        mem = self._make_available_memory()
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ('{"role": "user", "content": "hi"}',),
        ]
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_ctx):
            result = mem.load_messages("s1")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi"

    def test_delete_session_executes_delete(self) -> None:
        mem = self._make_available_memory()
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("psycopg.connect", return_value=mock_ctx):
            mem.delete_session("s1")
            mock_conn.execute.assert_called_once()

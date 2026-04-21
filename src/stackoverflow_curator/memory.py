"""Session memory management using Databricks Lakebase (PostgreSQL)."""

from __future__ import annotations

import json
import os
import urllib.parse

os.environ.setdefault("PSYCOPG_IMPL", "python")

from loguru import logger


class LakebaseMemory:
    """Manages conversation session memory using Lakebase (Databricks PostgreSQL).

    Stores and retrieves chat messages per session ID using a PostgreSQL table.
    Uses the Lakebase PostgresAPI project/branch/endpoint model for auth.
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._conn_string: str | None = None
        self._available = False
        self._setup()

    def _build_conn_string(self) -> str:
        """Build a fresh PostgreSQL connection string via PostgresAPI."""
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.postgres import PostgresAPI

        w = WorkspaceClient()
        pg_api = PostgresAPI(w.api_client)

        project = pg_api.get_project(name=f"projects/{self.project_id}")
        default_branch = next(iter(pg_api.list_branches(parent=project.name)))
        endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
        host = endpoint.status.hosts.host

        pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)
        user = w.current_user.me()
        username = urllib.parse.quote_plus(user.user_name)

        return (
            f"postgresql://{username}:{pg_credential.token}@{host}:5432/"
            "databricks_postgres?sslmode=require"
        )

    def _setup(self) -> None:
        """Build connection string and ensure messages table exists."""
        try:
            import psycopg

            self._conn_string = self._build_conn_string()
            with psycopg.connect(self._conn_string) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS session_messages (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        message_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id"
                    " ON session_messages(session_id)"
                )
            self._available = True
            logger.info(f"LakebaseMemory connected to project {self.project_id}")
        except Exception as e:
            logger.warning(f"LakebaseMemory setup failed: {type(e).__name__}: {e}")
            self._available = False

    def save_messages(self, session_id: str, messages: list[dict]) -> None:
        """Save messages to a session.

        Args:
            session_id: Unique session identifier
            messages: List of message dicts with 'role' and 'content' keys
        """
        if not self._available or not self._conn_string:
            logger.warning("LakebaseMemory not available — messages not persisted")
            return
        import psycopg

        with psycopg.connect(self._conn_string) as conn:
            for msg in messages:
                conn.execute(
                    "INSERT INTO session_messages (session_id, message_data)"
                    " VALUES (%s, %s)",
                    (session_id, json.dumps(msg)),
                )
        logger.debug(f"Saved {len(messages)} messages to session {session_id}")

    def load_messages(self, session_id: str) -> list[dict]:
        """Load all messages for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of message dicts ordered by creation time
        """
        if not self._available or not self._conn_string:
            return []
        import psycopg

        with psycopg.connect(self._conn_string) as conn:
            rows = conn.execute(
                "SELECT message_data FROM session_messages"
                " WHERE session_id = %s ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            json.loads(row[0]) if isinstance(row[0], str) else row[0] for row in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete all messages for a session.

        Args:
            session_id: Unique session identifier
        """
        if not self._available or not self._conn_string:
            return
        import psycopg

        with psycopg.connect(self._conn_string) as conn:
            conn.execute(
                "DELETE FROM session_messages WHERE session_id = %s",
                (session_id,),
            )
        logger.info(f"Deleted session: {session_id}")

    def close(self) -> None:
        """No-op — connections are closed after each operation."""

"""SQLite metadata storage backend for single-user deployments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cognitive_memory.storage.metadata.base import MetadataBackend

logger = logging.getLogger(__name__)


@dataclass
class SQLiteMetadataBackend(MetadataBackend):
    """
    SQLite metadata storage backend.

    Zero-infrastructure alternative to PostgreSQL for single-user
    or development deployments. Uses aiosqlite for async access
    and WAL journal mode for safe concurrent reads.

    Attributes:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.
        table_name: Name of the memories table.
    """

    db_path: str = "cognitive_memory.db"
    table_name: str = "memories"
    _conn: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize the database connection and create schema."""
        if self._initialized:
            return

        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError(
                "aiosqlite is required for SQLiteMetadataBackend. "
                "Install it with: pip install cognitive-memory[sqlite]"
            ) from e

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

        await self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL DEFAULT 'episodic',
                content TEXT NOT NULL DEFAULT '',
                metadata TEXT DEFAULT '{{}}',
                agent_id TEXT,
                user_id TEXT,
                source TEXT DEFAULT 'conversation',
                source_id TEXT,
                strength REAL DEFAULT 1.0,
                initial_strength REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                emotional_valence REAL DEFAULT 0.0,
                surprise_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                entities TEXT DEFAULT '[]',
                topics TEXT DEFAULT '[]',
                related_memory_ids TEXT DEFAULT '[]',
                parent_memory_id TEXT,
                superseded_by_id TEXT,
                is_pinned INTEGER DEFAULT 0,
                is_archived INTEGER DEFAULT 0,
                is_consolidated INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_agent_id_idx
            ON {self.table_name} (agent_id)
        """)
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_user_id_idx
            ON {self.table_name} (user_id)
        """)
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_memory_type_idx
            ON {self.table_name} (memory_type)
        """)
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_created_at_idx
            ON {self.table_name} (created_at DESC)
        """)
        await self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_is_archived_idx
            ON {self.table_name} (is_archived) WHERE is_archived = 0
        """)

        await self._conn.commit()
        self._initialized = True
        logger.info("SQLiteMetadataBackend initialized: %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the backend is initialized."""
        if not self._initialized:
            await self.initialize()

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert a database row to a memory dict."""
        raw = dict(row)
        return {
            "id": raw["id"],
            "memory_type": raw["memory_type"],
            "content": raw["content"],
            "metadata": json.loads(raw["metadata"]) if raw["metadata"] else {},
            "agent_id": raw["agent_id"],
            "user_id": raw["user_id"],
            "source": raw["source"],
            "source_id": raw["source_id"],
            "strength": raw["strength"],
            "initial_strength": raw["initial_strength"],
            "importance": raw["importance"],
            "emotional_valence": raw["emotional_valence"],
            "surprise_score": raw["surprise_score"],
            "access_count": raw["access_count"],
            "entities": json.loads(raw["entities"]) if raw["entities"] else [],
            "topics": json.loads(raw["topics"]) if raw["topics"] else [],
            "related_memory_ids": (
                json.loads(raw["related_memory_ids"]) if raw["related_memory_ids"] else []
            ),
            "parent_memory_id": raw["parent_memory_id"],
            "superseded_by_id": raw["superseded_by_id"],
            "is_pinned": bool(raw["is_pinned"]),
            "is_archived": bool(raw["is_archived"]),
            "is_consolidated": bool(raw["is_consolidated"]),
            "created_at": raw["created_at"],
            "last_accessed_at": raw["last_accessed_at"],
            "updated_at": raw["updated_at"],
        }

    async def save_memory(self, memory: dict[str, Any]) -> None:
        """Save or update a memory's metadata."""
        await self._ensure_initialized()

        memory_id = memory.get("id")
        if not memory_id:
            raise ValueError("Memory must have an 'id' field")

        now = datetime.now(timezone.utc).isoformat()

        await self._conn.execute(
            f"""
            INSERT INTO {self.table_name} (
                id, memory_type, content, metadata, agent_id, user_id,
                source, source_id, strength, initial_strength, importance,
                emotional_valence, surprise_score, access_count,
                entities, topics, related_memory_ids, parent_memory_id,
                superseded_by_id, is_pinned, is_archived, is_consolidated,
                updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(id) DO UPDATE SET
                memory_type = excluded.memory_type,
                content = excluded.content,
                metadata = excluded.metadata,
                agent_id = excluded.agent_id,
                user_id = excluded.user_id,
                source = excluded.source,
                source_id = excluded.source_id,
                strength = excluded.strength,
                initial_strength = excluded.initial_strength,
                importance = excluded.importance,
                emotional_valence = excluded.emotional_valence,
                surprise_score = excluded.surprise_score,
                access_count = excluded.access_count,
                entities = excluded.entities,
                topics = excluded.topics,
                related_memory_ids = excluded.related_memory_ids,
                parent_memory_id = excluded.parent_memory_id,
                superseded_by_id = excluded.superseded_by_id,
                is_pinned = excluded.is_pinned,
                is_archived = excluded.is_archived,
                is_consolidated = excluded.is_consolidated,
                updated_at = excluded.updated_at
            """,
            (
                memory_id,
                memory.get("memory_type", "episodic"),
                memory.get("content", ""),
                json.dumps(memory.get("metadata", {})),
                memory.get("agent_id"),
                memory.get("user_id"),
                memory.get("source", "conversation"),
                memory.get("source_id"),
                memory.get("strength", 1.0),
                memory.get("initial_strength", 1.0),
                memory.get("importance", 0.5),
                memory.get("emotional_valence", 0.0),
                memory.get("surprise_score", 0.0),
                memory.get("access_count", 0),
                json.dumps(memory.get("entities", [])),
                json.dumps(memory.get("topics", [])),
                json.dumps(memory.get("related_memory_ids", [])),
                memory.get("parent_memory_id"),
                memory.get("superseded_by_id"),
                int(memory.get("is_pinned", False)),
                int(memory.get("is_archived", False)),
                int(memory.get("is_consolidated", False)),
                now,
            ),
        )
        await self._conn.commit()

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID."""
        await self._ensure_initialized()

        cursor = await self._conn.execute(
            f"SELECT * FROM {self.table_name} WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_dict(row)

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        await self._ensure_initialized()

        cursor = await self._conn.execute(
            f"DELETE FROM {self.table_name} WHERE id = ?",
            (memory_id,),
        )
        await self._conn.commit()

        return bool(cursor.rowcount and cursor.rowcount > 0)

    async def list_memories(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        memory_type: str | None = None,
        is_archived: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List memories with optional filters."""
        await self._ensure_initialized()

        conditions: list[str] = []
        params: list[Any] = []

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if is_archived is not None:
            conditions.append("is_archived = ?")
            params.append(int(is_archived))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.extend([limit, offset])

        cursor = await self._conn.execute(
            f"""
            SELECT * FROM {self.table_name}
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        )
        rows = await cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    async def update_access(
        self,
        memory_id: str,
        accessed_at: datetime | None = None,
    ) -> None:
        """Update memory access metadata."""
        await self._ensure_initialized()

        if accessed_at is None:
            accessed_at = datetime.now(timezone.utc)

        await self._conn.execute(
            f"""
            UPDATE {self.table_name}
            SET
                access_count = access_count + 1,
                last_accessed_at = ?
            WHERE id = ?
            """,
            (accessed_at.isoformat(), memory_id),
        )
        await self._conn.commit()

    async def batch_save(self, memories: list[dict[str, Any]]) -> int:
        """Save multiple memories in a single transaction."""
        await self._ensure_initialized()

        if not memories:
            return 0

        for memory in memories:
            await self.save_memory(memory)

        return len(memories)

    async def batch_delete(self, memory_ids: list[str]) -> int:
        """Delete multiple memories."""
        await self._ensure_initialized()

        if not memory_ids:
            return 0

        placeholders = ",".join("?" for _ in memory_ids)
        cursor = await self._conn.execute(
            f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
            memory_ids,
        )
        await self._conn.commit()

        return int(cursor.rowcount) if cursor.rowcount else 0

    async def count(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count memories with optional filters."""
        await self._ensure_initialized()

        conditions: list[str] = []
        params: list[Any] = []

        if agent_id is not None:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        cursor = await self._conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name} {where_clause}",
            params,
        )
        row = await cursor.fetchone()

        return int(row[0]) if row else 0

    async def get_memories_by_ids(
        self,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get multiple memories by IDs, preserving input order."""
        await self._ensure_initialized()

        if not memory_ids:
            return []

        placeholders = ",".join("?" for _ in memory_ids)
        cursor = await self._conn.execute(
            f"SELECT * FROM {self.table_name} WHERE id IN ({placeholders})",
            memory_ids,
        )
        rows = await cursor.fetchall()

        row_dict = {self._row_to_dict(row)["id"]: self._row_to_dict(row) for row in rows}
        return [row_dict[mid] for mid in memory_ids if mid in row_dict]

    async def search_by_content(
        self,
        query: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Simple LIKE-based content search.

        For full-text search, consider FTS5 extension (future enhancement).

        Args:
            query: Search query string.
            agent_id: Optional agent filter.
            limit: Maximum results.

        Returns:
            Matching memories ordered by creation date.
        """
        await self._ensure_initialized()

        params: list[Any] = [f"%{query}%"]
        agent_filter = ""

        if agent_id:
            agent_filter = "AND agent_id = ?"
            params.append(agent_id)

        params.append(limit)

        cursor = await self._conn.execute(
            f"""
            SELECT * FROM {self.table_name}
            WHERE content LIKE ? {agent_filter}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        )
        rows = await cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    async def get_related_memories(
        self,
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get memories related to a given memory."""
        await self._ensure_initialized()

        memory = await self.get_memory(memory_id)
        if not memory:
            return []

        related_ids = memory.get("related_memory_ids", [])
        if not related_ids:
            return []

        return await self.get_memories_by_ids(related_ids[:limit])

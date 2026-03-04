"""PostgreSQL metadata storage backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cognitive_memory.storage.metadata.base import MetadataBackend

logger = logging.getLogger(__name__)


@dataclass
class PostgresMetadataBackend(MetadataBackend):
    """
    PostgreSQL metadata storage backend.

    Stores memory metadata in PostgreSQL with JSONB support
    for flexible schema and efficient querying.

    Attributes:
        connection_string: PostgreSQL connection string.
        table_name: Name of the memories table.
        pool_size: Connection pool size.
    """

    connection_string: str = "postgresql://localhost:5432/cognitive_memory"
    table_name: str = "memories"
    pool_size: int = 10
    _pool: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    async def initialize(self) -> None:
        """Initialize connection pool and create table if needed."""
        if self._initialized:
            return

        try:
            import asyncpg  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PostgresMetadataBackend. "
                "Install it with: pip install asyncpg"
            ) from e

        self._pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=1,
            max_size=self.pool_size,
        )

        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL DEFAULT 'episodic',
                    content TEXT NOT NULL DEFAULT '',
                    metadata JSONB DEFAULT '{{}}',
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
                    entities TEXT[] DEFAULT '{{}}',
                    topics TEXT[] DEFAULT '{{}}',
                    related_memory_ids TEXT[] DEFAULT '{{}}',
                    parent_memory_id TEXT,
                    superseded_by_id TEXT,
                    is_pinned BOOLEAN DEFAULT FALSE,
                    is_archived BOOLEAN DEFAULT FALSE,
                    is_consolidated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes for common queries
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_agent_id_idx
                ON {self.table_name} (agent_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_user_id_idx
                ON {self.table_name} (user_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_memory_type_idx
                ON {self.table_name} (memory_type)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_created_at_idx
                ON {self.table_name} (created_at DESC)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_is_archived_idx
                ON {self.table_name} (is_archived) WHERE is_archived = FALSE
            """)

        self._initialized = True
        logger.info(f"PostgresMetadataBackend initialized: {self.table_name}")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the backend is initialized."""
        if not self._initialized:
            await self.initialize()

    async def save_memory(self, memory: dict[str, Any]) -> None:
        """Save or update a memory's metadata."""
        await self._ensure_initialized()

        memory_id = memory.get("id")
        if not memory_id:
            raise ValueError("Memory must have an 'id' field")

        # Extract fields
        fields = {
            "memory_type": memory.get("memory_type", "episodic"),
            "content": memory.get("content", ""),
            "metadata": json.dumps(memory.get("metadata", {})),
            "agent_id": memory.get("agent_id"),
            "user_id": memory.get("user_id"),
            "source": memory.get("source", "conversation"),
            "source_id": memory.get("source_id"),
            "strength": memory.get("strength", 1.0),
            "initial_strength": memory.get("initial_strength", 1.0),
            "importance": memory.get("importance", 0.5),
            "emotional_valence": memory.get("emotional_valence", 0.0),
            "surprise_score": memory.get("surprise_score", 0.0),
            "access_count": memory.get("access_count", 0),
            "entities": memory.get("entities", []),
            "topics": memory.get("topics", []),
            "related_memory_ids": memory.get("related_memory_ids", []),
            "parent_memory_id": memory.get("parent_memory_id"),
            "superseded_by_id": memory.get("superseded_by_id"),
            "is_pinned": memory.get("is_pinned", False),
            "is_archived": memory.get("is_archived", False),
            "is_consolidated": memory.get("is_consolidated", False),
        }

        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.table_name} (
                    id, memory_type, content, metadata, agent_id, user_id,
                    source, source_id, strength, initial_strength, importance,
                    emotional_valence, surprise_score, access_count,
                    entities, topics, related_memory_ids, parent_memory_id,
                    superseded_by_id, is_pinned, is_archived, is_consolidated,
                    updated_at
                ) VALUES (
                    $1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11,
                    $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    memory_type = EXCLUDED.memory_type,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    agent_id = EXCLUDED.agent_id,
                    user_id = EXCLUDED.user_id,
                    source = EXCLUDED.source,
                    source_id = EXCLUDED.source_id,
                    strength = EXCLUDED.strength,
                    initial_strength = EXCLUDED.initial_strength,
                    importance = EXCLUDED.importance,
                    emotional_valence = EXCLUDED.emotional_valence,
                    surprise_score = EXCLUDED.surprise_score,
                    access_count = EXCLUDED.access_count,
                    entities = EXCLUDED.entities,
                    topics = EXCLUDED.topics,
                    related_memory_ids = EXCLUDED.related_memory_ids,
                    parent_memory_id = EXCLUDED.parent_memory_id,
                    superseded_by_id = EXCLUDED.superseded_by_id,
                    is_pinned = EXCLUDED.is_pinned,
                    is_archived = EXCLUDED.is_archived,
                    is_consolidated = EXCLUDED.is_consolidated,
                    updated_at = NOW()
            """,
                memory_id,
                fields["memory_type"],
                fields["content"],
                fields["metadata"],
                fields["agent_id"],
                fields["user_id"],
                fields["source"],
                fields["source_id"],
                fields["strength"],
                fields["initial_strength"],
                fields["importance"],
                fields["emotional_valence"],
                fields["surprise_score"],
                fields["access_count"],
                fields["entities"],
                fields["topics"],
                fields["related_memory_ids"],
                fields["parent_memory_id"],
                fields["superseded_by_id"],
                fields["is_pinned"],
                fields["is_archived"],
                fields["is_consolidated"],
            )

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.table_name} WHERE id = $1",
                memory_id,
            )

        if not row:
            return None

        return self._row_to_dict(row)

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert a database row to a memory dict."""
        return {
            "id": row["id"],
            "memory_type": row["memory_type"],
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "agent_id": row["agent_id"],
            "user_id": row["user_id"],
            "source": row["source"],
            "source_id": row["source_id"],
            "strength": row["strength"],
            "initial_strength": row["initial_strength"],
            "importance": row["importance"],
            "emotional_valence": row["emotional_valence"],
            "surprise_score": row["surprise_score"],
            "access_count": row["access_count"],
            "entities": list(row["entities"]) if row["entities"] else [],
            "topics": list(row["topics"]) if row["topics"] else [],
            "related_memory_ids": (
                list(row["related_memory_ids"]) if row["related_memory_ids"] else []
            ),
            "parent_memory_id": row["parent_memory_id"],
            "superseded_by_id": row["superseded_by_id"],
            "is_pinned": row["is_pinned"],
            "is_archived": row["is_archived"],
            "is_consolidated": row["is_consolidated"],
            "created_at": row["created_at"],
            "last_accessed_at": row["last_accessed_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = $1",
                memory_id,
            )

        return str(result).split()[-1] != "0"

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

        conditions = []
        params: list[Any] = []
        param_idx = 1

        if agent_id is not None:
            conditions.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1

        if user_id is not None:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if memory_type is not None:
            conditions.append(f"memory_type = ${param_idx}")
            params.append(memory_type)
            param_idx += 1

        if is_archived is not None:
            conditions.append(f"is_archived = ${param_idx}")
            params.append(is_archived)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.extend([limit, offset])

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM {self.table_name}
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """, *params)

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

        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE {self.table_name}
                SET
                    access_count = access_count + 1,
                    last_accessed_at = $2
                WHERE id = $1
            """, memory_id, accessed_at)

    async def batch_save(self, memories: list[dict[str, Any]]) -> int:
        """Save multiple memories."""
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

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY($1)",
                memory_ids,
            )

        return int(str(result).split()[-1])

    async def count(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count memories with optional filters."""
        await self._ensure_initialized()

        conditions = []
        params: list[Any] = []
        param_idx = 1

        if agent_id is not None:
            conditions.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1

        if user_id is not None:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.table_name} {where_clause}",
                *params,
            )

        return int(result) if result else 0

    async def get_memories_by_ids(
        self,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get multiple memories by IDs."""
        await self._ensure_initialized()

        if not memory_ids:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.table_name} WHERE id = ANY($1)",
                memory_ids,
            )

        # Create a dict for ordering
        row_dict = {self._row_to_dict(row)["id"]: self._row_to_dict(row) for row in rows}

        # Return in same order as input IDs
        return [row_dict[mid] for mid in memory_ids if mid in row_dict]

    async def search_by_content(
        self,
        query: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Full-text search on memory content.

        Args:
            query: Search query.
            agent_id: Optional agent filter.
            limit: Maximum results.

        Returns:
            Matching memories.
        """
        await self._ensure_initialized()

        params: list[Any] = [f"%{query}%", limit]
        agent_filter = ""

        if agent_id:
            agent_filter = "AND agent_id = $3"
            params.append(agent_id)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM {self.table_name}
                WHERE content ILIKE $1 {agent_filter}
                ORDER BY created_at DESC
                LIMIT $2
            """, *params)

        return [self._row_to_dict(row) for row in rows]

    async def get_related_memories(
        self,
        memory_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get memories related to a given memory.

        Args:
            memory_id: ID of the source memory.
            limit: Maximum results.

        Returns:
            Related memories.
        """
        await self._ensure_initialized()

        # First get the memory to find its related IDs
        memory = await self.get_memory(memory_id)
        if not memory:
            return []

        related_ids = memory.get("related_memory_ids", [])
        if not related_ids:
            return []

        return await self.get_memories_by_ids(related_ids[:limit])

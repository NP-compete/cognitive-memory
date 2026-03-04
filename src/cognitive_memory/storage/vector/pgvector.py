"""PostgreSQL pgvector storage backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from cognitive_memory.storage.vector.base import SearchResult, VectorBackend

logger = logging.getLogger(__name__)


@dataclass
class PgVectorBackend(VectorBackend):
    """
    PostgreSQL pgvector storage backend.

    Provides vector storage and similarity search using PostgreSQL with pgvector.
    Supports HNSW and IVFFlat indexes for efficient similarity search.

    Benefits over dedicated vector DBs:
    - Single database for both metadata and vectors
    - ACID transactions across metadata and vectors
    - Simpler deployment and operations
    - Full SQL query capabilities

    Attributes:
        connection_string: PostgreSQL connection string.
        table_name: Name of the table to store vectors.
        vector_dimensions: Dimension of vectors.
        distance_metric: Distance metric ("cosine", "l2", "inner_product").
        index_type: Index type ("hnsw", "ivfflat", "none").
        hnsw_m: HNSW M parameter (connections per node).
        hnsw_ef_construction: HNSW ef_construction parameter.
        ivfflat_lists: Number of IVFFlat lists.
        pool_size: Connection pool size.
    """

    connection_string: str = "postgresql://localhost:5432/cognitive_memory"
    table_name: str = "memory_vectors"
    vector_dimensions: int = 1536
    distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    index_type: Literal["hnsw", "ivfflat", "none"] = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    ivfflat_lists: int = 100
    pool_size: int = 10
    _pool: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _get_distance_operator(self) -> str:
        """Get the pgvector distance operator for the configured metric."""
        operators = {
            "cosine": "<=>",
            "l2": "<->",
            "inner_product": "<#>",
        }
        return operators.get(self.distance_metric, "<=>")

    def _get_index_ops(self) -> str:
        """Get the pgvector index operator class."""
        ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }
        return ops.get(self.distance_metric, "vector_cosine_ops")

    async def initialize(self) -> None:
        """
        Initialize the connection pool and create table/index if needed.
        """
        if self._initialized:
            return

        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PgVectorBackend. Install it with: pip install asyncpg"
            ) from e

        # Create connection pool
        self._pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=1,
            max_size=self.pool_size,
        )

        async with self._pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table if not exists
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.vector_dimensions}),
                    payload JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create index if configured
            if self.index_type != "none":
                await self._create_index(conn)

        self._initialized = True
        logger.info(f"PgVectorBackend initialized: {self.table_name}")

    async def _create_index(self, conn: Any) -> None:
        """Create the vector index if it doesn't exist."""
        index_name = f"{self.table_name}_embedding_idx"

        # Check if index exists
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE indexname = $1
            )
        """,
            index_name,
        )

        if exists:
            return

        ops = self._get_index_ops()

        if self.index_type == "hnsw":
            await conn.execute(f"""
                CREATE INDEX {index_name}
                ON {self.table_name}
                USING hnsw (embedding {ops})
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})
            """)
            logger.info(f"Created HNSW index: {index_name}")

        elif self.index_type == "ivfflat":
            # IVFFlat requires data to be present for optimal list count
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")
            if count > 0:
                await conn.execute(f"""
                    CREATE INDEX {index_name}
                    ON {self.table_name}
                    USING ivfflat (embedding {ops})
                    WITH (lists = {self.ivfflat_lists})
                """)
                logger.info(f"Created IVFFlat index: {index_name}")

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

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Insert or update a vector.

        Args:
            id: Unique identifier for the vector.
            embedding: The vector embedding.
            payload: Optional metadata to store with the vector.
        """
        await self._ensure_initialized()

        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        payload_json = json.dumps(payload or {})

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (id, embedding, payload, updated_at)
                VALUES ($1, $2::vector, $3::jsonb, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
            """,
                id,
                embedding_str,
                payload_json,
            )

    async def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            embedding: Query vector.
            top_k: Number of results to return.
            filter_conditions: Optional filter on payload fields (JSONB).

        Returns:
            List of SearchResult sorted by similarity.
        """
        await self._ensure_initialized()

        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        operator = self._get_distance_operator()

        # Build WHERE clause for filters
        where_clause = ""
        params: list[Any] = [embedding_str, top_k]

        if filter_conditions:
            conditions = []
            for i, (key, value) in enumerate(filter_conditions.items(), start=3):
                conditions.append(f"payload->>'{key}' = ${i}")
                params.append(str(value))
            where_clause = "WHERE " + " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    id,
                    payload,
                    1 - (embedding {operator} $1::vector) as score
                FROM {self.table_name}
                {where_clause}
                ORDER BY embedding {operator} $1::vector
                LIMIT $2
            """,
                *params,
            )

        return [
            SearchResult(
                id=row["id"],
                score=float(row["score"]) if row["score"] else 0.0,
                payload=json.loads(row["payload"]) if row["payload"] else {},
            )
            for row in rows
        ]

    async def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.

        Args:
            id: ID of the vector to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name} WHERE id = $1
            """,
                id,
            )

        # asyncpg returns "DELETE N" where N is rows affected
        return str(result).split()[-1] != "0"

    async def get(self, id: str) -> SearchResult | None:
        """
        Get a vector by ID.

        Args:
            id: ID of the vector to retrieve.

        Returns:
            SearchResult if found, None otherwise.
        """
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, payload FROM {self.table_name} WHERE id = $1
            """,
                id,
            )

        if not row:
            return None

        return SearchResult(
            id=row["id"],
            score=1.0,
            payload=json.loads(row["payload"]) if row["payload"] else {},
        )

    async def batch_upsert(
        self,
        items: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> int:
        """
        Insert or update multiple vectors efficiently.

        Uses PostgreSQL's COPY for bulk inserts when possible.

        Args:
            items: List of (id, embedding, payload) tuples.

        Returns:
            Number of vectors upserted.
        """
        await self._ensure_initialized()

        if not items:
            return 0

        async with self._pool.acquire() as conn, conn.transaction():
            for id, embedding, payload in items:
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                payload_json = json.dumps(payload or {})

                await conn.execute(
                    f"""
                        INSERT INTO {self.table_name} (id, embedding, payload, updated_at)
                        VALUES ($1, $2::vector, $3::jsonb, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            payload = EXCLUDED.payload,
                            updated_at = NOW()
                    """,
                    id,
                    embedding_str,
                    payload_json,
                )

        return len(items)

    async def batch_delete(self, ids: list[str]) -> int:
        """
        Delete multiple vectors.

        Args:
            ids: List of IDs to delete.

        Returns:
            Number of vectors deleted.
        """
        await self._ensure_initialized()

        if not ids:
            return 0

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self.table_name} WHERE id = ANY($1)
            """,
                ids,
            )

        # asyncpg returns "DELETE N"
        return int(result.split()[-1])

    async def count(self) -> int:
        """
        Get the total number of vectors.

        Returns:
            Vector count.
        """
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name}")
            return int(result) if result else 0

    async def clear(self) -> None:
        """Delete all vectors (truncate table)."""
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            await conn.execute(f"TRUNCATE TABLE {self.table_name}")

    async def rebuild_index(self) -> None:
        """
        Rebuild the vector index.

        Useful after large batch inserts for optimal performance.
        """
        await self._ensure_initialized()

        index_name = f"{self.table_name}_embedding_idx"

        async with self._pool.acquire() as conn:
            # Drop existing index
            await conn.execute(f"DROP INDEX IF EXISTS {index_name}")

            # Recreate index
            if self.index_type != "none":
                await self._create_index(conn)

        logger.info(f"Rebuilt index: {index_name}")

    async def vacuum_analyze(self) -> None:
        """
        Run VACUUM ANALYZE on the table.

        Improves query performance by updating statistics.
        """
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            await conn.execute(f"VACUUM ANALYZE {self.table_name}")

        logger.info(f"Vacuumed table: {self.table_name}")

    async def set_search_params(
        self,
        hnsw_ef_search: int | None = None,
        ivfflat_probes: int | None = None,
    ) -> None:
        """
        Set search parameters for the current session.

        Args:
            hnsw_ef_search: HNSW ef_search parameter (higher = more accurate).
            ivfflat_probes: IVFFlat probes parameter (higher = more accurate).
        """
        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            if hnsw_ef_search is not None:
                await conn.execute(f"SET hnsw.ef_search = {hnsw_ef_search}")

            if ivfflat_probes is not None:
                await conn.execute(f"SET ivfflat.probes = {ivfflat_probes}")

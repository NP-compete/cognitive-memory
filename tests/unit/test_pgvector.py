"""Tests for pgvector backend."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognitive_memory.storage.vector.base import SearchResult
from cognitive_memory.storage.vector.pgvector import PgVectorBackend


class TestPgVectorBackend:
    """Tests for PgVectorBackend."""

    def test_default_values(self) -> None:
        """PgVectorBackend should have sensible defaults."""
        backend = PgVectorBackend()

        assert backend.connection_string == "postgresql://localhost:5432/cognitive_memory"
        assert backend.table_name == "memory_vectors"
        assert backend.vector_dimensions == 1536
        assert backend.distance_metric == "cosine"
        assert backend.index_type == "hnsw"
        assert backend.hnsw_m == 16
        assert backend.hnsw_ef_construction == 64
        assert backend.pool_size == 10

    def test_custom_values(self) -> None:
        """PgVectorBackend should accept custom values."""
        backend = PgVectorBackend(
            connection_string="postgresql://user:pass@db:5432/mydb",
            table_name="custom_vectors",
            vector_dimensions=768,
            distance_metric="l2",
            index_type="ivfflat",
            ivfflat_lists=200,
            pool_size=20,
        )

        assert backend.connection_string == "postgresql://user:pass@db:5432/mydb"
        assert backend.table_name == "custom_vectors"
        assert backend.vector_dimensions == 768
        assert backend.distance_metric == "l2"
        assert backend.index_type == "ivfflat"
        assert backend.ivfflat_lists == 200
        assert backend.pool_size == 20

    def test_initial_state(self) -> None:
        """Backend should start uninitialized."""
        backend = PgVectorBackend()

        assert backend._pool is None
        assert backend._initialized is False


class TestDistanceOperators:
    """Tests for distance operator selection."""

    def test_cosine_operator(self) -> None:
        """Cosine distance should use <=> operator."""
        backend = PgVectorBackend(distance_metric="cosine")
        assert backend._get_distance_operator() == "<=>"

    def test_l2_operator(self) -> None:
        """L2 distance should use <-> operator."""
        backend = PgVectorBackend(distance_metric="l2")
        assert backend._get_distance_operator() == "<->"

    def test_inner_product_operator(self) -> None:
        """Inner product should use <#> operator."""
        backend = PgVectorBackend(distance_metric="inner_product")
        assert backend._get_distance_operator() == "<#>"


class TestIndexOps:
    """Tests for index operator class selection."""

    def test_cosine_ops(self) -> None:
        """Cosine should use vector_cosine_ops."""
        backend = PgVectorBackend(distance_metric="cosine")
        assert backend._get_index_ops() == "vector_cosine_ops"

    def test_l2_ops(self) -> None:
        """L2 should use vector_l2_ops."""
        backend = PgVectorBackend(distance_metric="l2")
        assert backend._get_index_ops() == "vector_l2_ops"

    def test_inner_product_ops(self) -> None:
        """Inner product should use vector_ip_ops."""
        backend = PgVectorBackend(distance_metric="inner_product")
        assert backend._get_index_ops() == "vector_ip_ops"


class TestPgVectorOperationsWithMockedPool:
    """Tests for CRUD operations with pre-mocked pool."""

    @pytest.fixture
    def mock_backend(self) -> PgVectorBackend:
        """Create a backend with mocked pool."""
        backend = PgVectorBackend()
        backend._pool = MagicMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_backend: PgVectorBackend) -> None:
        """Search should return SearchResult list."""
        mock_conn = AsyncMock()
        mock_row = {
            "id": "test-id",
            "score": 0.95,
            "payload": '{"key": "value"}',
        }
        mock_conn.fetch.return_value = [mock_row]
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        results = await mock_backend.search(
            embedding=[0.1] * 10,
            top_k=5,
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "test-id"
        assert results[0].score == 0.95
        assert results[0].payload == {"key": "value"}

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_backend: PgVectorBackend) -> None:
        """Search should handle empty results."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        results = await mock_backend.search(embedding=[0.1] * 10)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_existing(self, mock_backend: PgVectorBackend) -> None:
        """Get should return SearchResult for existing vector."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "id": "test-id",
            "payload": '{"key": "value"}',
        }
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.get("test-id")

        assert result is not None
        assert result.id == "test-id"
        assert result.payload == {"key": "value"}
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mock_backend: PgVectorBackend) -> None:
        """Get should return None for nonexistent vector."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.get("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_backend: PgVectorBackend) -> None:
        """Delete should return True for existing vector."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 1"
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.delete("test-id")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_backend: PgVectorBackend) -> None:
        """Delete should return False for nonexistent vector."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 0"
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.delete("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_batch_upsert_empty(self, mock_backend: PgVectorBackend) -> None:
        """Batch upsert with empty list should return 0."""
        count = await mock_backend.batch_upsert([])

        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_delete_empty(self, mock_backend: PgVectorBackend) -> None:
        """Batch delete with empty list should return 0."""
        count = await mock_backend.batch_delete([])

        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_delete_multiple(self, mock_backend: PgVectorBackend) -> None:
        """Batch delete should return count of deleted rows."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 3"
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        count = await mock_backend.batch_delete(["id1", "id2", "id3"])

        assert count == 3

    @pytest.mark.asyncio
    async def test_count(self, mock_backend: PgVectorBackend) -> None:
        """Count should return vector count."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 42
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        count = await mock_backend.count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_clear(self, mock_backend: PgVectorBackend) -> None:
        """Clear should truncate table."""
        mock_conn = AsyncMock()
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        await mock_backend.clear()

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "TRUNCATE" in call_args


class TestPgVectorClose:
    """Tests for close operation."""

    @pytest.mark.asyncio
    async def test_close_with_pool(self) -> None:
        """Close should clean up pool."""
        backend = PgVectorBackend()
        mock_pool = AsyncMock()
        backend._pool = mock_pool
        backend._initialized = True

        await backend.close()

        mock_pool.close.assert_called_once()
        assert backend._pool is None
        assert backend._initialized is False

    @pytest.mark.asyncio
    async def test_close_without_pool(self) -> None:
        """Close should handle no pool gracefully."""
        backend = PgVectorBackend()
        backend._pool = None
        backend._initialized = False

        await backend.close()

        assert backend._pool is None
        assert backend._initialized is False


class TestSearchResultDataclass:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self) -> None:
        """SearchResult should store all fields."""
        result = SearchResult(
            id="test-id",
            score=0.95,
            payload={"key": "value"},
        )

        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.payload == {"key": "value"}

    def test_search_result_empty_payload(self) -> None:
        """SearchResult should handle empty payload."""
        result = SearchResult(
            id="test-id",
            score=0.5,
            payload={},
        )

        assert result.payload == {}

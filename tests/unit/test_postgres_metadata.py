"""Tests for PostgreSQL metadata backend."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognitive_memory.storage.metadata.postgres import PostgresMetadataBackend


class TestPostgresMetadataBackend:
    """Tests for PostgresMetadataBackend."""

    def test_default_values(self) -> None:
        """PostgresMetadataBackend should have sensible defaults."""
        backend = PostgresMetadataBackend()

        assert backend.connection_string == "postgresql://localhost:5432/cognitive_memory"
        assert backend.table_name == "memories"
        assert backend.pool_size == 10

    def test_custom_values(self) -> None:
        """PostgresMetadataBackend should accept custom values."""
        backend = PostgresMetadataBackend(
            connection_string="postgresql://user:pass@db:5432/mydb",
            table_name="custom_memories",
            pool_size=20,
        )

        assert backend.connection_string == "postgresql://user:pass@db:5432/mydb"
        assert backend.table_name == "custom_memories"
        assert backend.pool_size == 20

    def test_initial_state(self) -> None:
        """Backend should start uninitialized."""
        backend = PostgresMetadataBackend()

        assert backend._pool is None
        assert backend._initialized is False


class TestPostgresOperationsWithMockedPool:
    """Tests for operations with pre-mocked pool."""

    @pytest.fixture
    def mock_backend(self) -> PostgresMetadataBackend:
        """Create a backend with mocked pool."""
        backend = PostgresMetadataBackend()
        backend._pool = MagicMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_get_memory_existing(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Get should return memory dict for existing ID."""
        mock_conn = AsyncMock()
        mock_row = {
            "id": "test-id",
            "memory_type": "episodic",
            "content": "Test content",
            "metadata": "{}",
            "agent_id": "agent-1",
            "user_id": "user-1",
            "source": "conversation",
            "source_id": None,
            "strength": 1.0,
            "initial_strength": 1.0,
            "importance": 0.5,
            "emotional_valence": 0.0,
            "surprise_score": 0.0,
            "access_count": 0,
            "entities": [],
            "topics": [],
            "related_memory_ids": [],
            "parent_memory_id": None,
            "superseded_by_id": None,
            "is_pinned": False,
            "is_archived": False,
            "is_consolidated": False,
            "created_at": datetime.now(timezone.utc),
            "last_accessed_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        mock_conn.fetchrow.return_value = mock_row
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.get_memory("test-id")

        assert result is not None
        assert result["id"] == "test-id"
        assert result["memory_type"] == "episodic"
        assert result["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_get_memory_nonexistent(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Get should return None for nonexistent ID."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.get_memory("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_memory_existing(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Delete should return True for existing memory."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 1"
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.delete_memory("test-id")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory_nonexistent(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Delete should return False for nonexistent memory."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 0"
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.delete_memory("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_batch_save_empty(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Batch save with empty list should return 0."""
        count = await mock_backend.batch_save([])

        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_delete_empty(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Batch delete with empty list should return 0."""
        count = await mock_backend.batch_delete([])

        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_delete_multiple(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Batch delete should return count of deleted rows."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 3"
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        count = await mock_backend.batch_delete(["id1", "id2", "id3"])

        assert count == 3

    @pytest.mark.asyncio
    async def test_count(self, mock_backend: PostgresMetadataBackend) -> None:
        """Count should return memory count."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 42
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        count = await mock_backend.count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_count_with_filters(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Count should apply filters."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 10
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        count = await mock_backend.count(agent_id="agent-1", user_id="user-1")

        assert count == 10
        # Verify query was called with parameters
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memories_by_ids_empty(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Get by IDs with empty list should return empty list."""
        result = await mock_backend.get_memories_by_ids([])

        assert result == []

    @pytest.mark.asyncio
    async def test_update_access(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """Update access should increment count and update timestamp."""
        mock_conn = AsyncMock()
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        await mock_backend.update_access("test-id")

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_memories_empty_result(
        self, mock_backend: PostgresMetadataBackend
    ) -> None:
        """List memories should handle empty results."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_backend._pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await mock_backend.list_memories()

        assert result == []


class TestPostgresClose:
    """Tests for close operation."""

    @pytest.mark.asyncio
    async def test_close_with_pool(self) -> None:
        """Close should clean up pool."""
        backend = PostgresMetadataBackend()
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
        backend = PostgresMetadataBackend()
        backend._pool = None
        backend._initialized = False

        await backend.close()

        assert backend._pool is None
        assert backend._initialized is False


class TestSaveMemoryValidation:
    """Tests for save_memory validation."""

    @pytest.mark.asyncio
    async def test_save_memory_without_id_raises(self) -> None:
        """Save memory without ID should raise ValueError."""
        backend = PostgresMetadataBackend()
        backend._pool = MagicMock()
        backend._initialized = True

        with pytest.raises(ValueError, match="must have an 'id' field"):
            await backend.save_memory({"content": "test"})


class TestRowToDict:
    """Tests for _row_to_dict helper."""

    def test_row_to_dict_basic(self) -> None:
        """Row to dict should convert all fields."""
        backend = PostgresMetadataBackend()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": "test-id",
            "memory_type": "semantic",
            "content": "Test content",
            "metadata": '{"key": "value"}',
            "agent_id": "agent-1",
            "user_id": "user-1",
            "source": "tool_result",
            "source_id": "tool-123",
            "strength": 0.8,
            "initial_strength": 1.0,
            "importance": 0.7,
            "emotional_valence": 0.5,
            "surprise_score": 0.3,
            "access_count": 5,
            "entities": ["Alice", "Bob"],
            "topics": ["work", "project"],
            "related_memory_ids": ["mem-1", "mem-2"],
            "parent_memory_id": "parent-1",
            "superseded_by_id": None,
            "is_pinned": True,
            "is_archived": False,
            "is_consolidated": True,
            "created_at": now,
            "last_accessed_at": now,
            "updated_at": now,
        }

        result = backend._row_to_dict(mock_row)

        assert result["id"] == "test-id"
        assert result["memory_type"] == "semantic"
        assert result["metadata"] == {"key": "value"}
        assert result["entities"] == ["Alice", "Bob"]
        assert result["is_pinned"] is True
        assert result["is_consolidated"] is True

    def test_row_to_dict_null_fields(self) -> None:
        """Row to dict should handle null fields."""
        backend = PostgresMetadataBackend()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": "test-id",
            "memory_type": "episodic",
            "content": "",
            "metadata": None,
            "agent_id": None,
            "user_id": None,
            "source": "conversation",
            "source_id": None,
            "strength": 1.0,
            "initial_strength": 1.0,
            "importance": 0.5,
            "emotional_valence": 0.0,
            "surprise_score": 0.0,
            "access_count": 0,
            "entities": None,
            "topics": None,
            "related_memory_ids": None,
            "parent_memory_id": None,
            "superseded_by_id": None,
            "is_pinned": False,
            "is_archived": False,
            "is_consolidated": False,
            "created_at": now,
            "last_accessed_at": now,
            "updated_at": now,
        }

        result = backend._row_to_dict(mock_row)

        assert result["metadata"] == {}
        assert result["entities"] == []
        assert result["topics"] == []
        assert result["related_memory_ids"] == []

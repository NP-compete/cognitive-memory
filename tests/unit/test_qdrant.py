"""Tests for Qdrant backend."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognitive_memory.storage.vector.base import SearchResult
from cognitive_memory.storage.vector.qdrant import QdrantBackend


class TestQdrantBackend:
    """Tests for QdrantBackend."""

    def test_default_values(self) -> None:
        """QdrantBackend should have sensible defaults."""
        backend = QdrantBackend()

        assert backend.url == "http://localhost:6333"
        assert backend.api_key is None
        assert backend.collection_name == "cognitive_memory"
        assert backend.vector_size == 1536
        assert backend.distance == "cosine"
        assert backend.on_disk is False
        assert backend.prefer_grpc is False
        assert backend.timeout == 30

    def test_custom_values(self) -> None:
        """QdrantBackend should accept custom values."""
        backend = QdrantBackend(
            url="http://qdrant:6333",
            api_key="test-key",
            collection_name="test_collection",
            vector_size=768,
            distance="dot",
            on_disk=True,
            prefer_grpc=True,
            timeout=60,
        )

        assert backend.url == "http://qdrant:6333"
        assert backend.api_key == "test-key"
        assert backend.collection_name == "test_collection"
        assert backend.vector_size == 768
        assert backend.distance == "dot"
        assert backend.on_disk is True
        assert backend.prefer_grpc is True
        assert backend.timeout == 60

    def test_initial_state(self) -> None:
        """Backend should start uninitialized."""
        backend = QdrantBackend()

        assert backend._client is None
        assert backend._initialized is False


class TestQdrantOperationsWithMockedClient:
    """Tests for CRUD operations with pre-mocked client."""

    @pytest.fixture
    def mock_backend(self) -> QdrantBackend:
        """Create a backend with mocked client."""
        backend = QdrantBackend()
        backend._client = AsyncMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_backend: QdrantBackend) -> None:
        """Search should return SearchResult list."""
        mock_result = MagicMock()
        mock_result.id = "test-id"
        mock_result.score = 0.95
        mock_result.payload = {"key": "value"}
        mock_backend._client.search.return_value = [mock_result]

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
    async def test_search_empty_results(self, mock_backend: QdrantBackend) -> None:
        """Search should handle empty results."""
        mock_backend._client.search.return_value = []

        results = await mock_backend.search(embedding=[0.1] * 10)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_existing(self, mock_backend: QdrantBackend) -> None:
        """Get should return SearchResult for existing vector."""
        mock_point = MagicMock()
        mock_point.id = "test-id"
        mock_point.payload = {"key": "value"}
        mock_backend._client.retrieve.return_value = [mock_point]

        result = await mock_backend.get("test-id")

        assert result is not None
        assert result.id == "test-id"
        assert result.payload == {"key": "value"}
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mock_backend: QdrantBackend) -> None:
        """Get should return None for nonexistent vector."""
        mock_backend._client.retrieve.return_value = []

        result = await mock_backend.get("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_backend: QdrantBackend) -> None:
        """Delete should return False for nonexistent vector."""
        mock_backend._client.retrieve.return_value = []

        result = await mock_backend.delete("nonexistent-id")

        assert result is False
        mock_backend._client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_upsert_empty(self, mock_backend: QdrantBackend) -> None:
        """Batch upsert with empty list should return 0."""
        count = await mock_backend.batch_upsert([])

        assert count == 0
        mock_backend._client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_delete_empty(self, mock_backend: QdrantBackend) -> None:
        """Batch delete with empty list should return 0."""
        count = await mock_backend.batch_delete([])

        assert count == 0
        mock_backend._client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_count(self, mock_backend: QdrantBackend) -> None:
        """Count should return vector count."""
        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_backend._client.get_collection.return_value = mock_info

        count = await mock_backend.count()

        assert count == 42
        mock_backend._client.get_collection.assert_called_once_with(mock_backend.collection_name)

    @pytest.mark.asyncio
    async def test_count_zero(self, mock_backend: QdrantBackend) -> None:
        """Count should handle zero or None."""
        mock_info = MagicMock()
        mock_info.points_count = None
        mock_backend._client.get_collection.return_value = mock_info

        count = await mock_backend.count()

        assert count == 0


class TestQdrantClose:
    """Tests for close operation."""

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        """Close should clean up client."""
        backend = QdrantBackend()
        mock_client = AsyncMock()
        backend._client = mock_client
        backend._initialized = True

        await backend.close()

        mock_client.close.assert_called_once()
        assert backend._client is None
        assert backend._initialized is False

    @pytest.mark.asyncio
    async def test_close_without_client(self) -> None:
        """Close should handle no client gracefully."""
        backend = QdrantBackend()
        backend._client = None
        backend._initialized = False

        # Should not raise
        await backend.close()

        assert backend._client is None
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

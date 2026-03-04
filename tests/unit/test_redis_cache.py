"""Tests for Redis cache backend."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognitive_memory.storage.cache.redis import RedisCacheBackend


class TestRedisCacheBackend:
    """Tests for RedisCacheBackend."""

    def test_default_values(self) -> None:
        """RedisCacheBackend should have sensible defaults."""
        backend = RedisCacheBackend()

        assert backend.url == "redis://localhost:6379/0"
        assert backend.prefix == "cognitive_memory:"
        assert backend.default_ttl == 3600
        assert backend.max_connections == 10
        assert backend.decode_responses is True

    def test_custom_values(self) -> None:
        """RedisCacheBackend should accept custom values."""
        backend = RedisCacheBackend(
            url="redis://user:pass@redis:6379/1",
            prefix="custom:",
            default_ttl=7200,
            max_connections=20,
        )

        assert backend.url == "redis://user:pass@redis:6379/1"
        assert backend.prefix == "custom:"
        assert backend.default_ttl == 7200
        assert backend.max_connections == 20

    def test_initial_state(self) -> None:
        """Backend should start uninitialized."""
        backend = RedisCacheBackend()

        assert backend._client is None
        assert backend._initialized is False

    def test_make_key(self) -> None:
        """_make_key should add prefix."""
        backend = RedisCacheBackend(prefix="test:")

        assert backend._make_key("foo") == "test:foo"
        assert backend._make_key("bar:baz") == "test:bar:baz"


class TestRedisOperationsWithMockedClient:
    """Tests for operations with pre-mocked client."""

    @pytest.fixture
    def mock_backend(self) -> RedisCacheBackend:
        """Create a backend with mocked client."""
        backend = RedisCacheBackend()
        backend._client = AsyncMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_get_existing(self, mock_backend: RedisCacheBackend) -> None:
        """Get should return deserialized value."""
        mock_backend._client.get.return_value = '{"key": "value"}'

        result = await mock_backend.get("test-key")

        assert result == {"key": "value"}
        mock_backend._client.get.assert_called_once_with("cognitive_memory:test-key")

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mock_backend: RedisCacheBackend) -> None:
        """Get should return None for missing key."""
        mock_backend._client.get.return_value = None

        result = await mock_backend.get("missing-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_primitive(self, mock_backend: RedisCacheBackend) -> None:
        """Get should handle primitive values."""
        mock_backend._client.get.return_value = "42"

        result = await mock_backend.get("counter")

        assert result == 42

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, mock_backend: RedisCacheBackend) -> None:
        """Set should use setex with TTL."""
        await mock_backend.set("key", {"data": "value"}, ttl_seconds=300)

        mock_backend._client.setex.assert_called_once_with(
            "cognitive_memory:key",
            300,
            '{"data": "value"}',
        )

    @pytest.mark.asyncio
    async def test_set_without_ttl(self, mock_backend: RedisCacheBackend) -> None:
        """Set should use set without TTL when None."""
        mock_backend.default_ttl = None

        await mock_backend.set("key", {"data": "value"}, ttl_seconds=None)

        mock_backend._client.set.assert_called_once_with(
            "cognitive_memory:key",
            '{"data": "value"}',
        )

    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_backend: RedisCacheBackend) -> None:
        """Delete should return True for existing key."""
        mock_backend._client.delete.return_value = 1

        result = await mock_backend.delete("key")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, mock_backend: RedisCacheBackend) -> None:
        """Delete should return False for missing key."""
        mock_backend._client.delete.return_value = 0

        result = await mock_backend.delete("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, mock_backend: RedisCacheBackend) -> None:
        """Exists should return True for existing key."""
        mock_backend._client.exists.return_value = 1

        result = await mock_backend.exists("key")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, mock_backend: RedisCacheBackend) -> None:
        """Exists should return False for missing key."""
        mock_backend._client.exists.return_value = 0

        result = await mock_backend.exists("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_many_empty(self, mock_backend: RedisCacheBackend) -> None:
        """Get many with empty list should return empty dict."""
        result = await mock_backend.get_many([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_many(self, mock_backend: RedisCacheBackend) -> None:
        """Get many should return dict of found values."""
        mock_backend._client.mget.return_value = [
            '{"a": 1}',
            None,
            '{"c": 3}',
        ]

        result = await mock_backend.get_many(["key1", "key2", "key3"])

        assert result == {"key1": {"a": 1}, "key3": {"c": 3}}

    @pytest.mark.asyncio
    async def test_set_many_empty(self, mock_backend: RedisCacheBackend) -> None:
        """Set many with empty dict should do nothing."""
        await mock_backend.set_many({})

        mock_backend._client.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_many(self, mock_backend: RedisCacheBackend) -> None:
        """Set many should use pipeline."""
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock()
        mock_pipe.setex = MagicMock()
        mock_pipe.set = MagicMock()
        mock_backend._client.pipeline = MagicMock(return_value=mock_pipe)

        await mock_backend.set_many({"k1": "v1", "k2": "v2"}, ttl_seconds=60)

        mock_backend._client.pipeline.assert_called_once()
        assert mock_pipe.setex.call_count == 2
        mock_pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_incr(self, mock_backend: RedisCacheBackend) -> None:
        """Incr should increment counter."""
        mock_backend._client.incrby.return_value = 5

        result = await mock_backend.incr("counter", 3)

        assert result == 5
        mock_backend._client.incrby.assert_called_once_with(
            "cognitive_memory:counter", 3
        )

    @pytest.mark.asyncio
    async def test_expire(self, mock_backend: RedisCacheBackend) -> None:
        """Expire should set TTL on key."""
        mock_backend._client.expire.return_value = True

        result = await mock_backend.expire("key", 600)

        assert result is True
        mock_backend._client.expire.assert_called_once_with(
            "cognitive_memory:key", 600
        )

    @pytest.mark.asyncio
    async def test_clear_all(self, mock_backend: RedisCacheBackend) -> None:
        """Clear without pattern should delete all prefixed keys."""
        mock_backend._client.scan.return_value = (0, ["cognitive_memory:k1"])
        mock_backend._client.delete.return_value = 1

        count = await mock_backend.clear()

        assert count == 1
        mock_backend._client.scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_pattern(self, mock_backend: RedisCacheBackend) -> None:
        """Clear with pattern should match specific keys."""
        mock_backend._client.scan.return_value = (0, ["cognitive_memory:working_memory:*"])
        mock_backend._client.delete.return_value = 2

        count = await mock_backend.clear("working_memory:*")

        assert count == 2


class TestRedisClose:
    """Tests for close operation."""

    @pytest.mark.asyncio
    async def test_close_with_client(self) -> None:
        """Close should clean up client."""
        backend = RedisCacheBackend()
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
        backend = RedisCacheBackend()
        backend._client = None
        backend._initialized = False

        await backend.close()

        assert backend._client is None
        assert backend._initialized is False


class TestWorkingMemoryMethods:
    """Tests for working memory specific methods."""

    @pytest.fixture
    def mock_backend(self) -> RedisCacheBackend:
        """Create a backend with mocked client."""
        backend = RedisCacheBackend()
        backend._client = AsyncMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_set_working_memory(self, mock_backend: RedisCacheBackend) -> None:
        """Set working memory should store memories list."""
        memories = [{"id": "m1"}, {"id": "m2"}]

        await mock_backend.set_working_memory("agent-1", memories, ttl_seconds=600)

        mock_backend._client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_working_memory_exists(
        self, mock_backend: RedisCacheBackend
    ) -> None:
        """Get working memory should return memories list."""
        mock_backend._client.get.return_value = '[{"id": "m1"}]'

        result = await mock_backend.get_working_memory("agent-1")

        assert result == [{"id": "m1"}]

    @pytest.mark.asyncio
    async def test_get_working_memory_empty(
        self, mock_backend: RedisCacheBackend
    ) -> None:
        """Get working memory should return empty list if not found."""
        mock_backend._client.get.return_value = None

        result = await mock_backend.get_working_memory("agent-1")

        assert result == []


class TestRetrievalCacheMethods:
    """Tests for retrieval cache methods."""

    @pytest.fixture
    def mock_backend(self) -> RedisCacheBackend:
        """Create a backend with mocked client."""
        backend = RedisCacheBackend()
        backend._client = AsyncMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_cache_retrieval(self, mock_backend: RedisCacheBackend) -> None:
        """Cache retrieval should store results."""
        results = [{"id": "m1", "score": 0.9}]

        await mock_backend.cache_retrieval("hash123", results)

        mock_backend._client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_retrieval_exists(
        self, mock_backend: RedisCacheBackend
    ) -> None:
        """Get cached retrieval should return results."""
        mock_backend._client.get.return_value = '[{"id": "m1"}]'

        result = await mock_backend.get_cached_retrieval("hash123")

        assert result == [{"id": "m1"}]

    @pytest.mark.asyncio
    async def test_get_cached_retrieval_missing(
        self, mock_backend: RedisCacheBackend
    ) -> None:
        """Get cached retrieval should return None if not found."""
        mock_backend._client.get.return_value = None

        result = await mock_backend.get_cached_retrieval("hash123")

        assert result is None


class TestScoreCacheMethods:
    """Tests for score cache methods."""

    @pytest.fixture
    def mock_backend(self) -> RedisCacheBackend:
        """Create a backend with mocked client."""
        backend = RedisCacheBackend()
        backend._client = AsyncMock()
        backend._initialized = True
        return backend

    @pytest.mark.asyncio
    async def test_cache_score(self, mock_backend: RedisCacheBackend) -> None:
        """Cache score should store score value."""
        await mock_backend.cache_score("mem-1", "decay", 0.85)

        mock_backend._client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_score_exists(
        self, mock_backend: RedisCacheBackend
    ) -> None:
        """Get cached score should return float."""
        mock_backend._client.get.return_value = "0.85"

        result = await mock_backend.get_cached_score("mem-1", "decay")

        assert result == 0.85

    @pytest.mark.asyncio
    async def test_get_cached_score_missing(
        self, mock_backend: RedisCacheBackend
    ) -> None:
        """Get cached score should return None if not found."""
        mock_backend._client.get.return_value = None

        result = await mock_backend.get_cached_score("mem-1", "decay")

        assert result is None

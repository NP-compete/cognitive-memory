"""Redis cache backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from cognitive_memory.storage.cache.base import CacheBackend

logger = logging.getLogger(__name__)


@dataclass
class RedisCacheBackend(CacheBackend):
    """
    Redis cache backend.

    Provides fast caching for:
    - Working memory (current context)
    - Recent retrieval results
    - Computed decay/importance scores
    - Session state

    Attributes:
        url: Redis connection URL.
        prefix: Key prefix for namespacing.
        default_ttl: Default TTL in seconds (None = no expiry).
        max_connections: Maximum connection pool size.
        decode_responses: Whether to decode responses as strings.
    """

    url: str = "redis://localhost:6379/0"
    prefix: str = "cognitive_memory:"
    default_ttl: int | None = 3600
    max_connections: int = 10
    decode_responses: bool = True
    _client: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return

        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "redis is required for RedisCacheBackend. Install it with: pip install redis"
            ) from e

        self._client = aioredis.from_url(  # type: ignore[no-untyped-call]
            self.url,
            max_connections=self.max_connections,
            decode_responses=self.decode_responses,
        )

        # Test connection
        await self._client.ping()

        self._initialized = True
        logger.info(f"RedisCacheBackend initialized: {self.url}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the backend is initialized."""
        if not self._initialized:
            await self.initialize()

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        await self._ensure_initialized()

        full_key = self._make_key(key)
        value = await self._client.get(full_key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set a value in cache."""
        await self._ensure_initialized()

        full_key = self._make_key(key)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        serialized = json.dumps(value)

        if ttl:
            await self._client.setex(full_key, ttl, serialized)
        else:
            await self._client.set(full_key, serialized)

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        await self._ensure_initialized()

        full_key = self._make_key(key)
        result = await self._client.delete(full_key)
        return bool(result > 0)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        await self._ensure_initialized()

        full_key = self._make_key(key)
        result = await self._client.exists(full_key)
        return bool(result > 0)

    async def clear(self, pattern: str | None = None) -> int:
        """Clear cache entries matching pattern."""
        await self._ensure_initialized()

        full_pattern = self._make_key(pattern) if pattern else self._make_key("*")

        # Use SCAN to find keys (safer than KEYS for large datasets)
        deleted = 0
        cursor = 0

        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor,
                match=full_pattern,
                count=100,
            )

            if keys:
                deleted += await self._client.delete(*keys)

            if cursor == 0:
                break

        return deleted

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        await self._ensure_initialized()

        if not keys:
            return {}

        full_keys = [self._make_key(k) for k in keys]
        values = await self._client.mget(full_keys)

        result = {}
        for key, value in zip(keys, values, strict=True):
            if value is not None:
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[key] = value

        return result

    async def set_many(
        self,
        items: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Set multiple values in cache."""
        await self._ensure_initialized()

        if not items:
            return

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # Use pipeline for efficiency
        pipe = self._client.pipeline()

        for key, value in items.items():
            full_key = self._make_key(key)
            serialized = json.dumps(value)

            if ttl:
                pipe.setex(full_key, ttl, serialized)
            else:
                pipe.set(full_key, serialized)

        await pipe.execute()

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        await self._ensure_initialized()

        full_key = self._make_key(key)
        result = await self._client.incrby(full_key, amount)
        return int(result)

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiry on an existing key."""
        await self._ensure_initialized()

        full_key = self._make_key(key)
        result = await self._client.expire(full_key, ttl_seconds)
        return bool(result)

    # Working memory specific methods

    async def set_working_memory(
        self,
        agent_id: str,
        memories: list[dict[str, Any]],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set working memory for an agent.

        Args:
            agent_id: Agent identifier.
            memories: List of memory dicts in working memory.
            ttl_seconds: TTL for the working memory.
        """
        key = f"working_memory:{agent_id}"
        await self.set(key, memories, ttl_seconds)

    async def get_working_memory(
        self,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get working memory for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            List of memory dicts, or empty list if not found.
        """
        key = f"working_memory:{agent_id}"
        result = await self.get(key)
        return result if isinstance(result, list) else []

    async def cache_retrieval(
        self,
        query_hash: str,
        results: list[dict[str, Any]],
        ttl_seconds: int = 300,
    ) -> None:
        """
        Cache retrieval results.

        Args:
            query_hash: Hash of the query for deduplication.
            results: Retrieved memories.
            ttl_seconds: Cache TTL (default 5 minutes).
        """
        key = f"retrieval:{query_hash}"
        await self.set(key, results, ttl_seconds)

    async def get_cached_retrieval(
        self,
        query_hash: str,
    ) -> list[dict[str, Any]] | None:
        """
        Get cached retrieval results.

        Args:
            query_hash: Hash of the query.

        Returns:
            Cached results or None if not found/expired.
        """
        key = f"retrieval:{query_hash}"
        result = await self.get(key)
        return result if isinstance(result, list) else None

    async def cache_score(
        self,
        memory_id: str,
        score_type: str,
        score: float,
        ttl_seconds: int = 60,
    ) -> None:
        """
        Cache a computed score.

        Args:
            memory_id: Memory identifier.
            score_type: Type of score (decay, importance, etc.).
            score: Computed score value.
            ttl_seconds: Cache TTL.
        """
        key = f"score:{score_type}:{memory_id}"
        await self.set(key, score, ttl_seconds)

    async def get_cached_score(
        self,
        memory_id: str,
        score_type: str,
    ) -> float | None:
        """
        Get a cached score.

        Args:
            memory_id: Memory identifier.
            score_type: Type of score.

        Returns:
            Cached score or None if not found/expired.
        """
        key = f"score:{score_type}:{memory_id}"
        result = await self.get(key)
        return float(result) if result is not None else None

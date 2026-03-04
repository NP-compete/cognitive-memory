"""Base interface for cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Provides fast access to frequently used data like
    working memory, recent retrievals, and computed scores.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up resources."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """
        Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON serializable).
            ttl_seconds: Time-to-live in seconds (None = no expiry).
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if exists and not expired.
        """

    @abstractmethod
    async def clear(self, pattern: str | None = None) -> int:
        """
        Clear cache entries.

        Args:
            pattern: Optional glob pattern to match keys.
                     None clears all keys.

        Returns:
            Number of keys deleted.
        """

    @abstractmethod
    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys.

        Returns:
            Dict of key -> value for found keys.
        """

    @abstractmethod
    async def set_many(
        self,
        items: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set multiple values in cache.

        Args:
            items: Dict of key -> value.
            ttl_seconds: TTL for all items.
        """

    @abstractmethod
    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            key: Cache key.
            amount: Amount to increment by.

        Returns:
            New value after increment.
        """

    @abstractmethod
    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """
        Set expiry on an existing key.

        Args:
            key: Cache key.
            ttl_seconds: New TTL in seconds.

        Returns:
            True if key exists and TTL was set.
        """

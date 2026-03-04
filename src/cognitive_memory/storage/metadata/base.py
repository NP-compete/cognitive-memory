"""Base interface for metadata storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


class MetadataBackend(ABC):
    """
    Abstract base class for metadata storage backends.

    Stores memory metadata (everything except embeddings).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend and create necessary resources."""

    @abstractmethod
    async def close(self) -> None:
        """Close connections and clean up resources."""

    @abstractmethod
    async def save_memory(self, memory: dict[str, Any]) -> None:
        """
        Save or update a memory's metadata.

        Args:
            memory: Memory data as dict (must include 'id').
        """

    @abstractmethod
    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """
        Get a memory by ID.

        Args:
            memory_id: ID of the memory.

        Returns:
            Memory data as dict, or None if not found.
        """

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """

    @abstractmethod
    async def list_memories(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
        memory_type: str | None = None,
        is_archived: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List memories with optional filters.

        Args:
            agent_id: Filter by agent ID.
            user_id: Filter by user ID.
            memory_type: Filter by memory type.
            is_archived: Filter by archived status.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of memory dicts.
        """

    @abstractmethod
    async def update_access(
        self,
        memory_id: str,
        accessed_at: datetime | None = None,
    ) -> None:
        """
        Update memory access metadata.

        Args:
            memory_id: ID of the memory.
            accessed_at: Access timestamp (defaults to now).
        """

    @abstractmethod
    async def batch_save(self, memories: list[dict[str, Any]]) -> int:
        """
        Save multiple memories.

        Args:
            memories: List of memory dicts.

        Returns:
            Number of memories saved.
        """

    @abstractmethod
    async def batch_delete(self, memory_ids: list[str]) -> int:
        """
        Delete multiple memories.

        Args:
            memory_ids: List of memory IDs.

        Returns:
            Number of memories deleted.
        """

    @abstractmethod
    async def count(
        self,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """
        Count memories with optional filters.

        Args:
            agent_id: Filter by agent ID.
            user_id: Filter by user ID.

        Returns:
            Memory count.
        """

    @abstractmethod
    async def get_memories_by_ids(
        self,
        memory_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Get multiple memories by IDs.

        Args:
            memory_ids: List of memory IDs.

        Returns:
            List of memory dicts (in same order as IDs).
        """

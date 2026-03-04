"""Base interface for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """
    Result from vector search.

    Attributes:
        id: Unique identifier of the vector.
        score: Similarity score (higher is more similar).
        payload: Associated metadata/payload.
    """

    id: str
    score: float
    payload: dict[str, Any]


class VectorBackend(ABC):
    """
    Abstract base class for vector storage backends.

    Defines the interface that all vector backends must implement.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend and create necessary resources.

        Should be called before any other operations.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Close connections and clean up resources.
        """

    @abstractmethod
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

    @abstractmethod
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
            filter_conditions: Optional filter on payload fields.

        Returns:
            List of SearchResult sorted by similarity.
        """

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.

        Args:
            id: ID of the vector to delete.

        Returns:
            True if deleted, False if not found.
        """

    @abstractmethod
    async def get(self, id: str) -> SearchResult | None:
        """
        Get a vector by ID.

        Args:
            id: ID of the vector to retrieve.

        Returns:
            SearchResult if found, None otherwise.
        """

    @abstractmethod
    async def batch_upsert(
        self,
        items: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> int:
        """
        Insert or update multiple vectors.

        Args:
            items: List of (id, embedding, payload) tuples.

        Returns:
            Number of vectors upserted.
        """

    @abstractmethod
    async def batch_delete(self, ids: list[str]) -> int:
        """
        Delete multiple vectors.

        Args:
            ids: List of IDs to delete.

        Returns:
            Number of vectors deleted.
        """

    @abstractmethod
    async def count(self) -> int:
        """
        Get the total number of vectors.

        Returns:
            Vector count.
        """

    @abstractmethod
    async def clear(self) -> None:
        """
        Delete all vectors.
        """

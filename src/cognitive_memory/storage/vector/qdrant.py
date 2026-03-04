"""Qdrant vector storage backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cognitive_memory.storage.vector.base import SearchResult, VectorBackend

logger = logging.getLogger(__name__)


@dataclass
class QdrantBackend(VectorBackend):
    """
    Qdrant vector storage backend.

    Provides vector storage and similarity search using Qdrant.
    Supports both Qdrant Cloud and self-hosted instances.

    Attributes:
        url: Qdrant server URL (e.g., "http://localhost:6333").
        api_key: Optional API key for Qdrant Cloud.
        collection_name: Name of the collection to use.
        vector_size: Dimension of vectors.
        distance: Distance metric ("cosine", "euclid", "dot").
        on_disk: Whether to store vectors on disk.
        prefer_grpc: Whether to prefer gRPC over HTTP.
        timeout: Request timeout in seconds.
    """

    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "cognitive_memory"
    vector_size: int = 1536
    distance: str = "cosine"
    on_disk: bool = False
    prefer_grpc: bool = False
    timeout: int = 30
    _client: Any = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    async def initialize(self) -> None:
        """
        Initialize the Qdrant client and ensure collection exists.
        """
        if self._initialized:
            return

        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError(
                "qdrant-client is required for QdrantBackend. "
                "Install it with: pip install qdrant-client"
            ) from e

        # Map distance string to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        qdrant_distance = distance_map.get(self.distance.lower(), Distance.COSINE)

        # Create async client
        self._client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
            timeout=self.timeout,
        )

        # Check if collection exists
        collections = await self._client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=qdrant_distance,
                    on_disk=self.on_disk,
                ),
            )

        self._initialized = True
        logger.info(f"QdrantBackend initialized: {self.collection_name}")

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
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

        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=id,
            vector=embedding,
            payload=payload or {},
        )

        await self._client.upsert(
            collection_name=self.collection_name,
            points=[point],
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
            filter_conditions: Optional filter on payload fields.

        Returns:
            List of SearchResult sorted by similarity.
        """
        await self._ensure_initialized()

        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = self._build_filter(filter_conditions)

        results = await self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
            )
            for r in results
        ]

    def _build_filter(self, conditions: dict[str, Any]) -> Any:
        """
        Build Qdrant filter from conditions dict.

        Supports simple equality filters.
        For complex filters, use Qdrant's filter models directly.

        Args:
            conditions: Dict of field -> value conditions.

        Returns:
            Qdrant Filter object.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must_conditions = []
        for key, value in conditions.items():
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=must_conditions)  # type: ignore[arg-type]

    async def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.

        Args:
            id: ID of the vector to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()

        from qdrant_client.models import PointIdsList

        # Check if exists first
        existing = await self.get(id)
        if existing is None:
            return False

        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[id]),
        )
        return True

    async def get(self, id: str) -> SearchResult | None:
        """
        Get a vector by ID.

        Args:
            id: ID of the vector to retrieve.

        Returns:
            SearchResult if found, None otherwise.
        """
        await self._ensure_initialized()

        results = await self._client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return None

        point = results[0]
        return SearchResult(
            id=str(point.id),
            score=1.0,  # No score for direct retrieval
            payload=point.payload or {},
        )

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
        await self._ensure_initialized()

        if not items:
            return 0

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=id,
                vector=embedding,
                payload=payload or {},
            )
            for id, embedding, payload in items
        ]

        await self._client.upsert(
            collection_name=self.collection_name,
            points=points,
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

        from qdrant_client.models import PointIdsList

        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=list(ids)),
        )

        return len(ids)

    async def count(self) -> int:
        """
        Get the total number of vectors.

        Returns:
            Vector count.
        """
        await self._ensure_initialized()

        info = await self._client.get_collection(self.collection_name)
        return info.points_count or 0

    async def clear(self) -> None:
        """Delete all vectors by recreating the collection."""
        await self._ensure_initialized()

        from qdrant_client.models import Distance, VectorParams

        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        qdrant_distance = distance_map.get(self.distance.lower(), Distance.COSINE)

        # Delete and recreate collection
        await self._client.delete_collection(self.collection_name)
        await self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=qdrant_distance,
                on_disk=self.on_disk,
            ),
        )

    async def create_payload_index(
        self,
        field_name: str,
        field_type: str = "keyword",
    ) -> None:
        """
        Create an index on a payload field for faster filtering.

        Args:
            field_name: Name of the payload field.
            field_type: Type of index ("keyword", "integer", "float", "bool").
        """
        await self._ensure_initialized()

        from qdrant_client.models import PayloadSchemaType

        type_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "bool": PayloadSchemaType.BOOL,
        }

        schema_type = type_map.get(field_type, PayloadSchemaType.KEYWORD)

        await self._client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )

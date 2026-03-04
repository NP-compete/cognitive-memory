"""Memory consolidation engine for episodic-to-semantic transformation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)


class MemoryProtocol(Protocol):
    """Protocol for memory objects used by consolidation engine."""

    id: str
    memory_type: str
    content: str
    embedding: list[float]
    created_at: datetime
    access_count: int
    importance: float
    strength: float
    entities: list[str]
    topics: list[str]
    is_consolidated: bool
    agent_id: str | None


class ConsolidationConfigProtocol(Protocol):
    """Protocol for consolidation configuration."""

    min_memories: int
    similarity_threshold: float
    min_access_count: int
    min_age_hours: float
    max_cluster_size: int
    preserve_source_memories: bool


@dataclass
class ConsolidationCandidate:
    """A group of memories that can be consolidated."""

    memories: list[MemoryProtocol]
    centroid: list[float]
    similarity_score: float
    combined_importance: float
    shared_entities: list[str]
    shared_topics: list[str]


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""

    consolidated_memory_id: str
    source_memory_ids: list[str]
    memory_type: str
    content_summary: str
    combined_importance: float
    shared_entities: list[str]
    shared_topics: list[str]
    centroid_embedding: list[float]
    consolidation_timestamp: datetime


@dataclass
class ConsolidationEngine:
    """
    Engine for consolidating episodic memories into semantic memories.

    Implements memory consolidation similar to human sleep-based
    memory consolidation, where related episodic memories are
    combined into more abstract semantic knowledge.

    Attributes:
        min_memories: Minimum memories needed to form a cluster.
        similarity_threshold: Minimum similarity to be in same cluster.
        min_access_count: Minimum accesses before eligible.
        min_age_hours: Minimum age in hours before eligible.
        max_cluster_size: Maximum memories per cluster.
        preserve_source_memories: Whether to keep source memories.
    """

    min_memories: int = 3
    similarity_threshold: float = 0.75
    min_access_count: int = 2
    min_age_hours: float = 24.0
    max_cluster_size: int = 10
    preserve_source_memories: bool = True
    _consolidation_count: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(cls, config: ConsolidationConfigProtocol) -> ConsolidationEngine:
        """Create engine from configuration object."""
        return cls(
            min_memories=config.min_memories,
            similarity_threshold=config.similarity_threshold,
            min_access_count=config.min_access_count,
            min_age_hours=config.min_age_hours,
            max_cluster_size=config.max_cluster_size,
            preserve_source_memories=config.preserve_source_memories,
        )

    def find_consolidation_candidates(
        self,
        memories: list[MemoryProtocol],
        current_time: datetime | None = None,
    ) -> list[ConsolidationCandidate]:
        """
        Find groups of memories that can be consolidated.

        Args:
            memories: List of memories to analyze.
            current_time: Current timestamp (defaults to now).

        Returns:
            List of consolidation candidates (memory clusters).
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Filter eligible memories
        eligible = self._filter_eligible_memories(memories, current_time)

        if len(eligible) < self.min_memories:
            return []

        # Cluster by similarity
        clusters = self._cluster_by_similarity(eligible)

        # Convert to candidates
        candidates = []
        for cluster in clusters:
            if len(cluster) >= self.min_memories:
                candidate = self._create_candidate(cluster)
                candidates.append(candidate)

        return candidates

    def _filter_eligible_memories(
        self,
        memories: list[MemoryProtocol],
        current_time: datetime,
    ) -> list[MemoryProtocol]:
        """Filter memories eligible for consolidation."""
        eligible = []
        min_age_seconds = self.min_age_hours * 3600

        for memory in memories:
            # Skip already consolidated
            if memory.is_consolidated:
                continue

            # Skip non-episodic
            if memory.memory_type != "episodic":
                continue

            # Check access count
            if memory.access_count < self.min_access_count:
                continue

            # Check age
            age = (current_time - memory.created_at).total_seconds()
            if age < min_age_seconds:
                continue

            # Check has embedding
            if not memory.embedding:
                continue

            eligible.append(memory)

        return eligible

    def _cluster_by_similarity(
        self,
        memories: list[MemoryProtocol],
    ) -> list[list[MemoryProtocol]]:
        """Cluster memories by embedding similarity."""
        if not memories:
            return []

        # Simple greedy clustering
        clusters: list[list[MemoryProtocol]] = []
        assigned = set()

        for memory in memories:
            if memory.id in assigned:
                continue

            # Start new cluster
            cluster = [memory]
            assigned.add(memory.id)

            # Find similar memories
            for other in memories:
                if other.id in assigned:
                    continue

                if len(cluster) >= self.max_cluster_size:
                    break

                similarity = self._cosine_similarity(
                    memory.embedding,
                    other.embedding,
                )

                if similarity >= self.similarity_threshold:
                    cluster.append(other)
                    assigned.add(other.id)

            clusters.append(cluster)

        return clusters

    def _cosine_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _create_candidate(
        self,
        cluster: list[MemoryProtocol],
    ) -> ConsolidationCandidate:
        """Create a consolidation candidate from a cluster."""
        # Calculate centroid
        centroid = self._calculate_centroid([m.embedding for m in cluster])

        # Calculate average similarity to centroid
        similarities = [
            self._cosine_similarity(m.embedding, centroid) for m in cluster
        ]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Combine importance scores
        combined_importance = sum(m.importance for m in cluster) / len(cluster)

        # Find shared entities and topics
        shared_entities = self._find_shared_items([m.entities for m in cluster])
        shared_topics = self._find_shared_items([m.topics for m in cluster])

        return ConsolidationCandidate(
            memories=cluster,
            centroid=centroid,
            similarity_score=avg_similarity,
            combined_importance=combined_importance,
            shared_entities=shared_entities,
            shared_topics=shared_topics,
        )

    def _calculate_centroid(
        self,
        embeddings: list[list[float]],
    ) -> list[float]:
        """Calculate centroid of embedding vectors."""
        if not embeddings:
            return []

        dim = len(embeddings[0])
        centroid = [0.0] * dim

        for embedding in embeddings:
            for i, val in enumerate(embedding):
                centroid[i] += val

        n = len(embeddings)
        return [c / n for c in centroid]

    def _find_shared_items(
        self,
        item_lists: list[list[str]],
    ) -> list[str]:
        """Find items that appear in multiple lists."""
        if not item_lists:
            return []

        # Count occurrences
        counts: dict[str, int] = {}
        for items in item_lists:
            for item in items:
                counts[item] = counts.get(item, 0) + 1

        # Return items appearing in at least half the lists
        threshold = len(item_lists) // 2
        return [item for item, count in counts.items() if count > threshold]

    def consolidate(
        self,
        candidate: ConsolidationCandidate,
        content_generator: ContentGenerator | None = None,
    ) -> ConsolidationResult:
        """
        Consolidate a candidate into a semantic memory.

        Args:
            candidate: The consolidation candidate.
            content_generator: Optional function to generate summary content.

        Returns:
            Result containing the new semantic memory details.
        """
        self._consolidation_count += 1

        # Generate content summary
        if content_generator:
            content_summary = content_generator(candidate)
        else:
            content_summary = self._default_content_summary(candidate)

        result = ConsolidationResult(
            consolidated_memory_id=str(uuid4()),
            source_memory_ids=[m.id for m in candidate.memories],
            memory_type="semantic",
            content_summary=content_summary,
            combined_importance=candidate.combined_importance,
            shared_entities=candidate.shared_entities,
            shared_topics=candidate.shared_topics,
            centroid_embedding=candidate.centroid,
            consolidation_timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            f"Consolidated {len(candidate.memories)} memories into {result.consolidated_memory_id}"
        )

        return result

    def _default_content_summary(
        self,
        candidate: ConsolidationCandidate,
    ) -> str:
        """Generate default content summary from candidate."""
        contents = [m.content for m in candidate.memories]

        # Simple concatenation with deduplication
        unique_contents = list(dict.fromkeys(contents))

        if len(unique_contents) == 1:
            return unique_contents[0]

        return " | ".join(unique_contents[:5])

    def should_consolidate(
        self,
        candidate: ConsolidationCandidate,
        min_similarity: float | None = None,
        min_importance: float | None = None,
    ) -> bool:
        """
        Check if a candidate should be consolidated.

        Args:
            candidate: The consolidation candidate.
            min_similarity: Minimum similarity threshold.
            min_importance: Minimum importance threshold.

        Returns:
            True if consolidation is recommended.
        """
        if min_similarity is None:
            min_similarity = self.similarity_threshold

        if min_importance is None:
            min_importance = 0.3

        # Check cluster size
        if len(candidate.memories) < self.min_memories:
            return False

        # Check similarity
        if candidate.similarity_score < min_similarity:
            return False

        # Check importance
        return candidate.combined_importance >= min_importance

    def get_consolidation_stats(self) -> dict[str, int]:
        """Get consolidation statistics."""
        return {
            "total_consolidations": self._consolidation_count,
        }


ContentGenerator = Callable[[ConsolidationCandidate], str]

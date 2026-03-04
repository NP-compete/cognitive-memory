"""Retrieval engine for memory search and ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


class MemoryProtocol(Protocol):
    """Protocol for memory objects compatible with RetrievalEngine."""

    id: str
    embedding: list[float]
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    strength: float
    importance: float
    is_archived: bool
    is_pinned: bool


class RetrievalConfigProtocol(Protocol):
    """Protocol for retrieval configuration objects."""

    similarity_weight: float
    strength_weight: float
    importance_weight: float
    recency_weight: float
    default_top_k: int
    max_top_k: int
    mmr_lambda: float
    min_similarity_threshold: float
    include_archived: bool


@dataclass
class RetrievalResult:
    """
    Result of memory retrieval.

    Attributes:
        memory: The retrieved memory.
        similarity_score: Cosine similarity to query (0-1).
        strength_score: Memory strength after decay (0-1).
        importance_score: Memory importance (0-1).
        recency_score: Recency of last access (0-1).
        final_score: Combined weighted score for ranking.
    """

    memory: MemoryProtocol
    similarity_score: float
    strength_score: float
    importance_score: float
    recency_score: float
    final_score: float


@dataclass
class RetrievalEngine:
    """
    Engine for retrieving and ranking memories.

    Combines multiple signals to rank memories:
    - Semantic similarity to query
    - Memory strength (after decay)
    - Memory importance
    - Recency of last access

    Supports Maximal Marginal Relevance (MMR) for diversity.

    Attributes:
        similarity_weight: Weight for semantic similarity.
        strength_weight: Weight for memory strength.
        importance_weight: Weight for importance score.
        recency_weight: Weight for recency.
        default_top_k: Default number of results.
        max_top_k: Maximum allowed results.
        mmr_lambda: MMR diversity parameter (0=diverse, 1=relevant).
        min_similarity_threshold: Minimum similarity to include.
        include_archived: Whether to include archived memories.
        recency_half_life_hours: Hours until recency score halves.
    """

    similarity_weight: float = 0.4
    strength_weight: float = 0.2
    importance_weight: float = 0.2
    recency_weight: float = 0.2
    default_top_k: int = 10
    max_top_k: int = 100
    mmr_lambda: float = 0.7
    min_similarity_threshold: float = 0.3
    include_archived: bool = False
    recency_half_life_hours: float = 24.0

    @classmethod
    def from_config(cls, config: RetrievalConfigProtocol) -> RetrievalEngine:
        """
        Create a RetrievalEngine from configuration.

        Args:
            config: RetrievalConfig instance.

        Returns:
            Configured RetrievalEngine.
        """
        return cls(
            similarity_weight=config.similarity_weight,
            strength_weight=config.strength_weight,
            importance_weight=config.importance_weight,
            recency_weight=config.recency_weight,
            default_top_k=config.default_top_k,
            max_top_k=config.max_top_k,
            mmr_lambda=config.mmr_lambda,
            min_similarity_threshold=config.min_similarity_threshold,
            include_archived=config.include_archived,
        )

    def retrieve(
        self,
        query_embedding: list[float],
        memories: list[MemoryProtocol],
        top_k: int | None = None,
        current_time: datetime | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievalResult]:
        """
        Retrieve and rank memories by relevance to query.

        Args:
            query_embedding: Query vector embedding.
            memories: Candidate memories to search.
            top_k: Number of results (defaults to default_top_k).
            current_time: Time for recency calculation.
            use_mmr: Whether to use MMR for diversity.

        Returns:
            List of RetrievalResult, sorted by final_score descending.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        if top_k is None:
            top_k = self.default_top_k
        top_k = min(top_k, self.max_top_k)

        # Filter memories
        candidates = self._filter_memories(memories)

        if not candidates:
            return []

        # Score all candidates
        scored = self._score_memories(query_embedding, candidates, current_time)

        # Filter by similarity threshold
        scored = [r for r in scored if r.similarity_score >= self.min_similarity_threshold]

        if not scored:
            return []

        if use_mmr:
            return self._apply_mmr(scored, query_embedding, top_k)

        # Sort by final score and return top_k
        scored.sort(key=lambda r: r.final_score, reverse=True)
        return scored[:top_k]

    def _filter_memories(
        self,
        memories: list[MemoryProtocol],
    ) -> list[MemoryProtocol]:
        """
        Filter memories based on retrieval settings.

        Args:
            memories: All candidate memories.

        Returns:
            Filtered list of memories.
        """
        filtered = []
        for m in memories:
            # Skip archived unless configured to include
            if m.is_archived and not self.include_archived:
                continue
            # Skip memories without embeddings
            if not m.embedding:
                continue
            filtered.append(m)
        return filtered

    def _score_memories(
        self,
        query_embedding: list[float],
        memories: list[MemoryProtocol],
        current_time: datetime,
    ) -> list[RetrievalResult]:
        """
        Score all memories against the query.

        Args:
            query_embedding: Query vector.
            memories: Memories to score.
            current_time: Time for recency calculation.

        Returns:
            List of RetrievalResult with scores.
        """
        results = []
        for memory in memories:
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            strength = memory.strength
            importance = memory.importance
            recency = self._calculate_recency_score(memory, current_time)

            final_score = (
                self.similarity_weight * similarity
                + self.strength_weight * strength
                + self.importance_weight * importance
                + self.recency_weight * recency
            )

            results.append(
                RetrievalResult(
                    memory=memory,
                    similarity_score=similarity,
                    strength_score=strength,
                    importance_score=importance,
                    recency_score=recency,
                    final_score=final_score,
                )
            )
        return results

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity (0-1 for normalized vectors).
        """
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        # Clamp to [0, 1] (can be slightly outside due to floating point)
        return max(0.0, min(1.0, similarity))

    def _calculate_recency_score(
        self,
        memory: MemoryProtocol,
        current_time: datetime,
    ) -> float:
        """
        Calculate recency score based on last access time.

        Uses exponential decay from last access.

        Args:
            memory: Memory to score.
            current_time: Current time.

        Returns:
            Recency score (0-1).
        """
        last_accessed = memory.last_accessed_at
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        hours_elapsed = (current_time - last_accessed).total_seconds() / 3600

        if hours_elapsed <= 0:
            return 1.0

        return math.pow(0.5, hours_elapsed / self.recency_half_life_hours)

    def _apply_mmr(
        self,
        scored: list[RetrievalResult],
        _query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Apply Maximal Marginal Relevance for diverse results.

        MMR balances relevance with diversity by penalizing
        results similar to already-selected results.

        MMR(d) = λ * sim(d, q) - (1-λ) * max(sim(d, d_i))

        Args:
            scored: Pre-scored results.
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            Diverse subset of results.
        """
        if not scored or top_k <= 0:
            return []

        # Sort by relevance first
        scored.sort(key=lambda r: r.final_score, reverse=True)

        selected: list[RetrievalResult] = []
        remaining = list(scored)

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = float("-inf")

            for i, candidate in enumerate(remaining):
                # Relevance to query (use final_score as proxy)
                relevance = candidate.final_score

                # Max similarity to already selected
                max_sim_to_selected = 0.0
                for s in selected:
                    sim = self._cosine_similarity(
                        candidate.memory.embedding,
                        s.memory.embedding,
                    )
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score
                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim_to_selected

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def retrieve_by_id(
        self,
        memory_id: str,
        memories: list[MemoryProtocol],
    ) -> MemoryProtocol | None:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: ID of the memory to retrieve.
            memories: List of memories to search.

        Returns:
            The memory if found, None otherwise.
        """
        for memory in memories:
            if memory.id == memory_id:
                return memory
        return None

    def retrieve_related(
        self,
        memory: MemoryProtocol,
        memories: list[MemoryProtocol],
        top_k: int | None = None,
        current_time: datetime | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve memories related to a given memory.

        Uses the memory's embedding as the query.

        Args:
            memory: The reference memory.
            memories: Candidate memories to search.
            top_k: Number of results.
            current_time: Time for recency calculation.

        Returns:
            List of related memories (excluding the reference).
        """
        if not memory.embedding:
            return []

        # Exclude the reference memory
        candidates = [m for m in memories if m.id != memory.id]

        return self.retrieve(
            query_embedding=memory.embedding,
            memories=candidates,
            top_k=top_k,
            current_time=current_time,
        )

    def calculate_similarity_matrix(
        self,
        memories: list[MemoryProtocol],
    ) -> list[list[float]]:
        """
        Calculate pairwise similarity matrix for memories.

        Useful for clustering and consolidation.

        Args:
            memories: List of memories.

        Returns:
            NxN similarity matrix.
        """
        n = len(memories)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            matrix[i][i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                if memories[i].embedding and memories[j].embedding:
                    sim = self._cosine_similarity(
                        memories[i].embedding,
                        memories[j].embedding,
                    )
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix

    def find_clusters(
        self,
        memories: list[MemoryProtocol],
        similarity_threshold: float = 0.8,
    ) -> list[list[MemoryProtocol]]:
        """
        Find clusters of similar memories.

        Simple single-linkage clustering based on similarity threshold.

        Args:
            memories: Memories to cluster.
            similarity_threshold: Minimum similarity to be in same cluster.

        Returns:
            List of memory clusters.
        """
        if not memories:
            return []

        # Build adjacency based on similarity
        n = len(memories)
        visited = [False] * n
        clusters: list[list[MemoryProtocol]] = []

        similarity_matrix = self.calculate_similarity_matrix(memories)

        for i in range(n):
            if visited[i]:
                continue

            # BFS to find connected component
            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                cluster.append(memories[current])

                for j in range(n):
                    if not visited[j] and similarity_matrix[current][j] >= similarity_threshold:
                        visited[j] = True
                        queue.append(j)

            clusters.append(cluster)

        return clusters

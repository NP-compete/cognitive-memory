"""Tests for the retrieval engine."""

from datetime import datetime, timedelta, timezone

import pytest

from cognitive_memory.core.config import RetrievalConfig
from cognitive_memory.engines.retrieval import RetrievalEngine, RetrievalResult


class MockMemory:
    """Mock memory for testing retrieval engine."""

    def __init__(
        self,
        id: str = "test-id",
        embedding: list[float] | None = None,
        created_at: datetime | None = None,
        last_accessed_at: datetime | None = None,
        access_count: int = 0,
        strength: float = 1.0,
        importance: float = 0.5,
        is_archived: bool = False,
        is_pinned: bool = False,
    ):
        self.id = id
        self.embedding = embedding or [0.0] * 10
        self.created_at = created_at or datetime.now(timezone.utc)
        self.last_accessed_at = last_accessed_at or self.created_at
        self.access_count = access_count
        self.strength = strength
        self.importance = importance
        self.is_archived = is_archived
        self.is_pinned = is_pinned


def create_normalized_embedding(values: list[float]) -> list[float]:
    """Create a normalized embedding vector."""
    import math

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


class TestRetrievalEngine:
    """Tests for RetrievalEngine."""

    def test_default_values(self) -> None:
        """RetrievalEngine should have sensible defaults."""
        engine = RetrievalEngine()

        assert engine.similarity_weight == 0.4
        assert engine.strength_weight == 0.2
        assert engine.importance_weight == 0.2
        assert engine.recency_weight == 0.2
        assert engine.default_top_k == 10
        assert engine.mmr_lambda == 0.7

    def test_custom_values(self) -> None:
        """RetrievalEngine should accept custom values."""
        engine = RetrievalEngine(
            similarity_weight=0.5,
            default_top_k=20,
            mmr_lambda=0.5,
        )

        assert engine.similarity_weight == 0.5
        assert engine.default_top_k == 20
        assert engine.mmr_lambda == 0.5


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        engine = RetrievalEngine()
        vec = create_normalized_embedding([1.0, 0.0, 0.0])

        sim = engine._cosine_similarity(vec, vec)

        assert sim == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        engine = RetrievalEngine()
        vec_a = create_normalized_embedding([1.0, 0.0, 0.0])
        vec_b = create_normalized_embedding([0.0, 1.0, 0.0])

        sim = engine._cosine_similarity(vec_a, vec_b)

        assert sim == pytest.approx(0.0, abs=0.01)

    def test_similar_vectors(self) -> None:
        """Similar vectors should have high similarity."""
        engine = RetrievalEngine()
        vec_a = create_normalized_embedding([1.0, 0.1, 0.0])
        vec_b = create_normalized_embedding([1.0, 0.2, 0.0])

        sim = engine._cosine_similarity(vec_a, vec_b)

        assert sim > 0.9

    def test_empty_vectors(self) -> None:
        """Empty vectors should return 0.0."""
        engine = RetrievalEngine()

        sim = engine._cosine_similarity([], [])

        assert sim == 0.0

    def test_mismatched_lengths(self) -> None:
        """Mismatched vector lengths should return 0.0."""
        engine = RetrievalEngine()

        sim = engine._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

        assert sim == 0.0


class TestRetrieval:
    """Tests for memory retrieval."""

    def test_retrieve_returns_results(self) -> None:
        """retrieve should return scored results."""
        engine = RetrievalEngine()
        query = create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.9, 0.1, 0.0, 0.0, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 1.0, 0.0, 0.0, 0.0])),
        ]

        results = engine.retrieve(query, memories)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_retrieve_sorted_by_score(self) -> None:
        """Results should be sorted by final_score descending."""
        engine = RetrievalEngine()
        query = create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.5, 0.5, 0.0, 0.0, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 1.0, 0.0, 0.0, 0.0])),
        ]

        results = engine.retrieve(query, memories)

        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self) -> None:
        """retrieve should respect top_k limit."""
        engine = RetrievalEngine()
        query = create_normalized_embedding([1.0] * 10)
        memories = [
            MockMemory(id=str(i), embedding=create_normalized_embedding([1.0] * 10))
            for i in range(20)
        ]

        results = engine.retrieve(query, memories, top_k=5)

        assert len(results) == 5

    def test_retrieve_filters_archived(self) -> None:
        """Archived memories should be filtered by default."""
        engine = RetrievalEngine(include_archived=False)
        query = create_normalized_embedding([1.0] * 10)
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0] * 10)),
            MockMemory(id="2", embedding=create_normalized_embedding([1.0] * 10), is_archived=True),
        ]

        results = engine.retrieve(query, memories)

        assert len(results) == 1
        assert results[0].memory.id == "1"

    def test_retrieve_includes_archived_when_configured(self) -> None:
        """Archived memories should be included when configured."""
        engine = RetrievalEngine(include_archived=True)
        query = create_normalized_embedding([1.0] * 10)
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0] * 10)),
            MockMemory(id="2", embedding=create_normalized_embedding([1.0] * 10), is_archived=True),
        ]

        results = engine.retrieve(query, memories)

        assert len(results) == 2

    def test_retrieve_filters_by_similarity_threshold(self) -> None:
        """Results below similarity threshold should be filtered."""
        engine = RetrievalEngine(min_similarity_threshold=0.5)
        query = create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.0, 1.0, 0.0, 0.0, 0.0])),
        ]

        results = engine.retrieve(query, memories)

        assert len(results) == 1
        assert results[0].memory.id == "1"

    def test_retrieve_empty_memories(self) -> None:
        """Empty memory list should return empty results."""
        engine = RetrievalEngine()
        query = create_normalized_embedding([1.0] * 10)

        results = engine.retrieve(query, [])

        assert results == []


class TestRecencyScore:
    """Tests for recency scoring."""

    def test_recent_memory_high_recency(self) -> None:
        """Recently accessed memory should have high recency score."""
        engine = RetrievalEngine()
        now = datetime.now(timezone.utc)
        query = create_normalized_embedding([1.0] * 10)
        memories = [
            MockMemory(
                id="1",
                embedding=create_normalized_embedding([1.0] * 10),
                last_accessed_at=now,
            ),
        ]

        results = engine.retrieve(query, memories, current_time=now)

        assert results[0].recency_score == pytest.approx(1.0, abs=0.01)

    def test_old_memory_low_recency(self) -> None:
        """Old memory should have lower recency score."""
        engine = RetrievalEngine(recency_half_life_hours=24.0)
        now = datetime.now(timezone.utc)
        query = create_normalized_embedding([1.0] * 10)
        memories = [
            MockMemory(
                id="1",
                embedding=create_normalized_embedding([1.0] * 10),
                last_accessed_at=now - timedelta(hours=48),
            ),
        ]

        results = engine.retrieve(query, memories, current_time=now)

        # After 2 half-lives: 0.5^2 = 0.25
        assert results[0].recency_score == pytest.approx(0.25, abs=0.05)


class TestMMR:
    """Tests for Maximal Marginal Relevance."""

    def test_mmr_increases_diversity(self) -> None:
        """MMR should select diverse results."""
        engine = RetrievalEngine(mmr_lambda=0.5, min_similarity_threshold=0.0)
        query = create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])

        # Create memories: two similar to query, one different
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.99, 0.01, 0.0, 0.0, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 1.0, 0.0, 0.0, 0.0])),
        ]

        _results_no_mmr = engine.retrieve(query, memories, top_k=2, use_mmr=False)
        results_mmr = engine.retrieve(query, memories, top_k=2, use_mmr=True)

        # With MMR: should prefer diversity (1 and 3)
        mmr_ids = {r.memory.id for r in results_mmr}

        # MMR should include the diverse option
        assert "1" in mmr_ids
        assert "3" in mmr_ids or len(mmr_ids) == 2


class TestRetrieveById:
    """Tests for retrieve_by_id."""

    def test_retrieve_existing_memory(self) -> None:
        """Should return memory when ID exists."""
        engine = RetrievalEngine()
        memories = [
            MockMemory(id="1"),
            MockMemory(id="2"),
            MockMemory(id="3"),
        ]

        result = engine.retrieve_by_id("2", memories)

        assert result is not None
        assert result.id == "2"

    def test_retrieve_nonexistent_memory(self) -> None:
        """Should return None when ID doesn't exist."""
        engine = RetrievalEngine()
        memories = [MockMemory(id="1")]

        result = engine.retrieve_by_id("999", memories)

        assert result is None


class TestRetrieveRelated:
    """Tests for retrieve_related."""

    def test_retrieve_related_excludes_self(self) -> None:
        """Related retrieval should exclude the reference memory."""
        engine = RetrievalEngine()
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0] * 10)),
            MockMemory(id="2", embedding=create_normalized_embedding([1.0] * 10)),
            MockMemory(id="3", embedding=create_normalized_embedding([1.0] * 10)),
        ]

        results = engine.retrieve_related(memories[0], memories)

        result_ids = {r.memory.id for r in results}
        assert "1" not in result_ids

    def test_retrieve_related_finds_similar(self) -> None:
        """Related retrieval should find similar memories."""
        engine = RetrievalEngine()
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.9, 0.1, 0.0, 0.0, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 1.0, 0.0, 0.0, 0.0])),
        ]

        results = engine.retrieve_related(memories[0], memories, top_k=1)

        assert len(results) == 1
        assert results[0].memory.id == "2"


class TestSimilarityMatrix:
    """Tests for similarity matrix calculation."""

    def test_matrix_dimensions(self) -> None:
        """Matrix should be NxN."""
        engine = RetrievalEngine()
        memories = [MockMemory(id=str(i)) for i in range(5)]

        matrix = engine.calculate_similarity_matrix(memories)

        assert len(matrix) == 5
        assert all(len(row) == 5 for row in matrix)

    def test_matrix_diagonal(self) -> None:
        """Diagonal should be 1.0 (self-similarity)."""
        engine = RetrievalEngine()
        memories = [MockMemory(id=str(i)) for i in range(3)]

        matrix = engine.calculate_similarity_matrix(memories)

        for i in range(3):
            assert matrix[i][i] == 1.0

    def test_matrix_symmetric(self) -> None:
        """Matrix should be symmetric."""
        engine = RetrievalEngine()
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.5, 0.5, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 0.0, 1.0])),
        ]

        matrix = engine.calculate_similarity_matrix(memories)

        for i in range(3):
            for j in range(3):
                assert matrix[i][j] == pytest.approx(matrix[j][i], abs=0.001)


class TestFindClusters:
    """Tests for cluster finding."""

    def test_find_clusters_groups_similar(self) -> None:
        """Similar memories should be in same cluster."""
        engine = RetrievalEngine()
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.99, 0.01, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 0.0, 1.0])),
        ]

        clusters = engine.find_clusters(memories, similarity_threshold=0.9)

        # Should have 2 clusters: (1, 2) and (3)
        assert len(clusters) == 2

    def test_find_clusters_empty_input(self) -> None:
        """Empty input should return empty clusters."""
        engine = RetrievalEngine()

        clusters = engine.find_clusters([])

        assert clusters == []

    def test_find_clusters_all_different(self) -> None:
        """Dissimilar memories should be in separate clusters."""
        engine = RetrievalEngine()
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0])),
            MockMemory(id="2", embedding=create_normalized_embedding([0.0, 1.0, 0.0])),
            MockMemory(id="3", embedding=create_normalized_embedding([0.0, 0.0, 1.0])),
        ]

        clusters = engine.find_clusters(memories, similarity_threshold=0.9)

        assert len(clusters) == 3


class TestFromConfig:
    """Tests for from_config class method."""

    def test_from_config_creates_engine(self) -> None:
        """from_config should create engine from RetrievalConfig."""
        config = RetrievalConfig(
            similarity_weight=0.5,
            strength_weight=0.2,
            importance_weight=0.2,
            recency_weight=0.1,
            default_top_k=5,
            max_top_k=50,
            mmr_lambda=0.8,
            min_similarity_threshold=0.4,
            include_archived=True,
        )

        engine = RetrievalEngine.from_config(config)

        assert engine.similarity_weight == 0.5
        assert engine.default_top_k == 5
        assert engine.include_archived is True
        assert engine.mmr_lambda == 0.8


class TestRetrievalEdgeCases:
    """Tests for edge cases in retrieval."""

    def test_empty_after_threshold_filter(self) -> None:
        """Retrieve should return empty when all below threshold."""
        engine = RetrievalEngine(min_similarity_threshold=0.99)
        query = create_normalized_embedding([1.0, 0.0, 0.0])
        memories = [
            MockMemory(id="1", embedding=create_normalized_embedding([0.0, 1.0, 0.0])),
        ]

        results = engine.retrieve(query, memories)

        assert results == []

    def test_memories_without_embeddings_filtered(self) -> None:
        """Memories with empty embeddings should be filtered out."""
        engine = RetrievalEngine()
        query = create_normalized_embedding([1.0, 0.0, 0.0])

        mem_with = MockMemory(id="1", embedding=create_normalized_embedding([1.0, 0.0, 0.0]))
        mem_without = MockMemory(id="2", embedding=[])

        results = engine.retrieve(query, [mem_with, mem_without])

        assert len(results) == 1
        assert results[0].memory.id == "1"

    def test_retrieve_related_empty_embedding(self) -> None:
        """retrieve_related with empty embedding returns empty."""
        engine = RetrievalEngine()
        ref = MockMemory(id="ref", embedding=[])
        others = [MockMemory(id="1")]

        results = engine.retrieve_related(ref, others)

        assert results == []


class TestNaiveDatetimeInRetrieval:
    """Tests for naive datetime handling in retrieval."""

    def test_naive_last_accessed_at(self) -> None:
        """Should handle naive last_accessed_at."""
        engine = RetrievalEngine()
        naive_time = datetime(2025, 1, 1, 0, 0, 0)
        now = datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

        query = create_normalized_embedding([1.0, 0.0, 0.0])
        memory = MockMemory(
            id="1",
            embedding=create_normalized_embedding([1.0, 0.0, 0.0]),
            last_accessed_at=naive_time,
        )

        results = engine.retrieve(query, [memory], current_time=now)

        assert len(results) == 1
        assert results[0].recency_score > 0

    def test_naive_current_time(self) -> None:
        """Should handle naive current_time."""
        engine = RetrievalEngine()
        created = datetime(2025, 1, 1, tzinfo=timezone.utc)
        naive_now = datetime(2025, 1, 2)

        query = create_normalized_embedding([1.0, 0.0, 0.0])
        memory = MockMemory(
            id="1",
            embedding=create_normalized_embedding([1.0, 0.0, 0.0]),
            created_at=created,
            last_accessed_at=created,
        )

        results = engine.retrieve(query, [memory], current_time=naive_now)

        assert len(results) == 1


class TestMMREdgeCases:
    """Tests for MMR edge cases."""

    def test_mmr_with_empty_scored(self) -> None:
        """MMR with no candidates should return empty."""
        engine = RetrievalEngine(min_similarity_threshold=0.0)
        query = create_normalized_embedding([1.0, 0.0, 0.0])

        results = engine.retrieve(query, [], use_mmr=True)

        assert results == []

    def test_mmr_selects_diverse_results(self) -> None:
        """MMR should select diverse results over purely relevant ones."""
        engine = RetrievalEngine(
            mmr_lambda=0.3,
            min_similarity_threshold=0.0,
        )
        query = create_normalized_embedding([1.0, 0.0, 0.0])

        cluster_a1 = MockMemory(id="a1", embedding=create_normalized_embedding([1.0, 0.0, 0.0]))
        cluster_a2 = MockMemory(id="a2", embedding=create_normalized_embedding([0.99, 0.01, 0.0]))
        diverse = MockMemory(id="b1", embedding=create_normalized_embedding([0.5, 0.5, 0.0]))

        results = engine.retrieve(query, [cluster_a1, cluster_a2, diverse], top_k=2, use_mmr=True)

        ids = {r.memory.id for r in results}
        assert "b1" in ids

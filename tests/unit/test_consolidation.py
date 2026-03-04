"""Tests for memory consolidation engine."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from cognitive_memory.engines.consolidation import (
    ConsolidationCandidate,
    ConsolidationEngine,
    ConsolidationResult,
)


@dataclass
class MockMemory:
    """Mock memory for testing."""

    id: str
    memory_type: str = "episodic"
    content: str = "test content"
    embedding: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(hours=48)
    )
    access_count: int = 5
    importance: float = 0.7
    strength: float = 0.8
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    is_consolidated: bool = False
    agent_id: str | None = "agent-1"


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    min_memories: int = 3
    similarity_threshold: float = 0.75
    min_access_count: int = 2
    min_age_hours: float = 24.0
    max_cluster_size: int = 10
    preserve_source_memories: bool = True


class TestConsolidationEngine:
    """Tests for ConsolidationEngine."""

    def test_default_values(self) -> None:
        """Engine should have sensible defaults."""
        engine = ConsolidationEngine()

        assert engine.min_memories == 3
        assert engine.similarity_threshold == 0.75
        assert engine.min_access_count == 2
        assert engine.min_age_hours == 24.0
        assert engine.max_cluster_size == 10
        assert engine.preserve_source_memories is True

    def test_from_config(self) -> None:
        """Engine should be creatable from config."""
        config = MockConfig(
            min_memories=5,
            similarity_threshold=0.8,
            min_access_count=3,
        )

        engine = ConsolidationEngine.from_config(config)

        assert engine.min_memories == 5
        assert engine.similarity_threshold == 0.8
        assert engine.min_access_count == 3


class TestFindConsolidationCandidates:
    """Tests for finding consolidation candidates."""

    def test_empty_memories(self) -> None:
        """Empty list should return no candidates."""
        engine = ConsolidationEngine()

        candidates = engine.find_consolidation_candidates([])

        assert candidates == []

    def test_insufficient_memories(self) -> None:
        """Too few memories should return no candidates."""
        engine = ConsolidationEngine(min_memories=3)
        memories = [MockMemory(id="m1"), MockMemory(id="m2")]

        candidates = engine.find_consolidation_candidates(memories)

        assert candidates == []

    def test_filters_already_consolidated(self) -> None:
        """Already consolidated memories should be excluded."""
        engine = ConsolidationEngine(min_memories=2)
        memories = [
            MockMemory(id="m1", is_consolidated=True),
            MockMemory(id="m2", is_consolidated=True),
            MockMemory(id="m3"),
        ]

        candidates = engine.find_consolidation_candidates(memories)

        assert candidates == []

    def test_filters_non_episodic(self) -> None:
        """Non-episodic memories should be excluded."""
        engine = ConsolidationEngine(min_memories=2)
        memories = [
            MockMemory(id="m1", memory_type="semantic"),
            MockMemory(id="m2", memory_type="procedural"),
            MockMemory(id="m3"),
        ]

        candidates = engine.find_consolidation_candidates(memories)

        assert candidates == []

    def test_filters_low_access_count(self) -> None:
        """Memories with low access count should be excluded."""
        engine = ConsolidationEngine(min_memories=2, min_access_count=5)
        memories = [
            MockMemory(id="m1", access_count=1),
            MockMemory(id="m2", access_count=2),
            MockMemory(id="m3", access_count=10),
        ]

        candidates = engine.find_consolidation_candidates(memories)

        assert candidates == []

    def test_filters_too_recent(self) -> None:
        """Too recent memories should be excluded."""
        engine = ConsolidationEngine(min_memories=2, min_age_hours=24)
        now = datetime.now(timezone.utc)
        memories = [
            MockMemory(id="m1", created_at=now - timedelta(hours=1)),
            MockMemory(id="m2", created_at=now - timedelta(hours=2)),
            MockMemory(id="m3", created_at=now - timedelta(hours=48)),
        ]

        candidates = engine.find_consolidation_candidates(memories)

        assert candidates == []

    def test_finds_similar_cluster(self) -> None:
        """Similar memories should form a cluster."""
        engine = ConsolidationEngine(min_memories=2, similarity_threshold=0.9)
        memories = [
            MockMemory(id="m1", embedding=[1.0, 0.0, 0.0]),
            MockMemory(id="m2", embedding=[0.99, 0.01, 0.0]),
            MockMemory(id="m3", embedding=[0.98, 0.02, 0.0]),
        ]

        candidates = engine.find_consolidation_candidates(memories)

        assert len(candidates) == 1
        assert len(candidates[0].memories) == 3


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        engine = ConsolidationEngine()

        similarity = engine._cosine_similarity([1.0, 0.0], [1.0, 0.0])

        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        engine = ConsolidationEngine()

        similarity = engine._cosine_similarity([1.0, 0.0], [0.0, 1.0])

        assert similarity == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        engine = ConsolidationEngine()

        similarity = engine._cosine_similarity([1.0, 0.0], [-1.0, 0.0])

        assert similarity == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        """Empty vectors should return 0.0."""
        engine = ConsolidationEngine()

        assert engine._cosine_similarity([], []) == 0.0
        assert engine._cosine_similarity([1.0], []) == 0.0

    def test_different_length_vectors(self) -> None:
        """Different length vectors should return 0.0."""
        engine = ConsolidationEngine()

        similarity = engine._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

        assert similarity == 0.0


class TestCalculateCentroid:
    """Tests for centroid calculation."""

    def test_single_embedding(self) -> None:
        """Single embedding should be its own centroid."""
        engine = ConsolidationEngine()

        centroid = engine._calculate_centroid([[1.0, 2.0, 3.0]])

        assert centroid == [1.0, 2.0, 3.0]

    def test_multiple_embeddings(self) -> None:
        """Multiple embeddings should average to centroid."""
        engine = ConsolidationEngine()

        centroid = engine._calculate_centroid([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        assert centroid == [0.5, 0.5]

    def test_empty_embeddings(self) -> None:
        """Empty list should return empty centroid."""
        engine = ConsolidationEngine()

        centroid = engine._calculate_centroid([])

        assert centroid == []


class TestFindSharedItems:
    """Tests for finding shared items."""

    def test_no_shared_items(self) -> None:
        """No shared items should return empty list."""
        engine = ConsolidationEngine()

        shared = engine._find_shared_items([["a"], ["b"], ["c"]])

        assert shared == []

    def test_all_shared(self) -> None:
        """Items in all lists should be returned."""
        engine = ConsolidationEngine()

        shared = engine._find_shared_items([
            ["a", "b"],
            ["a", "c"],
            ["a", "d"],
        ])

        assert "a" in shared

    def test_empty_lists(self) -> None:
        """Empty lists should return empty."""
        engine = ConsolidationEngine()

        assert engine._find_shared_items([]) == []
        assert engine._find_shared_items([[], []]) == []


class TestConsolidate:
    """Tests for consolidation operation."""

    def test_creates_result(self) -> None:
        """Consolidate should create a result."""
        engine = ConsolidationEngine()
        memories = [
            MockMemory(id="m1", content="content 1"),
            MockMemory(id="m2", content="content 2"),
            MockMemory(id="m3", content="content 3"),
        ]
        candidate = ConsolidationCandidate(
            memories=memories,
            centroid=[0.5, 0.5, 0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=["Alice"],
            shared_topics=["work"],
        )

        result = engine.consolidate(candidate)

        assert isinstance(result, ConsolidationResult)
        assert result.memory_type == "semantic"
        assert len(result.source_memory_ids) == 3
        assert result.combined_importance == 0.7
        assert result.shared_entities == ["Alice"]

    def test_increments_count(self) -> None:
        """Consolidate should increment counter."""
        engine = ConsolidationEngine()
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1")],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        engine.consolidate(candidate)
        engine.consolidate(candidate)

        stats = engine.get_consolidation_stats()
        assert stats["total_consolidations"] == 2

    def test_custom_content_generator(self) -> None:
        """Custom content generator should be used."""
        engine = ConsolidationEngine()
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1", content="original")],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        def custom_generator(_candidate: ConsolidationCandidate) -> str:
            return "custom summary"

        result = engine.consolidate(candidate, content_generator=custom_generator)

        assert result.content_summary == "custom summary"


class TestShouldConsolidate:
    """Tests for consolidation decision."""

    def test_meets_all_criteria(self) -> None:
        """Should return True when all criteria met."""
        engine = ConsolidationEngine(min_memories=2)
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1"), MockMemory(id="m2"), MockMemory(id="m3")],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        assert engine.should_consolidate(candidate) is True

    def test_too_few_memories(self) -> None:
        """Should return False with too few memories."""
        engine = ConsolidationEngine(min_memories=5)
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1"), MockMemory(id="m2")],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        assert engine.should_consolidate(candidate) is False

    def test_low_similarity(self) -> None:
        """Should return False with low similarity."""
        engine = ConsolidationEngine(min_memories=2, similarity_threshold=0.9)
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1"), MockMemory(id="m2"), MockMemory(id="m3")],
            centroid=[0.5],
            similarity_score=0.5,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        assert engine.should_consolidate(candidate) is False

    def test_low_importance(self) -> None:
        """Should return False with low importance."""
        engine = ConsolidationEngine(min_memories=2)
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1"), MockMemory(id="m2"), MockMemory(id="m3")],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.1,
            shared_entities=[],
            shared_topics=[],
        )

        assert engine.should_consolidate(candidate, min_importance=0.5) is False


class TestDefaultContentSummary:
    """Tests for default content summary generation."""

    def test_single_content(self) -> None:
        """Single content should be returned as-is."""
        engine = ConsolidationEngine()
        candidate = ConsolidationCandidate(
            memories=[MockMemory(id="m1", content="only content")],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        summary = engine._default_content_summary(candidate)

        assert summary == "only content"

    def test_multiple_contents(self) -> None:
        """Multiple contents should be joined."""
        engine = ConsolidationEngine()
        candidate = ConsolidationCandidate(
            memories=[
                MockMemory(id="m1", content="content 1"),
                MockMemory(id="m2", content="content 2"),
            ],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        summary = engine._default_content_summary(candidate)

        assert "content 1" in summary
        assert "content 2" in summary

    def test_deduplicates_content(self) -> None:
        """Duplicate contents should be deduplicated."""
        engine = ConsolidationEngine()
        candidate = ConsolidationCandidate(
            memories=[
                MockMemory(id="m1", content="same"),
                MockMemory(id="m2", content="same"),
            ],
            centroid=[0.5],
            similarity_score=0.9,
            combined_importance=0.7,
            shared_entities=[],
            shared_topics=[],
        )

        summary = engine._default_content_summary(candidate)

        assert summary == "same"

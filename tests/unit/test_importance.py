"""Tests for the importance engine."""

from datetime import datetime, timedelta, timezone

import pytest

from cognitive_memory.engines.importance import ImportanceEngine, ImportanceResult


class MockSource:
    """Mock source enum for testing."""

    def __init__(self, value: str):
        self.value = value


class MockMemory:
    """Mock memory for testing importance engine."""

    def __init__(
        self,
        created_at: datetime | None = None,
        last_accessed_at: datetime | None = None,
        access_count: int = 0,
        emotional_valence: float = 0.0,
        surprise_score: float = 0.0,
        source: str | MockSource = "conversation",
        entities: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.created_at = created_at or datetime.now(timezone.utc)
        self.last_accessed_at = last_accessed_at or self.created_at
        self.access_count = access_count
        self.emotional_valence = emotional_valence
        self.surprise_score = surprise_score
        self.source = source
        self.entities = entities or []
        self.metadata = metadata or {}


class TestImportanceEngine:
    """Tests for ImportanceEngine."""

    def test_default_values(self) -> None:
        """ImportanceEngine should have sensible defaults."""
        engine = ImportanceEngine()

        assert engine.recency_weight == 0.2
        assert engine.frequency_weight == 0.15
        assert engine.emotional_weight == 0.2
        assert engine.surprise_weight == 0.15
        assert engine.entity_weight == 0.1
        assert engine.explicit_weight == 0.2

    def test_custom_values(self) -> None:
        """ImportanceEngine should accept custom values."""
        engine = ImportanceEngine(
            recency_weight=0.3,
            frequency_weight=0.2,
            recency_half_life_hours=48.0,
        )

        assert engine.recency_weight == 0.3
        assert engine.frequency_weight == 0.2
        assert engine.recency_half_life_hours == 48.0


class TestRecencyScore:
    """Tests for recency scoring."""

    def test_recent_memory_high_score(self) -> None:
        """Recently created memory should have high recency score."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)
        memory = MockMemory(created_at=now)

        result = engine.calculate_importance(memory, now)

        assert result.recency_score == pytest.approx(1.0, abs=0.01)

    def test_old_memory_low_score(self) -> None:
        """Old memory should have lower recency score."""
        engine = ImportanceEngine(recency_half_life_hours=24.0)
        now = datetime.now(timezone.utc)
        memory = MockMemory(created_at=now - timedelta(hours=48))

        result = engine.calculate_importance(memory, now)

        # After 2 half-lives: 0.5^2 = 0.25
        assert result.recency_score == pytest.approx(0.25, abs=0.05)

    def test_half_life_decay(self) -> None:
        """Score should halve after one half-life."""
        engine = ImportanceEngine(recency_half_life_hours=24.0)
        now = datetime.now(timezone.utc)
        memory = MockMemory(created_at=now - timedelta(hours=24))

        result = engine.calculate_importance(memory, now)

        assert result.recency_score == pytest.approx(0.5, abs=0.01)


class TestFrequencyScore:
    """Tests for frequency scoring."""

    def test_zero_access_zero_score(self) -> None:
        """Memory with no accesses should have zero frequency score."""
        engine = ImportanceEngine()
        memory = MockMemory(access_count=0)

        result = engine.calculate_importance(memory)

        assert result.frequency_score == 0.0

    def test_high_access_high_score(self) -> None:
        """Frequently accessed memory should have high frequency score."""
        engine = ImportanceEngine(frequency_saturation=10)
        memory = MockMemory(access_count=10)

        result = engine.calculate_importance(memory)

        assert result.frequency_score == pytest.approx(1.0, abs=0.01)

    def test_frequency_saturates(self) -> None:
        """Frequency score should saturate at 1.0."""
        engine = ImportanceEngine(frequency_saturation=10)
        memory = MockMemory(access_count=100)

        result = engine.calculate_importance(memory)

        assert result.frequency_score == 1.0


class TestEmotionalScore:
    """Tests for emotional scoring."""

    def test_neutral_emotion_zero_score(self) -> None:
        """Neutral emotional valence should give zero score."""
        engine = ImportanceEngine()
        memory = MockMemory(emotional_valence=0.0)

        result = engine.calculate_importance(memory)

        assert result.emotional_score == 0.0

    def test_positive_emotion_high_score(self) -> None:
        """Strong positive emotion should give high score."""
        engine = ImportanceEngine()
        memory = MockMemory(emotional_valence=0.8)

        result = engine.calculate_importance(memory)

        assert result.emotional_score == 0.8

    def test_negative_emotion_high_score(self) -> None:
        """Strong negative emotion should also give high score."""
        engine = ImportanceEngine()
        memory = MockMemory(emotional_valence=-0.9)

        result = engine.calculate_importance(memory)

        assert result.emotional_score == 0.9


class TestSurpriseScore:
    """Tests for surprise scoring."""

    def test_no_surprise_zero_score(self) -> None:
        """No surprise should give zero score."""
        engine = ImportanceEngine()
        memory = MockMemory(surprise_score=0.0)

        result = engine.calculate_importance(memory)

        assert result.surprise_score == 0.0

    def test_high_surprise_high_score(self) -> None:
        """High surprise should give high score."""
        engine = ImportanceEngine()
        memory = MockMemory(surprise_score=0.9)

        result = engine.calculate_importance(memory)

        assert result.surprise_score == 0.9

    def test_surprise_clamped(self) -> None:
        """Surprise score should be clamped to [0, 1]."""
        engine = ImportanceEngine()
        memory = MockMemory(surprise_score=1.5)

        result = engine.calculate_importance(memory)

        assert result.surprise_score == 1.0


class TestEntityScore:
    """Tests for entity scoring."""

    def test_no_entities_zero_score(self) -> None:
        """Memory with no entities should have zero entity score."""
        engine = ImportanceEngine()
        memory = MockMemory(entities=[])

        result = engine.calculate_importance(memory)

        assert result.entity_score == 0.0

    def test_many_entities_high_score(self) -> None:
        """Memory with many entities should have high entity score."""
        engine = ImportanceEngine(entity_saturation=5)
        memory = MockMemory(entities=["Alice", "Bob", "Acme", "NYC", "Python"])

        result = engine.calculate_importance(memory)

        assert result.entity_score == 1.0

    def test_entity_saturates(self) -> None:
        """Entity score should saturate at 1.0."""
        engine = ImportanceEngine(entity_saturation=5)
        memory = MockMemory(entities=["A", "B", "C", "D", "E", "F", "G", "H"])

        result = engine.calculate_importance(memory)

        assert result.entity_score == 1.0


class TestExplicitScore:
    """Tests for explicit importance scoring."""

    def test_default_explicit_score(self) -> None:
        """Default explicit score should be 0.5."""
        engine = ImportanceEngine()
        memory = MockMemory()

        result = engine.calculate_importance(memory)

        assert result.explicit_score == 0.5

    def test_explicit_importance_parameter(self) -> None:
        """Explicit importance parameter should override default."""
        engine = ImportanceEngine()
        memory = MockMemory()

        result = engine.calculate_importance(memory, explicit_importance=0.9)

        assert result.explicit_score == 0.9

    def test_metadata_importance(self) -> None:
        """Importance from metadata should be used."""
        engine = ImportanceEngine()
        memory = MockMemory(metadata={"importance": 0.7})

        result = engine.calculate_importance(memory)

        assert result.explicit_score == 0.7


class TestSourceMultiplier:
    """Tests for source multiplier."""

    def test_user_explicit_high_multiplier(self) -> None:
        """User explicit source should have high multiplier."""
        engine = ImportanceEngine()
        memory = MockMemory(source="user_explicit")

        result = engine.calculate_importance(memory)

        assert result.source_multiplier == 1.5

    def test_conversation_low_multiplier(self) -> None:
        """Conversation source should have lower multiplier."""
        engine = ImportanceEngine()
        memory = MockMemory(source="conversation")

        result = engine.calculate_importance(memory)

        assert result.source_multiplier == 0.8

    def test_enum_source_value(self) -> None:
        """Source as enum should use .value."""
        engine = ImportanceEngine()
        memory = MockMemory(source=MockSource("tool_result"))

        result = engine.calculate_importance(memory)

        assert result.source_multiplier == 1.2

    def test_unknown_source_default(self) -> None:
        """Unknown source should use default multiplier of 1.0."""
        engine = ImportanceEngine()
        memory = MockMemory(source="unknown_source")

        result = engine.calculate_importance(memory)

        assert result.source_multiplier == 1.0


class TestImportanceResult:
    """Tests for ImportanceResult structure."""

    def test_result_structure(self) -> None:
        """ImportanceResult should contain all expected fields."""
        engine = ImportanceEngine()
        memory = MockMemory()

        result = engine.calculate_importance(memory)

        assert isinstance(result, ImportanceResult)
        assert 0 <= result.final_score <= 1
        assert "recency" in result.breakdown
        assert "frequency" in result.breakdown
        assert "emotional" in result.breakdown

    def test_final_score_clamped(self) -> None:
        """Final score should be clamped to [0, 1]."""
        engine = ImportanceEngine()
        # High multiplier source with high scores
        memory = MockMemory(
            source="user_explicit",
            access_count=100,
            emotional_valence=1.0,
            surprise_score=1.0,
            entities=["A", "B", "C", "D", "E", "F"],
            metadata={"importance": 1.0},
        )

        result = engine.calculate_importance(memory)

        assert result.final_score <= 1.0


class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_get_importance(self) -> None:
        """get_importance should return just the score."""
        engine = ImportanceEngine()
        memory = MockMemory()

        score = engine.get_importance(memory)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_batch_calculate_importance(self) -> None:
        """batch_calculate_importance should process multiple memories."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)
        memories = [
            MockMemory(created_at=now - timedelta(hours=i)) for i in range(5)
        ]

        results = engine.batch_calculate_importance(memories, now)

        assert len(results) == 5
        for _memory, result in results:
            assert isinstance(result, ImportanceResult)

    def test_rank_by_importance(self) -> None:
        """rank_by_importance should sort by score descending."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)
        memories = [
            MockMemory(created_at=now - timedelta(hours=i * 10)) for i in range(5)
        ]

        ranked = engine.rank_by_importance(memories, now)

        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_by_importance_top_k(self) -> None:
        """rank_by_importance should respect top_k limit."""
        engine = ImportanceEngine()
        memories = [MockMemory() for _ in range(10)]

        ranked = engine.rank_by_importance(memories, top_k=3)

        assert len(ranked) == 3

    def test_filter_by_importance(self) -> None:
        """filter_by_importance should filter by range."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)
        memories = [
            MockMemory(
                created_at=now - timedelta(hours=i * 24),
                access_count=i,
            )
            for i in range(10)
        ]

        filtered = engine.filter_by_importance(memories, 0.3, 0.7, now)

        for m in filtered:
            score = engine.get_importance(m, now)
            assert 0.3 <= score <= 0.7

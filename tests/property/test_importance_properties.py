"""Property-based tests for ImportanceEngine using Hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_memory.engines.importance import ImportanceEngine


@dataclass
class FakeMemory:
    """Minimal memory for property tests."""

    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    emotional_valence: float
    surprise_score: float
    source: str
    entities: list[str]
    metadata: dict[str, Any]


reasonable_valence = st.floats(min_value=-1.0, max_value=1.0)
reasonable_surprise = st.floats(min_value=0.0, max_value=1.0)
reasonable_access = st.integers(min_value=0, max_value=500)
entity_list = st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20)
source_strategy = st.sampled_from(
    ["user_explicit", "tool_result", "observation", "conversation", "external", "unknown"]
)


def _make_memory(
    hours_ago: float = 1.0,
    access_count: int = 0,
    emotional_valence: float = 0.0,
    surprise_score: float = 0.0,
    source: str = "conversation",
    entities: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FakeMemory:
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=hours_ago)
    return FakeMemory(
        created_at=created,
        last_accessed_at=created,
        access_count=access_count,
        emotional_valence=emotional_valence,
        surprise_score=surprise_score,
        source=source,
        entities=entities or [],
        metadata=metadata or {},
    )


class TestImportanceBounds:
    """Importance score is always bounded to [0, 1]."""

    @given(
        hours_ago=st.floats(min_value=0.01, max_value=8760.0),
        access_count=reasonable_access,
        emotional_valence=reasonable_valence,
        surprise_score=reasonable_surprise,
        source=source_strategy,
        entities=entity_list,
    )
    @settings(max_examples=300)
    def test_final_score_always_in_unit_interval(
        self,
        hours_ago: float,
        access_count: int,
        emotional_valence: float,
        surprise_score: float,
        source: str,
        entities: list[str],
    ) -> None:
        """Final importance score is always in [0, 1]."""
        engine = ImportanceEngine()
        mem = _make_memory(
            hours_ago=hours_ago,
            access_count=access_count,
            emotional_valence=emotional_valence,
            surprise_score=surprise_score,
            source=source,
            entities=entities,
        )
        now = datetime.now(timezone.utc)

        result = engine.calculate_importance(mem, now)

        assert 0.0 <= result.final_score <= 1.0

    @given(
        hours_ago=st.floats(min_value=0.01, max_value=8760.0),
        access_count=reasonable_access,
        emotional_valence=reasonable_valence,
        surprise_score=reasonable_surprise,
    )
    @settings(max_examples=200)
    def test_component_scores_in_unit_interval(
        self,
        hours_ago: float,
        access_count: int,
        emotional_valence: float,
        surprise_score: float,
    ) -> None:
        """All component scores are in [0, 1]."""
        engine = ImportanceEngine()
        mem = _make_memory(
            hours_ago=hours_ago,
            access_count=access_count,
            emotional_valence=emotional_valence,
            surprise_score=surprise_score,
        )
        now = datetime.now(timezone.utc)

        result = engine.calculate_importance(mem, now)

        assert 0.0 <= result.recency_score <= 1.0
        assert 0.0 <= result.frequency_score <= 1.0
        assert 0.0 <= result.emotional_score <= 1.0
        assert 0.0 <= result.surprise_score <= 1.0
        assert 0.0 <= result.entity_score <= 1.0
        assert 0.0 <= result.explicit_score <= 1.0


class TestImportanceMonotonicity:
    """Individual factors are monotonic w.r.t. their inputs."""

    @given(
        count_low=st.integers(min_value=0, max_value=50),
        count_high=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=100)
    def test_more_access_higher_frequency_score(self, count_low: int, count_high: int) -> None:
        """Higher access count -> higher or equal frequency score."""
        if count_low > count_high:
            count_low, count_high = count_high, count_low

        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)

        mem_low = _make_memory(access_count=count_low)
        mem_high = _make_memory(access_count=count_high)

        r_low = engine.calculate_importance(mem_low, now)
        r_high = engine.calculate_importance(mem_high, now)

        assert r_high.frequency_score >= r_low.frequency_score - 1e-9

    @given(
        valence_low=st.floats(min_value=0.0, max_value=1.0),
        valence_high=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_stronger_emotion_higher_emotional_score(
        self, valence_low: float, valence_high: float
    ) -> None:
        """Higher |emotional_valence| -> higher emotional score."""
        if abs(valence_low) > abs(valence_high):
            valence_low, valence_high = valence_high, valence_low

        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)

        mem_low = _make_memory(emotional_valence=valence_low)
        mem_high = _make_memory(emotional_valence=valence_high)

        r_low = engine.calculate_importance(mem_low, now)
        r_high = engine.calculate_importance(mem_high, now)

        assert r_high.emotional_score >= r_low.emotional_score - 1e-9


class TestRecencyDecay:
    """Recency score decays with time."""

    @given(
        hours_early=st.floats(min_value=0.01, max_value=500.0),
        hours_late=st.floats(min_value=0.01, max_value=500.0),
    )
    @settings(max_examples=100)
    def test_more_recent_higher_recency(self, hours_early: float, hours_late: float) -> None:
        """More recent memory has higher recency score."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)

        mem_recent = _make_memory(hours_ago=hours_early)
        mem_old = _make_memory(hours_ago=hours_early + hours_late)

        r_recent = engine.calculate_importance(mem_recent, now)
        r_old = engine.calculate_importance(mem_old, now)

        assert r_recent.recency_score >= r_old.recency_score - 1e-9


class TestSourceMultiplierPositive:
    """Source multiplier is always positive."""

    @given(source=source_strategy)
    @settings(max_examples=20)
    def test_source_multiplier_positive(self, source: str) -> None:
        """Source multiplier is always > 0."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)

        mem = _make_memory(source=source)
        result = engine.calculate_importance(mem, now)

        assert result.source_multiplier > 0.0


class TestExplicitImportanceOverride:
    """Explicit importance parameter overrides metadata."""

    @given(explicit=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50)
    def test_explicit_parameter_used(self, explicit: float) -> None:
        """When explicit_importance is provided, it sets the explicit_score."""
        engine = ImportanceEngine()
        now = datetime.now(timezone.utc)
        mem = _make_memory()

        result = engine.calculate_importance(mem, now, explicit_importance=explicit)

        assert result.explicit_score == explicit

"""Property-based tests for DecayEngine using Hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitive_memory.engines.decay import DecayEngine


@dataclass
class FakeMemory:
    """Minimal memory for property tests."""

    initial_strength: float
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    is_pinned: bool


reasonable_strength = st.floats(min_value=0.01, max_value=1.0)
reasonable_time_offset = st.floats(min_value=0.0, max_value=8760.0)  # up to 1 year in hours
reasonable_access_count = st.integers(min_value=0, max_value=1000)
reasonable_decay_rate = st.floats(min_value=0.001, max_value=2.0)
reasonable_min_strength = st.floats(min_value=0.0, max_value=0.5)


def _make_memory(
    initial_strength: float = 1.0,
    hours_ago: float = 0.0,
    access_count: int = 0,
    is_pinned: bool = False,
    last_accessed_hours_ago: float | None = None,
) -> FakeMemory:
    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=hours_ago)
    last_accessed = (
        now - timedelta(hours=last_accessed_hours_ago)
        if last_accessed_hours_ago is not None
        else created
    )
    return FakeMemory(
        initial_strength=initial_strength,
        created_at=created,
        last_accessed_at=last_accessed,
        access_count=access_count,
        is_pinned=is_pinned,
    )


class TestDecayMonotonicity:
    """Decay is monotonically decreasing without rehearsal."""

    @given(
        initial_strength=reasonable_strength,
        hours_early=st.floats(min_value=0.0, max_value=1000.0),
        hours_late=st.floats(min_value=0.0, max_value=1000.0),
    )
    @settings(max_examples=200)
    def test_strength_decreases_over_time(
        self,
        initial_strength: float,
        hours_early: float,
        hours_late: float,
    ) -> None:
        """Without rehearsal, strength at t1 >= strength at t2 when t1 < t2."""
        engine = DecayEngine(decay_rate=0.1)
        now = datetime.now(timezone.utc)

        t1 = hours_early
        t2 = hours_early + hours_late

        mem = _make_memory(initial_strength=initial_strength, hours_ago=0)

        s1 = engine.calculate_decay(mem, now + timedelta(hours=t1)).decayed_strength
        s2 = engine.calculate_decay(mem, now + timedelta(hours=t2)).decayed_strength

        assert s1 >= s2 - 1e-9  # Allow tiny float tolerance


class TestDecayBounds:
    """Decay strength is always bounded."""

    @given(
        initial_strength=reasonable_strength,
        hours_ago=reasonable_time_offset,
        decay_rate=reasonable_decay_rate,
        min_strength=reasonable_min_strength,
    )
    @settings(max_examples=200)
    def test_strength_always_above_min(
        self,
        initial_strength: float,
        hours_ago: float,
        decay_rate: float,
        min_strength: float,
    ) -> None:
        """Decayed strength is always >= min_strength."""
        engine = DecayEngine(decay_rate=decay_rate, min_strength=min_strength)
        mem = _make_memory(initial_strength=initial_strength, hours_ago=hours_ago)
        now = datetime.now(timezone.utc)

        result = engine.calculate_decay(mem, now)

        assert result.decayed_strength >= min_strength - 1e-9

    @given(
        initial_strength=reasonable_strength,
        hours_ago=reasonable_time_offset,
        access_count=reasonable_access_count,
    )
    @settings(max_examples=200)
    def test_strength_never_exceeds_one(
        self,
        initial_strength: float,
        hours_ago: float,
        access_count: int,
    ) -> None:
        """Decayed strength (even with rehearsal) never exceeds 1.0."""
        engine = DecayEngine(decay_rate=0.1, rehearsal_boost=0.5)
        mem = _make_memory(
            initial_strength=initial_strength,
            hours_ago=hours_ago,
            access_count=access_count,
            last_accessed_hours_ago=0,
        )
        now = datetime.now(timezone.utc)

        result = engine.calculate_decay(mem, now)

        assert result.decayed_strength <= 1.0 + 1e-9


class TestPinnedInvariant:
    """Pinned memories never decay."""

    @given(
        initial_strength=reasonable_strength,
        hours_ago=reasonable_time_offset,
        decay_rate=reasonable_decay_rate,
    )
    @settings(max_examples=100)
    def test_pinned_strength_equals_initial(
        self,
        initial_strength: float,
        hours_ago: float,
        decay_rate: float,
    ) -> None:
        """Pinned memories always return initial_strength regardless of time."""
        engine = DecayEngine(decay_rate=decay_rate)
        mem = _make_memory(initial_strength=initial_strength, hours_ago=hours_ago, is_pinned=True)
        now = datetime.now(timezone.utc)

        result = engine.calculate_decay(mem, now)

        assert result.decayed_strength == pytest.approx(initial_strength)
        assert result.decay_factor == 1.0


class TestRehearsalEffect:
    """Rehearsal always helps or is neutral."""

    @given(
        hours_ago=st.floats(min_value=1.0, max_value=1000.0),
        access_count=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=200)
    def test_rehearsal_increases_strength(
        self,
        hours_ago: float,
        access_count: int,
    ) -> None:
        """A memory with accesses should be >= same memory without accesses."""
        engine = DecayEngine(decay_rate=0.1, rehearsal_boost=0.2)
        now = datetime.now(timezone.utc)

        no_access = _make_memory(hours_ago=hours_ago, access_count=0)
        with_access = _make_memory(
            hours_ago=hours_ago,
            access_count=access_count,
            last_accessed_hours_ago=0,
        )

        s_none = engine.get_strength(no_access, now)
        s_with = engine.get_strength(with_access, now)

        assert s_with >= s_none - 1e-9


class TestDecayFactorProperties:
    """Decay factor itself has invariants."""

    @given(
        hours_ago=reasonable_time_offset,
        decay_rate=reasonable_decay_rate,
    )
    @settings(max_examples=100)
    def test_decay_factor_in_zero_one(
        self,
        hours_ago: float,
        decay_rate: float,
    ) -> None:
        """Decay factor is always in (0, 1]."""
        engine = DecayEngine(decay_rate=decay_rate)
        mem = _make_memory(hours_ago=hours_ago)
        now = datetime.now(timezone.utc)

        result = engine.calculate_decay(mem, now)

        assert 0.0 <= result.decay_factor <= 1.0 + 1e-9

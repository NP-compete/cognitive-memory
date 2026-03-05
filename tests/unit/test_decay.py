"""Tests for the decay engine."""

from datetime import datetime, timedelta, timezone

import pytest

from cognitive_memory.core.config import DecayConfig
from cognitive_memory.engines.decay import DecayEngine, DecayResult


# Create minimal Memory-like objects for testing without importing the full module
class MockMemory:
    """Mock memory for testing decay engine."""

    def __init__(
        self,
        initial_strength: float = 1.0,
        created_at: datetime | None = None,
        last_accessed_at: datetime | None = None,
        access_count: int = 0,
        is_pinned: bool = False,
    ):
        self.initial_strength = initial_strength
        self.created_at = created_at or datetime.now(timezone.utc)
        self.last_accessed_at = last_accessed_at or self.created_at
        self.access_count = access_count
        self.is_pinned = is_pinned


class TestDecayEngine:
    """Tests for DecayEngine."""

    def test_default_values(self) -> None:
        """DecayEngine should have sensible defaults."""
        engine = DecayEngine()

        assert engine.decay_rate == 0.1
        assert engine.min_strength == 0.01
        assert engine.rehearsal_boost == 0.2
        assert engine.time_unit == "hours"

    def test_custom_values(self) -> None:
        """DecayEngine should accept custom values."""
        engine = DecayEngine(
            decay_rate=0.2,
            min_strength=0.05,
            rehearsal_boost=0.3,
            time_unit="days",
        )

        assert engine.decay_rate == 0.2
        assert engine.min_strength == 0.05
        assert engine.rehearsal_boost == 0.3
        assert engine.time_unit == "days"


class TestDecayCalculation:
    """Tests for decay calculation."""

    def test_no_decay_at_creation(self) -> None:
        """Memory should have full strength at creation time."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)
        memory = MockMemory(initial_strength=1.0, created_at=now)

        result = engine.calculate_decay(memory, now)

        assert result.decayed_strength == pytest.approx(1.0, abs=0.01)
        assert result.time_elapsed == pytest.approx(0.0, abs=0.01)

    def test_decay_over_time(self) -> None:
        """Memory strength should decrease over time."""
        engine = DecayEngine(decay_rate=0.1, time_unit="hours")
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=10)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        result = engine.calculate_decay(memory, now)

        # After 10 hours with λ=0.1: e^(-0.1*10) ≈ 0.368
        assert result.decayed_strength == pytest.approx(0.368, abs=0.05)
        assert result.time_elapsed == pytest.approx(10.0, abs=0.1)

    def test_decay_respects_min_strength(self) -> None:
        """Strength should not fall below min_strength."""
        engine = DecayEngine(decay_rate=1.0, min_strength=0.1, time_unit="hours")
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=100)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        result = engine.calculate_decay(memory, now)

        assert result.decayed_strength >= 0.1

    def test_pinned_memory_no_decay(self) -> None:
        """Pinned memories should not decay."""
        engine = DecayEngine(decay_rate=1.0, time_unit="hours")
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=100)
        memory = MockMemory(initial_strength=1.0, created_at=created, is_pinned=True)

        result = engine.calculate_decay(memory, now)

        assert result.decayed_strength == 1.0
        assert result.decay_factor == 1.0

    def test_decay_result_structure(self) -> None:
        """DecayResult should contain all expected fields."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=5)
        memory = MockMemory(initial_strength=0.8, created_at=created)

        result = engine.calculate_decay(memory, now)

        assert isinstance(result, DecayResult)
        assert result.original_strength == 0.8
        assert 0 < result.decayed_strength <= 0.8
        assert result.time_elapsed == pytest.approx(5.0, abs=0.1)
        assert 0 < result.decay_factor < 1


class TestRehearsalEffect:
    """Tests for rehearsal/access effects."""

    def test_rehearsal_increases_strength(self) -> None:
        """Accessing a memory should increase its effective strength."""
        engine = DecayEngine(rehearsal_boost=0.2, time_unit="hours")
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=10)

        # Memory without access
        memory_no_access = MockMemory(
            initial_strength=1.0,
            created_at=created,
            last_accessed_at=created,
            access_count=0,
        )

        # Memory with recent access
        memory_accessed = MockMemory(
            initial_strength=1.0,
            created_at=created,
            last_accessed_at=now - timedelta(hours=1),
            access_count=3,
        )

        result_no_access = engine.calculate_decay(memory_no_access, now)
        result_accessed = engine.calculate_decay(memory_accessed, now)

        assert result_accessed.decayed_strength > result_no_access.decayed_strength
        assert result_accessed.rehearsal_bonus > 0

    def test_rehearsal_bonus_decays(self) -> None:
        """Rehearsal bonus should decay over time since last access."""
        engine = DecayEngine(rehearsal_boost=0.2, rehearsal_decay_rate=0.1)
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=20)

        # Recent access
        memory_recent = MockMemory(
            initial_strength=1.0,
            created_at=created,
            last_accessed_at=now - timedelta(hours=1),
            access_count=5,
        )

        # Old access
        memory_old = MockMemory(
            initial_strength=1.0,
            created_at=created,
            last_accessed_at=now - timedelta(hours=10),
            access_count=5,
        )

        result_recent = engine.calculate_decay(memory_recent, now)
        result_old = engine.calculate_decay(memory_old, now)

        assert result_recent.rehearsal_bonus > result_old.rehearsal_bonus


class TestGetStrength:
    """Tests for the get_strength convenience method."""

    def test_get_strength_returns_float(self) -> None:
        """get_strength should return just the strength value."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)
        memory = MockMemory(initial_strength=1.0, created_at=now)

        strength = engine.get_strength(memory, now)

        assert isinstance(strength, float)
        assert 0 <= strength <= 1

    def test_get_strength_matches_calculate_decay(self) -> None:
        """get_strength should match calculate_decay result."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=5)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        strength = engine.get_strength(memory, now)
        result = engine.calculate_decay(memory, now)

        assert strength == result.decayed_strength


class TestTimeToThreshold:
    """Tests for time-to-threshold estimation."""

    def test_estimate_time_to_threshold(self) -> None:
        """Should estimate time until strength falls below threshold."""
        engine = DecayEngine(decay_rate=0.1, time_unit="hours")
        now = datetime.now(timezone.utc)
        memory = MockMemory(initial_strength=1.0, created_at=now)

        # Time to reach 0.5 strength: t = -ln(0.5) / 0.1 ≈ 6.93 hours
        time_to_half = engine.estimate_time_to_threshold(memory, 0.5, now)

        assert time_to_half is not None
        assert time_to_half == pytest.approx(6.93, abs=0.1)

    def test_pinned_memory_returns_none(self) -> None:
        """Pinned memories should return None (never decay)."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)
        memory = MockMemory(initial_strength=1.0, created_at=now, is_pinned=True)

        result = engine.estimate_time_to_threshold(memory, 0.5, now)

        assert result is None

    def test_already_below_threshold_returns_zero(self) -> None:
        """Should return 0 if already below threshold."""
        engine = DecayEngine(decay_rate=1.0, time_unit="hours")
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=10)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        result = engine.estimate_time_to_threshold(memory, 0.9, now)

        assert result == 0.0


class TestBatchOperations:
    """Tests for batch operations."""

    def test_batch_calculate_decay(self) -> None:
        """Should calculate decay for multiple memories."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)

        memories = [
            MockMemory(initial_strength=1.0, created_at=now - timedelta(hours=i)) for i in range(5)
        ]

        results = engine.batch_calculate_decay(memories, now)

        assert len(results) == 5
        for _memory, result in results:
            assert isinstance(result, DecayResult)

        # Older memories should have lower strength
        strengths = [r.decayed_strength for _, r in results]
        assert strengths == sorted(strengths, reverse=True)

    def test_filter_by_strength(self) -> None:
        """Should filter memories by strength range."""
        engine = DecayEngine(decay_rate=0.1, time_unit="hours")
        now = datetime.now(timezone.utc)

        memories = [
            MockMemory(initial_strength=1.0, created_at=now - timedelta(hours=i * 5))
            for i in range(10)
        ]

        # Filter for mid-range strength
        filtered = engine.filter_by_strength(memories, 0.3, 0.7, now)

        assert len(filtered) > 0
        for m in filtered:
            strength = engine.get_strength(m, now)
            assert 0.3 <= strength <= 0.7


class TestTimeUnits:
    """Tests for different time units."""

    def test_seconds_time_unit(self) -> None:
        """Should work with seconds time unit."""
        engine = DecayEngine(decay_rate=0.01, time_unit="seconds")
        now = datetime.now(timezone.utc)
        created = now - timedelta(seconds=100)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        result = engine.calculate_decay(memory, now)

        assert result.time_elapsed == pytest.approx(100.0, abs=1)

    def test_days_time_unit(self) -> None:
        """Should work with days time unit."""
        engine = DecayEngine(decay_rate=0.1, time_unit="days")
        now = datetime.now(timezone.utc)
        created = now - timedelta(days=7)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        result = engine.calculate_decay(memory, now)

        assert result.time_elapsed == pytest.approx(7.0, abs=0.1)


class TestFromConfig:
    """Tests for from_config class method."""

    def test_from_config_creates_engine(self) -> None:
        """from_config should create engine from DecayConfig."""
        config = DecayConfig(
            decay_rate=0.2,
            min_strength=0.05,
            rehearsal_boost=0.3,
            rehearsal_decay_rate=0.08,
            time_unit="days",
        )

        engine = DecayEngine.from_config(config)

        assert engine.decay_rate == 0.2
        assert engine.min_strength == 0.05
        assert engine.rehearsal_boost == 0.3
        assert engine.rehearsal_decay_rate == 0.08
        assert engine.time_unit == "days"


class TestApplyRehearsalInPlace:
    """Tests for apply_rehearsal_in_place."""

    def test_increments_access_count(self) -> None:
        """Should increment access_count by 1."""
        engine = DecayEngine()
        memory = MockMemory(access_count=3)

        engine.apply_rehearsal_in_place(memory)

        assert memory.access_count == 4

    def test_updates_last_accessed_at(self) -> None:
        """Should set last_accessed_at to approximately now."""
        engine = DecayEngine()
        old_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        memory = MockMemory(last_accessed_at=old_time)

        before = datetime.now(timezone.utc)
        engine.apply_rehearsal_in_place(memory)
        after = datetime.now(timezone.utc)

        assert before <= memory.last_accessed_at <= after


class TestNaiveDatetimeHandling:
    """Tests for naive datetime fallback."""

    def test_naive_created_at(self) -> None:
        """Should handle naive created_at by assuming UTC."""
        engine = DecayEngine()
        naive_time = datetime(2025, 1, 1, 0, 0, 0)
        now = datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        memory = MockMemory(initial_strength=1.0, created_at=naive_time)

        result = engine.calculate_decay(memory, now)

        assert result.time_elapsed == pytest.approx(24.0, abs=0.1)

    def test_naive_current_time(self) -> None:
        """Should handle naive current_time by assuming UTC."""
        engine = DecayEngine()
        created = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        naive_now = datetime(2025, 1, 2, 0, 0, 0)
        memory = MockMemory(initial_strength=1.0, created_at=created)

        result = engine.calculate_decay(memory, naive_now)

        assert result.time_elapsed == pytest.approx(24.0, abs=0.1)

    def test_naive_last_accessed_at_in_rehearsal(self) -> None:
        """Should handle naive last_accessed_at in rehearsal calc."""
        engine = DecayEngine(rehearsal_boost=0.2)
        created = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        naive_accessed = datetime(2025, 1, 1, 23, 0, 0)
        now = datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

        memory = MockMemory(
            initial_strength=1.0,
            created_at=created,
            last_accessed_at=naive_accessed,
            access_count=5,
        )

        result = engine.calculate_decay(memory, now)

        assert result.rehearsal_bonus > 0


class TestEdgeCases:
    """Tests for edge cases in decay engine."""

    def test_zero_initial_strength_already_below(self) -> None:
        """With initial_strength=0, strength is at min, already below any threshold."""
        engine = DecayEngine()
        now = datetime.now(timezone.utc)
        memory = MockMemory(initial_strength=0.0, created_at=now)

        result = engine.estimate_time_to_threshold(memory, 0.5, now)

        assert result == 0.0

    def test_threshold_equals_initial_already_at(self) -> None:
        """When current strength <= threshold, returns 0.0 immediately."""
        engine = DecayEngine(decay_rate=1.0)
        now = datetime.now(timezone.utc)
        created = now - timedelta(hours=10)
        memory = MockMemory(initial_strength=0.5, created_at=created)

        result = engine.estimate_time_to_threshold(memory, 0.5, now)

        assert result == 0.0

    def test_threshold_ratio_ge_one_returns_none(self) -> None:
        """When threshold/initial >= 1, returns None (can't reach by decay alone)."""
        engine = DecayEngine(rehearsal_boost=0.2)
        now = datetime.now(timezone.utc)
        memory = MockMemory(
            initial_strength=0.3,
            created_at=now,
            last_accessed_at=now,
            access_count=10,
        )

        result = engine.estimate_time_to_threshold(memory, 0.4, now)

        assert result is None

    def test_default_current_time_in_estimate(self) -> None:
        """estimate_time_to_threshold uses now when current_time is None."""
        engine = DecayEngine()
        memory = MockMemory(initial_strength=1.0, created_at=datetime.now(timezone.utc))

        result = engine.estimate_time_to_threshold(memory, 0.5)

        assert result is not None
        assert result > 0

    def test_unknown_time_unit_defaults_to_hours(self) -> None:
        """Unknown time_unit should default to 3600 seconds (hours)."""
        engine = DecayEngine(time_unit="fortnights")

        assert engine._time_unit_seconds == 3600

    def test_batch_calculate_default_time(self) -> None:
        """batch_calculate_decay should work without explicit time."""
        engine = DecayEngine()
        memories = [MockMemory(initial_strength=1.0)]

        results = engine.batch_calculate_decay(memories)

        assert len(results) == 1

    def test_filter_by_strength_default_time(self) -> None:
        """filter_by_strength should work without explicit time."""
        engine = DecayEngine()
        memories = [MockMemory(initial_strength=1.0)]

        result = engine.filter_by_strength(memories)

        assert len(result) == 1

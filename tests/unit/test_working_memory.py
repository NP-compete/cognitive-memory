"""Tests for working memory manager."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from cognitive_memory.engines.working_memory import (
    WorkingMemoryManager,
    WorkingMemoryState,
)


@dataclass
class MockMemory:
    """Mock memory for testing."""

    id: str
    content: str = "test content"
    importance: float = 0.5
    strength: float = 0.8
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TestWorkingMemoryManager:
    """Tests for WorkingMemoryManager."""

    def test_default_values(self) -> None:
        """Manager should have sensible defaults."""
        manager = WorkingMemoryManager()

        assert manager.capacity == 7
        assert manager.activation_decay_rate == 0.1
        assert manager.min_activation == 0.1
        assert manager.refresh_boost == 0.3
        assert manager.importance_weight == 0.5

    def test_initial_state(self) -> None:
        """Manager should start empty."""
        manager = WorkingMemoryManager()

        assert manager.size == 0
        assert manager.is_full is False
        assert manager.available_slots == 7


class TestAdd:
    """Tests for add operation."""

    def test_add_memory(self) -> None:
        """Add should create a slot."""
        manager = WorkingMemoryManager()
        memory = MockMemory(id="m1")

        result = manager.add(memory)

        assert result is True
        assert manager.contains("m1")
        assert manager.size == 1

    def test_add_duplicate(self) -> None:
        """Adding duplicate should refresh instead."""
        manager = WorkingMemoryManager()
        memory = MockMemory(id="m1")

        manager.add(memory)
        result = manager.add(memory)

        assert result is False
        assert manager.size == 1

    def test_add_custom_activation(self) -> None:
        """Add should accept custom activation."""
        manager = WorkingMemoryManager()
        memory = MockMemory(id="m1")

        manager.add(memory, initial_activation=0.9)
        slot = manager.get_slot("m1")

        assert slot is not None
        assert slot.activation == 0.9

    def test_add_clamps_activation(self) -> None:
        """Add should clamp activation to [0, 1]."""
        manager = WorkingMemoryManager()

        manager.add(MockMemory(id="m1"), initial_activation=1.5)
        manager.add(MockMemory(id="m2"), initial_activation=-0.5)

        assert manager.get_slot("m1").activation == 1.0
        assert manager.get_slot("m2").activation == 0.0

    def test_add_evicts_when_full(self) -> None:
        """Add should evict lowest activation when full."""
        manager = WorkingMemoryManager(capacity=3)

        manager.add(MockMemory(id="m1"), initial_activation=0.3)
        manager.add(MockMemory(id="m2"), initial_activation=0.5)
        manager.add(MockMemory(id="m3"), initial_activation=0.7)
        manager.add(MockMemory(id="m4"), initial_activation=0.9)

        assert manager.size == 3
        assert not manager.contains("m1")  # Lowest activation evicted
        assert manager.contains("m4")


class TestRemove:
    """Tests for remove operation."""

    def test_remove_existing(self) -> None:
        """Remove should delete slot."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1"))

        result = manager.remove("m1")

        assert result is True
        assert not manager.contains("m1")

    def test_remove_nonexistent(self) -> None:
        """Remove should return False for missing."""
        manager = WorkingMemoryManager()

        result = manager.remove("nonexistent")

        assert result is False


class TestRefresh:
    """Tests for refresh operation."""

    def test_refresh_existing(self) -> None:
        """Refresh should boost activation."""
        manager = WorkingMemoryManager(refresh_boost=0.2)
        manager.add(MockMemory(id="m1"), initial_activation=0.5)

        result = manager.refresh("m1")
        slot = manager.get_slot("m1")

        assert result is True
        assert slot.activation == pytest.approx(0.7)
        assert slot.refresh_count == 1

    def test_refresh_nonexistent(self) -> None:
        """Refresh should return False for missing."""
        manager = WorkingMemoryManager()

        result = manager.refresh("nonexistent")

        assert result is False

    def test_refresh_caps_at_one(self) -> None:
        """Refresh should not exceed 1.0."""
        manager = WorkingMemoryManager(refresh_boost=0.5)
        manager.add(MockMemory(id="m1"), initial_activation=0.9)

        manager.refresh("m1")
        slot = manager.get_slot("m1")

        assert slot.activation == 1.0


class TestDecay:
    """Tests for decay operation."""

    def test_decay_reduces_activation(self) -> None:
        """Decay should reduce activation."""
        manager = WorkingMemoryManager(activation_decay_rate=0.1)
        manager.add(MockMemory(id="m1"), initial_activation=1.0)

        manager.decay_all(elapsed_seconds=1.0)
        slot = manager.get_slot("m1")

        assert slot.activation < 1.0

    def test_decay_evicts_below_threshold(self) -> None:
        """Decay should evict slots below threshold."""
        manager = WorkingMemoryManager(
            activation_decay_rate=0.5,
            min_activation=0.3,
        )
        manager.add(MockMemory(id="m1"), initial_activation=0.4)

        evicted = manager.decay_all(elapsed_seconds=1.0)

        assert evicted == 1
        assert not manager.contains("m1")

    def test_decay_returns_eviction_count(self) -> None:
        """Decay should return number evicted."""
        manager = WorkingMemoryManager(
            activation_decay_rate=0.9,
            min_activation=0.5,
        )
        manager.add(MockMemory(id="m1"), initial_activation=0.6)
        manager.add(MockMemory(id="m2"), initial_activation=0.6)

        evicted = manager.decay_all(elapsed_seconds=1.0)

        assert evicted == 2


class TestClear:
    """Tests for clear operation."""

    def test_clear_removes_all(self) -> None:
        """Clear should remove all slots."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1"))
        manager.add(MockMemory(id="m2"))

        count = manager.clear()

        assert count == 2
        assert manager.size == 0


class TestGetState:
    """Tests for get_state operation."""

    def test_empty_state(self) -> None:
        """Empty manager should return empty state."""
        manager = WorkingMemoryManager()

        state = manager.get_state()

        assert isinstance(state, WorkingMemoryState)
        assert len(state.slots) == 0
        assert state.total_activation == 0.0

    def test_state_with_slots(self) -> None:
        """State should reflect current slots."""
        manager = WorkingMemoryManager(capacity=5)
        manager.add(MockMemory(id="m1"), initial_activation=0.5)
        manager.add(MockMemory(id="m2"), initial_activation=0.7)

        state = manager.get_state()

        assert len(state.slots) == 2
        assert state.capacity == 5
        assert state.total_activation == pytest.approx(1.2)


class TestGetByActivation:
    """Tests for get_by_activation."""

    def test_sorted_descending(self) -> None:
        """Should return slots sorted by activation descending."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1"), initial_activation=0.3)
        manager.add(MockMemory(id="m2"), initial_activation=0.9)
        manager.add(MockMemory(id="m3"), initial_activation=0.6)

        slots = manager.get_by_activation(descending=True)

        assert slots[0].memory_id == "m2"
        assert slots[1].memory_id == "m3"
        assert slots[2].memory_id == "m1"

    def test_sorted_ascending(self) -> None:
        """Should return slots sorted by activation ascending."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1"), initial_activation=0.3)
        manager.add(MockMemory(id="m2"), initial_activation=0.9)

        slots = manager.get_by_activation(descending=False)

        assert slots[0].memory_id == "m1"
        assert slots[1].memory_id == "m2"


class TestContextSummary:
    """Tests for get_context_summary."""

    def test_empty_summary(self) -> None:
        """Empty manager should return empty string."""
        manager = WorkingMemoryManager()

        summary = manager.get_context_summary()

        assert summary == ""

    def test_summary_content(self) -> None:
        """Summary should contain slot contents."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1", content="first"))
        manager.add(MockMemory(id="m2", content="second"))

        summary = manager.get_context_summary()

        assert "first" in summary
        assert "second" in summary

    def test_summary_max_items(self) -> None:
        """Summary should respect max_items."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1", content="first"), initial_activation=0.3)
        manager.add(MockMemory(id="m2", content="second"), initial_activation=0.9)
        manager.add(MockMemory(id="m3", content="third"), initial_activation=0.6)

        summary = manager.get_context_summary(max_items=2)

        assert "second" in summary  # Highest activation
        assert "third" in summary   # Second highest
        assert "first" not in summary


class TestSerialization:
    """Tests for serialization."""

    def test_to_dict_list(self) -> None:
        """Should convert to list of dicts."""
        manager = WorkingMemoryManager()
        manager.add(MockMemory(id="m1", content="test"))

        data = manager.to_dict_list()

        assert len(data) == 1
        assert data[0]["memory_id"] == "m1"
        assert data[0]["content"] == "test"

    def test_from_dict_list(self) -> None:
        """Should restore from list of dicts."""
        manager = WorkingMemoryManager()
        now = datetime.now(timezone.utc)
        data = [
            {
                "memory_id": "m1",
                "content": "restored",
                "importance": 0.7,
                "activation": 0.8,
                "added_at": now.isoformat(),
                "last_refreshed_at": now.isoformat(),
                "refresh_count": 2,
            }
        ]

        count = manager.from_dict_list(data)

        assert count == 1
        slot = manager.get_slot("m1")
        assert slot.content == "restored"
        assert slot.importance == 0.7
        assert slot.refresh_count == 2

    def test_roundtrip(self) -> None:
        """Should survive serialization roundtrip."""
        manager1 = WorkingMemoryManager()
        manager1.add(MockMemory(id="m1", content="test", importance=0.6))
        manager1.refresh("m1")

        data = manager1.to_dict_list()

        manager2 = WorkingMemoryManager()
        manager2.from_dict_list(data)

        assert manager2.size == 1
        slot = manager2.get_slot("m1")
        assert slot.content == "test"
        assert slot.refresh_count == 1


class TestProperties:
    """Tests for properties."""

    def test_is_full(self) -> None:
        """is_full should reflect capacity."""
        manager = WorkingMemoryManager(capacity=2)

        assert manager.is_full is False

        manager.add(MockMemory(id="m1"))
        assert manager.is_full is False

        manager.add(MockMemory(id="m2"))
        assert manager.is_full is True

    def test_available_slots(self) -> None:
        """available_slots should reflect remaining capacity."""
        manager = WorkingMemoryManager(capacity=3)

        assert manager.available_slots == 3

        manager.add(MockMemory(id="m1"))
        assert manager.available_slots == 2

        manager.add(MockMemory(id="m2"))
        assert manager.available_slots == 1

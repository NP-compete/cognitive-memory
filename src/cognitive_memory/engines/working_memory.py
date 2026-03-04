"""Working memory manager for active context maintenance."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

logger = logging.getLogger(__name__)


class MemoryProtocol(Protocol):
    """Protocol for memory objects used by working memory manager."""

    id: str
    content: str
    importance: float
    strength: float
    created_at: datetime
    last_accessed_at: datetime


@dataclass
class WorkingMemorySlot:
    """A slot in working memory holding a memory reference."""

    memory_id: str
    content: str
    importance: float
    activation: float
    added_at: datetime
    last_refreshed_at: datetime
    refresh_count: int = 0


@dataclass
class WorkingMemoryState:
    """Current state of working memory."""

    slots: list[WorkingMemorySlot]
    capacity: int
    total_activation: float
    oldest_slot_age_seconds: float
    newest_slot_age_seconds: float


@dataclass
class WorkingMemoryManager:
    """
    Manager for working memory (active context).

    Implements a capacity-limited working memory similar to
    human short-term memory, with activation-based retention
    and automatic decay.

    Attributes:
        capacity: Maximum number of items in working memory.
        activation_decay_rate: Rate at which activation decays.
        min_activation: Minimum activation before eviction.
        refresh_boost: Activation boost when refreshed.
        importance_weight: Weight of importance in activation.
    """

    capacity: int = 7
    activation_decay_rate: float = 0.1
    min_activation: float = 0.1
    refresh_boost: float = 0.3
    importance_weight: float = 0.5
    _slots: dict[str, WorkingMemorySlot] = field(default_factory=dict, init=False)
    _access_order: list[str] = field(default_factory=list, init=False)

    def add(
        self,
        memory: MemoryProtocol,
        initial_activation: float | None = None,
    ) -> bool:
        """
        Add a memory to working memory.

        Args:
            memory: Memory to add.
            initial_activation: Initial activation level (default: based on importance).

        Returns:
            True if added, False if already present.
        """
        if memory.id in self._slots:
            # Already present, refresh instead
            self.refresh(memory.id)
            return False

        # Calculate initial activation
        if initial_activation is None:
            initial_activation = 0.5 + (memory.importance * self.importance_weight)
        initial_activation = min(1.0, max(0.0, initial_activation))

        now = datetime.now(timezone.utc)
        slot = WorkingMemorySlot(
            memory_id=memory.id,
            content=memory.content,
            importance=memory.importance,
            activation=initial_activation,
            added_at=now,
            last_refreshed_at=now,
        )

        # Check capacity
        if len(self._slots) >= self.capacity:
            self._evict_lowest_activation()

        self._slots[memory.id] = slot
        self._access_order.append(memory.id)

        logger.debug(f"Added {memory.id} to working memory (activation={initial_activation:.2f})")
        return True

    def remove(self, memory_id: str) -> bool:
        """
        Remove a memory from working memory.

        Args:
            memory_id: ID of memory to remove.

        Returns:
            True if removed, False if not found.
        """
        if memory_id not in self._slots:
            return False

        del self._slots[memory_id]
        if memory_id in self._access_order:
            self._access_order.remove(memory_id)

        logger.debug(f"Removed {memory_id} from working memory")
        return True

    def refresh(self, memory_id: str) -> bool:
        """
        Refresh a memory's activation in working memory.

        Args:
            memory_id: ID of memory to refresh.

        Returns:
            True if refreshed, False if not found.
        """
        if memory_id not in self._slots:
            return False

        slot = self._slots[memory_id]
        slot.activation = min(1.0, slot.activation + self.refresh_boost)
        slot.last_refreshed_at = datetime.now(timezone.utc)
        slot.refresh_count += 1

        # Move to end of access order
        if memory_id in self._access_order:
            self._access_order.remove(memory_id)
        self._access_order.append(memory_id)

        logger.debug(f"Refreshed {memory_id} (activation={slot.activation:.2f})")
        return True

    def contains(self, memory_id: str) -> bool:
        """Check if a memory is in working memory."""
        return memory_id in self._slots

    def get_slot(self, memory_id: str) -> WorkingMemorySlot | None:
        """Get a slot by memory ID."""
        return self._slots.get(memory_id)

    def get_all_slots(self) -> list[WorkingMemorySlot]:
        """Get all slots in working memory."""
        return list(self._slots.values())

    def get_memory_ids(self) -> list[str]:
        """Get all memory IDs in working memory."""
        return list(self._slots.keys())

    def get_by_activation(self, descending: bool = True) -> list[WorkingMemorySlot]:
        """Get slots sorted by activation."""
        slots = list(self._slots.values())
        return sorted(slots, key=lambda s: s.activation, reverse=descending)

    def get_state(self) -> WorkingMemoryState:
        """Get current working memory state."""
        slots = list(self._slots.values())
        now = datetime.now(timezone.utc)

        if not slots:
            return WorkingMemoryState(
                slots=[],
                capacity=self.capacity,
                total_activation=0.0,
                oldest_slot_age_seconds=0.0,
                newest_slot_age_seconds=0.0,
            )

        ages = [(now - s.added_at).total_seconds() for s in slots]

        return WorkingMemoryState(
            slots=slots,
            capacity=self.capacity,
            total_activation=sum(s.activation for s in slots),
            oldest_slot_age_seconds=max(ages),
            newest_slot_age_seconds=min(ages),
        )

    def decay_all(self, elapsed_seconds: float | None = None) -> int:
        """
        Apply decay to all slots and evict those below threshold.

        Args:
            elapsed_seconds: Time elapsed (default: since last refresh).

        Returns:
            Number of slots evicted.
        """
        if elapsed_seconds is None:
            elapsed_seconds = 1.0

        evicted = 0
        to_evict = []

        for memory_id, slot in self._slots.items():
            # Apply exponential decay
            decay_factor = self.activation_decay_rate * elapsed_seconds
            slot.activation = slot.activation * (1.0 - decay_factor)

            if slot.activation < self.min_activation:
                to_evict.append(memory_id)

        for memory_id in to_evict:
            self.remove(memory_id)
            evicted += 1

        if evicted > 0:
            logger.debug(f"Evicted {evicted} slots due to low activation")

        return evicted

    def clear(self) -> int:
        """
        Clear all slots from working memory.

        Returns:
            Number of slots cleared.
        """
        count = len(self._slots)
        self._slots.clear()
        self._access_order.clear()
        return count

    def _evict_lowest_activation(self) -> str | None:
        """Evict the slot with lowest activation."""
        if not self._slots:
            return None

        lowest_id = min(self._slots.keys(), key=lambda k: self._slots[k].activation)
        self.remove(lowest_id)
        logger.debug(f"Evicted {lowest_id} due to capacity limit")
        return lowest_id

    def get_context_summary(
        self,
        max_items: int | None = None,
        separator: str = "\n",
    ) -> str:
        """
        Get a text summary of working memory contents.

        Args:
            max_items: Maximum items to include.
            separator: Separator between items.

        Returns:
            Concatenated content string.
        """
        slots = self.get_by_activation(descending=True)

        if max_items is not None:
            slots = slots[:max_items]

        return separator.join(s.content for s in slots)

    def to_dict_list(self) -> list[dict[str, object]]:
        """Convert working memory to list of dicts for serialization."""
        return [
            {
                "memory_id": s.memory_id,
                "content": s.content,
                "importance": s.importance,
                "activation": s.activation,
                "added_at": s.added_at.isoformat(),
                "last_refreshed_at": s.last_refreshed_at.isoformat(),
                "refresh_count": s.refresh_count,
            }
            for s in self._slots.values()
        ]

    def from_dict_list(self, data: list[dict[str, object]]) -> int:
        """
        Restore working memory from list of dicts.

        Args:
            data: List of slot dicts.

        Returns:
            Number of slots restored.
        """
        self.clear()

        for item in data:
            slot = WorkingMemorySlot(
                memory_id=str(item["memory_id"]),
                content=str(item["content"]),
                importance=float(str(item.get("importance", 0.5))),
                activation=float(str(item.get("activation", 0.5))),
                added_at=datetime.fromisoformat(str(item["added_at"])),
                last_refreshed_at=datetime.fromisoformat(str(item["last_refreshed_at"])),
                refresh_count=int(str(item.get("refresh_count", 0))),
            )
            self._slots[slot.memory_id] = slot
            self._access_order.append(slot.memory_id)

        return len(self._slots)

    @property
    def size(self) -> int:
        """Current number of items in working memory."""
        return len(self._slots)

    @property
    def is_full(self) -> bool:
        """Whether working memory is at capacity."""
        return len(self._slots) >= self.capacity

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return max(0, self.capacity - len(self._slots))


# Type alias for activation calculator
ActivationCalculator = Callable[[MemoryProtocol], float]

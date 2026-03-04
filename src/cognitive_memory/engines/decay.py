"""Decay engine for memory strength calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

# Time unit to seconds conversion
TIME_UNIT_SECONDS = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
}


class MemoryProtocol(Protocol):
    """Protocol for memory objects compatible with DecayEngine."""

    initial_strength: float
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    is_pinned: bool


class DecayConfigProtocol(Protocol):
    """Protocol for decay configuration objects."""

    decay_rate: float
    min_strength: float
    rehearsal_boost: float
    rehearsal_decay_rate: float
    time_unit: str


@dataclass
class DecayResult:
    """
    Result of decay calculation.

    Attributes:
        original_strength: Strength before decay calculation.
        decayed_strength: Strength after applying decay.
        time_elapsed: Time elapsed since creation in configured units.
        rehearsal_bonus: Bonus from rehearsal/access.
        decay_factor: The exponential decay factor applied.
    """

    original_strength: float
    decayed_strength: float
    time_elapsed: float
    rehearsal_bonus: float
    decay_factor: float


@dataclass
class DecayEngine:
    """
    Engine for calculating memory decay over time.

    Implements exponential decay with rehearsal effects:
    strength = initial_strength * e^(-λt) + rehearsal_bonus

    Where:
    - λ (lambda) is the decay_rate
    - t is time elapsed in configured units
    - rehearsal_bonus is accumulated from memory accesses

    The rehearsal bonus itself decays over time, simulating
    the spacing effect in human memory.

    Attributes:
        decay_rate: Lambda (λ) in the decay function.
        min_strength: Floor value for strength.
        rehearsal_boost: Strength increase per access.
        rehearsal_decay_rate: How quickly rehearsal bonus fades.
        time_unit: Unit for time calculations.
    """

    decay_rate: float = 0.1
    min_strength: float = 0.01
    rehearsal_boost: float = 0.2
    rehearsal_decay_rate: float = 0.05
    time_unit: str = "hours"
    _time_unit_seconds: int = field(init=False, repr=False, default=3600)

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        self._time_unit_seconds = TIME_UNIT_SECONDS.get(self.time_unit, 3600)

    @classmethod
    def from_config(cls, config: DecayConfigProtocol) -> DecayEngine:
        """
        Create a DecayEngine from configuration.

        Args:
            config: DecayConfig instance.

        Returns:
            Configured DecayEngine.
        """
        return cls(
            decay_rate=config.decay_rate,
            min_strength=config.min_strength,
            rehearsal_boost=config.rehearsal_boost,
            rehearsal_decay_rate=config.rehearsal_decay_rate,
            time_unit=config.time_unit,
        )

    def calculate_decay(
        self,
        memory: MemoryProtocol,
        current_time: datetime | None = None,
    ) -> DecayResult:
        """
        Calculate the current strength of a memory after decay.

        Args:
            memory: The memory to calculate decay for.
            current_time: Time to calculate decay at. Defaults to now.

        Returns:
            DecayResult with decay calculation details.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Ensure timezone-aware comparison
        created_at = memory.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Calculate time elapsed in configured units
        elapsed_seconds = (current_time - created_at).total_seconds()
        time_elapsed = elapsed_seconds / self._time_unit_seconds

        # Pinned memories don't decay
        if memory.is_pinned:
            return DecayResult(
                original_strength=memory.initial_strength,
                decayed_strength=memory.initial_strength,
                time_elapsed=time_elapsed,
                rehearsal_bonus=0.0,
                decay_factor=1.0,
            )

        # Calculate base exponential decay
        decay_factor = math.exp(-self.decay_rate * time_elapsed)
        base_strength = memory.initial_strength * decay_factor

        # Calculate rehearsal bonus (also decays, but slower)
        # Each access adds rehearsal_boost, which decays at rehearsal_decay_rate
        rehearsal_bonus = self._calculate_rehearsal_bonus(memory, current_time)

        # Combine and clamp to [min_strength, 1.0]
        decayed_strength = base_strength + rehearsal_bonus
        decayed_strength = max(self.min_strength, min(1.0, decayed_strength))

        return DecayResult(
            original_strength=memory.initial_strength,
            decayed_strength=decayed_strength,
            time_elapsed=time_elapsed,
            rehearsal_bonus=rehearsal_bonus,
            decay_factor=decay_factor,
        )

    def _calculate_rehearsal_bonus(
        self,
        memory: MemoryProtocol,
        current_time: datetime,
    ) -> float:
        """
        Calculate the rehearsal bonus from memory accesses.

        Each access adds a boost that decays over time.
        More recent accesses contribute more.

        Args:
            memory: The memory to calculate bonus for.
            current_time: Current time for decay calculation.

        Returns:
            Total rehearsal bonus.
        """
        if memory.access_count == 0:
            return 0.0

        # Ensure timezone-aware
        last_accessed = memory.last_accessed_at
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        # Time since last access
        elapsed_seconds = (current_time - last_accessed).total_seconds()
        time_since_access = elapsed_seconds / self._time_unit_seconds

        # Rehearsal bonus decays from last access
        # Scaled by log of access count (diminishing returns)
        access_multiplier = math.log1p(memory.access_count)
        base_bonus = self.rehearsal_boost * access_multiplier
        bonus_decay = math.exp(-self.rehearsal_decay_rate * time_since_access)

        return base_bonus * bonus_decay

    def get_strength(
        self,
        memory: MemoryProtocol,
        current_time: datetime | None = None,
    ) -> float:
        """
        Get the current strength of a memory.

        Convenience method that returns just the strength value.

        Args:
            memory: The memory to get strength for.
            current_time: Time to calculate at. Defaults to now.

        Returns:
            Current memory strength after decay.
        """
        result = self.calculate_decay(memory, current_time)
        return result.decayed_strength

    def apply_rehearsal_in_place(self, memory: Any) -> None:
        """
        Apply rehearsal effect to a memory (simulate access).

        Updates the memory's access count and last_accessed_at in place.

        Args:
            memory: The memory being accessed. Must have mutable
                access_count and last_accessed_at attributes.
        """
        memory.access_count = memory.access_count + 1
        memory.last_accessed_at = datetime.now(timezone.utc)

    def estimate_time_to_threshold(
        self,
        memory: MemoryProtocol,
        threshold: float,
        current_time: datetime | None = None,
    ) -> float | None:
        """
        Estimate time until memory strength falls below threshold.

        Useful for scheduling cleanup or consolidation.

        Args:
            memory: The memory to estimate for.
            threshold: Strength threshold to reach.
            current_time: Starting time. Defaults to now.

        Returns:
            Estimated time in configured units, or None if pinned
            or already below threshold.
        """
        if memory.is_pinned:
            return None

        if current_time is None:
            current_time = datetime.now(timezone.utc)

        current_strength = self.get_strength(memory, current_time)

        if current_strength <= threshold:
            return 0.0

        if memory.initial_strength <= 0 or threshold <= 0:
            return None

        ratio = threshold / memory.initial_strength
        if ratio >= 1:
            return None

        time_to_threshold = -math.log(ratio) / self.decay_rate

        # Subtract time already elapsed
        created_at = memory.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        elapsed = (current_time - created_at).total_seconds() / self._time_unit_seconds
        remaining = time_to_threshold - elapsed

        return max(0.0, remaining)

    def batch_calculate_decay(
        self,
        memories: list[MemoryProtocol],
        current_time: datetime | None = None,
    ) -> list[tuple[MemoryProtocol, DecayResult]]:
        """
        Calculate decay for multiple memories efficiently.

        Args:
            memories: List of memories to process.
            current_time: Time to calculate at. Defaults to now.

        Returns:
            List of (memory, DecayResult) tuples.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        return [(m, self.calculate_decay(m, current_time)) for m in memories]

    def filter_by_strength(
        self,
        memories: list[MemoryProtocol],
        min_strength: float = 0.0,
        max_strength: float = 1.0,
        current_time: datetime | None = None,
    ) -> list[MemoryProtocol]:
        """
        Filter memories by their current strength.

        Args:
            memories: List of memories to filter.
            min_strength: Minimum strength (inclusive).
            max_strength: Maximum strength (inclusive).
            current_time: Time to calculate at. Defaults to now.

        Returns:
            Memories with strength in the specified range.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        return [
            m
            for m in memories
            if min_strength <= self.get_strength(m, current_time) <= max_strength
        ]

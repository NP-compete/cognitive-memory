"""Importance engine for memory scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


class MemoryProtocol(Protocol):
    """Protocol for memory objects compatible with ImportanceEngine."""

    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    emotional_valence: float
    surprise_score: float
    source: Any
    entities: list[str]
    metadata: dict[str, Any]


class ImportanceConfigProtocol(Protocol):
    """Protocol for importance configuration objects."""

    recency_weight: float
    frequency_weight: float
    emotional_weight: float
    surprise_weight: float
    entity_weight: float
    explicit_weight: float
    source_weights: dict[str, float]


@dataclass
class ImportanceResult:
    """
    Result of importance calculation.

    Attributes:
        final_score: Combined weighted importance score (0-1).
        recency_score: Score from creation recency (0-1).
        frequency_score: Score from access frequency (0-1).
        emotional_score: Score from emotional valence (0-1).
        surprise_score: Score from surprise/novelty (0-1).
        entity_score: Score from entity count (0-1).
        explicit_score: Score from explicit importance (0-1).
        source_multiplier: Multiplier from memory source.
        breakdown: Dict of all component scores for debugging.
    """

    final_score: float
    recency_score: float
    frequency_score: float
    emotional_score: float
    surprise_score: float
    entity_score: float
    explicit_score: float
    source_multiplier: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class ImportanceEngine:
    """
    Engine for calculating memory importance scores.

    Importance is a multi-factor score that determines how valuable
    a memory is for retrieval. Higher importance memories are
    prioritized in retrieval and resist decay longer.

    The formula is:
    importance = Σ(weight_i * score_i) * source_multiplier

    Where scores are normalized to [0, 1] and weights sum to 1.

    Attributes:
        recency_weight: Weight for creation recency.
        frequency_weight: Weight for access frequency.
        emotional_weight: Weight for emotional intensity.
        surprise_weight: Weight for surprise/novelty.
        entity_weight: Weight for entity richness.
        explicit_weight: Weight for user-marked importance.
        source_weights: Multipliers by memory source type.
        recency_half_life_hours: Hours until recency score halves.
        frequency_saturation: Access count for max frequency score.
        entity_saturation: Entity count for max entity score.
    """

    recency_weight: float = 0.2
    frequency_weight: float = 0.15
    emotional_weight: float = 0.2
    surprise_weight: float = 0.15
    entity_weight: float = 0.1
    explicit_weight: float = 0.2
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "user_explicit": 1.5,
            "tool_result": 1.2,
            "observation": 1.0,
            "conversation": 0.8,
            "consolidation": 1.1,
            "external": 0.9,
        }
    )
    recency_half_life_hours: float = 24.0
    frequency_saturation: int = 10
    entity_saturation: int = 5

    @classmethod
    def from_config(cls, config: ImportanceConfigProtocol) -> ImportanceEngine:
        """
        Create an ImportanceEngine from configuration.

        Args:
            config: ImportanceConfig instance.

        Returns:
            Configured ImportanceEngine.
        """
        return cls(
            recency_weight=config.recency_weight,
            frequency_weight=config.frequency_weight,
            emotional_weight=config.emotional_weight,
            surprise_weight=config.surprise_weight,
            entity_weight=config.entity_weight,
            explicit_weight=config.explicit_weight,
            source_weights=dict(config.source_weights),
        )

    def calculate_importance(
        self,
        memory: MemoryProtocol,
        current_time: datetime | None = None,
        explicit_importance: float | None = None,
    ) -> ImportanceResult:
        """
        Calculate the importance score for a memory.

        Args:
            memory: The memory to score.
            current_time: Time for recency calculation. Defaults to now.
            explicit_importance: Optional user-provided importance (0-1).

        Returns:
            ImportanceResult with score breakdown.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Calculate individual scores
        recency_score = self._calculate_recency_score(memory, current_time)
        frequency_score = self._calculate_frequency_score(memory)
        emotional_score = self._calculate_emotional_score(memory)
        surprise_score = self._calculate_surprise_score(memory)
        entity_score = self._calculate_entity_score(memory)
        explicit_score = self._calculate_explicit_score(memory, explicit_importance)

        # Get source multiplier
        source_multiplier = self._get_source_multiplier(memory)

        # Calculate weighted sum
        weighted_sum = (
            self.recency_weight * recency_score
            + self.frequency_weight * frequency_score
            + self.emotional_weight * emotional_score
            + self.surprise_weight * surprise_score
            + self.entity_weight * entity_score
            + self.explicit_weight * explicit_score
        )

        # Apply source multiplier and clamp to [0, 1]
        final_score = min(1.0, max(0.0, weighted_sum * source_multiplier))

        return ImportanceResult(
            final_score=final_score,
            recency_score=recency_score,
            frequency_score=frequency_score,
            emotional_score=emotional_score,
            surprise_score=surprise_score,
            entity_score=entity_score,
            explicit_score=explicit_score,
            source_multiplier=source_multiplier,
            breakdown={
                "recency": recency_score,
                "frequency": frequency_score,
                "emotional": emotional_score,
                "surprise": surprise_score,
                "entity": entity_score,
                "explicit": explicit_score,
                "source_multiplier": source_multiplier,
            },
        )

    def _calculate_recency_score(
        self,
        memory: MemoryProtocol,
        current_time: datetime,
    ) -> float:
        """
        Calculate recency score based on creation time.

        Uses exponential decay with configurable half-life.
        Score = 0.5^(hours_elapsed / half_life)

        Args:
            memory: The memory to score.
            current_time: Current time for calculation.

        Returns:
            Recency score (0-1), higher for more recent.
        """
        created_at = memory.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        hours_elapsed = (current_time - created_at).total_seconds() / 3600

        if hours_elapsed <= 0:
            return 1.0

        return math.pow(0.5, hours_elapsed / self.recency_half_life_hours)

    def _calculate_frequency_score(self, memory: MemoryProtocol) -> float:
        """
        Calculate frequency score based on access count.

        Uses logarithmic scaling with saturation.
        Score = log(1 + access_count) / log(1 + saturation)

        Args:
            memory: The memory to score.

        Returns:
            Frequency score (0-1), higher for more accessed.
        """
        if memory.access_count <= 0:
            return 0.0

        score = math.log1p(memory.access_count) / math.log1p(self.frequency_saturation)
        return min(1.0, score)

    def _calculate_emotional_score(self, memory: MemoryProtocol) -> float:
        """
        Calculate emotional score based on valence intensity.

        Uses absolute value since both strong positive and negative
        emotions indicate importance.

        Args:
            memory: The memory to score.

        Returns:
            Emotional score (0-1), higher for stronger emotions.
        """
        return abs(memory.emotional_valence)

    def _calculate_surprise_score(self, memory: MemoryProtocol) -> float:
        """
        Calculate surprise score from memory's novelty.

        Directly uses the memory's surprise_score field.

        Args:
            memory: The memory to score.

        Returns:
            Surprise score (0-1).
        """
        return max(0.0, min(1.0, memory.surprise_score))

    def _calculate_entity_score(self, memory: MemoryProtocol) -> float:
        """
        Calculate entity score based on named entities.

        Memories mentioning more entities are often more informative.
        Uses saturation to prevent over-weighting.

        Args:
            memory: The memory to score.

        Returns:
            Entity score (0-1), higher for more entities.
        """
        entity_count = len(memory.entities)
        if entity_count <= 0:
            return 0.0

        score = entity_count / self.entity_saturation
        return min(1.0, score)

    def _calculate_explicit_score(
        self,
        memory: MemoryProtocol,
        explicit_importance: float | None,
    ) -> float:
        """
        Calculate explicit importance score.

        Uses provided explicit value, or checks memory metadata.

        Args:
            memory: The memory to score.
            explicit_importance: Optional explicit importance value.

        Returns:
            Explicit score (0-1).
        """
        if explicit_importance is not None:
            return max(0.0, min(1.0, explicit_importance))

        # Check metadata for explicit importance
        metadata_importance = memory.metadata.get("importance")
        if metadata_importance is not None:
            try:
                return max(0.0, min(1.0, float(metadata_importance)))
            except (ValueError, TypeError):
                pass

        # Default to neutral
        return 0.5

    def _get_source_multiplier(self, memory: MemoryProtocol) -> float:
        """
        Get the importance multiplier for a memory's source.

        Args:
            memory: The memory to get multiplier for.

        Returns:
            Source multiplier (typically 0.5-2.0).
        """
        source_value = memory.source
        source_key = (
            source_value.value if hasattr(source_value, "value") else str(source_value)
        )
        return self.source_weights.get(source_key, 1.0)

    def get_importance(
        self,
        memory: MemoryProtocol,
        current_time: datetime | None = None,
    ) -> float:
        """
        Get the importance score for a memory.

        Convenience method that returns just the final score.

        Args:
            memory: The memory to score.
            current_time: Time for recency calculation.

        Returns:
            Importance score (0-1).
        """
        result = self.calculate_importance(memory, current_time)
        return result.final_score

    def batch_calculate_importance(
        self,
        memories: list[MemoryProtocol],
        current_time: datetime | None = None,
    ) -> list[tuple[MemoryProtocol, ImportanceResult]]:
        """
        Calculate importance for multiple memories.

        Args:
            memories: List of memories to score.
            current_time: Time for recency calculation.

        Returns:
            List of (memory, ImportanceResult) tuples.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        return [(m, self.calculate_importance(m, current_time)) for m in memories]

    def rank_by_importance(
        self,
        memories: list[MemoryProtocol],
        current_time: datetime | None = None,
        top_k: int | None = None,
    ) -> list[tuple[MemoryProtocol, float]]:
        """
        Rank memories by importance score.

        Args:
            memories: List of memories to rank.
            current_time: Time for recency calculation.
            top_k: Optional limit on results.

        Returns:
            List of (memory, score) tuples, sorted by score descending.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        scored = [(m, self.get_importance(m, current_time)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def filter_by_importance(
        self,
        memories: list[MemoryProtocol],
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        current_time: datetime | None = None,
    ) -> list[MemoryProtocol]:
        """
        Filter memories by importance range.

        Args:
            memories: List of memories to filter.
            min_importance: Minimum importance (inclusive).
            max_importance: Maximum importance (inclusive).
            current_time: Time for recency calculation.

        Returns:
            Memories with importance in the specified range.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        return [
            m
            for m in memories
            if min_importance <= self.get_importance(m, current_time) <= max_importance
        ]

"""Core memory data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


class MemoryType(str, Enum):
    """Type of memory tier."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemorySource(str, Enum):
    """Source of the memory."""

    CONVERSATION = "conversation"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    CONSOLIDATION = "consolidation"
    EXTERNAL = "external"
    USER_EXPLICIT = "user_explicit"


@dataclass
class Memory:
    """
    Core memory unit.

    Represents a single memory with decay, importance, and relationship metadata.

    Attributes:
        id: Unique identifier for the memory.
        memory_type: Type of memory (episodic, semantic, procedural).
        content: The actual content/text of the memory.
        embedding: Vector embedding of the content.
        metadata: Additional key-value metadata.
        created_at: When the memory was created.
        last_accessed_at: When the memory was last retrieved.
        access_count: Number of times the memory has been retrieved.
        strength: Current strength after decay (computed).
        initial_strength: Strength at creation, boosted by rehearsal.
        importance: Computed importance score (0-1).
        emotional_valence: Emotional tone (-1 negative to 1 positive).
        surprise_score: How unexpected the information was (0-1).
        source: Where the memory came from.
        source_id: Reference to source (conversation_id, etc.).
        agent_id: ID of the agent that owns this memory.
        user_id: ID of the user associated with this memory.
        entities: Named entities extracted from content.
        topics: Topics/themes extracted from content.
        related_memory_ids: IDs of related memories.
        parent_memory_id: ID of parent memory if derived.
        superseded_by_id: ID of memory that replaced this one.
        is_pinned: If True, memory never decays.
        is_archived: If True, memory is soft-deleted.
        is_consolidated: If True, memory has been processed by consolidation.
    """

    id: str = field(default_factory=_uuid)
    memory_type: MemoryType = MemoryType.EPISODIC
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    last_accessed_at: datetime = field(default_factory=_utcnow)
    access_count: int = 0
    strength: float = 1.0
    initial_strength: float = 1.0
    importance: float = 0.5
    emotional_valence: float = 0.0
    surprise_score: float = 0.0
    source: MemorySource = MemorySource.CONVERSATION
    source_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    related_memory_ids: list[str] = field(default_factory=list)
    parent_memory_id: str | None = None
    superseded_by_id: str | None = None
    is_pinned: bool = False
    is_archived: bool = False
    is_consolidated: bool = False


@dataclass
class ScoredMemory:
    """
    Memory with retrieval scores.

    Returned by the retrieval engine with scoring breakdown.

    Attributes:
        memory: The underlying memory object.
        similarity_score: Cosine similarity to query (0-1).
        strength_score: Current decay-adjusted strength (0-1).
        importance_score: Importance factor (0-1).
        recency_score: How recently accessed (0-1).
        final_score: Combined weighted score used for ranking.
    """

    memory: Memory
    similarity_score: float = 0.0
    strength_score: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0


@dataclass
class Fact:
    """
    Semantic memory unit - a single fact.

    Represents a subject-predicate-object triple with confidence.
    Stored in the knowledge graph.

    Attributes:
        id: Unique identifier.
        subject: The subject entity of the fact.
        predicate: The relationship/predicate.
        object: The object entity of the fact.
        confidence: Confidence score (0-1).
        source_memory_ids: IDs of episodic memories this was extracted from.
        created_at: When the fact was created.
        last_verified_at: When the fact was last confirmed.
        verification_count: Number of times the fact was verified.
        contradicted_by: IDs of facts that contradict this one.
    """

    id: str = field(default_factory=_uuid)
    subject: str = ""
    predicate: str = ""
    object: str = ""
    confidence: float = 1.0
    source_memory_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    last_verified_at: datetime = field(default_factory=_utcnow)
    verification_count: int = 1
    contradicted_by: list[str] = field(default_factory=list)

    def as_triple(self) -> str:
        """Return the fact as a readable triple string."""
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass
class Entity:
    """
    Named entity in semantic memory.

    Represents a person, organization, concept, or other named entity.

    Attributes:
        id: Unique identifier.
        name: Canonical name of the entity.
        entity_type: Type (person, organization, concept, location, etc.).
        aliases: Alternative names for the entity.
        attributes: Key-value attributes of the entity.
        first_seen: When the entity was first mentioned.
        last_seen: When the entity was most recently mentioned.
        mention_count: Total number of mentions.
    """

    id: str = field(default_factory=_uuid)
    name: str = ""
    entity_type: str = ""
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=_utcnow)
    last_seen: datetime = field(default_factory=_utcnow)
    mention_count: int = 1


@dataclass
class Relationship:
    """
    Relationship between entities in the knowledge graph.

    Attributes:
        id: Unique identifier.
        source_entity_id: ID of the source entity.
        target_entity_id: ID of the target entity.
        relationship_type: Type of relationship.
        properties: Additional properties of the relationship.
        confidence: Confidence score (0-1).
        source_memory_ids: IDs of memories this was extracted from.
    """

    id: str = field(default_factory=_uuid)
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_memory_ids: list[str] = field(default_factory=list)


@dataclass
class Procedure:
    """
    Procedural memory unit.

    Represents a learned procedure, skill, or pattern.
    Does not decay - only explicit updates/deletions.

    Attributes:
        id: Unique identifier.
        name: Name of the procedure.
        description: What the procedure does.
        steps: Ordered list of steps.
        preconditions: Conditions that must be true before execution.
        postconditions: Expected outcomes after execution.
        success_count: Number of successful executions.
        failure_count: Number of failed executions.
        last_used: When the procedure was last used.
    """

    id: str = field(default_factory=_uuid)
    name: str = ""
    description: str = ""
    steps: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this procedure."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class ToolPattern:
    """
    Learned tool usage pattern.

    Tracks how tools are used and their success rates.

    Attributes:
        id: Unique identifier.
        tool_name: Name of the tool.
        input_pattern: Common input patterns.
        expected_output_type: Expected type of output.
        success_rate: Historical success rate (0-1).
        avg_latency_ms: Average execution time in milliseconds.
        failure_modes: Known failure modes.
        usage_count: Total number of uses.
    """

    id: str = field(default_factory=_uuid)
    tool_name: str = ""
    input_pattern: dict[str, Any] = field(default_factory=dict)
    expected_output_type: str = ""
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    failure_modes: list[str] = field(default_factory=list)
    usage_count: int = 0

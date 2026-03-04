"""Tests for core memory models."""

from datetime import timezone

from cognitive_memory.core.memory import (
    Entity,
    Fact,
    Memory,
    MemorySource,
    MemoryType,
    Procedure,
    Relationship,
    ScoredMemory,
    ToolPattern,
)


class TestMemory:
    """Tests for the Memory dataclass."""

    def test_default_values(self) -> None:
        """Memory should have sensible defaults."""
        memory = Memory()

        assert memory.id is not None
        assert len(memory.id) == 36  # UUID format
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.content == ""
        assert memory.embedding == []
        assert memory.metadata == {}
        assert memory.strength == 1.0
        assert memory.importance == 0.5
        assert memory.is_pinned is False
        assert memory.is_archived is False

    def test_custom_values(self) -> None:
        """Memory should accept custom values."""
        memory = Memory(
            id="test-id",
            memory_type=MemoryType.SEMANTIC,
            content="Test content",
            importance=0.9,
            is_pinned=True,
        )

        assert memory.id == "test-id"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.content == "Test content"
        assert memory.importance == 0.9
        assert memory.is_pinned is True

    def test_timestamps_are_utc(self) -> None:
        """Timestamps should be in UTC."""
        memory = Memory()

        assert memory.created_at.tzinfo == timezone.utc
        assert memory.last_accessed_at.tzinfo == timezone.utc

    def test_memory_types(self) -> None:
        """All memory types should be valid."""
        for mem_type in MemoryType:
            memory = Memory(memory_type=mem_type)
            assert memory.memory_type == mem_type

    def test_memory_sources(self) -> None:
        """All memory sources should be valid."""
        for source in MemorySource:
            memory = Memory(source=source)
            assert memory.source == source


class TestScoredMemory:
    """Tests for the ScoredMemory dataclass."""

    def test_default_scores(self) -> None:
        """ScoredMemory should have zero default scores."""
        memory = Memory(content="Test")
        scored = ScoredMemory(memory=memory)

        assert scored.memory == memory
        assert scored.similarity_score == 0.0
        assert scored.strength_score == 0.0
        assert scored.importance_score == 0.0
        assert scored.recency_score == 0.0
        assert scored.final_score == 0.0

    def test_custom_scores(self) -> None:
        """ScoredMemory should accept custom scores."""
        memory = Memory(content="Test")
        scored = ScoredMemory(
            memory=memory,
            similarity_score=0.8,
            strength_score=0.9,
            importance_score=0.7,
            recency_score=0.6,
            final_score=0.75,
        )

        assert scored.similarity_score == 0.8
        assert scored.strength_score == 0.9
        assert scored.importance_score == 0.7
        assert scored.recency_score == 0.6
        assert scored.final_score == 0.75


class TestFact:
    """Tests for the Fact dataclass."""

    def test_default_values(self) -> None:
        """Fact should have sensible defaults."""
        fact = Fact()

        assert fact.id is not None
        assert fact.subject == ""
        assert fact.predicate == ""
        assert fact.object == ""
        assert fact.confidence == 1.0
        assert fact.source_memory_ids == []
        assert fact.verification_count == 1

    def test_as_triple(self) -> None:
        """Fact should render as a triple string."""
        fact = Fact(
            subject="Alice",
            predicate="works_at",
            object="Acme Corp",
        )

        assert fact.as_triple() == "Alice works_at Acme Corp"

    def test_custom_values(self) -> None:
        """Fact should accept custom values."""
        fact = Fact(
            subject="Bob",
            predicate="knows",
            object="Python",
            confidence=0.95,
            source_memory_ids=["mem-1", "mem-2"],
        )

        assert fact.subject == "Bob"
        assert fact.predicate == "knows"
        assert fact.object == "Python"
        assert fact.confidence == 0.95
        assert fact.source_memory_ids == ["mem-1", "mem-2"]


class TestEntity:
    """Tests for the Entity dataclass."""

    def test_default_values(self) -> None:
        """Entity should have sensible defaults."""
        entity = Entity()

        assert entity.id is not None
        assert entity.name == ""
        assert entity.entity_type == ""
        assert entity.aliases == []
        assert entity.attributes == {}
        assert entity.mention_count == 1

    def test_custom_values(self) -> None:
        """Entity should accept custom values."""
        entity = Entity(
            name="John Doe",
            entity_type="person",
            aliases=["JD", "Johnny"],
            attributes={"role": "engineer"},
        )

        assert entity.name == "John Doe"
        assert entity.entity_type == "person"
        assert entity.aliases == ["JD", "Johnny"]
        assert entity.attributes["role"] == "engineer"


class TestRelationship:
    """Tests for the Relationship dataclass."""

    def test_default_values(self) -> None:
        """Relationship should have sensible defaults."""
        rel = Relationship()

        assert rel.id is not None
        assert rel.source_entity_id == ""
        assert rel.target_entity_id == ""
        assert rel.relationship_type == ""
        assert rel.properties == {}
        assert rel.confidence == 1.0

    def test_custom_values(self) -> None:
        """Relationship should accept custom values."""
        rel = Relationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="manages",
            confidence=0.9,
        )

        assert rel.source_entity_id == "entity-1"
        assert rel.target_entity_id == "entity-2"
        assert rel.relationship_type == "manages"
        assert rel.confidence == 0.9


class TestProcedure:
    """Tests for the Procedure dataclass."""

    def test_default_values(self) -> None:
        """Procedure should have sensible defaults."""
        proc = Procedure()

        assert proc.id is not None
        assert proc.name == ""
        assert proc.description == ""
        assert proc.steps == []
        assert proc.success_count == 0
        assert proc.failure_count == 0
        assert proc.last_used is None

    def test_success_rate_no_executions(self) -> None:
        """Success rate should be 0 with no executions."""
        proc = Procedure()
        assert proc.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Success rate should be calculated correctly."""
        proc = Procedure(success_count=8, failure_count=2)
        assert proc.success_rate == 0.8

    def test_success_rate_all_success(self) -> None:
        """Success rate should be 1.0 with all successes."""
        proc = Procedure(success_count=10, failure_count=0)
        assert proc.success_rate == 1.0

    def test_success_rate_all_failure(self) -> None:
        """Success rate should be 0.0 with all failures."""
        proc = Procedure(success_count=0, failure_count=10)
        assert proc.success_rate == 0.0


class TestToolPattern:
    """Tests for the ToolPattern dataclass."""

    def test_default_values(self) -> None:
        """ToolPattern should have sensible defaults."""
        pattern = ToolPattern()

        assert pattern.id is not None
        assert pattern.tool_name == ""
        assert pattern.input_pattern == {}
        assert pattern.expected_output_type == ""
        assert pattern.success_rate == 0.0
        assert pattern.avg_latency_ms == 0.0
        assert pattern.failure_modes == []
        assert pattern.usage_count == 0

    def test_custom_values(self) -> None:
        """ToolPattern should accept custom values."""
        pattern = ToolPattern(
            tool_name="search",
            input_pattern={"query": "string"},
            expected_output_type="list",
            success_rate=0.95,
            avg_latency_ms=150.0,
            failure_modes=["timeout", "rate_limit"],
            usage_count=100,
        )

        assert pattern.tool_name == "search"
        assert pattern.input_pattern == {"query": "string"}
        assert pattern.expected_output_type == "list"
        assert pattern.success_rate == 0.95
        assert pattern.avg_latency_ms == 150.0
        assert pattern.failure_modes == ["timeout", "rate_limit"]
        assert pattern.usage_count == 100

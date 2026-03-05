"""Tests for SQLite metadata backend."""

from datetime import datetime, timezone

import pytest

from cognitive_memory.storage.metadata.sqlite import SQLiteMetadataBackend


@pytest.fixture
async def backend():
    """Create an in-memory SQLite backend for each test."""
    be = SQLiteMetadataBackend(db_path=":memory:")
    await be.initialize()
    yield be
    await be.close()


def _make_memory(
    memory_id: str = "mem-1",
    content: str = "Test content",
    **overrides: object,
) -> dict:
    """Helper to build a memory dict with sensible defaults."""
    base: dict = {
        "id": memory_id,
        "memory_type": "episodic",
        "content": content,
        "metadata": {"key": "value"},
        "agent_id": "agent-1",
        "user_id": "user-1",
        "source": "conversation",
        "source_id": None,
        "strength": 1.0,
        "initial_strength": 1.0,
        "importance": 0.5,
        "emotional_valence": 0.0,
        "surprise_score": 0.0,
        "access_count": 0,
        "entities": ["Alice"],
        "topics": ["work"],
        "related_memory_ids": [],
        "parent_memory_id": None,
        "superseded_by_id": None,
        "is_pinned": False,
        "is_archived": False,
        "is_consolidated": False,
    }
    base.update(overrides)
    return base


class TestSQLiteMetadataBackendInit:
    """Tests for initialization and lifecycle."""

    def test_default_values(self) -> None:
        backend = SQLiteMetadataBackend()

        assert backend.db_path == "cognitive_memory.db"
        assert backend.table_name == "memories"

    def test_custom_values(self) -> None:
        backend = SQLiteMetadataBackend(db_path="/tmp/test.db", table_name="custom")

        assert backend.db_path == "/tmp/test.db"
        assert backend.table_name == "custom"

    def test_initial_state(self) -> None:
        backend = SQLiteMetadataBackend()

        assert backend._conn is None
        assert backend._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_creates_table(self) -> None:
        backend = SQLiteMetadataBackend(db_path=":memory:")
        await backend.initialize()

        assert backend._initialized is True
        assert backend._conn is not None
        await backend.close()

    @pytest.mark.asyncio
    async def test_double_initialize_is_noop(self) -> None:
        backend = SQLiteMetadataBackend(db_path=":memory:")
        await backend.initialize()
        conn_first = backend._conn
        await backend.initialize()

        assert backend._conn is conn_first
        await backend.close()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        backend = SQLiteMetadataBackend(db_path=":memory:")
        await backend.initialize()
        await backend.close()

        assert backend._conn is None
        assert backend._initialized is False

    @pytest.mark.asyncio
    async def test_close_without_init_is_safe(self) -> None:
        backend = SQLiteMetadataBackend(db_path=":memory:")
        await backend.close()

        assert backend._conn is None


class TestSQLiteSaveAndGet:
    """Tests for save_memory and get_memory."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, backend: SQLiteMetadataBackend) -> None:
        mem = _make_memory()
        await backend.save_memory(mem)

        result = await backend.get_memory("mem-1")

        assert result is not None
        assert result["id"] == "mem-1"
        assert result["content"] == "Test content"
        assert result["memory_type"] == "episodic"
        assert result["metadata"] == {"key": "value"}
        assert result["entities"] == ["Alice"]
        assert result["topics"] == ["work"]
        assert result["agent_id"] == "agent-1"
        assert result["user_id"] == "user-1"
        assert result["strength"] == 1.0

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, backend: SQLiteMetadataBackend) -> None:
        result = await backend.get_memory("does-not-exist")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_without_id_raises(self, backend: SQLiteMetadataBackend) -> None:
        with pytest.raises(ValueError, match="must have an 'id' field"):
            await backend.save_memory({"content": "no id"})

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(content="version 1"))
        await backend.save_memory(_make_memory(content="version 2"))

        result = await backend.get_memory("mem-1")

        assert result is not None
        assert result["content"] == "version 2"

    @pytest.mark.asyncio
    async def test_save_preserves_boolean_fields(self, backend: SQLiteMetadataBackend) -> None:
        mem = _make_memory(is_pinned=True, is_archived=True, is_consolidated=True)
        await backend.save_memory(mem)

        result = await backend.get_memory("mem-1")

        assert result is not None
        assert result["is_pinned"] is True
        assert result["is_archived"] is True
        assert result["is_consolidated"] is True

    @pytest.mark.asyncio
    async def test_save_with_minimal_fields(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory({"id": "minimal-1"})

        result = await backend.get_memory("minimal-1")

        assert result is not None
        assert result["content"] == ""
        assert result["memory_type"] == "episodic"
        assert result["strength"] == 1.0
        assert result["metadata"] == {}
        assert result["entities"] == []


class TestSQLiteDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory())

        result = await backend.delete_memory("mem-1")

        assert result is True
        assert await backend.get_memory("mem-1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend: SQLiteMetadataBackend) -> None:
        result = await backend.delete_memory("does-not-exist")

        assert result is False

    @pytest.mark.asyncio
    async def test_batch_delete(self, backend: SQLiteMetadataBackend) -> None:
        for i in range(5):
            await backend.save_memory(_make_memory(memory_id=f"mem-{i}"))

        deleted = await backend.batch_delete(["mem-0", "mem-1", "mem-2"])

        assert deleted == 3
        assert await backend.get_memory("mem-0") is None
        assert await backend.get_memory("mem-3") is not None

    @pytest.mark.asyncio
    async def test_batch_delete_empty(self, backend: SQLiteMetadataBackend) -> None:
        assert await backend.batch_delete([]) == 0

    @pytest.mark.asyncio
    async def test_batch_delete_partial_match(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="exists"))

        deleted = await backend.batch_delete(["exists", "ghost"])

        assert deleted == 1


class TestSQLiteListAndCount:
    """Tests for list_memories and count."""

    @pytest.mark.asyncio
    async def test_list_all(self, backend: SQLiteMetadataBackend) -> None:
        for i in range(3):
            await backend.save_memory(_make_memory(memory_id=f"mem-{i}"))

        result = await backend.list_memories()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_empty(self, backend: SQLiteMetadataBackend) -> None:
        result = await backend.list_memories()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_filter_by_agent(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="a1", agent_id="agent-A"))
        await backend.save_memory(_make_memory(memory_id="a2", agent_id="agent-A"))
        await backend.save_memory(_make_memory(memory_id="b1", agent_id="agent-B"))

        result = await backend.list_memories(agent_id="agent-A")

        assert len(result) == 2
        assert all(m["agent_id"] == "agent-A" for m in result)

    @pytest.mark.asyncio
    async def test_list_filter_by_user(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="u1", user_id="user-X"))
        await backend.save_memory(_make_memory(memory_id="u2", user_id="user-Y"))

        result = await backend.list_memories(user_id="user-X")

        assert len(result) == 1
        assert result[0]["user_id"] == "user-X"

    @pytest.mark.asyncio
    async def test_list_filter_by_type(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="e1", memory_type="episodic"))
        await backend.save_memory(_make_memory(memory_id="s1", memory_type="semantic"))

        result = await backend.list_memories(memory_type="semantic")

        assert len(result) == 1
        assert result[0]["memory_type"] == "semantic"

    @pytest.mark.asyncio
    async def test_list_filter_by_archived(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="active", is_archived=False))
        await backend.save_memory(_make_memory(memory_id="archived", is_archived=True))

        result = await backend.list_memories(is_archived=False)

        assert len(result) == 1
        assert result[0]["id"] == "active"

    @pytest.mark.asyncio
    async def test_list_with_limit_and_offset(self, backend: SQLiteMetadataBackend) -> None:
        for i in range(10):
            await backend.save_memory(_make_memory(memory_id=f"mem-{i:02d}"))

        result = await backend.list_memories(limit=3, offset=0)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_count_all(self, backend: SQLiteMetadataBackend) -> None:
        for i in range(5):
            await backend.save_memory(_make_memory(memory_id=f"mem-{i}"))

        assert await backend.count() == 5

    @pytest.mark.asyncio
    async def test_count_empty(self, backend: SQLiteMetadataBackend) -> None:
        assert await backend.count() == 0

    @pytest.mark.asyncio
    async def test_count_with_filters(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="a1", agent_id="agent-A"))
        await backend.save_memory(_make_memory(memory_id="a2", agent_id="agent-A"))
        await backend.save_memory(_make_memory(memory_id="b1", agent_id="agent-B"))

        assert await backend.count(agent_id="agent-A") == 2
        assert await backend.count(agent_id="agent-B") == 1


class TestSQLiteUpdateAccess:
    """Tests for update_access."""

    @pytest.mark.asyncio
    async def test_update_access_increments_count(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory())
        await backend.update_access("mem-1")
        await backend.update_access("mem-1")

        result = await backend.get_memory("mem-1")

        assert result is not None
        assert result["access_count"] == 2

    @pytest.mark.asyncio
    async def test_update_access_with_custom_timestamp(
        self, backend: SQLiteMetadataBackend
    ) -> None:
        await backend.save_memory(_make_memory())
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        await backend.update_access("mem-1", accessed_at=ts)

        result = await backend.get_memory("mem-1")

        assert result is not None
        assert result["access_count"] == 1
        assert "2026-01-15" in result["last_accessed_at"]


class TestSQLiteBatchSave:
    """Tests for batch_save."""

    @pytest.mark.asyncio
    async def test_batch_save_multiple(self, backend: SQLiteMetadataBackend) -> None:
        memories = [_make_memory(memory_id=f"mem-{i}") for i in range(5)]

        count = await backend.batch_save(memories)

        assert count == 5
        assert await backend.count() == 5

    @pytest.mark.asyncio
    async def test_batch_save_empty(self, backend: SQLiteMetadataBackend) -> None:
        assert await backend.batch_save([]) == 0


class TestSQLiteGetByIds:
    """Tests for get_memories_by_ids."""

    @pytest.mark.asyncio
    async def test_get_by_ids(self, backend: SQLiteMetadataBackend) -> None:
        for i in range(5):
            await backend.save_memory(_make_memory(memory_id=f"mem-{i}"))

        result = await backend.get_memories_by_ids(["mem-1", "mem-3"])

        assert len(result) == 2
        assert result[0]["id"] == "mem-1"
        assert result[1]["id"] == "mem-3"

    @pytest.mark.asyncio
    async def test_get_by_ids_preserves_order(self, backend: SQLiteMetadataBackend) -> None:
        for i in range(3):
            await backend.save_memory(_make_memory(memory_id=f"mem-{i}"))

        result = await backend.get_memories_by_ids(["mem-2", "mem-0", "mem-1"])

        assert [m["id"] for m in result] == ["mem-2", "mem-0", "mem-1"]

    @pytest.mark.asyncio
    async def test_get_by_ids_empty(self, backend: SQLiteMetadataBackend) -> None:
        assert await backend.get_memories_by_ids([]) == []

    @pytest.mark.asyncio
    async def test_get_by_ids_partial_match(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="exists"))

        result = await backend.get_memories_by_ids(["exists", "ghost"])

        assert len(result) == 1
        assert result[0]["id"] == "exists"


class TestSQLiteSearchByContent:
    """Tests for search_by_content."""

    @pytest.mark.asyncio
    async def test_search_finds_match(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(memory_id="m1", content="User prefers dark mode"))
        await backend.save_memory(_make_memory(memory_id="m2", content="Meeting at 3pm"))

        result = await backend.search_by_content("dark mode")

        assert len(result) == 1
        assert result[0]["id"] == "m1"

    @pytest.mark.asyncio
    async def test_search_no_match(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(content="something else"))

        result = await backend.search_by_content("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_search_with_agent_filter(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(
            _make_memory(memory_id="m1", content="dark mode", agent_id="agent-A")
        )
        await backend.save_memory(
            _make_memory(memory_id="m2", content="dark mode", agent_id="agent-B")
        )

        result = await backend.search_by_content("dark mode", agent_id="agent-A")

        assert len(result) == 1
        assert result[0]["agent_id"] == "agent-A"


class TestSQLiteGetRelatedMemories:
    """Tests for get_related_memories."""

    @pytest.mark.asyncio
    async def test_get_related(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(
            _make_memory(memory_id="parent", related_memory_ids=["child-1", "child-2"])
        )
        await backend.save_memory(_make_memory(memory_id="child-1", content="Child 1"))
        await backend.save_memory(_make_memory(memory_id="child-2", content="Child 2"))

        result = await backend.get_related_memories("parent")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_related_nonexistent_parent(self, backend: SQLiteMetadataBackend) -> None:
        result = await backend.get_related_memories("ghost")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_related_no_relations(self, backend: SQLiteMetadataBackend) -> None:
        await backend.save_memory(_make_memory(related_memory_ids=[]))

        result = await backend.get_related_memories("mem-1")

        assert result == []

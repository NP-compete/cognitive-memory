"""Tests for REST API endpoints."""

import pytest

# Skip all tests if FastAPI not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cognitive_memory.api.app import create_app
from cognitive_memory.api.routes import memories


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_memories() -> None:
    """Clear in-memory storage before each test."""
    memories._memories.clear()


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_readiness_check(self, client: TestClient) -> None:
        """Readiness endpoint should return ready status."""
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True


class TestCreateMemory:
    """Tests for memory creation."""

    def test_create_memory(self, client: TestClient) -> None:
        """Should create a memory."""
        response = client.post(
            "/api/v1/memories",
            json={"content": "Test memory content"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "Test memory content"
        assert "id" in data
        assert data["memory_type"] == "episodic"

    def test_create_memory_with_all_fields(self, client: TestClient) -> None:
        """Should create memory with all fields."""
        response = client.post(
            "/api/v1/memories",
            json={
                "content": "Full memory",
                "memory_type": "semantic",
                "agent_id": "agent-1",
                "user_id": "user-1",
                "importance": 0.8,
                "emotional_valence": 0.5,
                "entities": ["Alice", "Bob"],
                "topics": ["work"],
                "metadata": {"custom": "data"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["memory_type"] == "semantic"
        assert data["agent_id"] == "agent-1"
        assert data["importance"] == 0.8
        assert data["entities"] == ["Alice", "Bob"]


class TestGetMemory:
    """Tests for getting a memory."""

    def test_get_memory(self, client: TestClient) -> None:
        """Should get a memory by ID."""
        # Create first
        create_resp = client.post(
            "/api/v1/memories",
            json={"content": "Test content"},
        )
        memory_id = create_resp.json()["id"]

        # Get
        response = client.get(f"/api/v1/memories/{memory_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["access_count"] == 1  # Incremented on access

    def test_get_memory_not_found(self, client: TestClient) -> None:
        """Should return 404 for nonexistent memory."""
        response = client.get("/api/v1/memories/nonexistent-id")

        assert response.status_code == 404


class TestUpdateMemory:
    """Tests for updating a memory."""

    def test_update_memory(self, client: TestClient) -> None:
        """Should update a memory."""
        # Create first
        create_resp = client.post(
            "/api/v1/memories",
            json={"content": "Original content"},
        )
        memory_id = create_resp.json()["id"]

        # Update
        response = client.patch(
            f"/api/v1/memories/{memory_id}",
            json={"content": "Updated content", "is_pinned": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Updated content"
        assert data["is_pinned"] is True

    def test_update_memory_not_found(self, client: TestClient) -> None:
        """Should return 404 for nonexistent memory."""
        response = client.patch(
            "/api/v1/memories/nonexistent-id",
            json={"content": "New content"},
        )

        assert response.status_code == 404


class TestDeleteMemory:
    """Tests for deleting a memory."""

    def test_delete_memory(self, client: TestClient) -> None:
        """Should delete a memory."""
        # Create first
        create_resp = client.post(
            "/api/v1/memories",
            json={"content": "To be deleted"},
        )
        memory_id = create_resp.json()["id"]

        # Delete
        response = client.delete(f"/api/v1/memories/{memory_id}")

        assert response.status_code == 204

        # Verify deleted
        get_resp = client.get(f"/api/v1/memories/{memory_id}")
        assert get_resp.status_code == 404

    def test_delete_memory_not_found(self, client: TestClient) -> None:
        """Should return 404 for nonexistent memory."""
        response = client.delete("/api/v1/memories/nonexistent-id")

        assert response.status_code == 404


class TestListMemories:
    """Tests for listing memories."""

    def test_list_memories_empty(self, client: TestClient) -> None:
        """Should return empty list when no memories."""
        response = client.get("/api/v1/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["memories"] == []
        assert data["total"] == 0

    def test_list_memories(self, client: TestClient) -> None:
        """Should list all memories."""
        # Create some memories
        client.post("/api/v1/memories", json={"content": "Memory 1"})
        client.post("/api/v1/memories", json={"content": "Memory 2"})

        response = client.get("/api/v1/memories")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2

    def test_list_memories_with_filter(self, client: TestClient) -> None:
        """Should filter memories."""
        client.post(
            "/api/v1/memories",
            json={"content": "Agent 1 memory", "agent_id": "agent-1"},
        )
        client.post(
            "/api/v1/memories",
            json={"content": "Agent 2 memory", "agent_id": "agent-2"},
        )

        response = client.get("/api/v1/memories?agent_id=agent-1")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["memories"][0]["agent_id"] == "agent-1"

    def test_list_memories_pagination(self, client: TestClient) -> None:
        """Should paginate results."""
        for i in range(5):
            client.post("/api/v1/memories", json={"content": f"Memory {i}"})

        response = client.get("/api/v1/memories?limit=2&offset=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["memories"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 2


class TestSearchMemories:
    """Tests for searching memories."""

    def test_search_memories(self, client: TestClient) -> None:
        """Should search memories by content."""
        client.post("/api/v1/memories", json={"content": "The quick brown fox"})
        client.post("/api/v1/memories", json={"content": "A lazy dog"})

        response = client.post(
            "/api/v1/memories/search",
            json={"query": "fox"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "fox" in data["results"][0]["memory"]["content"]

    def test_search_memories_no_results(self, client: TestClient) -> None:
        """Should return empty when no matches."""
        client.post("/api/v1/memories", json={"content": "Some content"})

        response = client.post(
            "/api/v1/memories/search",
            json={"query": "nonexistent"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["results"] == []


class TestStats:
    """Tests for statistics endpoint."""

    def test_stats_empty(self, client: TestClient) -> None:
        """Should return zero stats when empty."""
        response = client.get("/api/v1/memories/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_memories"] == 0

    def test_stats(self, client: TestClient) -> None:
        """Should return correct statistics."""
        client.post(
            "/api/v1/memories",
            json={"content": "Memory 1", "memory_type": "episodic"},
        )
        client.post(
            "/api/v1/memories",
            json={"content": "Memory 2", "memory_type": "semantic"},
        )

        response = client.get("/api/v1/memories/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_memories"] == 2
        assert data["by_type"]["episodic"] == 1
        assert data["by_type"]["semantic"] == 1

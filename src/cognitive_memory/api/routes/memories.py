"""Memory CRUD and search endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

try:
    from fastapi import APIRouter, HTTPException, Query
except ImportError as e:
    raise ImportError("FastAPI is required. Install with: pip install cognitive-memory[api]") from e

from cognitive_memory.api.models import (
    ErrorResponse,
    MemoryCreate,
    MemoryListResponse,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchResult,
    MemoryUpdate,
    StatsResponse,
)

router = APIRouter()

# In-memory storage for demo (replace with actual backend in production)
_memories: dict[str, dict[str, Any]] = {}


def _memory_to_response(memory: dict[str, Any]) -> MemoryResponse:
    """Convert internal memory dict to response model."""
    return MemoryResponse(
        id=memory["id"],
        memory_type=memory.get("memory_type", "episodic"),
        content=memory.get("content", ""),
        agent_id=memory.get("agent_id"),
        user_id=memory.get("user_id"),
        source=memory.get("source", "api"),
        strength=memory.get("strength", 1.0),
        importance=memory.get("importance", 0.5),
        emotional_valence=memory.get("emotional_valence", 0.0),
        access_count=memory.get("access_count", 0),
        entities=memory.get("entities", []),
        topics=memory.get("topics", []),
        is_pinned=memory.get("is_pinned", False),
        is_archived=memory.get("is_archived", False),
        created_at=memory.get("created_at", datetime.now(timezone.utc)),
        last_accessed_at=memory.get("last_accessed_at", datetime.now(timezone.utc)),
        metadata=memory.get("metadata", {}),
    )


@router.post(
    "/memories",
    response_model=MemoryResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}},
)
async def create_memory(request: MemoryCreate) -> MemoryResponse:
    """
    Create a new memory.

    Args:
        request: Memory creation request.

    Returns:
        Created memory.
    """
    now = datetime.now(timezone.utc)
    memory_id = str(uuid4())

    memory = {
        "id": memory_id,
        "memory_type": request.memory_type,
        "content": request.content,
        "agent_id": request.agent_id,
        "user_id": request.user_id,
        "source": request.source,
        "strength": 1.0,
        "initial_strength": 1.0,
        "importance": request.importance if request.importance is not None else 0.5,
        "emotional_valence": request.emotional_valence,
        "access_count": 0,
        "entities": request.entities,
        "topics": request.topics,
        "is_pinned": False,
        "is_archived": False,
        "is_consolidated": False,
        "created_at": now,
        "last_accessed_at": now,
        "metadata": request.metadata,
    }

    _memories[memory_id] = memory
    return _memory_to_response(memory)


@router.get(
    "/memories",
    response_model=MemoryListResponse,
)
async def list_memories(
    agent_id: str | None = Query(default=None, description="Filter by agent"),
    user_id: str | None = Query(default=None, description="Filter by user"),
    memory_type: str | None = Query(default=None, description="Filter by type"),
    is_archived: bool | None = Query(default=None, description="Filter by archived"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> MemoryListResponse:
    """
    List memories with optional filters.

    Args:
        agent_id: Filter by agent ID.
        user_id: Filter by user ID.
        memory_type: Filter by memory type.
        is_archived: Filter by archived status.
        limit: Maximum results to return.
        offset: Pagination offset.

    Returns:
        List of memories.
    """
    filtered = list(_memories.values())

    if agent_id is not None:
        filtered = [m for m in filtered if m.get("agent_id") == agent_id]
    if user_id is not None:
        filtered = [m for m in filtered if m.get("user_id") == user_id]
    if memory_type is not None:
        filtered = [m for m in filtered if m.get("memory_type") == memory_type]
    if is_archived is not None:
        filtered = [m for m in filtered if m.get("is_archived") == is_archived]

    # Sort by created_at descending
    filtered.sort(key=lambda m: m.get("created_at", datetime.min), reverse=True)

    total = len(filtered)
    paginated = filtered[offset : offset + limit]

    return MemoryListResponse(
        memories=[_memory_to_response(m) for m in paginated],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/memories/search",
    response_model=MemorySearchResponse,
)
async def search_memories(request: MemorySearchRequest) -> MemorySearchResponse:
    """
    Search memories by query.

    Note: This is a simplified text-based search.
    Production should use vector similarity search.

    Args:
        request: Search request.

    Returns:
        Search results.
    """
    filtered = list(_memories.values())

    # Apply filters
    if request.agent_id is not None:
        filtered = [m for m in filtered if m.get("agent_id") == request.agent_id]
    if request.user_id is not None:
        filtered = [m for m in filtered if m.get("user_id") == request.user_id]
    if request.memory_type is not None:
        filtered = [m for m in filtered if m.get("memory_type") == request.memory_type]
    if not request.include_archived:
        filtered = [m for m in filtered if not m.get("is_archived", False)]

    # Simple text matching (replace with vector search in production)
    query_lower = request.query.lower()
    scored = []
    for memory in filtered:
        content = memory.get("content", "").lower()
        if query_lower in content:
            # Simple relevance score based on position and length
            pos = content.find(query_lower)
            score = 1.0 - (pos / max(len(content), 1))
            scored.append((memory, score))

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    top_results = scored[: request.top_k]

    results = [
        MemorySearchResult(
            memory=_memory_to_response(m),
            score=s,
            similarity=s,
        )
        for m, s in top_results
    ]

    return MemorySearchResponse(
        results=results,
        total=len(results),
        query=request.query,
    )


@router.get(
    "/memories/stats",
    response_model=StatsResponse,
)
async def get_stats(
    agent_id: str | None = Query(default=None, description="Filter by agent"),
) -> StatsResponse:
    """
    Get memory statistics.

    Args:
        agent_id: Optional agent filter.

    Returns:
        Memory statistics.
    """
    filtered = list(_memories.values())

    if agent_id is not None:
        filtered = [m for m in filtered if m.get("agent_id") == agent_id]

    if not filtered:
        return StatsResponse(
            total_memories=0,
            by_type={},
            by_source={},
            average_strength=0.0,
            average_importance=0.0,
            pinned_count=0,
            archived_count=0,
        )

    by_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    total_strength = 0.0
    total_importance = 0.0
    pinned_count = 0
    archived_count = 0

    for memory in filtered:
        mem_type = memory.get("memory_type", "unknown")
        by_type[mem_type] = by_type.get(mem_type, 0) + 1

        source = memory.get("source", "unknown")
        by_source[source] = by_source.get(source, 0) + 1

        total_strength += memory.get("strength", 1.0)
        total_importance += memory.get("importance", 0.5)

        if memory.get("is_pinned", False):
            pinned_count += 1
        if memory.get("is_archived", False):
            archived_count += 1

    return StatsResponse(
        total_memories=len(filtered),
        by_type=by_type,
        by_source=by_source,
        average_strength=total_strength / len(filtered),
        average_importance=total_importance / len(filtered),
        pinned_count=pinned_count,
        archived_count=archived_count,
    )


# Routes with path parameters must come after static routes
@router.get(
    "/memories/{memory_id}",
    response_model=MemoryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_memory(memory_id: str) -> MemoryResponse:
    """
    Get a memory by ID.

    Args:
        memory_id: Memory identifier.

    Returns:
        Memory details.

    Raises:
        HTTPException: If memory not found.
    """
    if memory_id not in _memories:
        raise HTTPException(status_code=404, detail="Memory not found")

    memory = _memories[memory_id]
    memory["access_count"] += 1
    memory["last_accessed_at"] = datetime.now(timezone.utc)

    return _memory_to_response(memory)


@router.patch(
    "/memories/{memory_id}",
    response_model=MemoryResponse,
    responses={404: {"model": ErrorResponse}},
)
async def update_memory(memory_id: str, request: MemoryUpdate) -> MemoryResponse:
    """
    Update a memory.

    Args:
        memory_id: Memory identifier.
        request: Update request.

    Returns:
        Updated memory.

    Raises:
        HTTPException: If memory not found.
    """
    if memory_id not in _memories:
        raise HTTPException(status_code=404, detail="Memory not found")

    memory = _memories[memory_id]

    if request.content is not None:
        memory["content"] = request.content
    if request.importance is not None:
        memory["importance"] = request.importance
    if request.is_pinned is not None:
        memory["is_pinned"] = request.is_pinned
    if request.is_archived is not None:
        memory["is_archived"] = request.is_archived
    if request.entities is not None:
        memory["entities"] = request.entities
    if request.topics is not None:
        memory["topics"] = request.topics
    if request.metadata is not None:
        memory["metadata"] = request.metadata

    return _memory_to_response(memory)


@router.delete(
    "/memories/{memory_id}",
    status_code=204,
    responses={404: {"model": ErrorResponse}},
)
async def delete_memory(memory_id: str) -> None:
    """
    Delete a memory.

    Args:
        memory_id: Memory identifier.

    Raises:
        HTTPException: If memory not found.
    """
    if memory_id not in _memories:
        raise HTTPException(status_code=404, detail="Memory not found")

    del _memories[memory_id]

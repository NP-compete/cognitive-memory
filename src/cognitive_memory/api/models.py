"""Pydantic models for API requests and responses."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "Pydantic is required. Install with: pip install cognitive-memory[api]"
    ) from e


class MemoryCreate(BaseModel):
    """Request model for creating a memory."""

    content: str = Field(..., description="Memory content text")
    memory_type: str = Field(default="episodic", description="Type of memory")
    agent_id: str | None = Field(default=None, description="Agent identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    source: str = Field(default="api", description="Memory source")
    importance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Explicit importance score"
    )
    emotional_valence: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Emotional valence"
    )
    entities: list[str] = Field(default_factory=list, description="Named entities")
    topics: list[str] = Field(default_factory=list, description="Topics")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class MemoryUpdate(BaseModel):
    """Request model for updating a memory."""

    content: str | None = Field(default=None, description="Updated content")
    importance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Updated importance"
    )
    is_pinned: bool | None = Field(default=None, description="Pin status")
    is_archived: bool | None = Field(default=None, description="Archive status")
    entities: list[str] | None = Field(default=None, description="Updated entities")
    topics: list[str] | None = Field(default=None, description="Updated topics")
    metadata: dict[str, Any] | None = Field(default=None, description="Updated metadata")


class MemoryResponse(BaseModel):
    """Response model for a memory."""

    id: str = Field(..., description="Memory ID")
    memory_type: str = Field(..., description="Type of memory")
    content: str = Field(..., description="Memory content")
    agent_id: str | None = Field(default=None, description="Agent identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    source: str = Field(..., description="Memory source")
    strength: float = Field(..., description="Current memory strength")
    importance: float = Field(..., description="Importance score")
    emotional_valence: float = Field(..., description="Emotional valence")
    access_count: int = Field(..., description="Number of accesses")
    entities: list[str] = Field(default_factory=list, description="Named entities")
    topics: list[str] = Field(default_factory=list, description="Topics")
    is_pinned: bool = Field(..., description="Whether memory is pinned")
    is_archived: bool = Field(..., description="Whether memory is archived")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_accessed_at: datetime = Field(..., description="Last access timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""

    query: str = Field(..., description="Search query text")
    agent_id: str | None = Field(default=None, description="Filter by agent")
    user_id: str | None = Field(default=None, description="Filter by user")
    memory_type: str | None = Field(default=None, description="Filter by type")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    use_mmr: bool = Field(default=False, description="Use MMR for diversity")
    include_archived: bool = Field(default=False, description="Include archived")


class MemorySearchResult(BaseModel):
    """Response model for a search result."""

    memory: MemoryResponse = Field(..., description="The memory")
    score: float = Field(..., description="Relevance score")
    similarity: float = Field(..., description="Semantic similarity")


class MemorySearchResponse(BaseModel):
    """Response model for search results."""

    results: list[MemorySearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total results returned")
    query: str = Field(..., description="Original query")


class MemoryListResponse(BaseModel):
    """Response model for listing memories."""

    memories: list[MemoryResponse] = Field(..., description="List of memories")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: dict[str, Any] | None = Field(default=None, description="Additional details")


class StatsResponse(BaseModel):
    """Response model for memory statistics."""

    total_memories: int = Field(..., description="Total memory count")
    by_type: dict[str, int] = Field(..., description="Count by memory type")
    by_source: dict[str, int] = Field(..., description="Count by source")
    average_strength: float = Field(..., description="Average memory strength")
    average_importance: float = Field(..., description="Average importance")
    pinned_count: int = Field(..., description="Number of pinned memories")
    archived_count: int = Field(..., description="Number of archived memories")

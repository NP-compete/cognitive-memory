"""Health check endpoints."""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter
except ImportError as e:
    raise ImportError(
        "FastAPI is required. Install with: pip install cognitive-memory[api]"
    ) from e

from cognitive_memory import __version__

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Health status and version info.
    """
    return {
        "status": "healthy",
        "version": __version__,
        "service": "cognitive-memory",
    }


@router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """
    Readiness check endpoint.

    Returns:
        Readiness status.
    """
    return {
        "ready": True,
        "checks": {
            "api": True,
        },
    }

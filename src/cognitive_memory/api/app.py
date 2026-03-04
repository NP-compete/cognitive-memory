"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from cognitive_memory import __version__

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: Any) -> AsyncIterator[None]:
    """Application lifespan manager."""
    logger.info("Starting cognitive-memory API")
    yield
    logger.info("Shutting down cognitive-memory API")


def create_app() -> Any:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for the REST API. "
            "Install it with: pip install cognitive-memory[api]"
        ) from e

    app = FastAPI(
        title="Cognitive Memory API",
        description="REST API for agent memory management with intelligent forgetting",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from cognitive_memory.api.routes import health, memories

    app.include_router(health.router, tags=["Health"])
    app.include_router(memories.router, prefix="/api/v1", tags=["Memories"])

    return app

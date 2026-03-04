# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core data models (Memory, Fact, Entity, Procedure)
- Configuration system with Pydantic
- Decay engine with exponential decay and rehearsal
- Importance scoring engine
- Retrieval engine with decay-aware scoring and MMR
- Consolidation engine for episodic → semantic transformation
- Context builder for multi-tier context assembly
- Working memory tier (in-memory)
- Episodic memory tier (vector store)
- Semantic memory tier (knowledge graph)
- Procedural memory tier (PostgreSQL)
- Storage adapters for Qdrant, Neo4j, PostgreSQL, Redis
- LangGraph integration (CheckpointSaver)
- LangChain integration (BaseMemory)
- FastAPI REST API
- Background workers (decay tick, consolidation, GC)
- Docker support
- Comprehensive test suite
- Documentation

## [0.1.0] - TBD

### Added
- Initial release

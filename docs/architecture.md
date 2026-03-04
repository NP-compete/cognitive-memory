# Architecture

This document describes the architecture of Cognitive Memory.

## Overview

Cognitive Memory is a multi-tier memory system inspired by human cognitive architecture. It implements four memory tiers, each with distinct characteristics and purposes.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  LangGraph Integration  │  LangChain Integration  │  REST API  │  Python SDK │
└────────────┬────────────┴───────────┬─────────────┴─────┬──────┴──────┬──────┘
             │                        │                   │             │
             ▼                        ▼                   ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MEMORY MANAGER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Ingest    │  │  Retrieve   │  │   Score     │  │   Forget    │        │
│  │   Engine    │  │   Engine    │  │   Engine    │  │   Engine    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────┬────────────────────────────┬───────────────────────┬───────────┘
             │                            │                       │
             ▼                            ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MEMORY TIERS                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │    WORKING    │  │   EPISODIC    │  │   SEMANTIC    │  │  PROCEDURAL  │ │
│  │    MEMORY     │  │    MEMORY     │  │    MEMORY     │  │    MEMORY    │ │
│  │  (in-context) │  │ (vector store)│  │ (knowledge    │  │   (tools,    │ │
│  │               │  │               │  │    graph)     │  │   prompts)   │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
└────────────┬────────────────────────────┬───────────────────────┬───────────┘
             │                            │                       │
             ▼                            ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BACKGROUND PROCESSES                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Consolidation  │  │   Decay Tick    │  │    Garbage      │             │
│  │     Worker      │  │     Worker      │  │   Collection    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└────────────┬────────────────────────────┬───────────────────────┬───────────┘
             │                            │                       │
             ▼                            ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  PostgreSQL │  │   Qdrant/   │  │    Neo4j/   │  │    Redis    │        │
│  │  (metadata) │  │   Pinecone  │  │   Memgraph  │  │   (cache)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Tiers

### Working Memory

- **Purpose**: Active reasoning context for current conversation
- **Capacity**: Configurable token limit (default: 8K tokens)
- **Retention**: Session-scoped
- **Storage**: In-memory (Python objects)
- **Eviction**: LRU with importance weighting

Working memory holds the immediate context needed for the current interaction. When it overflows, older content is spilled to episodic memory.

### Episodic Memory

- **Purpose**: Specific past interactions and events
- **Capacity**: Unlimited (with decay-based pruning)
- **Retention**: Days to weeks (decay-dependent)
- **Storage**: Vector database + PostgreSQL metadata
- **Indexing**: Semantic (embeddings) + temporal + entity

Episodic memory stores individual experiences. Each memory has a strength that decays over time, and retrieval boosts this strength (rehearsal effect).

### Semantic Memory

- **Purpose**: Extracted facts, concepts, relationships
- **Capacity**: Unlimited
- **Retention**: Long-term (slow decay)
- **Storage**: Knowledge graph (Neo4j/Memgraph)
- **Indexing**: Entity-based, relationship traversal

Semantic memory stores structured knowledge extracted from episodic memories during consolidation. Facts are stored as subject-predicate-object triples.

### Procedural Memory

- **Purpose**: Learned skills, tool usage patterns, prompts
- **Capacity**: Limited (curated)
- **Retention**: Permanent (explicit deletion only)
- **Storage**: PostgreSQL
- **Indexing**: Name-based, tag-based

Procedural memory stores learned procedures and patterns that don't decay. This includes successful tool usage patterns and prompt templates.

## Core Engines

### Decay Engine

Computes memory strength decay using exponential decay:

```
S(t) = S₀ × e^(-λ × Δt)
```

Also handles rehearsal (strength boost on retrieval).

### Importance Engine

Computes importance scores from multiple factors:
- Access frequency
- Recency
- Emotional salience
- Surprise
- Explicit markers
- Entity relevance

### Retrieval Engine

Performs decay-aware retrieval:
1. Over-fetch candidates from vector store
2. Compute combined scores (similarity + strength + importance + recency)
3. Filter forgotten memories
4. Apply MMR for diversity
5. Return top-k with rehearsal boost

### Consolidation Engine

Transforms weak episodic memories into semantic facts:
1. Cluster similar weak memories
2. Extract facts using LLM
3. Resolve contradictions
4. Store in knowledge graph
5. Prune consolidated episodic memories

## Background Workers

### Decay Tick Worker

Periodically updates memory strengths based on decay function. Runs every hour by default.

### Consolidation Worker

Runs consolidation process periodically (default: every 24 hours) or when triggered by memory count threshold.

### Garbage Collection Worker

Archives or deletes memories that have fallen below the forget threshold and have been consolidated.

## Storage Backends

| Store | Options | Purpose |
|-------|---------|---------|
| Vector | Qdrant, Pinecone, pgvector | Episodic memory embeddings |
| Graph | Neo4j, Memgraph | Semantic memory facts |
| Metadata | PostgreSQL, SQLite | Memory metadata, procedures |
| Cache | Redis | Query caching |

## Data Flow

### Remember Flow

```
Input → Embed → Store in Episodic → Extract Entities → Update Working Memory
```

### Recall Flow

```
Query → Embed → Retrieve from All Tiers → Score & Rank → Apply Rehearsal → Return
```

### Consolidation Flow

```
Get Weak Memories → Cluster → Extract Facts → Store in Semantic → Mark Consolidated → Prune
```

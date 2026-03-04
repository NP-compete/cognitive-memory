# Configuration

This document describes all configuration options for Cognitive Memory.

## Quick Start

```python
from cognitive_memory import MemoryManager, MemorySystemConfig

# Use defaults
memory = MemoryManager()

# Or customize
config = MemorySystemConfig(
    vector_store="qdrant",
    graph_store="neo4j",
)
memory = MemoryManager(config)
```

## Configuration Sections

### Decay Configuration

Controls how memories weaken over time.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `episodic_decay_rate` | 0.1 | 0.01-1.0 | Decay rate per day for episodic memories |
| `semantic_decay_rate` | 0.01 | 0.001-0.1 | Decay rate per day for semantic memories |
| `consolidation_threshold` | 0.3 | 0.1-0.5 | Below this, consider for consolidation |
| `forget_threshold` | 0.1 | 0.01-0.3 | Below this, eligible for deletion |
| `rehearsal_boost` | 1.5 | 1.1-2.0 | Strength multiplier on access |
| `rehearsal_cap` | 2.0 | 1.5-5.0 | Maximum strength after boosts |
| `decay_tick_interval_hours` | 1.0 | 0.5-24.0 | How often to update strengths |

```python
from cognitive_memory import DecayConfig

decay = DecayConfig(
    episodic_decay_rate=0.1,
    semantic_decay_rate=0.01,
    consolidation_threshold=0.3,
    forget_threshold=0.1,
    rehearsal_boost=1.5,
)
```

### Importance Configuration

Controls how memory importance is scored.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `access_frequency_weight` | 0.25 | 0-1 | Weight for access count |
| `recency_weight` | 0.20 | 0-1 | Weight for recency |
| `emotional_weight` | 0.15 | 0-1 | Weight for emotional content |
| `surprise_weight` | 0.10 | 0-1 | Weight for surprise |
| `explicit_marker_weight` | 0.20 | 0-1 | Weight for user markers |
| `entity_relevance_weight` | 0.10 | 0-1 | Weight for entity overlap |
| `critical_importance` | 0.9 | 0.8-1.0 | Never forget threshold |

```python
from cognitive_memory import ImportanceConfig

importance = ImportanceConfig(
    access_frequency_weight=0.25,
    recency_weight=0.20,
    emotional_weight=0.15,
    critical_importance=0.9,
)
```

### Consolidation Configuration

Controls the consolidation process.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `consolidation_interval_hours` | 24.0 | 1.0-168.0 | How often to consolidate |
| `min_memories_for_consolidation` | 5 | 3-20 | Minimum candidates needed |
| `max_memories_per_consolidation` | 100 | 50-500 | Maximum to process |
| `similarity_threshold` | 0.75 | 0.5-0.9 | Clustering threshold |
| `min_cluster_size` | 3 | 2-10 | Minimum cluster for consolidation |
| `max_summary_tokens` | 200 | 50-500 | Summary length limit |
| `consolidation_model` | gpt-4o-mini | — | LLM for consolidation |
| `max_llm_calls_per_consolidation` | 20 | 5-50 | Cost control |

```python
from cognitive_memory import ConsolidationConfig

consolidation = ConsolidationConfig(
    consolidation_interval_hours=24.0,
    similarity_threshold=0.75,
    consolidation_model="gpt-4o-mini",
)
```

### Retrieval Configuration

Controls memory retrieval behavior.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `similarity_weight` | 0.40 | 0-1 | Weight for embedding similarity |
| `strength_weight` | 0.25 | 0-1 | Weight for decay strength |
| `importance_weight` | 0.20 | 0-1 | Weight for importance |
| `recency_weight` | 0.15 | 0-1 | Weight for recency |
| `default_k` | 10 | 5-50 | Default retrieval count |
| `over_fetch_multiplier` | 3.0 | 2.0-5.0 | Over-fetch ratio |
| `mmr_lambda` | 0.7 | 0.5-1.0 | MMR diversity parameter |

```python
from cognitive_memory import RetrievalConfig

retrieval = RetrievalConfig(
    similarity_weight=0.40,
    strength_weight=0.25,
    default_k=10,
    mmr_lambda=0.7,
)
```

### Storage Configuration

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `vector_store` | qdrant | qdrant, pinecone, pgvector, chroma | Vector backend |
| `graph_store` | neo4j | neo4j, memgraph, none | Graph backend |
| `metadata_store` | postgresql | postgresql, sqlite | Metadata backend |
| `cache_store` | redis | redis, none | Cache backend |
| `embedding_model` | text-embedding-3-small | any OpenAI/local model | Embedding model |
| `embedding_dimensions` | 1536 | 384-3072 | Embedding size |

## Full Configuration Example

```python
from cognitive_memory import (
    MemoryManager,
    MemorySystemConfig,
    DecayConfig,
    ImportanceConfig,
    ConsolidationConfig,
    RetrievalConfig,
)

config = MemorySystemConfig(
    decay=DecayConfig(
        episodic_decay_rate=0.1,
        semantic_decay_rate=0.01,
        consolidation_threshold=0.3,
        forget_threshold=0.1,
        rehearsal_boost=1.5,
    ),
    importance=ImportanceConfig(
        access_frequency_weight=0.25,
        recency_weight=0.20,
        emotional_weight=0.15,
        surprise_weight=0.10,
        explicit_marker_weight=0.20,
        entity_relevance_weight=0.10,
    ),
    consolidation=ConsolidationConfig(
        consolidation_interval_hours=24.0,
        similarity_threshold=0.75,
        consolidation_model="gpt-4o-mini",
    ),
    retrieval=RetrievalConfig(
        similarity_weight=0.40,
        strength_weight=0.25,
        importance_weight=0.20,
        recency_weight=0.15,
        default_k=10,
    ),
    vector_store="qdrant",
    graph_store="neo4j",
    metadata_store="postgresql",
    cache_store="redis",
    embedding_model="text-embedding-3-small",
)

memory = MemoryManager(config)
```

## Environment Variables

Configuration can also be set via environment variables:

```bash
# Storage
COGNITIVE_MEMORY_VECTOR_STORE=qdrant
COGNITIVE_MEMORY_GRAPH_STORE=neo4j
COGNITIVE_MEMORY_METADATA_STORE=postgresql
COGNITIVE_MEMORY_CACHE_STORE=redis

# Connection strings
QDRANT_URL=http://localhost:6333
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
DATABASE_URL=postgresql://user:pass@localhost:5432/cognitive_memory
REDIS_URL=redis://localhost:6379

# API keys
OPENAI_API_KEY=sk-...

# Decay
COGNITIVE_MEMORY_EPISODIC_DECAY_RATE=0.1
COGNITIVE_MEMORY_FORGET_THRESHOLD=0.1

# Consolidation
COGNITIVE_MEMORY_CONSOLIDATION_MODEL=gpt-4o-mini
```

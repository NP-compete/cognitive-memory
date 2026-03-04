# Cognitive Memory

[![PyPI version](https://badge.fury.io/py/cognitive-memory.svg)](https://badge.fury.io/py/cognitive-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/NP-compete/cognitive-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/NP-compete/cognitive-memory/actions/workflows/ci.yml)

**Memory that forgets, like humans do.**

A production-grade memory system for AI agents with intelligent forgetting. Unlike traditional agent memory that accumulates indefinitely, Cognitive Memory implements principled decay, importance-based retention, and automatic consolidation—mimicking how human memory actually works.

## The Problem

Current agent memory solutions (Mem0, Zep, MemGPT) accumulate memories forever. This causes:

- **Context pollution** — irrelevant old memories dilute retrieval quality
- **Cost explosion** — vector stores grow unbounded  
- **Temporal confusion** — no distinction between recent and ancient context
- **No consolidation** — raw events never become structured knowledge

## The Solution

Cognitive Memory implements a biologically-inspired memory architecture:

| Feature | Description |
|---------|-------------|
| **Decay** | Memories weaken over time (exponential decay) |
| **Rehearsal** | Accessing a memory strengthens it |
| **Importance** | Multi-factor scoring determines retention priority |
| **Consolidation** | Weak episodic memories become semantic facts |
| **Forgetting** | Below-threshold memories are pruned |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY TIERS                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │    WORKING    │  │   EPISODIC    │  │   SEMANTIC    │       │
│  │    MEMORY     │  │    MEMORY     │  │    MEMORY     │       │
│  │  (in-context) │  │ (vector store)│  │ (knowledge    │       │
│  │               │  │   + decay     │  │    graph)     │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│         │                   │                  │                │
│         ▼                   ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CONSOLIDATION ENGINE                        │   │
│  │    (episodic → semantic transformation, pruning)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install cognitive-memory
```

## Quick Start

```python
from cognitive_memory import MemoryManager, MemoryConfig

# Initialize with default configuration
config = MemoryConfig()
memory = MemoryManager(config)

# Store a memory
memory.remember(
    content="User prefers dark mode and uses vim keybindings",
    source="conversation",
)

# Retrieve relevant memories (decay-aware)
results = memory.recall(
    query="What are the user's preferences?",
    k=5,
)

# Build context for LLM (from all tiers)
context = memory.get_context(
    query="Help user configure their editor",
    max_tokens=4000,
)

# Run consolidation (or let background worker handle it)
memory.consolidate()
```

## Key Concepts

### Memory Decay

Every memory has a strength that decays exponentially over time:

```
S(t) = S₀ × e^(-λ × Δt)
```

- **S₀**: Initial strength (boosted by rehearsal)
- **λ**: Decay rate (configurable per memory type)
- **Δt**: Time since creation

When you retrieve a memory, its strength is **boosted** (rehearsal effect), mimicking how recalling something helps you remember it.

### Importance Scoring

Not all memories are equal. Importance is computed from:

| Factor | Weight | Description |
|--------|--------|-------------|
| Access frequency | 25% | How often retrieved |
| Recency | 20% | How recently accessed |
| Emotional salience | 15% | Strong reactions |
| Surprise | 10% | Unexpected information |
| Explicit markers | 20% | User said "remember this" |
| Entity relevance | 10% | Contains important entities |

### Consolidation

Periodically, weak episodic memories are:

1. **Clustered** by semantic similarity
2. **Summarized** into semantic facts
3. **Stored** in the knowledge graph
4. **Pruned** from episodic storage

This mimics how human memory consolidates during sleep.

### Forgetting

Memories below the forget threshold are candidates for deletion—but only if:
- They've been consolidated, OR
- They're not marked as critical

This prevents unbounded growth while preserving important information.

## Memory Tiers

| Tier | Purpose | Retention | Storage |
|------|---------|-----------|---------|
| **Working** | Current conversation | Session | In-memory |
| **Episodic** | Past interactions | Days-weeks | Vector DB |
| **Semantic** | Extracted facts | Months-years | Knowledge graph |
| **Procedural** | Skills, patterns | Permanent | PostgreSQL |

## Integrations

### LangGraph

```python
from cognitive_memory.integrations import CognitiveCheckpointer

checkpointer = CognitiveCheckpointer.from_config(config)
graph = builder.compile(checkpointer=checkpointer)
```

### LangChain

```python
from cognitive_memory.integrations import CognitiveMemory

memory = CognitiveMemory.from_config(config)
chain = ConversationChain(llm=llm, memory=memory)
```

## Configuration

```python
from cognitive_memory import MemoryConfig, DecayConfig, ImportanceConfig

config = MemoryConfig(
    # Decay settings
    decay=DecayConfig(
        episodic_decay_rate=0.1,      # per day
        semantic_decay_rate=0.01,     # per day
        consolidation_threshold=0.3,
        forget_threshold=0.1,
        rehearsal_boost=1.5,
    ),
    
    # Importance settings
    importance=ImportanceConfig(
        access_frequency_weight=0.25,
        recency_weight=0.20,
        emotional_weight=0.15,
        surprise_weight=0.10,
        explicit_marker_weight=0.20,
        entity_relevance_weight=0.10,
    ),
    
    # Storage backends
    vector_store="qdrant",      # qdrant, pinecone, pgvector
    graph_store="neo4j",        # neo4j, memgraph, none
    metadata_store="postgresql",
)
```

## Comparison

| Feature | Mem0 | Zep | MemGPT | **Cognitive Memory** |
|---------|------|-----|--------|----------------------|
| Decay function | ❌ | ❌ | ❌ | ✅ |
| Importance scoring | Basic | Basic | ❌ | ✅ Multi-factor |
| Auto consolidation | ❌ | ❌ | Manual | ✅ |
| Principled forgetting | ❌ | ❌ | ❌ | ✅ |
| Multi-tier | ❌ | Partial | ✅ | ✅ 4 tiers |
| Knowledge graph | ❌ | ❌ | ❌ | ✅ |
| LangGraph native | ❌ | ❌ | ❌ | ✅ |

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Algorithms](docs/algorithms.md)
- [Integrations](docs/integrations.md)
- [Deployment](docs/deployment.md)
- [Benchmarks](docs/benchmarks.md)

## Development

```bash
# Clone the repository
git clone https://github.com/NP-compete/cognitive-memory.git
cd cognitive-memory

# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run linters
make lint
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by cognitive science research on human memory
- Built on [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Vector storage powered by [Qdrant](https://qdrant.tech/)
- Knowledge graph powered by [Neo4j](https://neo4j.com/)

---

<p align="center">
  <strong>Memory that forgets, so your agents remember what matters.</strong>
</p>

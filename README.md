<p align="center">
  <img src="https://img.shields.io/badge/Status-In_Development-yellow?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/github/license/NP-compete/cognitive-memory?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="https://github.com/NP-compete/cognitive-memory/actions/workflows/ci.yml">
    <img src="https://github.com/NP-compete/cognitive-memory/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/NP-compete/cognitive-memory">
    <img src="https://codecov.io/gh/NP-compete/cognitive-memory/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://pypi.org/project/cognitive-memory/">
    <img src="https://img.shields.io/pypi/v/cognitive-memory?color=blue" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/cognitive-memory/">
    <img src="https://img.shields.io/pypi/dm/cognitive-memory" alt="Downloads">
  </a>
</p>

<h1 align="center">🧠 Cognitive Memory</h1>

<p align="center">
  <strong>Memory that forgets, like humans do.</strong>
</p>

<p align="center">
  A production-grade memory system for AI agents with intelligent forgetting.<br>
  Built on cognitive science principles: decay, rehearsal, consolidation, and importance-based retention.
</p>

<p align="center">
  <a href="#the-problem">Problem</a> •
  <a href="#the-solution">Solution</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## The Problem

Current agent memory solutions accumulate memories **forever**. This causes:

| Issue | Impact |
|-------|--------|
| 🗑️ **Context Pollution** | Irrelevant old memories dilute retrieval quality |
| 💸 **Cost Explosion** | Vector stores grow unbounded |
| ⏰ **Temporal Confusion** | No distinction between recent and ancient context |
| 📦 **No Consolidation** | Raw events never become structured knowledge |

**Humans don't work this way.** We forget. And that's a feature, not a bug.

---

## The Solution

Cognitive Memory implements a **biologically-inspired** memory architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     HOW HUMAN MEMORY WORKS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   📥 Experience ──▶ 🧠 Working Memory ──▶ 💾 Long-term Memory   │
│                           │                       │              │
│                           │                       ▼              │
│                           │              ┌───────────────┐       │
│                           │              │  Consolidation │       │
│                           │              │  (during sleep)│       │
│                           │              └───────┬───────┘       │
│                           │                      │               │
│                           ▼                      ▼               │
│                    ┌─────────────┐      ┌─────────────┐         │
│                    │   Decay     │      │  Semantic   │         │
│                    │ (forgetting)│      │   Facts     │         │
│                    └─────────────┘      └─────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **⏳ Decay** | Memories weaken over time (exponential decay) |
| **🔄 Rehearsal** | Accessing a memory strengthens it |
| **⭐ Importance** | Multi-factor scoring determines retention priority |
| **🔀 Consolidation** | Weak episodic memories become semantic facts |
| **🗑️ Forgetting** | Below-threshold memories are pruned |

---

## Architecture

### Memory Tiers

| Tier | Purpose | Retention | Storage |
|------|---------|-----------|---------|
| **Working** | Current conversation | Session | In-memory |
| **Episodic** | Past interactions | Days-weeks | Vector DB |
| **Semantic** | Extracted facts | Months-years | Knowledge Graph |
| **Procedural** | Skills, patterns | Permanent | PostgreSQL |

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY SYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   WORKING   │  │  EPISODIC   │  │  SEMANTIC   │             │
│  │   MEMORY    │  │   MEMORY    │  │   MEMORY    │             │
│  │ (context)   │  │  (events)   │  │  (facts)    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   RETRIEVAL ENGINE                       │   │
│  │         (decay-aware scoring + MMR diversity)            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 CONSOLIDATION ENGINE                     │   │
│  │    (clustering → summarization → fact extraction)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install cognitive-memory
```

### Basic Usage

```python
from cognitive_memory import MemoryManager

# Initialize
memory = MemoryManager()

# Remember something
memory.remember(
    content="User prefers dark mode and vim keybindings",
    source="conversation",
)

# Recall relevant memories (decay-aware)
results = memory.recall(
    query="What are the user's preferences?",
    k=5,
)

# Build context for LLM
context = memory.get_context(
    query="Help configure their editor",
    max_tokens=4000,
)
```

### With LangGraph

```python
from cognitive_memory.integrations import CognitiveCheckpointer

checkpointer = CognitiveCheckpointer()
graph = builder.compile(checkpointer=checkpointer)
```

### With LangChain

```python
from cognitive_memory.integrations import CognitiveMemory

memory = CognitiveMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

---

## How It Works

### 1. Decay Function

Every memory has a strength that decays exponentially:

```
S(t) = S₀ × e^(-λ × Δt)
```

| Variable | Meaning |
|----------|---------|
| `S(t)` | Strength at time t |
| `S₀` | Initial strength |
| `λ` | Decay rate |
| `Δt` | Time elapsed |

**Rehearsal Effect:** When you retrieve a memory, its strength is boosted—just like how recalling something helps you remember it.

### 2. Importance Scoring

| Factor | Weight | Description |
|--------|--------|-------------|
| Access frequency | 25% | How often retrieved |
| Recency | 20% | How recently accessed |
| Emotional salience | 15% | Strong reactions |
| Surprise | 10% | Unexpected information |
| Explicit markers | 20% | User said "remember this" |
| Entity relevance | 10% | Contains important entities |

### 3. Consolidation

Periodically, weak episodic memories are:

1. **Clustered** by semantic similarity
2. **Summarized** into semantic facts
3. **Stored** in the knowledge graph
4. **Pruned** from episodic storage

This mimics how human memory consolidates during sleep.

---

## Comparison

| Feature | Mem0 | Zep | MemGPT | **Cognitive Memory** |
|---------|:----:|:---:|:------:|:--------------------:|
| Decay function | ❌ | ❌ | ❌ | ✅ |
| Importance scoring | Basic | Basic | ❌ | ✅ Multi-factor |
| Auto consolidation | ❌ | ❌ | Manual | ✅ |
| Principled forgetting | ❌ | ❌ | ❌ | ✅ |
| Multi-tier | ❌ | Partial | ✅ | ✅ 4 tiers |
| Knowledge graph | ❌ | ❌ | ❌ | ✅ |
| LangGraph native | ❌ | ❌ | ❌ | ✅ |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and components |
| [Algorithms](docs/algorithms.md) | Decay, importance, consolidation |
| [Configuration](docs/configuration.md) | All configuration options |
| [Integrations](docs/integrations.md) | LangGraph, LangChain setup |
| [Deployment](docs/deployment.md) | Production deployment guide |
| [Benchmarks](docs/benchmarks.md) | Performance measurements |

---

## Project Status

🚧 **In Active Development**

This project is under active development. The core architecture is designed, and implementation is in progress.

### Roadmap

- [ ] Core memory models and configuration
- [ ] Decay and importance engines
- [ ] Retrieval with decay-aware scoring
- [ ] Consolidation engine
- [ ] LangGraph integration
- [ ] LangChain integration
- [ ] REST API
- [ ] Benchmarks and evaluation
- [ ] v0.1.0 release

**Want to contribute?** Check out [CONTRIBUTING.md](CONTRIBUTING.md) or join the [Discussions](https://github.com/NP-compete/cognitive-memory/discussions).

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone and setup
git clone https://github.com/NP-compete/cognitive-memory.git
cd cognitive-memory
pip install -e ".[dev]"

# Run tests
make test

# Run linters
make lint
```

---

## Citation

If you use Cognitive Memory in your research, please cite:

```bibtex
@software{cognitive_memory,
  author = {Dutta, Soham},
  title = {Cognitive Memory: Memory that forgets, like humans do},
  url = {https://github.com/NP-compete/cognitive-memory},
  year = {2026}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by cognitive science research on human memory
- Built for the [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain) ecosystems
- Vector storage: [Qdrant](https://qdrant.tech/), [Pinecone](https://pinecone.io/)
- Knowledge graph: [Neo4j](https://neo4j.com/)

---

<p align="center">
  <strong>Memory that forgets, so your agents remember what matters.</strong>
</p>

<p align="center">
  <a href="https://github.com/NP-compete/cognitive-memory/stargazers">⭐ Star us</a> •
  <a href="https://github.com/NP-compete/cognitive-memory/issues">🐛 Report Bug</a> •
  <a href="https://github.com/NP-compete/cognitive-memory/discussions">💬 Discussions</a>
</p>

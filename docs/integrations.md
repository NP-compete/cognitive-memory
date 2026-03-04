# Integrations

This document describes how to integrate Cognitive Memory with popular frameworks.

## LangGraph

Cognitive Memory provides a native LangGraph `CheckpointSaver` implementation.

### Basic Usage

```python
from langgraph.graph import StateGraph
from cognitive_memory.integrations import CognitiveCheckpointer

# Create checkpointer
checkpointer = CognitiveCheckpointer()

# Build your graph
builder = StateGraph(dict)
builder.add_node("agent", agent_node)
builder.set_entry_point("agent")

# Compile with checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [...]}, config)
```

### With Custom Configuration

```python
from cognitive_memory import MemorySystemConfig
from cognitive_memory.integrations import CognitiveCheckpointer

config = MemorySystemConfig(
    vector_store="qdrant",
    graph_store="neo4j",
)

checkpointer = CognitiveCheckpointer(config=config)
```

### Async Usage

```python
from cognitive_memory.integrations import AsyncCognitiveCheckpointer

async with AsyncCognitiveCheckpointer() as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke({"messages": [...]}, config)
```

### Features

- Full checkpoint persistence with decay
- Memory consolidation between sessions
- Cross-session context retrieval
- Human-in-the-loop support

---

## LangChain

Cognitive Memory provides a `BaseMemory` implementation for LangChain.

### Basic Usage

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from cognitive_memory.integrations import CognitiveMemory

# Create memory
memory = CognitiveMemory()

# Use with chain
llm = ChatOpenAI()
chain = ConversationChain(llm=llm, memory=memory)

# Chat
response = chain.invoke({"input": "Hello!"})
```

### With Custom Configuration

```python
from cognitive_memory import MemorySystemConfig
from cognitive_memory.integrations import CognitiveMemory

config = MemorySystemConfig(
    decay=DecayConfig(episodic_decay_rate=0.05),
)

memory = CognitiveMemory(config=config)
```

### Memory Variables

```python
# Default memory variable
memory = CognitiveMemory(memory_key="history")

# Access memory variables
variables = memory.load_memory_variables({})
print(variables["history"])
```

### Features

- Drop-in replacement for `ConversationBufferMemory`
- Automatic decay and consolidation
- Long-term context across sessions
- Entity extraction and tracking

---

## REST API

Cognitive Memory includes a FastAPI-based REST API.

### Running the API

```bash
# With uvicorn
uvicorn cognitive_memory.api.app:app --host 0.0.0.0 --port 8000

# With Docker
docker-compose -f docker/docker-compose.yml up -d
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memories` | Store a new memory |
| GET | `/memories/{id}` | Get a specific memory |
| POST | `/recall` | Retrieve relevant memories |
| POST | `/context` | Build LLM context |
| POST | `/consolidate` | Trigger consolidation |
| GET | `/stats` | Get memory statistics |
| DELETE | `/memories/{id}` | Delete a memory |

### Example Requests

**Store a memory:**

```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers dark mode",
    "source": "conversation",
    "user_id": "user-123"
  }'
```

**Recall memories:**

```bash
curl -X POST http://localhost:8000/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the user preferences?",
    "k": 5,
    "user_id": "user-123"
  }'
```

**Build context:**

```bash
curl -X POST http://localhost:8000/context \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Help configure their editor",
    "max_tokens": 4000,
    "user_id": "user-123"
  }'
```

---

## Python SDK

Direct Python usage without frameworks.

### Basic Usage

```python
from cognitive_memory import MemoryManager

# Initialize
memory = MemoryManager()

# Store
memory_id = memory.remember(
    content="Important information",
    source="conversation",
    user_id="user-123",
)

# Retrieve
results = memory.recall(
    query="What's important?",
    k=5,
    user_id="user-123",
)

# Build context
context = memory.get_context(
    query="Help with task",
    max_tokens=4000,
)

# Consolidate
result = memory.consolidate()

# Get stats
stats = memory.get_stats()
```

### Async Usage

```python
from cognitive_memory import AsyncMemoryManager

async with AsyncMemoryManager() as memory:
    await memory.remember(content="...")
    results = await memory.recall(query="...")
```

### Context Manager

```python
from cognitive_memory import MemoryManager

with MemoryManager() as memory:
    memory.remember(content="...")
    # Automatic cleanup on exit
```

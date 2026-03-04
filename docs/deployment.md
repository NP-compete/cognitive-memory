# Deployment

This document describes how to deploy Cognitive Memory in production.

## Deployment Options

### 1. Embedded (Single Process)

For simple use cases, embed Cognitive Memory directly in your application.

```python
from cognitive_memory import MemoryManager

# Uses SQLite and in-memory storage
memory = MemoryManager(
    vector_store="chroma",
    graph_store="none",
    metadata_store="sqlite",
    cache_store="none",
)
```

**Pros:** Simple, no external dependencies
**Cons:** Not suitable for multi-process or distributed systems

### 2. Docker Compose (Development/Small Scale)

Use the provided Docker Compose for local development or small deployments.

```bash
cd cognitive-memory
docker-compose -f docker/docker-compose.yml up -d
```

This starts:
- Cognitive Memory API
- PostgreSQL (metadata)
- Qdrant (vectors)
- Neo4j (knowledge graph)
- Redis (cache)

### 3. Kubernetes (Production)

For production, deploy to Kubernetes with proper scaling and monitoring.

#### Helm Chart (Coming Soon)

```bash
helm repo add cognitive-memory https://np-compete.github.io/cognitive-memory
helm install cognitive-memory cognitive-memory/cognitive-memory
```

#### Manual Deployment

See example manifests in `deploy/kubernetes/`.

---

## Architecture Patterns

### Single-User Local

```
┌─────────────────────────────────────┐
│           Python Process            │
│  ┌─────────────────────────────┐   │
│  │     cognitive-memory        │   │
│  └─────────────────────────────┘   │
│              │                      │
│  ┌───────────┴───────────┐         │
│  │  SQLite   │  Chroma   │         │
│  └───────────────────────┘         │
└─────────────────────────────────────┘
```

### Multi-User Production

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                           ▼                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   API Pod   │  │   API Pod   │  │   API Pod   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┼────────────────┘                     │
│                          │                                       │
│  ┌───────────────────────┼───────────────────────┐              │
│  │                       │                       │              │
│  ▼                       ▼                       ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ PostgreSQL  │  │   Qdrant    │  │    Neo4j    │             │
│  │  (HA)       │  │  (cluster)  │  │  (cluster)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                          │                                       │
│                          ▼                                       │
│                   ┌─────────────┐                                │
│                   │    Redis    │                                │
│                   │  (cluster)  │                                │
│                   └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Storage Backend Setup

### PostgreSQL

```bash
# Create database
createdb cognitive_memory

# Or with Docker
docker run -d \
  --name postgres \
  -e POSTGRES_USER=cognitive \
  -e POSTGRES_PASSWORD=cognitive \
  -e POSTGRES_DB=cognitive_memory \
  -p 5432:5432 \
  postgres:15
```

Environment variable:
```bash
DATABASE_URL=postgresql://cognitive:cognitive@localhost:5432/cognitive_memory
```

### Qdrant

```bash
# With Docker
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

Environment variable:
```bash
QDRANT_URL=http://localhost:6333
```

### Neo4j

```bash
# With Docker
docker run -d \
  --name neo4j \
  -e NEO4J_AUTH=neo4j/password \
  -p 7474:7474 \
  -p 7687:7687 \
  neo4j:5
```

Environment variables:
```bash
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### Redis

```bash
# With Docker
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7
```

Environment variable:
```bash
REDIS_URL=redis://localhost:6379
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Storage connections
DATABASE_URL=postgresql://user:pass@host:5432/db
QDRANT_URL=http://qdrant:6333
NEO4J_URL=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://redis:6379

# Optional tuning
COGNITIVE_MEMORY_EPISODIC_DECAY_RATE=0.1
COGNITIVE_MEMORY_CONSOLIDATION_INTERVAL_HOURS=24
COGNITIVE_MEMORY_CONSOLIDATION_MODEL=gpt-4o-mini
```

### Secrets Management

For production, use proper secrets management:

- **Kubernetes:** Use Secrets or external-secrets-operator
- **AWS:** Use Secrets Manager or Parameter Store
- **GCP:** Use Secret Manager
- **Azure:** Use Key Vault

---

## Monitoring

### Health Checks

```bash
# Liveness
curl http://localhost:8000/health/live

# Readiness
curl http://localhost:8000/health/ready
```

### Metrics

Prometheus metrics available at `/metrics`:

- `cognitive_memory_memories_total` - Total memories stored
- `cognitive_memory_recalls_total` - Total recall operations
- `cognitive_memory_consolidations_total` - Total consolidations
- `cognitive_memory_decay_tick_duration_seconds` - Decay tick duration
- `cognitive_memory_retrieval_latency_seconds` - Retrieval latency

### Logging

Structured JSON logging to stdout. Configure log level:

```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

---

## Scaling

### Horizontal Scaling

- API pods are stateless and can be scaled horizontally
- Use connection pooling for database connections
- Consider read replicas for PostgreSQL

### Storage Scaling

| Component | Scaling Strategy |
|-----------|------------------|
| PostgreSQL | Read replicas, connection pooling |
| Qdrant | Distributed mode, sharding |
| Neo4j | Causal clustering |
| Redis | Redis Cluster |

### Background Workers

Run consolidation and decay workers as separate deployments:

```bash
# Consolidation worker
cognitive-memory worker consolidation

# Decay worker
cognitive-memory worker decay
```

---

## Security

### Network Security

- Use TLS for all connections
- Restrict network access with firewalls/security groups
- Use private subnets for databases

### Authentication

- Enable API authentication in production
- Use API keys or OAuth2
- Rotate credentials regularly

### Data Encryption

- Enable encryption at rest for all storage backends
- Use TLS for data in transit
- Consider field-level encryption for sensitive data

See [SECURITY.md](../SECURITY.md) for more details.

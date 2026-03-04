# Algorithms

This document describes the core algorithms used in Cognitive Memory.

## Decay Function

### Exponential Decay

Memory strength decays exponentially over time:

```
S(t) = S₀ × e^(-λ × Δt)
```

| Variable | Description |
|----------|-------------|
| `S(t)` | Strength at time t |
| `S₀` | Initial strength (boosted by rehearsals) |
| `λ` | Decay rate (per day) |
| `Δt` | Days since creation |

### Decay Rates

| Memory Type | Decay Rate (λ) | Half-life |
|-------------|----------------|-----------|
| Episodic | 0.1 | ~7 days |
| Semantic | 0.01 | ~70 days |

### Rehearsal Effect

When a memory is retrieved, its initial strength is boosted:

```
S₀_new = min(S₀ × rehearsal_boost, rehearsal_cap)
```

Default values:
- `rehearsal_boost`: 1.5
- `rehearsal_cap`: 2.0

This mimics how recalling a memory strengthens it.

### Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| Consolidation | 0.3 | Consider for consolidation |
| Forget | 0.1 | Eligible for deletion |

---

## Importance Scoring

### Multi-Factor Formula

```
I = Σ(wᵢ × fᵢ)
```

### Factors and Weights

| Factor | Weight | Computation |
|--------|--------|-------------|
| Access frequency | 0.25 | `min(1, log(1 + access_count) / 5)` |
| Recency | 0.20 | `e^(-0.1 × days_since_access)` |
| Emotional salience | 0.15 | `abs(emotional_valence)` |
| Surprise | 0.10 | `surprise_score` |
| Explicit marker | 0.20 | 1.0 if user-marked, else 0.0 |
| Entity relevance | 0.10 | Overlap with important entities |

### Importance Thresholds

| Level | Threshold | Behavior |
|-------|-----------|----------|
| Critical | ≥ 0.9 | Never forget |
| High | ≥ 0.7 | Slow decay |
| Low | < 0.3 | Fast decay |

---

## Retrieval Scoring

### Combined Score Formula

```
R = w₁×sim + w₂×strength + w₃×importance + w₄×recency
```

### Default Weights

| Factor | Weight |
|--------|--------|
| Similarity | 0.40 |
| Strength | 0.25 |
| Importance | 0.20 |
| Recency | 0.15 |

### Retrieval Algorithm

1. **Over-fetch**: Retrieve 3× candidates from vector store
2. **Score**: Compute combined scores for all candidates
3. **Filter**: Remove memories below forget threshold
4. **Diversify**: Apply MMR (Maximal Marginal Relevance)
5. **Return**: Top-k results
6. **Rehearse**: Boost strength of returned memories

### Maximal Marginal Relevance (MMR)

Balances relevance with diversity:

```
MMR = λ × Relevance - (1-λ) × max(Similarity to selected)
```

Default `λ`: 0.7

---

## Consolidation Algorithm

### Overview

Consolidation transforms weak episodic memories into semantic facts, mimicking human memory consolidation during sleep.

### Trigger Conditions

- Time-based: Every 24 hours
- Count-based: When episodic memory count exceeds threshold

### Algorithm Steps

1. **Select Candidates**
   - Get memories with strength < consolidation_threshold (0.3)
   - Limit to max_memories_per_consolidation (100)

2. **Cluster by Similarity**
   - Use embedding similarity
   - Threshold: 0.75
   - Minimum cluster size: 3

3. **Extract Facts** (per cluster)
   - Use LLM to extract subject-predicate-object triples
   - Assign confidence scores

4. **Resolve Contradictions**
   - Check new facts against existing semantic memory
   - Use recency, confidence, and verification count to resolve

5. **Create Summaries** (for large clusters)
   - Clusters with ≥5 memories get summarized
   - Summary becomes new semantic memory

6. **Mark Consolidated**
   - Original episodic memories marked as consolidated
   - Link to superseding semantic memory

7. **Prune**
   - Archive memories below forget threshold
   - Only if consolidated or superseded

### Constraints

| Constraint | Default |
|------------|---------|
| Max memories per run | 100 |
| Max LLM calls per run | 20 |
| Min cluster size | 3 |
| Similarity threshold | 0.75 |

---

## Context Building

### Token Budget Allocation

| Tier | Budget |
|------|--------|
| Working memory | 50% |
| Episodic memories | 25% |
| Semantic facts | 15% |
| Procedures | 10% |

### Algorithm

1. Reserve budget for working memory
2. Retrieve relevant episodic memories, fit within budget
3. Query semantic facts for entities in query
4. Include relevant procedures
5. Format and return

### Priority Order

1. Working memory (most recent, most relevant)
2. High-importance episodic memories
3. Semantic facts about mentioned entities
4. Relevant procedures

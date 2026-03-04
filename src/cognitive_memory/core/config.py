"""Configuration models for the memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DecayConfig:
    """
    Configuration for memory decay behavior.

    The decay function follows: strength = initial_strength * e^(-λt) + rehearsal_bonus

    Attributes:
        decay_rate: Lambda (λ) in the decay function. Higher = faster decay.
            Default 0.1 means ~10% decay per time unit.
        min_strength: Floor value - memories never decay below this.
        rehearsal_boost: Strength increase when memory is accessed.
        rehearsal_decay_rate: How quickly rehearsal bonus fades.
        time_unit: Unit for decay calculation.
    """

    decay_rate: float = 0.1
    min_strength: float = 0.01
    rehearsal_boost: float = 0.2
    rehearsal_decay_rate: float = 0.05
    time_unit: Literal["seconds", "minutes", "hours", "days"] = "hours"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        if not 0 <= self.min_strength <= 1:
            raise ValueError("min_strength must be between 0 and 1")
        if self.rehearsal_boost < 0:
            raise ValueError("rehearsal_boost must be non-negative")
        if self.rehearsal_decay_rate < 0:
            raise ValueError("rehearsal_decay_rate must be non-negative")


@dataclass
class ImportanceConfig:
    """
    Configuration for importance scoring.

    Final importance = Σ(weight_i * factor_i) normalized to [0, 1]

    Attributes:
        recency_weight: Weight for how recently the memory was created.
        frequency_weight: Weight for access frequency.
        emotional_weight: Weight for emotional valence.
        surprise_weight: Weight for surprise/novelty score.
        entity_weight: Weight for number of entities mentioned.
        explicit_weight: Weight for user-marked importance.
        source_weights: Importance multipliers by memory source.
    """

    recency_weight: float = 0.2
    frequency_weight: float = 0.15
    emotional_weight: float = 0.2
    surprise_weight: float = 0.15
    entity_weight: float = 0.1
    explicit_weight: float = 0.2
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "user_explicit": 1.5,
            "tool_result": 1.2,
            "observation": 1.0,
            "conversation": 0.8,
            "consolidation": 1.1,
            "external": 0.9,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        weights = [
            self.recency_weight,
            self.frequency_weight,
            self.emotional_weight,
            self.surprise_weight,
            self.entity_weight,
            self.explicit_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")


@dataclass
class ConsolidationConfig:
    """
    Configuration for memory consolidation (episodic -> semantic).

    Attributes:
        min_memories_for_consolidation: Minimum episodic memories needed
            before consolidation triggers.
        similarity_threshold: Minimum similarity for memories to be
            grouped together during consolidation.
        consolidation_interval_hours: How often consolidation runs.
        max_memories_per_batch: Maximum memories to process per batch.
        fact_confidence_threshold: Minimum confidence to extract a fact.
        entity_mention_threshold: Minimum mentions to create an entity.
        preserve_source_memories: Keep episodic memories after consolidation.
    """

    min_memories_for_consolidation: int = 5
    similarity_threshold: float = 0.8
    consolidation_interval_hours: float = 24.0
    max_memories_per_batch: int = 100
    fact_confidence_threshold: float = 0.7
    entity_mention_threshold: int = 2
    preserve_source_memories: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.min_memories_for_consolidation < 1:
            raise ValueError("min_memories_for_consolidation must be at least 1")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.consolidation_interval_hours <= 0:
            raise ValueError("consolidation_interval_hours must be positive")
        if self.max_memories_per_batch < 1:
            raise ValueError("max_memories_per_batch must be at least 1")
        if not 0 <= self.fact_confidence_threshold <= 1:
            raise ValueError("fact_confidence_threshold must be between 0 and 1")
        if self.entity_mention_threshold < 1:
            raise ValueError("entity_mention_threshold must be at least 1")


@dataclass
class RetrievalConfig:
    """
    Configuration for memory retrieval.

    Final score = w_sim * similarity + w_str * strength + w_imp * importance + w_rec * recency

    Attributes:
        similarity_weight: Weight for semantic similarity.
        strength_weight: Weight for memory strength (after decay).
        importance_weight: Weight for importance score.
        recency_weight: Weight for recency of last access.
        default_top_k: Default number of memories to retrieve.
        max_top_k: Maximum allowed top_k value.
        mmr_lambda: Lambda for Maximal Marginal Relevance (diversity).
            0 = max diversity, 1 = max relevance.
        min_similarity_threshold: Minimum similarity to include in results.
        include_archived: Whether to include archived memories.
    """

    similarity_weight: float = 0.4
    strength_weight: float = 0.2
    importance_weight: float = 0.2
    recency_weight: float = 0.2
    default_top_k: int = 10
    max_top_k: int = 100
    mmr_lambda: float = 0.7
    min_similarity_threshold: float = 0.3
    include_archived: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        weights = [
            self.similarity_weight,
            self.strength_weight,
            self.importance_weight,
            self.recency_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")
        if self.max_top_k < self.default_top_k:
            raise ValueError("max_top_k must be >= default_top_k")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if not 0 <= self.min_similarity_threshold <= 1:
            raise ValueError("min_similarity_threshold must be between 0 and 1")


@dataclass
class StorageConfig:
    """
    Configuration for storage backends.

    Attributes:
        vector_backend: Vector store backend ("qdrant", "pinecone", "pgvector").
        graph_backend: Graph store backend ("neo4j", "memgraph", "none").
        metadata_backend: Metadata store backend ("postgresql", "sqlite").
        cache_backend: Cache backend ("redis", "memory", "none").
        vector_dimensions: Embedding vector dimensions.
        connection_pool_size: Database connection pool size.
    """

    vector_backend: Literal["qdrant", "pinecone", "pgvector"] = "qdrant"
    graph_backend: Literal["neo4j", "memgraph", "none"] = "none"
    metadata_backend: Literal["postgresql", "sqlite"] = "postgresql"
    cache_backend: Literal["redis", "memory", "none"] = "redis"
    vector_dimensions: int = 1536
    connection_pool_size: int = 10

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.vector_dimensions < 1:
            raise ValueError("vector_dimensions must be at least 1")
        if self.connection_pool_size < 1:
            raise ValueError("connection_pool_size must be at least 1")


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation.

    Attributes:
        provider: Embedding provider ("openai", "cohere", "local").
        model: Model name for embeddings.
        dimensions: Output embedding dimensions.
        batch_size: Batch size for embedding generation.
        max_retries: Maximum retries on failure.
        timeout_seconds: Timeout for embedding requests.
    """

    provider: Literal["openai", "cohere", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class MemorySystemConfig:
    """
    Top-level configuration for the entire memory system.

    Attributes:
        decay: Decay engine configuration.
        importance: Importance scoring configuration.
        consolidation: Consolidation configuration.
        retrieval: Retrieval configuration.
        storage: Storage backend configuration.
        embedding: Embedding configuration.
        agent_id: Default agent ID for memories.
        user_id: Default user ID for memories.
        enable_background_workers: Enable background decay/consolidation.
        log_level: Logging level.
    """

    decay: DecayConfig = field(default_factory=DecayConfig)
    importance: ImportanceConfig = field(default_factory=ImportanceConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    agent_id: str | None = None
    user_id: str | None = None
    enable_background_workers: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def from_env(cls) -> MemorySystemConfig:
        """
        Create configuration from environment variables.

        Environment variables are prefixed with COGNITIVE_MEMORY_.
        Nested configs use double underscore: COGNITIVE_MEMORY_DECAY__DECAY_RATE=0.2

        Returns:
            MemorySystemConfig with values from environment.
        """
        import os

        config = cls()

        # Simple top-level overrides
        if agent_id := os.getenv("COGNITIVE_MEMORY_AGENT_ID"):
            config.agent_id = agent_id
        if user_id := os.getenv("COGNITIVE_MEMORY_USER_ID"):
            config.user_id = user_id
        log_level = os.getenv("COGNITIVE_MEMORY_LOG_LEVEL")
        if log_level and log_level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            config.log_level = log_level  # type: ignore[assignment]
        if workers := os.getenv("COGNITIVE_MEMORY_ENABLE_BACKGROUND_WORKERS"):
            config.enable_background_workers = workers.lower() == "true"

        # Decay config
        if decay_rate := os.getenv("COGNITIVE_MEMORY_DECAY__DECAY_RATE"):
            config.decay.decay_rate = float(decay_rate)
        if min_strength := os.getenv("COGNITIVE_MEMORY_DECAY__MIN_STRENGTH"):
            config.decay.min_strength = float(min_strength)

        # Retrieval config
        if top_k := os.getenv("COGNITIVE_MEMORY_RETRIEVAL__DEFAULT_TOP_K"):
            config.retrieval.default_top_k = int(top_k)

        # Storage config
        vector_backend = os.getenv("COGNITIVE_MEMORY_STORAGE__VECTOR_BACKEND")
        if vector_backend and vector_backend in ("qdrant", "pinecone", "pgvector"):
            config.storage.vector_backend = vector_backend  # type: ignore[assignment]

        # Embedding config
        if embedding_model := os.getenv("COGNITIVE_MEMORY_EMBEDDING__MODEL"):
            config.embedding.model = embedding_model

        return config

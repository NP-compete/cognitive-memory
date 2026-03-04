"""Tests for configuration models."""

import os
from unittest.mock import patch

import pytest

from cognitive_memory.core.config import (
    ConsolidationConfig,
    DecayConfig,
    EmbeddingConfig,
    ImportanceConfig,
    MemorySystemConfig,
    RetrievalConfig,
    StorageConfig,
)


class TestDecayConfig:
    """Tests for DecayConfig."""

    def test_default_values(self) -> None:
        """DecayConfig should have sensible defaults."""
        config = DecayConfig()

        assert config.decay_rate == 0.1
        assert config.min_strength == 0.01
        assert config.rehearsal_boost == 0.2
        assert config.time_unit == "hours"

    def test_custom_values(self) -> None:
        """DecayConfig should accept custom values."""
        config = DecayConfig(
            decay_rate=0.2,
            min_strength=0.05,
            rehearsal_boost=0.3,
            time_unit="days",
        )

        assert config.decay_rate == 0.2
        assert config.min_strength == 0.05
        assert config.rehearsal_boost == 0.3
        assert config.time_unit == "days"

    def test_negative_decay_rate_raises(self) -> None:
        """Negative decay_rate should raise ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be non-negative"):
            DecayConfig(decay_rate=-0.1)

    def test_invalid_min_strength_raises(self) -> None:
        """min_strength outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="min_strength must be between 0 and 1"):
            DecayConfig(min_strength=1.5)

        with pytest.raises(ValueError, match="min_strength must be between 0 and 1"):
            DecayConfig(min_strength=-0.1)

    def test_negative_rehearsal_boost_raises(self) -> None:
        """Negative rehearsal_boost should raise ValueError."""
        with pytest.raises(ValueError, match="rehearsal_boost must be non-negative"):
            DecayConfig(rehearsal_boost=-0.1)


class TestImportanceConfig:
    """Tests for ImportanceConfig."""

    def test_default_values(self) -> None:
        """ImportanceConfig should have sensible defaults."""
        config = ImportanceConfig()

        assert config.recency_weight == 0.2
        assert config.frequency_weight == 0.15
        assert config.emotional_weight == 0.2
        assert "user_explicit" in config.source_weights

    def test_default_source_weights(self) -> None:
        """Source weights should have expected values."""
        config = ImportanceConfig()

        assert config.source_weights["user_explicit"] == 1.5
        assert config.source_weights["conversation"] == 0.8

    def test_negative_weight_raises(self) -> None:
        """Negative weights should raise ValueError."""
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            ImportanceConfig(recency_weight=-0.1)


class TestConsolidationConfig:
    """Tests for ConsolidationConfig."""

    def test_default_values(self) -> None:
        """ConsolidationConfig should have sensible defaults."""
        config = ConsolidationConfig()

        assert config.min_memories_for_consolidation == 5
        assert config.similarity_threshold == 0.8
        assert config.consolidation_interval_hours == 24.0
        assert config.preserve_source_memories is True

    def test_invalid_min_memories_raises(self) -> None:
        """min_memories_for_consolidation < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_memories_for_consolidation must be at least 1"):
            ConsolidationConfig(min_memories_for_consolidation=0)

    def test_invalid_similarity_threshold_raises(self) -> None:
        """similarity_threshold outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            ConsolidationConfig(similarity_threshold=1.5)

    def test_invalid_interval_raises(self) -> None:
        """Non-positive interval should raise ValueError."""
        with pytest.raises(ValueError, match="consolidation_interval_hours must be positive"):
            ConsolidationConfig(consolidation_interval_hours=0)


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_values(self) -> None:
        """RetrievalConfig should have sensible defaults."""
        config = RetrievalConfig()

        assert config.similarity_weight == 0.4
        assert config.default_top_k == 10
        assert config.max_top_k == 100
        assert config.mmr_lambda == 0.7
        assert config.include_archived is False

    def test_negative_weight_raises(self) -> None:
        """Negative weights should raise ValueError."""
        with pytest.raises(ValueError, match="All weights must be non-negative"):
            RetrievalConfig(similarity_weight=-0.1)

    def test_invalid_top_k_raises(self) -> None:
        """Invalid top_k values should raise ValueError."""
        with pytest.raises(ValueError, match="default_top_k must be at least 1"):
            RetrievalConfig(default_top_k=0)

    def test_max_top_k_less_than_default_raises(self) -> None:
        """max_top_k < default_top_k should raise ValueError."""
        with pytest.raises(ValueError, match="max_top_k must be >= default_top_k"):
            RetrievalConfig(default_top_k=20, max_top_k=10)

    def test_invalid_mmr_lambda_raises(self) -> None:
        """mmr_lambda outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="mmr_lambda must be between 0 and 1"):
            RetrievalConfig(mmr_lambda=1.5)


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_values(self) -> None:
        """StorageConfig should have sensible defaults."""
        config = StorageConfig()

        assert config.vector_backend == "qdrant"
        assert config.graph_backend == "none"
        assert config.metadata_backend == "postgresql"
        assert config.cache_backend == "redis"
        assert config.vector_dimensions == 1536

    def test_invalid_dimensions_raises(self) -> None:
        """vector_dimensions < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="vector_dimensions must be at least 1"):
            StorageConfig(vector_dimensions=0)

    def test_invalid_pool_size_raises(self) -> None:
        """connection_pool_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="connection_pool_size must be at least 1"):
            StorageConfig(connection_pool_size=0)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self) -> None:
        """EmbeddingConfig should have sensible defaults."""
        config = EmbeddingConfig()

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536
        assert config.batch_size == 100

    def test_invalid_dimensions_raises(self) -> None:
        """dimensions < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be at least 1"):
            EmbeddingConfig(dimensions=0)

    def test_invalid_timeout_raises(self) -> None:
        """Non-positive timeout should raise ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            EmbeddingConfig(timeout_seconds=0)


class TestMemorySystemConfig:
    """Tests for MemorySystemConfig."""

    def test_default_values(self) -> None:
        """MemorySystemConfig should have sensible defaults."""
        config = MemorySystemConfig()

        assert isinstance(config.decay, DecayConfig)
        assert isinstance(config.importance, ImportanceConfig)
        assert isinstance(config.consolidation, ConsolidationConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert config.agent_id is None
        assert config.enable_background_workers is True
        assert config.log_level == "INFO"

    def test_nested_config_override(self) -> None:
        """Nested configs should be overridable."""
        config = MemorySystemConfig(
            decay=DecayConfig(decay_rate=0.5),
            retrieval=RetrievalConfig(default_top_k=20),
        )

        assert config.decay.decay_rate == 0.5
        assert config.retrieval.default_top_k == 20

    def test_from_env_basic(self) -> None:
        """from_env should read basic environment variables."""
        env_vars = {
            "COGNITIVE_MEMORY_AGENT_ID": "test-agent",
            "COGNITIVE_MEMORY_USER_ID": "test-user",
            "COGNITIVE_MEMORY_LOG_LEVEL": "DEBUG",
            "COGNITIVE_MEMORY_ENABLE_BACKGROUND_WORKERS": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = MemorySystemConfig.from_env()

            assert config.agent_id == "test-agent"
            assert config.user_id == "test-user"
            assert config.log_level == "DEBUG"
            assert config.enable_background_workers is False

    def test_from_env_nested(self) -> None:
        """from_env should read nested config environment variables."""
        env_vars = {
            "COGNITIVE_MEMORY_DECAY__DECAY_RATE": "0.25",
            "COGNITIVE_MEMORY_RETRIEVAL__DEFAULT_TOP_K": "15",
            "COGNITIVE_MEMORY_STORAGE__VECTOR_BACKEND": "pinecone",
            "COGNITIVE_MEMORY_EMBEDDING__MODEL": "text-embedding-3-large",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = MemorySystemConfig.from_env()

            assert config.decay.decay_rate == 0.25
            assert config.retrieval.default_top_k == 15
            assert config.storage.vector_backend == "pinecone"
            assert config.embedding.model == "text-embedding-3-large"

    def test_from_env_ignores_invalid_log_level(self) -> None:
        """from_env should ignore invalid log levels."""
        env_vars = {
            "COGNITIVE_MEMORY_LOG_LEVEL": "INVALID",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = MemorySystemConfig.from_env()
            assert config.log_level == "INFO"  # Default

    def test_from_env_ignores_invalid_backend(self) -> None:
        """from_env should ignore invalid backend values."""
        env_vars = {
            "COGNITIVE_MEMORY_STORAGE__VECTOR_BACKEND": "invalid",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = MemorySystemConfig.from_env()
            assert config.storage.vector_backend == "qdrant"  # Default

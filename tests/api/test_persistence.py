"""
Persistence Layer Tests

Comprehensive tests for database persistence including:
- In-memory repository
- SQLite repository
- Entity models
- Unit of Work pattern
"""

import asyncio
import os
import tempfile
from datetime import datetime
from typing import Dict, Any

import pytest

from api.persistence import (
    BaseEntity,
    Agent,
    Conversation,
    TrainingJob,
    APIKey,
    InMemoryRepository,
    SQLiteRepository,
    UnitOfWork,
    DatabaseConfig,
    DatabaseBackend,
    init_database,
    get_database,
    close_database,
)


# ============================================================================
# Entity Model Tests
# ============================================================================

class TestBaseEntity:
    """Tests for BaseEntity class."""

    def test_default_id_generation(self):
        """Entity should generate UUID by default."""
        entity = BaseEntity()
        assert entity.id is not None
        assert len(entity.id) == 36  # UUID format

    def test_default_timestamps(self):
        """Entity should have default timestamps."""
        entity = BaseEntity()
        assert entity.created_at is not None
        assert entity.updated_at is not None
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)

    def test_to_dict(self):
        """Entity should serialize to dictionary."""
        entity = BaseEntity(id="test-id")
        data = entity.to_dict()

        assert data["id"] == "test-id"
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self):
        """Entity should deserialize from dictionary."""
        data = {
            "id": "test-id",
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T10:30:00",
        }
        entity = BaseEntity.from_dict(data)

        assert entity.id == "test-id"
        assert isinstance(entity.created_at, datetime)


class TestAgentEntity:
    """Tests for Agent entity."""

    def test_agent_creation(self):
        """Test agent with all fields."""
        agent = Agent(
            name="TestAgent",
            model_name="gpt-4",
            system_prompt="You are helpful.",
            temperature=0.7,
            max_tokens=2048,
        )

        assert agent.name == "TestAgent"
        assert agent.model_name == "gpt-4"
        assert agent.system_prompt == "You are helpful."
        assert agent.temperature == 0.7
        assert agent.max_tokens == 2048
        assert agent.status == "active"

    def test_agent_defaults(self):
        """Test agent default values."""
        agent = Agent()

        assert agent.name == ""
        assert agent.model_name == "gpt-4"
        assert agent.temperature == 0.7
        assert agent.max_tokens == 2048
        assert agent.total_tokens_used == 0

    def test_agent_to_dict(self):
        """Test agent serialization."""
        agent = Agent(name="TestAgent")
        data = agent.to_dict()

        assert data["name"] == "TestAgent"
        assert "model_name" in data
        assert "config" in data


class TestConversationEntity:
    """Tests for Conversation entity."""

    def test_conversation_creation(self):
        """Test conversation with all fields."""
        conv = Conversation(
            agent_id="agent-123",
            user_id="user-456",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert conv.agent_id == "agent-123"
        assert conv.user_id == "user-456"
        assert len(conv.messages) == 1
        assert conv.status == "active"

    def test_conversation_defaults(self):
        """Test conversation default values."""
        conv = Conversation()

        assert conv.agent_id == ""
        assert conv.messages == []
        assert conv.metadata == {}
        assert conv.total_tokens == 0


class TestTrainingJobEntity:
    """Tests for TrainingJob entity."""

    def test_training_job_creation(self):
        """Test training job with all fields."""
        job = TrainingJob(
            agent_id="agent-123",
            strategy="grpo",
            prompts=["prompt1", "prompt2"],
            num_iterations=10,
        )

        assert job.agent_id == "agent-123"
        assert job.strategy == "grpo"
        assert len(job.prompts) == 2
        assert job.num_iterations == 10
        assert job.status == "pending"

    def test_training_job_defaults(self):
        """Test training job defaults."""
        job = TrainingJob()

        assert job.status == "pending"
        assert job.current_iteration == 0
        assert job.average_reward == 0.0


class TestAPIKeyEntity:
    """Tests for APIKey entity."""

    def test_api_key_creation(self):
        """Test API key creation."""
        key = APIKey(
            key_hash="hash123",
            name="Test Key",
            user_id="user-123",
            roles=["admin", "trainer"],
        )

        assert key.key_hash == "hash123"
        assert key.name == "Test Key"
        assert "admin" in key.roles
        assert key.is_active is True


# ============================================================================
# In-Memory Repository Tests
# ============================================================================

class TestInMemoryRepository:
    """Tests for in-memory repository."""

    @pytest.fixture
    def repo(self):
        """Create a fresh repository for each test."""
        return InMemoryRepository(Agent)

    @pytest.mark.asyncio
    async def test_create(self, repo):
        """Test entity creation."""
        agent = Agent(name="TestAgent")
        created = await repo.create(agent)

        assert created.id == agent.id
        assert created.name == "TestAgent"
        assert created.created_at is not None

    @pytest.mark.asyncio
    async def test_get(self, repo):
        """Test entity retrieval."""
        agent = Agent(name="TestAgent")
        await repo.create(agent)

        retrieved = await repo.get(agent.id)
        assert retrieved is not None
        assert retrieved.name == "TestAgent"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, repo):
        """Test getting non-existent entity."""
        result = await repo.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update(self, repo):
        """Test entity update."""
        agent = Agent(name="Original")
        await repo.create(agent)

        agent.name = "Updated"
        updated = await repo.update(agent)

        assert updated.name == "Updated"
        assert updated.updated_at > updated.created_at

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, repo):
        """Test updating non-existent entity."""
        agent = Agent(id="nonexistent", name="Test")

        with pytest.raises(ValueError, match="not found"):
            await repo.update(agent)

    @pytest.mark.asyncio
    async def test_delete(self, repo):
        """Test entity deletion."""
        agent = Agent(name="ToDelete")
        await repo.create(agent)

        result = await repo.delete(agent.id)
        assert result is True

        retrieved = await repo.get(agent.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, repo):
        """Test deleting non-existent entity."""
        result = await repo.delete("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_list(self, repo):
        """Test listing entities."""
        for i in range(5):
            await repo.create(Agent(name=f"Agent{i}"))

        items = await repo.list(limit=10)
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_list_pagination(self, repo):
        """Test listing with pagination."""
        for i in range(10):
            await repo.create(Agent(name=f"Agent{i}"))

        page1 = await repo.list(limit=5, offset=0)
        page2 = await repo.list(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5
        # Should be different items
        page1_ids = {a.id for a in page1}
        page2_ids = {a.id for a in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_list_with_filter(self, repo):
        """Test listing with filters."""
        await repo.create(Agent(name="Active", status="active"))
        await repo.create(Agent(name="Inactive", status="inactive"))
        await repo.create(Agent(name="Active2", status="active"))

        items = await repo.list(filters={"status": "active"})
        assert len(items) == 2
        assert all(a.status == "active" for a in items)

    @pytest.mark.asyncio
    async def test_count(self, repo):
        """Test counting entities."""
        for i in range(5):
            await repo.create(Agent(name=f"Agent{i}"))

        count = await repo.count()
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_with_filter(self, repo):
        """Test counting with filters."""
        await repo.create(Agent(status="active"))
        await repo.create(Agent(status="inactive"))
        await repo.create(Agent(status="active"))

        count = await repo.count(filters={"status": "active"})
        assert count == 2

    @pytest.mark.asyncio
    async def test_clear(self, repo):
        """Test clearing all entities."""
        for i in range(5):
            await repo.create(Agent(name=f"Agent{i}"))

        await repo.clear()
        count = await repo.count()
        assert count == 0


# ============================================================================
# SQLite Repository Tests
# ============================================================================

class TestSQLiteRepository:
    """Tests for SQLite repository."""

    @pytest.fixture
    async def repo(self):
        """Create a SQLite repository with temp file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        repo = SQLiteRepository(Agent, db_path)
        await repo.connect()
        yield repo
        await repo.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_create_and_get(self, repo):
        """Test create and get operations."""
        agent = Agent(name="SQLiteAgent")
        created = await repo.create(agent)

        retrieved = await repo.get(created.id)
        assert retrieved is not None
        assert retrieved.name == "SQLiteAgent"

    @pytest.mark.asyncio
    async def test_update(self, repo):
        """Test update operation."""
        agent = Agent(name="Original")
        await repo.create(agent)

        agent.name = "Updated"
        await repo.update(agent)

        retrieved = await repo.get(agent.id)
        assert retrieved.name == "Updated"

    @pytest.mark.asyncio
    async def test_delete(self, repo):
        """Test delete operation."""
        agent = Agent(name="ToDelete")
        await repo.create(agent)

        result = await repo.delete(agent.id)
        assert result is True

        retrieved = await repo.get(agent.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list(self, repo):
        """Test list operation."""
        for i in range(3):
            await repo.create(Agent(name=f"Agent{i}"))

        items = await repo.list()
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_count(self, repo):
        """Test count operation."""
        for i in range(3):
            await repo.create(Agent(name=f"Agent{i}"))

        count = await repo.count()
        assert count == 3


# ============================================================================
# Unit of Work Tests
# ============================================================================

class TestUnitOfWork:
    """Tests for Unit of Work pattern."""

    @pytest.mark.asyncio
    async def test_memory_backend_initialization(self):
        """Test UoW with memory backend."""
        config = DatabaseConfig(backend=DatabaseBackend.MEMORY)
        uow = UnitOfWork(config)
        await uow.connect()

        assert uow.agents is not None
        assert uow.conversations is not None
        assert uow.training_jobs is not None
        assert uow.api_keys is not None

        await uow.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test UoW as context manager."""
        config = DatabaseConfig(backend=DatabaseBackend.MEMORY)

        async with UnitOfWork(config) as uow:
            agent = Agent(name="ContextTest")
            await uow.agents.create(agent)

            retrieved = await uow.agents.get(agent.id)
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_multiple_repositories(self):
        """Test working with multiple repositories."""
        config = DatabaseConfig(backend=DatabaseBackend.MEMORY)

        async with UnitOfWork(config) as uow:
            # Create agent
            agent = Agent(name="TestAgent")
            await uow.agents.create(agent)

            # Create conversation
            conv = Conversation(agent_id=agent.id)
            await uow.conversations.create(conv)

            # Create training job
            job = TrainingJob(agent_id=agent.id)
            await uow.training_jobs.create(job)

            # Verify all were created
            assert await uow.agents.count() == 1
            assert await uow.conversations.count() == 1
            assert await uow.training_jobs.count() == 1

    @pytest.mark.asyncio
    async def test_not_connected_error(self):
        """Test error when accessing repository before connect."""
        config = DatabaseConfig(backend=DatabaseBackend.MEMORY)
        uow = UnitOfWork(config)

        with pytest.raises(RuntimeError, match="not connected"):
            _ = uow.agents


# ============================================================================
# Database Configuration Tests
# ============================================================================

class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()

        assert config.backend == DatabaseBackend.MEMORY
        assert config.pool_size == 5
        assert config.auto_migrate is True

    def test_config_from_env(self):
        """Test loading config from environment."""
        os.environ["DB_BACKEND"] = "memory"
        os.environ["DB_POOL_SIZE"] = "10"

        config = DatabaseConfig.from_env()

        assert config.backend == DatabaseBackend.MEMORY
        assert config.pool_size == 10

        # Cleanup
        del os.environ["DB_BACKEND"]
        del os.environ["DB_POOL_SIZE"]


# ============================================================================
# Global Database Instance Tests
# ============================================================================

class TestGlobalDatabase:
    """Tests for global database instance."""

    @pytest.mark.asyncio
    async def test_init_and_get_database(self):
        """Test initializing and getting global database."""
        config = DatabaseConfig(backend=DatabaseBackend.MEMORY)
        db = await init_database(config)

        assert db is not None
        assert get_database() is db

        await close_database()
        assert get_database() is None

    @pytest.mark.asyncio
    async def test_close_without_init(self):
        """Test closing database that was never initialized."""
        # Should not raise
        await close_database()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

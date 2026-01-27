"""
Tests for memory management and evaluation framework:
- Conversation memory (short-term, long-term, episodic, semantic)
- Entity and fact extraction
- Evaluation metrics
- A/B testing
- Streaming services
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Memory Management Tests
# ============================================================================


class TestMemoryConfig:
    """Test MemoryConfig settings."""

    def test_memory_config_defaults(self):
        """Test default memory configuration."""
        from stateset_agents.core.memory import MemoryConfig

        config = MemoryConfig()
        assert config.max_short_term_turns == 20
        assert config.max_short_term_tokens == 4000
        assert config.enable_entity_extraction == True
        assert config.enable_fact_extraction == True

    def test_memory_config_custom(self):
        """Test custom memory configuration."""
        from stateset_agents.core.memory import MemoryConfig

        config = MemoryConfig(
            max_short_term_turns=50,
            max_short_term_tokens=8000,
            enable_entity_extraction=False,
        )
        assert config.max_short_term_turns == 50
        assert config.max_short_term_tokens == 8000
        assert config.enable_entity_extraction == False


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test creating a memory entry."""
        from stateset_agents.core.memory import MemoryEntry, MemoryType

        entry = MemoryEntry(
            id="test_123",
            content="This is a test memory",
            memory_type=MemoryType.SHORT_TERM,
        )

        assert entry.id == "test_123"
        assert entry.content == "This is a test memory"
        assert entry.memory_type == MemoryType.SHORT_TERM
        assert entry.importance == 0.5

    def test_memory_entry_decay(self):
        """Test importance decay."""
        from stateset_agents.core.memory import MemoryEntry, MemoryType

        entry = MemoryEntry(
            id="test",
            content="content",
            memory_type=MemoryType.SHORT_TERM,
            importance=1.0,
        )

        entry.decay_importance(0.9)
        assert entry.importance == 0.9

        entry.decay_importance(0.9)
        assert entry.importance == pytest.approx(0.81)

    def test_memory_entry_to_dict(self):
        """Test converting memory entry to dict."""
        from stateset_agents.core.memory import MemoryEntry, MemoryType

        entry = MemoryEntry(
            id="test",
            content="content",
            memory_type=MemoryType.LONG_TERM,
        )

        data = entry.to_dict()
        assert data["id"] == "test"
        assert data["content"] == "content"
        assert data["memory_type"] == "long_term"


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test creating an entity."""
        from stateset_agents.core.memory import Entity

        entity = Entity(
            name="Alice",
            entity_type="person",
        )

        assert entity.name == "Alice"
        assert entity.entity_type == "person"
        assert entity.mentions == 1

    def test_entity_update(self):
        """Test updating entity with new mention."""
        from stateset_agents.core.memory import Entity

        entity = Entity(name="Bob", entity_type="person")
        entity.update("Bob went to the store")
        entity.update("Bob came home")

        assert entity.mentions == 3
        assert len(entity.context) == 2


class TestEntityExtractor:
    """Test EntityExtractor functionality."""

    def test_extract_email(self):
        """Test extracting email addresses."""
        from stateset_agents.core.memory import EntityExtractor

        extractor = EntityExtractor()
        entities = extractor.extract("Contact me at test@example.com")

        assert "email" in entities
        assert "test@example.com" in entities["email"]

    def test_extract_phone(self):
        """Test extracting phone numbers."""
        from stateset_agents.core.memory import EntityExtractor

        extractor = EntityExtractor()
        entities = extractor.extract("Call me at 555-123-4567")

        assert "phone" in entities
        assert "555-123-4567" in entities["phone"]

    def test_extract_money(self):
        """Test extracting monetary amounts."""
        from stateset_agents.core.memory import EntityExtractor

        extractor = EntityExtractor()
        entities = extractor.extract("The price is $99.99")

        assert "money" in entities
        assert "$99.99" in entities["money"]


class TestContextWindow:
    """Test ContextWindow dataclass."""

    def test_context_window_creation(self):
        """Test creating a context window."""
        from stateset_agents.core.memory import ContextWindow

        context = ContextWindow(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        )

        assert len(context.messages) == 2
        assert context.summary is None
        assert context.total_tokens == 0

    def test_context_window_to_messages(self):
        """Test converting context to messages format."""
        from stateset_agents.core.memory import ContextWindow

        context = ContextWindow(
            messages=[{"role": "user", "content": "Hello"}],
            summary="Previous conversation about greetings",
            entities={"person": ["Alice"]},
        )

        messages = context.to_messages(include_summary=True)

        # Should include system message with summary
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "Previous conversation summary" in messages[0]["content"]


class TestConversationMemory:
    """Test ConversationMemory functionality."""

    def test_memory_initialization(self):
        """Test memory initializes correctly."""
        from stateset_agents.core.memory import ConversationMemory, MemoryConfig

        config = MemoryConfig(max_short_term_turns=10)
        memory = ConversationMemory(config)

        # Memory should start empty
        context = memory.get_context_for_generation()
        assert len(context.messages) == 0

    def test_add_turn_sync(self):
        """Test adding conversation turns."""
        from stateset_agents.core.memory import ConversationMemory, MemoryConfig

        # Use a larger limit to avoid overflow handling
        config = MemoryConfig(max_short_term_turns=100)
        memory = ConversationMemory(config)

        memory.add_turn({"role": "user", "content": "Hello"})
        memory.add_turn({"role": "assistant", "content": "Hi there!"})

        context = memory.get_context_for_generation()
        assert len(context.messages) == 2
        assert context.messages[0]["role"] == "user"
        assert context.messages[1]["role"] == "assistant"

    def test_working_memory(self):
        """Test working memory for task context."""
        from stateset_agents.core.memory import ConversationMemory

        memory = ConversationMemory()

        # Set working memory
        memory.set_working_memory("current_task", "Analyzing code")
        memory.set_working_memory("user_preference", "verbose")

        assert memory.get_working_memory("current_task") == "Analyzing code"
        assert memory.get_working_memory("user_preference") == "verbose"
        assert memory.get_working_memory("nonexistent") is None

    def test_clear_working_memory(self):
        """Test clearing working memory."""
        from stateset_agents.core.memory import ConversationMemory

        memory = ConversationMemory()
        memory.set_working_memory("key", "value")
        memory.clear_working_memory()

        assert memory.get_working_memory("key") is None


# ============================================================================
# Evaluation Framework Tests
# ============================================================================


class TestEvaluationConfig:
    """Test EvaluationConfig settings."""

    def test_evaluation_config_defaults(self):
        """Test default evaluation configuration."""
        from stateset_agents.core.evaluation import EvaluationConfig

        config = EvaluationConfig()
        assert "relevance" in config.metrics
        assert config.parallel_workers == 4
        assert config.timeout_seconds == 30.0

    def test_evaluation_config_custom(self):
        """Test custom evaluation configuration."""
        from stateset_agents.core.evaluation import EvaluationConfig

        config = EvaluationConfig(
            metrics=["relevance", "helpfulness"],
            parallel_workers=8,
            timeout_seconds=60.0,
        )
        assert len(config.metrics) == 2
        assert config.parallel_workers == 8


class TestEvaluationSample:
    """Test EvaluationSample dataclass."""

    def test_sample_creation(self):
        """Test creating an evaluation sample."""
        from stateset_agents.core.evaluation import EvaluationSample

        sample = EvaluationSample(
            id="test_001",
            input="What is Python?",
            expected_output="Python is a programming language.",
        )

        assert sample.id == "test_001"
        assert sample.input == "What is Python?"
        assert sample.expected_output == "Python is a programming language."


class TestEvaluationMetrics:
    """Test individual evaluation metrics."""

    @pytest.mark.asyncio
    async def test_relevance_metric(self):
        """Test relevance scoring."""
        from stateset_agents.core.evaluation import RelevanceMetric

        metric = RelevanceMetric()
        result = await metric.compute(
            input_text="What is Python programming?",
            output_text="Python is a high-level programming language known for its simplicity.",
        )

        assert result.metric_name == "relevance"
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_coherence_metric(self):
        """Test coherence scoring."""
        from stateset_agents.core.evaluation import CoherenceMetric

        metric = CoherenceMetric()

        # Coherent response
        coherent_result = await metric.compute(
            input_text="Tell me about Python",
            output_text="Python is a programming language. It is widely used for web development. Many companies use Python for their projects.",
        )

        # Incoherent response
        incoherent_result = await metric.compute(
            input_text="Tell me about Python",
            output_text="",
        )

        assert coherent_result.value > incoherent_result.value

    @pytest.mark.asyncio
    async def test_helpfulness_metric(self):
        """Test helpfulness scoring."""
        from stateset_agents.core.evaluation import HelpfulnessMetric

        metric = HelpfulnessMetric()

        helpful_result = await metric.compute(
            input_text="How do I install Python?",
            output_text="To install Python, visit python.org, download the installer for your operating system, and run it. Make sure to check 'Add to PATH' during installation.",
        )

        unhelpful_result = await metric.compute(
            input_text="How do I install Python?",
            output_text="I don't know.",
        )

        assert helpful_result.value > unhelpful_result.value

    @pytest.mark.asyncio
    async def test_latency_metric(self):
        """Test latency measurement."""
        from stateset_agents.core.evaluation import LatencyMetric

        metric = LatencyMetric()

        result = await metric.compute(
            input_text="test",
            output_text="response",
            context={"latency_ms": 100},
        )

        assert result.metric_name == "latency"

    @pytest.mark.asyncio
    async def test_safety_metric(self):
        """Test safety scoring."""
        from stateset_agents.core.evaluation import SafetyMetric

        metric = SafetyMetric()

        safe_result = await metric.compute(
            input_text="Tell me about Python",
            output_text="Python is a great programming language for beginners.",
        )

        # Safety metric should give high score to safe content
        assert safe_result.value >= 0.5


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test creating a metric result."""
        from stateset_agents.core.evaluation import MetricResult

        result = MetricResult(
            metric_name="test_metric",
            value=0.85,
            metadata={"key": "value"},
        )

        assert result.metric_name == "test_metric"
        assert result.value == 0.85
        assert result.metadata["key"] == "value"


class TestAgentEvaluator:
    """Test the AgentEvaluator class."""

    @pytest.mark.asyncio
    async def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        from stateset_agents.core.evaluation import AgentEvaluator, EvaluationConfig

        config = EvaluationConfig(metrics=["relevance"])
        evaluator = AgentEvaluator(config)

        assert evaluator is not None
        assert evaluator.config == config


class TestABTestRunner:
    """Test A/B test runner."""

    def test_ab_test_runner_creation(self):
        """Test creating A/B test runner."""
        from stateset_agents.core.evaluation import ABTestRunner, EvaluationConfig

        config = EvaluationConfig(metrics=["relevance", "helpfulness"])
        runner = ABTestRunner(config)

        assert runner.config == config
        assert runner.evaluator is not None

    def test_ab_test_runner_default_config(self):
        """Test A/B test runner with default config."""
        from stateset_agents.core.evaluation import ABTestRunner

        runner = ABTestRunner()

        assert runner.config is not None
        assert "relevance" in runner.config.metrics


# ============================================================================
# Streaming Service Tests
# ============================================================================


class TestStreamingService:
    """Test streaming service functionality."""

    @pytest.mark.asyncio
    async def test_stream_event_creation(self):
        """Test creating stream events."""
        from api.streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            data={"token": "Hello"},
        )

        assert event.event_type == StreamEventType.TOKEN
        assert event.data["token"] == "Hello"
        assert event.id is not None

    @pytest.mark.asyncio
    async def test_stream_event_to_sse(self):
        """Test converting event to SSE format."""
        from api.streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            event_type=StreamEventType.TOKEN,
            data={"token": "World"},
        )

        sse = event.to_sse()

        assert "event: token" in sse
        assert "data:" in sse
        assert "World" in sse

    @pytest.mark.asyncio
    async def test_stream_event_to_dict(self):
        """Test converting event to dict."""
        from api.streaming import StreamEvent, StreamEventType

        event = StreamEvent(
            event_type=StreamEventType.CHUNK,
            data={"content": "test"},
        )

        data = event.to_dict()
        assert data["type"] == "chunk"
        assert data["data"]["content"] == "test"

    @pytest.mark.asyncio
    async def test_streaming_service_mock_response(self):
        """Test streaming service with mock agent."""
        from api.streaming import StreamingService, StreamEventType

        service = StreamingService(agent=None)

        events = []
        async for event in service.stream_response("Hello"):
            events.append(event)

        # Should have metadata, tokens, and done events
        event_types = [e.event_type for e in events]
        assert StreamEventType.METADATA in event_types
        assert StreamEventType.TOKEN in event_types
        assert StreamEventType.DONE in event_types

    @pytest.mark.asyncio
    async def test_batch_item_creation(self):
        """Test creating batch items."""
        from api.streaming import BatchItem

        item = BatchItem(
            id="item_1",
            message="Hello, world!",
            context={"key": "value"},
        )

        assert item.id == "item_1"
        assert item.message == "Hello, world!"
        assert item.context["key"] == "value"

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing."""
        from api.streaming import StreamingService, BatchItem

        service = StreamingService(agent=None)

        items = [
            BatchItem(id="1", message="Hello"),
            BatchItem(id="2", message="World"),
        ]

        result = await service.process_batch(items, parallel=2)

        assert result.total_items == 2
        assert result.completed + result.failed == 2
        assert result.batch_id is not None

    @pytest.mark.asyncio
    async def test_batch_with_progress(self):
        """Test batch processing with progress callback."""
        from api.streaming import StreamingService, BatchItem

        service = StreamingService(agent=None)

        items = [
            BatchItem(id="1", message="Test 1"),
            BatchItem(id="2", message="Test 2"),
        ]

        progress_updates = []

        def on_progress(done, total, result):
            progress_updates.append((done, total))

        await service.process_batch(items, parallel=2, progress_callback=on_progress)

        assert len(progress_updates) == 2

    @pytest.mark.asyncio
    async def test_batch_response_model(self):
        """Test BatchResponse model."""
        from api.streaming import BatchResponse, BatchItemResult

        result = BatchResponse(
            batch_id="batch_123",
            total_items=10,
            completed=8,
            failed=2,
            results=[],
            total_latency_ms=1500.0,
        )

        assert result.batch_id == "batch_123"
        assert result.total_items == 10
        assert result.completed == 8

    @pytest.mark.asyncio
    async def test_streaming_batch_progress(self):
        """Test streaming batch progress events."""
        from api.streaming import StreamingService, BatchItem, StreamEventType

        service = StreamingService(agent=None)

        items = [
            BatchItem(id="1", message="Test"),
        ]

        events = []
        async for event in service.stream_batch_progress(items, parallel=1):
            events.append(event)

        # Should have metadata and done events
        event_types = [e.event_type for e in events]
        assert StreamEventType.METADATA in event_types
        assert StreamEventType.DONE in event_types


class TestStreamingRequest:
    """Test StreamingRequest model."""

    def test_streaming_request_defaults(self):
        """Test default streaming request values."""
        from api.streaming import StreamingRequest

        request = StreamingRequest(message="Hello")

        assert request.message == "Hello"
        assert request.temperature == 0.7
        assert request.max_tokens == 512
        assert request.stream_tokens == True


class TestStreamEventType:
    """Test StreamEventType enum."""

    def test_event_type_values(self):
        """Test all event type values."""
        from api.streaming import StreamEventType

        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.CHUNK.value == "chunk"
        assert StreamEventType.METADATA.value == "metadata"
        assert StreamEventType.TOOL_CALL.value == "tool_call"
        assert StreamEventType.ERROR.value == "error"
        assert StreamEventType.DONE.value == "done"


# ============================================================================
# Integration Tests
# ============================================================================


class TestMemoryEvaluationIntegration:
    """Integration tests for memory and evaluation."""

    @pytest.mark.asyncio
    async def test_streaming_complete_workflow(self):
        """Test complete streaming workflow."""
        from api.streaming import StreamingService, StreamEventType

        service = StreamingService(agent=None)

        # Stream a response
        all_tokens = []
        metadata = None
        async for event in service.stream_response("Hello, how are you?"):
            if event.event_type == StreamEventType.METADATA:
                metadata = event.data
            elif event.event_type == StreamEventType.TOKEN:
                all_tokens.append(event.data.get("token", ""))
            elif event.event_type == StreamEventType.DONE:
                break

        assert metadata is not None
        assert len(all_tokens) > 0
        assert "stream_id" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

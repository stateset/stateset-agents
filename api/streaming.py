"""
Streaming API Endpoints for StateSet Agents

This module provides Server-Sent Events (SSE) and streaming endpoints
for real-time agent responses.

Features:
- SSE streaming for chat responses
- WebSocket streaming support
- Batch processing with streaming status
- Progress callbacks for long operations

Example:
    # Client-side SSE consumption:
    const eventSource = new EventSource('/api/v1/chat/stream?message=Hello');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.token);
    };
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TOKEN = "token"  # Individual token
    CHUNK = "chunk"  # Text chunk
    METADATA = "metadata"  # Response metadata
    TOOL_CALL = "tool_call"  # Function/tool call
    TOOL_RESULT = "tool_result"  # Tool execution result
    ERROR = "error"  # Error event
    DONE = "done"  # Stream complete


@dataclass
class StreamEvent:
    """A streaming event.

    Attributes:
        event_type: Type of event
        data: Event data
        id: Event ID
        timestamp: Event timestamp
    """

    event_type: StreamEventType
    data: Any
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data_json = json.dumps({
            "type": self.event_type.value,
            "data": self.data,
            "id": self.id,
            "timestamp": self.timestamp,
        })
        return f"id: {self.id}\nevent: {self.event_type.value}\ndata: {data_json}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "id": self.id,
            "timestamp": self.timestamp,
        }


class StreamingRequest(BaseModel):
    """Request for streaming chat."""

    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    system_prompt: Optional[str] = Field(None, description="System prompt override")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    stream_tokens: bool = Field(True, description="Stream individual tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how are you?",
                "conversation_id": "conv_abc123",
                "temperature": 0.7,
                "max_tokens": 512,
            }
        }


class BatchItem(BaseModel):
    """Single item in a batch request."""

    id: str = Field(..., description="Unique item ID")
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    """Batch processing request."""

    items: List[BatchItem] = Field(..., min_length=1, max_length=100)
    parallel: int = Field(4, ge=1, le=10, description="Parallel processing")
    stream_progress: bool = Field(True, description="Stream progress updates")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {"id": "1", "message": "Hello"},
                    {"id": "2", "message": "How are you?"},
                ],
                "parallel": 4,
            }
        }


class BatchItemResult(BaseModel):
    """Result for a batch item."""

    id: str
    status: str  # "completed", "failed"
    response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float


class BatchResponse(BaseModel):
    """Batch processing response."""

    batch_id: str
    total_items: int
    completed: int
    failed: int
    results: List[BatchItemResult]
    total_latency_ms: float


class StreamingService:
    """Service for streaming operations."""

    def __init__(self, agent: Any = None):
        """Initialize streaming service.

        Args:
            agent: Agent to use for generation
        """
        self.agent = agent
        self._active_streams: Dict[str, Dict[str, Any]] = {}

    def set_agent(self, agent: Any) -> None:
        """Set the agent for streaming."""
        self.agent = agent

    async def stream_response(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the agent.

        Args:
            message: User message
            conversation_id: Optional conversation ID
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            StreamEvent objects
        """
        stream_id = uuid.uuid4().hex[:12]
        self._active_streams[stream_id] = {
            "started": time.time(),
            "conversation_id": conversation_id,
        }

        try:
            # Emit metadata
            yield StreamEvent(
                event_type=StreamEventType.METADATA,
                data={
                    "stream_id": stream_id,
                    "conversation_id": conversation_id,
                    "model": getattr(self.agent.config, "model_name", "unknown") if self.agent else "mock",
                },
            )

            if self.agent is None:
                # Mock streaming for testing
                mock_response = "Hello! I'm a mock agent response. How can I help you today?"
                for word in mock_response.split():
                    yield StreamEvent(
                        event_type=StreamEventType.TOKEN,
                        data={"token": word + " "},
                    )
                    await asyncio.sleep(0.05)
            else:
                # Use agent's streaming method if available
                if hasattr(self.agent, "generate_response_stream"):
                    messages = [{"role": "user", "content": message}]
                    if system_prompt:
                        messages.insert(0, {"role": "system", "content": system_prompt})

                    async for token in self.agent.generate_response_stream(messages):
                        yield StreamEvent(
                            event_type=StreamEventType.TOKEN,
                            data={"token": token},
                        )
                else:
                    # Fallback to non-streaming
                    response = await self.agent.generate_response(message)
                    yield StreamEvent(
                        event_type=StreamEventType.CHUNK,
                        data={"content": response},
                    )

            # Emit completion
            yield StreamEvent(
                event_type=StreamEventType.DONE,
                data={
                    "stream_id": stream_id,
                    "duration_ms": (time.time() - self._active_streams[stream_id]["started"]) * 1000,
                },
            )

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
            )
        finally:
            self._active_streams.pop(stream_id, None)

    async def process_batch(
        self,
        items: List[BatchItem],
        parallel: int = 4,
        progress_callback: Optional[Callable[[int, int, BatchItemResult], None]] = None,
    ) -> BatchResponse:
        """Process a batch of requests.

        Args:
            items: Batch items to process
            parallel: Number of parallel workers
            progress_callback: Optional callback for progress

        Returns:
            BatchResponse with all results
        """
        batch_id = uuid.uuid4().hex[:12]
        start_time = time.time()
        results: List[BatchItemResult] = []
        semaphore = asyncio.Semaphore(parallel)

        async def process_item(item: BatchItem) -> BatchItemResult:
            async with semaphore:
                item_start = time.time()
                try:
                    if self.agent:
                        response = await self.agent.generate_response(item.message)
                    else:
                        # Mock response
                        await asyncio.sleep(0.1)
                        response = f"Mock response for: {item.message[:50]}"

                    return BatchItemResult(
                        id=item.id,
                        status="completed",
                        response=response,
                        latency_ms=(time.time() - item_start) * 1000,
                    )
                except Exception as e:
                    return BatchItemResult(
                        id=item.id,
                        status="failed",
                        error=str(e),
                        latency_ms=(time.time() - item_start) * 1000,
                    )

        # Process items
        tasks = [process_item(item) for item in items]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(items), result)

        # Sort by original order
        id_order = {item.id: i for i, item in enumerate(items)}
        results.sort(key=lambda r: id_order.get(r.id, 0))

        return BatchResponse(
            batch_id=batch_id,
            total_items=len(items),
            completed=sum(1 for r in results if r.status == "completed"),
            failed=sum(1 for r in results if r.status == "failed"),
            results=results,
            total_latency_ms=(time.time() - start_time) * 1000,
        )

    async def stream_batch_progress(
        self,
        items: List[BatchItem],
        parallel: int = 4,
    ) -> AsyncIterator[StreamEvent]:
        """Process batch with streaming progress.

        Args:
            items: Batch items
            parallel: Parallel workers

        Yields:
            Progress events
        """
        batch_id = uuid.uuid4().hex[:12]
        completed = 0

        yield StreamEvent(
            event_type=StreamEventType.METADATA,
            data={
                "batch_id": batch_id,
                "total_items": len(items),
            },
        )

        def on_progress(done: int, total: int, result: BatchItemResult):
            nonlocal completed
            completed = done

        # Start batch processing
        batch_task = asyncio.create_task(
            self.process_batch(items, parallel, on_progress)
        )

        # Stream progress
        last_completed = 0
        while not batch_task.done():
            if completed > last_completed:
                yield StreamEvent(
                    event_type=StreamEventType.CHUNK,
                    data={
                        "completed": completed,
                        "total": len(items),
                        "progress_pct": completed / len(items) * 100,
                    },
                )
                last_completed = completed
            await asyncio.sleep(0.1)

        # Get final results
        try:
            batch_result = await batch_task
            yield StreamEvent(
                event_type=StreamEventType.DONE,
                data=batch_result.model_dump(),
            )
        except Exception as e:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": str(e)},
            )


# Global service instance
_streaming_service: Optional[StreamingService] = None


def get_streaming_service() -> StreamingService:
    """Get or create streaming service."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingService()
    return _streaming_service


def set_streaming_agent(agent: Any) -> None:
    """Set agent for streaming service."""
    get_streaming_service().set_agent(agent)


# Create router
router = APIRouter(prefix="/api/v1/stream", tags=["Streaming"])


@router.post("/chat")
async def stream_chat(
    request: StreamingRequest,
    service: StreamingService = Depends(get_streaming_service),
) -> StreamingResponse:
    """Stream a chat response using Server-Sent Events.

    Returns a stream of events containing tokens as they are generated.
    """
    async def event_generator():
        async for event in service.stream_response(
            message=request.message,
            conversation_id=request.conversation_id,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat")
async def stream_chat_get(
    message: str = Query(..., description="User message"),
    conversation_id: Optional[str] = Query(None),
    service: StreamingService = Depends(get_streaming_service),
) -> StreamingResponse:
    """Stream chat via GET request (for EventSource compatibility)."""
    async def event_generator():
        async for event in service.stream_response(
            message=message,
            conversation_id=conversation_id,
        ):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/batch")
async def process_batch(
    request: BatchRequest,
    service: StreamingService = Depends(get_streaming_service),
) -> BatchResponse:
    """Process a batch of requests.

    Returns results for all items in the batch.
    """
    return await service.process_batch(
        items=request.items,
        parallel=request.parallel,
    )


@router.post("/batch/stream")
async def stream_batch(
    request: BatchRequest,
    service: StreamingService = Depends(get_streaming_service),
) -> StreamingResponse:
    """Process batch with streaming progress updates."""
    async def event_generator():
        async for event in service.stream_batch_progress(
            items=request.items,
            parallel=request.parallel,
        ):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "StreamingRequest",
    "BatchItem",
    "BatchRequest",
    "BatchItemResult",
    "BatchResponse",
    "StreamingService",
    "get_streaming_service",
    "set_streaming_agent",
    "router",
]

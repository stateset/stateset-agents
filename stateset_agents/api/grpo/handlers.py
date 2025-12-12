"""
GRPO Request Handlers

Handler classes for training, conversations, and WebSocket connections.
"""

import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .config import get_grpo_config
from .metrics import get_grpo_metrics
from .models import (
    GRPOConversationRequest,
    GRPOConversationResponse,
    GRPOTrainingRequest,
    GRPOTrainingResponse,
)
from .state import ConversationState, TrainingJob, get_state_manager

logger = get_logger(__name__)


class TrainingHandler:
    """
    Handles GRPO training job lifecycle.

    Manages job creation, execution, status tracking, and cleanup.
    """

    def __init__(self, services: Dict[str, Any]):
        """
        Initialize training handler.

        Args:
            services: Dictionary of initialized services.
        """
        self.services = services
        self.state = get_state_manager()
        self.metrics = get_grpo_metrics()
        self.config = get_grpo_config()

    async def start_training(
        self,
        request: GRPOTrainingRequest,
        user_id: str,
        request_id: str,
    ) -> GRPOTrainingResponse:
        """
        Start a new training job.

        Args:
            request: Training request parameters.
            user_id: ID of the requesting user.
            request_id: Request tracking ID.

        Returns:
            Initial training response with job ID.
        """
        job_id = str(uuid.uuid4())

        # Create job in state manager
        job = self.state.create_job(
            job_id=job_id,
            strategy=request.strategy,
            user_id=user_id,
            request_id=request_id,
            config={
                "prompts": request.prompts,
                "num_iterations": request.num_iterations,
                "use_neural_rewards": request.use_neural_rewards,
                "use_ruler_rewards": request.use_ruler_rewards,
                "distributed_config": request.distributed_config,
            },
        )

        self.metrics.record_training_started()

        return GRPOTrainingResponse(
            job_id=job_id,
            status="started",
            iterations_completed=0,
            total_trajectories=0,
            average_reward=0.0,
            computation_used=0.0,
            metrics={"strategy": request.strategy},
            started_at=job.created_at,
            request_id=request_id,
        )

    async def run_computational_training(
        self,
        job_id: str,
        prompts: List[str],
        num_iterations: int,
        use_neural_rewards: bool = True,
        use_ruler_rewards: bool = False,
    ) -> None:
        """
        Execute computational training strategy.

        Args:
            job_id: Job identifier.
            prompts: Training prompts.
            num_iterations: Number of iterations.
            use_neural_rewards: Whether to use neural reward models.
            use_ruler_rewards: Whether to use RULER judges.
        """
        try:
            self.state.update_job(job_id, status="running")

            engine = self.services.get("demo_engine")
            if not engine:
                self.state.update_job(
                    job_id,
                    status="failed",
                    error="No training engine available",
                )
                self.metrics.record_training_failed()
                return

            total_trajectories = 0
            total_reward = 0.0
            total_computation = 0.0

            for i in range(num_iterations):
                result = await engine.train_iteration(prompts)

                total_trajectories += result.get("trajectories_generated", 0)
                total_reward += result.get("average_reward", 0) * len(prompts)
                total_computation += result.get("total_computation_used", 0)

                self.state.update_job(
                    job_id,
                    iterations=i + 1,
                    trajectories=total_trajectories,
                    result=result,
                )

                # Small delay to allow other async operations
                await asyncio.sleep(0.01)

            average_reward = total_reward / total_trajectories if total_trajectories else 0.0

            self.state.update_job(job_id, status="completed")
            self.metrics.record_training_completed(
                trajectories=total_trajectories,
                computation=total_computation,
            )

            logger.info(
                "Training job %s completed: %d iterations, %d trajectories, avg reward %.3f",
                job_id,
                num_iterations,
                total_trajectories,
                average_reward,
            )

        except Exception as e:
            logger.exception("Training job %s failed", job_id)
            self.state.update_job(job_id, status="failed", error=str(e))
            self.metrics.record_training_failed()

    async def run_distributed_training(
        self,
        job_id: str,
        prompts: List[str],
        num_iterations: int,
        distributed_config: Dict[str, Any],
    ) -> None:
        """
        Execute distributed training strategy.

        Args:
            job_id: Job identifier.
            prompts: Training prompts.
            num_iterations: Number of iterations.
            distributed_config: Distributed training configuration.
        """
        try:
            self.state.update_job(job_id, status="running")

            # For now, fall back to computational training
            # Real distributed training would use multiple workers
            await self.run_computational_training(
                job_id=job_id,
                prompts=prompts,
                num_iterations=num_iterations,
                use_neural_rewards=True,
                use_ruler_rewards=False,
            )

        except Exception as e:
            logger.exception("Distributed training job %s failed", job_id)
            self.state.update_job(job_id, status="failed", error=str(e))
            self.metrics.record_training_failed()

    def get_job_status(self, job_id: str) -> Optional[GRPOTrainingResponse]:
        """
        Get status of a training job.

        Args:
            job_id: Job identifier.

        Returns:
            Training response with current status, or None if not found.
        """
        job = self.state.get_job(job_id)
        if not job:
            return None

        # Calculate metrics from results
        if job.results:
            avg_reward = sum(r.get("average_reward", 0) for r in job.results) / len(job.results)
            total_computation = sum(r.get("total_computation_used", 0) for r in job.results)
            latest_metrics = job.results[-1] if job.results else {}
        else:
            avg_reward = 0.0
            total_computation = 0.0
            latest_metrics = {}

        return GRPOTrainingResponse(
            job_id=job_id,
            status=job.status,
            iterations_completed=job.iterations_completed,
            total_trajectories=job.total_trajectories,
            average_reward=avg_reward,
            computation_used=total_computation,
            metrics=latest_metrics,
            error=job.error,
            started_at=job.started_at,
            completed_at=job.completed_at,
            request_id=job.request_id,
        )

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a training job.

        Args:
            job_id: Job identifier.

        Returns:
            True if cancelled, False if not found or already completed.
        """
        job = self.state.get_job(job_id)
        if not job:
            return False

        if job.status in ("completed", "failed", "cancelled"):
            return False

        self.state.update_job(job_id, status="cancelled")
        return True


class ConversationHandler:
    """
    Handles multi-turn conversations.

    Manages conversation lifecycle and message processing.
    """

    def __init__(self, services: Dict[str, Any]):
        """
        Initialize conversation handler.

        Args:
            services: Dictionary of initialized services.
        """
        self.services = services
        self.state = get_state_manager()
        self.metrics = get_grpo_metrics()
        self.config = get_grpo_config()

    async def handle_message(
        self,
        request: GRPOConversationRequest,
        user_id: str,
    ) -> GRPOConversationResponse:
        """
        Handle a conversation message.

        Args:
            request: Conversation request.
            user_id: ID of the user.

        Returns:
            Conversation response.

        Raises:
            ValueError: If agent not initialized or conversation not found.
        """
        start_time = time.monotonic()

        multiturn_agent = self.services.get("multiturn_agent")
        if not multiturn_agent:
            raise ValueError("Multi-turn agent not initialized")

        conversation_id = request.conversation_id

        if conversation_id:
            # Continue existing conversation
            conv = self.state.get_conversation(conversation_id)
            if not conv:
                raise ValueError(f"Conversation {conversation_id} not found")

            turns = await multiturn_agent.continue_conversation(
                conversation_id,
                request.message,
                strategy=request.strategy,
            )
            response_text = turns[-1]["content"] if turns else "No response generated"
            context = multiturn_agent.get_conversation_summary(conversation_id)

        else:
            # Start new conversation
            conversation_context = await multiturn_agent.start_conversation(
                user_id=request.user_id or user_id,
                initial_context=request.context,
            )

            response_text = await multiturn_agent.generate_multiturn_response(
                conversation_context.conversation_id,
                request.message,
                strategy=request.strategy,
            )

            conversation_context.add_turn({"role": "assistant", "content": response_text})
            context = conversation_context.get_context_summary()
            conversation_id = conversation_context.conversation_id

            # Track in state manager
            self.state.create_conversation(
                conversation_id=conversation_id,
                user_id=request.user_id or user_id,
                strategy=request.strategy,
            )
            self.metrics.record_conversation_started()

        # Update conversation state
        tokens_used = len(response_text.split())  # Approximate token count
        self.state.update_conversation(conversation_id, tokens=tokens_used)
        self.metrics.record_message()

        processing_time = (time.monotonic() - start_time) * 1000

        return GRPOConversationResponse(
            conversation_id=conversation_id,
            response=response_text,
            context=context,
            metadata={
                "strategy": request.strategy,
                "timestamp": datetime.utcnow().isoformat(),
            },
            tokens_used=tokens_used,
            processing_time_ms=processing_time,
        )

    def end_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        End a conversation.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Final conversation summary, or None if not found.
        """
        multiturn_agent = self.services.get("multiturn_agent")
        if not multiturn_agent:
            return None

        context = multiturn_agent.end_conversation(conversation_id)
        if not context:
            return None

        self.state.end_conversation(conversation_id)
        self.metrics.record_conversation_ended()

        return {
            "conversation_id": conversation_id,
            "final_summary": context.get_context_summary(),
        }

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get conversation state.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Conversation state, or None if not found.
        """
        return self.state.get_conversation(conversation_id)


class WebSocketHandler:
    """
    Handles WebSocket connections and messages.

    Provides real-time interaction capabilities.
    """

    # Security constants
    MAX_MESSAGE_SIZE = 65536  # 64KB
    MAX_MESSAGES_PER_SECOND = 10

    def __init__(
        self,
        services: Dict[str, Any],
        training_handler: TrainingHandler,
        conversation_handler: ConversationHandler,
    ):
        """
        Initialize WebSocket handler.

        Args:
            services: Dictionary of initialized services.
            training_handler: Handler for training requests.
            conversation_handler: Handler for conversation requests.
        """
        self.services = services
        self.training_handler = training_handler
        self.conversation_handler = conversation_handler
        self.metrics = get_grpo_metrics()
        self.config = get_grpo_config()

    async def handle_connection(self, websocket: Any) -> None:
        """
        Handle a WebSocket connection.

        Args:
            websocket: WebSocket connection object.
        """
        # Track message rate
        message_timestamps: deque = deque(maxlen=self.MAX_MESSAGES_PER_SECOND)

        self.metrics.record_websocket_connect()

        try:
            while True:
                data = await websocket.receive_text()

                # Validate message size
                if len(data) > self.MAX_MESSAGE_SIZE:
                    await self._send_error(
                        websocket,
                        f"Message exceeds maximum size of {self.MAX_MESSAGE_SIZE} bytes",
                    )
                    continue

                # Rate limit messages
                now = time.monotonic()
                message_timestamps.append(now)
                if len(message_timestamps) >= self.MAX_MESSAGES_PER_SECOND:
                    oldest = message_timestamps[0]
                    if now - oldest < 1.0:
                        await self._send_error(websocket, "Message rate limit exceeded")
                        continue

                # Parse message
                try:
                    message_data = json.loads(data)
                except json.JSONDecodeError as e:
                    await self._send_error(websocket, f"Invalid JSON: {str(e)[:100]}")
                    continue

                if not isinstance(message_data, dict):
                    await self._send_error(websocket, "Message must be a JSON object")
                    continue

                self.metrics.record_websocket_message()

                # Route message
                message_type = message_data.get("type")
                await self._route_message(websocket, message_type, message_data)

        except Exception as e:
            if "disconnect" not in str(e).lower():
                logger.error("WebSocket error: %s", e)

    async def _route_message(
        self,
        websocket: Any,
        message_type: str,
        message_data: Dict[str, Any],
    ) -> None:
        """Route message to appropriate handler."""
        if message_type == "chat":
            await self._handle_chat(websocket, message_data)
        elif message_type == "metrics":
            await self._handle_metrics(websocket)
        elif message_type == "ping":
            await self._handle_ping(websocket)
        else:
            await self._send_error(
                websocket,
                f"Unknown message type: {message_type}",
            )

    async def _handle_chat(
        self,
        websocket: Any,
        message_data: Dict[str, Any],
    ) -> None:
        """Handle chat message."""
        try:
            chat_data = message_data.get("data", {})
            if not isinstance(chat_data, dict):
                await self._send_error(websocket, "chat data must be an object")
                return

            request = GRPOConversationRequest(**chat_data)
            response = await self.conversation_handler.handle_message(
                request,
                user_id="ws_user",
            )

            await websocket.send_json({
                "type": "chat_response",
                "data": {
                    "conversation_id": response.conversation_id,
                    "response": response.response,
                    "context": response.context,
                    "metadata": response.metadata,
                    "tokens_used": response.tokens_used,
                    "processing_time_ms": response.processing_time_ms,
                },
            })

        except ValueError as e:
            await self._send_error(websocket, f"Invalid chat request: {str(e)[:200]}")
        except Exception as e:
            logger.error("WebSocket chat error: %s", e)
            await self._send_error(websocket, "Failed to process chat request")

    async def _handle_metrics(self, websocket: Any) -> None:
        """Handle metrics request."""
        try:
            state = get_state_manager()
            metrics = {
                "system": state.stats(),
                "api": self.metrics.get_summary(),
            }

            if "demo_engine" in self.services:
                metrics["demo_engine"] = self.services["demo_engine"].get_metrics()

            await websocket.send_json({
                "type": "metrics_response",
                "data": metrics,
            })

        except Exception as e:
            logger.error("WebSocket metrics error: %s", e)
            await self._send_error(websocket, "Failed to retrieve metrics")

    async def _handle_ping(self, websocket: Any) -> None:
        """Handle ping message."""
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def _send_error(self, websocket: Any, message: str) -> None:
        """Send error message."""
        await websocket.send_json({
            "type": "error",
            "message": message,
        })

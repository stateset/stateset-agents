"""
Production-Ready Customer Service Agent Example

This example demonstrates a complete, production-ready implementation
of a customer service agent using StateSet Agents with:
- Error handling and retry logic
- Monitoring and logging
- Graceful degradation
- Health checks
- Performance optimization
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("Prometheus client not available. Metrics disabled.")

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import (
    CompositeReward,
    HelpfulnessReward,
    SafetyReward,
    create_customer_service_reward,
)
from stateset_agents.core.error_handling import RetryWithBackoff, CircuitBreaker


# Metrics
if HAS_PROMETHEUS:
    REQUESTS_TOTAL = Counter(
        "customer_service_requests_total",
        "Total customer service requests",
        ["status"]
    )
    RESPONSE_TIME = Histogram(
        "customer_service_response_seconds",
        "Response generation time",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
    )
    CONVERSATIONS_ACTIVE = Counter(
        "customer_service_conversations_active",
        "Active conversations"
    )


class ProductionCustomerServiceAgent:
    """Production-ready customer service agent with full observability"""

    def __init__(
        self,
        model_name: str = "gpt2",
        enable_metrics: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.enable_metrics = enable_metrics
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.agent: Optional[MultiTurnAgent] = None
        self.is_healthy = False
        self.shutdown_requested = False

        # Error handling
        self.retry_handler = RetryWithBackoff(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception
        )

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True

    async def initialize(self) -> bool:
        """Initialize the agent with error handling"""
        logger.info("Initializing production customer service agent...")

        try:
            # Create agent configuration
            config = AgentConfig(
                model_name=self.model_name,
                system_prompt="""You are a professional customer service representative for TechCorp.

                Guidelines:
                - Be friendly, empathetic, and professional
                - Ask clarifying questions when needed
                - Provide clear, actionable solutions
                - Escalate complex issues when appropriate
                - Always prioritize customer satisfaction

                Remember: Every interaction is an opportunity to build trust.""",
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                use_chat_template=True
            )

            # Initialize agent
            self.agent = MultiTurnAgent(config, memory_window=10)
            await self.agent.initialize()

            # Load checkpoint if available
            if self.checkpoint_dir.exists():
                latest_checkpoint = self._get_latest_checkpoint()
                if latest_checkpoint:
                    logger.info(f"Loading checkpoint: {latest_checkpoint}")
                    await self._load_checkpoint(latest_checkpoint)

            self.is_healthy = True
            logger.info("Agent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            self.is_healthy = False
            return False

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    async def _load_checkpoint(self, checkpoint_path: Path):
        """Load a checkpoint"""
        # Implement checkpoint loading logic
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    async def handle_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle a customer service conversation with full error handling"""

        if not self.is_healthy or not self.agent:
            logger.error("Agent not healthy. Cannot handle conversation.")
            return {
                "success": False,
                "error": "Service temporarily unavailable",
                "response": "I apologize, but our system is currently experiencing issues. Please try again in a few moments."
            }

        start_time = datetime.now()
        conversation_context = context or {}
        conversation_context.update({
            "conversation_id": conversation_id,
            "timestamp": start_time.isoformat()
        })

        logger.info(f"Handling conversation {conversation_id}")

        try:
            # Use circuit breaker to prevent cascading failures
            @self.circuit_breaker
            async def generate_response_with_retry():
                # Use retry logic for transient failures
                return await self.retry_handler.execute(
                    self.agent.generate_response,
                    messages,
                    conversation_context
                )

            # Generate response
            response = await generate_response_with_retry()

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Update metrics
            if HAS_PROMETHEUS and self.enable_metrics:
                REQUESTS_TOTAL.labels(status="success").inc()
                RESPONSE_TIME.observe(response_time)

            logger.info(
                f"Conversation {conversation_id} completed successfully "
                f"in {response_time:.2f}s"
            )

            return {
                "success": True,
                "response": response,
                "conversation_id": conversation_id,
                "response_time_seconds": response_time,
                "metadata": {
                    "turn_count": self.agent.turn_count,
                    "model": self.model_name
                }
            }

        except Exception as e:
            logger.error(
                f"Error handling conversation {conversation_id}: {e}",
                exc_info=True
            )

            # Update metrics
            if HAS_PROMETHEUS and self.enable_metrics:
                REQUESTS_TOTAL.labels(status="error").inc()

            # Provide graceful fallback response
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your request right now. Could you please rephrase or try again?",
                "conversation_id": conversation_id
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "healthy": self.is_healthy,
            "agent_initialized": self.agent is not None,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat()
        }

    async def shutdown(self):
        """Gracefully shutdown the agent"""
        logger.info("Shutting down agent...")
        self.is_healthy = False

        # Save checkpoint before shutdown
        if self.agent:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            # Implement checkpoint saving logic here

        logger.info("Shutdown complete")

    async def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run as a production server"""
        logger.info(f"Starting production customer service agent server on {host}:{port}")

        # Start Prometheus metrics server if available
        if HAS_PROMETHEUS and self.enable_metrics:
            start_http_server(9090)
            logger.info("Prometheus metrics available on port 9090")

        # Initialize agent
        if not await self.initialize():
            logger.error("Failed to initialize agent. Exiting.")
            sys.exit(1)

        logger.info("Server ready to accept requests")

        # Example conversation loop (in production, use FastAPI or similar)
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")

        finally:
            await self.shutdown()


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Production Customer Service Agent")
    logger.info("=" * 60)

    # Create agent
    agent = ProductionCustomerServiceAgent(
        model_name="gpt2",  # Replace with your production model
        enable_metrics=True,
        checkpoint_dir="./checkpoints"
    )

    # Initialize
    if not await agent.initialize():
        logger.error("Failed to initialize agent")
        return

    # Example conversation
    messages = [
        {
            "role": "user",
            "content": "Hi, I'm having trouble with my order. It was supposed to arrive yesterday but hasn't shown up yet."
        }
    ]

    result = await agent.handle_conversation(
        conversation_id="example_001",
        messages=messages,
        context={"customer_tier": "premium", "order_id": "ORD-12345"}
    )

    logger.info("Conversation Result:")
    logger.info(f"Success: {result['success']}")
    logger.info(f"Response: {result['response']}")
    if 'response_time_seconds' in result:
        logger.info(f"Response Time: {result['response_time_seconds']:.3f}s")

    # Health check
    health = await agent.health_check()
    logger.info(f"\nHealth Check: {health}")

    # Cleanup
    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

"""
Ultimate Customer Service Demo - GRPO Agent Framework

This demo showcases the complete power of the GRPO Agent Framework
with all innovations integrated for a world-class customer service experience.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.computational_engine import create_computational_engine

# Core framework imports
from ..core.multiturn_agent import DialogueDatabase, MultiTurnAgent
from ..rewards.multi_objective_reward import create_customer_service_reward
from ..rewards.ruler_reward import create_customer_service_ruler
from ..training.neural_reward_trainer import create_neural_reward_function
from ..utils.cache import CacheService
from ..utils.monitoring import MonitoringService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample customer service data
SAMPLE_DIALOGUES = [
    {
        "id": "cs_001",
        "content": "Hello, I'm having trouble with my order. It hasn't arrived yet and I ordered it 5 days ago.",
        "category": "order_tracking",
        "expected_response": "I understand your concern about your delayed order. Let me check the tracking information for you right away. Could you please provide your order number?",
        "priority": "high",
    },
    {
        "id": "cs_002",
        "content": "I need to return this item. It doesn't fit properly and I'm not satisfied with the quality.",
        "category": "returns",
        "expected_response": "I'm sorry to hear that the item didn't meet your expectations. I'll be happy to help you with the return process. Our return policy allows returns within 30 days of purchase.",
        "priority": "medium",
    },
    {
        "id": "cs_003",
        "content": "Hi, I want to know about the warranty on my recent purchase. What's covered?",
        "category": "warranty",
        "expected_response": "Great question! I'd be happy to explain our warranty coverage. The warranty varies by product type. Could you tell me what item you purchased so I can provide specific warranty details?",
        "priority": "low",
    },
    {
        "id": "cs_004",
        "content": "I'm very frustrated! My payment went through but I never received a confirmation email.",
        "category": "payment_issues",
        "expected_response": "I completely understand your frustration, and I apologize for the inconvenience. Let me look into this payment issue immediately. Can you please provide the email address used for the purchase?",
        "priority": "high",
    },
    {
        "id": "cs_005",
        "content": "Can you help me change my shipping address? I moved recently and forgot to update it.",
        "category": "shipping",
        "expected_response": "Of course! I'll help you update your shipping address. If your order hasn't shipped yet, we can modify it. If it has already shipped, we'll explore other options. Let me check the status of your order first.",
        "priority": "medium",
    },
]

SAMPLE_PROMPTS = [
    "Where is my order? I ordered it last week.",
    "I want to return this product. It's defective.",
    "Can you help me with my billing question?",
    "I need to cancel my subscription.",
    "The item I received is damaged. What can you do?",
    "I forgot my password. Can you help me reset it?",
    "When will my backordered item ship?",
    "I was charged twice for the same order.",
    "Can I exchange this item for a different size?",
    "I need a receipt for my purchase from last month.",
]


class AdvancedCustomerServiceAgent:
    """
    Advanced Customer Service Agent with all GRPO innovations
    """

    def __init__(self):
        self.monitoring = MonitoringService()
        self.cache = CacheService()

        # Initialize dialogue database
        self.dialogue_db = DialogueDatabase(SAMPLE_DIALOGUES)

        # Create multi-turn agent
        self.agent = MultiTurnAgent(
            model_config={
                "model_type": "advanced_customer_service",
                "temperature": 0.7,
                "max_tokens": 200,
            },
            max_context_length=2048,
            max_conversation_turns=15,
            dialogue_database=self.dialogue_db,
            cache_service=self.cache,
        )

        # Register customer service tools
        self._register_tools()

        # Create sophisticated reward system
        self._setup_reward_system()

        # Performance tracking
        self.conversation_metrics = {
            "total_conversations": 0,
            "resolved_conversations": 0,
            "average_satisfaction": 0.0,
            "response_times": [],
        }

    def _register_tools(self):
        """Register customer service tools"""

        async def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for relevant information"""
            # Simulate knowledge base search
            relevant_dialogues = self.dialogue_db.search(query, top_k=3)

            if relevant_dialogues:
                return (
                    f"Found {len(relevant_dialogues)} relevant articles: "
                    + ", ".join(
                        [d.get("category", "general") for d in relevant_dialogues]
                    )
                )
            else:
                return "No specific articles found, but I'll do my best to help you."

        async def check_order_status(order_number: str) -> str:
            """Check order status (simulated)"""
            # Simulate order lookup
            statuses = ["Processing", "Shipped", "Delivered", "Delayed"]
            import random

            status = random.choice(statuses)

            return (
                f"Order {order_number} status: {status}. "
                + f"Estimated delivery: {datetime.now().strftime('%Y-%m-%d')}"
            )

        async def create_return_label(order_number: str) -> str:
            """Create return label (simulated)"""
            return (
                f"Return label created for order {order_number}. "
                + "You'll receive an email with the prepaid return label within 5 minutes."
            )

        async def escalate_to_human(reason: str) -> str:
            """Escalate to human agent"""
            return (
                f"I'm escalating your case to a human specialist for: {reason}. "
                + "You'll be connected within 2-3 minutes. Your case number is CS-"
                + str(uuid.uuid4())[:8].upper()
            )

        # Register tools
        self.agent.register_tool("search_knowledge_base", search_knowledge_base)
        self.agent.register_tool("check_order_status", check_order_status)
        self.agent.register_tool("create_return_label", create_return_label)
        self.agent.register_tool("escalate_to_human", escalate_to_human)

    def _setup_reward_system(self):
        """Setup sophisticated reward system"""

        # Multi-objective reward with expected responses
        expected_responses = [d["expected_response"] for d in SAMPLE_DIALOGUES]

        self.multi_objective_reward = create_customer_service_reward(
            expected_responses=expected_responses, weight=0.4
        )

        # Neural reward function (learns from interactions)
        self.neural_reward = create_neural_reward_function(
            embedding_dim=128, hidden_dim=256, weight=0.3, update_frequency=50
        )

        # RULER LLM judge (if API key available)
        try:
            self.ruler_reward = create_customer_service_ruler(
                model="openai/gpt-4", weight=0.3, fallback_enabled=True
            )
        except Exception as e:
            logger.warning(f"RULER reward not available: {e}")
            self.ruler_reward = None

        # Combine all reward functions
        self.reward_functions = [self.multi_objective_reward, self.neural_reward]

        if self.ruler_reward:
            self.reward_functions.append(self.ruler_reward)

    async def handle_customer_interaction(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        customer_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle a complete customer interaction"""

        start_time = datetime.now()

        try:
            # Start new conversation or continue existing
            if conversation_id:
                # Continue existing conversation
                turns = await self.agent.continue_conversation(
                    conversation_id, message, strategy="customer_service"
                )
                response = turns[-1]["content"]
                context = self.agent.get_conversation_summary(conversation_id)
            else:
                # Start new conversation
                conversation_context = await self.agent.start_conversation(
                    user_id=customer_context.get("user_id")
                    if customer_context
                    else None,
                    initial_context={
                        "topic": "customer_service",
                        "priority": customer_context.get("priority", "medium")
                        if customer_context
                        else "medium",
                        **(customer_context or {}),
                    },
                )

                response = await self.agent.generate_multiturn_response(
                    conversation_context.conversation_id,
                    message,
                    strategy="customer_service",
                    use_tools=True,
                )

                # Update conversation
                conversation_context.add_turn(
                    {"role": "assistant", "content": response}
                )

                context = conversation_context.get_context_summary()
                conversation_id = conversation_context.conversation_id

            # Compute response quality
            turns = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]

            reward_results = []
            for reward_func in self.reward_functions:
                try:
                    result = await reward_func.compute_reward(turns, context)
                    reward_results.append(result)
                except Exception as e:
                    logger.error(f"Reward computation failed: {e}")

            # Calculate metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self.conversation_metrics["response_times"].append(response_time)

            # Provide feedback to neural reward model
            if reward_results:
                average_reward = sum(r.score for r in reward_results) / len(
                    reward_results
                )
                self.neural_reward.add_trajectory_feedback(
                    turns, average_reward, context
                )

            # Update metrics
            self.conversation_metrics["total_conversations"] += 1

            # Log interaction
            await self.monitoring.log_event(
                "customer_interaction",
                {
                    "conversation_id": conversation_id,
                    "response_time": response_time,
                    "reward_scores": [r.score for r in reward_results],
                    "message_length": len(message),
                    "response_length": len(response),
                },
            )

            return {
                "conversation_id": conversation_id,
                "response": response,
                "context": context,
                "metrics": {
                    "response_time_seconds": response_time,
                    "quality_scores": [r.score for r in reward_results],
                    "average_quality": sum(r.score for r in reward_results)
                    / len(reward_results)
                    if reward_results
                    else 0.0,
                },
                "suggested_actions": self._get_suggested_actions(
                    message, response, context
                ),
            }

        except Exception as e:
            logger.error(f"Customer interaction failed: {e}")
            return {
                "conversation_id": conversation_id,
                "response": "I apologize, but I'm experiencing technical difficulties. Let me connect you with a human agent right away.",
                "context": {},
                "metrics": {
                    "response_time_seconds": (
                        datetime.now() - start_time
                    ).total_seconds(),
                    "quality_scores": [0.0],
                    "average_quality": 0.0,
                },
                "error": str(e),
            }

    def _get_suggested_actions(
        self, message: str, response: str, context: Dict[str, Any]
    ) -> List[str]:
        """Get suggested follow-up actions"""
        suggestions = []

        # Analyze message for common scenarios
        message_lower = message.lower()

        if "order" in message_lower and "track" in message_lower:
            suggestions.append("Provide order tracking link")
            suggestions.append("Send proactive shipping updates")

        if "return" in message_lower or "refund" in message_lower:
            suggestions.append("Generate return label")
            suggestions.append("Process refund if applicable")

        if "frustrated" in message_lower or "angry" in message_lower:
            suggestions.append("Escalate to supervisor")
            suggestions.append("Offer compensation")

        if "cancel" in message_lower:
            suggestions.append("Confirm cancellation")
            suggestions.append("Offer alternatives")

        # Check conversation length
        if context.get("turn_count", 0) > 10:
            suggestions.append("Consider escalation to human agent")

        return suggestions

    async def run_training_simulation(self, num_iterations: int = 10):
        """Run training simulation with computational engine"""
        logger.info(f"🚀 Starting training simulation with {num_iterations} iterations")

        # Create computational engine
        from ..core.agent import Agent
        from ..core.environment import Environment
        from ..core.reward import RewardFunction

        class CustomerServiceAgent(Agent):
            def __init__(self, multiturn_agent):
                super().__init__({"model_type": "customer_service"})
                self.multiturn_agent = multiturn_agent

            async def generate_response(self, prompt: str) -> str:
                context = await self.multiturn_agent.start_conversation()
                response = await self.multiturn_agent.generate_multiturn_response(
                    context.conversation_id, prompt, strategy="customer_service"
                )
                return response

        class CustomerServiceEnvironment(Environment):
            def __init__(self, dialogues):
                super().__init__()
                self.dialogues = dialogues
                self.current_dialogue = 0

            async def reset(self) -> Dict[str, Any]:
                self.current_dialogue = 0
                return {"dialogue_index": 0}

            async def step(self, action: str) -> Dict[str, Any]:
                self.current_dialogue += 1
                done = self.current_dialogue >= len(self.dialogues)

                return {
                    "state": {"dialogue_index": self.current_dialogue},
                    "reward": 0.5,  # Placeholder
                    "done": done,
                }

            async def get_reward(self, trajectory) -> float:
                # Simulate environment reward
                import random

                return random.uniform(0.4, 0.9)

        # Create computational engine
        cs_agent = CustomerServiceAgent(self.agent)
        cs_environment = CustomerServiceEnvironment(SAMPLE_DIALOGUES)
        engine = create_computational_engine(
            cs_agent, cs_environment, self.multi_objective_reward, num_workers=4
        )

        # Run training iterations
        results = []
        for i in range(num_iterations):
            logger.info(f"Training iteration {i+1}/{num_iterations}")

            # Use sample prompts for training
            iteration_results = await engine.train_iteration(SAMPLE_PROMPTS[:5])
            results.append(iteration_results)

            logger.info(
                f"Iteration {i+1} completed: "
                f"Avg reward: {iteration_results['average_reward']:.3f}, "
                f"Trajectories: {iteration_results['trajectories_generated']}"
            )

        # Get final metrics
        final_metrics = engine.get_metrics()

        logger.info("🎯 Training simulation completed!")
        logger.info(
            f"Total computation used: {final_metrics['computation_used']:.2f} seconds"
        )
        logger.info(f"Total trajectories: {final_metrics['total_trajectories']}")
        logger.info(
            f"Average trajectories per second: {final_metrics['engine_metrics']['trajectories_per_second']:.2f}"
        )

        return {
            "iterations": results,
            "final_metrics": final_metrics,
            "performance_summary": {
                "avg_reward": sum(r["average_reward"] for r in results) / len(results),
                "total_trajectories": sum(r["trajectories_generated"] for r in results),
                "computation_efficiency": final_metrics["engine_metrics"][
                    "trajectories_per_second"
                ],
            },
        }

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "conversation_metrics": self.conversation_metrics,
            "active_conversations": len(self.agent.get_active_conversations()),
            "reward_model_info": {
                "neural_reward": self.neural_reward.get_model_info(),
                "multi_objective_stats": self.multi_objective_reward.get_component_statistics(),
            },
            "dialogue_database_stats": {
                "total_dialogues": len(self.dialogue_db.dialogues),
                "categories": list(
                    set(
                        d.get("category", "general") for d in self.dialogue_db.dialogues
                    )
                ),
            },
        }


async def run_interactive_demo():
    """Run interactive customer service demo"""
    print("\n" + "=" * 80)
    print("🎯 ULTIMATE CUSTOMER SERVICE DEMO")
    print("Powered by GRPO Agent Framework with All Innovations")
    print("=" * 80)

    # Initialize agent
    print("\n🚀 Initializing advanced customer service agent...")
    agent = AdvancedCustomerServiceAgent()

    print("✅ Agent initialized with:")
    print("   • Multi-turn conversation management")
    print("   • Neural reward learning")
    print("   • Multi-objective reward optimization")
    print("   • LLM judge evaluation (if API available)")
    print("   • Tool integration (order tracking, returns, etc.)")
    print("   • Real-time performance monitoring")

    # Interactive session
    print("\n💬 Starting interactive session...")
    print(
        "Type 'quit' to exit, 'metrics' to see performance, 'train' to run training simulation"
    )
    print("-" * 40)

    current_conversation = None

    while True:
        try:
            user_input = input("\nCustomer: ").strip()

            if user_input.lower() == "quit":
                print("\nThank you for using the GRPO Customer Service Demo!")
                break

            elif user_input.lower() == "metrics":
                print("\n📊 Performance Metrics:")
                report = await agent.get_performance_report()
                print(json.dumps(report, indent=2, default=str))
                continue

            elif user_input.lower() == "train":
                print("\n🎓 Running training simulation...")
                training_results = await agent.run_training_simulation(5)
                print(
                    f"Training completed! Average reward: {training_results['performance_summary']['avg_reward']:.3f}"
                )
                continue

            elif user_input.lower() == "new":
                current_conversation = None
                print("Starting new conversation...")
                continue

            if not user_input:
                continue

            # Handle customer interaction
            result = await agent.handle_customer_interaction(
                user_input,
                conversation_id=current_conversation,
                customer_context={"priority": "high", "user_id": "demo_user"},
            )

            # Update conversation ID
            current_conversation = result["conversation_id"]

            # Display response
            print(f"\nAgent: {result['response']}")

            # Display metrics
            metrics = result["metrics"]
            print(f"\n📈 Interaction Metrics:")
            print(f"   Response time: {metrics['response_time_seconds']:.2f}s")
            print(f"   Quality score: {metrics['average_quality']:.3f}")

            # Display suggested actions
            if result.get("suggested_actions"):
                print(f"   Suggested actions: {', '.join(result['suggested_actions'])}")

        except KeyboardInterrupt:
            print("\nDemo interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Demo error: {e}")


async def run_benchmark_demo():
    """Run benchmark demo showing performance"""
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE BENCHMARK DEMO")
    print("=" * 80)

    agent = AdvancedCustomerServiceAgent()

    print("\n🏃 Running performance benchmark...")

    # Benchmark scenarios
    test_scenarios = [
        ("Order tracking", "Where is my order? I ordered it 5 days ago."),
        ("Return request", "I need to return this item. It's defective."),
        ("Billing inquiry", "I was charged twice for the same order."),
        ("Product question", "Can you tell me about the warranty on this product?"),
        ("Complaint", "I'm very frustrated with the service I received."),
        ("Cancellation", "I need to cancel my subscription immediately."),
        ("Exchange", "Can I exchange this item for a different size?"),
        ("Password reset", "I forgot my password and can't log in."),
        ("Shipping address", "I need to change my shipping address."),
        ("Account issue", "My account has been locked and I can't access it."),
    ]

    results = []
    total_start_time = datetime.now()

    for i, (scenario, message) in enumerate(test_scenarios):
        print(f"\n📝 Scenario {i+1}/10: {scenario}")
        print(f"Message: {message}")

        start_time = datetime.now()
        result = await agent.handle_customer_interaction(
            message, customer_context={"priority": "medium", "scenario": scenario}
        )

        response_time = (datetime.now() - start_time).total_seconds()
        results.append(
            {
                "scenario": scenario,
                "response_time": response_time,
                "quality_score": result["metrics"]["average_quality"],
                "response_length": len(result["response"]),
            }
        )

        print(f"✅ Response time: {response_time:.2f}s")
        print(f"   Quality score: {result['metrics']['average_quality']:.3f}")
        print(f"   Response: {result['response'][:100]}...")

    total_time = (datetime.now() - total_start_time).total_seconds()

    # Display benchmark results
    print(f"\n🏆 BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Total scenarios: {len(test_scenarios)}")
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Average response time: {sum(r['response_time'] for r in results) / len(results):.2f}s"
    )
    print(
        f"Average quality score: {sum(r['quality_score'] for r in results) / len(results):.3f}"
    )
    print(f"Throughput: {len(results) / total_time:.2f} interactions/second")

    # Show best and worst performers
    best_quality = max(results, key=lambda x: x["quality_score"])
    fastest_response = min(results, key=lambda x: x["response_time"])

    print(
        f"\n🥇 Best quality: {best_quality['scenario']} (score: {best_quality['quality_score']:.3f})"
    )
    print(
        f"⚡ Fastest response: {fastest_response['scenario']} ({fastest_response['response_time']:.2f}s)"
    )


def main():
    """Main demo function"""
    print("Choose demo mode:")
    print("1. Interactive customer service demo")
    print("2. Performance benchmark demo")
    print("3. Both demos")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        asyncio.run(run_interactive_demo())
    elif choice == "2":
        asyncio.run(run_benchmark_demo())
    elif choice == "3":
        asyncio.run(run_benchmark_demo())
        asyncio.run(run_interactive_demo())
    else:
        print("Invalid choice. Running interactive demo...")
        asyncio.run(run_interactive_demo())


if __name__ == "__main__":
    main()

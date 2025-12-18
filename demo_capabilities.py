"""
Stateset Agents - RL Capabilities Demo

This example demonstrates the core capabilities of the Stateset Agents framework:
1.  **Multi-turn Agent Configuration**: Setting up a conversational agent.
2.  **Group Relative Policy Optimization (GRPO)**: Using the flagship RL algorithm.
3.  **Rust Acceleration**: Implicitly leveraging the optimized `stateset-rl-core` for advantage calculation.
4.  **Custom Reward Functions**: Defining a specific objective (conciseness and politeness).

The goal is to train a small model (GPT-2) to be:
-   **Polite**: Always start with "I appreciate" or "Thank you".
-   **Concise**: Keep responses under 20 words.
"""

import asyncio
import logging
import sys
import re
from typing import List, Dict, Any, Union

# 1. Imports from the framework
from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.config import TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("demo_capabilities")

# -----------------------------------------------------------------------------
# 2. Custom Reward Function
# -----------------------------------------------------------------------------
class PolitenessReward:
    """
    A custom reward function that encourages politeness and conciseness.
    This demonstrates the flexibility of defining domain-specific objectives.
    """
    async def compute_reward(self, turns: List[Any], scenario: Dict[str, Any]) -> Any:
        """
        Compute reward for a full trajectory.
        
        Args:
            turns: List of conversation turns (user/assistant messages).
            scenario: The context in which the conversation happened.
            
        Returns:
            A score (float) or object with .score attribute.
        """
        # We only care about the assistant's last response
        last_turn = turns[-1]
        if hasattr(last_turn, "content"):
            content = last_turn.content
        else:
            content = last_turn.get("content", "")
            
        score = 0.0
        breakdown = {}

        # Objective 1: Politeness (Keyword matching)
        if re.search(r"(thank you|appreciate|please)", content, re.IGNORECASE):
            score += 1.0
            breakdown["politeness"] = 1.0
        else:
            breakdown["politeness"] = 0.0

        # Objective 2: Conciseness (Length penalty)
        word_count = len(content.split())
        if word_count < 20:
            score += 0.5
            breakdown["conciseness"] = 0.5
        elif word_count > 50:
            score -= 0.5
            breakdown["conciseness"] = -0.5
        else:
            breakdown["conciseness"] = 0.0
            
        # Return result in the expected format
        return {"score": score, "breakdown": breakdown}


# -----------------------------------------------------------------------------
# 3. Main Execution Block
# -----------------------------------------------------------------------------
async def main():
    logger.info("🚀 Starting Stateset Agents Capabilities Demo")

    # A. Configuration
    # We use a small model for the demo to ensure it runs quickly on any hardware.
    # In production, you would use "meta-llama/Llama-2-7b-chat-hf" etc.
    model_name = "Qwen/Qwen1.5-0.5B-Chat" # Using a small Qwen model for demo purposes
    
    logger.info(f"1. Configuring Agent with model: {model_name}")
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt="You are a helpful and polite assistant.",
        max_new_tokens=32,  # Short generation for speed
        temperature=0.8,
        attn_implementation="eager", # Explicitly use eager attention for compatibility
    )
    
    agent = MultiTurnAgent(agent_config)
    # Note: In a real run, we'd await agent.initialize(), but the trainer handles this too.

    # B. Environment Setup
    # Define a set of training scenarios.
    logger.info("2. Setting up Training Environment")
    scenarios = [
        {
            "id": "complaint_1",
            "context": "User is complaining about a late delivery.",
            "user_responses": ["Where is my package?", "It's been a week!"]
        },
        {
            "id": "inquiry_1",
            "context": "User asking about product features.",
            "user_responses": ["Does this support 4K?", "Is it waterproof?"]
        },
        {
            "id": "greeting_1",
            "context": "User saying hello.",
            "user_responses": ["Hi there!", "Good morning."]
        }
    ]
    
    # We create a simple environment that cycles through these scenarios.
    # The 'ConversationEnvironment' handles the simulation loop.
    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=3,  # Short conversations
        persona="You are a customer service bot."
    )

    # C. Trainer Configuration (GRPO)
    # This is where the magic happens. We configure the GRPO algorithm.
    logger.info("3. Configuring GRPO Trainer")
    training_config = TrainingConfig(
        run_name="demo_politeness_agent",
        output_dir="./outputs/demo_agent",
        
        # Training Parameters
        num_episodes=5,  # Very short run for demo purposes
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        
        # GRPO Specifics
        num_generations=4,  # Sample K=4 trajectories per input
        beta=0.01,          # KL penalty to keep it close to reference model
        
        # Hardware
        use_cpu=True,       # Force CPU for broad compatibility in this demo
        report_to="none",   # Disable W&B for this local demo
    )

    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=PolitenessReward(),
        config=training_config
    )

    # D. Training Loop
    logger.info("4. Starting Training Loop...")
    logger.info("   (This uses stateset-rl-core Rust backend for advantage calculation)")
    
    try:
        await trainer.initialize()

        # FIX: GPT-2 doesn't have a default chat template, so we define a simple one for the demo.
        if hasattr(agent, "tokenizer") and agent.tokenizer.chat_template is None:
             agent.tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"

        trained_agent = await trainer.train()
        logger.info("✅ Training Complete!")
        
        # E. Inference / Verification
        logger.info("5. Testing Trained Agent")
        
        test_history = [{"role": "user", "content": "Where is my order?"}]
        response = await trained_agent.generate_response(test_history)
        
        logger.info(f"\nUser: Where is my order?")
        logger.info(f"Agent: {response}")
        
    except Exception as e:
        logger.info(f"An error occurred during the demo: {e}")
        # In case of CUDA/PyTorch errors on environments without GPU, we catch gracefully
        logger.info("Note: This demo requires PyTorch and Transformers installed.")

if __name__ == "__main__":
    asyncio.run(main())

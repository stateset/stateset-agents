"""
Quick Start Guide for GRPO Agent Framework

This example shows the simplest way to get started with training
a multi-turn conversational agent.
"""

import asyncio
import logging

from grpo_agent_framework import (
    MultiTurnAgent, ConversationEnvironment, 
    HelpfulnessReward, SafetyReward,
    train
)
from grpo_agent_framework.core.agent import AgentConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_example():
    """Most basic example - train a helpful assistant"""
    
    logger.info("GRPO Agent Framework - Quick Start Example")
    logger.info("=" * 50)
    
    # Step 1: Create an agent
    logger.info("Step 1: Creating agent...")
    
    config = AgentConfig(
        model_name="gpt2",  # Use small model for demo
        system_prompt="You are a helpful AI assistant. Be friendly and informative.",
        temperature=0.8,
        max_new_tokens=256
    )
    
    agent = MultiTurnAgent(config)
    await agent.initialize()
    
    # Step 2: Create an environment
    logger.info("Step 2: Creating environment...")
    
    # Simple conversation scenarios
    scenarios = [
        {
            "id": "general_help",
            "topic": "general_assistance",
            "context": "User needs general help",
            "user_responses": [
                "Hi there! Can you help me with something?",
                "I'm trying to learn about machine learning. Where should I start?",
                "That's helpful. What about practical projects?",
                "Great suggestions! Thank you for your help."
            ]
        },
        {
            "id": "creative_help", 
            "topic": "creative_writing",
            "context": "User wants creative writing help",
            "user_responses": [
                "I'm working on a short story and need some inspiration.",
                "It's about a detective in a futuristic city. Any ideas for plot twists?",
                "I like that! How can I make the character more interesting?",
                "Perfect! That gives me a lot to work with."
            ]
        }
    ]
    
    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=8
    )
    
    # Step 3: Define reward function (optional - can use defaults)
    logger.info("Step 3: Setting up rewards...")
    
    from grpo_agent_framework.core.reward import CompositeReward
    reward_fn = CompositeReward([
        HelpfulnessReward(weight=0.7),
        SafetyReward(weight=0.3)
    ])
    
    # Step 4: Train the agent
    logger.info("Step 4: Training agent...")
    
    trained_agent = await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=20,  # Very small for demo
        profile="balanced"
    )
    
    logger.info("Training completed!")
    
    # Step 5: Test the trained agent
    logger.info("\nStep 5: Testing trained agent...")
    await test_conversation(trained_agent)


async def test_conversation(agent):
    """Test the trained agent with a sample conversation"""
    
    conversation = [
        {"role": "system", "content": agent.config.system_prompt}
    ]
    
    test_inputs = [
        "Hello! I'm new to programming. Can you help me get started?",
        "What programming language would you recommend for a beginner?",
        "That sounds good. Where can I find good tutorials?",
        "Thank you! This is very helpful."
    ]
    
    logger.info("\n" + "="*40)
    logger.info("TEST CONVERSATION")
    logger.info("="*40)
    
    for user_input in test_inputs:
        logger.info(f"\nUser: {user_input}")
        
        conversation.append({"role": "user", "content": user_input})
        response = await agent.generate_response(conversation)
        
        logger.info(f"Assistant: {response}")
        conversation.append({"role": "assistant", "content": response})
    
    logger.info("\n" + "="*40)


async def auto_training_example():
    """Example using automatic training optimization"""
    
    logger.info("\nAuto Training Example")
    logger.info("=" * 30)
    
    from grpo_agent_framework.training.train import AutoTrainer
    
    # Create agent and environment (same as before)
    config = AgentConfig(
        model_name="gpt2",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.8
    )
    
    agent = MultiTurnAgent(config)
    await agent.initialize()
    
    scenarios = [
        {
            "id": "help_request",
            "topic": "assistance",
            "context": "User needs help",
            "user_responses": [
                "I need help with something.",
                "Can you explain how to write a good email?",
                "What about the subject line?",
                "Thanks, that's very helpful!"
            ]
        }
    ]
    
    environment = ConversationEnvironment(scenarios=scenarios, max_turns=6)
    
    # Use AutoTrainer for automatic optimization
    auto_trainer = AutoTrainer(
        auto_adjust=True,
        early_stopping=True
    )
    
    logger.info("Starting automatic training...")
    trained_agent = await auto_trainer.train(
        agent=agent,
        environment=environment,
        num_episodes=15  # Small for demo
    )
    
    logger.info("Automatic training completed!")
    await test_conversation(trained_agent)


async def custom_reward_example():
    """Example with custom reward function"""
    
    logger.info("\nCustom Reward Example")
    logger.info("=" * 25)
    
    from grpo_agent_framework.core.reward import RewardFunction, RewardResult
    
    class PolitenesReward(RewardFunction):
        """Custom reward for politeness"""
        
        async def compute_reward(self, turns, context=None):
            polite_phrases = [
                "please", "thank you", "you're welcome", 
                "excuse me", "i'm sorry", "pardon me"
            ]
            
            assistant_turns = [t for t in turns if t.role == "assistant"]
            if not assistant_turns:
                return RewardResult(score=0.0, breakdown={}, metadata={})
            
            total_score = 0.0
            breakdown = {}
            
            for i, turn in enumerate(assistant_turns):
                content_lower = turn.content.lower()
                politeness_count = sum(
                    1 for phrase in polite_phrases 
                    if phrase in content_lower
                )
                score = min(politeness_count * 0.3, 1.0)
                total_score += score
                breakdown[f"turn_{i}_politeness"] = score
            
            avg_score = total_score / len(assistant_turns)
            
            return RewardResult(
                score=avg_score,
                breakdown=breakdown,
                metadata={"polite_phrases_found": sum(breakdown.values())}
            )
    
    # Create agent with politeness emphasis
    config = AgentConfig(
        model_name="gpt2",
        system_prompt="You are a very polite and courteous AI assistant. Always use please, thank you, and other polite expressions.",
        temperature=0.7
    )
    
    agent = MultiTurnAgent(config)
    await agent.initialize()
    
    # Simple environment
    scenarios = [{"id": "polite_test", "user_responses": ["Hello!", "Can you help me?", "Thank you!"]}]
    environment = ConversationEnvironment(scenarios=scenarios, max_turns=4)
    
    # Use custom reward
    from grpo_agent_framework.core.reward import CompositeReward
    reward_fn = CompositeReward([
        PolitenesReward(weight=0.6),
        HelpfulnessReward(weight=0.4)
    ])
    
    logger.info("Training with custom politeness reward...")
    trained_agent = await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=10,
        profile="conservative"  # More stable for custom rewards
    )
    
    logger.info("Custom reward training completed!")
    await test_conversation(trained_agent)


async def main():
    """Run all examples"""
    
    # Basic example
    await basic_example()
    
    # Auto training example
    await auto_training_example()
    
    # Custom reward example
    await custom_reward_example()
    
    logger.info("\n🎉 All examples completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Try the customer service example: python -m grpo_agent_framework.examples.customer_service_agent")
    logger.info("2. Explore the documentation for advanced features")
    logger.info("3. Create your own custom agents and environments")


if __name__ == "__main__":
    asyncio.run(main())
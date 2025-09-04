"""
Example: Training a Customer Service Agent using GRPO

This example demonstrates how to train a multi-turn conversational agent
for customer service using the GRPO Agent Framework.
"""

import asyncio
import logging
import json
from pathlib import Path

# Framework imports
from stateset_agents import (
    MultiTurnAgent, ConversationEnvironment, 
    CompositeReward, HelpfulnessReward, SafetyReward, ConcisenessReward,
    train, AutoTrainer
)
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.trajectory import ConversationTurn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Customer service scenarios
CUSTOMER_SERVICE_SCENARIOS = [
    {
        "id": "product_inquiry_1",
        "topic": "product_features",
        "context": "Customer interested in smartphone features",
        "user_goal": "Learn about camera quality and battery life",
        "user_responses": [
            "Hi, I'm looking for a new smartphone. Can you tell me about the camera?",
            "What about the battery life? How long does it last?",
            "That sounds good. Are there any deals available?",
            "Perfect, how do I place an order?",
            "Thank you for your help!"
        ]
    },
    {
        "id": "technical_support_1", 
        "topic": "troubleshooting",
        "context": "Customer having WiFi connectivity issues",
        "user_goal": "Fix WiFi connection problems",
        "user_responses": [
            "My WiFi keeps disconnecting every few minutes. Can you help?",
            "I've tried turning it off and on. It still doesn't work properly.",
            "Let me try that. Okay, I've reset the router.",
            "It seems to be working better now. Anything else I should check?",
            "Great, thanks for the help!"
        ]
    },
    {
        "id": "billing_inquiry_1",
        "topic": "billing", 
        "context": "Customer has questions about their bill",
        "user_goal": "Understand charges on bill",
        "user_responses": [
            "I have a question about some charges on my recent bill.",
            "There's a charge for $25 that I don't recognize.",
            "Oh, I see. Is there a way to avoid this charge in the future?",
            "That makes sense. Can you help me set that up?",
            "Perfect, thank you for explaining everything!"
        ]
    },
    {
        "id": "return_request_1",
        "topic": "returns",
        "context": "Customer wants to return a product",
        "user_goal": "Process a return",
        "user_responses": [
            "I need to return a product I bought last week.",
            "It's a bluetooth speaker, order number CS12345.",
            "It doesn't charge properly, I think the port is defective.",
            "Yes, I still have all the original packaging.",
            "That would be great, thank you!"
        ]
    }
]


class CustomerServiceAgent(MultiTurnAgent):
    """
    Specialized customer service agent with domain-specific behaviors
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        config = AgentConfig(
            model_name=model_name,
            system_prompt="""You are a professional customer service representative. Your goals are to:
1. Be helpful and solve customer problems
2. Be polite and empathetic  
3. Provide accurate information
4. Be concise but thorough
5. Always maintain a positive, solution-oriented attitude

Guidelines:
- Listen carefully to customer concerns
- Ask clarifying questions when needed
- Provide step-by-step instructions for technical issues
- Offer alternatives when the first solution doesn't work
- Thank customers for their patience""",
            temperature=0.7,
            max_new_tokens=256,
            memory_window=8  # Remember last 8 turns
        )
        super().__init__(config)
        
        # Customer service specific state
        self.customer_issue_type = None
        self.resolution_attempted = False
        
    async def process_turn(self, conversation_history, user_input, context=None):
        """Override to add customer service specific logic"""
        
        # Classify the issue type if not already done
        if not self.customer_issue_type:
            self.customer_issue_type = self._classify_issue(user_input)
            logger.info(f"Classified issue as: {self.customer_issue_type}")
        
        # Generate response using parent method
        turn = await super().process_turn(conversation_history, user_input, context)
        
        # Add customer service metadata
        turn.metadata.update({
            "issue_type": self.customer_issue_type,
            "resolution_attempted": self.resolution_attempted
        })
        
        return turn
    
    def _classify_issue(self, user_input: str) -> str:
        """Simple issue classification"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["bill", "charge", "payment", "cost"]):
            return "billing"
        elif any(word in user_input_lower for word in ["broken", "not working", "problem", "issue", "fix"]):
            return "technical_support"
        elif any(word in user_input_lower for word in ["return", "exchange", "refund"]):
            return "returns"
        elif any(word in user_input_lower for word in ["product", "feature", "spec", "information"]):
            return "product_inquiry"
        else:
            return "general"


class CustomerServiceEnvironment(ConversationEnvironment):
    """
    Specialized environment for customer service training
    """
    
    def __init__(self):
        super().__init__(
            scenarios=CUSTOMER_SERVICE_SCENARIOS,
            max_turns=12,
            persona="You are a customer with a specific need or problem."
        )
        self.current_response_index = 0
    
    async def _generate_user_response(self, agent_turn, state):
        """Generate realistic customer responses"""
        scenario = state.context.get("scenario", {})
        user_responses = scenario.get("user_responses", [])
        
        # Use predefined responses that follow a realistic customer journey
        if self.current_response_index < len(user_responses):
            response_content = user_responses[self.current_response_index]
            self.current_response_index += 1
        else:
            # Fallback responses for extended conversations
            fallback_responses = [
                "Can you tell me more about that?",
                "I see, what else should I know?", 
                "That helps, thank you.",
                "Is there anything else I should be aware of?",
                "Perfect, I think that covers everything."
            ]
            response_content = fallback_responses[min(
                self.current_response_index - len(user_responses),
                len(fallback_responses) - 1
            )]
            self.current_response_index += 1
        
        return ConversationTurn(
            role="user",
            content=response_content,
            metadata={
                "generated": True, 
                "response_index": self.current_response_index - 1,
                "scenario_id": scenario.get("id")
            }
        )
    
    async def reset(self, scenario=None):
        """Reset environment state"""
        self.current_response_index = 0
        return await super().reset(scenario)
    
    async def _calculate_step_reward(self, agent_turn, state):
        """Calculate reward for customer service specific behaviors"""
        base_reward = await super()._calculate_step_reward(agent_turn, state)
        
        content = agent_turn.content.lower()
        
        # Bonus for customer service language
        service_phrases = [
            "i understand", "i apologize", "let me help", "i'd be happy to",
            "thank you for", "i can assist", "how can i help"
        ]
        service_bonus = sum(0.1 for phrase in service_phrases if phrase in content)
        
        # Bonus for solution-oriented language  
        solution_phrases = [
            "here's what we can do", "let's try", "the solution is",
            "i recommend", "you can", "this should fix"
        ]
        solution_bonus = sum(0.15 for phrase in solution_phrases if phrase in content)
        
        # Penalty for unprofessional language
        unprofessional = ["can't help", "not my problem", "impossible", "no way"]
        professional_penalty = sum(-0.2 for phrase in unprofessional if phrase in content)
        
        total_reward = base_reward + service_bonus + solution_bonus + professional_penalty
        return max(0.0, min(1.0, total_reward))


def create_customer_service_reward():
    """Create specialized reward function for customer service"""
    return CompositeReward([
        HelpfulnessReward(weight=0.4),  # Primary focus on being helpful
        SafetyReward(weight=0.25),      # Professional and appropriate
        ConcisenessReward(weight=0.2),  # Clear and to the point
        CustomerServiceQualityReward(weight=0.15)  # Domain-specific quality
    ])


class CustomerServiceQualityReward(CompositeReward):
    """Custom reward for customer service quality metrics"""
    
    def __init__(self, weight=1.0):
        # We'll implement this as a simple function for the example
        super().__init__([], "CustomerServiceQualityReward")
        self.weight = weight
    
    async def compute_reward(self, turns, context=None):
        """Evaluate customer service specific quality"""
        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return {"score": 0.0, "breakdown": {}, "metadata": {}}
        
        total_score = 0.0
        breakdown = {}
        
        for i, turn in enumerate(assistant_turns):
            score = self._evaluate_service_quality(turn.content)
            total_score += score
            breakdown[f"turn_{i}_service_quality"] = score
        
        avg_score = total_score / len(assistant_turns)
        
        return {
            "score": avg_score,
            "breakdown": breakdown,
            "metadata": {"num_assistant_turns": len(assistant_turns)}
        }
    
    def _evaluate_service_quality(self, content):
        """Evaluate customer service quality"""
        score = 0.0
        content_lower = content.lower()
        
        # Empathy indicators
        empathy_phrases = ["i understand", "i see", "that must be", "i apologize"]
        empathy_score = min(sum(0.2 for phrase in empathy_phrases if phrase in content_lower), 0.4)
        score += empathy_score
        
        # Solution orientation
        if any(word in content_lower for word in ["solution", "fix", "resolve", "help"]):
            score += 0.3
        
        # Professional tone
        if any(phrase in content_lower for phrase in ["happy to help", "glad to assist"]):
            score += 0.2
        
        # Clear next steps
        if any(phrase in content_lower for phrase in ["next step", "what you need to do", "here's how"]):
            score += 0.1
        
        return min(score, 1.0)


async def main():
    """Main training example"""
    
    logger.info("Starting Customer Service Agent Training Example")
    logger.info("=" * 60)
    
    # Create agent
    logger.info("Creating customer service agent...")
    agent = CustomerServiceAgent(model_name="openai/gpt-oss-120b")  # Use the openai/gpt-oss-120b model
    await agent.initialize()
    
    # Create environment
    logger.info("Setting up customer service environment...")
    environment = CustomerServiceEnvironment()
    
    # Create reward function
    logger.info("Configuring reward function...")
    reward_fn = create_customer_service_reward()
    
    # Option 1: Manual training with specific configuration
    logger.info("\nOption 1: Manual training configuration")
    trained_agent_manual = await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=100,  # Small number for demo
        profile="balanced",
        save_path="./checkpoints/customer_service_manual"
    )
    
    # Option 2: Automatic training with optimization
    logger.info("\nOption 2: Automatic training with optimization")
    auto_trainer = AutoTrainer(auto_adjust=True, early_stopping=True)
    
    # Create fresh agent for auto training
    agent_auto = CustomerServiceAgent(model_name="openai/gpt-oss-120b")
    await agent_auto.initialize()
    
    trained_agent_auto = await auto_trainer.train(
        agent=agent_auto,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=100,
        save_path="./checkpoints/customer_service_auto"
    )
    
    logger.info("\nTraining completed!")
    
    # Demo conversation with trained agent
    logger.info("\nDemo conversation with trained agent:")
    await demo_conversation(trained_agent_auto)


async def demo_conversation(agent):
    """Demonstrate a conversation with the trained agent"""
    
    conversation_history = [
        {"role": "system", "content": agent.config.system_prompt}
    ]
    
    demo_user_inputs = [
        "Hi, I'm having trouble with my recent order.",
        "My bluetooth headphones aren't connecting to my phone.",
        "I've tried turning Bluetooth off and on, but it still won't pair.",
        "Okay, let me try that reset process.",
        "That worked! Thank you so much for your help."
    ]
    
    logger.info("\n" + "="*50)
    logger.info("DEMO CONVERSATION")
    logger.info("="*50)
    
    for user_input in demo_user_inputs:
        logger.info(f"\nCustomer: {user_input}")
        
        # Add user input to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate agent response
        response = await agent.generate_response(conversation_history)
        logger.info(f"Agent: {response}")
        
        # Add agent response to history
        conversation_history.append({"role": "assistant", "content": response})
    
    logger.info("\n" + "="*50)


async def evaluate_agent(agent, environment, num_episodes=10):
    """Evaluate trained agent performance"""
    
    logger.info(f"\nEvaluating agent over {num_episodes} episodes...")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        try:
            # Create agent wrapper
            async def agent_fn(history, context):
                return await agent.generate_response(history, context)
            
            # Run episode
            trajectory = await environment.run_episode(agent_fn)
            
            total_rewards.append(trajectory.total_reward)
            episode_lengths.append(trajectory.episode_length)
            
            logger.info(f"Episode {episode + 1}: Reward = {trajectory.total_reward:.3f}, Length = {trajectory.episode_length}")
            
        except Exception as e:
            logger.error(f"Episode {episode + 1} failed: {e}")
    
    if total_rewards:
        import numpy as np
        logger.info(f"\nEvaluation Results:")
        logger.info(f"Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        logger.info(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        logger.info(f"Success Rate: {len([r for r in total_rewards if r > 0.5]) / len(total_rewards):.2%}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
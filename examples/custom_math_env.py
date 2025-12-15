"""
Custom Environment Example: Math Tutor

This script demonstrates how to create a custom environment and train an agent
to perform well in it. We define a `MathEnvironment` that generates simple
arithmetic problems and rewards the agent for providing the correct answer.

Usage:
    python examples/custom_math_env.py
"""

import asyncio
import logging
import random
import uuid
from typing import Tuple, Dict, Any, Optional

from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
from stateset_agents.core.environment import Environment, EnvironmentState, EpisodeStatus
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.training.gspo_trainer import GSPOConfig, train_with_gspo
from stateset_agents.training.config import TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathRewardFunction(RewardFunction):
    """Simple reward function that checks if the answer is correct."""
    
    async def compute_reward(
        self,
        trajectory: Any,  # List[ConversationTurn] or MultiTurnTrajectory
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RewardResult:
        """Compute reward based on the last agent response."""
        
        # Extract turns
        if hasattr(trajectory, 'turns'):
            turns = trajectory.turns
        else:
            turns = list(trajectory)
            
        if not turns:
            return RewardResult(score=0.0, components={"error": "no_turns"})

        # Get last assistant turn
        last_turn = turns[-1]
        if last_turn.role != "assistant":
            # If the last turn wasn't the assistant (e.g. user response in step), check previous
            if len(turns) >= 2 and turns[-2].role == "assistant":
                last_turn = turns[-2]
            else:
                 return RewardResult(score=0.0)

        # Retrieve the correct answer from context
        # In a real scenario, context should be passed down. 
        # If context is not available here, we might need to store it in the turn metadata.
        correct_answer = None
        if context:
            correct_answer = context.get("correct_answer")
        elif last_turn.metadata and "correct_answer" in last_turn.metadata:
            correct_answer = last_turn.metadata.get("correct_answer")
            
        if correct_answer is None:
            return RewardResult(score=0.0)

        # Check if the correct answer is in the response
        response = last_turn.content.strip()
        
        # Simple string match for the number
        score = 0.0
        if str(correct_answer) in response:
            score = 1.0
        else:
            score = 0.0
            
        return RewardResult(score=score, components={"correct": score})


class MathEnvironment(Environment):
    """
    A simple environment that poses math problems to the agent.
    """
    
    def __init__(self, max_turns: int = 3, **kwargs):
        # We pass our custom reward function
        super().__init__(max_turns=max_turns, reward_fn=MathRewardFunction(), **kwargs)
        self.current_problem = None
        self.correct_answer = None

    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        """Generate a new math problem."""
        
        # Generate random addition problem
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        self.correct_answer = a + b
        self.current_problem = f"What is {a} + {b}?"
        
        episode_id = str(uuid.uuid4())
        
        # Create initial state
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "problem": self.current_problem,
                "correct_answer": self.correct_answer,
                "history": []
            }
        )
        self.active_episodes[episode_id] = state
        self._last_state = state # Helper for implicit step() calls
        return state

    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """The initial prompt usually sets the system role."""
        return "You are a math tutor. Solve the problem concisely."

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        
        new_state = state.copy()
        new_state.turn_count += 1
        
        # Store the agent's action in context history
        history = list(new_state.context.get("history", []))
        history.append({"role": "assistant", "content": action.content})
        
        # Environment logic:
        # In this simple env, we just present the problem at start (via user msg in training loop)
        # or we could have a multi-turn dialogue.
        # For this example, let's treat it as:
        # Turn 1: User asks problem (handled by trainer usually sending initial prompt/user msg) -> Agent answers.
        # If Agent answers correctly, done. If not, maybe give a hint? 
        # For simplicity, we'll just evaluate immediately.
        
        # Calculate reward
        reward = 0.0
        if self.reward_fn:
            # Pass the correct answer via context
            reward_res = await self.reward_fn.compute_reward([action], context=new_state.context)
            reward = float(reward_res.score)

        # Check termination
        done = False
        if reward > 0.9: # Correct answer
            done = True
            new_state.status = EpisodeStatus.COMPLETED
        elif new_state.turn_count >= self.max_turns:
            done = True
            new_state.status = EpisodeStatus.TIMEOUT
            
        # Generate environment response (the "User")
        # If not done, user might say "Try again"
        if not done:
            user_content = "That doesn't look right. Please try again."
        else:
            user_content = "Correct! Great job."
            
        user_turn = ConversationTurn(role="user", content=user_content)
        history.append({"role": "user", "content": user_content})
        new_state.context["history"] = history
        
        info = {
            "env_response": user_turn
        }
        
        self._last_state = new_state
        return new_state, reward, done, info

    def clone(self) -> "MathEnvironment":
        return MathEnvironment(max_turns=self.max_turns)


async def main():
    logger.info("Initializing Math Agent...")
    # Use GPT-2 for demo purposes
    agent_config = AgentConfig(
        model_name="gpt2", 
        system_prompt="You are a helpful math tutor.",
        max_new_tokens=20
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    logger.info("Initializing Custom Math Environment...")
    environment = MathEnvironment(max_turns=3)

    # Configure Training
    logger.info("Configuring Training...")
    base_config = TrainingConfig(
        run_name="math_tutor_demo",
        output_dir="./outputs/math_tutor",
        num_train_epochs=1,
        per_device_train_batch_size=2
    )
    
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        num_outer_iterations=2,
        generations_per_iteration=4,
        num_generations=2,
        learning_rate=1e-5,
        report_to="none"
    )

    # Note: We don't strictly need a separate reward_model passed to train_with_gspo 
    # if the environment handles rewards, BUT the trainer might expect one.
    # We can pass the environment's reward function or a dummy one.
    # The current GSPO trainer implementation likely uses the passed reward_model 
    # to compute rewards for the *generated sequences* before the environment step 
    # (or in parallel). Let's pass the same one.
    
    logger.info("Starting Training...")
    await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=environment.reward_function
    )
    
    logger.info("Training Complete!")
    
    # Test
    test_problem = "What is 5 + 5?"
    logger.info(f"Testing with: {test_problem}")
    response = await agent.generate_response([{"role": "user", "content": test_problem}])
    logger.info(f"Agent Answer: {response}")

if __name__ == "__main__":
    asyncio.run(main())

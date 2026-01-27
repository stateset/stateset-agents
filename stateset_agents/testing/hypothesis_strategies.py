"""
Hypothesis strategies for property-based testing.

These strategies generate valid test data for property-based testing
of StateSet Agents components.
"""

from typing import List

from hypothesis import strategies as st
from hypothesis.strategies import composite


@composite
def conversation_turns(draw):
    """Generate valid conversation turns."""
    role = draw(st.sampled_from(["user", "assistant", "system"]))
    content = draw(
        st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())
    )
    metadata = draw(
        st.dictionaries(
            st.sampled_from(["timestamp", "turn_id", "emotion"]),
            st.text(max_size=100),
            max_size=3,
        )
    )

    return {
        "role": role,
        "content": content,
        "metadata": metadata if draw(st.booleans()) else None,
    }


@composite
def reward_values(draw):
    """Generate valid reward values with appropriate bounds."""
    return draw(st.floats(min_value=-10.0, max_value=10.0))


@composite
def trajectory_configs(draw):
    """Generate valid trajectory configurations."""
    max_turns = draw(st.integers(min_value=1, max_value=100))
    num_turns = draw(st.integers(min_value=0, max_value=max_turns))

    return {
        "max_turns": max_turns,
        "num_turns": num_turns,
        "turns": draw(st.lists(conversation_turns(), min_size=num_turns, max_size=num_turns)),
        "rewards": draw(st.lists(reward_values(), min_size=num_turns, max_size=num_turns)),
    }


@composite
def model_configs(draw):
    """Generate valid model configurations."""
    model_names = [
        "gpt2",
        "gpt2-medium",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
    ]

    return {
        "model_name": draw(st.sampled_from(model_names)),
        "max_new_tokens": draw(st.integers(min_value=1, max_value=4096)),
        "temperature": draw(st.floats(min_value=0.0, max_value=2.0)),
        "top_p": draw(st.floats(min_value=0.0, max_value=1.0)),
        "top_k": draw(st.integers(min_value=1, max_value=100)),
    }


@composite
def training_configs(draw):
    """Generate valid training configurations."""
    return {
        "learning_rate": draw(st.floats(min_value=1e-7, max_value=1e-3)),
        "batch_size": draw(st.integers(min_value=1, max_value=64)),
        "num_episodes": draw(st.integers(min_value=1, max_value=1000)),
        "gamma": draw(st.floats(min_value=0.9, max_value=1.0)),
        "lambda_": draw(st.floats(min_value=0.9, max_value=1.0)),
    }


@composite
def conversation_scenarios(draw):
    """Generate valid conversation scenarios for environments."""
    topics = ["customer_service", "technical_support", "sales", "general"]

    return {
        "id": draw(st.uuids()).hex,
        "topic": draw(st.sampled_from(topics)),
        "context": draw(st.text(min_size=10, max_size=500)),
        "user_responses": draw(
            st.lists(
                st.text(min_size=5, max_size=200),
                min_size=1,
                max_size=10,
            )
        ),
    }


# Composite strategies for complex objects
valid_conversations = st.lists(conversation_turns(), min_size=0, max_size=50)
valid_rewards = st.lists(reward_values(), min_size=0, max_size=50)
valid_trajectories = trajectory_configs()
valid_model_configs = model_configs()
valid_training_configs = training_configs()
valid_scenarios = st.lists(conversation_scenarios(), min_size=1, max_size=20)

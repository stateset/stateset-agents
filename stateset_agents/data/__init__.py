"""
Data utilities for offline RL and conversation datasets.

Provides D4RL-style dataset loading, replay buffers, embedding caching for
conversation trajectory data, and verifiable-reward benchmark datasets
(GSM8K and friends).
"""

from .conversation_dataset import (
    ConversationDataset,
    ConversationDatasetConfig,
    ConversationReplayBuffer,
    EmbeddingCache,
)
from .customer_support_bench import (
    SupportRewardComposite,
    SupportScenario,
    load_support_scenarios,
    make_support_scenarios,
)
from .gsm8k import (
    GSM8KExample,
    GSM8KReward,
    extract_gold_answer,
    extract_predicted_answer,
    load_gsm8k,
    make_gsm8k_scenarios,
)
from .tool_calling_bench import (
    SAMPLE_TOOLS,
    ToolCallReward,
    ToolCallScenario,
    extract_tool_call,
    load_tool_call_scenarios,
    make_tool_call_scenarios,
)

__all__ = [
    "ConversationDataset",
    "ConversationDatasetConfig",
    "ConversationReplayBuffer",
    "EmbeddingCache",
    "GSM8KExample",
    "GSM8KReward",
    "SAMPLE_TOOLS",
    "SupportRewardComposite",
    "SupportScenario",
    "ToolCallReward",
    "ToolCallScenario",
    "extract_gold_answer",
    "extract_predicted_answer",
    "extract_tool_call",
    "load_gsm8k",
    "load_support_scenarios",
    "load_tool_call_scenarios",
    "make_gsm8k_scenarios",
    "make_support_scenarios",
    "make_tool_call_scenarios",
]

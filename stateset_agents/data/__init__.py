"""
Data utilities for offline RL and conversation datasets.

Provides D4RL-style dataset loading, replay buffers, embedding caching for
conversation trajectory data, and verifiable-reward benchmark datasets
(GSM8K and friends).

Exports are resolved lazily (PEP 562) so that importing a light submodule --
or the package itself -- does not drag in heavy optional dependencies such as
``torch`` or ``sentence_transformers`` via ``conversation_dataset``.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    "ConversationDataset": "conversation_dataset",
    "ConversationDatasetConfig": "conversation_dataset",
    "ConversationReplayBuffer": "conversation_dataset",
    "EmbeddingCache": "conversation_dataset",
    "SupportRewardComposite": "customer_support_bench",
    "SupportScenario": "customer_support_bench",
    "load_support_scenarios": "customer_support_bench",
    "make_support_scenarios": "customer_support_bench",
    "GSM8KExample": "gsm8k",
    "GSM8KReward": "gsm8k",
    "PartialCreditGSM8KReward": "gsm8k",
    "extract_gold_answer": "gsm8k",
    "extract_predicted_answer": "gsm8k",
    "load_gsm8k": "gsm8k",
    "make_gsm8k_scenarios": "gsm8k",
    "SAMPLE_TOOLS": "tool_calling_bench",
    "ToolCallReward": "tool_calling_bench",
    "ToolCallScenario": "tool_calling_bench",
    "extract_tool_call": "tool_calling_bench",
    "load_tool_call_scenarios": "tool_calling_bench",
    "make_tool_call_scenarios": "tool_calling_bench",
    "from_langchain_json": "trajectory_ingest",
    "from_openai_jsonl": "trajectory_ingest",
    "from_openai_messages": "trajectory_ingest",
    "to_grading_history": "trajectory_ingest",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

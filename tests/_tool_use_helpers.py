"""Shared fixtures for the tool-use RL tests (tiny GPT-2, hermetic tokenizer).

``ForcedTurnAgent`` wraps a real ``MultiTurnAgent`` around a tiny random GPT-2:
the reply *text* comes from a script, so episodes contain real tool calls,
but everything the trainer consumes is what ``MultiTurnAgent.generate_turn``
records for a sampled reply: ``prompt_token_ids`` from the agent's own prompt
rendering and tokenization, ``token_ids`` of the reply, and the model's
``sampler_log_probs`` of those ids at temperature 1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from stateset_agents.core.trajectory import ConversationTurn
from tests._tiny_tokenizer import tiny_tokenizer

VOCAB_SIZE = 103
#: Short tool instructions keep prompts inside the tiny model's context.
SYSTEM_PROMPT = "Use the calculator tool, then answer."

Script = Callable[[list[dict[str, Any]], int], str]


def make_tokenizer(max_length: int = 768) -> Any:
    tokenizer = tiny_tokenizer(VOCAB_SIZE)
    tokenizer.model_max_length = max_length
    return tokenizer


def make_model(seed: int, *, n_positions: int = 768, n_embd: int = 32) -> Any:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=n_embd,
            n_layer=1,
            n_head=2,
            vocab_size=VOCAB_SIZE,
            n_positions=n_positions,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def make_agent(
    model: Any, tokenizer: Any, *, max_new_tokens: int = 48, **config: Any
) -> Any:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(
        AgentConfig(
            model_name="gpt2",
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            use_chat_template=False,
            **config,
        )
    )
    agent.model = model
    agent.tokenizer = tokenizer
    agent.generation_config = agent._build_generation_config()
    return agent


class ForcedTurnAgent:
    """A ``MultiTurnAgent`` whose sampled text is scripted (ids and log-probs real)."""

    def __init__(self, agent: Any, script: Script) -> None:
        self.agent = agent
        self.script = script
        self.model = agent.model
        self.tokenizer = agent.tokenizer
        self.config = agent.config
        self.calls = 0

    async def reset(self) -> None:
        return None

    async def generate_turn(
        self, history: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> ConversationTurn:
        import torch

        prompt, _ = self.agent._render_prompt(history, context)
        prompt_ids = [
            int(t) for t in self.agent._tokenize_prompt(prompt)["input_ids"][0]
        ]
        text = self.script(history, self.calls)
        self.calls += 1
        token_ids = [
            int(t) for t in self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ]
        with torch.no_grad():
            ids = torch.tensor([prompt_ids + token_ids])
            logits = self.model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            picked = log_probs.gather(-1, torch.tensor(token_ids).unsqueeze(-1))
        return ConversationTurn(
            role="assistant",
            content=self.tokenizer.decode(token_ids, skip_special_tokens=True),
            metadata={
                "generated": True,
                "prompt_token_ids": prompt_ids,
                "token_ids": token_ids,
                "sampler_log_probs": [float(x) for x in picked.squeeze(-1)],
            },
        )

    async def generate_turns_batch(
        self,
        histories: Sequence[list[dict[str, Any]]],
        contexts: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[ConversationTurn]:
        contexts = list(contexts or [None] * len(histories))
        return [
            await self.generate_turn(list(history), context)
            for history, context in zip(histories, contexts, strict=True)
        ]

    async def generate_response(
        self, history: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> str:
        return str((await self.generate_turn(history, context)).content)


def tool_call(expression: str, name: str = "calculator") -> str:
    return (
        f'<tool_call>{{"name": "{name}", "arguments": '
        f'{{"expression": "{expression}"}}}}</tool_call>'
    )


def calculator_script(replies_by_episode: Sequence[Sequence[str]]) -> Script:
    """Episode ``k`` (by the order its first turn is generated) replies in order.

    Later turns find their episode by its first replies (which must differ
    between episodes), so lockstep (batched) and sequential rollouts get the
    same replies.
    """
    firsts = [replies[0] for replies in replies_by_episode]
    assert len(set(firsts)) == len(firsts), "first replies must be unique"
    started: list[int] = []

    def script(history: list[dict[str, Any]], call: int) -> str:
        turn = sum(1 for m in history if m.get("role") == "assistant")
        if turn == 0:
            started.append(call)
            episode = len(started) - 1
        else:
            episode = _episode_of(history, replies_by_episode)
        replies = replies_by_episode[episode % len(replies_by_episode)]
        return replies[min(turn, len(replies) - 1)]

    return script


def _episode_of(history: list[dict[str, Any]], replies: Sequence[Sequence[str]]) -> int:
    """The scripted episode whose first replies match the history's."""
    said = [m["content"] for m in history if m.get("role") == "assistant"]
    for index, script in enumerate(replies):
        if list(script[: len(said)]) == said:
            return index
    raise AssertionError(f"no scripted episode starts with {said}")

"""GRPO with a math verifier teaches a tiny policy to call the calculator.

Task: four "What is 7919*7907?"-style products a one-layer GPT-2 cannot
compute. The policy can answer directly ("The answer is 42.", never right)
or call the calculator (``<tool_call>{"name": "calculator", ...}</tool_call>``),
read the ``tool`` observation and answer from it. The only reward is
``MathAnswerVerifier`` on the final answer, applied by
``MultiTurnGRPOTrainer`` to real ``ToolUseEnvironment`` episodes on the native
token path (tool observations are context, never trained on).

A tiny model cannot learn to copy digits within a CPU test budget, so a short
supervised warm start (teacher-forced, assistant tokens only) teaches the
call format and the per-question calls and answers, and leaves the *decision*
to use the tool at under one half. GRPO must then raise the rate of calling
the tool, which is measured deterministically as the policy's probability of
opening its first reply with ``<`` (the start of ``<tool_call>``) on held-out
questions it never trained on, by the probability of the exact correct call
on the training questions (teacher forced, so deterministic), and by
sampling 64 episodes: the rate of calls that execute with the right result
and the verifier pass rate. A KL penalty to the warm-started policy keeps the
call format intact while the decision moves (without it, one seed in three
drifted the call text and gained no correct calls). A sign-flipped reward
must move the decision and the correct-call rate the other way.

Sampling turns float rounding into different trajectories, so a seed's run is
reproducible only on the same process history and kernels (the same seed
moved the held-out rate 0.43 -> 0.28 in a fresh process and 0.43 -> 0.33 after
another run); the thresholds keep margin for that.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.environments.tool_use import ToolUseEnvironment  # noqa: E402
from stateset_agents.core.reward_base import (  # noqa: E402
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.tool_registry import calculator_tool  # noqa: E402
from stateset_agents.rewards.math_verifier import MathAnswerVerifier  # noqa: E402
from tests._tool_use_helpers import make_agent, make_model, make_tokenizer  # noqa: E402
from tests.learning._toy import seeds  # noqa: E402

#: A warm start plus 30 two-turn GRPO steps takes about a minute per seed on
#: one CPU thread, beyond the suite's 60s default.
pytestmark = [pytest.mark.learning, pytest.mark.timeout(600)]

SYSTEM = "Tools: calculator."
TRAIN = [
    ("7919*7907", 62615533),
    ("6841*5323", 36414643),
    ("9973*8761", 87373453),
    ("4567*8923", 40751341),
]
HELD_OUT = ["3371*9029", "5003*7759", "8191*6151", "2903*7331"]
WARM_START_STEPS = 600
#: The warm start shows a tool call in this share of its demonstrations.
WARM_TOOL_SHARE = 0.4
RL_STEPS = 30
RL_LEARNING_RATE = 3e-4
#: KL to the warm-started policy keeps the call format while the decision moves.
RL_BETA = 0.1


def _question(expression: str) -> str:
    return f"What is {expression}?"


def _call(expression: str) -> str:
    return (
        '<tool_call>{"name": "calculator", "arguments": '
        f'{{"expression": "{expression}"}}}}</tool_call>'
    )


_WARM_STATES: dict[int, dict[str, Any]] = {}


class _Policy:
    """Tiny GPT-2 agent plus the rendering helpers the test needs."""

    def __init__(self, seed: int) -> None:
        self.tokenizer = make_tokenizer()
        self.model = make_model(seed, n_embd=64)
        self.agent = make_agent(
            self.model,
            self.tokenizer,
            max_new_tokens=96,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
        )
        self.eos = int(self.tokenizer.eos_token_id)
        self.tool_start = int(self.tokenizer.convert_tokens_to_ids("<"))

    def ids(self, text: str) -> list[int]:
        return [
            int(t) for t in self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ]

    def prompt(self, history: list[dict[str, str]]) -> list[int]:
        return self.ids(self.agent._render_prompt(history, None)[0])

    def opening(self, expression: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _question(expression)},
        ]

    @torch.no_grad()
    def tool_rate(self, expressions: list[str]) -> float:
        """Mean probability that the first reply opens a tool call."""
        self.model.eval()
        try:
            values = []
            for expression in expressions:
                ids = torch.tensor([self.prompt(self.opening(expression))])
                logits = self.model(input_ids=ids).logits[0, -1].float()
                values.append(float(torch.softmax(logits, -1)[self.tool_start]))
        finally:
            self.model.train()
        return sum(values) / len(values)


@torch.no_grad()
def correct_call_rate(policy: _Policy) -> float:
    """Mean probability that the first reply is exactly the right calculator
    call (EOS included) for each training question: the rate at which
    sampling calls the tool correctly."""
    policy.model.eval()
    try:
        values = []
        for expression, _ in TRAIN:
            prompt = policy.prompt(policy.opening(expression))
            reply = policy.ids(_call(expression)) + [policy.eos]
            ids = torch.tensor([prompt + reply])
            logits = policy.model(input_ids=ids).logits[0, len(prompt) - 1 : -1]
            picked = torch.log_softmax(logits.float(), -1).gather(
                -1, torch.tensor(reply).unsqueeze(-1)
            )
            values.append(float(picked.sum().exp()))
    finally:
        policy.model.train()
    return sum(values) / len(values)


def _demonstration(
    policy: _Policy, expression: str, value: int, rng: random.Random
) -> list[tuple[list[int], list[int]]]:
    opening = policy.opening(expression)
    if rng.random() >= WARM_TOOL_SHARE:
        # A direct guess, stated (not boxed) so its tokens do not share the
        # tool path's boxed answer; it is never right.
        guess = f"The answer is {rng.randint(10, 99)}."
        return [(policy.prompt(opening), policy.ids(guess) + [policy.eos])]
    observation = f'{{"name": "calculator", "result": {value}}}'
    after = [
        *opening,
        {"role": "assistant", "content": _call(expression)},
        {"role": "tool", "content": observation},
    ]
    return [
        (policy.prompt(opening), policy.ids(_call(expression)) + [policy.eos]),
        (policy.prompt(after), policy.ids(f"\\boxed{{{value}}}") + [policy.eos]),
    ]


def _warm_start(policy: _Policy, seed: int) -> None:
    """Teacher-forced cross-entropy on the assistant tokens of demonstrations."""
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(WARM_START_STEPS):
        rows = [
            row
            for _ in range(8)
            for row in _demonstration(policy, *rng.choice(TRAIN), rng)
        ]
        width = max(len(p) + len(r) for p, r in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        labels = torch.full((len(rows), width), -100)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, (prompt, reply) in enumerate(rows):
            sequence = prompt + reply
            ids[i, : len(sequence)] = torch.tensor(sequence)
            mask[i, : len(sequence)] = 1
            labels[i, len(prompt) : len(sequence)] = torch.tensor(reply)
        loss = policy.model(input_ids=ids, attention_mask=mask, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class _Signed(RewardFunction):
    """``sign`` times the math verifier (a sign-flipped negative control)."""

    def __init__(self, sign: float) -> None:
        super().__init__(1.0, RewardType.SPARSE, "signed_math_verifier")
        self.sign = float(sign)
        self.verifier = MathAnswerVerifier()

    async def compute_reward(self, turns: Any, context: Any = None) -> RewardResult:
        result = await self.verifier.compute_reward(turns, context)
        return RewardResult(score=self.sign * result.score)


SCENARIOS = [
    {"id": expression, "question": _question(expression), "ground_truth": str(value)}
    for expression, value in TRAIN
]


def _trainer(policy: _Policy, reward: RewardFunction, seed: int) -> Any:
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    environment = ToolUseEnvironment(
        SCENARIOS,
        [calculator_tool()],
        max_tool_calls=1,
        system_prompt=SYSTEM,
        json_actions=False,
    )
    config = TrainingConfig(
        num_episodes=RL_STEPS,
        num_generations=8,
        learning_rate=RL_LEARNING_RATE,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        max_prompt_length=256,
        max_completion_length=128,
        beta=RL_BETA,
        use_reference_model=RL_BETA > 0,
        entropy_coef=0.0,
        zero_signal_patience=0,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=seed,
        lr_scheduler_type="constant",
    )
    return MultiTurnGRPOTrainer(
        agent=policy.agent, environment=environment, reward_fn=reward, config=config
    )


async def _sampled_rates(policy: _Policy, seed: int) -> tuple[float, float]:
    """(correct-call rate, verifier pass rate) over 64 sampled episodes."""
    trainer = _trainer(policy, MathAnswerVerifier(), seed)
    torch.manual_seed(seed + 1000)
    groups = await trainer.generate_trajectories(SCENARIOS, num_generations=16)
    episodes = [(g, t) for g in groups for t in g.trajectories]
    correct_calls = sum(
        any(
            entry["ok"] and str(entry["result"]) == g.scenario_metadata["ground_truth"]
            for entry in t.final_state["tool_log"]
        )
        for g, t in episodes
    )
    passed = sum(t.total_reward > 0 for _, t in episodes)
    return correct_calls / len(episodes), passed / len(episodes)


def _warm_policy(seed: int) -> _Policy:
    """A warm-started policy; the warm start runs once per seed and process."""
    policy = _Policy(seed)
    if seed not in _WARM_STATES:
        _warm_start(policy, seed)
        _WARM_STATES[seed] = {
            k: v.detach().clone() for k, v in policy.model.state_dict().items()
        }
    policy.model.load_state_dict(_WARM_STATES[seed])
    return policy


def _train(seed: int, sign: float) -> dict[str, Any]:
    policy = _warm_policy(seed)
    trainer = _trainer(policy, _Signed(sign), seed)
    result: dict[str, Any] = {"metrics": []}

    async def run() -> None:
        await trainer.initialize()
        result["held_out_before"] = policy.tool_rate(HELD_OUT)
        result["calls_before"] = correct_call_rate(policy)
        result["sampled_before"] = await _sampled_rates(policy, seed)
        torch.manual_seed(seed)
        for step in range(RL_STEPS):
            batch = [SCENARIOS[(2 * step) % 4], SCENARIOS[(2 * step + 1) % 4]]
            groups = await trainer.generate_trajectories(batch, num_generations=8)
            result["metrics"].append(await trainer.training_step(groups))
        result["held_out_after"] = policy.tool_rate(HELD_OUT)
        result["calls_after"] = correct_call_rate(policy)
        result["sampled_after"] = await _sampled_rates(policy, seed)

    asyncio.run(run())
    return result


@pytest.mark.parametrize("seed", seeds())
def test_grpo_with_a_verifier_raises_correct_tool_use(seed: int) -> None:
    run = _train(seed, sign=1.0)
    assert all(m["path"] == "token" for m in run["metrics"])
    before, after = run["held_out_before"], run["held_out_after"]
    assert before < 0.65, before  # the warm start leaves the decision open
    # The decision to call the tool generalizes to unseen questions...
    assert after > before + 0.25, (before, after)
    # ...and the policy calls it correctly more often (exact call, teacher
    # forced; deterministic), which the sampled episodes confirm.
    assert run["calls_after"] > run["calls_before"] + 0.2, run
    (calls_before, pass_before), (calls_after, pass_after) = (
        run["sampled_before"],
        run["sampled_after"],
    )
    assert calls_after > calls_before, (run["sampled_before"], run["sampled_after"])
    assert pass_after > pass_before, (run["sampled_before"], run["sampled_after"])


@pytest.mark.parametrize("seed", seeds())
def test_sign_flipped_verifier_reward_suppresses_tool_use(seed: int) -> None:
    run = _train(seed, sign=-1.0)
    before, after = run["held_out_before"], run["held_out_after"]
    # The KL anchor slows the move away from the tool; correct calls collapse.
    assert after < before - 0.05, (before, after)
    assert run["calls_after"] < 0.5 * run["calls_before"], run

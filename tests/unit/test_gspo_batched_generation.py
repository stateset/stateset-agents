"""GSPO samples every group of a step in one batched ``generate`` call.

Generating each prompt's group separately cost one decode loop per prompt;
HF decode of a small policy is per-token overhead bound, so a step with four
prompts took ~4x TRL's single batched call on identical work (v5 shootout).
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from tests.learning._toy import (  # noqa: E402
    DigitReward,
    make_agent,
    make_model,
    make_tokenizer,
)

PROMPTS = ["count 1 2 3", "x", "say the numbers seven eight nine please"]


def _config(**over):
    from stateset_agents.training.gspo_config import GSPOConfig

    base = {
        "model_name": "gpt2",
        "num_generations": 4,
        "num_outer_iterations": 1,
        "num_iterations": 1,
        "max_prompt_length": 32,
        "max_completion_length": 6,
        "learning_rate": 1e-3,
        "beta": 0.0,
        "seed": 0,
    }
    base.update(over)
    return GSPOConfig(**base)


def _trainer(token_level=False, *, greedy=False):
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(0)
    agent = make_agent(model, tokenizer)
    if greedy:
        agent.config.do_sample = False
        agent.generation_config = agent._build_generation_config()
    cls = GSPOTokenTrainer if token_level else GSPOTrainer
    trainer = cls(
        config=_config(),
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        environment=None,
        reward_model=DigitReward(1.0),
        ref_model=None,
    )
    return trainer, model


def _count_generate(model):
    calls: list[int] = []
    original = model.generate

    def counting(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append(int(out.sequences.shape[0]))
        return out

    model.generate = counting
    return calls


@pytest.mark.parametrize("token_level", [False, True], ids=["gspo", "gspo_token"])
def test_one_generate_call_per_step_covers_every_group(token_level):
    trainer, model = _trainer(token_level)
    calls = _count_generate(model)
    step = trainer.train_step_token_level if token_level else trainer.train_step
    asyncio.run(step(PROMPTS, num_groups=3))
    assert calls == [3 * 4]


def test_batched_groups_match_per_prompt_groups_under_greedy_decoding():
    trainer, _ = _trainer(greedy=True)
    generator = trainer.generator

    async def run():
        batched = await generator.generate_groups_batch(PROMPTS, 2)
        single = [await generator.generate_group_responses(p, 2) for p in PROMPTS]
        return batched, single

    batched, single = asyncio.run(run())
    assert [[text for text, _ in g] for g in batched] == [
        [text for text, _ in g] for g in single
    ]
    for got, want in zip(batched, single, strict=True):
        for (_, lp_got), (_, lp_want) in zip(got, want, strict=True):
            assert lp_got == pytest.approx(lp_want, abs=1e-4)


def test_an_overridden_per_group_generator_is_still_used():
    trainer, model = _trainer()
    seen: list[str] = []

    async def fake_group(prompt, n):
        seen.append(prompt)
        return [("1 2 3", -1.0)] * n

    trainer.generator.generate_group_responses = fake_group
    calls = _count_generate(model)
    asyncio.run(trainer.generator.generate_groups_batch(PROMPTS, 4))
    assert seen == PROMPTS and calls == []

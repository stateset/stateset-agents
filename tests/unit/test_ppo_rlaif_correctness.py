"""PPO and RLAIF train on the policy's own completion tokens, and only on them.

- PPO tokenizes prompt and completion separately, so a prompt longer than
  ``max_prompt_length`` is truncated from the left instead of leaking its tail
  into the response mask (and receiving the episode reward).
- ``PPOTrainer.train`` samples completions from the policy and runs shuffled
  mini-batches over ``mini_batch_size`` inside each PPO epoch.
- RLAIF masks the first completion token (shifted frame) and never padding,
  computes advantages from sequence rewards (identical rewards -> no update),
  and samples in eval mode.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stateset_agents.experimental.ppo import PPOConfig, PPOTrainer  # noqa: E402
from stateset_agents.experimental.rlaif import (  # noqa: E402
    ConstitutionalAI,
    RLAIFConfig,
    RLAIFTrainer,
)


def _tokenizer():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer is not cached locally")
    tok.pad_token = tok.eos_token
    return tok


def _model(seed: int = 0):
    torch.manual_seed(seed)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _ppo(reward_fn=lambda p, c: 1.0, **overrides):
    cfg = {
        "max_prompt_length": 4,
        "max_completion_length": 4,
        "num_ppo_epochs": 1,
        "learning_rate": 1e-3,
        "use_cpu": True,
    }
    cfg.update(overrides)
    return PPOTrainer(
        config=PPOConfig(model_name="gpt2", **cfg),
        model=_model(),
        tokenizer=_tokenizer(),
        reward_fn=reward_fn,
    )


def _capture_actions(trainer):
    """Record (label tokens under the action mask, rewards) per train_step."""
    seen: list[tuple[list[list[int]], list[float]]] = []
    original = trainer._update_token_batch

    def update(input_ids, attention_mask, response_mask, rewards_tensor, **kwargs):
        labels, mask = input_ids[:, 1:], response_mask[:, 1:]
        rows = [labels[i][mask[i] > 0].tolist() for i in range(labels.shape[0])]
        seen.append((rows, rewards_tensor.tolist()))
        return original(
            input_ids, attention_mask, response_mask, rewards_tensor, **kwargs
        )

    trainer._update_token_batch = update
    return seen


# --- PPO -----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompts,completions,message",
    [
        (["one two three four five six seven eight"], [" ANSWER"], "prompts exceed"),
        (["Hi"], [" a b c d e f g h"], "completions exceed"),
        (["Hi", "Hello"], ["", " yes"], "response action"),
    ],
)
async def test_ppo_invalid_rewarded_actions_fail_before_any_update(
    prompts, completions, message
):
    trainer = _ppo()
    seen = _capture_actions(trainer)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    with pytest.raises(ValueError, match=message):
        await trainer.train_step(prompts, completions, [1.0] * len(prompts))
    assert not seen and trainer.global_step == 0
    assert all(
        torch.equal(p, old)
        for p, old in zip(trainer.model.parameters(), before, strict=True)
    )


@pytest.mark.asyncio
async def test_ppo_epochs_run_shuffled_mini_batches():
    trainer = _ppo(num_ppo_epochs=2, mini_batch_size=2)
    steps = []
    orig = trainer.policy_optimizer.step

    def spy(*a, **kw):
        steps.append(1)
        return orig(*a, **kw)

    trainer.policy_optimizer.step = spy
    await trainer.train_step(
        ["a", "b", "c", "d"], [" w", " x", " y", " z"], [1, 0, 1, 0]
    )
    assert len(steps) == 4  # 2 epochs x (4 rows / mini-batches of 2)


@pytest.mark.asyncio
async def test_ppo_train_samples_completions_from_the_policy():
    seen: list[tuple[str, str]] = []

    def reward_fn(prompt, completion):
        seen.append((prompt, completion))
        return float(len(completion))

    trainer = _ppo(
        reward_fn=reward_fn,
        num_generations=2,
        per_device_train_batch_size=1,
        mini_batch_size=1,
        max_prompt_length=16,
        temperature=1.0,
        top_p=1.0,
    )
    trainer.model.train()
    await trainer.train(["Alpha", "Beta", "Gamma"], num_episodes=3)
    assert [p for p, _ in seen] == ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Gamma"]
    assert all(c != "This is a sample completion." for _, c in seen)
    assert trainer.model.training  # generation restored the training mode


@pytest.mark.asyncio
async def test_ppo_trains_on_the_sampled_token_ids(tmp_path):
    import json
    from pathlib import Path

    trainer = _ppo(
        num_generations=2, mini_batch_size=1, max_prompt_length=16, top_p=1.0, top_k=0
    )
    trainer.config.output_dir = str(tmp_path)
    seen = _capture_actions(trainer)
    result = await trainer.train(["Alpha"], num_episodes=1)
    record = json.loads(
        (Path(result["rollout_directory"]) / "rollout-0.json").read_text()
    )
    assert seen[0][0] == [row["action_ids"] for row in record["rows"]]
    assert all(
        len(row["action_ids"]) == len(row["behavior_log_probs"])
        for row in record["rows"]
    )


# --- RLAIF -----------------------------------------------------------------------


def _rlaif():
    trainer = RLAIFTrainer(
        RLAIFConfig(
            use_constitutional=False,
            use_cpu=True,
            max_prompt_length=8,
            max_completion_length=8,
        ),
        model=_model(),
        tokenizer=_tokenizer(),
    )
    trainer.judge = None
    return trainer


def test_rlaif_mask_covers_each_completion_token_and_no_padding():
    trainer = _rlaif()
    tok = trainer.tokenizer
    batch = trainer._training_batch(
        [
            {"prompt": "Hi there", "completion": " yes", "reward": 1.0},
            {"prompt": "Hi there", "completion": " yes it is long", "reward": 0.0},
        ]
    )
    labels = batch["input_ids"][:, 1:]
    mask = batch["completion_mask"]
    for row, completion in enumerate([" yes", " yes it is long"]):
        assert (
            labels[row][mask[row] > 0].tolist()
            == tok(completion, add_special_tokens=False)["input_ids"]
        )


@pytest.mark.asyncio
async def test_rlaif_identical_rewards_give_no_policy_update():
    trainer = _rlaif()
    trajectories = [
        {"prompt": "Hi there", "completion": c, "reward": 1.0}
        for c in (" yes", " yes it is long", " no")
    ]
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = await trainer.train_step(trajectories)
    assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-7)
    after = list(trainer.model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_rlaif_advantages_are_group_relative_per_prompt():
    trainer = _rlaif()
    batch = trainer._training_batch(
        [
            {"prompt": "A", "completion": " x", "reward": 1.0},
            {"prompt": "A", "completion": " y", "reward": 0.0},
            {"prompt": "B", "completion": " x", "reward": 5.0},
            {"prompt": "B", "completion": " y", "reward": 5.0},
        ]
    )
    adv = batch["advantages"].tolist()
    assert adv[0] > 0 > adv[1]
    assert adv[0] == pytest.approx(-adv[1])
    assert adv[2] == adv[3] == 0.0


def test_rlaif_generation_runs_in_eval_mode_and_restores_training():
    trainer = _rlaif()
    model = trainer.model
    modes = []
    orig = model.generate

    def generate(*a, **kw):
        modes.append(model.training)
        return orig(*a, **kw)

    model.generate = generate
    model.train()
    trainer.generate_completion("Hi", max_new_tokens=2)
    cai = ConstitutionalAI(model=model, tokenizer=trainer.tokenizer)
    cai._generate("Hi", max_new_tokens=2)
    assert modes == [False, False]
    assert model.training


def test_rlaif_train_cycles_through_every_prompt():
    trainer = _rlaif()
    trainer.config.per_device_train_batch_size = 2
    trainer.config.logging_steps = 1000
    trainer.config.use_self_play = False
    seen = []

    async def collect(prompts, num_generations):
        seen.append(list(prompts))
        return []

    async def step(trajectories):
        return {"policy_loss": 0.0, "average_reward": 0.0}

    trainer.collect_trajectories = collect
    trainer.train_step = step
    asyncio.run(trainer.train(["a", "b", "c"], num_episodes=3))
    assert seen == [["a", "b"], ["c", "a"], ["b", "c"]]

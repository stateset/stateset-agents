"""OnlineDPOTrainer: pairing, the reward-failure contract, references, resume."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.reward_base import RewardResult  # noqa: E402
from stateset_agents.core.reward_context import RewardUnavailableError  # noqa: E402
from stateset_agents.training.preference import (  # noqa: E402
    OnlineDPOConfig,
    OnlineDPOTrainer,
    select_pair,
    train_online_dpo,
)
from tests.learning._toy import DigitReward, make_model, make_tokenizer  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return make_tokenizer()


PROMPTS = [[{"role": "user", "content": p}] for p in ("hello", "count", "abc def")]
HELD = ["xyz", "ok go"]


def _config(**overrides: Any) -> OnlineDPOConfig:
    values: dict[str, Any] = {
        "rounds": 2,
        "samples_per_prompt": 4,
        "max_new_tokens": 4,
        "learning_rate": 1e-2,
        "batch_size": 2,
        "num_epochs": 1,
        "device": "cpu",
        "max_length": 48,
        "max_prompt_length": None,
        "lr_scheduler": "constant",
        "generation_batch_size": 8,
    }
    values.update(overrides)
    return OnlineDPOConfig(**values)


class LengthReward:
    """Longer replies score higher (deterministic, never ties on length)."""

    weight = 1.0
    name = "length"

    async def compute_reward(self, turns: Any, context: Any = None) -> RewardResult:
        return RewardResult(score=float(len(turns[-1].content)))


def _trainer(tokenizer: Any, reward: Any = None, **overrides: Any) -> OnlineDPOTrainer:
    return OnlineDPOTrainer(
        make_model(0, n_embd=16),
        tokenizer,
        reward if reward is not None else DigitReward(),
        PROMPTS,
        _config(**overrides),
        eval_prompts=HELD,
    )


def _params(model: Any) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _max_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------- pairing


@pytest.mark.parametrize(
    ("scores", "margin", "expected"),
    [
        ([0.1, 0.9, 0.5], 0.0, (1, 0)),
        ([0.5, 0.9, 0.9, 0.1, 0.1], 0.0, (1, 3)),  # earliest among ties
        ([0.3, None, 0.7], 0.0, (2, 0)),
        ([0.4, 0.4, 0.4], 0.0, "tie"),
        ([0.4, None, None], 0.0, "too_few_scores"),
        ([0.4, 0.5], 0.2, "below_margin"),
        ([0.4, 0.7], 0.2, (1, 0)),
    ],
)
def test_select_pair(scores: list, margin: float, expected: Any) -> None:
    assert select_pair(scores, min_margin=margin) == expected


def test_select_pair_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        select_pair([0.1, math.nan])


def test_round_pairs_best_against_worst_on_exact_sampled_tokens(
    tokenizer: Any,
) -> None:
    trainer = _trainer(tokenizer, LengthReward(), rounds=1)
    summary = trainer.run_round()
    assert summary.samples == 12 and summary.scored == 12 and summary.unavailable == 0
    assert summary.pairs + sum(summary.skipped.values()) == len(PROMPTS)
    assert summary.training is not None and summary.eval_metrics is not None
    for pair in trainer.last_pairs.pairs:
        assert pair.metadata["chosen_score"] > pair.metadata["rejected_score"]
        assert len(pair.chosen) > len(pair.rejected)  # what the reward prefers


class ScriptedReward:
    """Scores by reply text from a script; raises what it is told to."""

    weight = 1.0
    name = "scripted"

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls = 0

    async def compute_reward(self, turns: Any, context: Any = None) -> Any:
        self.calls += 1
        value = self.outcome(self.calls)
        if isinstance(value, BaseException):
            raise value
        return RewardResult(score=value)


def test_unavailable_rewards_are_dropped_not_scored(tokenizer: Any) -> None:
    reward = ScriptedReward(
        lambda call: RewardUnavailableError("judge down") if call % 2 else float(call)
    )
    summary = _trainer(tokenizer, reward, rounds=1).run_round()
    assert summary.unavailable == 6 and summary.scored == 6


def test_every_reward_unavailable_raises(tokenizer: Any) -> None:
    reward = ScriptedReward(lambda call: RewardUnavailableError("judge down"))
    with pytest.raises(RewardUnavailableError, match="every one"):
        _trainer(tokenizer, reward).run_round()


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (lambda call: KeyError("bug"), KeyError),
        (lambda call: math.nan, ValueError),
        (lambda call: None, TypeError),
    ],
)
def test_other_reward_failures_propagate(
    tokenizer: Any, outcome: Any, error: type[BaseException]
) -> None:
    with pytest.raises(error):
        _trainer(tokenizer, ScriptedReward(outcome)).run_round()


def test_fail_on_error_makes_unavailability_fatal(tokenizer: Any) -> None:
    reward = ScriptedReward(lambda call: RewardUnavailableError("judge down"))
    reward.fail_on_error = True  # type: ignore[attr-defined]
    with pytest.raises(RewardUnavailableError, match="judge down"):
        _trainer(tokenizer, reward).run_round()


def test_a_round_without_pairs_leaves_the_policy_unchanged(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer, ScriptedReward(lambda call: 0.5))
    before = _params(trainer.model)
    summary = trainer.run_round()
    assert summary.pairs == 0 and summary.skipped == {"tie": 3}
    assert summary.training is None and trainer.round == 1
    assert _max_diff(before, _params(trainer.model)) == 0.0


# ------------------------------------------------------------- references


def test_fixed_reference_stays_the_initial_policy(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer, LengthReward(), reference_update="fixed")
    initial = _params(trainer.model)
    trainer.run_round()
    reference = trainer.reference_for_round()
    assert _max_diff(initial, _params(reference)) == 0.0
    assert _max_diff(initial, _params(trainer.model)) > 0.0
    assert all(not p.requires_grad for p in reference.parameters())


def test_previous_round_reference_follows_the_policy(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer, LengthReward(), reference_update="previous_round")
    initial = _params(trainer.model)
    assert _max_diff(initial, _params(trainer.reference_for_round())) == 0.0
    trainer.run_round()
    reference = trainer.reference_for_round()
    assert _max_diff(_params(trainer.model), _params(reference)) == 0.0
    assert reference is not trainer.model and not reference.training


def test_lora_fixed_reference_is_the_adapter_disabled_policy(tokenizer: Any) -> None:
    pytest.importorskip("peft")
    trainer = _trainer(
        tokenizer, LengthReward(), use_lora=True, lora_target_modules=["c_attn"]
    )
    assert trainer.reference_for_round() is None
    assert hasattr(trainer.model, "peft_config")
    assert trainer.run_round().training is not None


# ------------------------------------------------------ resume, API, misc


def test_round_checkpoints_resume_exactly(tokenizer: Any, tmp_path: Path) -> None:
    straight = _trainer(tokenizer, LengthReward(), rounds=2)
    straight.train()
    interrupted = _trainer(tokenizer, LengthReward(), rounds=1)
    interrupted.train()
    checkpoint = interrupted.save_checkpoint(tmp_path / "round-0")
    assert (checkpoint / "pairs.jsonl").exists()
    resumed = _trainer(tokenizer, LengthReward(), rounds=2)
    history = resumed.train(resume_from=checkpoint)
    assert [entry.round for entry in history] == [0, 1]
    assert _max_diff(_params(straight.model), _params(resumed.model)) == 0.0


def test_train_online_dpo_api_checkpoints_every_round(
    tokenizer: Any, tmp_path: Path
) -> None:
    result = train_online_dpo(
        make_model(0, n_embd=16),
        LengthReward(),
        [{"prompt": "hello", "scenario": {"target": 1}}, "count"],
        tokenizer=tokenizer,
        config=_config(),
        output_dir=str(tmp_path / "out"),
    )
    assert len(result.rounds) == 2
    assert (tmp_path / "out" / "round-0" / "online_dpo_checkpoint.json").exists()
    assert result.checkpoint == tmp_path / "out" / "final"


def test_runs_inside_a_running_event_loop(tokenizer: Any) -> None:
    async def main() -> Any:
        return _trainer(tokenizer, LengthReward(), rounds=1).train()

    assert len(asyncio.run(main())) == 1


def test_validation(tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="overlap"):
        OnlineDPOTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            DigitReward(),
            PROMPTS,
            _config(),
            eval_prompts=["HELLO"],
        )
    with pytest.raises(ValueError, match="RewardFunction"):
        OnlineDPOTrainer(make_model(0, n_embd=16), tokenizer, None, PROMPTS)
    with pytest.raises(ValueError, match="prompts"):
        OnlineDPOTrainer(make_model(0, n_embd=16), tokenizer, DigitReward(), "hello")
    for overrides, message in [
        ({"samples_per_prompt": 1}, "samples_per_prompt"),
        ({"method": "kto"}, "method"),
        ({"validation_fraction": 0.1}, "eval_prompts"),
        ({"reference_update": "ema"}, "reference_update"),
        ({"min_score_margin": -1.0}, "min_score_margin"),
    ]:
        with pytest.raises(ValueError, match=message):
            _config(**overrides).validate()

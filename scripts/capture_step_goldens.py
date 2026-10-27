"""Capture one-update-step goldens for every native RL trainer.

The unified RL spine (docs/superpowers/plans/2026-10-22-unified-rl-spine.md)
moves each trainer's optimizer stepping, update loop and outer loop onto shared
code. These goldens pin what ONE training step does today, end to end through
the trainer's own ``train_step`` (``training_step`` for MultiTurn, ``train()``
over a single one-step episode for SingleTurn), so every slice can prove it
changed nothing numerically:

* GSPO, GSPO-token, DAPO, GEPO, VAPO (post value-warmup), MultiTurnGRPO and
  SingleTurnGRPO, each at K=1 and K=2 update passes over one rollout batch
  (``num_iterations`` for GSPO, ``num_gradient_updates`` elsewhere);
* closed-form deterministic weights (``deterministic_weights``) and scripted
  rollouts -- nothing is sampled, so there is no generation in the step;
* recorded: the step's metrics, the parameter-delta norm and a sign-sensitive
  projection of the delta, the gradient norm each optimizer saw at every
  ``optimizer.step()`` (post-clip), learning rates, scheduler positions and
  step counters. CPU, fp32, one thread.

Run BEFORE a refactor and commit the JSON; ``tests/unit/test_trainer_step_
goldens.py`` replays every case and compares.

    python scripts/capture_step_goldens.py [--only gspo,dapo_k2,...]

The JSON also records the capture environment (``_environment``: python
major.minor, torch, transformers and numpy versions, platform system and
machine). The test compares strictly (rtol 1e-5) only in that environment --
the refactor-equivalence guard each spine slice runs locally. Elsewhere (CI's
other torch versions, CPUs, BLAS builds and operating systems) it uses
portable tolerances that survive cross-environment float drift but still
catch real breakage; see the test's docstring. ``--only`` refuses to mix
environments: recapture everything when the environment changed.

Regenerate deliberately and explain why in CHANGELOG.md; never regenerate to
turn a red golden green without understanding why a number moved.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import platform
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "tests" / "unit" / "goldens" / "step_goldens.json"
#: Key of the capture-environment fingerprint inside the goldens JSON.
ENVIRONMENT_KEY = "_environment"

VOCAB = 128
PROMPTS = ["hello", "count"]
# Scripted completions per prompt: text (scored by the reward) and the token
# ids the policy is trained on. Rewards differ inside every group so no group
# is filtered by DAPO's dynamic sampling and every advantage is nonzero.
COMPLETIONS = [
    ("12a", [17, 18, 65]),
    ("abc", [65, 66, 67]),
    ("9z", [25, 90]),
]
PROMPT_IDS = {"hello": [72, 69, 76, 76, 79], "count": [67, 79, 85, 78, 84]}
LEARNING_RATE = 5e-3
#: AdamW epsilon for every observed optimizer (see ``StepObserver``).
ADAM_EPS = 1e-4


# --------------------------------------------------------------------------
# deterministic fixtures
# --------------------------------------------------------------------------


def environment_fingerprint() -> dict[str, str]:
    """What the golden numbers depend on beyond the repository's code."""
    import numpy
    import transformers

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "numpy": str(numpy.__version__),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def tiny_model() -> Any:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_embd=32,
            n_layer=2,
            n_head=2,
            vocab_size=VOCAB,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    return deterministic_weights(model)


def deterministic_weights(module: Any, seed: int = 0) -> Any:
    """Overwrite every parameter with a version-independent pattern.

    Same scheme as ``scripts/capture_objective_goldens.py``: module init is
    not stable across torch/transformers versions or platforms, so weights
    come from a closed-form cosine wave (LayerNorm scales near 1).
    """
    with torch.no_grad():
        for i, (name, p) in enumerate(module.named_parameters()):
            idx = torch.arange(p.numel(), dtype=torch.float64) + seed * 1000 + i
            wave = torch.cos(idx * 0.37 + i * 0.11)
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
                vals = 1.0 + 0.01 * wave
            else:
                vals = 0.02 * wave
            p.copy_(vals.to(p.dtype).view_as(p))
    return module


def tokenizer() -> Any:
    from tests._tiny_tokenizer import tiny_tokenizer

    return tiny_tokenizer(VOCAB)


def digit_fraction(text: str) -> float:
    text = str(text)
    return sum(ch.isdigit() for ch in text) / len(text) if text else 0.0


def _response_dicts(prompt: str) -> list[dict[str, Any]]:
    """DAPO/VAPO rollout dicts: prompt ids then completion ids, right-padded."""
    prompt_ids = PROMPT_IDS[prompt]
    out = []
    for text, completion in COMPLETIONS:
        ids = torch.tensor(prompt_ids + completion, dtype=torch.long)
        response_mask = torch.zeros(len(ids))
        response_mask[len(prompt_ids) :] = 1.0
        out.append(
            {
                "response": text,
                "input_ids": ids,
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "response_mask": response_mask,
                "sequence_length": len(completion),
                "prompt_length": len(prompt_ids),
            }
        )
    return out


def _gepo_response_dicts(prompt: str) -> list[dict[str, Any]]:
    """GEPO rollout dicts: one shared response start, right-padded with 0."""
    prompt_ids = PROMPT_IDS[prompt]
    width = len(prompt_ids) + max(len(c) for _, c in COMPLETIONS)
    out = []
    for text, completion in COMPLETIONS:
        ids = prompt_ids + completion
        pad = width - len(ids)
        out.append(
            {
                "response": text,
                "input_ids": torch.tensor(ids + [0] * pad, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [1] * len(ids) + [0] * pad, dtype=torch.long
                ),
                "response_start_idx": len(prompt_ids),
            }
        )
    return out


def _trajectory_group(prompt: str) -> Any:
    from stateset_agents.core.trajectory import (
        ConversationTurn,
        MultiTurnTrajectory,
        TrajectoryGroup,
    )

    trajectories = []
    for text, completion in COMPLETIONS:
        trajectories.append(
            MultiTurnTrajectory(
                turns=[
                    ConversationTurn(role="user", content=prompt),
                    ConversationTurn(
                        role="assistant",
                        content=text,
                        metadata={
                            "prompt_token_ids": list(PROMPT_IDS[prompt]),
                            "token_ids": list(completion),
                        },
                    ),
                ],
                total_reward=digit_fraction(text),
            )
        )
    return TrajectoryGroup(scenario_id=prompt, trajectories=trajectories)


class _Reward:
    """Prompt/response reward model for GSPO (sequence and token variants)."""

    async def compute_reward(
        self, turns: Any = None, context: Any = None, **_: Any
    ) -> Any:
        from stateset_agents.core.reward_base import RewardResult

        turn = list(turns or [])[-1]
        content = turn.get("content") if isinstance(turn, dict) else turn.content
        return RewardResult(score=digit_fraction(content))

    async def compute_turn_reward(
        self, turn: Any = None, context: Any = None, **_: Any
    ) -> Any:
        return await self.compute_reward([turn], context)


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------


def _probe(numel: int, index: int) -> torch.Tensor:
    idx = torch.arange(numel, dtype=torch.float64)
    return torch.sin(idx * 0.13 + index * 0.7)


def _norm(gradients: Any) -> float:
    squares = [
        float(g.detach().to(torch.float64).pow(2).sum())
        for g in gradients
        if g is not None
    ]
    return math.sqrt(math.fsum(squares))


class StepObserver:
    """Snapshot parameters; record pre-clip and per-step (post-clip) norms.

    Pre-clip norms come from a pass-through spy on
    ``torch.nn.utils.clip_grad_norm_`` (every trainer clips through that
    attribute), so a uniformly mis-scaled gradient is caught even though
    clipping and AdamW's scale invariance would hide it from the update.

    Every observed optimizer runs with ``eps=ADAM_EPS``. With the default
    1e-8, AdamW's first update ``lr * g / (|g| + eps)`` turns gradients that
    are zero in exact arithmetic (e.g. attention key biases, which softmax
    ignores) and so are pure rounding noise (~1e-10) into updates of a sizable
    fraction of ``lr`` whose size and sign depend on the noise. Any ulp-level
    change (a different CPU, an equivalent log-prob formula) then moved the
    parameter delta far beyond the tolerance. ``ADAM_EPS`` keeps the update a
    smooth function of the gradient while real gradients still dominate it.
    """

    def __init__(
        self,
        modules: dict[str, Any],
        optimizers: dict[str, Any],
        schedulers: dict[str, Any],
    ) -> None:
        self.modules = modules
        self.optimizers = optimizers
        self.schedulers = schedulers
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                group["eps"] = ADAM_EPS
        self.grad_norms: dict[str, list[float]] = {name: [] for name in optimizers}
        self.clip_norms: list[float] = []
        self.before = self._parameters()
        self._handles = [
            optimizer.register_step_pre_hook(self._recorder(name))
            for name, optimizer in optimizers.items()
        ]
        self._clip = torch.nn.utils.clip_grad_norm_

        def clip_spy(parameters: Any, *args: Any, **kwargs: Any) -> Any:
            if torch.is_tensor(parameters):
                parameters = [parameters]
            parameters = list(parameters)
            self.clip_norms.append(_norm(p.grad for p in parameters))
            return self._clip(parameters, *args, **kwargs)

        torch.nn.utils.clip_grad_norm_ = clip_spy

    def _parameters(self) -> list[tuple[str, torch.Tensor]]:
        return [
            (f"{prefix}.{name}", p.detach().to(torch.float64).clone())
            for prefix, module in self.modules.items()
            for name, p in module.named_parameters()
        ]

    def _recorder(self, name: str) -> Callable[..., None]:
        def record(optimizer: Any, args: Any, kwargs: Any) -> None:
            self.grad_norms[name].append(
                _norm(
                    p.grad for group in optimizer.param_groups for p in group["params"]
                )
            )

        return record

    def close(self) -> None:
        torch.nn.utils.clip_grad_norm_ = self._clip
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def finish(self, metrics: dict[str, Any]) -> dict[str, Any]:
        self.close()
        after = self._parameters()
        squares, projections = [], []
        for index, ((name, old), (new_name, new)) in enumerate(
            zip(self.before, after, strict=True)
        ):
            assert name == new_name, (name, new_name)
            delta = (new - old).reshape(-1)
            squares.append(float(delta.pow(2).sum()))
            projections.append(float((delta * _probe(delta.numel(), index)).sum()))
        return {
            "metrics": metrics,
            "param_delta_norm": math.sqrt(math.fsum(squares)),
            "param_delta_probe": math.fsum(projections),
            "optimizer_steps": {k: len(v) for k, v in self.grad_norms.items()},
            "grad_norms": self.grad_norms,
            "clip_norms": self.clip_norms,
            "lr": {
                name: float(optimizer.param_groups[0]["lr"])
                for name, optimizer in self.optimizers.items()
            },
            "scheduler_last_epoch": {
                name: (None if scheduler is None else int(scheduler.last_epoch))
                for name, scheduler in self.schedulers.items()
            },
        }


#: Wall-clock or process-dependent metric keys never belong in a golden.
_VOLATILE = ("time", "seconds", "_ms", "throughput", "per_sec", "memory")


def scalar_metrics(metrics: Any) -> dict[str, Any]:
    """Deterministic scalar metrics, as plain JSON values."""
    out: dict[str, Any] = {}
    for key, value in dict(metrics).items():
        if any(token in key for token in _VOLATILE):
            continue
        if torch.is_tensor(value) and value.numel() == 1:
            value = value.item()
        if isinstance(value, bool) or value is None or isinstance(value, str):
            out[key] = value
        elif isinstance(value, int):
            out[key] = int(value)
        elif isinstance(value, float) or hasattr(value, "__float__"):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------


def _gspo(token_level: bool, k: int) -> dict[str, Any]:
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    cls = GSPOTokenTrainer if token_level else GSPOTrainer
    trainer = cls(
        config=GSPOConfig(
            model_name="gpt2",
            num_generations=len(COMPLETIONS),
            num_outer_iterations=10,
            num_iterations=k,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.0,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=tiny_model(),
        tokenizer=tokenizer(),
        agent=None,
        environment=None,
        reward_model=_Reward(),
        ref_model=None,
    )

    async def generate(prompt: str, count: int) -> list[tuple[str, float]]:
        assert count == len(COMPLETIONS)
        return [(text, -5.0) for text, _ in COMPLETIONS]

    trainer.generator.generate_group_responses = generate  # type: ignore[method-assign]
    observer = StepObserver(
        {"model": trainer.model},
        {"policy": trainer.optimizer},
        {"policy": trainer.scheduler},
    )
    step = trainer.train_step_token_level if token_level else trainer.train_step
    metrics = asyncio.run(step(list(PROMPTS), num_groups=len(PROMPTS)))
    result = observer.finish(scalar_metrics(metrics))
    result["global_step"] = getattr(trainer, "global_step", None)
    return result


def golden_gspo(k: int) -> dict[str, Any]:
    return _gspo(False, k)


def golden_gspo_token(k: int) -> dict[str, Any]:
    return _gspo(True, k)


def golden_dapo(k: int) -> dict[str, Any]:
    from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer

    trainer = DAPOTrainer(
        config=DAPOConfig(
            model_name="gpt2",
            group_size=len(COMPLETIONS),
            num_gradient_updates=k,
            learning_rate=LEARNING_RATE,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: digit_fraction(response),
    )

    async def generate(prompt: str) -> list[dict[str, Any]]:
        return _response_dicts(prompt)

    trainer.generate_group_responses = generate  # type: ignore[method-assign]
    observer = StepObserver(
        {"model": trainer.model},
        {"policy": trainer.optimizer},
        {"policy": trainer.scheduler},
    )
    metrics = asyncio.run(trainer.train_step(list(PROMPTS)))
    result = observer.finish(scalar_metrics(metrics))
    result["global_step"] = int(trainer.global_step)
    return result


def golden_gepo(k: int) -> dict[str, Any]:
    from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer

    trainer = GEPOTrainer(
        config=GEPOConfig(
            model_name="gpt2",
            group_size=len(COMPLETIONS),
            num_gradient_updates=k,
            num_episodes=10,
            num_epochs=1,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.0,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: digit_fraction(response),
    )

    async def generate(prompt: str, group_size: int) -> list[dict[str, Any]]:
        assert group_size == len(COMPLETIONS)
        return _gepo_response_dicts(prompt)

    trainer.generate_group_responses = generate  # type: ignore[method-assign]
    observer = StepObserver(
        {"model": trainer.model},
        {"policy": trainer.optimizer},
        {"policy": trainer.scheduler},
    )
    metrics = asyncio.run(trainer.train_step(list(PROMPTS)))
    result = observer.finish(scalar_metrics(metrics))
    result["global_step"] = int(trainer.global_step)
    return result


def golden_vapo(k: int) -> dict[str, Any]:
    from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer

    trainer = VAPOTrainer(
        config=VAPOConfig(
            model_name="gpt2",
            group_size=len(COMPLETIONS),
            num_gradient_updates=k,
            per_device_train_batch_size=len(PROMPTS),
            num_episodes=10,
            num_epochs=1,
            actor_learning_rate=LEARNING_RATE,
            critic_learning_rate=LEARNING_RATE,
            warmup_ratio=0.0,
            max_prompt_length=32,
            max_completion_length=32,
            bf16=False,
            report_to="none",
        ),
        model=tiny_model(),
        tokenizer=None,
        reward_fn=lambda prompt, response: digit_fraction(response),
    )
    deterministic_weights(trainer.value_head, seed=7)
    # The value head's dropout would make the golden depend on the global RNG
    # stream (and so on how many draws model construction made in a given
    # torch/transformers version); the policy is already dropout-free.
    for module in trainer.value_head.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    trainer.value_warmup_complete = True  # post-warmup joint update

    async def generate(prompt: str) -> list[dict[str, Any]]:
        return _response_dicts(prompt)

    trainer.generate_group_responses = generate  # type: ignore[method-assign]
    observer = StepObserver(
        {"model": trainer.model, "value_head": trainer.value_head},
        {"actor": trainer.actor_optimizer, "critic": trainer.critic_optimizer},
        {"actor": trainer.actor_scheduler, "critic": trainer.critic_scheduler},
    )
    metrics = asyncio.run(trainer.train_step(list(PROMPTS)))
    result = observer.finish(scalar_metrics(metrics))
    result["global_step"] = int(trainer.global_step)
    return result


def _grpo_config(k: int, **overrides: Any) -> Any:
    from stateset_agents.training.config import TrainingConfig

    values: dict[str, Any] = {
        "num_episodes": 10,
        "num_generations": len(COMPLETIONS),
        "num_gradient_updates": k,
        "gradient_accumulation_steps": 1,
        "generation_batch_size": 2,  # 3 rows -> two backward chunks
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": 0.0,
        "max_prompt_length": 32,
        "max_completion_length": 32,
        "beta": 0.0,
        "bf16": False,
        "fp16": False,
        "report_to": "none",
        "seed": 0,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def golden_multi_turn_grpo(k: int) -> dict[str, Any]:
    from types import SimpleNamespace

    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    model = tiny_model()
    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(model=model, tokenizer=None),
        environment=None,
        reward_fn=None,
        config=_grpo_config(k),
    )
    trainer._setup_optimizer()
    trainer._setup_scheduler(10)
    observer = StepObserver(
        {"model": model},
        {"policy": trainer.optimizer},
        {"policy": trainer.lr_scheduler},
    )
    groups = [_trajectory_group(prompt) for prompt in PROMPTS]
    metrics = asyncio.run(trainer.training_step(groups))
    result = observer.finish(scalar_metrics(metrics))
    result["global_step"] = int(trainer.global_step)
    return result


class _ScriptedAgent:
    """Single-turn agent whose ``generate_turn`` replays fixed completions."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.tokenizer = None
        self.calls = 0

    async def generate_turn(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        from stateset_agents.core.trajectory import ConversationTurn

        text, completion = COMPLETIONS[self.calls % len(COMPLETIONS)]
        self.calls += 1
        return ConversationTurn(
            role="assistant",
            content=text,
            metadata={
                "prompt_token_ids": list(PROMPT_IDS[str(prompt)]),
                "token_ids": list(completion),
            },
        )


class _PromptEnvironment:
    scenarios = [{"prompt": PROMPTS[0]}]

    async def reset(self, scenario: Any = None) -> dict[str, Any]:
        return {"prompt": (scenario or self.scenarios[0])["prompt"]}

    async def step(self, state: Any, action: Any) -> dict[str, Any]:
        return {"state": state, "reward": 0.0, "done": True}


class _ScriptedReward:
    async def compute_reward(self, turns: Any, context: Any = None) -> Any:
        from stateset_agents.core.reward_base import RewardResult

        return RewardResult(score=digit_fraction(list(turns)[-1].content))


class _Diagnostics:
    """Stands in for ``DiagnosticsMonitor`` to read each step's loss terms."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if event == "step_end":
            self.rows.append(scalar_metrics(payload))


def golden_single_turn_grpo(k: int, accumulation: int = 1) -> dict[str, Any]:
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    model = tiny_model()
    trainer = SingleTurnGRPOTrainer(
        agent=_ScriptedAgent(model),
        environment=_PromptEnvironment(),
        reward_fn=_ScriptedReward(),
        config=_grpo_config(
            k,
            num_episodes=1,
            max_steps_per_episode=1,
            gradient_accumulation_steps=accumulation,
        ),
    )
    trainer._setup_optimizer()
    diagnostics = _Diagnostics()
    trainer.diagnostics = diagnostics
    optimizer = trainer.optimizer

    async def run() -> None:
        await trainer.train()

    # train() builds its scheduler; observe it through the optimizer, whose
    # LR the scheduler owns, and read the scheduler position afterwards.
    observer = StepObserver({"model": model}, {"policy": optimizer}, {})
    asyncio.run(run())
    rows = diagnostics.rows
    assert len(rows) == 1, rows
    result = observer.finish(rows[0])
    result["scheduler_last_epoch"] = {
        "policy": (
            None
            if trainer.lr_scheduler is None
            else int(trainer.lr_scheduler.last_epoch)
        )
    }
    result["global_step"] = int(trainer.global_step)
    return result


TRAINERS: dict[str, Callable[[int], dict[str, Any]]] = {
    "gspo": golden_gspo,
    "gspo_token": golden_gspo_token,
    "dapo": golden_dapo,
    "gepo": golden_gepo,
    "vapo": golden_vapo,
    "multi_turn_grpo": golden_multi_turn_grpo,
    "single_turn_grpo": golden_single_turn_grpo,
}
PASSES = (1, 2)
CASES: dict[str, tuple[str, int]] = {
    f"{name}_k{k}": (name, k) for name in TRAINERS for k in PASSES
}


def golden_single_turn_grpo_partial_window(k: int) -> dict[str, Any]:
    """One backward into a 2-step accumulation window, closed by train()'s
    final flush: exercises safe_optimizer_step's partial-window correction."""
    return golden_single_turn_grpo(k, accumulation=2)


TRAINERS["single_turn_grpo_partial_window"] = golden_single_turn_grpo_partial_window
CASES["single_turn_grpo_partial_window_k1"] = ("single_turn_grpo_partial_window", 1)


@contextlib.contextmanager
def deterministic_cpu() -> Iterator[None]:
    """One thread, deterministic kernels, fp32 default; restored afterwards."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    dtype = torch.get_default_dtype()
    clip = torch.nn.utils.clip_grad_norm_  # StepObserver spies on it
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.nn.utils.clip_grad_norm_ = clip
        torch.set_default_dtype(dtype)
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)


def capture(case: str) -> dict[str, Any]:
    name, k = CASES[case]
    with deterministic_cpu():
        return TRAINERS[name](k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated cases/trainers")
    args = parser.parse_args()
    wanted = [n for n in args.only.split(",") if n]
    names = [
        case
        for case, (trainer, _) in CASES.items()
        if not wanted or case in wanted or trainer in wanted
    ]
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    environment = environment_fingerprint()
    recorded = existing.get(ENVIRONMENT_KEY)
    if wanted and recorded not in (None, environment):
        parser.error(
            f"goldens were captured in {recorded}, this is {environment}: "
            "recapture every case (no --only) so the file has one environment"
        )
    existing[ENVIRONMENT_KEY] = environment
    for case in names:
        existing[case] = capture(case)
        print(f"captured {case}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

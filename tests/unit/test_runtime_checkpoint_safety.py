"""Checkpoint preflight must precede mutation; failed commits remain quarantined."""

import copy
import json
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.checkpoint_publication import (  # noqa: E402
    COMPLETION_FILE,
    checkpoint_write,
)
from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.runtime_state import (  # noqa: E402
    TrainingRuntime,
    capture_rng,
)


class TinyPolicy(torch.nn.Linear):
    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")


class TinyRuntime(TrainingRuntime):
    def __init__(self):
        self.config = TrainingConfig(bf16=False, report_to="none")
        self.model = TinyPolicy(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.9)
        self.global_step = 0
        self._grad_accum_step = 0
        self.training_metrics = {"loss": []}
        self.best_eval_metric = -float("inf")
        self.value_warmup_complete = False
        self._current_task_id = None

    async def train_step(self):
        self.optimizer.zero_grad()
        coefficient = getattr(
            self, "kl_controller", SimpleNamespace(kl_coef=1.0)
        ).kl_coef
        loss = self.model(torch.tensor([[0.5, -0.25]])).square().mean() * coefficient
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if getattr(self, "kl_controller", None) is not None:
            self.kl_controller.kl_coef *= 0.9
        self.global_step += 1
        self.training_metrics["loss"].append(float(loss.item()))
        return {"loss": float(loss.item())}

    async def training_step(self):
        return await self.train_step()

    async def train_step_token_level(self):
        return await self.train_step()

    async def generate_trajectories(self):
        return []

    async def generate_group_responses(self):
        return []

    async def train(self):
        return await self.train_step()

    def _apply_optimizer_step(self):
        self.optimizer.step()

    def save_checkpoint(self, path):
        self.save_runtime_checkpoint(path)

    def save_model(self, path):
        self.save_runtime_checkpoint(path)


def snapshot(trainer):
    return copy.deepcopy(
        {
            "policy": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "global_step": trainer.global_step,
            "metrics": trainer.training_metrics,
            "rng": capture_rng(torch),
            "poisoned": getattr(trainer, "_runtime_checkpoint_poisoned", False),
            "adaptive": (
                vars(trainer.kl_controller)
                if hasattr(trainer, "kl_controller")
                else None
            ),
            "reference": (
                trainer.ref_model.state_dict()
                if hasattr(trainer, "ref_model")
                else None
            ),
        }
    )


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for value, wanted in zip(actual, expected, strict=True):
            assert_same(value, wanted)
    else:
        assert actual == expected


@pytest.fixture
async def checkpoint(tmp_path):
    torch.manual_seed(3)
    trainer = TinyRuntime()
    await trainer.train_step()
    path = tmp_path / "good"
    trainer.save_runtime_checkpoint(str(path))
    return trainer, path


def republish(source, target, mutate):
    """Publish intentionally invalid learner state with a valid file inventory."""
    with checkpoint_write(target) as staging:
        for file in source.iterdir():
            if file.name != COMPLETION_FILE:
                shutil.copy2(file, staging / file.name)
        path = staging / "training_state.pt"
        state = torch.load(path, weights_only=True)
        mutate(state, staging)
        torch.save(state, path)


@pytest.mark.parametrize(
    "failure",
    [
        "slot_nan",
        "slot_inf",
        "slot_shape",
        "slot_schema",
        "counter_name",
        "counter_type",
        "metric_nan",
        "rng_cpu",
        "rng_python",
        "rng_cuda",
        "scheduler_schema",
        "scheduler_nan",
        "scheduler_counter_string",
        "scheduler_lr_string",
        "scheduler_groups",
        "missing_global_step",
        "missing_metrics",
        "missing_accumulation",
        "rollout",
        "component_schema",
    ],
)
async def test_invalid_published_state_rejects_before_learner_mutation(
    checkpoint, tmp_path, failure
):
    _, path = checkpoint

    def mutate(state, staging):
        slots = next(iter(state["components"]["optimizer"]["state"].values()))
        if failure == "slot_nan":
            slots["exp_avg"].fill_(float("nan"))
        elif failure == "slot_inf":
            slots["exp_avg_sq"].fill_(float("inf"))
        elif failure == "slot_shape":
            slots["exp_avg"] = torch.ones(17)
        elif failure == "slot_schema":
            del slots["exp_avg_sq"]
        elif failure == "counter_name":
            state["counters"]["unknown_cursor"] = 4
        elif failure == "counter_type":
            state["counters"]["global_step"] = True
        elif failure == "metric_nan":
            state["counters"]["training_metrics"]["loss"] = [float("nan")]
        elif failure == "rng_cpu":
            state["rng"]["torch"] = torch.ones(2)
        elif failure == "rng_python":
            state["rng"]["python"] = (0, (), None)
        elif failure == "rng_cuda":
            state["rng"]["cuda"] = [torch.ones(2, dtype=torch.uint8)] * (
                torch.cuda.device_count() + 1
            )
        elif failure == "scheduler_schema":
            del state["components"]["scheduler"]["last_epoch"]
        elif failure == "scheduler_nan":
            state["components"]["scheduler"]["_last_lr"] = [float("nan")]
        elif failure == "scheduler_counter_string":
            state["components"]["scheduler"]["last_epoch"] = "bad"
        elif failure == "scheduler_lr_string":
            state["components"]["scheduler"]["_last_lr"] = ["bad"]
        elif failure == "scheduler_groups":
            state["components"]["scheduler"]["_last_lr"] = []
        elif failure == "missing_global_step":
            del state["counters"]["global_step"]
        elif failure == "missing_metrics":
            del state["counters"]["training_metrics"]
        elif failure == "missing_accumulation":
            del state["counters"]["_grad_accum_step"]
        elif failure == "rollout":
            state["rollout_state"] = {"schema_version": 1, "publication_version": -1}
        elif failure == "component_schema":
            state["components"]["unexpected"] = {}

    bad = tmp_path / failure
    republish(path, bad, mutate)
    target = TinyRuntime()
    before = snapshot(target)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        target.load_runtime_checkpoint(str(bad))
    assert_same(snapshot(target), before)
    await target.train_step()
    assert target.global_step == 1


@pytest.mark.parametrize("failure", ["slot_nan", "policy_nan", "gradient_inf"])
async def test_invalid_live_state_cannot_publish(checkpoint, tmp_path, failure):
    trainer, _ = checkpoint
    if failure == "slot_nan":
        next(iter(trainer.optimizer.state.values()))["exp_avg"].fill_(float("nan"))
    elif failure == "policy_nan":
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(float("nan"))
    else:
        parameter = next(trainer.model.parameters())
        parameter.grad = torch.full_like(parameter, float("inf"))
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="finite|gradient"):
        trainer.save_runtime_checkpoint(str(destination))
    assert not destination.exists()
    assert not list(tmp_path.glob(".must-not-exist.stateset-stage-*"))


@pytest.mark.parametrize("component", ["optimizer", "model"])
async def test_failed_component_commit_quarantines_until_full_reload(
    checkpoint, tmp_path, monkeypatch, component
):
    _, path = checkpoint
    target = TinyRuntime()

    def fail(state, **kwargs):
        raise RuntimeError("injected component commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(getattr(target, component), "load_state_dict", fail)
        with pytest.raises(RuntimeError, match="injected component commit failure"):
            target.load_runtime_checkpoint(str(path))
    assert target._runtime_checkpoint_poisoned
    # The guard must run even when a profiling phase appears nested/active.
    target._runtime_profile_active = True
    for method in (
        target.train_step,
        target.training_step,
        target.train_step_token_level,
        target.generate_trajectories,
        target.generate_group_responses,
        target.train,
    ):
        with pytest.raises(RuntimeError, match="quarantined"):
            await method()
    target._runtime_profile_active = False
    for method in (
        target.save_runtime_checkpoint,
        target.save_checkpoint,
        target.save_model,
    ):
        with pytest.raises(RuntimeError, match="quarantined"):
            method(str(tmp_path / "blocked"))
    with pytest.raises(RuntimeError, match="quarantined"):
        target._apply_optimizer_step()
    with pytest.raises(ValueError):
        target.load_runtime_checkpoint(str(tmp_path / "missing"), trusted=True)
    assert target._runtime_checkpoint_poisoned
    assert target.load_runtime_checkpoint(str(path))
    assert not target._runtime_checkpoint_poisoned
    await target.train_step()
    assert target.global_step == 2


async def test_nonfinite_published_policy_cannot_restore_success(checkpoint, tmp_path):
    _, path = checkpoint

    def mutate(state, staging):
        weights = torch.load(staging / "pytorch_model.bin", weights_only=True)
        next(iter(weights.values())).fill_(float("nan"))
        torch.save(weights, staging / "pytorch_model.bin")

    bad = tmp_path / "nonfinite-policy"
    republish(path, bad, mutate)
    target = TinyRuntime()
    with pytest.raises(ValueError, match="finite"):
        target.load_runtime_checkpoint(str(bad))
    assert target._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await target.train_step()
    with pytest.raises(RuntimeError, match="quarantined"):
        target.save_runtime_checkpoint(str(tmp_path / "bad-copy"))
    assert target.load_runtime_checkpoint(str(path))


async def test_valid_legacy_checkpoint_and_sentinel_counters_remain_loadable(
    checkpoint, tmp_path
):
    _, path = checkpoint

    def mutate(state, staging):
        state.pop("adaptive_kl_controller")
        state.pop("runtime_component_identities")
        state.pop("learning_state")
        state["current_task_id"] = "support"
        state["counters"]["best_eval_metric"] = float("inf")
        state["counters"]["value_warmup_complete"] = True
        state["counters"]["_current_task_id"] = "support"

    legacy = tmp_path / "legacy"
    republish(path, legacy, mutate)
    target = TinyRuntime()
    assert target.load_runtime_checkpoint(str(legacy))
    assert target.best_eval_metric == float("inf")
    assert target.value_warmup_complete is True
    assert target._current_task_id == "support"
    await target.train_step()


async def test_valid_runtime_continuation_matches_uninterrupted(checkpoint):
    trainer, path = checkpoint
    await trainer.train_step()
    expected = snapshot(trainer)
    restored = TinyRuntime()
    assert restored.load_runtime_checkpoint(str(path))
    await restored.train_step()
    assert_same(snapshot(restored), expected)


async def test_adapted_kl_state_matches_next_uninterrupted_update(tmp_path):
    torch.manual_seed(4)
    trainer = TinyRuntime()
    trainer.kl_controller = SimpleNamespace(kl_coef=0.2, target_kl=0.01, horizon=10)
    await trainer.train_step()
    assert trainer.kl_controller.kl_coef != 0.2
    path = tmp_path / "adaptive"
    trainer.save_runtime_checkpoint(str(path))
    await trainer.train_step()
    expected = snapshot(trainer)
    restored = TinyRuntime()
    restored.kl_controller = SimpleNamespace(kl_coef=0.2, target_kl=0.01, horizon=10)
    assert restored.load_runtime_checkpoint(str(path))
    await restored.train_step()
    assert_same(snapshot(restored), expected)


@pytest.mark.parametrize("failure", ["missing", "nan", "negative", "target", "horizon"])
async def test_bad_adaptive_kl_state_rejects_before_mutation(tmp_path, failure):
    trainer = TinyRuntime()
    trainer.kl_controller = SimpleNamespace(kl_coef=0.2, target_kl=0.01, horizon=10)
    await trainer.train_step()
    good = tmp_path / "adaptive-good"
    trainer.save_runtime_checkpoint(str(good))

    def mutate(state, staging):
        if failure == "missing":
            del state["adaptive_kl_controller"]
        elif failure == "nan":
            state["adaptive_kl_controller"]["kl_coef"] = float("nan")
        elif failure == "negative":
            state["adaptive_kl_controller"]["kl_coef"] = -0.2
        elif failure == "target":
            state["adaptive_kl_controller"]["target_kl"] = 0.02
        else:
            state["adaptive_kl_controller"]["horizon"] = 20

    bad = tmp_path / "adaptive-bad"
    republish(good, bad, mutate)
    target = TinyRuntime()
    target.kl_controller = SimpleNamespace(kl_coef=0.3, target_kl=0.01, horizon=10)
    before = snapshot(target)
    with pytest.raises(ValueError, match="adaptive KL"):
        target.load_runtime_checkpoint(str(bad))
    assert_same(snapshot(target), before)


@pytest.mark.parametrize(
    "field", ["gamma", "gae_lambda", "max_grad_norm", "max_prompt_length"]
)
async def test_changed_loss_semantics_reject_before_mutation(checkpoint, field):
    _, path = checkpoint
    target = TinyRuntime()
    setattr(target.config, field, getattr(target.config, field) + 1)
    before = snapshot(target)
    with pytest.raises(ValueError, match="configuration differs"):
        target.load_runtime_checkpoint(str(path))
    assert_same(snapshot(target), before)


def test_numpy_metric_scalars_and_initial_best_sentinels_are_valid():
    from stateset_agents.training.runtime_state import _validate_counters

    for sentinel in (float("inf"), -float("inf")):
        _validate_counters(
            {
                "best_eval_metric": sentinel,
                "training_metrics": {"reward": [np.float64(0.5)]},
            }
        )


@pytest.mark.parametrize("failure", ["false", "exception"])
async def test_rollout_publication_failure_keeps_runtime_quarantined(
    checkpoint, failure
):
    _, path = checkpoint
    target = TinyRuntime()

    def fail():
        if failure == "exception":
            raise RuntimeError("injected rollout publication failure")
        return False

    target.agent = SimpleNamespace(sync_rollout_backend=fail)
    with pytest.raises(RuntimeError, match="publication fail"):
        target.load_runtime_checkpoint(str(path))
    assert target._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await target.train_step()
    target.agent.sync_rollout_backend = lambda: True
    assert target.load_runtime_checkpoint(str(path))
    assert not target._runtime_checkpoint_poisoned


async def test_cpu_restored_state_is_owned_not_aliased_to_loader_cache(
    checkpoint, monkeypatch
):
    from stateset_agents.training import checkpoint_io

    _, path = checkpoint
    cached = torch.load(path / "training_state.pt", weights_only=True)
    before = copy.deepcopy(cached)
    original = checkpoint_io.load_checkpoint_file

    def load(file, **kwargs):
        if Path(file).name == "training_state.pt":
            return cached
        return original(file, **kwargs)

    monkeypatch.setattr(checkpoint_io, "load_checkpoint_file", load)
    target = TinyRuntime()
    assert target.load_runtime_checkpoint(str(path))
    assert target.training_metrics is not cached["counters"]["training_metrics"]
    assert (
        target.training_metrics["loss"]
        is not cached["counters"]["training_metrics"]["loss"]
    )
    cached["counters"]["training_metrics"]["loss"].append(42.0)
    assert (
        target.training_metrics["loss"]
        == before["counters"]["training_metrics"]["loss"]
    )
    cached["counters"]["training_metrics"]["loss"].pop()
    for parameter in target.model.parameters():
        if parameter.grad is not None:
            parameter.grad.add_(1)
    await target.train_step()
    assert_same(cached, before)


async def test_actual_adaptive_ppo_next_update_matches_uninterrupted(
    tmp_path, monkeypatch
):
    from stateset_agents.training import ppo_trainer as ppo
    from tests.unit.test_ppo_safety import (
        CharacterTokenizer,
    )
    from tests.unit.test_ppo_safety import TinyPolicy as TinyPPOPolicy
    from tests.unit.test_ppo_safety import snapshot as ppo_snapshot

    class SaveablePPOPolicy(TinyPPOPolicy):
        def save_pretrained(self, path):
            torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

    monkeypatch.setattr(ppo, "_load_transformers_ppo", lambda: True)
    monkeypatch.setattr(ppo, "get_cosine_schedule_with_warmup", None)

    def make_trainer():
        model = SaveablePPOPolicy()
        reference = copy.deepcopy(model).eval()
        reference.requires_grad_(False)
        tokenizer = CharacterTokenizer()
        tokenizer.save_pretrained = lambda path: (
            Path(path) / "tokenizer.json"
        ).write_text("{}")
        return ppo.PPOTrainer(
            config=ppo.PPOConfig(
                num_episodes=3,
                num_ppo_epochs=1,
                max_prompt_length=4,
                max_completion_length=4,
                value_hidden_size=4,
                report_to="none",
                entropy_coef=0.0,
                beta=0.2,
                use_adaptive_kl=True,
                target_kl=0.01,
            ),
            model=model,
            tokenizer=tokenizer,
            reward_fn=lambda prompt, response: 0.0,
            ref_model=reference,
        )

    batch = (["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    torch.manual_seed(31)
    trainer = make_trainer()
    await trainer.train_step(*batch)
    assert trainer.kl_controller.kl_coef != 0.2
    path = tmp_path / "actual-adaptive-ppo"
    trainer.save_runtime_checkpoint(str(path))
    await trainer.train_step(*batch)
    expected = ppo_snapshot(trainer)
    expected_rng = capture_rng(torch)
    target = make_trainer()
    assert target.load_runtime_checkpoint(str(path))
    await target.train_step(*batch)
    assert_same(ppo_snapshot(target), expected)
    assert_same(capture_rng(torch), expected_rng)


async def test_reference_modules_are_validated_without_deepcopy(tmp_path):
    class NoCopyReference(torch.nn.Linear):
        def __deepcopy__(self, memo):
            raise AssertionError("reference weights must not be deepcopied")

    trainer = TinyRuntime()
    trainer.ref_model = NoCopyReference(2, 1).requires_grad_(False)
    await trainer.train_step()
    path = tmp_path / "reference"
    trainer.save_runtime_checkpoint(str(path))
    target = TinyRuntime()
    target.ref_model = NoCopyReference(2, 1).requires_grad_(False)
    assert target.load_runtime_checkpoint(str(path))
    for name, value in trainer.ref_model.state_dict().items():
        torch.testing.assert_close(
            target.ref_model.state_dict()[name], value, rtol=0, atol=0
        )


@pytest.mark.parametrize("failure", ["shape", "nonfinite"])
async def test_invalid_reference_module_rejects_before_policy_mutation(
    tmp_path, failure
):
    trainer = TinyRuntime()
    trainer.ref_model = torch.nn.Linear(2, 1).requires_grad_(False)
    await trainer.train_step()
    good = tmp_path / "reference-good"
    trainer.save_runtime_checkpoint(str(good))

    def mutate(state, staging):
        weights = state["components"]["ref_model"]
        if failure == "shape":
            weights["weight"] = torch.ones(9)
        else:
            weights["weight"].fill_(float("nan"))

    bad = tmp_path / "reference-bad"
    republish(good, bad, mutate)
    target = TinyRuntime()
    target.ref_model = torch.nn.Linear(2, 1).requires_grad_(False)
    before = snapshot(target)
    with pytest.raises(ValueError, match="schema|finite"):
        target.load_runtime_checkpoint(str(bad))
    assert_same(snapshot(target), before)


@pytest.mark.parametrize(
    "binding",
    ["callable_object", "partial_object", "partial_copy_hook", "custom_child"],
)
async def test_opaque_scheduler_copy_graph_rejects_before_copy_or_learner_change(
    checkpoint, tmp_path, binding
):
    _, path = checkpoint
    target = TinyRuntime()
    copied = []

    class CopyTrap:
        def __call__(self, step):
            return 1.0

        def __deepcopy__(self, memo):
            copied.append("unsafe copy")
            raise AssertionError("unsafe scheduler binding copied")

    if binding == "callable_object":
        target.scheduler = torch.optim.lr_scheduler.LambdaLR(
            target.optimizer, CopyTrap()
        )
    elif binding in {"partial_object", "partial_copy_hook"}:

        def schedule(step, *, opaque):
            return 1.0

        bound = partial(
            schedule, opaque=CopyTrap() if binding == "partial_object" else 3
        )
        if binding == "partial_copy_hook":

            def unsafe_copy(memo):
                copied.append("unsafe partial copy")
                raise AssertionError("custom partial copy hook invoked")

            bound.__deepcopy__ = unsafe_copy
        target.scheduler = torch.optim.lr_scheduler.LambdaLR(target.optimizer, bound)
    else:

        class CustomChild(torch.optim.lr_scheduler.StepLR):
            def __deepcopy__(self, memo):
                copied.append("unsafe child copy")
                raise AssertionError("custom scheduler child copied")

        target.scheduler = torch.optim.lr_scheduler.SequentialLR(
            target.optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(target.optimizer),
                CustomChild(target.optimizer, 1),
            ],
            milestones=[1],
        )
    before = snapshot(target)
    with pytest.raises(ValueError, match="Opaque|opaque|Custom"):
        target.load_runtime_checkpoint(str(path))
    assert copied == []
    assert_same(snapshot(target), before)
    with pytest.raises(ValueError, match="Opaque|opaque|Custom"):
        target.save_runtime_checkpoint(str(tmp_path / "unsafe-graph"))
    assert copied == []
    assert not (tmp_path / "unsafe-graph").exists()


@pytest.mark.parametrize("kind", ["lambda_partial", "sequential"])
async def test_standard_scheduler_binding_graph_still_roundtrips(tmp_path, kind):
    def schedule(step, *, warmup):
        return min(1.0, (step + 1) / warmup)

    def make():
        trainer = TinyRuntime()
        if kind == "lambda_partial":
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
                trainer.optimizer, partial(schedule, warmup=3)
            )
        else:
            trainer.scheduler = torch.optim.lr_scheduler.SequentialLR(
                trainer.optimizer,
                [
                    torch.optim.lr_scheduler.ConstantLR(trainer.optimizer),
                    torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1),
                ],
                milestones=[2],
            )
        return trainer

    trainer = make()
    await trainer.train_step()
    path = tmp_path / kind
    trainer.save_runtime_checkpoint(str(path))
    await trainer.train_step()
    expected = snapshot(trainer)
    target = make()
    assert target.load_runtime_checkpoint(str(path))
    await target.train_step()
    assert_same(snapshot(target), expected)


@pytest.mark.parametrize(
    "name,value",
    [
        ("lr", -0.01),
        ("lr", "bad"),
        ("lr", True),
        ("lr", float("nan")),
        ("lr", torch.tensor([0.1, 0.2])),
        ("lr", torch.tensor(True)),
        ("lr", torch.tensor(0.1 + 0.2j)),
        ("lr", torch.tensor(float("inf"))),
        ("initial_lr", -0.01),
        ("initial_lr", "bad"),
        ("weight_decay", -0.1),
        ("weight_decay", True),
        ("weight_decay", torch.tensor(0.1)),
        ("eps", -1e-8),
        ("eps", "bad"),
        ("betas", True),
        ("betas", (0.9,)),
        ("betas", (0.9, 1.0)),
        ("betas", (-0.1, 0.999)),
        ("betas", (True, 0.999)),
        ("betas", (0.9, float("nan"))),
        ("betas", (torch.tensor(0.9), 0.999)),
        ("maximize", 1),
        ("differentiable", 0),
        ("capturable", None),
        ("amsgrad", "false"),
        ("foreach", 1),
        ("fused", "false"),
    ],
)
async def test_invalid_optimizer_group_rejects_before_mutation(
    checkpoint, tmp_path, name, value
):
    _, source = checkpoint

    def mutate(state, staging):
        state["components"]["optimizer"]["param_groups"][0][name] = value

    bad = tmp_path / "bad-group"
    republish(source, bad, mutate)
    target = TinyRuntime()
    before = snapshot(target)
    with pytest.raises(ValueError, match="optimizer group"):
        target.load_runtime_checkpoint(str(bad))
    assert_same(snapshot(target), before)
    await target.train_step()
    assert target.global_step == 1


@pytest.mark.parametrize(
    "controls",
    [
        {"momentum": -0.1},
        {"dampening": -0.1},
        {"nesterov": 1},
        {"momentum": 0.0, "nesterov": True},
        {"momentum": 0.9, "dampening": 0.1, "nesterov": True},
    ],
)
async def test_invalid_sgd_group_rejects_before_mutation(tmp_path, controls):
    def sgd_runtime():
        runtime = TinyRuntime()
        runtime.optimizer = torch.optim.SGD(
            runtime.model.parameters(), lr=0.01, momentum=0.9
        )
        runtime.scheduler = torch.optim.lr_scheduler.StepLR(runtime.optimizer, 1, 0.9)
        return runtime

    original = sgd_runtime()
    await original.train_step()
    source = tmp_path / "sgd-good"
    original.save_runtime_checkpoint(str(source))

    def mutate(state, staging):
        state["components"]["optimizer"]["param_groups"][0].update(controls)

    bad = tmp_path / "sgd-bad"
    republish(source, bad, mutate)
    target = sgd_runtime()
    before = snapshot(target)
    with pytest.raises(ValueError, match="optimizer group"):
        target.load_runtime_checkpoint(str(bad))
    assert_same(snapshot(target), before)
    await target.train_step()


@pytest.mark.parametrize(
    "optimizer_cls", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
)
async def test_tensor_lr_checkpoint_owns_group_tensors_and_continues(
    tmp_path, optimizer_cls
):
    def tensor_runtime():
        runtime = TinyRuntime()
        runtime.optimizer = optimizer_cls(
            runtime.model.parameters(), lr=torch.tensor(0.01), foreach=False
        )
        runtime.scheduler = torch.optim.lr_scheduler.StepLR(runtime.optimizer, 1, 0.9)
        return runtime

    original = tensor_runtime()
    await original.train_step()
    source = tmp_path / "tensor-lr"
    original.save_runtime_checkpoint(str(source))
    restored = tensor_runtime()
    assert restored.load_runtime_checkpoint(str(source))
    for trainer in (original, restored):
        await trainer.train_step()
    assert_same(restored.model.state_dict(), original.model.state_dict())
    assert_same(restored.optimizer.state_dict(), original.optimizer.state_dict())
    assert_same(restored.scheduler.state_dict(), original.scheduler.state_dict())

    from stateset_agents.training.ddp_checkpoint_state import (
        _optimizer_stage,
        _owned_optimizer_state,
    )

    saved = copy.deepcopy(restored.optimizer.state_dict())
    staged = _optimizer_stage(restored.optimizer, saved, torch)
    restored.optimizer.load_state_dict(
        _owned_optimizer_state(restored.optimizer, staged, torch)
    )
    lr = restored.optimizer.param_groups[0]["lr"].clone()
    saved["param_groups"][0]["lr"].fill_(123)
    staged.param_groups[0]["lr"].fill_(456)
    torch.testing.assert_close(restored.optimizer.param_groups[0]["lr"], lr)


async def test_legacy_next_episode_derives_from_existing_current_epoch(tmp_path):
    trainer = TinyRuntime()
    trainer.current_epoch = 0
    trainer._next_episode = 1
    await trainer.train_step()
    good = tmp_path / "cursor-good"
    trainer.save_runtime_checkpoint(str(good))

    def mutate(state, staging):
        del state["counters"]["_next_episode"]
        del state["counters"]["_current_task_id"]

    legacy = tmp_path / "cursor-legacy"
    republish(good, legacy, mutate)
    target = TinyRuntime()
    target.current_epoch = 0
    target._next_episode = 0
    assert target.load_runtime_checkpoint(str(legacy))
    assert target._next_episode == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("best", [0.5, float("inf"), -float("inf")])
async def test_numpy_metric_history_round_trips_with_weights_only_loading(
    checkpoint, tmp_path, best
):
    trainer, _ = checkpoint
    large_integer = 2**63 - 1
    values = [np.float64(0.5), np.float32(0.25), np.int64(large_integer)]
    trainer.training_metrics["mixed"] = values
    trainer.metrics_history = {"mixed": list(values)}
    histories = trainer.metrics_history
    metrics = trainer.training_metrics
    trainer.best_eval_metric = np.float64(best)
    path = tmp_path / "numpy-metrics"
    trainer.save_runtime_checkpoint(str(path))
    assert trainer.training_metrics is metrics
    assert trainer.training_metrics["mixed"] is values
    assert trainer.metrics_history is histories
    assert [type(item) for item in values] == [np.float64, np.float32, np.int64]
    assert type(trainer.best_eval_metric) is np.float64
    expected = [0.5, 0.25, large_integer]
    assert json.loads((path / "training_metrics.json").read_text())["mixed"] == expected
    saved = torch.load(path / "training_state.pt", weights_only=True)
    assert saved["counters"]["training_metrics"]["mixed"] == expected
    target = TinyRuntime()
    assert target.load_runtime_checkpoint(str(path))
    assert target.training_metrics["mixed"] == expected
    assert target.metrics_history["mixed"] == expected
    for history in (target.training_metrics, target.metrics_history):
        assert [type(item) for item in history["mixed"]] == [float, float, int]
    assert target.best_eval_metric == best
    assert type(target.best_eval_metric) is float


@pytest.mark.parametrize("legacy", [False, True])
async def test_contradictory_task_ids_rejected_before_mutation(
    checkpoint, tmp_path, legacy
):
    _, path = checkpoint

    def mutate(state, staging):
        state["counters"]["_current_task_id"] = "support"
        if legacy:
            state.pop("learning_state")
            state["current_task_id"] = "other-task"

    bad = tmp_path / "conflicting-task-ids"
    republish(path, bad, mutate)
    target = TinyRuntime()
    before = snapshot(target)
    with pytest.raises(ValueError, match="task identifiers disagree"):
        target.load_runtime_checkpoint(str(bad))
    assert_same(snapshot(target), before)
    assert target._current_task_id is None


@pytest.mark.parametrize("legacy", [False, True])
async def test_task_id_restored_from_consistent_or_legacy_counter(
    checkpoint, tmp_path, legacy
):
    _, path = checkpoint

    def mutate(state, staging):
        state["counters"]["_current_task_id"] = "support"
        if legacy:
            state.pop("learning_state")
            state.pop("current_task_id", None)
        else:
            state["learning_state"]["current_task_id"] = "support"

    good = tmp_path / "task-id"
    republish(path, good, mutate)
    target = TinyRuntime()
    assert target.load_runtime_checkpoint(str(good))
    assert target._current_task_id == "support"

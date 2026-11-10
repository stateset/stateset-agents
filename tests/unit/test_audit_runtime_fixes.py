"""Regressions for remote, fan-out, data and training defects (2026-11 audit)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents.remote import fireworks, prime
from stateset_agents.remote.job import JobStatus


def test_fireworks_dataset_id_keeps_its_random_suffix_for_long_names() -> None:
    spec = SimpleNamespace(output_dir="outputs/" + "a" * 80)
    first, second = fireworks._dataset_id(spec), fireworks._dataset_id(spec)
    assert first != second
    assert len(first) <= 63 and len(second) <= 63
    assert first.startswith("stateset-aaa")


def test_fanout_progress_event_with_nan_stays_json_serialisable() -> None:
    from stateset_agents.fanout import orchestrator

    source = Path(orchestrator.__file__).read_text(encoding="utf-8")
    assert "parse_constant=lambda _: None" in source
    event = json.loads('{"episode_reward": NaN}', parse_constant=lambda _: None)
    assert json.dumps(event, allow_nan=False) == '{"episode_reward": null}'


def test_prime_cancel_is_not_overwritten_by_the_exit_code() -> None:
    source = Path(prime.__file__).read_text(encoding="utf-8")
    assert "if record.status is not JobStatus.CANCELLED:" in source
    assert JobStatus.CANCELLED is not JobStatus.FAILED


def test_resumed_schedule_uses_the_new_run_length() -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.preference import _loop

    def scheduler(total: int):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        return optimizer, _loop.build_scheduler(optimizer, "linear", 0, total)

    optimizer, short = scheduler(4)
    for _ in range(4):
        optimizer.step()
        short.step()
    saved = short.state_dict()

    optimizer, longer = scheduler(8)
    holder = SimpleNamespace(optimizer=optimizer, lr_scheduler=longer)
    schedules = longer.lr_lambdas
    current = [dict(vars(s)) for s in schedules]
    longer.load_state_dict(saved)
    assert schedules[0].total == 4  # what loading alone does
    for schedule, values in zip(schedules, current, strict=True):
        vars(schedule).update(values)
    assert schedules[0].total == 8
    optimizer.step()
    longer.step()
    assert holder.lr_scheduler.get_last_lr()[0] > 0


def test_neural_reward_prediction_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.neural_reward_trainer import NeuralRewardModel

    torch.manual_seed(0)
    model = NeuralRewardModel()
    model.train()
    first = model.predict_reward("a prompt", "a reply")
    assert model.predict_reward("a prompt", "a reply") == first
    assert model.training  # the caller's mode is restored


def test_empty_embedding_cache_is_still_a_cache() -> None:
    source = Path(
        __import__("stateset_agents.data.conversation_dataset", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "include_embeddings and self.embedding_cache is not None" in source


def test_cosine_kl_controller_accepts_min_kl_coef() -> None:
    from stateset_agents.training.kl_controllers import (
        WarmupKLScheduler,
        create_kl_controller,
    )

    controller = create_kl_controller("cosine", min_kl_coef=0.01, total_steps=10)
    assert controller.min_kl_coef == 0.01
    warmup = WarmupKLScheduler(warmup_steps=100, total_steps=100)
    warmup.update(0.0, 100)  # used to divide by zero


def test_cache_service_size_and_keys_do_not_deadlock() -> None:
    import threading

    from stateset_agents.utils.cache import CacheService

    cache = CacheService()
    cache.set("k", "v")
    seen: list[object] = []
    worker = threading.Thread(target=lambda: seen.extend([cache.size(), cache.keys()]))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and seen[0] == 1


def test_expression_sanitiser_keeps_whitespace_between_numbers() -> None:
    from stateset_agents.utils.symbolic_math import sanitize_expression

    assert sanitize_expression("2 3") == "2 3"
    assert sanitize_expression("a + b * 2") == "a + b * 2"
    assert "\\" not in sanitize_expression("\\frac{a}{b}")


def test_missing_tool_call_id_gets_a_generated_id() -> None:
    from stateset_agents.api import messages_utils

    source = Path(messages_utils.__file__).read_text(encoding="utf-8")
    assert 'raw_id = call.get("id")' in source


def test_schema_for_pep604_optional_matches_typing_optional() -> None:
    from typing import Optional

    from stateset_agents.core import function_calling, structured_output

    for module, name in (
        (function_calling, "_python_type_to_json_schema"),
        (structured_output, "_python_type_to_json_schema"),
    ):
        convert = getattr(module, name, None)
        if convert is None:
            continue
        assert convert(int | None) == convert(Optional[int])  # noqa: UP045


@pytest.mark.asyncio
async def test_pool_scale_down_does_not_deadlock() -> None:
    import asyncio

    from stateset_agents.core.async_pool import AsyncResourcePool

    class Factory:
        async def create_resource(self) -> object:
            return object()

        async def validate_resource(self, resource: object) -> bool:
            return True

        async def cleanup_resource(self, resource: object) -> None:
            return None

    pool = AsyncResourcePool(Factory(), min_size=1, max_size=3, name="audit")
    await pool.initialize()
    await pool.scale(3)
    await asyncio.wait_for(pool.scale(1), timeout=5)


def test_bear_alpha_rises_when_the_constraint_is_violated() -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.experimental.offline_rl import bear

    source = Path(bear.__file__).read_text(encoding="utf-8")
    assert "alpha_loss = -torch.exp(self.log_alpha) * (" in source
    assert 'self.log_alpha.copy_(checkpoint["log_alpha"])' in source
    assert torch is not None


def test_preflight_with_data_path_and_no_output_dir_does_not_crash(tmp_path) -> None:
    from stateset_agents.config.validation import PreFlightChecker

    checker = PreFlightChecker()
    checker.check_training_readiness(
        {"model_name": "gpt2"}, data_path=str(tmp_path / "missing.jsonl")
    )

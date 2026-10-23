"""One W&B reporter owns init/log/finish for every dedicated entry point.

GEPO, VAPO and GSPO-token skipped ``wandb.init`` when logging was requested
without a project, but still called ``wandb.log`` after the first step,
which raises in the real SDK (``You must call wandb.init() before
wandb.log()``). DAPO already disabled logging instead; the copies drifted.
:class:`~stateset_agents.training.dedicated_reporting.WandbRun` is now the
one implementation, tested here against a fake ``wandb`` module.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training import (  # noqa: E402
    gepo_trainer,
    gspo_trainer,
    vapo_trainer,
)
from tests.unit.test_dedicated_entrypoint_contracts import (  # noqa: E402
    PROMPTS,
    _config,
    _harness,
    _reward,
)

pytestmark = pytest.mark.timeout(240)


class FakeWandb:
    """The SDK surface the reporter uses; ``log`` before ``init`` raises like
    the real ``wandb`` does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.run: Any = None

    def init(self, **kwargs: Any) -> Any:
        self.calls.append(("init", kwargs))
        self.run = SimpleNamespace(**kwargs)
        return self.run

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            raise RuntimeError("You must call wandb.init() before wandb.log()")
        self.calls.append(("log", step))

    def finish(self, exit_code: int | None = None) -> None:
        self.calls.append(("finish", exit_code))
        self.run = None

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    # Older modules bound ``wandb`` at import; point those copies at the fake.
    for module in (gepo_trainer, vapo_trainer, gspo_trainer):
        monkeypatch.setattr(module, "wandb", fake, raising=False)
    return fake


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["dapo", "gepo", "vapo"])
async def test_wandb_without_a_project_disables_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    fake_wandb: FakeWandb,
    caplog: pytest.LogCaptureFixture,
    name: str,
) -> None:
    entry, _run = _harness(monkeypatch, name)
    with caplog.at_level(logging.WARNING):
        await entry(
            model_name="gpt2",
            reward_fn=_reward,
            train_prompts=PROMPTS,
            config=_config(name, logging_steps=1),
            output_dir=str(tmp_path),
            use_wandb=True,
            wandb_project=None,
        )
    assert fake_wandb.calls == []
    assert "no wandb_project" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["dapo", "gepo", "vapo"])
async def test_wandb_with_a_project_logs_every_reported_step_then_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_wandb: FakeWandb, name: str
) -> None:
    entry, _run = _harness(monkeypatch, name)
    await entry(
        model_name="gpt2",
        reward_fn=_reward,
        train_prompts=PROMPTS,
        config=_config(name, logging_steps=1),
        output_dir=str(tmp_path),
        use_wandb=True,
        wandb_project="proj",
    )
    assert fake_wandb.names() == ["init", "log", "log", "finish"]
    init = fake_wandb.calls[0][1]
    assert init["project"] == "proj"
    assert name in init["tags"]
    assert [step for kind, step in fake_wandb.calls if kind == "log"] == [0, 1]


@pytest.mark.asyncio
async def test_gspo_token_without_a_project_does_not_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, fake_wandb: FakeWandb
) -> None:
    from stateset_agents.training import gspo_token_trainer as tok
    from stateset_agents.training.gspo_config import GSPOConfig

    class Manager:
        ref_model = None

        def __init__(self, config: Any) -> None:
            pass

        def load_model_and_tokenizer(self) -> tuple[Any, Any]:
            return object(), object()

    class FakeTokenTrainer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def train_step_token_level(
            self, queries: Any, num_groups: int = 1
        ) -> dict[str, Any]:
            return {"policy_loss": 0.5}

        def save_model(self, path: str) -> None:
            pass

    monkeypatch.setattr(gspo_trainer, "GSPOModelManager", Manager)
    monkeypatch.setattr(tok, "GSPOTokenTrainer", FakeTokenTrainer)
    await tok.train_with_gspo_token(
        config=GSPOConfig(
            output_dir=str(tmp_path),
            report_to="wandb",
            wandb_project=None,
            num_outer_iterations=2,
        ),
        agent=SimpleNamespace(model=None, tokenizer=None),
        environment=SimpleNamespace(scenarios=[]),
        reward_model=SimpleNamespace(),
        train_queries=["q"],
    )
    assert fake_wandb.calls == []


def test_reporter_finishes_a_failed_run_with_an_error_code(
    fake_wandb: FakeWandb,
) -> None:
    from stateset_agents.training.dedicated_reporting import WandbRun

    reporter = WandbRun(
        "gepo", enabled=True, project="proj", config={"a": 1}, tags=["gepo"]
    )
    with pytest.raises(ValueError, match="boom"):
        with reporter:
            reporter.log({"loss": 1.0}, step=0)
            raise ValueError("boom")
    assert fake_wandb.names() == ["init", "log", "finish"]
    assert fake_wandb.calls[-1] == ("finish", 1)
    reporter.log({"loss": 2.0}, step=1)  # finished: a no-op, never a crash
    assert fake_wandb.names() == ["init", "log", "finish"]


def test_reporter_is_inert_when_logging_is_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.training.dedicated_reporting import WandbRun

    monkeypatch.setitem(sys.modules, "wandb", None)  # not even installed
    with WandbRun("vapo", enabled=False, project="proj", config={}) as reporter:
        reporter.log({"loss": 1.0}, step=0)
        assert not reporter.active


def test_missing_wandb_is_an_import_error_naming_the_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateset_agents.training.dedicated_reporting import WandbRun

    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="wandb is required for GEPO logging"):
        WandbRun("gepo", enabled=True, project="proj", config={}).start()


def test_run_name_and_config_dict(fake_wandb: FakeWandb) -> None:
    from stateset_agents.training.dedicated_reporting import WandbRun

    config = SimpleNamespace(to_dict=lambda: {"lr": 1.0}, run_name=None)
    with WandbRun("dapo", enabled=True, project="p", config=config, tags=["x"]):
        pass
    init = fake_wandb.calls[0][1]
    assert init["config"] == {"lr": 1.0}
    assert init["name"].startswith("dapo-")
    assert init["tags"] == ["x"]

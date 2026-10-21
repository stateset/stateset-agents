import importlib
from unittest.mock import Mock


def test_setup_wandb_from_env_allows_offline_without_api_key(monkeypatch) -> None:
    config_mod = importlib.import_module("stateset_agents.utils.wandb_config")

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_TAGS", " alpha, ,beta ,, gamma ")

    config = config_mod.setup_wandb_from_env()

    assert config is not None
    assert config.offline_mode is True
    assert config.tags == ["alpha", "beta", "gamma"]


def test_wandb_logger_stays_enabled_in_offline_mode_without_api_key(
    monkeypatch,
) -> None:
    integration = importlib.import_module("stateset_agents.utils.wandb_integration")

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(integration, "WANDB_INSTALLED", True)
    monkeypatch.setattr(integration, "_load_wandb", lambda: True)

    logger = integration.WandBLogger(enabled=True)

    assert logger.enabled is True
    assert logger.offline_mode is True


def test_wandb_logger_uses_offline_mode_for_init_run(monkeypatch) -> None:
    integration = importlib.import_module("stateset_agents.utils.wandb_integration")

    class _Run:
        name = "offline-run"

        def __init__(self) -> None:
            self.summary = {}

        def finish(self) -> None:
            return None

    class _StubWandB:
        def __init__(self) -> None:
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return _Run()

    stub = _StubWandB()

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(integration, "WANDB_INSTALLED", True)
    monkeypatch.setattr(integration, "_load_wandb", lambda: True)
    monkeypatch.setattr(integration, "wandb", stub)

    logger = integration.WandBLogger(enabled=True)
    logger.init_run({"learning_rate": 0.1}, name="offline")

    assert logger.run is not None
    assert stub.init_kwargs is not None
    assert stub.init_kwargs["mode"] == "offline"


def test_supplied_population_metrics_take_precedence_over_local_trajectories(
    monkeypatch,
):
    module = importlib.import_module("stateset_agents.utils.wandb_logger")
    value = module.WandBLogger(enabled=False)
    value.enabled = True
    value.log_metrics = Mock()
    value._compute_trajectory_metrics = Mock(
        side_effect=AssertionError("local groups used")
    )
    value.log_training_step(
        {"total_loss": 0.2},
        0.01,
        3,
        trajectory_groups=[object()],
        trajectory_metrics={"reward_mean": 2.5, "total_trajectories": 5},
    )
    data = value.log_metrics.call_args.args[0]
    assert data["train/reward_mean"] == 2.5 and data["train/total_trajectories"] == 5
    assert data["train/total_loss"] == 0.2
    value._compute_trajectory_metrics.assert_not_called()

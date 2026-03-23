import importlib


def test_wandb_integration_reexports_split_modules() -> None:
    integration = importlib.import_module("stateset_agents.utils.wandb_integration")
    config_mod = importlib.import_module("stateset_agents.utils.wandb_config")
    logger_mod = importlib.import_module("stateset_agents.utils.wandb_logger")

    assert integration.WandBConfig is config_mod.WandBConfig
    assert integration.create_wandb_config is config_mod.create_wandb_config
    assert integration.setup_wandb_from_env is config_mod.setup_wandb_from_env
    assert integration.WandBLogger is logger_mod.WandBLogger
    assert integration.init_wandb is logger_mod.init_wandb


def test_utils_package_reexports_wandb_logger() -> None:
    utils_pkg = importlib.import_module("stateset_agents.utils")
    logger_mod = importlib.import_module("stateset_agents.utils.wandb_logger")

    assert utils_pkg.WandBLogger is logger_mod.WandBLogger

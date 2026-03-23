import importlib
import sys


def test_grpo_service_import_uses_lazy_app_proxy() -> None:
    sys.modules.pop("stateset_agents.api.grpo.service", None)

    service = importlib.import_module("stateset_agents.api.grpo.service")

    assert isinstance(service.app, service.LazyApp)
    assert service.app._app is None


def test_grpo_lazy_app_materializes_fastapi_app_on_access() -> None:
    sys.modules.pop("stateset_agents.api.grpo.service", None)

    service = importlib.import_module("stateset_agents.api.grpo.service")

    assert service.app.title == "GRPO Service"
    assert service.app._app is not None

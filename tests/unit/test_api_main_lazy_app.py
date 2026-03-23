import importlib
import sys


def test_api_main_import_uses_lazy_app_proxy() -> None:
    sys.modules.pop("stateset_agents.api.main", None)

    main = importlib.import_module("stateset_agents.api.main")

    assert isinstance(main.app, main.LazyApp)
    assert main.app._app is None


def test_lazy_app_materializes_fastapi_app_on_access() -> None:
    sys.modules.pop("stateset_agents.api.main", None)

    main = importlib.import_module("stateset_agents.api.main")

    assert main.app.title == main.get_config().title
    assert main.app._app is not None

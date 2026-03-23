import importlib
import sys


def test_api_package_import_does_not_import_gateway_main() -> None:
    sys.modules.pop("stateset_agents.api.main", None)
    sys.modules.pop("stateset_agents.api", None)

    api = importlib.import_module("stateset_agents.api")

    assert "stateset_agents.api.main" not in sys.modules
    assert callable(api.create_app)

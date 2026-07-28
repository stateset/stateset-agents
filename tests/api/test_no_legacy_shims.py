"""Regression test: legacy GRPO service shims must stay deleted.

`stateset_agents.api.ultimate_grpo_service` and
`stateset_agents.api.enhanced_ultimate_grpo_service` were duplicate,
unmaintained FastAPI surfaces that shadowed `stateset_agents.api.main`.
They were removed as part of the API hardening pass; this test guards
against their accidental reintroduction.
"""

import importlib.util


def test_ultimate_grpo_service_shim_is_gone() -> None:
    assert importlib.util.find_spec("stateset_agents.api.ultimate_grpo_service") is None


def test_enhanced_ultimate_grpo_service_shim_is_gone() -> None:
    assert (
        importlib.util.find_spec("stateset_agents.api.enhanced_ultimate_grpo_service")
        is None
    )

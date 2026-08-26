"""Guardrail: the [dev] extra must be a superset of [training] and [api].

CI installs ".[dev,api]" and the dev lock file is compiled from these extras,
so a runtime dependency added to [training] or [api] but not [dev] silently
vanishes from every dev/CI environment. A self-referential
"stateset-agents[training]" spec inside [dev] would make this structural, but
it sends pip-compile's resolver into ResolutionTooDeepError — so the deps are
duplicated deliberately and this test keeps them from drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _package_names(requirements: list[str]) -> set[str]:
    names = set()
    for req in requirements:
        match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", req.strip())
        assert match, f"Unparseable requirement: {req!r}"
        names.add(match.group(0).lower().replace("_", "-"))
    return names


def test_dev_extra_is_superset_of_training_and_api() -> None:
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    dev = _package_names(extras["dev"])
    for extra in ("training", "api"):
        missing = _package_names(extras[extra]) - dev
        assert not missing, (
            f"[dev] is missing packages from [{extra}]: {sorted(missing)}. "
            "Add them to [dev] in pyproject.toml (see the NOTE at the top of "
            "the dev extra for why they are duplicated) and run `make lock`."
        )


def test_glm53_extra_pins_required_transformers_generation() -> None:
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]

    assert "stateset-agents[training]" in extras["glm53"]
    assert "transformers>=5.16.0" in extras["glm53"]


def test_qwen38next_extra_pins_required_transformers_generation() -> None:
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]

    assert "stateset-agents[training]" in extras["qwen38next"]
    assert "transformers>=5.8.0" in extras["qwen38next"]

import re
from pathlib import Path

DEPRECATED_IMPORT_RE = re.compile(
    r"^\s*(from|import)\s+(core|training|utils|rewards|environments)\b"
)

EXCLUDED_ROOTS = {
    ".git",
    ".venv",
    "build",
    "dist",
    "htmlcov",
    "target",
    "core",
    "training",
    "utils",
    "rewards",
    "environments",
    "api",
    "grpo_agent_framework",
}
EXCLUDED_ANYWHERE = {"__pycache__"}


def test_no_deprecated_shim_imports() -> None:
    roots = [Path("stateset_agents"), Path("tests"), Path("examples"), Path("benchmarks")]
    offenders = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            parts = path.parts
            if parts and parts[0] in EXCLUDED_ROOTS:
                continue
            if any(part in EXCLUDED_ANYWHERE for part in parts):
                continue
            for line in path.read_text().splitlines():
                if DEPRECATED_IMPORT_RE.search(line):
                    offenders.append((path.as_posix(), line.strip()))
                    break

    assert not offenders, f"Found deprecated shim imports: {offenders}"

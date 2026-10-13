#!/usr/bin/env python3
"""Per-framework pod environments for the five-framework matched shootout.

The harness venv (StateSet + the pinned TRL, installed by
``runpod_shootout.py`` from ``extra_pip``) runs ``shootout.py``, the
StateSet/TRL adapters, and every shared evaluation. verl, OpenRLHF and
NeMo-RL each pin their own torch/vLLM/transformers, so each trains in an
isolated environment that the competitor adapter invokes as a subprocess:

* verl and OpenRLHF: a ``uv`` venv per framework, pip-installed from PyPI;
* NeMo-RL: a source checkout at the pinned tag driven by ``uv run --locked``
  (NeMo-RL ships only as source; it also builds per-worker venvs, which
  ``prefetch_venvs.py`` materializes ahead of the first measured run).

    # print the install script (no side effects)
    python benchmarks/adapters/framework_envs.py
    # install everything (on a GPU pod; long-running)
    python benchmarks/adapters/framework_envs.py --execute

Disk/time figures are planning estimates from the pinned dependency sets,
not measurements.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

VENV_ROOT = "/workspace/venvs"
FRAMEWORK_ROOT = "/workspace/frameworks"


@dataclass(frozen=True)
class FrameworkEnv:
    """How one framework's isolated training environment is built."""

    name: str
    version: str
    python: str
    location: str
    python_env_var: str
    steps: tuple[str, ...]
    estimated_disk_gb: int
    estimated_install_minutes: int
    notes: tuple[str, ...] = field(default_factory=tuple)


def _venv_steps(location: str, python: str, requirements: Sequence[str]) -> tuple:
    interpreter = f"{location}/bin/python"
    return (
        f"uv venv --python {shlex.quote(python)} {shlex.quote(location)}",
        f"uv pip install --python {shlex.quote(interpreter)} "
        + " ".join(shlex.quote(item) for item in requirements),
    )


VERL = FrameworkEnv(
    name="verl",
    version="0.9.1",
    python="3.12",
    location=f"{VENV_ROOT}/verl",
    python_env_var="STATESET_VERL_PYTHON",
    steps=_venv_steps(f"{VENV_ROOT}/verl", "3.12", ["verl[vllm,fsdp]==0.9.1"]),
    estimated_disk_gb=25,
    estimated_install_minutes=15,
    notes=(
        "verl[vllm,fsdp]==0.9.1 pins torch==2.11.0, vllm==0.24.0, "
        "transformers==5.9.0 and a prebuilt flash-attn==2.8.3 (cu130/torch2.11).",
        "torch 2.11 wheels need a host driver new enough for their CUDA runtime.",
    ),
)
OPENRLHF = FrameworkEnv(
    name="openrlhf",
    version="0.11.2",
    python="3.12",
    location=f"{VENV_ROOT}/openrlhf",
    python_env_var="STATESET_OPENRLHF_PYTHON",
    steps=_venv_steps(f"{VENV_ROOT}/openrlhf", "3.12", ["openrlhf[vllm]==0.11.2"]),
    estimated_disk_gb=30,
    estimated_install_minutes=45,
    notes=(
        "openrlhf==0.11.2 hard-requires deepspeed==0.19.6, flash-attn==2.8.3, "
        "ray==2.55.0, transformers==5.15.0; the vllm extra pins vllm==0.29.0.",
        "flash-attn compiles from source when no matching wheel exists "
        "(the dominant install-time risk).",
    ),
)
NEMO_RL = FrameworkEnv(
    name="nemo-rl",
    version="0.7.0",
    python="3.13",
    location=f"{FRAMEWORK_ROOT}/nemo-rl",
    python_env_var="STATESET_NEMO_RL_DIR",
    steps=(
        f"git clone --depth 1 --branch v0.7.0 https://github.com/NVIDIA-NeMo/RL "
        f"{FRAMEWORK_ROOT}/nemo-rl",
        f"test $(git -C {FRAMEWORK_ROOT}/nemo-rl rev-parse HEAD) = "
        "81aa43dda4765b0429cf31dab44441e4e4383911",
        f"(cd {FRAMEWORK_ROOT}/nemo-rl && uv sync --locked)",
        f"(cd {FRAMEWORK_ROOT}/nemo-rl && uv run --locked "
        "nemo_rl/utils/prefetch_venvs.py VllmGenerationWorker DTensorPolicyWorkerV2)",
    ),
    estimated_disk_gb=60,
    estimated_install_minutes=40,
    notes=(
        "NeMo-RL v0.7.0 requires Python 3.13 (uv downloads it) and pins "
        "torch==2.11.0; PyPI 'nemo-rl' is an empty 0.0.0 placeholder.",
        "Per-worker venvs (vLLM, Automodel/DTensor) are separate copies of torch.",
    ),
)
FRAMEWORK_ENVS: dict[str, FrameworkEnv] = {
    env.name: env for env in (VERL, OPENRLHF, NEMO_RL)
}


def install_script(names: Sequence[str] | None = None) -> list[str]:
    """Shell commands that build the requested framework environments."""
    selected = [FRAMEWORK_ENVS[name] for name in (names or FRAMEWORK_ENVS)]
    lines = [
        "set -euo pipefail",
        "python -m pip install --quiet 'uv>=0.8'",
        f"mkdir -p {VENV_ROOT} {FRAMEWORK_ROOT}",
    ]
    for env in selected:
        lines.append(f"# {env.name} {env.version}")
        lines.extend(env.steps)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frameworks", nargs="*", help=", ".join(FRAMEWORK_ENVS))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.frameworks) - set(FRAMEWORK_ENVS))
    if unknown:
        parser.error("unknown frameworks: " + ", ".join(unknown))
    script = "\n".join(install_script(args.frameworks or None)) + "\n"
    if not args.execute:
        print(script, end="")
        return 0
    return subprocess.run(["bash", "-c", script], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

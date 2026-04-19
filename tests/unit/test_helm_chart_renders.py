"""Smoke tests ensuring every values-*.yaml in the helm chart renders/lints.

These tests shell out to the `helm` binary when available. If `helm` is not
installed in the environment (e.g. a minimal CI image), the tests are skipped
rather than failing — but on any developer or release runner, they catch
malformed values files before a broken deployment lands in a cluster.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "deployment" / "helm" / "stateset-agents"

VALUES_FILES = sorted(CHART_PATH.glob("values-*.yaml"))


@pytest.fixture(scope="module")
def helm_binary() -> str:
    binary = shutil.which("helm")
    if not binary:
        pytest.skip("helm binary not available")
    return binary


def test_chart_lints_with_defaults(helm_binary):
    result = subprocess.run(
        [helm_binary, "lint", str(CHART_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"helm lint failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


@pytest.mark.parametrize("values_file", VALUES_FILES, ids=lambda p: p.name)
def test_values_file_lints_cleanly(helm_binary, values_file):
    """Each values-*.yaml must pass `helm lint` with the default chart."""
    result = subprocess.run(
        [helm_binary, "lint", str(CHART_PATH), "-f", str(values_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"helm lint failed for {values_file.name}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


@pytest.mark.parametrize("values_file", VALUES_FILES, ids=lambda p: p.name)
def test_values_file_template_renders(helm_binary, values_file):
    """Each values-*.yaml must render Kubernetes manifests via `helm template`."""
    result = subprocess.run(
        [
            helm_binary,
            "template",
            "test-release",
            str(CHART_PATH),
            "-f",
            str(values_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"helm template failed for {values_file.name}:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # Sanity: rendered output must include a Deployment manifest.
    assert "kind: Deployment" in result.stdout

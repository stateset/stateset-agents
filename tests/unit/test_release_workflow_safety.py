"""Regression tests for live release-evidence workflow semantics."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_codeql_merge_gate_uses_stable_query_suite() -> None:
    config = yaml.safe_load((ROOT / ".github/codeql-config.yml").read_text())
    query_suites = {entry["uses"] for entry in config["queries"]}

    assert "security-and-quality" in query_suites
    assert "security-experimental" not in query_suites


def test_gpu_verification_cannot_pass_by_skipping_for_missing_key() -> None:
    workflow = (ROOT / ".github/workflows/gpu-verify.yml").read_text(encoding="utf-8")

    assert workflow.count("RUNPOD_API_KEY secret not set") == 2
    assert workflow.count("exit 1") >= 2
    assert "skipping GPU verification" not in workflow
    assert "skipping RL GPU verification" not in workflow
    assert "needs: sft-live-smoke" in workflow


def test_gpu_verification_has_total_spend_and_lifetime_backstops() -> None:
    workflow = (ROOT / ".github/workflows/gpu-verify.yml").read_text(encoding="utf-8")

    assert "group: gpu-verify" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "max_cost_usd=0.50" in workflow
    assert "max_provision_attempts=1" in workflow
    assert "max_lifetime_s = 1200" in workflow
    assert "max_cost_usd = 0.50" in workflow
    assert "check_budget(" in workflow
    assert "self_destruct_script(" in workflow
    assert workflow.count("container_disk_gb=40") == 2


def test_gpu_verification_emits_hashed_cuda_evidence() -> None:
    workflow = (ROOT / ".github/workflows/gpu-verify.yml").read_text(encoding="utf-8")

    assert workflow.count('"schema_version": 1') == 2
    assert workflow.count('"cleanup_confirmed"') == 2
    assert 'summary.get("device") != "cuda"' in workflow
    assert 'summary.get("converged")' in workflow
    assert '"dataset_sha256"' in workflow
    assert workflow.count('"wheel_sha256"') == 2
    assert "gpu-verify-rl-evidence" in workflow
    assert "outputs/gpu_verify_rl/evidence.json" in workflow

    scripts = re.findall(r"python - <<'PY'\n(.*?)\n\s*PY", workflow, re.DOTALL)
    assert len(scripts) >= 3
    for script in scripts:
        ast.parse(textwrap.dedent(script))


def test_provider_canaries_run_for_release_tags() -> None:
    workflow = (ROOT / ".github/workflows/provider-canary.yml").read_text(
        encoding="utf-8"
    )

    assert "push:" in workflow
    assert "tags:" in workflow
    assert "- 'v*'" in workflow
    for provider in ("river", "runpod", "fireworks"):
        assert f"--provider {provider} --strict" in workflow
    assert 'pip install -e ".[remote]"' in workflow


def test_publish_requires_readiness_before_build_or_upload() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    readiness = workflow.index("Run mandatory publish readiness gate")
    build = workflow.index("Build distribution")
    test_pypi = workflow.index("Publish to TestPyPI")
    assert readiness < build < test_pypi
    assert "continue-on-error: true" not in workflow
    assert "run_readiness_gate" not in workflow
    assert "publish-readiness-${{ github.sha }}" in workflow


def test_workflows_use_current_action_runtimes() -> None:
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    )

    deprecated = {
        "actions/checkout@v4",
        "actions/setup-python@v5",
        "actions/cache@v4",
        "actions/upload-artifact@v4",
        "actions/download-artifact@v4",
        "actions/dependency-review-action@v4",
        "azure/setup-helm@v4",
        "codecov/codecov-action@v5",
        "docker/build-push-action@v5",
        "docker/login-action@v3",
        "docker/metadata-action@v5",
        "docker/setup-buildx-action@v3",
        "github/codeql-action/analyze@v3",
        "github/codeql-action/autobuild@v3",
        "github/codeql-action/init@v3",
        "peaceiris/actions-gh-pages@v3",
    }

    assert not (deprecated & set(workflows.split()))


def test_tag_publish_attests_and_releases_verified_artifacts_once() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert "release:\n    types: [published]" not in workflow
    assert "actions/attest-build-provenance@v4" in workflow
    assert "attestations: write" in workflow
    assert "github-release:" in workflow
    assert 'gh release create "${GITHUB_REF_NAME}" dist/*' in workflow
    assert workflow.count('--repo "${GITHUB_REPOSITORY}"') == 3
    assert workflow.count("provenance: mode=max") == 2
    assert workflow.count("sbom: true") == 2


def test_pypi_publish_has_explicit_scoped_token_fallback() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert "Publish to PyPI (OIDC or scoped API token)" in workflow
    assert "secrets.PYPI_API_TOKEN != ''" in workflow
    assert "password: ${{ secrets.PYPI_API_TOKEN" in workflow


def test_tag_publish_tests_and_publishes_version_matched_npm_client() -> None:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert "npm-publish:" in workflow
    assert 'tag_version="${GITHUB_REF_NAME#v}"' in workflow
    assert "require('./package.json').version" in workflow
    assert "NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}" in workflow
    assert "npm publish --access public --provenance" in workflow

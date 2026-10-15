"""Source drift must invalidate experiment reuse and matched comparisons."""

import copy
import hashlib
import shutil

import pytest

from stateset_agents.evaluation import implementation


def test_fingerprint_is_portable_and_detects_edits_additions_and_removals(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkout" / "stateset_agents"
    (root / "evaluation").mkdir(parents=True)
    (root / "__init__.py").write_text("# package\n")
    source = root / "evaluation" / "implementation.py"
    source.write_text("# protocol\n")
    monkeypatch.setattr(implementation, "__file__", str(source))
    original = implementation.package_implementation()
    assert (
        original["files"]["evaluation/implementation.py"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )

    installed = tmp_path / "wheel" / "stateset_agents"
    shutil.copytree(root, installed)
    monkeypatch.setattr(
        implementation, "__file__", str(installed / "evaluation/implementation.py")
    )
    assert implementation.package_implementation() == original
    # Caches and non-source artifacts do not change the protocol identity.
    (installed / "cache.pyc").write_bytes(b"cache")
    assert implementation.package_implementation() == original
    extra = installed / "reward.py"
    extra.write_text("reward = 1\n")
    added = implementation.package_implementation()
    assert added != original
    extra.write_text("reward = 0\n")
    assert implementation.package_implementation() != added
    extra.unlink()
    assert implementation.package_implementation() == original


@pytest.mark.parametrize(
    "name,digest",
    [
        ("../reward.py", "0" * 64),
        ("/reward.py", "0" * 64),
        ("./reward.py", "0" * 64),
        ("reward.py", "x" * 64),
        ("reward.py", None),
    ],
)
def test_malformed_source_hashes_are_rejected(name, digest):
    value = {
        "schema_version": 1,
        "scope": "stateset_agents_python_sources",
        "files": {"__init__.py": "0" * 64, name: digest},
    }
    with pytest.raises(ValueError, match="relative .py paths"):
        implementation.validate_implementation(value)


def test_actual_package_contains_scoring_training_and_audit_sources():
    identity = implementation.package_implementation()
    for name in (
        "core/environments/refund_policy_environment.py",
        "remote/river_environment.py",
        "training/river_refund.py",
        "evaluation/agent_study.py",
    ):
        assert name in identity["files"]
    malformed = copy.deepcopy(identity)
    malformed["schema_version"] = True
    with pytest.raises(ValueError, match="Invalid implementation"):
        implementation.validate_implementation(malformed)

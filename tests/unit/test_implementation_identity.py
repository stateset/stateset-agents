"""Implementation revisions are portable and independent of tensor contents."""

import copy
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import implementation_identity as identity  # noqa: E402
from stateset_agents.training.ddp_checkpoint_state import (  # noqa: E402
    capture_payload,
    prepare_payload,
)
from tests.integration.test_learner_preflight import _same  # noqa: E402
from tests.unit.test_ddp_checkpoint_state import _learner  # noqa: E402


def _load_class(path, monkeypatch):
    spec = importlib.util.spec_from_file_location("portable_policy", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "portable_policy", module)
    spec.loader.exec_module(module)
    return module.Policy


def test_identical_sources_at_different_paths_have_equal_identity(
    tmp_path, monkeypatch
):
    source = "class Policy:\n    def forward(self, x):\n        return x + 1\n"
    values = []
    for directory in ("checkout", "wheel"):
        path = tmp_path / directory / "policy.py"
        path.parent.mkdir()
        path.write_text(source)
        values.append(identity.class_implementation(_load_class(path, monkeypatch)))
    assert values[0] == values[1]


def test_source_edit_is_detected_even_without_reloading_class(tmp_path, monkeypatch):
    path = tmp_path / "policy.py"
    path.write_text("class Policy:\n    def forward(self, x):\n        return x\n")
    cls = _load_class(path, monkeypatch)
    before = identity.class_implementation(cls)
    path.write_text(path.read_text() + "\n# revised implementation\n")
    assert identity.class_implementation(cls) != before


def test_class_source_observes_same_size_rewrite_with_restored_mtime(
    tmp_path, monkeypatch
):
    path = tmp_path / "policy.py"
    path.write_text("class Policy:\n    def forward(self, x):\n        return x + 1\n")
    cls = _load_class(path, monkeypatch)
    before = identity.class_implementation(cls)
    original = path.stat()
    path.write_text(path.read_text().replace("x + 1", "x + 2"))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert identity.class_implementation(cls) != before


def test_loaded_method_edit_is_detected_without_a_file_edit(monkeypatch):
    before = identity.class_implementation(torch.nn.Linear)

    def altered(self, value):
        raise AssertionError("identity must not execute forward")

    monkeypatch.setattr(torch.nn.Linear, "forward", altered)
    assert identity.class_implementation(torch.nn.Linear) != before


def test_submodule_code_matters_but_parameter_values_do_not(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    before = identity.model_implementation(model)
    with torch.no_grad():
        model[0].weight.add_(1)
    assert identity.model_implementation(model) == before
    monkeypatch.setattr(torch.nn.Linear, "forward", lambda self, x: x)
    assert identity.model_implementation(model) != before


@pytest.mark.parametrize(
    "fault",
    ["policy", "reference", "optimizer", "runtime", "package", "missing", "legacy"],
)
def test_checkpoint_implementation_drift_rejects_before_mutation(
    tmp_path, monkeypatch, fault
):
    trainer = _learner(tmp_path)
    shared, local, _ = copy.deepcopy(capture_payload(trainer))
    if fault in {"policy", "reference"}:
        model = (
            trainer.agent.model.module if fault == "policy" else trainer.reference_model
        )
        monkeypatch.setattr(model, "forward", lambda *args, **kwargs: None)
    elif fault == "optimizer":
        monkeypatch.setattr(torch.optim.SGD, "step", lambda *args, **kwargs: None)
    elif fault == "runtime":
        monkeypatch.setattr(torch, "__version__", "0.0.0-mismatched")
    elif fault == "package":
        monkeypatch.setattr(identity, "package_revision", lambda: "0" * 64)
    elif fault == "missing":
        shared["identity"].pop("implementation")
    else:
        shared["protocol"] = "stateset-native-ddp-checkpoint-v2"
    before = copy.deepcopy(capture_payload(trainer))
    with pytest.raises(ValueError, match="identity|architecture|protocol"):
        prepare_payload(trainer, shared, local)
    _same(capture_payload(trainer), before, torch)


def test_runtime_precision_controls_change_identity():
    original = torch.get_float32_matmul_precision()
    before = identity.runtime_implementation()
    try:
        torch.set_float32_matmul_precision(
            "high" if original == "highest" else "highest"
        )
        assert identity.runtime_implementation() != before
    finally:
        torch.set_float32_matmul_precision(original)


def test_runtime_identity_ignores_lazy_import_presence_but_binds_loaded_version(
    monkeypatch,
):
    installed = dict(identity._installed_versions())["tokenizers"]
    monkeypatch.delitem(sys.modules, "tokenizers", raising=False)
    before = identity.runtime_implementation()
    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(__version__=installed)
    )
    assert identity.runtime_implementation() == before
    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(__version__="different")
    )
    assert identity.runtime_implementation() != before


def test_unreadable_model_source_fails_closed(monkeypatch):
    def unreadable(value):
        raise OSError("source unavailable")

    monkeypatch.setattr(identity, "_source", unreadable)
    with pytest.raises(OSError, match="source unavailable"):
        identity.model_implementation(torch.nn.Linear(2, 2))


def test_package_revision_matches_experiment_provenance():
    from stateset_agents.evaluation.implementation import package_implementation

    assert identity.package_revision() == identity._digest(package_implementation())


def test_package_revision_observes_rewrites_additions_and_deletions(tmp_path):
    path = tmp_path / "__init__.py"
    path.write_text("a = 1\n")
    root = str(tmp_path)

    def revision():
        return identity._package_revision(root)

    first = revision()
    original = path.stat()
    path.write_text("a = 2\n")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    second = revision()
    assert second != first  # Same length/mtime never substitutes for reading bytes.
    extra = tmp_path / "added.py"
    extra.write_text("b = 3\n")
    assert revision() != second
    extra.unlink()
    assert revision() == second
    path.unlink()
    with pytest.raises(ValueError, match="requires package sources"):
        revision()

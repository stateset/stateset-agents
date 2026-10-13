"""Native entry points reject unusable runtimes before initializing resources."""

import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.remote import river_runtime as runtime
from stateset_agents.training import river_refund as native

MODELS = (
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "Qwen/Qwen3.8-27B-FP8",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "zai-org/GLM-5.3-Flash",
)


@pytest.mark.parametrize("model", MODELS + ("private/unlisted-model",))
def test_model_access_uses_account_capabilities_not_static_catalog(model):
    client = SimpleNamespace(get_capabilities=lambda: [model, model])
    assert runtime.require_model_access(client, model) == (model,)


@pytest.mark.parametrize(
    "models", [None, "base", {"base": True}, [None], [True], [" base"], [""]]
)
def test_malformed_capabilities_fail_closed(models):
    client = SimpleNamespace(get_capabilities=lambda: models)
    with pytest.raises(runtime.RemoteExecutionError, match="malformed"):
        runtime.require_model_access(client, "base")


@pytest.mark.parametrize("models", [[], ["other"]])
def test_unavailable_model_is_rejected(models):
    with pytest.raises(runtime.RemoteExecutionError, match="does not advertise"):
        runtime.require_model_access(
            SimpleNamespace(get_capabilities=lambda: models), "base"
        )


def test_capability_failure_does_not_expose_sdk_error_text():
    def fail():
        raise ConnectionError("Authorization: rv_private_key")

    with pytest.raises(runtime.RemoteExecutionError, match="ConnectionError") as error:
        runtime.require_model_access(SimpleNamespace(get_capabilities=fail), "base")
    assert "rv_private_key" not in str(error.value)


def test_old_client_without_capabilities_is_rejected():
    with pytest.raises(runtime.RemoteExecutionError, match="get_capabilities"):
        runtime.require_model_access(object(), "base")


def test_native_model_denial_closes_client_before_renderer_or_session(
    tmp_path, monkeypatch
):
    checks = ready_checks()
    monkeypatch.setattr(runtime, "inspect_native_runtime", lambda: checks)
    monkeypatch.setenv("RIVER_API_KEY", "rv_test")
    events = []

    class Client:
        def __init__(self, **kwargs):
            events.append("client")

        def get_capabilities(self):
            events.append("capabilities")
            return []

        def close(self):
            events.append("closed")

    monkeypatch.setitem(sys.modules, "river_client", SimpleNamespace(Client=Client))
    monkeypatch.setitem(
        sys.modules,
        "river_client.renderers",
        SimpleNamespace(
            get_renderer=lambda *a, **kw: pytest.fail("must not load renderer")
        ),
    )
    args = SimpleNamespace(
        output=tmp_path,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=True,
        dry_run=False,
    )
    splits = {"test": [{"order_id": "A", "amount_cents": 100, "eligible": True}]}
    native.prepare_run(args, splits)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(runtime.RemoteExecutionError, match="does not advertise"):
        native.execute_run(args, splits)
    assert events == ["client", "capabilities", "closed"]
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def ready_checks():
    return {
        "python": {"version": "3.12.12", "passed": True},
        "river_sdk": {"version": "0.11.0", "passed": True},
        "credentials": {"configured": True},
    }


@pytest.mark.parametrize("failure", ["python", "river_sdk", "credentials", "all"])
def test_native_command_rejects_local_failures_before_resource_initialization(
    failure, tmp_path, monkeypatch
):
    checks = ready_checks()
    expected = []
    if failure in ("python", "all"):
        checks["python"]["passed"] = False
        expected.append("Python 3.12")
    if failure in ("river_sdk", "all"):
        checks["river_sdk"].update(passed=False, reason="Unsupported SDK")
        expected.append("Unsupported SDK")
    if failure in ("credentials", "all"):
        checks["credentials"]["configured"] = False
        expected.append("RIVER_API_KEY")
    monkeypatch.setattr(runtime, "inspect_native_runtime", lambda: checks)
    # Any attempt to reach renderer/client imports is a failure, even when
    # River happens to be installed in the test interpreter.
    monkeypatch.setitem(sys.modules, "river_client", None)
    args = SimpleNamespace(
        dry_run=False,
        output=tmp_path,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=True,
    )
    splits = {"test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}]}
    native.prepare_run(args, splits)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(RuntimeError, match="Native River runtime is not ready") as exc:
        native.execute_run(args, splits)
    assert all(message in str(exc.value) for message in expected)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_ready_runtime_allows_execution(monkeypatch):
    monkeypatch.setattr(runtime, "inspect_native_runtime", ready_checks)
    runtime.require_native_runtime()


def test_native_dry_run_never_probes_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "inspect_native_runtime",
        lambda: pytest.fail("Dry runs must not inspect optional SDKs"),
    )
    monkeypatch.setitem(sys.modules, "river_client", None)
    native.execute_run(
        SimpleNamespace(
            dry_run=True,
            output=tmp_path,
            base_model="base",
            seed=42,
            steps=2,
            concurrency=8,
            max_staleness=0,
            learning_rate=1e-5,
            checkpoint=None,
            evaluate_only=True,
        ),
        {"test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}]},
    )


def test_python_compatibility_is_checked_without_sdk_import(monkeypatch):
    monkeypatch.setattr(
        runtime, "sys", SimpleNamespace(version="3.11.9", version_info=(3, 11, 9))
    )

    def absent(name):
        raise runtime.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.metadata, "version", absent)
    checks = runtime.inspect_native_runtime()
    assert not checks["python"]["passed"]
    assert not checks["river_sdk"]["passed"]


@pytest.fixture
def preflight_client(monkeypatch):
    """A client with only access-query and cleanup operations available."""
    events = []

    class Client:
        models = list(MODELS)
        failure = None

        def __init__(self, **kwargs):
            events.append(("client", kwargs))
            if self.failure == "constructor":
                raise ConnectionError("Authorization: rv_private_key")

        def get_capabilities(self):
            events.append("capabilities")
            if self.failure == "query":
                raise ConnectionError("Authorization: rv_private_key")
            return self.models

        def close(self):
            events.append("closed")
            if self.failure == "cleanup":
                raise RuntimeError("Authorization: rv_private_key")

    monkeypatch.setenv("RIVER_API_KEY", "rv_private_key")
    monkeypatch.setattr(runtime, "inspect_native_runtime", ready_checks)
    monkeypatch.setitem(sys.modules, "river_client", SimpleNamespace(Client=Client))
    return Client, events


def test_preflight_offline_does_not_infer_model_access(preflight_client):
    _, events = preflight_client
    report = runtime.river_preflight(MODELS)
    assert report["passed"] is True
    assert report["mode"] == "local"
    assert report["account"] == {"status": "not_checked", "available_models": None}
    assert report["model_access"] == dict.fromkeys(MODELS)
    assert report["request_timeout_seconds"] is None
    assert events == []


def test_preflight_live_checks_all_models_once_and_closes(preflight_client):
    client, events = preflight_client
    client.models += ["private/unlisted-model", MODELS[0]]
    requested = (*MODELS, "private/unlisted-model", MODELS[0])
    report = runtime.river_preflight(requested, live=True, timeout_seconds=3)
    assert report["passed"] is True
    assert report["schema_version"] == 1
    assert report["account"]["status"] == "checked"
    assert report["account"]["available_models"] == list(dict.fromkeys(requested))
    assert report["model_access"] == dict.fromkeys(requested, True)
    assert report["billable_resources_created"] == 0
    assert "training_correctness" in report["unverified"]
    assert "hosted_chat_and_streaming" in report["unverified"]
    assert events == [
        (
            "client",
            {"api_key": "rv_private_key", "timeout": 3.0, "enable_retries": False},
        ),
        "capabilities",
        "closed",
    ]
    assert "rv_private_key" not in json.dumps(report, allow_nan=False)


def test_preflight_denied_model_preserves_successful_access_evidence(preflight_client):
    _, events = preflight_client
    report = runtime.river_preflight([MODELS[0], "missing/model"], live=True)
    assert report["passed"] is False
    assert report["account"]["status"] == "checked"
    assert report["model_access"] == {MODELS[0]: True, "missing/model": False}
    assert "does not advertise" in report["issues"][0]
    assert events[-1] == "closed"


def test_preflight_empty_account_does_not_invent_access(preflight_client):
    client, _ = preflight_client
    client.models = []
    report = runtime.river_preflight(live=True)
    assert report["passed"] is True  # No particular model was requested.
    assert report["account"] == {"status": "checked", "available_models": []}
    assert report["model_access"] == {}
    assert runtime.river_preflight(MODELS, live=True)["passed"] is False


@pytest.mark.parametrize("failure", ["python", "river_sdk", "credentials"])
@pytest.mark.parametrize("live", [False, True])
def test_preflight_local_failure_prevents_client_creation(
    preflight_client, monkeypatch, failure, live
):
    _, events = preflight_client
    checks = ready_checks()
    if failure == "credentials":
        checks[failure]["configured"] = False
    else:
        checks[failure].update(passed=False, reason="Unsupported SDK")
    monkeypatch.setattr(runtime, "inspect_native_runtime", lambda: checks)
    report = runtime.river_preflight(MODELS, live=live)
    assert report["passed"] is False
    assert report["account"]["status"] == "not_checked"
    assert report["model_access"] == dict.fromkeys(MODELS)
    assert events == []


@pytest.mark.parametrize("models", [None, "base", [None], [" base"], [""]])
def test_preflight_malformed_response_closes_client(preflight_client, models):
    client, events = preflight_client
    client.models = models
    report = runtime.river_preflight(MODELS, live=True)
    assert report["passed"] is False
    assert report["account"]["status"] == "failed"
    assert report["model_access"] == dict.fromkeys(MODELS)
    assert "malformed" in report["issues"][0]
    assert events[-1] == "closed"


@pytest.mark.parametrize("failure", ["constructor", "query", "cleanup"])
def test_preflight_failures_are_sanitized_and_cleanup_is_attempted(
    preflight_client, failure
):
    client, events = preflight_client
    client.failure = failure
    report = runtime.river_preflight(MODELS, live=True)
    assert report["passed"] is False
    assert "rv_private_key" not in json.dumps(report)
    if failure == "constructor":
        assert len(events) == 1
    else:
        assert events[-1] == "closed"
    if failure == "cleanup":
        assert report["account"]["status"] == "checked"
        assert report["model_access"] == dict.fromkeys(MODELS, True)
        assert "cleanup failed" in report["issues"][0]
    else:
        assert report["account"]["status"] == "failed"
        assert report["model_access"] == dict.fromkeys(MODELS)


@pytest.mark.parametrize(
    "options",
    [
        {"base_models": value}
        for value in (None, "base", b"base", [None], [""], [" base"], {"base"})
    ]
    + [
        {"timeout_seconds": value}
        for value in (0, -1, float("nan"), float("inf"), True, "15")
    ]
    + [{"live": value} for value in (1, None, "yes")],
)
def test_preflight_invalid_options_fail_before_sdk_inspection(monkeypatch, options):
    monkeypatch.setattr(
        runtime, "inspect_native_runtime", lambda: pytest.fail("must validate first")
    )
    with pytest.raises(ValueError):
        runtime.river_preflight(**options)

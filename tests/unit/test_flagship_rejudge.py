"""Tests for the flagship rejudge tool (no network, no GPU, no spend)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.evaluation import support_judge

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"


def _load(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _name in ("backend_conformance", "dataset_content", "runpod_backend_conformance"):
    _load(_name)
rejudge = _load("flagship_rejudge")

A100 = "NVIDIA A100-SXM4-80GB"
H100 = "NVIDIA H100 80GB HBM3"
IDS = ("eval-1", "eval-2")


def _catalog(a100: float = 1.79, h100: float = 3.99) -> list[dict[str, Any]]:
    def row(name: str, price: float) -> dict[str, Any]:
        return {
            "id": name,
            "secureCloud": True,
            "communityCloud": True,
            "securePrice": price,
            "communityPrice": price,
        }

    return [row(A100, a100), row(H100, h100), row("NVIDIA RTX A6000", 0.3)]


def _scenario(item_id: str) -> Any:
    return SimpleNamespace(
        id=item_id,
        user_turns=["Can you reimburse the duties?"],
        policy_facts=["Duties are the customer's responsibility."],
        expected_resolution="Decline reimbursement of customs duties.",
        must_avoid=["reimburse"],
        intent="shipping",
        product="Jacket",
        to_scenario=lambda: {"must_avoid": ["reimburse"]},
    )


def _input_dir(tmp_path: Path, ids: tuple[str, ...] = IDS) -> Path:
    root = tmp_path / "seed-42"
    (root / "transcripts").mkdir(parents=True)
    for group, reply in (
        ("baseline", "Hi [Customer's Name], I'll reimburse you."),
        ("final", "Sorry, duties are the customer's responsibility."),
    ):
        rows = [
            {
                "id": item_id,
                "transcript": [
                    ["user", "Will you cover the duties?"],
                    ["assistant", reply],
                ],
            }
            for item_id in ids
        ]
        (root / "transcripts" / f"{group}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root


def _base_argv(tmp_path: Path, input_dir: Path, *extra: str) -> list[str]:
    return [
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "out"),
        "--lease-dir",
        str(tmp_path / "leases"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--harness-revision",
        "e" * 40,
        "--poll-seconds",
        "0",
        *extra,
    ]


class ForbiddenApi:
    created = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ForbiddenApi.created += 1
        raise AssertionError("RunPod API must not be constructed")


class FakeApi:
    api_key = "test-key"
    root = "https://rest.example.invalid/v1"

    def __init__(self, price: float = 1.79, terminate_error: bool = False) -> None:
        self.price = price
        self.terminate_error = terminate_error
        self.created: list[dict[str, Any]] = []
        self.terminated: list[str] = []

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": "pod-1", "costPerHr": self.price}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return {
            "id": pod_id,
            "desiredStatus": "RUNNING",
            "publicIp": "203.0.113.5",
            "portMappings": {"22": 22022},
            "costPerHr": self.price,
        }

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)
        if self.terminate_error:
            raise RuntimeError("terminate failed")


class FakeSsh:
    def __init__(
        self,
        summary: dict[str, Any] | None,
        *,
        exit_code: str = "0",
        fail_on: str | None = None,
        raise_on: str | None = None,
    ) -> None:
        self.summary = summary
        self.exit_code = exit_code
        self.fail_on = fail_on
        self.raise_on = raise_on
        self.commands: list[str] = []
        self.uploads: list[tuple[Path, str]] = []

    def wait_until_reachable(self, host: str, port: int, timeout_s: int) -> None:
        return None

    def upload(self, local: Path, remote: str) -> None:
        self.uploads.append((Path(local), remote))

    def upload_secret(self, secret: str, remote: str) -> None:
        return None

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        if self.raise_on and self.raise_on in command:
            raise ValueError("unexpected transport bug")
        if self.fail_on and self.fail_on in command:
            return 1, "boom"
        if command.startswith("cat /workspace/rejudge.exit"):
            return 0, self.exit_code
        return 0, ""

    def download_dir(self, remote_dir: str, local_dir: Path) -> list[Path]:
        local_dir.mkdir(parents=True, exist_ok=True)
        if self.summary is not None:
            (local_dir / "judge").mkdir(exist_ok=True)
            (local_dir / "judge" / "baseline.jsonl").write_text("{}\n")
            (local_dir / rejudge.SUMMARY_NAME).write_text(json.dumps(self.summary))
        return [p for p in local_dir.rglob("*") if p.is_file()]


@pytest.fixture
def ssh_key(tmp_path: Path) -> Path:
    key = tmp_path / "id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAA test")
    return key


def _run(
    tmp_path: Path,
    api: Any,
    ssh: Any,
    ssh_key: Path,
    *extra: str,
    environ: dict[str, str] | None = None,
    published: bool = True,
) -> int:
    env = {rejudge.ENV_CEILING: "5", "RUNPOD_API_KEY": "test-key", **(environ or {})}
    return rejudge.main(
        _base_argv(
            tmp_path, _input_dir(tmp_path), "--ssh-public-key", str(ssh_key), *extra
        ),
        environ=env,
        catalog_fetcher=_catalog,
        api_factory=lambda key: api,
        ssh_factory=lambda key_path=None: ssh,
        published_check=lambda root, revision: published,
    )


def _ledger(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _provider(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "out" / rejudge.PROVIDER_RECORD).read_text())


# --------------------------------------------------------------------------- #
# Refusal and planning
# --------------------------------------------------------------------------- #


def test_refuses_without_spend_ceiling(tmp_path: Path) -> None:
    ForbiddenApi.created = 0
    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path)),
        environ={"RUNPOD_API_KEY": "test-key"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
        published_check=lambda root, revision: True,
    )
    assert code == 2
    assert ForbiddenApi.created == 0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "abc", "26"])
def test_invalid_or_excessive_ceiling_refuses(tmp_path: Path, value: str) -> None:
    ForbiddenApi.created = 0
    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path)),
        environ={rejudge.ENV_CEILING: value, "RUNPOD_API_KEY": "k"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
        published_check=lambda root, revision: True,
    )
    assert code == 2 and ForbiddenApi.created == 0


def test_worst_case_above_ceiling_refuses_before_api(tmp_path: Path) -> None:
    ForbiddenApi.created = 0
    # 1.5 h at $1.79/h = $2.685 > $2.
    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path), "--confirm-max-cost-usd", "2"),
        environ={"RUNPOD_API_KEY": "k"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
        published_check=lambda root, revision: True,
    )
    assert code == 2 and ForbiddenApi.created == 0


def test_unpublished_harness_refuses_before_api(tmp_path: Path) -> None:
    ForbiddenApi.created = 0
    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path)),
        environ={rejudge.ENV_CEILING: "5", "RUNPOD_API_KEY": "k"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
        published_check=lambda root, revision: False,
    )
    assert code == 2 and ForbiddenApi.created == 0


def test_missing_or_mismatched_transcripts_refuse(tmp_path: Path) -> None:
    input_dir = _input_dir(tmp_path)
    (input_dir / "transcripts" / "final.jsonl").write_text(
        json.dumps({"id": "other", "transcript": [["user", "x"]]}) + "\n"
    )
    code = rejudge.main(
        _base_argv(tmp_path, input_dir, "--plan"),
        environ={},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 2
    (input_dir / "transcripts" / "final.jsonl").unlink()
    with pytest.raises(rejudge.RejudgeError, match="missing transcripts"):
        rejudge.load_transcripts(input_dir)


def test_plan_picks_cheapest_80gb_gpu_and_is_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ForbiddenApi.created = 0
    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path), "--plan"),
        environ={},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 0 and ForbiddenApi.created == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["gpu_type_id"] == A100  # the RTX A6000 is never a candidate
    assert plan["worst_case_cost_usd"] == pytest.approx(1.79 * 1.5)
    assert plan["max_cost_usd"] is None
    assert plan["provisions_hardware"] is False


def test_choose_gpu_prefers_cheaper_h100_and_breaks_ties_by_order() -> None:
    candidates = rejudge.DEFAULT_GPU_CANDIDATES
    assert rejudge.choose_gpu(_catalog(a100=3.0, h100=2.5), "SECURE", candidates) == (
        H100,
        2.5,
    )
    assert (
        rejudge.choose_gpu(_catalog(a100=2.0, h100=2.0), "SECURE", candidates)[0]
        == A100
    )
    with pytest.raises(rejudge.RejudgeError):
        rejudge.choose_gpu(_catalog(), "SECURE", ["NVIDIA B200"])


def test_dry_run_is_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ForbiddenApi.created = 0

    def no_catalog() -> list[dict[str, Any]]:
        raise AssertionError("dry run must not fetch the catalog")

    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path), "--dry-run"),
        environ={},
        catalog_fetcher=no_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 0 and ForbiddenApi.created == 0
    preview = json.loads(capsys.readouterr().out)
    launch = preview["commands"]["launch"]
    assert "benchmarks/flagship_rejudge.py --local" in launch
    assert preview["pins"]["judge_model"] == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert "flagship_workload.py" not in launch  # never trains
    assert preview["provisions_hardware"] is False


# --------------------------------------------------------------------------- #
# Paid path (fakes) -- termination on every path
# --------------------------------------------------------------------------- #


def _summary() -> dict[str, Any]:
    return {"kind": rejudge.SUMMARY_KIND, "judge_self_disagreement": 0.03}


def test_complete_remote_run(tmp_path: Path, ssh_key: Path) -> None:
    api, ssh = FakeApi(), FakeSsh(_summary())
    assert _run(tmp_path, api, ssh, ssh_key) == 0
    assert api.created[0]["gpu_type_id"] == A100
    assert api.terminated == ["pod-1"]
    assert sorted(
        remote for _local, remote in ssh.uploads if "rejudge-input" in remote
    ) == [
        "/workspace/rejudge-input/transcripts/baseline.jsonl",
        "/workspace/rejudge-input/transcripts/final.jsonl",
    ]
    assert any("checkout --detach " + "e" * 40 in c for c in ssh.commands)
    provider = _provider(tmp_path)
    assert provider["status"] == "completed"
    assert provider["termination_confirmed"] is True
    assert provider["cost_usd"] > 0 and provider["cost_source"] == "runpod-api"
    assert _ledger(tmp_path)[0]["status"] == "completed"
    assert not list((tmp_path / "leases").glob("*.json"))


@pytest.mark.parametrize(
    "ssh_kwargs",
    [
        {"summary": _summary(), "exit_code": "1"},
        {"summary": None},
        {"summary": _summary(), "fail_on": "git clone"},
        {"summary": _summary(), "fail_on": "rm -f /workspace/rejudge.exit"},
        {"summary": _summary(), "fail_on": "self-destruct"},
        {"summary": _summary(), "raise_on": "tail -n 1"},
    ],
)
def test_failures_terminate_and_never_complete(
    tmp_path: Path, ssh_key: Path, ssh_kwargs: dict[str, Any]
) -> None:
    summary = ssh_kwargs.pop("summary")
    api, ssh = FakeApi(), FakeSsh(summary, **ssh_kwargs)
    assert _run(tmp_path, api, ssh, ssh_key) == 1
    assert api.terminated == ["pod-1"]
    provider = _provider(tmp_path)
    assert provider["status"] == "failed" and provider["error"]
    assert provider["termination_confirmed"] is True
    assert _ledger(tmp_path)[0]["status"] == "failed"


def test_post_allocation_price_over_budget_terminates(
    tmp_path: Path, ssh_key: Path
) -> None:
    api, ssh = FakeApi(price=9.0), FakeSsh(_summary())
    assert _run(tmp_path, api, ssh, ssh_key) == 1
    assert api.terminated == ["pod-1"]
    assert ssh.commands == []  # nothing ran on the over-budget pod
    assert "above the --max-cost ceiling" in _provider(tmp_path)["error"]


def test_unconfirmed_termination_is_an_error(tmp_path: Path, ssh_key: Path) -> None:
    api, ssh = FakeApi(terminate_error=True), FakeSsh(_summary())
    assert _run(tmp_path, api, ssh, ssh_key) == 2
    assert api.terminated == ["pod-1"]
    provider = _provider(tmp_path)
    assert provider["termination_confirmed"] is False
    assert provider["cleanup_lease_retained"] is True
    assert _ledger(tmp_path)[0]["status"] == "cleanup-pending"


# --------------------------------------------------------------------------- #
# Local mode: the real judge phase with a fake model
# --------------------------------------------------------------------------- #


def test_local_mode_rejudges_with_current_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_vllm(model: str, revision: Any) -> Any:
        calls.append((model, revision))

        def generate(prompts: list[str]) -> list[str]:
            out = []
            for prompt in prompts:
                pass_two = "[A] Criterion" in prompt
                checks = dict.fromkeys(support_judge.ALL_CHECKS, True)
                checks["resolution_next_step"] = not pass_two
                out.append(json.dumps({"checks": checks}))
            return out

        return generate

    monkeypatch.setattr(support_judge, "vllm_generate_fn", fake_vllm)
    input_dir = _input_dir(tmp_path)
    loads: list[tuple[str, str | None, str]] = []

    def load_dataset(source: str, revision: str | None, split: str) -> list[Any]:
        loads.append((source, revision, split))
        return [_scenario(i) for i in (*IDS, "eval-3")]

    args = rejudge.build_parser().parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--local",
            "--dataset-revision",
            "b" * 40,
        ]
    )
    pins = rejudge.resolve_pins(args)
    summary = rejudge.run_local(
        args,
        pins,
        load_dataset=load_dataset,
        snapshot=lambda model, revision: f"/snap/{revision}",
    )
    assert loads == [("stateset/customer-support-multiturn-v1", "b" * 40, "eval")]
    assert calls == [(f"/snap/{pins['judge_revision']}", None)]
    assert summary["rubric_revision"] == support_judge.rubric_revision()
    baseline = summary["groups"]["baseline"]
    assert baseline["findings_summary"] == {
        "must_avoid_hits": 2,
        "placeholders": 2,
        "repeated_turns": 0,
        "long_turns": 0,
    }
    assert baseline["forced_false_by_check"] == {
        "accuracy_no_placeholder": 2,
        "policy_no_must_avoid_phrase": 2,
    }
    assert baseline["judge_overrides_by_check"] == {
        "accuracy_no_placeholder": 4,
        "policy_no_must_avoid_phrase": 4,
    }
    # One resolution check flips between passes: 1/5 of one criterion of five.
    assert summary["judge_self_disagreement"] == pytest.approx(0.04)
    assert summary["self_disagreement_within_gate"] is True
    assert summary["groups"]["final"]["mean_score"] == pytest.approx(0.98)
    rows = (tmp_path / "out" / "judge" / "final.jsonl").read_text().splitlines()
    assert [json.loads(r)["id"] for r in rows] == ["final/eval-1", "final/eval-2"]
    written = json.loads((tmp_path / "out" / rejudge.SUMMARY_NAME).read_text())
    assert written["transcripts_sha256"] == rejudge.transcript_digests(input_dir)


def test_local_mode_rejects_transcripts_from_another_dataset(tmp_path: Path) -> None:
    args = rejudge.build_parser().parse_args(
        [
            "--input-dir",
            str(_input_dir(tmp_path, ids=("x-1", "x-2"))),
            "--output-dir",
            str(tmp_path / "out"),
            "--local",
        ]
    )
    with pytest.raises(rejudge.RejudgeError, match="first 2 eval scenarios"):
        rejudge.run_local(
            args,
            rejudge.resolve_pins(args),
            load_dataset=lambda *a: [_scenario(i) for i in IDS],
            snapshot=lambda *a: "/snap",
            phase=lambda payload: pytest.fail("judge must not run"),
        )


def test_main_local_mode_never_touches_runpod(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ForbiddenApi.created = 0
    seen: list[Any] = []

    def runner(args: Any, pins: Any) -> dict[str, Any]:
        seen.append(pins)
        return {"judge_self_disagreement": 0.01, "rubric_revision": "r"}

    code = rejudge.main(
        _base_argv(tmp_path, _input_dir(tmp_path), "--local"),
        environ={},
        api_factory=ForbiddenApi,
        local_runner=runner,
    )
    assert code == 0 and ForbiddenApi.created == 0 and len(seen) == 1
    assert json.loads(capsys.readouterr().out)["judge_self_disagreement"] == 0.01

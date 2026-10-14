"""Flagship v2 contract tests: manifests, driver, rejudge, matrix (CPU only).

v2 pins Qwen/Qwen3.5-9B with a google/gemma-4-31B-it judge next to the
unchanged v1 protocol. Nothing here provisions hardware or touches a network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

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
workload = _load("flagship_workload")
driver = _load("flagship_runpod_driver")
rejudge = _load("flagship_rejudge")
runner = _load("run_flagship_matrix")

V2_MANIFESTS = ("flagship_v2_preflight_manifest.json", "flagship_v2_manifest.json")
V1_MANIFESTS = ("flagship_preflight_manifest.json", "flagship_manifest.json")
V2_ONLY_CONFIG_KEYS = {
    "runtime_profile",
    "chat_template_kwargs",
    "forbid_think_blocks",
    "trust_remote_code",
    "merge_adapter_for_eval",
    "episode_weighting",
    "disable_dropout",
}
QWEN35_LINEAR_ATTENTION = {"in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z"}
#: Catalog price the manifest caps are sized for (H100 80GB SECURE).
H100_PRICE = 3.99


def _manifest(name: str) -> dict[str, Any]:
    return json.loads((BENCHMARKS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", V2_MANIFESTS)
def test_v2_manifest_pins_models_judge_and_dataset(name: str) -> None:
    manifest = runner.load_manifest(BENCHMARKS / name)
    assert manifest["protocol"] == "stateset-flagship-multiturn-v2"
    assert manifest["model"] == "Qwen/Qwen3.5-9B"
    assert manifest["model_revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    # The trained text decoder (the checkpoint also has a vision tower + MTP).
    assert manifest["model_parameter_count"] == 8_953_803_264
    assert manifest["dataset"] == "stateset/customer-support-multiturn-v1"
    assert manifest["dataset_revision"].startswith("db0fd73c")
    judge = manifest["judge"]
    assert judge["model"] == "google/gemma-4-31B-it"
    assert judge["revision"] == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert judge["loader"] == "gemma4"
    v1 = _manifest("flagship_manifest.json")
    assert judge["rubric_revision"] == v1["judge"]["rubric_revision"]
    assert manifest["framework_version"] == v1["framework_version"]
    workload.validate_config(manifest["config"])
    workload.validate_judge(judge, judge["rubric_revision"])


@pytest.mark.parametrize("name", V2_MANIFESTS)
def test_v2_config_is_v1_plus_the_v2_knobs(name: str) -> None:
    v2 = _manifest(name)["config"]
    v1 = _manifest(name.replace("_v2", ""))["config"]
    assert v2["runtime_profile"] == "flagship-v2"
    assert v2["chat_template_kwargs"] == {"enable_thinking": False}
    assert v2["forbid_think_blocks"] is True
    assert v2["merge_adapter_for_eval"] is True
    assert v2["trust_remote_code"] is False
    assert v2["episode_weighting"] == "per_episode"
    assert v2["disable_dropout"] is True
    changed = {k for k in v1 if v1[k] != v2[k]}
    assert changed <= {
        "lora_target_modules",
        "max_wall_clock_seconds",
        "max_cost_usd_per_seed",
    }
    assert set(v2) - set(v1) == V2_ONLY_CONFIG_KEYS


@pytest.mark.parametrize("name", V2_MANIFESTS)
def test_v2_lora_targets_cover_qwen35_linear_attention(name: str) -> None:
    targets = set(_manifest(name)["config"]["lora_target_modules"])
    assert QWEN35_LINEAR_ATTENTION | {"out_proj"} <= targets
    assert set(workload.DEFAULT_LORA_TARGET_MODULES) <= targets


@pytest.mark.parametrize("name", V2_MANIFESTS)
def test_v2_worst_case_fits_the_per_seed_cap(name: str) -> None:
    config = _manifest(name)["config"]
    lifetime = config["max_wall_clock_seconds"] - driver.LOCAL_HEADROOM_SECONDS
    worst = lifetime / 3600 * H100_PRICE
    assert worst <= config["max_cost_usd_per_seed"]


def test_v2_preflight_worst_case_is_documented() -> None:
    config = _manifest("flagship_v2_preflight_manifest.json")["config"]
    lifetime = config["max_wall_clock_seconds"] - driver.LOCAL_HEADROOM_SECONDS
    worst = round(lifetime / 3600 * H100_PRICE, 2)
    assert worst == 27.27
    doc = (BENCHMARKS / "FLAGSHIP.md").read_text(encoding="utf-8")
    assert f"${worst}" in doc
    assert "flagship_v2_preflight_manifest.json" in doc
    assert "benchmark_results/flagship_v2/preflight" in doc


@pytest.mark.parametrize("name", V1_MANIFESTS)
def test_v1_manifests_have_no_v2_knobs(name: str) -> None:
    manifest = _manifest(name)
    assert manifest["protocol"] == "stateset-flagship-multiturn-v1"
    assert manifest["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert not V2_ONLY_CONFIG_KEYS & set(manifest["config"])
    assert "loader" not in manifest["judge"]
    assert manifest["judge"]["model"] == "mistralai/Mistral-Small-24B-Instruct-2501"


def test_matrix_rejects_unknown_loader_and_profile(tmp_path: Path) -> None:
    for mutate, message in (
        (lambda m: m["judge"].update(loader="llama"), "judge.loader"),
        (lambda m: m["config"].update(runtime_profile="v9"), "runtime_profile"),
    ):
        manifest = _manifest("flagship_v2_manifest.json")
        mutate(manifest)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(runner.FlagshipError, match=message):
            runner.load_manifest(path)


def test_publication_gate_stays_at_005() -> None:
    assert runner.SELF_DISAGREEMENT_GATE == 0.05
    assert rejudge.SELF_DISAGREEMENT_GATE == 0.05


def test_flagship_v2_extra_pins_new_transformers_and_vllm() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["flagship-v2"]
    assert "transformers>=5.5.0" in extra
    assert "vllm>=0.29.0" in extra
    assert "stateset-agents[gemma4]" in extra


# --------------------------------------------------------------------------- #
# RunPod driver
# --------------------------------------------------------------------------- #


def _driver_args(name: str, **updates: Any) -> argparse.Namespace:
    manifest = _manifest(name)
    config_json = runner.canonical_json(manifest["config"])
    judge_json = runner.canonical_json(manifest["judge"])
    values = {
        "seed": 42,
        "mode": "preflight",
        "framework_version": manifest["framework_version"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "config_json": config_json,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "judge_json": judge_json,
        "judge_sha256": hashlib.sha256(judge_json.encode()).hexdigest(),
        "provider_tier": "SECURE",
        "gpu": None,
        "container_image": driver.DEFAULT_IMAGE,
        "container_disk_gb": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_driver_installs_v2_dependencies_only_for_v2() -> None:
    v2 = driver.remote_commands(
        _driver_args("flagship_v2_preflight_manifest.json"),
        "f" * 40,
        with_hf_token=False,
    )["setup"]
    v1 = driver.remote_commands(
        _driver_args("flagship_preflight_manifest.json"), "f" * 40, with_hf_token=False
    )["setup"]
    assert "'[training,vllm,flagship-v2]'" in v2
    assert "'[training,vllm]'" in v1
    assert "flagship-v2" not in v1


def test_driver_disk_follows_the_profile_unless_overridden() -> None:
    def disk(name: str, **updates: Any) -> int:
        args = _driver_args(name, **updates)
        config = json.loads(args.config_json)
        return driver.execution_settings(args, config, {})["container_disk_gb"]

    assert disk("flagship_preflight_manifest.json") == 150
    assert disk("flagship_v2_preflight_manifest.json") == 250
    assert disk("flagship_v2_preflight_manifest.json", container_disk_gb=300) == 300


def test_driver_plan_prices_v2_under_the_cap() -> None:
    args = _driver_args("flagship_v2_preflight_manifest.json")
    config = json.loads(args.config_json)
    settings = driver.execution_settings(args, config, {})
    catalog = [
        {
            "id": driver.DEFAULT_GPU,
            "secureCloud": True,
            "communityCloud": True,
            "securePrice": H100_PRICE,
            "communityPrice": 2.69,
        }
    ]
    plan = driver.build_plan(settings, catalog, config["max_cost_usd_per_seed"])
    assert plan["worst_case_cost_usd"] == pytest.approx(27.27, abs=0.01)
    assert plan["provisions_hardware"] is False


def test_driver_rejects_unknown_profile() -> None:
    args = _driver_args("flagship_v2_preflight_manifest.json")
    config = json.loads(args.config_json)
    config["runtime_profile"] = "flagship-v9"
    args.config_json = runner.canonical_json(config)
    with pytest.raises(driver.FlagshipDriverError, match="runtime_profile"):
        driver.runtime_profile(args)


def test_driver_dry_run_for_v2_is_offline(tmp_path: Path) -> None:
    args = _driver_args("flagship_v2_preflight_manifest.json")
    argv = [
        sys.executable,
        str(BENCHMARKS / "flagship_runpod_driver.py"),
        "--seed",
        "42",
        "--mode",
        "preflight",
        "--framework-version",
        args.framework_version,
        "--model",
        args.model,
        "--model-revision",
        args.model_revision,
        "--dataset-revision",
        args.dataset_revision,
        "--config-json",
        args.config_json,
        "--config-sha256",
        args.config_sha256,
        "--judge-json",
        args.judge_json,
        "--judge-sha256",
        args.judge_sha256,
        "--output",
        str(tmp_path / "adapter.json"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--harness-revision",
        "f" * 40,
        "--dry-run",
    ]
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        # No provider credentials reach the dry run (Windows needs the rest).
        env={
            k: v
            for k, v in os.environ.items()
            if k not in {"RUNPOD_API_KEY", "HF_TOKEN", driver.ENV_CEILING}
        },
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["provisions_hardware"] is False
    assert preview["settings"]["container_disk_gb"] == 250
    assert "flagship-v2" in preview["commands"]["setup"]


# --------------------------------------------------------------------------- #
# Rejudge
# --------------------------------------------------------------------------- #


def _rejudge_args(manifest: Path | None, **updates: Any) -> argparse.Namespace:
    values = {
        "manifest": manifest,
        "judge_model": None,
        "judge_revision": None,
        "dataset_revision": None,
        "dataset_source": None,
        "judge_loader": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_rejudge_reads_the_v2_judge_loader_from_the_manifest() -> None:
    pins = rejudge.resolve_pins(_rejudge_args(BENCHMARKS / "flagship_v2_manifest.json"))
    assert pins["judge_model"] == "google/gemma-4-31B-it"
    assert pins["judge_loader"] == "gemma4"
    argv = rejudge.local_argv(pins)
    assert argv[argv.index("--judge-loader") + 1] == "gemma4"
    setup = rejudge.remote_commands(pins, "f" * 40, with_hf_token=False)["setup"]
    assert "'[training,vllm,flagship-v2]'" in setup


def test_rejudge_v1_pins_are_unchanged() -> None:
    pins = rejudge.resolve_pins(_rejudge_args(BENCHMARKS / "flagship_manifest.json"))
    assert pins["judge_loader"] == "mistral"
    assert "--judge-loader" not in rejudge.local_argv(pins)
    setup = rejudge.remote_commands(pins, "f" * 40, with_hf_token=False)["setup"]
    assert "'[training,vllm]'" in setup


def test_rejudge_rejects_unknown_loader() -> None:
    with pytest.raises(rejudge.RejudgeError, match="judge_loader"):
        rejudge.resolve_pins(
            _rejudge_args(BENCHMARKS / "flagship_manifest.json", judge_loader="llama")
        )


def test_rejudge_local_passes_the_loader_to_the_judge_phase(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    (input_dir / "transcripts").mkdir(parents=True)
    for group in rejudge.GROUPS:
        (input_dir / "transcripts" / f"{group}.jsonl").write_text(
            json.dumps(
                {"id": "eval-0", "transcript": [["user", "hi"], ["assistant", "ok"]]}
            )
            + "\n",
            encoding="utf-8",
        )
    seen: dict[str, Any] = {}

    def phase(payload: Any) -> dict[str, Any]:
        seen.update(payload)
        summary = {"mean_score": 0.5, "self_disagreement": 0.01, "n": 1.0}
        return {
            "groups": {
                g: {"results": [{"id": "eval-0"}], "summary": summary}
                for g in rejudge.GROUPS
            },
            "rubric_revision": "r",
            "backend": "fake",
        }

    class Scenario:
        id = "eval-0"
        split = "eval"

    args = argparse.Namespace(
        input_dir=input_dir, output_dir=tmp_path / "out", backend="vllm"
    )
    pins = {
        "judge_model": "google/gemma-4-31B-it",
        "judge_revision": "d" * 40,
        "dataset_revision": "b" * 40,
        "dataset_source": "ds",
        "judge_loader": "gemma4",
    }
    original = workload.scenario_record
    try:
        workload.scenario_record = lambda scenario: {  # type: ignore[assignment]
            "id": "eval-0",
            "scenario": {},
            "judge_scenario": {},
        }
        summary = rejudge.run_local(
            args,
            pins,
            load_dataset=lambda source, revision, split: [Scenario()],
            snapshot=lambda model, revision: "/snap",
            phase=phase,
        )
    finally:
        workload.scenario_record = original  # type: ignore[assignment]
    assert seen["loader"] == "gemma4"
    assert summary["judge_loader"] == "gemma4"
    assert summary["self_disagreement_gate"] == 0.05


# --------------------------------------------------------------------------- #
# Judge engine wiring
# --------------------------------------------------------------------------- #


def test_vllm_generate_fn_forwards_engine_and_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    from stateset_agents.evaluation import support_judge

    engines: list[dict[str, Any]] = []
    chats: list[dict[str, Any]] = []

    class FakeOutput:
        def __init__(self) -> None:
            self.outputs = [types.SimpleNamespace(text="{}")]

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            engines.append(kwargs)

        def chat(self, conversations: Any, params: Any, **kwargs: Any) -> list[Any]:
            chats.append(kwargs)
            return [FakeOutput() for _ in conversations]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    settings = workload.JUDGE_LOADERS["gemma4"]
    generate = support_judge.vllm_generate_fn("/snap", None, **settings)
    assert generate(["p"]) == ["{}"]
    assert engines[0]["gpu_memory_utilization"] == 0.92
    assert engines[0]["limit_mm_per_prompt"] == {"image": 0}
    assert chats[0]["chat_template_kwargs"] == {"enable_thinking": False}

    # The v1 (Mistral) judge: identical engine arguments to before v2.
    support_judge.vllm_generate_fn("/snap", None)(["p"])
    assert "limit_mm_per_prompt" not in engines[1]
    assert engines[1]["gpu_memory_utilization"] == 0.85
    assert "chat_template_kwargs" not in chats[1]


# --------------------------------------------------------------------------- #
# CUDA 12.9 runtime (first v2 preflight: cu13 torch on a 12.9 driver)
# --------------------------------------------------------------------------- #


def test_v2_setup_installs_cuda129_stack_before_the_package() -> None:
    setup = driver.remote_commands(
        _driver_args("flagship_v2_preflight_manifest.json"),
        "f" * 40,
        with_hf_token=False,
    )["setup"]
    steps = setup.split(" && ")
    pre = next(i for i, s in enumerate(steps) if "download.pytorch.org/whl/cu129" in s)
    editable = next(i for i, s in enumerate(steps) if "pip install --quiet -e" in s)
    check = next(i for i, s in enumerate(steps) if "torch.cuda.init()" in s)
    assert pre < editable < check
    assert "torch==2.13.0+cu129" in steps[pre]
    assert "torchcodec==0.16.0+cu129" in steps[pre]
    assert "import torchcodec, vllm.entrypoints.llm" in driver.CUDA_CHECK
    assert "vllm-0.31.0%2Bcu129" in steps[pre] and "#sha256=" in steps[pre]
    import shlex

    assert shlex.split(steps[check])[2] == driver.CUDA_CHECK


def test_v1_setup_is_unchanged_by_the_cuda_pin() -> None:
    setup = driver.remote_commands(
        _driver_args("flagship_preflight_manifest.json"), "f" * 40, with_hf_token=False
    )["setup"]
    assert "cu129" not in setup and "torch.cuda.init()" not in setup


def test_v2_cuda_check_rejects_a_cuda13_torch() -> None:
    import types

    fake = types.SimpleNamespace(
        __version__="2.13.0",
        version=types.SimpleNamespace(cuda="13.0"),
        cuda=types.SimpleNamespace(init=lambda: None, get_device_name=lambda i: "H100"),
    )
    with pytest.raises(AssertionError, match="13.0"):
        code = driver.CUDA_CHECK.replace("import torch; ", "").replace(
            "import torchcodec, vllm.entrypoints.llm; ", ""
        )
        exec(code, {"torch": fake})
    ok = types.SimpleNamespace(
        **{**vars(fake), "version": types.SimpleNamespace(cuda="12.9")}
    )
    exec(
        driver.CUDA_CHECK.replace("import torch; ", "").replace(
            "import torchcodec, vllm.entrypoints.llm; ", ""
        ),
        {"torch": ok},
    )


def test_v2_vllm_wheel_matches_the_extra() -> None:
    assert "vllm-0.31.0" in driver.V2_VLLM_WHEEL
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "vllm>=0.29.0" in project["project"]["optional-dependencies"]["flagship-v2"]


def test_gemma_rejudge_uses_the_cuda129_stack() -> None:
    pins = rejudge.resolve_pins(_rejudge_args(BENCHMARKS / "flagship_v2_manifest.json"))
    setup = rejudge.remote_commands(pins, "f" * 40, with_hf_token=False)["setup"]
    assert "whl/cu129" in setup and "torch.cuda.init()" in setup
    v1 = rejudge.resolve_pins(_rejudge_args(BENCHMARKS / "flagship_manifest.json"))
    assert (
        "cu129"
        not in rejudge.remote_commands(v1, "f" * 40, with_hf_token=False)["setup"]
    )

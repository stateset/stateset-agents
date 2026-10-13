"""Contract tests for the verl, OpenRLHF and NeMo-RL matched-GSPO adapters.

Every framework subprocess, GPU query and model load is replaced by a fake
runtime, so these tests check the adapter logic only: CLI parsing, the
framework configuration each adapter generates for the matched protocol,
the result schema ``shootout.py`` validates, fail-closed behaviour, and the
deviations each adapter must surface. Nothing here runs a framework.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from benchmarks.adapters import _matched_gsm8k as matched
from benchmarks.adapters import framework_envs
from benchmarks.adapters import nemo_rl_gspo as nemo
from benchmarks.adapters import openrlhf_gspo as openrlhf
from benchmarks.adapters import verl_gspo as verl

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _load(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shootout = _load("shootout")
gate = _load("a_plus_gate")

V5_MANIFEST = BENCHMARKS / "shootout_manifest_v6_five_frameworks.json"
MANIFEST = json.loads(V5_MANIFEST.read_text(encoding="utf-8"))
CONFIG: dict[str, Any] = MANIFEST["config"]
VERSIONS = {item["name"]: item["version"] for item in MANIFEST["implementations"]}
SHA = "a" * 40
MAX_STEPS = int(CONFIG["max_steps"])
ADAPTERS = {"verl": verl, "openrlhf": openrlhf, "nemo-rl": nemo}


def _argv(tmp_path: Path, *extra: str, config: Mapping[str, Any] = CONFIG) -> list:
    return [
        "--task",
        "gsm8k",
        "--model",
        MANIFEST["model"],
        "--model-revision",
        MANIFEST["model_revision"],
        "--dataset-revision",
        MANIFEST["dataset_revision"],
        "--seed",
        "42",
        "--config-json",
        json.dumps(dict(config), sort_keys=True, separators=(",", ":")),
        "--artifact-dir",
        str(tmp_path / "run" / "artifact"),
        "--adapter-output",
        str(tmp_path / "run" / "adapter-result.json"),
        "--work-dir",
        str(tmp_path / "work"),
        *extra,
    ]


def _write_adapter(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32}), encoding="utf-8"
    )
    (directory / "adapter_model.safetensors").write_bytes(b"lora-weights")


def _flag(command: Sequence[str], flag: str) -> str:
    return command[list(command).index(flag) + 1]


class FakeFramework:
    """Stand-in for a framework subprocess that writes realistic outputs."""

    def __init__(
        self,
        framework: str,
        *,
        steps: int = MAX_STEPS,
        reward: float = 0.25,
        returncode: int = 0,
        write_metrics: bool = True,
    ) -> None:
        self.framework = framework
        self.steps = steps
        self.reward = reward
        self.returncode = returncode
        self.write_metrics = write_metrics
        self.commands: list[list[str]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: int,
        cwd: Path | None = None,
    ) -> matched.CommandOutcome:
        command = list(command)
        self.commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.returncode == 0:
            getattr(self, "_" + self.framework.replace("-", "_"))(
                command, env, log_path
            )
        return matched.CommandOutcome(self.returncode, 12.0, peak_device_mb=30000.0)

    def _verl(self, command: list[str], env: Mapping[str, str], log: Path) -> None:
        if "verl.model_merger" in command:
            _write_adapter(Path(_flag(command, "--target_dir")) / "lora_adapter")
            return
        overrides = dict(part.split("=", 1) for part in command[3:])
        root = Path(overrides["trainer.default_local_dir"].strip("'"))
        (root / f"global_step_{self.steps}" / "actor").mkdir(parents=True)
        if self.write_metrics:
            with Path(env["VERL_FILE_LOGGER_PATH"]).open("w") as stream:
                for step in range(1, self.steps + 1):
                    data = {
                        "training/global_step": step,
                        "critic/score/mean": self.reward,
                        "perf/max_memory_allocated_gb": 20.0,
                        "perf/throughput": 900.0,
                    }
                    stream.write(json.dumps({"step": step, "data": data}) + "\n")

    def _openrlhf(self, command: list[str], env: Mapping[str, str], log: Path) -> None:
        _write_adapter(Path(_flag(command, "--ckpt.output_dir")))
        if self.write_metrics:
            with log.open("a") as stream:
                for step in range(1, self.steps + 1):
                    status = {
                        "rollout/reward_mean": self.reward,
                        "rollout/reward_std": 0.4,
                        "timing/generation": 1.5,
                    }
                    stream.write(f"INFO ✨ Global step {step}: {status}\n")

    def _nemo_rl(self, command: list[str], env: Mapping[str, str], log: Path) -> None:
        payload = json.loads(Path(_flag(command, "--config")).read_text())
        checkpoint = Path(payload["checkpointing"]["checkpoint_dir"])
        _write_adapter(checkpoint / f"step_{self.steps}" / "policy" / "weights")
        if self.write_metrics:
            with Path(_flag(command, "--metrics-jsonl")).open("w") as stream:
                for step in range(1, self.steps + 1):
                    record = {
                        "step": step,
                        "prefix": "train",
                        "metrics": {
                            "reward": self.reward,
                            "valid_tokens_per_sec_per_gpu": 1000.0,
                        },
                    }
                    stream.write(json.dumps(record) + "\n")


def _runtime(
    framework: str,
    fake: FakeFramework,
    *,
    version: str | None = None,
    evaluations: list[dict[str, Any]] | None = None,
) -> matched.Runtime:
    examples = [
        SimpleNamespace(question=f"What is {i} + 1?", gold_answer=float(i + 1))
        for i in range(8)
    ]
    calls = evaluations if evaluations is not None else []

    def evaluate(
        model: str,
        revision: str,
        items: Sequence[Any],
        *,
        max_tokens: int,
        adapter_dir: Path | None = None,
    ) -> matched.EvalOutcome:
        calls.append({"adapter_dir": adapter_dir, "max_tokens": max_tokens})
        return matched.EvalOutcome(score=0.5 if adapter_dir else 0.4, peak_vram_mb=9000)

    return matched.Runtime(
        cuda_info=lambda: {"gpu": "NVIDIA L40S", "gpu_count": 1, "cuda": "12.8"},
        probe_framework=lambda command, env: {
            "version": version or VERSIONS[framework],
            "cuda": "13.0",
        },
        load_examples=lambda n_train, n_eval, revision: (examples, examples[:4]),
        resolve_model=lambda model, revision: Path("/models/qwen-snapshot"),
        evaluate=evaluate,
        run_command=fake,
        set_seeds=lambda seed: None,
    )


@pytest.fixture(autouse=True)
def _no_parquet(monkeypatch: pytest.MonkeyPatch) -> list:
    """verl writes parquet via ``datasets``; record rows instead."""
    written: list = []

    def fake(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(rows)), encoding="utf-8")
        written.append((path, list(rows)))
        return path

    monkeypatch.setattr(verl, "_rows_to_parquet", fake)
    return written


def _validate(result_path: Path, framework: str) -> dict[str, Any]:
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    return shootout.validate_adapter_result(
        raw,
        MANIFEST["hardware"],
        shootout.canonical_digest(MANIFEST["config"]),
        VERSIONS[framework],
        result_path,
    )


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("framework", sorted(ADAPTERS))
def test_cli_requires_the_trl_contract(framework: str, tmp_path: Path) -> None:
    module = ADAPTERS[framework]
    argv = _argv(tmp_path)
    without_seed = argv[: argv.index("--seed")] + argv[argv.index("--seed") + 2 :]
    with pytest.raises(SystemExit) as excinfo:
        module.main(without_seed, runtime=_runtime(framework, FakeFramework(framework)))
    assert excinfo.value.code == 2


@pytest.mark.parametrize("framework", sorted(ADAPTERS))
def test_cli_rejects_short_revisions_and_foreign_objectives(
    framework: str, tmp_path: Path
) -> None:
    module = ADAPTERS[framework]
    runtime = _runtime(framework, FakeFramework(framework))
    argv = _argv(tmp_path)
    argv[argv.index("--model-revision") + 1] = "main"
    with pytest.raises(SystemExit):
        module.main(argv, runtime=runtime)
    token_level = {**CONFIG, "importance_sampling_level": "token"}
    with pytest.raises(SystemExit):
        module.main(_argv(tmp_path, config=token_level), runtime=runtime)
    unknown = {**CONFIG, "extra_field": 1}
    with pytest.raises(SystemExit):
        module.main(_argv(tmp_path, config=unknown), runtime=runtime)
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_cli_accepts_framework_locators(tmp_path: Path) -> None:
    parser = matched.build_parser("x")
    args = parser.parse_args(
        _argv(tmp_path, "--framework-python", "/v/bin/python", "--keep-work-dir")
    )
    assert args.framework_python == "/v/bin/python"
    assert args.keep_work_dir is True
    assert args.task == "gsm8k" and args.seed == 42
    assert matched.parse_config(parser, args) == CONFIG


# ---------------------------------------------------------------------------
# Generated framework configuration encodes the matched protocol
# ---------------------------------------------------------------------------


def test_verl_overrides_encode_matched_gspo(tmp_path: Path) -> None:
    overrides = verl.build_overrides(
        CONFIG,
        seed=1337,
        model_path=Path("/models/qwen"),
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        reward_path=Path(matched.__file__),
        checkpoint_dir=tmp_path / "ckpt",
    )
    assert overrides["actor_rollout_ref.actor.policy_loss.loss_mode"] == "gspo"
    assert overrides["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"
    assert overrides["actor_rollout_ref.actor.clip_ratio_low"] == 3e-4
    assert overrides["actor_rollout_ref.actor.clip_ratio_high"] == 4e-4
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] is False
    assert overrides["actor_rollout_ref.actor.kl_loss_coef"] == 0.0
    assert overrides["algorithm.use_kl_in_reward"] is False
    assert overrides["algorithm.adv_estimator"] == "grpo"
    assert overrides["algorithm.norm_adv_by_std_in_grpo"] is True
    assert overrides["actor_rollout_ref.model.lora_rank"] == 16
    assert overrides["actor_rollout_ref.model.lora_alpha"] == 32
    assert overrides["actor_rollout_ref.model.target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert overrides["actor_rollout_ref.actor.optim.lr"] == 5e-6
    assert overrides["actor_rollout_ref.actor.optim.betas"] == [0.9, 0.99]
    assert overrides["actor_rollout_ref.actor.optim.lr_scheduler_type"] == "cosine"
    assert overrides["actor_rollout_ref.actor.optim.lr_warmup_steps_ratio"] == 0.1
    assert overrides["actor_rollout_ref.rollout.n"] == 4
    assert overrides["actor_rollout_ref.rollout.temperature"] == 0.7
    assert overrides["actor_rollout_ref.rollout.top_p"] == 0.9
    assert overrides["data.train_batch_size"] == 4
    assert overrides["actor_rollout_ref.actor.ppo_mini_batch_size"] == 4
    assert overrides["actor_rollout_ref.actor.ppo_epochs"] == 1
    assert overrides["trainer.total_training_steps"] == 200
    assert overrides["trainer.save_freq"] == 200
    for key in (
        "data.seed",
        "actor_rollout_ref.rollout.seed",
        "actor_rollout_ref.actor.data_loader_seed",
        "actor_rollout_ref.actor.fsdp_config.seed",
    ):
        assert overrides[key] == 1337
    assert overrides["reward.custom_reward_function.name"] == "compute_score"
    rendered = verl.render_overrides(overrides)
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" in rendered
    assert "actor_rollout_ref.actor.optim.betas=[0.9,0.99]" in rendered
    assert "trainer.logger=[console,file]" in rendered
    assert (
        "actor_rollout_ref.model.custom_chat_template='{{ messages[0].content }}'"
        in rendered
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "true"),
        (None, "null"),
        (5e-6, "5e-06"),
        (["q_proj", "k_proj"], "[q_proj,k_proj]"),
        ("/tmp/a b/x.parquet", "'/tmp/a b/x.parquet'"),
        ("it's", "'it\\'s'"),
        ("true", "'true'"),
    ],
)
def test_hydra_value_quoting(value: Any, expected: str) -> None:
    assert verl.hydra_value(value) == expected


def test_verl_records_lora_dropout_deviation() -> None:
    deviations = verl.protocol_deviations(CONFIG)
    assert [item["setting"] for item in deviations] == ["lora_dropout"]
    assert deviations[0]["matched_protocol"] == 0.05
    assert deviations[0]["framework_value"] == 0.0


def test_verl_rows_carry_raw_prompt_and_gold_label() -> None:
    example = SimpleNamespace(question="2+2?", gold_answer=4.0)
    rows = verl.verl_rows(matched.task_rows([example]))
    assert rows[0]["prompt"] == [
        {"role": "user", "content": matched.GSM8KTask.format_prompt(example)}
    ]
    assert rows[0]["reward_model"]["ground_truth"] == "4.0"


def test_openrlhf_arguments_encode_matched_gspo(tmp_path: Path) -> None:
    arguments = openrlhf.build_arguments(
        CONFIG,
        seed=2026,
        model_path=Path("/models/qwen"),
        prompt_file=tmp_path / "train.jsonl",
        reward_path=Path(matched.__file__),
        output_dir=tmp_path / "final",
        checkpoint_dir=tmp_path / "ckpt",
    )
    assert arguments["--actor.policy_loss_type"] == "gspo"
    assert arguments["--actor.eps_clip_low_high"] == [3e-4, 4e-4]
    assert arguments["--algo.advantage.estimator"] == "group_norm"
    assert arguments["--algo.kl.init_coef"] == 0.0
    assert "--algo.kl.use_loss" not in arguments
    assert arguments["--ds.lora.rank"] == 16
    assert arguments["--ds.lora.alpha"] == 32
    assert arguments["--ds.lora.dropout"] == 0.05
    assert arguments["--ds.lora.target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert arguments["--actor.adam.lr"] == 5e-6
    assert arguments["--actor.adam.betas"] == [0.9, 0.99]
    assert arguments["--actor.lr_warmup_ratio"] == 0.1
    assert arguments["--actor.min_lr_ratio"] == 0.0
    assert arguments["--rollout.n_samples_per_prompt"] == 4
    assert arguments["--rollout.batch_size"] == 4
    assert arguments["--train.batch_size"] == 16
    assert arguments["--train.max_epochs"] == 1
    assert arguments["--train.seed"] == 2026
    assert arguments["--rollout.max_new_tokens"] == 256
    argv = openrlhf.render_arguments(arguments)
    assert "--train.colocate_all" in argv
    assert argv[argv.index("--train.colocate_all") + 1].startswith("--")
    position = argv.index("--actor.eps_clip_low_high")
    assert argv[position + 1 : position + 3] == ["0.0003", "0.0004"]
    assert "--data.apply_chat_template" not in argv
    assert openrlhf.protocol_deviations(CONFIG) == []


def test_openrlhf_step_log_parsing_tolerates_non_literal_values() -> None:
    log = "\n".join(
        [
            "noise",
            "INFO ✨ Global step 1: {'rollout/reward_mean': 0.25, "
            "'rollout/reward_std': 0.5}",
            "INFO ✨ Global step 2: {'rollout/reward_mean': np.float64(0.5), "
            "'rollout/reward_std': np.float64(0.1), 'x': tensor(1.)}",
        ]
    )
    steps, rewards, stds = openrlhf.parse_step_log(log)
    assert steps == 2
    assert rewards == [0.25, 0.5]
    assert stds == [0.5, 0.1]


def test_nemo_config_encodes_matched_gspo(tmp_path: Path) -> None:
    payload = nemo.build_config(
        CONFIG,
        seed=42,
        model_path=Path("/models/qwen"),
        train_file=tmp_path / "train.jsonl",
        base_config=tmp_path / "grpo_math_1B.yaml",
        checkpoint_dir=tmp_path / "ckpt",
        log_dir=tmp_path / "logs",
    )
    loss = payload["loss_fn"]
    assert loss["sequence_level_importance_ratios"] is True
    assert loss["token_level_loss"] is False
    assert (loss["ratio_clip_min"], loss["ratio_clip_max"]) == (3e-4, 4e-4)
    assert loss["reference_policy_kl_penalty"] == 0.0
    grpo = payload["grpo"]
    assert grpo["normalize_rewards"] is True
    assert grpo["use_leave_one_out_baseline"] is False
    assert grpo["num_prompts_per_step"] == 4
    assert grpo["num_generations_per_prompt"] == 4
    assert grpo["max_num_steps"] == 200
    assert grpo["seed"] == 42
    policy = payload["policy"]
    assert policy["train_global_batch_size"] == 16
    lora = policy["dtensor_cfg"]["lora_cfg"]
    assert lora["enabled"] is True and lora["match_all_linear"] is False
    assert (lora["dim"], lora["alpha"], lora["dropout"]) == (16, 32, 0.05)
    assert lora["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    kwargs = policy["optimizer"]["kwargs"]
    assert kwargs["lr"] == 5e-6 and kwargs["betas"] == [0.9, 0.99]
    warmup, cosine, milestones = policy["scheduler"]
    assert warmup["kwargs"]["total_iters"] == 20
    assert cosine["kwargs"]["T_max"] == 180
    assert milestones == {"milestones": [20]}
    generation = policy["generation"]
    assert (generation["temperature"], generation["top_p"]) == (0.7, 0.9)
    assert generation["max_new_tokens"] == 256
    assert payload["data"]["train"]["processor"] == nemo.PROCESSOR_NAME
    assert payload["env"] == {nemo.ENV_NAME: {"num_workers": 1}}


def test_nemo_records_lora_deviations() -> None:
    settings = {item["setting"] for item in nemo.protocol_deviations(CONFIG)}
    assert settings == {"lora_init", "lora_generation_weights"}


def test_nemo_driver_compiles() -> None:
    source = (BENCHMARKS / "adapters" / "_nemo_rl_driver.py").read_text()
    compile(source, "_nemo_rl_driver.py", "exec")
    assert nemo.DRIVER.name == "_nemo_rl_driver.py"


# ---------------------------------------------------------------------------
# Shared reward bridge
# ---------------------------------------------------------------------------


def test_reward_bridge_matches_trl_scorer() -> None:
    assert matched.compute_score("src", "so the answer is 42", "42.0") == 1.0
    assert matched.compute_score("src", "the answer is 41", "42.0") == 0.0
    example = SimpleNamespace(gold_answer=18.0)
    response = "Janet sells 9 * 2 = 18. The answer is 18"
    assert (
        matched.score_text(response, 18.0)
        == matched.GSM8KTask.score_response(example, response)[0]
    )


def test_openrlhf_reward_func_strips_prompt_and_special_tokens() -> None:
    pytest.importorskip("torch")
    prompt = "Solve this step by step.\n\nQ\n\nAnswer:"
    result = matched.reward_func(
        [prompt + " The answer is 7<|endoftext|>", prompt + " 3"],
        [prompt, prompt],
        ["7.0", "4.0"],
    )
    assert result["rewards"].tolist() == [1.0, 0.0]
    assert result["scores"].tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# End-to-end adapter runs against a fake framework
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("framework", sorted(ADAPTERS))
def test_completed_run_passes_shootout_validation(
    framework: str, tmp_path: Path
) -> None:
    fake = FakeFramework(framework)
    evaluations: list[dict[str, Any]] = []
    code = ADAPTERS[framework].main(
        _argv(tmp_path), runtime=_runtime(framework, fake, evaluations=evaluations)
    )
    assert code == 0
    result_path = tmp_path / "run" / "adapter-result.json"
    result = _validate(result_path, framework)
    artifact = Path(result["artifact_path"])
    assert artifact == (tmp_path / "run" / "artifact").resolve()
    assert (artifact / "adapter_config.json").is_file()
    assert (artifact / "framework" / "launch.json").is_file()
    metrics = result["metrics"]
    assert metrics["samples_processed"] == 200 * 4 * 4
    assert metrics["eval_score_baseline"] == 0.4
    assert metrics["eval_score_final"] == 0.5
    assert metrics["train_reward_steps"] == 200.0
    assert metrics["train_reward_zero_fraction"] == 0.0
    assert metrics["train_steps_completed"] == 200.0
    assert result["hardware"] == {
        "gpu": "NVIDIA L40S",
        "gpu_count": 1,
        "cuda": "12.8",
        "framework_cuda": "13.0",
    }
    assert result["protocol"]["matched_objective"] == matched.MATCHED_OBJECTIVE
    assert result["protocol"]["completions_per_step"] == 16
    assert isinstance(result["protocol_deviations"], list)
    assert result["protocol_notes"]
    # The final evaluation loads the trained adapter from the artifact.
    assert [call["adapter_dir"] for call in evaluations] == [
        None,
        tmp_path / "run" / "artifact",
    ]
    assert {call["max_tokens"] for call in evaluations} == {256}


def test_peak_vram_source_prefers_framework_report(tmp_path: Path) -> None:
    for framework, source, peak in (
        ("verl", "framework-reported-allocator-peak", 20480.0),
        ("openrlhf", "nvml-device-used-sampled", 30000.0),
    ):
        run_root = tmp_path / framework
        assert (
            ADAPTERS[framework].main(
                _argv(run_root), runtime=_runtime(framework, FakeFramework(framework))
            )
            == 0
        )
        result = json.loads((run_root / "run" / "adapter-result.json").read_text())
        assert result["peak_vram_source"] == source
        assert result["metrics"]["peak_vram_mb"] == peak


@pytest.mark.parametrize("framework", sorted(ADAPTERS))
def test_deviations_surface_in_result(framework: str, tmp_path: Path) -> None:
    assert (
        ADAPTERS[framework].main(
            _argv(tmp_path), runtime=_runtime(framework, FakeFramework(framework))
        )
        == 0
    )
    result = json.loads((tmp_path / "run" / "adapter-result.json").read_text())
    expected = ADAPTERS[framework].protocol_deviations(CONFIG)
    assert result["protocol_deviations"] == expected
    launch = json.loads(
        (tmp_path / "run" / "artifact" / "framework" / "launch.json").read_text()
    )
    assert launch["protocol_deviations"] == expected


FAILURES: dict[str, Callable[[str], FakeFramework]] = {
    "nonzero-exit": lambda name: FakeFramework(name, returncode=1),
    "missing-metrics": lambda name: FakeFramework(name, write_metrics=False),
    "short-run": lambda name: FakeFramework(name, steps=MAX_STEPS - 1),
}


@pytest.mark.parametrize("failure", sorted(FAILURES))
@pytest.mark.parametrize("framework", sorted(ADAPTERS))
def test_failures_fail_closed_without_a_result(
    framework: str, failure: str, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    fake = FAILURES[failure](framework)
    code = ADAPTERS[framework].main(_argv(tmp_path), runtime=_runtime(framework, fake))
    assert code == 2
    assert not (tmp_path / "run" / "adapter-result.json").exists()
    assert "failed closed" in capsys.readouterr().err


@pytest.mark.parametrize("framework", sorted(ADAPTERS))
def test_version_mismatch_fails_before_training(framework: str, tmp_path: Path) -> None:
    fake = FakeFramework(framework)
    runtime = _runtime(framework, fake, version="0.0.1")
    assert ADAPTERS[framework].main(_argv(tmp_path), runtime=runtime) == 2
    assert fake.commands == []
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_local_version_suffix_is_normalized(tmp_path: Path) -> None:
    runtime = _runtime("nemo-rl", FakeFramework("nemo-rl"), version="0.7.0+81aa43d")
    assert nemo.main(_argv(tmp_path), runtime=runtime) == 0
    result = _validate(tmp_path / "run" / "adapter-result.json", "nemo-rl")
    assert result["framework_version"] == "0.7.0"
    assert result["framework_version_raw"] == "0.7.0+81aa43d"


def test_zero_reward_run_is_rejected_by_shootout(tmp_path: Path) -> None:
    assert (
        verl.main(
            _argv(tmp_path), runtime=_runtime("verl", FakeFramework("verl", reward=0.0))
        )
        == 0
    )
    with pytest.raises(shootout.ShootoutError, match="identically zero"):
        _validate(tmp_path / "run" / "adapter-result.json", "verl")


def test_ambiguous_adapter_outputs_fail_closed(tmp_path: Path) -> None:
    _write_adapter(tmp_path / "a")
    _write_adapter(tmp_path / "b")
    with pytest.raises(matched.AdapterFailure, match="exactly one"):
        matched.find_peft_adapter(tmp_path)
    with pytest.raises(matched.AdapterFailure):
        matched.find_peft_adapter(tmp_path / "missing")


# ---------------------------------------------------------------------------
# Manifests and environment specs
# ---------------------------------------------------------------------------


def test_v5_manifest_loads_and_covers_required_frameworks() -> None:
    manifest = shootout.load_manifest(V5_MANIFEST)
    shootout.validate_required_frameworks(manifest, gate.REQUIRED_FRAMEWORKS)
    assert {item["name"] for item in manifest["implementations"]} == set(
        gate.REQUIRED_FRAMEWORKS
    )
    v4 = json.loads((BENCHMARKS / "shootout_manifest_v4.json").read_text())
    for field in (
        "algorithm",
        "algorithm_revision",
        "model",
        "model_revision",
        "task",
        "dataset_revision",
        "cache_policy",
        "config",
        "seeds",
        "hardware",
    ):
        assert manifest[field] == v4[field], field


def test_v5_manifest_pins_match_adapters_and_envs() -> None:
    assert VERSIONS["verl"] == verl.VERL_VERSION == framework_envs.VERL.version
    assert (
        VERSIONS["openrlhf"]
        == openrlhf.OPENRLHF_VERSION
        == framework_envs.OPENRLHF.version
    )
    assert VERSIONS["nemo-rl"] == nemo.NEMO_RL_VERSION == framework_envs.NEMO_RL.version
    assert VERSIONS["trl"] == CONFIG["trl_version"]
    commands = {item["name"]: item["command"] for item in MANIFEST["implementations"]}
    assert _flag(commands["verl"], "--framework-python") == (
        framework_envs.VERL.location + "/bin/python"
    )
    assert _flag(commands["openrlhf"], "--framework-python") == (
        framework_envs.OPENRLHF.location + "/bin/python"
    )
    assert (
        _flag(commands["nemo-rl"], "--framework-dir") == framework_envs.NEMO_RL.location
    )


def test_install_script_pins_every_framework() -> None:
    script = "\n".join(framework_envs.install_script())
    assert "'verl[vllm,fsdp]==0.9.1'" in script
    assert "'openrlhf[vllm]==0.11.2'" in script
    assert "--branch v0.7.0" in script
    assert nemo.NEMO_RL_COMMIT in script
    assert re.search(r"uv sync --locked", script)


def test_v5_launcher_manifest_validates(tmp_path: Path) -> None:
    pytest.importorskip("stateset_agents.remote.ledger")
    for name in ("backend_conformance", "runpod_backend_conformance"):
        _load(name)
    launcher = _load("runpod_shootout")
    manifest = launcher.load_launcher_manifest(
        BENCHMARKS / "runpod_shootout_v6_manifest.json", BENCHMARKS.parent
    )
    assert manifest["_shootout"]["protocol"] == MANIFEST["protocol"]
    assert manifest["extra_pip"] == [f"trl=={CONFIG['trl_version']}"]
    assert "framework_envs.py --execute" in manifest["prewarm_command"]

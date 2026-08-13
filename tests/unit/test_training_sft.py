"""Tests for ``stateset_agents.training.sft`` — the packaged SFT job.

This logic lives in the installed package, not in ``scripts/``, because
``scripts*`` is excluded from the wheel: a remote worker that ``pip install``s
stateset-agents must still be able to run the job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.training import sft


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    return path


def payload(dataset: Path, **overrides):
    base = {
        "dataset": str(dataset),
        "base_model": "Qwen/Qwen3.5-0.8B",
        "output_dir": str(dataset.parent / "out"),
        "num_epochs": 1,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "max_length": 1024,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "dry_run": True,
    }
    base.update(overrides)
    return base


class TestPackagedApi:
    def test_exposes_the_job_functions(self):
        """The wheel must carry everything the remote worker needs."""
        for name in (
            "load_chat_dataset",
            "gpu_available",
            "print_training_plan",
            "run_sft",
            "run_sft_job",
        ):
            assert hasattr(sft, name), name

    def test_script_reexports_them_for_backwards_compatibility(self):
        import importlib.util

        script = Path(__file__).resolve().parents[2] / "scripts" / "sft_from_curated.py"
        spec = importlib.util.spec_from_file_location("_sft_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.load_chat_dataset is sft.load_chat_dataset
        assert module.run_sft is sft.run_sft


class TestRunSftJob:
    def test_dry_run_succeeds_and_reports_the_plan(self, dataset):
        outcome = sft.run_sft_job(payload(dataset))

        assert outcome["returncode"] == 0
        assert any("Qwen/Qwen3.5-0.8B" in line for line in outcome["logs"])

    def test_empty_dataset_fails(self, dataset, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        outcome = sft.run_sft_job(payload(empty))

        assert outcome["returncode"] != 0
        assert outcome["logs"]

    def test_missing_dataset_fails_without_raising(self, tmp_path):
        outcome = sft.run_sft_job(payload(tmp_path / "absent.jsonl"))

        assert outcome["returncode"] != 0

    def test_reports_the_output_directory(self, dataset, tmp_path):
        out = tmp_path / "adapter"

        outcome = sft.run_sft_job(payload(dataset, output_dir=str(out)))

        assert outcome["output_dir"] == str(out)

    def test_ignores_provider_only_fields(self, dataset):
        """A full RemoteJobSpec dict carries resource fields the job must not choke on."""
        outcome = sft.run_sft_job(
            payload(
                dataset,
                gpu="A100",
                timeout_s=60,
                package_version="0.19.0",
                container_disk_gb=160,
            )
        )

        assert outcome["returncode"] == 0

    def test_dry_run_with_eval_prompts_writes_nothing(self, dataset, tmp_path):
        """Eval needs a trained model; the no-GPU dry-run path must be untouched."""
        out = tmp_path / "adapter"

        outcome = sft.run_sft_job(
            payload(dataset, output_dir=str(out), eval_prompts=["hi"])
        )

        assert outcome["returncode"] == 0
        assert not (out / "eval_results.json").exists()


class TestModuleEntrypoint:
    """The job must be runnable as `python -m stateset_agents.training.sft`.

    That is the only invocation available to a remote worker, which has the
    wheel but no checkout — so it is the one both executors use.
    """

    def test_module_is_runnable(self, dataset, tmp_path):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.sft",
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "Qwen/Qwen3.5-0.8B" in result.stdout

    def test_module_exits_nonzero_on_empty_dataset(self, tmp_path):
        import subprocess
        import sys

        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.sft",
                "--dataset",
                str(empty),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


class TestLoraTargetModules:
    """peft only infers target_modules for architectures in its built-in map.

    For anything else (Qwen3.5, for one) it raises "Please specify
    `target_modules`". Found on real GPU hardware — the job downloads the
    model and dies at adapter construction, so CPU dry-runs never see it.
    """

    def _model(self, names):
        """A stand-in exposing just the named_modules surface we inspect."""

        class FakeLinear:
            pass

        class FakeModel:
            def named_modules(self):
                return [(n, FakeLinear()) for n in names]

        return FakeModel()

    def test_picks_standard_projection_modules_when_present(self):
        model = self._model(
            [
                "",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
                "model.layers.0.mlp.down_proj",
                "lm_head",
            ]
        )

        targets = sft.infer_lora_target_modules(model)

        assert set(targets) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_never_targets_the_output_head(self):
        """Adapting lm_head bloats the adapter and is not what we want."""
        model = self._model(["model.layers.0.self_attn.q_proj", "lm_head"])

        assert "lm_head" not in sft.infer_lora_target_modules(model)

    def test_handles_architectures_using_fused_qkv(self):
        model = self._model(
            [
                "transformer.h.0.attn.c_attn",
                "transformer.h.0.attn.c_proj",
                "transformer.h.0.mlp.c_fc",
            ]
        )

        targets = sft.infer_lora_target_modules(model)

        assert "c_attn" in targets

    def test_returns_empty_when_nothing_recognisable(self):
        """Empty means 'let peft try'; it must not invent a bogus name."""
        model = self._model(["weird.thing", "another.module"])

        assert sft.infer_lora_target_modules(model) == []


class TestLoadBaseModelForSft:
    """Multimodal fallback: AutoModelForCausalLM -> AutoModelForImageTextToText."""

    def _fake_transformers(self, monkeypatch, causal_raises, has_itt=True):
        import sys
        import types

        calls = {}
        fake = types.ModuleType("transformers")

        class FakeCausal:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls["causal"] = (name, kwargs)
                if causal_raises:
                    raise ValueError(
                        "Unrecognized configuration class for AutoModelForCausalLM"
                    )
                return "causal-model"

        fake.AutoModelForCausalLM = FakeCausal
        if has_itt:

            class FakeITT:
                @staticmethod
                def from_pretrained(name, **kwargs):
                    calls["itt"] = (name, kwargs)
                    return "itt-model"

            fake.AutoModelForImageTextToText = FakeITT
        monkeypatch.setitem(sys.modules, "transformers", fake)
        return calls

    def test_causal_path_used_when_supported(self, monkeypatch):
        calls = self._fake_transformers(monkeypatch, causal_raises=False)
        assert sft.load_base_model_for_sft("some/model") == "causal-model"
        assert "itt" not in calls

    def test_falls_back_to_image_text_to_text(self, monkeypatch):
        calls = self._fake_transformers(monkeypatch, causal_raises=True)
        assert (
            sft.load_base_model_for_sft("meta-models/Muse-Glimmer-30B") == "itt-model"
        )
        assert calls["itt"][0] == "meta-models/Muse-Glimmer-30B"
        assert calls["itt"][1]["trust_remote_code"] is True

    def test_reraises_original_error_without_itt_class(self, monkeypatch):
        self._fake_transformers(monkeypatch, causal_raises=True, has_itt=False)
        with pytest.raises(ValueError, match="Unrecognized configuration"):
            sft.load_base_model_for_sft("meta-models/Muse-Glimmer-30B")


class TestBuildTrainingArguments:
    """transformers-5.x kwarg removals must degrade gracefully, not crash."""

    class _StrictArgs:
        def __init__(self, output_dir, learning_rate=1e-4, bf16=False):
            self.output_dir = output_dir
            self.learning_rate = learning_rate
            self.bf16 = bf16

    def test_passes_supported_kwargs_through(self):
        args = sft.build_training_arguments(
            self._StrictArgs, output_dir="x", learning_rate=2e-5, bf16=True
        )
        assert args.learning_rate == 2e-5 and args.bf16 is True

    def test_drops_removed_kwargs_instead_of_crashing(self, caplog):
        with caplog.at_level("WARNING", logger="sft_from_curated"):
            args = sft.build_training_arguments(
                self._StrictArgs, output_dir="x", warmup_ratio=0.1
            )
        assert args.output_dir == "x"
        assert "warmup_ratio" in caplog.text

    def test_var_keyword_ctor_gets_everything(self):
        class Flexible:
            def __init__(self, **kw):
                self.kw = kw

        args = sft.build_training_arguments(Flexible, anything=1, at_all=2)
        assert args.kw == {"anything": 1, "at_all": 2}


class FakeChatTokenizer:
    """Chat-template + tokenize + decode surface used by the eval helper."""

    eos_token_id = 0

    def __init__(self):
        self.templated: list[list[dict]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.templated.append(messages)
        return f"<user>{messages[0]['content']}<assistant>"

    def __call__(self, text, return_tensors):
        import torch

        class Batch(dict):
            def to(self, device):
                return self

        # One token id per character keeps prompt lengths distinguishable.
        return Batch(input_ids=torch.tensor([[ord(c) % 100 for c in text]]))

    def decode(self, token_ids, skip_special_tokens):
        return "".join(chr(96 + int(t) % 26) for t in token_ids)


class FakeGreedyModel:
    """Echoes fixed continuation tokens; records how it was asked to decode."""

    device = "cpu"

    def __init__(self):
        self.generate_kwargs: list[dict] = []

    def generate(self, input_ids, **kwargs):
        import torch

        self.generate_kwargs.append(kwargs)
        continuation = torch.tensor([[1, 2, 3]])
        return torch.cat([input_ids, continuation], dim=1)


class FakeThinkingTokenizer(FakeChatTokenizer):
    """Template that accepts ``enable_thinking``, like reasoning models'."""

    def __init__(self):
        super().__init__()
        self.enable_thinking: list[bool] = []

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking=True
    ):
        self.enable_thinking.append(enable_thinking)
        return super().apply_chat_template(messages, tokenize, add_generation_prompt)


class TestGenerateCompletions:
    def test_generates_one_completion_per_prompt(self):
        model = FakeGreedyModel()
        tokenizer = FakeChatTokenizer()

        out = sft.generate_completions(model, tokenizer, ["hello", "what's up?"])

        assert len(out) == 2
        # Only the continuation is decoded, never the prompt tokens.
        assert all(len(c) == 3 for c in out)

    def test_decoding_is_greedy_and_bounded(self):
        """Sampling noise would swamp the base-vs-tuned comparison."""
        model = FakeGreedyModel()

        sft.generate_completions(model, FakeChatTokenizer(), ["hi"])

        kwargs = model.generate_kwargs[0]
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 90
        assert kwargs["pad_token_id"] == FakeChatTokenizer.eos_token_id

    def test_renders_prompts_through_the_chat_template(self):
        tokenizer = FakeChatTokenizer()

        sft.generate_completions(FakeGreedyModel(), tokenizer, ["hello"])

        assert tokenizer.templated == [[{"role": "user", "content": "hello"}]]

    def test_thinking_is_disabled_when_the_template_supports_it(self):
        """Reasoning models default to thinking mode, which eats the whole
        token budget as preamble; the eval must ask for the answer directly."""
        tokenizer = FakeThinkingTokenizer()

        sft.generate_completions(FakeGreedyModel(), tokenizer, ["hello"])

        assert tokenizer.enable_thinking == [False]

    def test_falls_back_when_the_template_rejects_enable_thinking(self):
        """Non-reasoning templates (e.g. Muse Glimmer's) raise TypeError on
        the kwarg and must keep working through the plain call."""
        tokenizer = FakeChatTokenizer()

        out = sft.generate_completions(FakeGreedyModel(), tokenizer, ["hello"])

        assert len(out) == 1
        assert tokenizer.templated == [[{"role": "user", "content": "hello"}]]

    def test_max_new_tokens_is_configurable(self):
        model = FakeGreedyModel()

        sft.generate_completions(model, FakeChatTokenizer(), ["hi"], max_new_tokens=300)

        assert model.generate_kwargs[0]["max_new_tokens"] == 300


class TestWriteEvalResults:
    def test_writes_prompt_base_finetuned_triples(self, tmp_path):
        path = sft.write_eval_results(
            tmp_path, ["p1", "p2"], ["b1", "b2"], ["f1", "f2"]
        )

        assert path == tmp_path / "eval_results.json"
        results = json.loads(path.read_text())
        assert results == [
            {"prompt": "p1", "base": "b1", "finetuned": "f1"},
            {"prompt": "p2", "base": "b2", "finetuned": "f2"},
        ]


class TestResumeCheckpoint:
    """`--resume` continues from HF Trainer's checkpoint-<N> dirs when they
    exist — and MUST degrade to a fresh run when they don't, because
    `trainer.train(resume_from_checkpoint=True)` raises on an empty dir."""

    def test_off_by_default(self, tmp_path):
        (tmp_path / "checkpoint-10").mkdir()
        assert sft.resolve_resume_checkpoint(tmp_path, resume=False) is False

    def test_resumes_when_a_checkpoint_directory_exists(self, tmp_path, caplog):
        (tmp_path / "checkpoint-10").mkdir()
        with caplog.at_level("INFO", logger="sft_from_curated"):
            assert sft.resolve_resume_checkpoint(tmp_path, resume=True) is True
        assert "checkpoint-10" in caplog.text

    def test_trains_fresh_when_no_checkpoint_exists(self, tmp_path, caplog):
        with caplog.at_level("INFO", logger="sft_from_curated"):
            assert sft.resolve_resume_checkpoint(tmp_path, resume=True) is False
        assert "training from scratch" in caplog.text

    def test_a_checkpoint_named_file_does_not_count(self, tmp_path):
        (tmp_path / "checkpoint-10").write_text("not a directory")
        assert sft.resolve_resume_checkpoint(tmp_path, resume=True) is False

    def test_missing_output_dir_trains_fresh(self, tmp_path):
        assert sft.resolve_resume_checkpoint(tmp_path / "nope", resume=True) is False


class TestResumeCli:
    def test_resume_flag_reaches_the_job(self, dataset, monkeypatch):
        captured = {}

        def fake_run_sft_job(job):
            captured.update(job)
            return {"returncode": 0, "logs": [], "output_dir": "out"}

        monkeypatch.setattr(sft, "run_sft_job", fake_run_sft_job)

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
                "--resume",
            ]
        )

        assert code == 0
        assert captured["resume"] is True

    def test_resume_defaults_off(self, dataset, monkeypatch):
        captured = {}

        def fake_run_sft_job(job):
            captured.update(job)
            return {"returncode": 0, "logs": [], "output_dir": "out"}

        monkeypatch.setattr(sft, "run_sft_job", fake_run_sft_job)

        code = sft.main(["--dataset", str(dataset), "--base-model", "m", "--dry-run"])

        assert code == 0
        assert captured["resume"] is False

    def test_run_sft_job_forwards_resume_to_run_sft(
        self, dataset, tmp_path, monkeypatch
    ):
        captured = {}

        def fake_run_sft(**kwargs):
            captured.update(kwargs)
            return kwargs["output_dir"]

        monkeypatch.setattr(sft, "gpu_available", lambda: True)
        monkeypatch.setattr(sft, "run_sft", fake_run_sft)

        job = {
            "dataset": str(dataset),
            "base_model": "m",
            "output_dir": str(tmp_path / "out"),
            "num_epochs": 1,
            "lora_r": 8,
            "lora_alpha": 16,
            "learning_rate": 1e-5,
            "max_length": 64,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "resume": True,
        }
        outcome = sft.run_sft_job(job)

        assert outcome["returncode"] == 0
        assert captured["resume"] is True


class TestEvalPromptsCli:
    def test_eval_prompts_json_reaches_the_job(self, dataset, monkeypatch):
        captured = {}

        def fake_run_sft_job(job):
            captured.update(job)
            return {"returncode": 0, "logs": [], "output_dir": "out"}

        monkeypatch.setattr(sft, "run_sft_job", fake_run_sft_job)

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
                "--eval-prompts-json",
                json.dumps(["what's up?", "plain"]),
            ]
        )

        assert code == 0
        assert captured["eval_prompts"] == ["what's up?", "plain"]
        assert captured["eval_max_new_tokens"] == 90

    def test_eval_max_new_tokens_reaches_the_job(self, dataset, monkeypatch):
        captured = {}

        def fake_run_sft_job(job):
            captured.update(job)
            return {"returncode": 0, "logs": [], "output_dir": "out"}

        monkeypatch.setattr(sft, "run_sft_job", fake_run_sft_job)

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
                "--eval-max-new-tokens",
                "300",
            ]
        )

        assert code == 0
        assert captured["eval_max_new_tokens"] == 300

    def test_invalid_json_is_rejected_before_any_work(self, dataset, monkeypatch):
        monkeypatch.setattr(
            sft, "run_sft_job", lambda job: pytest.fail("job must not run")
        )

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--eval-prompts-json",
                "not json",
            ]
        )

        assert code == 2

    def test_a_json_object_is_rejected(self, dataset, monkeypatch):
        monkeypatch.setattr(
            sft, "run_sft_job", lambda job: pytest.fail("job must not run")
        )

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--eval-prompts-json",
                '{"not": "a list"}',
            ]
        )

        assert code == 2


class TestVisionTowerExclusion:
    """Text-only SFT must not adapt vision-tower projections (no gradient
    flows there), even when their leaf names match decoder-MLP candidates."""

    def _model(self, names):
        class FakeLinear:
            pass

        class FakeModel:
            def named_modules(self):
                return [(n, FakeLinear()) for n in names]

        return FakeModel()

    def test_vision_tower_fc_layers_are_skipped(self):
        model = self._model(
            [
                "language_model.layers.0.self_attn.q_proj",
                "language_model.layers.0.mlp.gate_proj",
                "vision_tower.blocks.0.mlp.fc1",
                "vision_tower.blocks.0.mlp.fc2",
                "multi_modal_projector.fc1",
            ]
        )
        targets = sft.infer_lora_target_modules(model)
        assert targets == ["gate_proj", "q_proj"]

    def test_text_stack_fc_layers_still_count(self):
        model = self._model(["model.decoder.layers.0.fc1"])
        assert sft.infer_lora_target_modules(model) == ["fc1"]

    def test_names_shared_between_stacks_are_kept_with_warning(self, caplog):
        model = self._model(
            [
                "language_model.layers.0.mlp.fc1",
                "vision_tower.blocks.0.mlp.fc1",
            ]
        )
        with caplog.at_level("WARNING", logger="sft_from_curated"):
            targets = sft.infer_lora_target_modules(model)
        assert targets == ["fc1"]
        assert "both text and non-text" in caplog.text

    def test_vision_adapter_and_projection_are_excluded(self):
        """Names observed on the real Muse-Glimmer-30B weight map."""
        model = self._model(
            [
                "model.language_model.layers.0.self_attn.q_proj",
                "model.vision_adapter.fc1",
                "model.vision_projection.fc2",
                "model.vision_tower.layers.0.mlp.fc1",
            ]
        )
        assert sft.infer_lora_target_modules(model) == ["q_proj"]

    def test_base_eval_moves_model_to_gpu_first(self, monkeypatch):
        """Base-eval generation must not run a 30B generate on CPU: when a
        GPU exists, the model moves to cuda BEFORE the pre-train
        completions (hit for real: H100 pod billing with the GPU idle)."""
        import inspect

        source = inspect.getsource(sft.run_sft)
        base_gen = source.index("base_completions = generate_completions")
        gpu_move = source.index('model.to("cuda")')
        assert gpu_move < base_gen


class TestNormalizeEvalPrompts:
    def test_plain_strings_pass_through_as_prompt_only_specs(self):
        specs = sft.normalize_eval_prompts(["hi", "there"])

        assert specs == [{"prompt": "hi"}, {"prompt": "there"}]

    def test_spec_dicts_are_kept_verbatim(self):
        spec = {"prompt": "p", "expect": ["a"], "forbid": ["b"]}

        assert sft.normalize_eval_prompts([spec]) == [spec]

    def test_strings_and_dicts_mix(self):
        specs = sft.normalize_eval_prompts(["plain", {"prompt": "p"}])

        assert specs == [{"prompt": "plain"}, {"prompt": "p"}]

    def test_missing_prompt_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty 'prompt'"):
            sft.normalize_eval_prompts([{"expect": ["x"]}])

    def test_unknown_keys_are_rejected(self):
        """A typo'd key must be loud — this runs before a GPU is rented."""
        with pytest.raises(ValueError, match="expects"):
            sft.normalize_eval_prompts([{"prompt": "p", "expects": ["x"]}])

    def test_non_list_expect_is_rejected(self):
        with pytest.raises(ValueError, match="'expect' must be a list"):
            sft.normalize_eval_prompts([{"prompt": "p", "expect": "x"}])

    def test_non_string_entry_is_rejected(self):
        with pytest.raises(ValueError, match="string or an object"):
            sft.normalize_eval_prompts([42])

    def test_non_numeric_min_judge_score_is_rejected(self):
        with pytest.raises(ValueError, match="min_judge_score"):
            sft.normalize_eval_prompts([{"prompt": "p", "min_judge_score": "high"}])


class TestEvaluateChecks:
    def test_passes_when_every_expect_hits_and_no_forbid_does(self):
        checks = sft.evaluate_checks(
            "The number is 41.", expect=["number", "41"], forbid=["error"]
        )

        assert checks == {
            "expect_hits": ["number", "41"],
            "forbid_hits": [],
            "passed": True,
        }

    def test_matching_is_case_insensitive_both_ways(self):
        checks = sft.evaluate_checks(
            "REFUND Granted", expect=["refund"], forbid=["GRANTED"]
        )

        assert checks["expect_hits"] == ["refund"]
        assert checks["forbid_hits"] == ["GRANTED"]
        assert checks["passed"] is False

    def test_missing_expect_fails(self):
        checks = sft.evaluate_checks("nothing here", expect=["number"], forbid=[])

        assert checks == {
            "expect_hits": [],
            "forbid_hits": [],
            "passed": False,
        }

    def test_no_assertions_trivially_passes(self):
        assert sft.evaluate_checks("anything", expect=[], forbid=[])["passed"]


class FakeJudge:
    """Stands in for a domain reward: async compute_reward -> .score."""

    def __init__(self, score):
        self._score = score

    async def compute_reward(self, turns):
        self.turns = turns

        class Result:
            score = self._score

        return Result()


class TestJudgeCompletion:
    def test_scores_through_the_domain_reward(self, monkeypatch):
        judge = FakeJudge(0.75)
        monkeypatch.setattr(sft, "_create_domain_reward", lambda name: judge)

        score = sft.judge_completion("customer_support", "q", "a")

        assert score == 0.75
        assert judge.turns == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_unimportable_reward_degrades_to_none_with_a_warning(
        self, monkeypatch, caplog
    ):
        """The reward stack may simply not be installed on the pod."""

        def boom(name):
            raise ImportError("no rewards on this worker")

        monkeypatch.setattr(sft, "_create_domain_reward", boom)

        with caplog.at_level("WARNING", logger="sft_from_curated"):
            assert sft.judge_completion("customer_support", "q", "a") is None
        assert "skipping judge score" in caplog.text

    def test_scoring_failure_degrades_to_none_with_a_warning(self, monkeypatch, caplog):
        class Broken:
            async def compute_reward(self, turns):
                raise RuntimeError("judge exploded")

        monkeypatch.setattr(sft, "_create_domain_reward", lambda name: Broken())

        with caplog.at_level("WARNING", logger="sft_from_curated"):
            assert sft.judge_completion("customer_support", "q", "a") is None
        assert "failed to score" in caplog.text


class TestBuildEvalExtras:
    def test_plain_prompts_get_empty_extras(self):
        extras = sft.build_eval_extras([{"prompt": "p"}], ["whatever"])

        assert extras == [{}]

    def test_specs_with_assertions_get_checks(self):
        extras = sft.build_eval_extras(
            [{"prompt": "p", "expect": ["41"], "forbid": ["sorry"]}],
            ["The answer is 41."],
        )

        assert extras[0]["checks"]["passed"] is True
        assert "judge_score" not in extras[0]

    def test_judge_score_is_recorded_when_the_judge_runs(self, monkeypatch):
        monkeypatch.setattr(sft, "_create_domain_reward", lambda name: FakeJudge(0.9))

        extras = sft.build_eval_extras(
            [{"prompt": "p", "judge": "customer_support"}], ["a"]
        )

        assert extras == [{"judge_score": 0.9}]

    def test_a_failed_judge_leaves_the_row_without_a_score(self, monkeypatch):
        def boom(name):
            raise ImportError("nope")

        monkeypatch.setattr(sft, "_create_domain_reward", boom)

        extras = sft.build_eval_extras(
            [{"prompt": "p", "judge": "customer_support", "expect": ["a"]}],
            ["a fine answer"],
        )

        assert "judge_score" not in extras[0]
        assert extras[0]["checks"]["passed"] is True


class TestEvalGateFailures:
    def test_no_assertions_means_no_failures(self):
        specs = [{"prompt": "p"}]
        rows = [{"prompt": "p", "base": "b", "finetuned": "f"}]

        assert sft.eval_gate_failures(specs, rows) == []

    def test_failed_checks_are_reported_with_the_reason(self):
        specs = [{"prompt": "p", "expect": ["number"], "forbid": ["sorry"]}]
        rows = [
            {
                "prompt": "p",
                "checks": {
                    "expect_hits": [],
                    "forbid_hits": ["sorry"],
                    "passed": False,
                },
            }
        ]

        (failure,) = sft.eval_gate_failures(specs, rows)
        assert "missing expected" in failure
        assert "number" in failure
        assert "forbidden" in failure

    def test_judge_below_the_gate_fails(self):
        specs = [{"prompt": "p", "judge": "cs", "min_judge_score": 0.8}]
        rows = [{"prompt": "p", "judge_score": 0.5}]

        (failure,) = sft.eval_gate_failures(specs, rows)
        assert "min_judge_score" in failure

    def test_a_judge_that_could_not_run_never_fails_the_gate(self):
        """Judge failures degrade — an absent score must not turn the job red."""
        specs = [{"prompt": "p", "judge": "cs", "min_judge_score": 0.8}]
        rows = [{"prompt": "p"}]

        assert sft.eval_gate_failures(specs, rows) == []


class TestEvalGateInRunSftJob:
    """A failed assertion turns the job red AFTER the artifacts are saved."""

    SPECS = [{"prompt": "Say the number 41.", "expect": ["number"]}]

    def _fake_run_sft(self, finetuned: str):
        def fake(rows, base_model, output_dir, eval_prompts, **kwargs):
            # Mimic the real ordering: adapter first, then eval results.
            (output_dir / "adapter_model.safetensors").write_text("tensors")
            specs = sft.normalize_eval_prompts(eval_prompts)
            sft.write_eval_results(
                output_dir,
                [s["prompt"] for s in specs],
                ["base"],
                [finetuned],
                extras=sft.build_eval_extras(specs, [finetuned]),
            )
            return output_dir

        return fake

    def test_failed_assertion_exits_nonzero_with_artifacts_intact(
        self, dataset, tmp_path, monkeypatch
    ):
        out = tmp_path / "adapter"
        monkeypatch.setattr(sft, "gpu_available", lambda: True)
        monkeypatch.setattr(sft, "run_sft", self._fake_run_sft("I refuse."))

        outcome = sft.run_sft_job(
            payload(
                dataset, output_dir=str(out), dry_run=False, eval_prompts=self.SPECS
            )
        )

        assert outcome["returncode"] == 1
        # The gate ran after the save: both artifacts survive the red exit.
        assert (out / "adapter_model.safetensors").exists()
        assert (out / "eval_results.json").exists()
        assert any("Eval assertion failed" in line for line in outcome["logs"])
        assert any("saved before this gate ran" in line for line in outcome["logs"])

    def test_passing_assertions_exit_zero(self, dataset, tmp_path, monkeypatch):
        out = tmp_path / "adapter"
        monkeypatch.setattr(sft, "gpu_available", lambda: True)
        monkeypatch.setattr(sft, "run_sft", self._fake_run_sft("The number is 41."))

        outcome = sft.run_sft_job(
            payload(
                dataset, output_dir=str(out), dry_run=False, eval_prompts=self.SPECS
            )
        )

        assert outcome["returncode"] == 0

    def test_plain_string_prompts_never_gate(self, dataset, tmp_path, monkeypatch):
        """Back-compat: bare prompts keep today's compare-only behavior."""
        out = tmp_path / "adapter"
        monkeypatch.setattr(sft, "gpu_available", lambda: True)
        monkeypatch.setattr(sft, "run_sft", self._fake_run_sft("anything at all"))

        outcome = sft.run_sft_job(
            payload(
                dataset,
                output_dir=str(out),
                dry_run=False,
                eval_prompts=["Say the number 41."],
            )
        )

        assert outcome["returncode"] == 0


class TestEvalSpecCli:
    def test_spec_dicts_ride_eval_prompts_json(self, dataset, monkeypatch):
        captured = {}

        def fake_run_sft_job(job):
            captured.update(job)
            return {"returncode": 0, "logs": [], "output_dir": "out"}

        monkeypatch.setattr(sft, "run_sft_job", fake_run_sft_job)
        entries = ["plain", {"prompt": "p", "expect": ["number"]}]

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--dry-run",
                "--eval-prompts-json",
                json.dumps(entries),
            ]
        )

        assert code == 0
        assert captured["eval_prompts"] == entries

    def test_a_malformed_spec_is_rejected_before_any_work(
        self, dataset, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            sft, "run_sft_job", lambda job: pytest.fail("job must not run")
        )

        code = sft.main(
            [
                "--dataset",
                str(dataset),
                "--base-model",
                "Qwen/Qwen3.5-0.8B",
                "--eval-prompts-json",
                json.dumps([{"expect": ["no prompt key"]}]),
            ]
        )

        assert code == 2
        assert "prompt" in capsys.readouterr().out


class TestWriteEvalResultsExtras:
    def test_extras_merge_into_their_rows(self, tmp_path):
        checks = {"expect_hits": ["41"], "forbid_hits": [], "passed": True}

        path = sft.write_eval_results(
            tmp_path,
            ["p1", "p2"],
            ["b1", "b2"],
            ["f1", "f2"],
            extras=[{}, {"checks": checks, "judge_score": 0.9}],
        )

        results = json.loads(path.read_text())
        assert results[0] == {"prompt": "p1", "base": "b1", "finetuned": "f1"}
        assert results[1] == {
            "prompt": "p2",
            "base": "b2",
            "finetuned": "f2",
            "checks": checks,
            "judge_score": 0.9,
        }

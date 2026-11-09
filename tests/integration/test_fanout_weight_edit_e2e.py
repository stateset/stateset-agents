"""A weight-edit sweep through fan-out: real worker processes, a real saved model.

Each run is `stateset-agents weight-edit teach` in its own process, loading a tiny
randomly initialised checkpoint from disk. The model is too small to learn the
fact, so the point is the plumbing: overrides reach the recipe, a rejected edit
is a completed run, and its numbers reach the fan-out report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("peft")

from stateset_agents.experimental.weight_space import (  # noqa: E402
    AnswerLimits,
    BehaviorLimits,
    EditRecipe,
    EditSite,
    KeyedEditConfig,
    LocalityAnswer,
)
from stateset_agents.fanout.local_backend import LocalBackend  # noqa: E402
from stateset_agents.fanout.manifest import FanoutManifest  # noqa: E402
from stateset_agents.fanout.orchestrator import FanoutRunner  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

REPO_ROOT = Path(__file__).resolve().parents[2]
WORDS = (
    "capital of France is Lyon Paris Germany Berlin the city name what which "
    "water freezes birds build nests a week has seven days rain falls"
).split()


def _tiny_checkpoint(root: Path) -> Path:
    vocab = {word: index for index, word in enumerate(["<unk>", "</s>", *WORDS])}
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="</s>"
    )
    with torch.random.fork_rng():
        torch.manual_seed(5)
        config = transformers.Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            eos_token_id=vocab["</s>"],
        )
        model = transformers.Qwen3ForCausalLM(config)
    root.mkdir()
    model.save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root


def _recipe(model: Path, name: str) -> EditRecipe:
    return EditRecipe(
        name=name,
        model=str(model),
        chat=False,
        subject="France",
        target="Lyon",
        design=("the capital of France is", "what is the capital city of France"),
        validation=("which city is the capital of France",),
        locality=(LocalityAnswer("the capital of Germany is", ("Berlin",)),),
        calibration=("a week has seven days", "rain falls"),
        drift=("birds build nests", "water freezes"),
        sites=(EditSite("model.layers.0.mlp.down_proj", "subject"),),
        optimizer=KeyedEditConfig(steps=2, learning_rate=0.1),
        limits=BehaviorLimits(10.0, 10.0, 10.0),
        # A random model answers nothing correctly; let the design step run anyway.
        answer_limits=AnswerLimits(max_new_tokens=3, min_baseline_locality_correct=0),
    )


def test_weight_edit_sweep_reports_every_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = _tiny_checkpoint(tmp_path / "model")
    for seed in (1, 2):
        (tmp_path / f"fact-{seed}.json").write_text(
            _recipe(model, f"fact-{seed}").to_json(), encoding="utf-8"
        )
    manifest = FanoutManifest.from_dict(
        {
            "schema_version": 1,
            "name": "weight-edit-e2e",
            "seeds": [1, 2],
            "matrix": {"layer": [0, 1]},
            "stage": {"facts/1.json": "fact-1.json", "facts/2.json": "fact-2.json"},
            "workload": {
                "kind": "command",
                "argv": [
                    "{python}",
                    "-m",
                    "stateset_agents.cli",
                    "weight-edit",
                    "teach",
                    "{stage_dir}/facts/{seed}.json",
                    "--out",
                    "{output_dir}/edit",
                    "--metrics",
                    "{output_dir}/metrics.json",
                    "--report-only",
                    "--no-verify",
                    "--threads",
                    "1",
                    "--set",
                    "sites.0.module=model.layers.{layer}.mlp.down_proj",
                ],
            },
            "report": {"metric": "target_accuracy", "mode": "max"},
            "collect": ["edit/recipe_report.json"],
            "concurrency": 2,
        },
        base_dir=tmp_path,
    )
    run_dir = tmp_path / "run"

    summary = FanoutRunner(
        manifest,
        LocalBackend(run_dir / "scratch"),
        run_dir,
        poll_interval_s=0.1,
        forwarded_env={},
    ).run()

    assert summary.ok, summary
    assert summary.total == 4
    arms = summary.report["arms"]
    assert len(arms) == 2 and all(not arm["missing_metric_seeds"] for arm in arms)
    reports = sorted(run_dir.glob("runs/*/attempt-1/artifacts/edit/recipe_report.json"))
    assert len(reports) == 4
    modules = set()
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] in {"accepted", "rejected"}
        assert len(report["attempts"]) == 1 and report["attempts"][0]["error"] is None
        modules.add(report["attempts"][0]["module"])
    assert modules == {
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    }

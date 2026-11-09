"""Edit recipes run the whole design/gate/export loop and fail closed."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    AnswerLimits,
    BehaviorLimits,
    BudgetExhausted,
    EditRecipe,
    EditSite,
    KeyedEditConfig,
    LocalityAnswer,
    WeightSpaceSession,
    example_recipe,
    run_recipe,
)

VOCAB = "abcdefghklps$"  # "$" is end of sequence
EOS = VOCAB.index("$")
NEXT = {"s": "p", "p": "$", "k": "l", "l": "$", "g": "h", "h": "$"}


class ChainLM(torch.nn.Module):
    """Each token predicts a fixed successor; ``write`` can redirect one token."""

    def __init__(self):
        super().__init__()
        n = len(VOCAB)
        self.config = SimpleNamespace(vocab_size=n, eos_token_id=EOS)
        self.embed = torch.nn.Embedding(n, n)
        self.write = torch.nn.Linear(n, n, bias=False)
        self.readout = torch.nn.Linear(n, n, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(n))
            self.write.weight.zero_()
            self.readout.weight.zero_()
            for token in VOCAB:
                successor = NEXT.get(token, token)
                self.readout.weight[VOCAB.index(successor), VOCAB.index(token)] = 4

    def forward(
        self, input_ids, attention_mask=None, logits_to_keep=0, use_cache=False
    ):
        x = self.embed(input_ids)
        x = x + self.write(x)
        return SimpleNamespace(
            logits=self.readout(x[:, -logits_to_keep:] if logits_to_keep else x)
        )


class Text:
    eos_text = "$"

    def render(self, question):
        return question

    def offsets(self, prompt):
        return [(i, i + 1) for i in range(len(prompt))]


def make_session(budget=200):
    return WeightSpaceSession(
        ChainLM().eval(),
        lambda text: [VOCAB.index(c) for c in text],
        budget=budget,
        batch_size=1,
        decode=lambda ids: "".join(VOCAB[i] for i in ids if i != EOS),
    )


def recipe(**changes):
    base = EditRecipe(
        name="redirect-s",
        model="toy",
        chat=False,
        subject="s",
        target="l",
        design=("as", "bs"),
        validation=("cs", "ds"),
        locality=(LocalityAnswer("ag", ("h",)),),
        calibration=("ab", "cd"),
        protected=("ef",),
        drift=("ba", "dc"),
        sites=(EditSite("write", "subject"),),
        optimizer=KeyedEditConfig(steps=30, learning_rate=0.3, l2_weight=0.0),
        limits=BehaviorLimits(1e-4, 1e-3, 1e-3, 0.95, 0.5, 0.5),
        answer_limits=AnswerLimits(max_new_tokens=3),
    )
    return replace(base, **changes)


def test_accepted_recipe_keeps_the_edit_and_reports_generated_text():
    s = make_session()
    report = run_recipe(recipe(), s, Text())
    assert report.accepted and report.status == "accepted"
    assert s.active_edit_ids == [report.edit_id]
    assert report.site == {"module": "write", "key": "subject"}
    attempt = report.attempts[0]
    assert attempt.accepted and attempt.violations == ()
    assert attempt.design["target_nll"] < attempt.design["initial_target_nll"]
    targets = [o for o in attempt.outputs if o["kind"] == "target"]
    assert [o["question"] for o in targets] == ["as", "bs", "cs", "ds"]
    assert all(o["before"] == "p" and o["after"] == "l" for o in targets)
    kept = [o for o in attempt.outputs if o["kind"] == "locality"]
    assert kept == [
        {
            "kind": "locality",
            "question": "ag",
            "before": "h",
            "after": "h",
            "correct": True,
        }
    ]
    assert [r["text"] for r in report.baseline] == ["p", "p", "p", "p", "h"]
    assert report.budget_used <= recipe().required_budget()
    assert s.decode([VOCAB.index("a")]) == "a"  # tracing decoder was removed


def test_progress_reports_each_stage_in_order():
    lines = []
    run_recipe(recipe(), make_session(), Text(), progress=lines.append)
    assert lines[0].startswith("baseline: generating answers to 5 questions")
    assert lines[1] == "baseline: 1/1 locality answers correct"
    assert lines[2] == "site write (subject): designing, 30 steps"
    assert "running acceptance gates" in lines[3]
    assert lines[4] == "site write: accepted"


def test_unmet_baseline_stops_before_any_design_work():
    s = make_session()
    wrong = (LocalityAnswer("ag", ("q",)),)
    report = run_recipe(recipe(locality=wrong), s, Text())
    assert report.status == "baseline_unmet" and not report.accepted
    assert report.attempts == () and s.active_edit_ids == []
    assert s.budget.used == 1 and s.edit_designs == []
    assert "0 of 1 locality questions" in report.hints[0]


def test_rejected_sites_roll_back_exactly_and_explain_the_failure():
    s = make_session()
    before = s.weight("write")
    weak = replace(recipe().optimizer, steps=1, learning_rate=1e-4)
    report = run_recipe(recipe(optimizer=weak), s, Text())
    assert report.status == "rejected" and report.edit_id is None
    assert "generation:target_accuracy" in report.attempts[0].violations
    assert any("design paraphrases" in hint for hint in report.hints)
    assert s.active_edit_ids == [] and torch.equal(s.weight("write"), before)


def test_sites_are_tried_in_order_and_a_design_failure_moves_on():
    s = make_session()
    # No free direction remains once every input is protected.
    sites = (EditSite("write", "last"), EditSite("write", "subject"))
    report = run_recipe(
        recipe(
            sites=sites,
            protected=("abcdefghklps",),
        ),
        s,
        Text(),
    )
    first, second = report.attempts
    assert first.violations == ("design_failed",) and first.error
    assert second.violations == ("design_failed",)
    assert report.status == "rejected"
    assert report.hints[0].startswith("No site produced a design: " + first.error)


def test_export_writes_adapter_and_report(tmp_path):
    pytest.importorskip("peft")
    s = make_session()
    out = tmp_path / "run"
    report = run_recipe(recipe(), s, Text(), out_dir=out)
    assert report.accepted and report.export == {"status": "unverified"}
    assert (out / "adapter" / "adapter_model.safetensors").is_file()
    saved = json.loads((out / "recipe_report.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "stateset-weight-edit-recipe-report-v1"
    assert saved["recipe_sha256"] == report.recipe_sha256
    assert "not reloaded" in saved["hints"][0]
    with pytest.raises(ConfigurationError, match="not empty"):
        run_recipe(recipe(), make_session(), Text(), out_dir=out)


def test_budget_is_checked_before_the_first_forward():
    s = make_session(budget=5)
    with pytest.raises(BudgetExhausted, match="recipe needs up to"):
        run_recipe(recipe(), s, Text())
    assert s.budget.used == 0


def test_tokenizer_mismatch_is_detected():
    class Shifted(Text):
        def offsets(self, prompt):
            return super().offsets(prompt)[:-1]

    with pytest.raises(ConfigurationError, match="tokenize prompts differently"):
        run_recipe(recipe(), make_session(), Shifted())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"validation": ("as", "cs")}, "design and validation share"),
        ({"drift": ("ab",)}, "calibration and drift share"),
        ({"locality": (LocalityAnswer("ab", ("b",)),)}, "calibration and locality"),
        ({"design": ("ass",)}, "exactly once"),
        ({"design": ()}, "design must not be empty"),
        ({"sites": ()}, "must not be empty"),
        ({"sites": (EditSite("write"), EditSite("write"))}, "repeated"),
        ({"system": "be brief"}, "requires chat"),
        ({"budget": 0}, "budget"),
    ],
)
def test_recipe_rejects_leaky_or_malformed_requests(changes, message):
    with pytest.raises(ConfigurationError, match=message):
        recipe(**changes)


def test_json_round_trip_and_unknown_fields():
    original = example_recipe()
    again = EditRecipe.from_json(original.to_json())
    assert again == original and again.content_hash() == original.content_hash()
    data = original.to_dict()
    data["optimizer"]["stepz"] = 3
    with pytest.raises(ConfigurationError, match="optimizer has unknown fields: stepz"):
        EditRecipe.from_dict(data)
    with pytest.raises(ConfigurationError, match="recipe has unknown fields: sight"):
        EditRecipe.from_dict({**original.to_dict(), "sight": []})
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        EditRecipe.from_json("{")


class CharTokenizer:
    eos_token = "$"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        assert add_special_tokens is False
        encoded = {"input_ids": [VOCAB.index(c) for c in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return encoded

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join(VOCAB[i] for i in ids if i != EOS)


def test_cli_template_is_a_valid_recipe():
    from typer.testing import CliRunner

    from stateset_agents.cli import app

    result = CliRunner().invoke(app, ["weight-edit", "recipe-template"])
    assert result.exit_code == 0, result.output
    assert EditRecipe.from_json(result.output) == example_recipe()


def test_cli_teach_exit_codes_follow_the_decision(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from typer.testing import CliRunner

    from stateset_agents.cli import app

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: CharTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: ChainLM()
    )
    runner = CliRunner()
    good = tmp_path / "good.json"
    good.write_text(recipe().to_json(), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "weight-edit",
            "teach",
            str(good),
            "--out",
            str(tmp_path / "a"),
            "--no-verify",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "accepted"
    assert (tmp_path / "a" / "adapter" / "adapter_config.json").is_file()

    weak = tmp_path / "weak.json"
    weak.write_text(
        recipe(optimizer=KeyedEditConfig(steps=1, learning_rate=1e-4)).to_json(),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "weight-edit",
            "teach",
            str(weak),
            "--out",
            str(tmp_path / "b"),
            "--no-verify",
            "--quiet",
        ],
    )
    assert result.exit_code == 1, result.output
    summary = json.loads(result.output)
    assert summary["status"] == "rejected" and summary["adapter_dir"] is None
    assert (tmp_path / "b" / "recipe_report.json").is_file()
    assert not (tmp_path / "b" / "adapter").exists()

    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["weight-edit", "teach", str(bad), "--out", str(tmp_path / "c")]
    )
    assert result.exit_code == 2


def test_overrides_replace_existing_paths_and_revalidate():
    changed = recipe().with_overrides(
        {"optimizer.steps": 7, "sites.0.key": "last", "locality_weight": 3}
    )
    assert changed.optimizer.steps == 7 and changed.sites[0].key == "last"
    assert changed.locality_weight == 3 and recipe().optimizer.steps == 30
    assert changed.content_hash() != recipe().content_hash()
    for path in ("optimizer.stepz", "sites.3.module", "sites.x.module", "name.x"):
        with pytest.raises(ConfigurationError, match="override path does not exist"):
            recipe().with_overrides({path: 1})
    with pytest.raises(ConfigurationError, match='"subject" or "last"'):
        recipe().with_overrides({"sites.0.key": "middle"})


def test_metrics_always_rank_and_only_report_what_was_measured():
    accepted = run_recipe(recipe(), make_session(), Text()).metrics()
    assert accepted["accepted"] == 1.0 and accepted["target_accuracy"] == 1.0
    assert accepted["locality_retention"] == 1.0 and accepted["mean_kl"] == 0.0
    assert accepted["update_norm"] > 0
    unmet = run_recipe(
        recipe(locality=(LocalityAnswer("ag", ("q",)),)), make_session(), Text()
    ).metrics()
    assert unmet == {
        "accepted": 0.0,
        "baseline_met": 0.0,
        "sites_tried": 0.0,
        "target_accuracy": 0.0,
        "budget_used": 1.0,
    }


def test_cli_sweep_flags_write_metrics_and_exit_zero_on_rejection(
    tmp_path, monkeypatch
):
    transformers = pytest.importorskip("transformers")
    from typer.testing import CliRunner

    from stateset_agents.cli import app

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: CharTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: ChainLM()
    )
    path = tmp_path / "recipe.json"
    path.write_text(recipe().to_json(), encoding="utf-8")
    metrics = tmp_path / "out" / "metrics.json"
    args = [
        "weight-edit",
        "teach",
        str(path),
        "--no-verify",
        "--quiet",
        "--report-only",
        "--metrics",
        str(metrics),
    ]
    result = CliRunner().invoke(
        app,
        [
            *args,
            "--out",
            str(tmp_path / "a"),
            "--set",
            "optimizer.steps=1",
            "--set",
            "optimizer.learning_rate=0.0001",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "rejected"
    assert json.loads(metrics.read_text(encoding="utf-8"))["accepted"] == 0.0
    result = CliRunner().invoke(
        app, [*args, "--out", str(tmp_path / "b"), "--set", "optimizer.stepz=1"]
    )
    assert result.exit_code == 2 and "override path" in result.output
    result = CliRunner().invoke(
        app, [*args, "--out", str(tmp_path / "c"), "--set", "x"]
    )
    assert result.exit_code == 2


def _transformer_session():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import copy

    with torch.random.fork_rng():
        torch.manual_seed(17)
        config = transformers.Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            eos_token_id=15,
        )
        model = transformers.Qwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    parameter.normal_(0, 0.3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    letters = "abcdefghijklmno$"
    session = WeightSpaceSession(
        model,
        lambda text: [letters.index(c) for c in text],
        budget=200,
        batch_size=1,
        decode=lambda ids: "".join(letters[i] for i in ids if i != 15),
    )
    return session, copy.deepcopy(model)


def _transformer_recipe(**changes):
    # A random model answers nothing; the gates are opened so the export and
    # verification path runs on a real attention model and a real PEFT reload.
    base = EditRecipe(
        name="tiny",
        model="tiny",
        chat=False,
        subject="m",
        target="l",
        design=("abm", "cdm"),
        validation=("efm",),
        locality=(LocalityAnswer("ghi", ("j",)),),
        calibration=("abc", "def"),
        drift=("ghj", "kno"),
        sites=(EditSite("model.layers.0.mlp.down_proj", "subject"),),
        optimizer=KeyedEditConfig(steps=3, learning_rate=0.05),
        limits=BehaviorLimits(50.0, 50.0, 50.0),
        answer_limits=AnswerLimits(
            max_new_tokens=2,
            min_target_accuracy=0.0,
            min_baseline_locality_correct=0,
            max_answer_occurrences=None,
        ),
        export_limits=BehaviorLimits(1e-6, 1e-5, 1e-4, min_each_target_gain=-1e-4),
    )
    return replace(base, **changes)


@pytest.mark.parametrize("mode", ["merged", "adapter"])
def test_accepted_edit_is_exported_and_verified_through_peft(tmp_path, mode):
    s, fresh = _transformer_session()
    report = run_recipe(
        _transformer_recipe(),
        s,
        Text(),
        out_dir=tmp_path / "run",
        fresh_base_model=fresh,
        verify_mode=mode,
    )
    assert report.status == "accepted", report
    assert report.export["status"] == "passed" and report.export["mode"] == mode
    assert report.export["violations"] == [] and report.hints == ()
    assert s.active_edit_ids == [report.edit_id]
    saved = json.loads(
        (tmp_path / "run" / "adapter" / "weight_space_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"] == "passed"
    assert report.budget_used <= _transformer_recipe().required_budget()


def test_failed_verification_reverts_the_edit(tmp_path):
    s, fresh = _transformer_session()
    before = s.weight("model.layers.0.mlp.down_proj")
    # A reloaded adapter cannot beat the live edit by a full nat on the target.
    impossible = BehaviorLimits(50.0, 50.0, 50.0, min_each_target_gain=1.0)
    report = run_recipe(
        _transformer_recipe(export_limits=impossible),
        s,
        Text(),
        out_dir=tmp_path / "run",
        fresh_base_model=fresh,
    )
    assert report.status == "export_rejected" and not report.accepted
    assert report.edit_id is None and report.export["status"] == "rejected"
    assert s.active_edit_ids == []
    assert torch.equal(s.weight("model.layers.0.mlp.down_proj"), before)
    assert "reverted" in report.hints[0]


def test_verification_requires_an_output_directory():
    s, fresh = _transformer_session()
    with pytest.raises(ConfigurationError, match="needs out_dir"):
        run_recipe(_transformer_recipe(), s, Text(), fresh_base_model=fresh)
    assert s.budget.used == 0

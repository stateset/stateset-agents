"""lm-eval capability regression hook, driven by a fake harness (no real runs)."""

from __future__ import annotations

import json
import math
import sys

import pytest

from stateset_agents.evaluation import capability_regression as cr
from stateset_agents.evaluation.capability_regression import (
    CapabilityEvalSettings,
    CapabilityEvalUnavailable,
    compare_capability_reports,
    load_harness,
    run_capability_eval,
    validate_capability_report,
)


class FakeHarness:
    """Stands in for the ``lm_eval`` module: records the call, returns scores."""

    __version__ = "0.4.9-fake"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def simple_evaluate(self, **kwargs):
        self.calls.append(kwargs)
        results = {}
        for task in kwargs["tasks"]:
            entry = {"alias": task}
            entry.update(self.scores[task])
            results[task] = entry
        return {
            "results": results,
            "versions": dict.fromkeys(kwargs["tasks"], 1.0),
            "n-shot": dict.fromkeys(kwargs["tasks"], kwargs["num_fewshot"] or 0),
            "n-samples": {
                task: {"original": 1000, "effective": 100} for task in kwargs["tasks"]
            },
            "higher_is_better": {
                "wikitext": {"word_perplexity": False},
            },
        }


BASE_SCORES = {
    "hellaswag": {
        "acc,none": 0.52,
        "acc_stderr,none": 0.005,
        "acc_norm,none": 0.70,
        "acc_norm_stderr,none": 0.004,
    },
    "gsm8k": {
        "exact_match,strict-match": 0.40,
        "exact_match_stderr,strict-match": 0.013,
        "exact_match,flexible-extract": 0.45,
    },
    "wikitext": {
        "word_perplexity,none": 12.0,
        "bits_per_byte,none": 0.6,
        "word_perplexity_stderr,none": "N/A",
    },
}
SETTINGS = CapabilityEvalSettings(
    tasks=("hellaswag", "gsm8k", "wikitext"), num_fewshot=0, limit=100
)


def report_for(scores, *, adapter=None, settings=SETTINGS):
    return run_capability_eval(
        "Qwen/Qwen3.5-0.8B", settings, adapter=adapter, harness=FakeHarness(scores)
    )


def shifted(task, metric, delta):
    scores = json.loads(json.dumps(BASE_SCORES))
    scores[task][metric] += delta
    return scores


def test_run_passes_model_args_tasks_and_seeds_to_the_harness():
    harness = FakeHarness(BASE_SCORES)
    run_capability_eval(
        "Qwen/Qwen3.5-0.8B",
        SETTINGS,
        adapter="outputs/sft_v1",
        revision="a" * 40,
        harness=harness,
    )
    (call,) = harness.calls
    assert call["model"] == "hf"
    assert call["model_args"] == (
        "pretrained=Qwen/Qwen3.5-0.8B,dtype=bfloat16,peft=outputs/sft_v1,revision="
        + "a" * 40
    )
    assert call["tasks"] == ["hellaswag", "gsm8k", "wikitext"]
    assert call["num_fewshot"] == 0 and call["limit"] == 100
    assert call["random_seed"] == call["torch_random_seed"] == 1234
    assert call["log_samples"] is False


def test_report_uses_the_repo_report_shape_and_headline_metrics():
    report = report_for(BASE_SCORES, adapter="outputs/sft_v1")
    assert report["schema_version"] == 1 and report["kind"] == "capability_eval"
    assert report["base_model"] == "Qwen/Qwen3.5-0.8B" and report["seed"] == 1234
    assert report["selected_checkpoint"]["path"] == "outputs/sft_v1"
    assert report["harness"] == {
        "name": "lm-evaluation-harness",
        "version": "0.4.9-fake",
    }
    hellaswag = report["tasks"]["hellaswag"]
    assert hellaswag["primary_metric"] == "acc_norm" and hellaswag["score"] == 0.70
    assert hellaswag["stderr"] == {"acc": 0.005, "acc_norm": 0.004}
    assert hellaswag["samples"] == 100
    gsm8k = report["tasks"]["gsm8k"]
    assert gsm8k["primary_metric"] == "exact_match,strict-match"
    assert gsm8k["higher_is_better"] is True
    wikitext = report["tasks"]["wikitext"]
    assert wikitext["primary_metric"] == "word_perplexity"
    assert wikitext["higher_is_better"] is False
    assert wikitext["stderr"] == {}  # "N/A" is dropped, never NaN
    assert report["metrics"] == {"hellaswag": 0.70, "gsm8k": 0.40, "wikitext": 12.0}
    json.dumps(report, allow_nan=False)


def test_primary_metric_override_and_missing_metric():
    report = run_capability_eval(
        "m",
        SETTINGS,
        harness=FakeHarness(BASE_SCORES),
        primary_metrics={"hellaswag": "acc"},
    )
    assert report["tasks"]["hellaswag"]["score"] == 0.52
    with pytest.raises(ValueError, match="no metric 'f1'"):
        run_capability_eval(
            "m",
            SETTINGS,
            harness=FakeHarness(BASE_SCORES),
            primary_metrics={"gsm8k": "f1"},
        )


def test_missing_tasks_and_bad_harness_output_are_rejected():
    partial = {k: v for k, v in BASE_SCORES.items() if k != "gsm8k"}

    class Partial(FakeHarness):
        def simple_evaluate(self, **kwargs):
            return super().simple_evaluate(
                **{**kwargs, "tasks": ["hellaswag", "wikitext"]}
            )

    with pytest.raises(ValueError, match="missing tasks"):
        run_capability_eval("m", SETTINGS, harness=Partial(partial))

    class Nothing:
        def simple_evaluate(self, **kwargs):
            return None

    with pytest.raises(ValueError, match="no results"):
        run_capability_eval("m", SETTINGS, harness=Nothing())
    with pytest.raises(ValueError, match="without ','"):
        run_capability_eval(
            "m,trust_remote_code=True", SETTINGS, harness=FakeHarness(BASE_SCORES)
        )


@pytest.mark.parametrize(
    "options",
    [
        {"tasks": ()},
        {"tasks": ("a", "a")},
        {"tasks": ("a",), "num_fewshot": -1},
        {"tasks": ("a",), "limit": 0},
        {"tasks": ("a",), "limit": 1.5},
        {"tasks": ("a",), "batch_size": 0},
        {"tasks": ("a",), "seed": -2},
    ],
)
def test_settings_validation(options):
    with pytest.raises(ValueError):
        CapabilityEvalSettings(**options)


def test_comparison_passes_within_the_allowed_regression():
    baseline = report_for(BASE_SCORES)
    candidate = report_for(shifted("hellaswag", "acc_norm,none", -0.005), adapter="a")
    gate = compare_capability_reports(baseline, candidate, max_regression=0.01)
    assert gate["passed"] is True and gate["regressions"] == []
    task = gate["tasks"]["hellaswag"]
    assert task["delta"] == pytest.approx(-0.005)
    assert task["regression"] == pytest.approx(0.005)
    assert task["baseline_stderr"] == 0.004
    assert gate["evidence_hashes"]["baseline"] != gate["evidence_hashes"]["candidate"]
    assert gate["checkpoints"]["candidate"]["adapter"] == "a"


def test_comparison_fails_on_a_regression_beyond_the_threshold():
    baseline = report_for(BASE_SCORES)
    candidate = report_for(shifted("gsm8k", "exact_match,strict-match", -0.05))
    gate = compare_capability_reports(baseline, candidate, max_regression=0.02)
    assert gate["passed"] is False
    assert gate["regressions"] == ["gsm8k"]
    assert gate["gates"]["no_task_regression"] is False
    # A per-task allowance can accept it explicitly.
    loose = compare_capability_reports(
        baseline, candidate, max_regression=0.02, task_max_regression={"gsm8k": 0.06}
    )
    assert loose["passed"] is True
    # Relative thresholds scale with the baseline score (0.40 * 0.1 = 0.04).
    relative = compare_capability_reports(
        baseline, candidate, max_regression=0.1, relative=True
    )
    assert relative["tasks"]["gsm8k"]["allowed_regression"] == pytest.approx(0.04)
    assert relative["passed"] is False


def test_lower_is_better_metrics_regress_when_they_rise():
    baseline = report_for(BASE_SCORES)
    worse = report_for(shifted("wikitext", "word_perplexity,none", +0.5))
    better = report_for(shifted("wikitext", "word_perplexity,none", -3.0))
    assert compare_capability_reports(baseline, worse, max_regression=0.1)[
        "regressions"
    ] == ["wikitext"]
    assert (
        compare_capability_reports(baseline, better, max_regression=0.1)["passed"]
        is True
    )


def test_comparison_requires_matching_settings_and_tasks():
    baseline = report_for(BASE_SCORES)
    other = CapabilityEvalSettings(tasks=SETTINGS.tasks, num_fewshot=5, limit=100)
    gate = compare_capability_reports(baseline, report_for(BASE_SCORES, settings=other))
    assert gate["passed"] is False and gate["gates"]["same_settings"] is False
    # Batch size is a throughput knob, not part of the measurement.
    batched = CapabilityEvalSettings(
        tasks=SETTINGS.tasks, num_fewshot=0, limit=100, batch_size=8
    )
    assert compare_capability_reports(
        baseline, report_for(BASE_SCORES, settings=batched)
    )["passed"]
    fewer = CapabilityEvalSettings(
        tasks=("hellaswag", "wikitext"), num_fewshot=0, limit=100
    )
    gate = compare_capability_reports(baseline, report_for(BASE_SCORES, settings=fewer))
    assert gate["gates"]["all_tasks_scored"] is False and "gsm8k" in gate["regressions"]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_capability_reports(baseline, baseline, max_regression=-0.1)
    with pytest.raises(ValueError, match="baseline lacks"):
        compare_capability_reports(
            baseline, baseline, task_max_regression={"mmlu": 0.1}
        )


def test_validation_rejects_tampered_reports():
    report = report_for(BASE_SCORES)
    validate_capability_report(report)
    for mutate in (
        lambda r: r.update(kind="other"),
        lambda r: r["tasks"].pop("gsm8k"),
        lambda r: r["tasks"]["gsm8k"].update(score=math.nan),
        lambda r: r["tasks"]["gsm8k"].update(primary_metric="f1"),
        lambda r: r["settings"].update(extra=True),
        lambda r: r.update(selected_checkpoint={}),
    ):
        broken = json.loads(json.dumps(report))
        mutate(broken)
        with pytest.raises(ValueError):
            validate_capability_report(broken)


def test_missing_lm_eval_raises_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    with pytest.raises(
        CapabilityEvalUnavailable, match=r"stateset-agents\[eval-harness\]"
    ):
        load_harness()
    with pytest.raises(ImportError):
        run_capability_eval("m", SETTINGS)


def test_cli_run_and_compare(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cr, "load_harness", lambda: FakeHarness(BASE_SCORES))
    base = tmp_path / "base.json"
    argv = [
        "run",
        "--checkpoint",
        "m",
        "--tasks",
        "hellaswag,gsm8k,wikitext",
        "--num-fewshot",
        "0",
        "--limit",
        "100",
    ]
    assert cr.main([*argv, "--output", str(base)]) == 0
    assert json.loads(base.read_text())["settings"]["limit"] == 100
    assert cr.main([*argv, "--output", str(base)]) == 2  # never overwrites

    monkeypatch.setattr(
        cr,
        "load_harness",
        lambda: FakeHarness(shifted("hellaswag", "acc_norm,none", -0.1)),
    )
    candidate = tmp_path / "candidate.json"
    assert cr.main([*argv, "--adapter", "out/sft", "--output", str(candidate)]) == 0
    gate_path = tmp_path / "gate.json"
    code = cr.main(
        [
            "compare",
            "--baseline",
            str(base),
            "--candidate",
            str(candidate),
            "--output",
            str(gate_path),
        ]
    )
    assert code == 1
    assert json.loads(gate_path.read_text())["regressions"] == ["hellaswag"]
    assert (
        cr.main(
            [
                "compare",
                "--baseline",
                str(base),
                "--candidate",
                str(candidate),
                "--task-max-regression",
                "hellaswag=0.2",
            ]
        )
        == 0
    )
    assert (
        cr.main(
            [
                "compare",
                "--baseline",
                str(base),
                "--candidate",
                str(candidate),
                "--task-max-regression",
                "nonsense",
            ]
        )
        == 2
    )


def test_cli_reports_a_missing_harness(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    code = cr.main(
        [
            "run",
            "--checkpoint",
            "m",
            "--tasks",
            "arc_easy",
            "--output",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 2
    assert "eval-harness" in capsys.readouterr().err

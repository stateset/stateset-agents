"""Held-out evidence validation and paired improvement gate regressions."""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    compare_study,
    content_hash,
    evaluation_report,
    summarize_outcomes,
    validate_report,
)


def test_offline_evidence_imports_do_not_load_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import examples
from stateset_agents.evaluation.agent_runs import compare_runs
assert 'examples.customer_service_agent' not in sys.modules
assert not any(name in sys.modules for name in ('torch', 'transformers', 'river_client'))
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "package,submodule,export",
    [
        ("examples", "customer_service_agent", "CustomerServiceAgent"),
        ("stateset_agents.evaluation", "sim_to_real_metrics", "SimToRealEvaluator"),
    ],
)
def test_lazy_package_exports_preserve_public_api(
    monkeypatch, package, submodule, export
):
    import importlib

    module = importlib.import_module(package)
    sentinel = object()
    monkeypatch.delitem(module.__dict__, export, raising=False)
    monkeypatch.setitem(
        sys.modules, f"{package}.{submodule}", SimpleNamespace(**{export: sentinel})
    )
    try:
        assert getattr(module, export) is sentinel
        with pytest.raises(AttributeError):
            _ = module.unknown_export
    finally:
        module.__dict__.pop(export, None)


def run(seed=42, successes=2):
    return evaluation_report(
        environment="refund-v1",
        base_model="base",
        seed=seed,
        cases={str(i): {"order_id": str(i), "amount_cents": 100} for i in range(4)},
        settings={"max_generated_tokens": 100, "temperature": 0},
        checkpoint={"path": f"river://model-{seed}-{successes}"},
        outcomes=[
            {
                "case_id": str(i),
                "success": i < successes,
                "reward": 1 if i < successes else -1,
                "policy_violations": 0,
                "tool_calls": 3,
                "generated_tokens": 20,
                "elapsed_seconds": 0.5,
                "truncated": None,
            }
            for i in range(4)
        ],
    )


def test_matched_seeds_and_case_pairing_produce_transparent_evidence():
    before = [run(seed) for seed in (1, 2, 3)]
    after = [run(seed, 3) for seed in (3, 1, 2)]
    for report in after:
        report["outcomes"].reverse()  # asynchronous completion order is irrelevant
    result = compare_runs(before, after)
    assert result["passed"]
    assert result["mean_success_gain"] == 0.25
    assert result["seed_gain_stddev"] == 0
    assert all(p["wins"] == 1 and p["regressions"] == 0 for p in result["pairs"])
    assert result["pairs"][0]["candidate"]["generated_tokens"] == 80


def test_uncertainty_is_nonzero_even_for_perfect_sample():
    report = run(successes=4)
    low, high = report["success_wilson_95"]
    assert 0 < low < 1 and high == pytest.approx(1)


def test_resource_comparison_counts_failed_attempts_and_ignores_cached_totals():
    baseline, candidate = run(successes=2), run(successes=3)
    for report, usage in ((baseline, [100, 100, 300, 300]), (candidate, [100] * 4)):
        for outcome, tokens in zip(report["outcomes"], usage, strict=True):
            outcome["generated_tokens"] = tokens
        report["generated_tokens"] = 0  # Cached values must not drive economics.
        report["tool_calls"] = 0
    candidate["outcomes"][3]["tool_calls"] = 5
    candidate["outcomes"][3]["elapsed_seconds"] = 1.5
    pair = compare_runs([baseline], [candidate], min_seeds=1)["pairs"][0]
    assert pair["resource_usage"] == {
        "baseline": {
            "generated_tokens_per_case": 200,
            "generated_tokens_per_success": 400,
        },
        "candidate": {
            "generated_tokens_per_case": 100,
            "generated_tokens_per_success": pytest.approx(400 / 3),
        },
        "candidate_minus_baseline": {
            "generated_tokens": -400,
            "tool_calls": 2,
            "trajectory_seconds": 1,
        },
    }
    assert pair["baseline"]["tool_calls"] == 12
    assert pair["candidate"]["tool_calls"] == 14


def test_zero_success_has_unknown_cost_per_success_including_family_reports():
    baseline, candidate = run(successes=0), run(successes=2)
    for report in (baseline, candidate):
        report["case_families"] = {
            str(i): "easy" if i < 2 else "hard" for i in range(4)
        }
    pair = compare_runs([baseline], [candidate], min_seeds=1)["pairs"][0]
    assert pair["resource_usage"]["baseline"]["generated_tokens_per_success"] is None
    easy = pair["families"]["easy"]["resource_usage"]
    hard = pair["families"]["hard"]["resource_usage"]
    assert easy["candidate"]["generated_tokens_per_case"] == 20
    assert easy["candidate"]["generated_tokens_per_success"] == 20
    assert hard["candidate"]["generated_tokens_per_success"] is None
    # Every derived value can be published as strict JSON, including zero-success arms.
    json.dumps(pair, allow_nan=False)


def test_more_usage_is_visible_without_changing_learning_gates():
    baseline, candidate = run(), run(successes=3)
    original = compare_runs([baseline], [candidate], min_seeds=1)
    for outcome in candidate["outcomes"]:
        outcome.update(generated_tokens=200, tool_calls=30, elapsed_seconds=5)
    changed = compare_runs([baseline], [candidate], min_seeds=1)
    assert original["gates"] == changed["gates"]
    assert original["inference"] == changed["inference"]
    assert changed["passed"]
    assert changed["pairs"][0]["resource_usage"]["candidate_minus_baseline"] == {
        "generated_tokens": 720,
        "tool_calls": 108,
        "trajectory_seconds": 18,
    }


def test_usage_deltas_preserve_large_integer_counters():
    baseline, candidate = run(), run(successes=3)
    for report in (baseline, candidate):
        report["outcomes"][0].update(generated_tokens=2**53, tool_calls=2**53)
    candidate["outcomes"][0]["generated_tokens"] += 1
    candidate["outcomes"][0]["tool_calls"] += 1
    pair = compare_runs([baseline], [candidate], min_seeds=1)["pairs"][0]
    delta = pair["resource_usage"]["candidate_minus_baseline"]
    assert type(delta["generated_tokens"]) is int and delta["generated_tokens"] == 1
    assert type(delta["tool_calls"]) is int and delta["tool_calls"] == 1


def test_unrepresentable_usage_ratio_is_rejected_explicitly():
    baseline, candidate = run(), run(successes=3)
    candidate["outcomes"][0]["generated_tokens"] = 10**400
    with pytest.raises(ValueError, match="Token efficiency"):
        compare_runs([baseline], [candidate], min_seeds=1)


def test_zero_recorded_tokens_with_success_remain_zero():
    baseline, candidate = run(successes=0), run(successes=3)
    for report in (baseline, candidate):
        for outcome in report["outcomes"]:
            outcome["generated_tokens"] = 0
    usage = compare_runs([baseline], [candidate], min_seeds=1)["pairs"][0][
        "resource_usage"
    ]
    assert usage["baseline"]["generated_tokens_per_case"] == 0
    assert usage["baseline"]["generated_tokens_per_success"] is None
    assert usage["candidate"]["generated_tokens_per_success"] == 0


def test_summary_keeps_integer_counters_exact_above_float_precision():
    outcomes = run()["outcomes"][:2]
    outcomes[0].update(generated_tokens=2**53 + 1, tool_calls=2**53 + 1)
    outcomes[1].update(generated_tokens=2, tool_calls=1)
    summary = summarize_outcomes(outcomes)
    assert type(summary["generated_tokens"]) is int
    assert summary["generated_tokens"] == 2**53 + 3
    assert summary["mean_tool_calls"] == 2**52 + 1


def test_integral_sdk_float_counters_remain_supported():
    outcomes = run()["outcomes"]
    for outcome in outcomes:
        outcome.update(generated_tokens=20.0, tool_calls=3.0, policy_violations=0.0)
    summary = summarize_outcomes(outcomes)
    assert summary["generated_tokens"] == 80
    assert summary["mean_tool_calls"] == 3


@pytest.mark.parametrize(
    "field", ["generated_tokens", "tool_calls", "policy_violations"]
)
@pytest.mark.parametrize("value", [True, -1, 0.5, float("nan"), float("inf"), "3"])
def test_invalid_counters_fail_with_actionable_errors(field, value):
    outcomes = run()["outcomes"]
    outcomes[0][field] = value
    with pytest.raises(ValueError, match=field):
        summarize_outcomes(outcomes)


@pytest.mark.parametrize(
    "field,value", [("elapsed_seconds", 10**400), ("reward", 10**400)]
)
def test_unrepresentable_scalar_metrics_raise_value_error(field, value):
    outcomes = run()["outcomes"]
    outcomes[0][field] = value
    with pytest.raises(ValueError, match=field):
        summarize_outcomes(outcomes)


@pytest.mark.parametrize(
    "field,value", [("elapsed_seconds", 1e308), ("tool_calls", 10**400)]
)
def test_nonfinite_aggregate_cannot_be_published(field, value):
    outcomes = run()["outcomes"]
    for outcome in outcomes:
        outcome[field] = value
    with pytest.raises(ValueError, match="aggregate"):
        summarize_outcomes(outcomes)


def test_aggregate_duration_is_independent_of_completion_order():
    outcomes = run()["outcomes"][:3]
    for outcome, seconds in zip(outcomes, (1e16, 1.0, 1.0), strict=True):
        outcome["elapsed_seconds"] = seconds
    before = summarize_outcomes(outcomes)
    after = summarize_outcomes(list(reversed(outcomes)))
    assert before == after
    assert before["trajectory_seconds"] == 1e16 + 2


def test_report_owns_nested_evidence_and_protocol_snapshots():
    original = run()
    outcomes = list(reversed(original["outcomes"]))
    for outcome in outcomes:
        outcome["environment_trace"] = {"steps": [{"action": {"content": "original"}}]}
    settings = {"sampling": {"temperature": 0}}
    checkpoint = {"path": "river://snapshot", "metadata": {"step": 1}}
    implementation = {
        "schema_version": 1,
        "scope": "stateset_agents_python_sources",
        "files": {"__init__.py": "a" * 64},
    }
    report = evaluation_report(
        environment="refund-v1",
        base_model="base",
        seed=42,
        cases={str(i): {"order_id": str(i)} for i in range(4)},
        settings=settings,
        checkpoint=checkpoint,
        outcomes=outcomes,
        implementation=implementation,
    )
    identity = content_hash(report)
    assert [o["case_id"] for o in outcomes] == ["3", "2", "1", "0"]
    settings["sampling"]["temperature"] = 1
    checkpoint["metadata"]["step"] = 999
    implementation["files"]["__init__.py"] = "b" * 64
    outcomes[-1]["success"] = False
    outcomes[-1]["environment_trace"]["steps"][0]["action"]["content"] = "changed"
    assert content_hash(report) == identity
    assert report["passed"] == validate_report(report)["passed"] == 2
    report["outcomes"][0]["environment_trace"]["steps"].clear()
    assert len(outcomes[-1]["environment_trace"]["steps"]) == 1


@pytest.mark.parametrize("mutation", ["candidate", "later_seed", "missing"])
def test_comparison_rejects_mixed_implementations(mutation):
    from stateset_agents.evaluation.implementation import package_implementation

    before = [run(seed) for seed in (1, 2, 3)]
    after = [run(seed, 3) for seed in (1, 2, 3)]
    identity = package_implementation()
    for report in before + after:
        report["implementation"] = copy.deepcopy(identity)
    assert compare_runs(before, after)["protocol"]["implementation"] == identity
    if mutation == "missing":
        del after[0]["implementation"]
    elif mutation == "candidate":
        after[0]["implementation"]["files"]["__init__.py"] = "0" * 64
    else:
        for report in (before[1], after[1]):
            report["implementation"]["files"]["__init__.py"] = "0" * 64
    with pytest.raises(ValueError, match="Mismatched implementation"):
        compare_runs(before, after)


def test_explicitly_invalid_implementation_cannot_be_treated_as_legacy():
    report = run()
    report["implementation"] = None
    with pytest.raises(ValueError, match="Invalid implementation"):
        validate_report(report)


@pytest.mark.parametrize(
    "seeds,comparisons,passed",
    [(3, 1, False), (5, 1, True), (5, 3, False), (6, 3, True)],
)
def test_exact_seed_test_and_multiple_comparison_gate(seeds, comparisons, passed):
    result = compare_runs(
        [run(s) for s in range(seeds)],
        [run(s, 3) for s in range(seeds)],
        require_significance=True,
        comparisons=comparisons,
    )
    assert result["passed"] is passed
    assert result["inference"]["p_value"] == 2**-seeds
    assert result["inference"]["bonferroni_adjusted_p_value"] == comparisons * 2**-seeds


def test_sign_test_counts_losses_and_excludes_ties():
    result = compare_runs(
        [run(s) for s in range(4)],
        [run(s, n) for s, n in enumerate([3, 3, 1, 2])],
    )
    assert result["inference"]["p_value"] == 0.5  # P(Binomial(3, .5) >= 2)
    assert result["inference"]["tied_seeds"] == 1
    result = compare_runs([run()], [run()])
    assert result["inference"]["p_value"] == 1


def test_reusing_one_checkpoint_cannot_pass_independent_seed_gate():
    candidates = [run(s, 3) for s in range(5)]
    for candidate in candidates:
        candidate["selected_checkpoint"]["path"] = "river://same-model"
    result = compare_runs(
        [run(s) for s in range(5)], candidates, require_significance=True
    )
    assert result["gates"]["seed_significance"]
    assert not result["gates"]["distinct_candidate_checkpoints"]
    assert not result["passed"]


@pytest.mark.parametrize(
    "options",
    [
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": float("nan")},
        {"comparisons": 0},
        {"comparisons": True},
        {"require_significance": 1},
    ],
)
def test_invalid_inference_options_are_rejected(options):
    with pytest.raises(ValueError):
        compare_runs([run()], [run(successes=3)], **options)


def four_arm_study():
    arms = {}
    for successes, name in enumerate(("base", "sft", "rejection_sft", "rl")):
        arms[name] = [run(seed, successes) for seed in range(6)]
    return arms


def test_four_arm_study_corrects_all_comparisons():
    result = compare_study(four_arm_study())
    assert result["passed"]
    assert set(result["comparisons"]) == {"base", "sft", "rejection_sft"}
    for comparison in result["comparisons"].values():
        assert comparison["inference"]["bonferroni_adjusted_p_value"] == 3 / 64


def test_study_cannot_hide_the_strongest_baseline_or_missing_seed():
    arms = four_arm_study()
    arms["sft"] = [run(seed, 4) for seed in range(6)]
    result = compare_study(arms)
    assert not result["passed"] and not result["gates"]["rl_beats_sft"]
    assert result["gates"]["rl_beats_base"]
    arms["sft"].pop()
    with pytest.raises(ValueError, match="seed sets"):
        compare_study(arms)
    del arms["sft"]
    with pytest.raises(ValueError, match="exactly"):
        compare_study(arms)


@pytest.mark.parametrize("source", ["base", "sft", "rejection_sft"])
def test_study_rejects_reused_trained_checkpoints(source):
    arms = four_arm_study()
    arms["rl"][0]["selected_checkpoint"] = arms[source][0]["selected_checkpoint"]
    result = compare_study(arms)
    assert not result["passed"]


def test_study_cli_reports_all_arms_and_preserves_evidence(tmp_path):
    arms = four_arm_study()
    args = ["benchmark", "compare-agent-study", "--strict"]
    for name, reports in arms.items():
        for report in reports:
            path = tmp_path / f"{name}-{report['seed']}.json"
            path.write_text(json.dumps(report))
            args.extend(["--" + name.replace("_", "-"), str(path)])
    output = tmp_path / "study.json"
    result = CliRunner().invoke(app, args + ["--output", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["passed"]
    result = CliRunner().invoke(
        app, args + ["--output", str(output), "--min-seeds", "7"]
    )
    assert result.exit_code == 1, result.output
    assert not json.loads(output.read_text())["passed"]
    original = path.read_text()
    result = CliRunner().invoke(app, args + ["--output", str(path)])
    assert result.exit_code == 2
    assert path.read_text() == original


def test_family_regression_cannot_hide_behind_improved_average():
    baseline, candidate = run(successes=1), run(successes=4)
    families = {"0": "chargeback", "1": "ordinary", "2": "ordinary", "3": "ordinary"}
    baseline["case_families"] = candidate["case_families"] = families
    candidate["outcomes"][0].update(success=False, reward=-1)
    result = compare_runs([baseline], [candidate], min_seeds=1)
    assert result["mean_success_gain"] == 0.5
    assert result["gates"]["mean_success_gain"]
    assert not result["passed"] and not result["gates"]["no_family_regression"]
    assert result["pairs"][0]["families"]["chargeback"]["success_gain"] == -1
    assert result["pairs"][0]["families"]["ordinary"]["candidate"]["total"] == 3


@pytest.mark.parametrize(
    "metric,value", [("policy_violations", 1), ("truncated", "length")]
)
def test_family_safety_regression_is_detected_even_when_aggregate_rate_is_unchanged(
    metric, value
):
    baseline, candidate = run(successes=0), run(successes=2)
    labels = {"0": "ordinary", "1": "ordinary", "2": "hard", "3": "hard"}
    baseline["case_families"] = candidate["case_families"] = labels
    baseline["outcomes"][0][metric] = value
    candidate["outcomes"][2][metric] = value
    result = compare_runs([baseline], [candidate], min_seeds=1)
    assert result["gates"]["mean_success_gain"]
    assert result["gates"]["no_violation_regression"]
    assert result["gates"]["no_truncation_regression"]
    assert not result["gates"]["no_family_regression"]


@pytest.mark.parametrize(
    "labels", [{"0": "only_one"}, {str(i): " " for i in range(4)}, None]
)
def test_incomplete_family_labels_are_rejected(labels):
    report = run()
    report["case_families"] = labels
    with pytest.raises(ValueError, match="label every"):
        validate_report(report)


def test_comparisons_reject_removed_or_reassigned_family_labels():
    baseline, candidate = run(), run(successes=3)
    baseline["case_families"] = {str(i): "ordinary" for i in range(4)}
    with pytest.raises(ValueError, match="families differ"):
        compare_runs([baseline], [candidate])
    candidate["case_families"] = {str(i): "hard" for i in range(4)}
    with pytest.raises(ValueError, match="families differ"):
        compare_runs([baseline], [candidate])


def test_scenario_coverage_must_match_between_seeds():
    reports = [run(seed) for seed in (1, 2)]
    for i, report in enumerate(reports):
        report["case_families"] = {str(k): f"family-{i}" for k in range(4)}
    with pytest.raises(ValueError, match="coverage"):
        compare_runs(reports, copy.deepcopy(reports))


@pytest.mark.parametrize(
    "field,value",
    [
        ("reward", float("nan")),
        ("generated_tokens", -1),
        ("generated_tokens", 1.5),
        ("policy_violations", True),
        ("elapsed_seconds", float("inf")),
        ("success", 1),
        ("truncated", False),
        ("tool_calls", -1),
    ],
)
def test_invalid_outcome_data_is_rejected(field, value):
    report = run()
    report["outcomes"][0][field] = value
    with pytest.raises(ValueError):
        validate_report(report)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "unknown", "truncated_success", "unsafe_success"],
)
def test_incomplete_or_impossible_outcomes_are_rejected(mutation):
    report = run()
    if mutation == "missing":
        report["outcomes"].pop()
    elif mutation == "duplicate":
        report["outcomes"].append(report["outcomes"][0])
    elif mutation == "unknown":
        report["outcomes"][0]["case_id"] = "unknown"
    elif mutation == "truncated_success":
        report["outcomes"][0]["truncated"] = "length"
    else:
        report["outcomes"][0]["policy_violations"] = 1
    with pytest.raises(ValueError):
        validate_report(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "refund-v2"),
        ("base_model", "larger-model"),
        ("settings", {"max_generated_tokens": 1000}),
        ("case_hashes", {str(i): "a" * 64 for i in range(4)}),
    ],
)
def test_incomparable_protocols_and_changed_cases_fail(field, value):
    candidate = run(successes=3)
    candidate[field] = value
    with pytest.raises(ValueError, match="Mismatched|cases differ"):
        compare_runs([run()], [candidate])


def test_duplicate_or_unpaired_seeds_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        compare_runs([run(), run()], [run()])
    with pytest.raises(ValueError, match="seed sets"):
        compare_runs([run(1)], [run(2)])


def test_one_seed_and_stale_top_level_totals_cannot_fake_a_pass():
    candidate = run(successes=3)
    candidate["passed"], candidate["success_rate"] = 999, 1
    result = compare_runs([run()], [candidate])
    assert not result["passed"]
    assert not result["gates"]["enough_seeds"]
    assert result["mean_success_gain"] == 0.25


@pytest.mark.parametrize(
    "kind,gate",
    [
        ("policy_violations", "no_violation_regression"),
        ("truncated", "no_truncation_regression"),
    ],
)
def test_success_gain_cannot_mask_safety_or_truncation_regression(kind, gate):
    candidate = run(successes=3)
    candidate["outcomes"][-1][kind] = 1 if kind == "policy_violations" else "length"
    result = compare_runs([run()], [candidate], min_seeds=1)
    assert not result["passed"] and not result["gates"][gate]
    assert result["gates"]["mean_success_gain"]


def test_mean_gain_cannot_mask_a_regressing_seed():
    result = compare_runs([run(1), run(2)], [run(1, 4), run(2, 1)], min_seeds=2)
    assert result["gates"]["mean_success_gain"]
    assert not result["gates"]["improved_every_seed"]
    assert not result["passed"]


def test_cli_emits_failure_report_and_preserves_source_evidence(tmp_path):
    baseline, candidate, output = [
        tmp_path / f"{name}.json" for name in ("base", "rl", "comparison")
    ]
    baseline.write_text(json.dumps(run()))
    candidate.write_text(json.dumps(run(successes=3)))
    args = [
        "benchmark",
        "compare-agents",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
        "--strict",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert not json.loads(output.read_text())["passed"]
    result = CliRunner().invoke(app, args + ["--min-seeds", "1"])
    assert result.exit_code == 0, result.output
    usage = json.loads(output.read_text())["pairs"][0]["resource_usage"]
    assert usage["baseline"]["generated_tokens_per_success"] == 40
    assert usage["candidate"]["generated_tokens_per_success"] == pytest.approx(80 / 3)
    result = CliRunner().invoke(
        app,
        args
        + [
            "--min-seeds",
            "1",
            "--require-significance",
            "--alpha",
            "0.01",
            "--comparisons",
            "3",
        ],
    )
    assert result.exit_code == 1, result.output
    inference = json.loads(output.read_text())["inference"]
    assert inference["alpha"] == 0.01 and inference["planned_comparisons"] == 3
    original = baseline.read_text()
    result = CliRunner().invoke(app, args + ["--output", str(baseline)])
    assert result.exit_code == 2
    assert baseline.read_text() == original


def prepared_args(tmp_path):
    return SimpleNamespace(
        output=tmp_path,
        base_model="base",
        seed=42,
        steps=20,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=True,
        dry_run=True,
    )


def test_manifest_allows_matching_resume_but_refuses_config_or_data_changes(tmp_path):
    from examples.river_refund_rl import prepare_run

    args = prepared_args(tmp_path)
    splits = {"test": [{"order_id": "A", "amount_cents": 100, "eligible": True}]}
    prepare_run(args, splits)
    args.dry_run = False
    prepare_run(args, splits)
    changed = copy.copy(args)
    changed.learning_rate *= 2
    with pytest.raises(ValueError, match="different experiment"):
        prepare_run(changed, splits)
    (tmp_path / "test.json").write_text("[]")
    with pytest.raises(ValueError, match="cases have changed"):
        prepare_run(args, splits)


def test_manifest_refuses_completed_test_overwrite_and_legacy_directory(tmp_path):
    from examples.river_refund_rl import prepare_run

    args = prepared_args(tmp_path)
    splits = {"test": [{"order_id": "A", "amount_cents": 100, "eligible": True}]}
    (tmp_path / "old_results.json").write_text("{}")
    with pytest.raises(ValueError, match="no run manifest"):
        prepare_run(args, splits)
    (tmp_path / "old_results.json").unlink()
    prepare_run(args, splits)
    (tmp_path / "test_results.json").write_text("{}")
    args.dry_run = False
    with pytest.raises(ValueError, match="already completed"):
        prepare_run(args, splits)

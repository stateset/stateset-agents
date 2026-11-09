"""The fan-out manifest: N independent runs, described in one validated file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.fanout.manifest import FanoutManifest, ManifestError


def _manifest(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "name": "demo",
        "seeds": [1, 2, 3],
        "workload": {
            "kind": "command",
            "argv": [
                "{python}",
                "train.py",
                "--seed",
                "{seed}",
                "--out",
                "{output_dir}",
            ],
        },
        "report": {"metric": "reward", "mode": "max"},
    }
    base.update(overrides)
    return base


class TestExpansion:
    def test_no_matrix_gives_one_arm_per_seed(self) -> None:
        runs = FanoutManifest.from_dict(_manifest()).expand()

        assert [run.seed for run in runs] == [1, 2, 3]
        assert len({run.arm_id for run in runs}) == 1
        assert len({run.run_id for run in runs}) == 3

    def test_matrix_is_the_cartesian_product_with_every_seed(self) -> None:
        manifest = FanoutManifest.from_dict(
            _manifest(
                seeds=[1, 2],
                matrix={"lr": [0.1, 0.2], "beta": [1, 2, 3]},
                workload={
                    "kind": "command",
                    "argv": ["t", "{seed}", "{output_dir}", "{lr}", "{beta}"],
                },
            )
        )

        runs = manifest.expand()

        assert len(runs) == 2 * 3 * 2
        assert len({run.arm_id for run in runs}) == 6
        assert {(run.params["lr"], run.params["beta"], run.seed) for run in runs} == {
            (lr, beta, seed)
            for lr in (0.1, 0.2)
            for beta in (1, 2, 3)
            for seed in (1, 2)
        }

    def test_run_ids_do_not_depend_on_matrix_key_order(self) -> None:
        argv = ["t", "{seed}", "{output_dir}", "{a}", "{b}"]
        first = FanoutManifest.from_dict(
            _manifest(
                matrix={"a": [1], "b": [2]}, workload={"kind": "command", "argv": argv}
            )
        )
        second = FanoutManifest.from_dict(
            _manifest(
                matrix={"b": [2], "a": [1]}, workload={"kind": "command", "argv": argv}
            )
        )

        assert [r.run_id for r in first.expand()] == [r.run_id for r in second.expand()]

    def test_run_ids_are_safe_as_a_single_path_component(self) -> None:
        runs = FanoutManifest.from_dict(
            _manifest(
                seeds=[7],
                matrix={"model": ["org/model:../../etc"]},
                workload={
                    "kind": "command",
                    "argv": ["t", "{seed}", "{output_dir}", "{model}"],
                },
            )
        ).expand()

        assert "/" not in runs[0].run_id
        assert ".." not in runs[0].run_id


class TestSweepsThatWouldSilentlyDoNothing:
    """Each of these would burn a paid fan-out producing fake or empty results."""

    def test_command_must_receive_the_seed(self) -> None:
        with pytest.raises(ManifestError, match=r"\{seed\}"):
            FanoutManifest.from_dict(
                _manifest(workload={"kind": "command", "argv": ["t", "{output_dir}"]})
            )

    def test_command_must_receive_the_output_dir(self) -> None:
        with pytest.raises(ManifestError, match=r"\{output_dir\}"):
            FanoutManifest.from_dict(
                _manifest(workload={"kind": "command", "argv": ["t", "{seed}"]})
            )

    def test_matrix_key_unused_by_the_command_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="lr"):
            FanoutManifest.from_dict(_manifest(matrix={"lr": [0.1, 0.2]}))

    def test_unknown_placeholder_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="typo"):
            FanoutManifest.from_dict(
                _manifest(
                    workload={
                        "kind": "command",
                        "argv": ["t", "{seed}", "{output_dir}", "{typo}"],
                    }
                )
            )

    def test_train_matrix_key_must_be_a_training_field(self) -> None:
        with pytest.raises(ManifestError, match="learning_rat"):
            FanoutManifest.from_dict(
                _manifest(
                    matrix={"learning_rat": [1e-5]},
                    workload={"kind": "train", "agent": {"model_name": "stub://x"}},
                )
            )

    def test_train_matrix_accepts_training_fields_and_objective(self) -> None:
        manifest = FanoutManifest.from_dict(
            _manifest(
                matrix={"learning_rate": [1e-5], "objective": ["gspo", "gspo_token"]},
                workload={"kind": "train", "agent": {"model_name": "stub://x"}},
            )
        )

        assert len(manifest.expand()) == 3 * 2

    def test_train_objective_must_be_a_known_preset(self) -> None:
        with pytest.raises(ManifestError, match="nope"):
            FanoutManifest.from_dict(
                _manifest(
                    matrix={"objective": ["nope"]},
                    workload={"kind": "train", "agent": {"model_name": "stub://x"}},
                )
            )

    @pytest.mark.parametrize("name", ["seed", "output_dir", "python", "run_id"])
    def test_matrix_key_cannot_shadow_a_builtin_placeholder(self, name: str) -> None:
        """The builtin wins, so every arm would run the identical command."""
        with pytest.raises(ManifestError, match=name):
            FanoutManifest.from_dict(
                _manifest(
                    matrix={name: [1, 2]},
                    workload={
                        "kind": "command",
                        "argv": ["t", "{seed}", "{output_dir}", f"{{{name}}}"],
                    },
                )
            )

    def test_train_objective_must_be_one_the_trainer_can_drive(self) -> None:
        """Otherwise every sandbox boots, installs, and then fails the same way."""
        with pytest.raises(ManifestError, match="dapo"):
            FanoutManifest.from_dict(
                _manifest(
                    matrix={"objective": ["dapo"]},
                    workload={"kind": "train", "agent": {"model_name": "stub://x"}},
                )
            )

    def test_train_objective_list_matches_the_real_trainer(self) -> None:
        """Guards the preflight against drifting from what the trainer accepts."""
        from types import SimpleNamespace

        from stateset_agents.fanout.manifest import TRAIN_OBJECTIVE_RATIOS
        from stateset_agents.training.loss_computation import _grpo_objective
        from stateset_agents.training.objectives import OBJECTIVES, compatible_presets

        allowed = set(compatible_presets(TRAIN_OBJECTIVE_RATIOS))
        for name in OBJECTIVES:
            config = SimpleNamespace(objective=name, objective_overrides=None)
            if name in allowed:
                _grpo_objective(config)
            else:
                with pytest.raises(ValueError):
                    _grpo_objective(config)

    def test_train_environment_typo_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="scenario"):
            FanoutManifest.from_dict(
                _manifest(
                    workload={
                        "kind": "train",
                        "agent": {"model_name": "stub://x"},
                        "environment": {"scenario": []},
                    },
                )
            )

    def test_train_reward_must_be_buildable(self) -> None:
        with pytest.raises(ManifestError, match="helpfullness"):
            FanoutManifest.from_dict(
                _manifest(
                    workload={
                        "kind": "train",
                        "agent": {"model_name": "stub://x"},
                        "reward": {"helpfullness": 1.0},
                    },
                )
            )

    def test_workflow_regression_refuses_unknown_params(self) -> None:
        with pytest.raises(ManifestError, match="temperature"):
            FanoutManifest.from_dict(
                _manifest(
                    matrix={"temperature": [1.0]},
                    workload={"kind": "workflow_regression"},
                )
            )

    def test_unknown_top_level_key_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="concurency"):
            FanoutManifest.from_dict(_manifest(concurency=4))

    def test_unknown_resource_key_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="cpu"):
            FanoutManifest.from_dict(_manifest(resources={"cpu": "2"}))

    def test_duplicate_seeds_are_not_replicates(self) -> None:
        with pytest.raises(ManifestError, match="unique"):
            FanoutManifest.from_dict(_manifest(seeds=[1, 1]))

    def test_matrix_explosion_is_refused_before_anything_runs(self) -> None:
        argv = ["t", "{seed}", "{output_dir}", "{a}", "{b}"]
        with pytest.raises(ManifestError, match="1000"):
            FanoutManifest.from_dict(
                _manifest(
                    seeds=list(range(100)),
                    matrix={"a": list(range(10)), "b": list(range(10))},
                    workload={"kind": "command", "argv": argv},
                )
            )


class TestShapeValidation:
    @pytest.mark.parametrize("bad", [[], "1", [True], [-1], [2**32], [1.5]])
    def test_bad_seeds(self, bad: Any) -> None:
        with pytest.raises(ManifestError, match="seeds"):
            FanoutManifest.from_dict(_manifest(seeds=bad))

    @pytest.mark.parametrize("mode", ["maximum", "", None])
    def test_bad_report_mode(self, mode: Any) -> None:
        with pytest.raises(ManifestError, match="mode"):
            FanoutManifest.from_dict(_manifest(report={"metric": "r", "mode": mode}))

    def test_unsupported_schema_version(self) -> None:
        with pytest.raises(ManifestError, match="schema_version"):
            FanoutManifest.from_dict(_manifest(schema_version=2))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), {"a": 1}, [1]])
    def test_matrix_values_must_be_finite_scalars(self, value: Any) -> None:
        argv = ["t", "{seed}", "{output_dir}", "{a}"]
        with pytest.raises(ManifestError, match="matrix"):
            FanoutManifest.from_dict(
                _manifest(
                    matrix={"a": [value]}, workload={"kind": "command", "argv": argv}
                )
            )

    @pytest.mark.parametrize("seconds", [59, 85801, 86400, 86401])
    def test_timeout_must_fit_a_non_durable_sandbox(self, seconds: int) -> None:
        with pytest.raises(ManifestError, match="timeout_seconds"):
            FanoutManifest.from_dict(_manifest(resources={"timeout_seconds": seconds}))

    def test_shared_capacity_cannot_have_a_gpu(self) -> None:
        with pytest.raises(ManifestError, match="shared"):
            FanoutManifest.from_dict(
                _manifest(
                    resources={
                        "resource_class": "shared",
                        "gpu": {"type": "nvidia-l4", "count": 1},
                    }
                )
            )

    @pytest.mark.parametrize("path", ["../escape.txt", "/abs/path.txt", "a/../../b"])
    def test_stage_destinations_cannot_escape(self, path: str, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="stage"):
            FanoutManifest.from_dict(
                _manifest(stage={path: "local.txt"}), base_dir=tmp_path
            )

    def test_stage_source_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="missing.txt"):
            FanoutManifest.from_dict(
                _manifest(stage={"data.txt": "missing.txt"}), base_dir=tmp_path
            )

    def test_stage_source_over_the_upload_limit_is_refused(
        self, tmp_path: Path
    ) -> None:
        big = tmp_path / "big.bin"
        big.write_bytes(b"\0" * (8 * 1024 * 1024))
        with pytest.raises(ManifestError, match="big.bin"):
            FanoutManifest.from_dict(
                _manifest(stage={"big.bin": "big.bin"}), base_dir=tmp_path
            )

    @pytest.mark.parametrize("pattern", ["../x", "/etc/passwd"])
    def test_collect_patterns_stay_inside_the_output(self, pattern: str) -> None:
        with pytest.raises(ManifestError, match="collect"):
            FanoutManifest.from_dict(_manifest(collect=[pattern]))


class TestSecrets:
    """The journal is written to disk and manifests get committed to repos."""

    @pytest.mark.parametrize(
        "name",
        ["HF_TOKEN", "OPENAI_API_KEY", "WANDB_APIKEY", "DB_PASSWORD", "MY_SECRET"],
    )
    def test_literal_secret_values_are_refused(self, name: str) -> None:
        with pytest.raises(ManifestError, match=name):
            FanoutManifest.from_dict(_manifest(env={name: "value"}))

    def test_plain_env_is_allowed(self) -> None:
        manifest = FanoutManifest.from_dict(_manifest(env={"LOG_LEVEL": "debug"}))

        assert manifest.env == {"LOG_LEVEL": "debug"}

    def test_forward_env_names_are_kept_without_values(self) -> None:
        manifest = FanoutManifest.from_dict(_manifest(forward_env=["HF_TOKEN"]))

        assert manifest.forward_env == ("HF_TOKEN",)
        assert "HF_TOKEN" in manifest.to_dict()["forward_env"]

    @pytest.mark.parametrize("name", ["STATESET_API_KEY", "STATESET_AUTH_TOKEN"])
    def test_control_plane_credentials_never_enter_a_workload(self, name: str) -> None:
        with pytest.raises(ManifestError, match=name):
            FanoutManifest.from_dict(_manifest(forward_env=[name]))


class TestSandboxLifetime:
    def test_sandbox_outlives_its_run_so_output_can_be_collected(self) -> None:
        """Expiry counts from create. Equal limits would delete the run's output."""
        resources = FanoutManifest.from_dict(
            _manifest(resources={"timeout_seconds": 600})
        ).resources

        assert resources.sandbox_lifetime_seconds >= 600 + 300

    def test_the_longest_run_still_fits_a_non_durable_sandbox(self) -> None:
        resources = FanoutManifest.from_dict(
            _manifest(resources={"timeout_seconds": 85800})
        ).resources

        assert resources.sandbox_lifetime_seconds <= 86400


class TestCost:
    def test_worst_case_covers_every_run_at_every_attempt(self) -> None:
        manifest = FanoutManifest.from_dict(
            _manifest(
                seeds=[1, 2],
                max_attempts=3,
                resources={"max_cost_cents": 25},
            )
        )

        # 2 runs x 3 attempts x $0.25: a retry bills again.
        assert manifest.worst_case_cost_usd() == pytest.approx(1.50)

    def test_worst_case_is_unknown_without_a_per_run_budget(self) -> None:
        assert FanoutManifest.from_dict(_manifest()).worst_case_cost_usd() is None


class TestIdentity:
    def test_digest_ignores_concurrency_so_a_resume_can_change_it(self) -> None:
        low = FanoutManifest.from_dict(_manifest(concurrency=2))
        high = FanoutManifest.from_dict(_manifest(concurrency=50))

        assert low.digest() == high.digest()

    def test_digest_tracks_what_the_runs_do(self) -> None:
        first = FanoutManifest.from_dict(_manifest())
        second = FanoutManifest.from_dict(_manifest(seeds=[1, 2, 4]))

        assert first.digest() != second.digest()

    def test_round_trips_through_its_serialized_form(self) -> None:
        manifest = FanoutManifest.from_dict(_manifest(concurrency=3))

        restored = FanoutManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

        assert restored.digest() == manifest.digest()
        assert restored.concurrency == 3

    def test_digest_tracks_staged_file_content(self, tmp_path: Path) -> None:
        data = tmp_path / "data.txt"
        data.write_text("one")
        before = FanoutManifest.from_dict(
            _manifest(stage={"data.txt": "data.txt"}), base_dir=tmp_path
        ).digest()

        data.write_text("two")
        after = FanoutManifest.from_dict(
            _manifest(stage={"data.txt": "data.txt"}), base_dir=tmp_path
        ).digest()

        assert before != after


class TestPackage:
    def test_pypi_source_needs_a_pinned_version(self) -> None:
        with pytest.raises(ManifestError, match="version"):
            FanoutManifest.from_dict(_manifest(package={"source": "pypi"}))

    def test_wheel_source_needs_an_existing_wheel(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="wheel"):
            FanoutManifest.from_dict(
                _manifest(package={"source": "wheel", "wheel": "nope.whl"}),
                base_dir=tmp_path,
            )

    def test_unknown_source_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="source"):
            FanoutManifest.from_dict(_manifest(package={"source": "conda"}))


class TestLoading:
    def test_loads_yaml_relative_to_its_own_directory(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("x")
        path = tmp_path / "fanout.yaml"
        path.write_text(
            "schema_version: 1\n"
            "name: demo\n"
            "seeds: [1]\n"
            "workload:\n"
            "  kind: command\n"
            "  argv: ['t', '{seed}', '{output_dir}']\n"
            "report: {metric: r, mode: max}\n"
            "stage: {data.txt: data.txt}\n"
        )

        manifest = FanoutManifest.from_file(path)

        assert manifest.stage_files() == {"data.txt": tmp_path / "data.txt"}

    def test_malformed_file_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("seeds: [1\n")

        with pytest.raises(ManifestError, match="broken.yaml"):
            FanoutManifest.from_file(path)

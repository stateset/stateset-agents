"""Small, private experiment API with sealed attempts and paired quality gates."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    content_hash,
    validate_report,
)
from stateset_agents.evaluation.implementation import package_implementation

from .datasets import verify_dataset
from .storage import private_directory, read_json, workspace_lock, write_json


@dataclass(frozen=True)
class ExperimentConfig:
    """Prespecified River continuation and held-out comparison protocol.

    The first seed selects the staging candidate before test inspection. Hosted
    checkpoints are mutable provider pointers, not immutable weight attestations.
    """

    base_model: str
    parent_checkpoint: str
    seeds: tuple[int, ...] = (42,)
    evidence_mode: str = "live"
    adapter_id: str = "river-refund-delegation-v1"
    environment: str = "refund-delegation-v1"
    min_gain: float = 0.03
    require_significance: bool = False
    min_cases: int = 32
    epochs: int = 1
    lora_rank: int = 16
    learning_rate: float = 2e-5
    batch_size: int = 2
    max_length: int = 1024
    request_timeout_seconds: int = 180
    evaluation: dict[str, Any] = field(
        default_factory=lambda: {
            "max_turns": 4,
            "max_generated_tokens": 1024,
            "max_context_tokens": 8192,
            "max_turn_tokens": 256,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": -1,
            "group_size": 1,
            "thinking": False,
            "truncation_reward": -1.0,
        }
    )

    def __post_init__(self) -> None:
        """Reject invalid protocols before any provider work is possible."""
        for value in (self.base_model, self.adapter_id, self.environment):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Model, adapter and environment must be nonempty")
        if not isinstance(
            self.parent_checkpoint, str
        ) or not self.parent_checkpoint.startswith("river://"):
            raise ValueError("An explicit River parent checkpoint is required")
        if self.evidence_mode not in ("live", "scripted"):
            raise ValueError("Unknown evidence mode")
        if (
            not self.seeds
            or any(type(s) is not int or not 0 <= s <= 2**32 - 1 for s in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("Seeds must be distinct uint32 values")
        if type(self.require_significance) is not bool:
            raise ValueError("require_significance must be boolean")
        if self.require_significance and len(self.seeds) < 6:
            raise ValueError("Significance mode requires at least six training seeds")
        for name in (
            "min_cases",
            "epochs",
            "lora_rank",
            "batch_size",
            "max_length",
            "request_timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_gain", "learning_rate"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if not isinstance(self.evaluation, dict) or not self.evaluation:
            raise ValueError("Evaluation protocol is required")
        content_hash(asdict(self))


class ExperimentBackend(Protocol):
    """Versioned training/evaluation adapter; implementations own provider clients."""

    evidence_mode: str
    adapter_id: str

    def train(
        self, config: ExperimentConfig, dataset: Path, output: Path, seed: int
    ) -> dict[str, Any]:
        """Return a checkpoint pointer; never automatically retry optimizer work."""
        ...

    def evaluate(
        self,
        config: ExperimentConfig,
        cases: dict[str, Any],
        checkpoint: str,
        seed: int,
        split: str,
        implementation: dict[str, Any],
        output: Path,
    ) -> dict[str, Any]:
        """Return a complete report, auditing tool traces before returning."""
        ...


class Experiment:
    """Resume completed stages; leave interrupted provider attempts sealed.

    Workspace records are private local provenance, not signed attestations.
    A lost training acknowledgment needs provider reconciliation and a new plan;
    rerunning this API never silently repeats that training seed.
    """

    def __init__(self, root: Path) -> None:
        self.root = private_directory(root)

    @classmethod
    def create(cls, root: Path, dataset: Path, config: ExperimentConfig) -> Experiment:
        """Freeze data, source identity, thresholds and checkpoint selection."""
        manifest = verify_dataset(dataset)
        experiment = cls(root)
        with workspace_lock(experiment.root):
            if any(p.name != ".workflow.lock" for p in experiment.root.iterdir()):
                raise ValueError("Experiment workspace must be empty")
            snapshot = private_directory(experiment.root / "dataset")
            for name in ("train.json", "validation.json", "test.json", "dataset.json"):
                write_json(snapshot / name, read_json(dataset / name))
            if manifest["review_sha256"] is not None:
                write_json(snapshot / "review.json", read_json(dataset / "review.json"))
            for split in ("validation", "test"):
                cases = experiment._cases(split)
                if len(cases) < config.min_cases:
                    raise ValueError(
                        "Held-out split is smaller than the planned minimum"
                    )
            experiment._validate_native_cases(config)
            plan = {
                "schema_version": 1,
                "experiment_id": str(uuid.uuid4()),
                "dataset_origin": str(dataset.expanduser().absolute()),
                "config": asdict(config),
                "dataset_sha256": content_hash(manifest),
                "implementation": package_implementation(),
                "selected_seed": config.seeds[0],
                "checkpoint_semantics": "hosted_mutable_pointer_weights_only_fresh_optimizer",
                "cost_usd": None,
                "evaluation_scope": (
                    "synthetic_policy_replay"
                    if config.adapter_id == "river-refund-delegation-v1"
                    else "adapter_reported"
                ),
                "production_quality_established": False,
            }
            write_json(experiment.root / "plan.json", plan)
        return experiment

    def _cases(self, split: str) -> dict[str, Any]:
        rows = read_json(self.root / "dataset" / f"{split}.json")
        cases: dict[str, Any] = {}
        for row in rows:
            case = row.get("evaluation_case")
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("family"), str)
                or not case["family"]
            ):
                raise ValueError(
                    "Held-out records require evaluation_case with a scenario family"
                )
            key = row["record_id"]
            if key in cases:
                raise ValueError("Duplicate held-out record identity")
            cases[key] = case
        return cases

    def _validate_native_cases(self, config: ExperimentConfig) -> None:
        if config.adapter_id != "river-refund-delegation-v1":
            return
        from stateset_agents.core.environments.refund_policy_environment import (
            RefundPolicyEnvironment,
        )

        if config.environment != "refund-delegation-v1":
            raise ValueError(
                "River adapter requires the synthetic delegation environment"
            )
        if (
            config.evaluation
            != ExperimentConfig(config.base_model, config.parent_checkpoint).evaluation
        ):
            raise ValueError("River adapter requires the fixed evaluation settings")
        if len(config.seeds) > 1:
            from stateset_agents.evaluation.replication import (
                validate_river_replication_seeds,
            )

            validate_river_replication_seeds(
                config.seeds,
                max_groups=max(
                    len(self._cases(split)) for split in ("validation", "test")
                ),
                group_size=1,
                max_sample_calls=1024,
            )
        seen: set[str] = set()
        for split in ("train", "validation", "test"):
            fingerprints: set[str] = set()
            for key, case in self._cases(split).items():
                RefundPolicyEnvironment.validate_scenario(case)
                if case["order_id"] != key:
                    raise ValueError("Refund record_id must equal order_id")
                fingerprints.add(RefundPolicyEnvironment.case_fingerprint(case))
            if seen & fingerprints:
                raise ValueError("Refund order facts overlap across dataset partitions")
            seen.update(fingerprints)

    def verify(self) -> dict[str, Any]:
        """Rehash data and source before continuing a frozen experiment."""
        plan = read_json(self.root / "plan.json")
        if plan.get("schema_version") != 1:
            raise ValueError("Unknown experiment schema")
        self._validate_native_cases(ExperimentConfig(**plan["config"]))
        if (
            content_hash(verify_dataset(self.root / "dataset"))
            != plan["dataset_sha256"]
        ):
            raise ValueError("Dataset manifest changed")
        if plan["implementation"] != package_implementation():
            raise ValueError("Package sources changed; use the original installation")
        return dict(plan)

    def _test_claim(self, plan: dict[str, Any], *, create: bool = False) -> None:
        origin = private_directory(Path(plan["dataset_origin"]))
        claim = {
            "experiment_id": plan["experiment_id"],
            "plan_sha256": content_hash(plan),
        }
        with workspace_lock(origin):
            if content_hash(verify_dataset(origin)) != plan["dataset_sha256"]:
                raise ValueError("Original dataset changed; retain its test-use ledger")
            path = origin / "test-use.json"
            if create and not path.exists():
                write_json(path, claim)
            if read_json(path) != claim:
                raise ValueError(
                    "This dataset test split was already consumed by another experiment"
                )

    def _backend(
        self, backend: ExperimentBackend, plan: dict[str, Any]
    ) -> ExperimentConfig:
        config = ExperimentConfig(**plan["config"])
        if (
            backend.adapter_id != config.adapter_id
            or backend.evidence_mode != config.evidence_mode
        ):
            raise ValueError("Backend does not match the sealed experiment protocol")
        return config

    def _attempt(self, name: str, plan: dict[str, Any]) -> Path:
        output = private_directory(self.root / name)
        write_json(output / "attempt.json", {"plan_sha256": content_hash(plan)})
        return output

    def _receipt(self, name: str, plan: dict[str, Any]) -> dict[str, Any]:
        directory = self.root / name
        attempt = read_json(directory / "attempt.json")
        receipt = read_json(directory / "result.json")
        if attempt != {"plan_sha256": content_hash(plan)} or receipt[
            "plan_sha256"
        ] != content_hash(plan):
            raise ValueError("Stage belongs to a different experiment")
        for key in ("pointer", "report"):
            if key in receipt and content_hash(receipt[key]) != receipt.get(
                f"{key}_sha256"
            ):
                raise ValueError("Stage payload changed")
        return dict(receipt)

    def train(self, backend: ExperimentBackend) -> None:
        """Train each seed once; retain unknown outcomes without resubmission."""
        with workspace_lock(self.root):
            plan = self.verify()
            config = self._backend(backend, plan)
            for seed in config.seeds:
                name = f"train-{seed}"
                if (self.root / name / "result.json").exists():
                    self._training(seed, plan)
                    continue
                output = self._attempt(name, plan)
                pointer = backend.train(
                    config, self.root / "dataset" / "train.json", output, seed
                )
                self._validate_pointer(pointer, config)
                write_json(
                    output / "result.json",
                    {
                        "plan_sha256": content_hash(plan),
                        "pointer": pointer,
                        "pointer_sha256": content_hash(pointer),
                        "cost_usd": None,
                    },
                )

    @staticmethod
    def _validate_pointer(pointer: dict[str, Any], config: ExperimentConfig) -> None:
        if (
            not isinstance(pointer, dict)
            or pointer.get("base_model") != config.base_model
            or not isinstance(pointer.get("checkpoint"), str)
            or not pointer["checkpoint"].startswith("river://")
            or pointer["checkpoint"] == config.parent_checkpoint
        ):
            raise ValueError(
                "Training must produce a distinct checkpoint for the planned model"
            )

    def _training(self, seed: int, plan: dict[str, Any]) -> dict[str, Any]:
        receipt = self._receipt(f"train-{seed}", plan)
        pointer = receipt["pointer"]
        self._validate_pointer(pointer, ExperimentConfig(**plan["config"]))
        if content_hash(pointer) != receipt["pointer_sha256"]:
            raise ValueError("Training pointer changed")
        return dict(pointer)

    def _validate_evaluation(
        self,
        report: dict[str, Any],
        plan: dict[str, Any],
        seed: int,
        split: str,
        checkpoint: str,
    ) -> None:
        validate_report(report, expected_split=split)
        config = plan["config"]
        cases = self._cases(split)
        expected = {
            "seed": seed,
            "environment": config["environment"],
            "base_model": config["base_model"],
            "settings": config["evaluation"],
            "implementation": plan["implementation"],
            "case_hashes": {key: content_hash(case) for key, case in cases.items()},
            "case_families": {key: case["family"] for key, case in cases.items()},
        }
        if (
            any(report.get(key) != value for key, value in expected.items())
            or report["selected_checkpoint"]["path"] != checkpoint
        ):
            raise ValueError(
                "Evaluation does not match planned cases, checkpoint or protocol"
            )

    def evaluate(
        self, backend: ExperimentBackend, *, split: str = "validation"
    ) -> None:
        """Audit paired validation first; open the test split only after it passes."""
        if split not in ("validation", "test"):
            raise ValueError("Expected validation or test")
        with workspace_lock(self.root):
            plan = self.verify()
            config = self._backend(backend, plan)
            if split == "test" and not self._comparison(plan, "validation")["passed"]:
                raise ValueError("Validation gate failed; test remains unopened")
            if split == "test":
                self._test_claim(plan, create=True)
            for seed in config.seeds:
                candidate = self._training(seed, plan)["checkpoint"]
                for arm, checkpoint in (
                    ("baseline", config.parent_checkpoint),
                    ("candidate", candidate),
                ):
                    name = f"{split}-{arm}-{seed}"
                    if (self.root / name / "result.json").exists():
                        self._report(plan, seed, split, arm, checkpoint)
                        continue
                    output = self._attempt(name, plan)
                    report = backend.evaluate(
                        config,
                        self._cases(split),
                        checkpoint,
                        seed,
                        split,
                        plan["implementation"],
                        output,
                    )
                    self._validate_evaluation(report, plan, seed, split, checkpoint)
                    write_json(
                        output / "result.json",
                        {
                            "plan_sha256": content_hash(plan),
                            "report": report,
                            "report_sha256": content_hash(report),
                        },
                    )

    def _report(
        self, plan: dict[str, Any], seed: int, split: str, arm: str, checkpoint: str
    ) -> dict[str, Any]:
        receipt = self._receipt(f"{split}-{arm}-{seed}", plan)
        report = receipt["report"]
        if receipt["report_sha256"] != content_hash(report):
            raise ValueError("Evaluation report changed")
        self._validate_evaluation(report, plan, seed, split, checkpoint)
        return dict(report)

    def _comparison(self, plan: dict[str, Any], split: str) -> dict[str, Any]:
        config = ExperimentConfig(**plan["config"])
        if split == "test":
            self._test_claim(plan)
        baselines, candidates = [], []
        for seed in config.seeds:
            baselines.append(
                self._report(plan, seed, split, "baseline", config.parent_checkpoint)
            )
            candidates.append(
                self._report(
                    plan,
                    seed,
                    split,
                    "candidate",
                    self._training(seed, plan)["checkpoint"],
                )
            )
        result = compare_runs(
            baselines,
            candidates,
            min_seeds=len(config.seeds),
            min_gain=config.min_gain,
            require_significance=config.require_significance,
            expected_split=split,
        )
        result["gates"]["zero_candidate_safety_failures"] = all(
            o["policy_violations"] == 0 and o["truncated"] is None
            for r in candidates
            for o in r["outcomes"]
        )
        result["gates"]["distinct_training_checkpoints"] = len(
            {r["selected_checkpoint"]["path"] for r in candidates}
        ) == len(config.seeds)
        result["passed"] = all(result["gates"].values())
        return dict(result)

    def compare(self, *, split: str = "test") -> dict[str, Any]:
        """Return a private paired comparison, recomputing every gate from reports."""
        if split not in ("validation", "test"):
            raise ValueError("Expected validation or test")
        with workspace_lock(self.root):
            return self._comparison(self.verify(), split)

    def approve(self, *, reviewer: str) -> dict[str, Any]:
        """Record explicit staging approval; never certify hosted production weights."""
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("An operator reviewer is required")
        with workspace_lock(self.root):
            plan = self.verify()
            comparison = self._comparison(plan, "test")
            if not comparison["passed"]:
                raise ValueError("Test quality gate failed")
            approval = {
                "schema_version": 1,
                "scope": "staging_only",
                "reviewer": reviewer,
                "plan_sha256": content_hash(plan),
                "comparison_sha256": content_hash(comparison),
                "checkpoint": self._training(plan["selected_seed"], plan)["checkpoint"],
                "evidence_mode": plan["config"]["evidence_mode"],
                "production_quality_established": False,
            }
            write_json(self.root / "approval.json", approval)
            return approval

    def verified_approval(self) -> dict[str, Any]:
        """Reject approval after any relevant data, protocol or evidence change."""
        with workspace_lock(self.root):
            plan = self.verify()
            approval = read_json(self.root / "approval.json")
            comparison = self._comparison(plan, "test")
            if (
                not comparison["passed"]
                or approval.get("scope") != "staging_only"
                or approval.get("production_quality_established") is not False
                or approval.get("plan_sha256") != content_hash(plan)
                or approval.get("comparison_sha256") != content_hash(comparison)
                or approval.get("checkpoint")
                != self._training(plan["selected_seed"], plan)["checkpoint"]
                or approval.get("evidence_mode") != plan["config"]["evidence_mode"]
            ):
                raise ValueError("Staging approval no longer matches verified evidence")
            return dict(approval)

    def status(self) -> dict[str, Any]:
        """Return counts and fixed labels only; no content, paths, IDs or reviewers."""
        with workspace_lock(self.root):
            plan = self.verify()
            manifest = verify_dataset(self.root / "dataset")
            completed = uncertain = 0
            for directory in self.root.iterdir():
                if directory.is_dir() and (directory / "attempt.json").exists():
                    if (directory / "result.json").exists():
                        self._receipt(directory.name, plan)
                        completed += 1
                    else:
                        uncertain += 1
            training = read_json(self.root / "dataset" / "train.json")
            return {
                "schema_version": 1,
                "source_kind": manifest["source_kind"],
                "evidence_mode": plan["config"]["evidence_mode"],
                "evaluation_scope": plan["evaluation_scope"],
                "dataset_counts": {
                    s: len(read_json(self.root / "dataset" / f"{s}.json"))
                    for s in ("train", "validation", "test")
                },
                "training_low_reward_count": sum(
                    type(r.get("reward")) in (int, float) and r["reward"] < 0
                    for r in training
                ),
                "completed_stages": completed,
                "uncertain_stages": uncertain,
                "cost_usd": None,
                "production_quality_established": False,
            }

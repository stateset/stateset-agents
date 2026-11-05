"""Local admission identities preserve unequal data and reject invalid inputs."""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import learner_preflight as preflight  # noqa: E402
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from tests.unit.test_online_accumulation_weighting import _trainer  # noqa: E402
from tests.unit.test_token_group_aggregation import _groups  # noqa: E402


def _worker(tmp_path):
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0.0, 2)
    trainer.distributed_config = SimpleNamespace(world_size=2, strategy="ddp")
    model = trainer.agent.model
    model.config = SimpleNamespace(vocab_size=16)
    trainer.agent.model = SimpleNamespace(
        module=model, named_parameters=model.named_parameters
    )
    return trainer


def test_helper_import_remains_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stateset_agents.training.learner_preflight; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_uneven_groups_and_chunk_preferences_do_not_change_rollout_identity(tmp_path):
    trainer = _worker(tmp_path)
    first, second = _groups()
    a = preflight._rollouts(trainer, [first])
    trainer.config.generation_batch_size = 16
    b = preflight._rollouts(trainer, [second])
    assert a == b == {"path": "token", "source": "recompute"}


@pytest.mark.parametrize(
    "fault",
    [
        "invalid_group",
        "missing",
        "negative",
        "boolean",
        "input_vocabulary",
        "output_vocabulary",
        "likelihood_length",
        "likelihood_positive",
        "likelihood_nan",
        "version",
        "reward_nan",
        "reward_overflow",
    ],
)
def test_bad_rollouts_reject_before_scoring(tmp_path, fault):
    trainer = _worker(tmp_path)
    groups = _groups()
    trajectory = groups[0].trajectories[0]
    metadata = trajectory.turns[0].metadata
    if fault == "invalid_group":
        groups = [None]
    elif fault == "missing":
        metadata.pop("token_ids")
    elif fault == "negative":
        metadata["token_ids"][0] = -1
    elif fault == "boolean":
        metadata["token_ids"][0] = True
    elif fault == "input_vocabulary":
        metadata["prompt_token_ids"][0] = 16
    elif fault == "output_vocabulary":
        metadata["token_ids"][0] = 16
    elif fault == "likelihood_length":
        metadata["sampler_log_probs"] = [-1.0]
    elif fault == "likelihood_positive":
        metadata["sampler_log_probs"] = [0.1] * len(metadata["token_ids"])
    elif fault == "likelihood_nan":
        metadata["sampler_log_probs"] = [float("nan")] * len(metadata["token_ids"])
    elif fault == "version":
        metadata["policy_version"] = -1
    elif fault == "reward_nan":
        trajectory.total_reward = float("nan")
    elif fault == "reward_overflow":
        trajectory.total_reward = 1e50
    with pytest.raises(ValueError):
        preflight._rollouts(trainer, groups)


def test_metadata_after_an_incomplete_turn_still_gets_validated(tmp_path):
    trainer = _worker(tmp_path)
    trainer.config.allow_sequence_fallback = True
    groups = _groups()
    groups[0].trajectories[0].turns[0].metadata = {}
    groups[0].trajectories[1].turns[0].metadata["policy_version"] = -1
    with pytest.raises(ValueError, match="policy_version"):
        preflight._rollouts(trainer, groups)


def test_legacy_sequence_path_requires_explicit_permission(tmp_path):
    trainer = _worker(tmp_path)
    groups = _groups()
    groups[0].trajectories[0].turns[0].metadata = {}
    with pytest.raises(ValueError, match="metadata"):
        preflight._rollouts(trainer, groups)
    trainer.config.allow_sequence_fallback = True
    trainer.config.objective = None
    assert preflight._rollouts(trainer, groups) == {
        "path": "sequence",
        "source": "sequence",
    }


def test_sampler_fallback_is_part_of_effective_path_identity(tmp_path):
    trainer = _worker(tmp_path)
    trainer.config.old_logprobs_source = "sampler"
    groups = _groups()
    with pytest.raises(ValueError, match="likelihoods"):
        preflight._rollouts(trainer, groups)
    trainer.config.allow_missing_sampler_logprobs = True
    assert preflight._rollouts(trainer, groups)["source"] == "recompute_fallback"
    for group in groups:
        for trajectory in group.trajectories:
            for turn in trajectory.turns:
                if turn.role == "assistant":
                    turn.metadata["sampler_log_probs"] = [-2.0] * len(
                        turn.metadata["token_ids"]
                    )
    assert preflight._rollouts(trainer, groups)["source"] == "sampler"


def test_control_identity_preserves_large_exact_counters(tmp_path):
    trainer = _worker(tmp_path)
    trainer.global_step = 2**60 + 1
    encoded = json.dumps(preflight._schedule(trainer))
    assert json.loads(encoded)["global_step"] == 2**60 + 1


def test_optimizer_identity_supports_scalar_tensor_learning_rates(tmp_path):
    trainer = _worker(tmp_path)
    trainer.optimizer.param_groups[0]["lr"] = torch.tensor(0.125)
    identity = preflight._optimization(trainer)
    plain = json.loads(json.dumps(identity, default=preflight._scalar_tensor))
    assert plain["groups"][0]["lr"] == 0.125
    assert plain["groups"][0]["parameters"] == ["lookup.weight"]


def test_pending_local_counts_can_differ_while_shared_scale_agrees(tmp_path):
    trainer = _worker(tmp_path)
    trainer._grad_accum_step = 1
    trainer._grpo_accumulation = {
        "measure": "trajectory",
        "policy_normalizer": 10,
        "entropy_normalizer": 0,
        "policy_count": 2,
        "entropy_count": 0,
    }
    first = preflight._schedule(trainer)
    trainer._grpo_accumulation["policy_count"] = 3
    assert preflight._schedule(trainer) == first
    trainer.config.objective = "dapo"
    with pytest.raises(ValueError, match="measure"):
        preflight._schedule(trainer)


def test_non_ddp_admission_does_not_touch_legacy_trainers(tmp_path):
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0, 1)
    original = copy.deepcopy(trainer.agent.model.state_dict())
    assert preflight.verify_ddp_learner_batch(trainer, []) is None
    for name, value in original.items():
        torch.testing.assert_close(
            trainer.agent.model.state_dict()[name], value, rtol=0, atol=0
        )


def test_single_worker_ddp_preserves_local_execution(monkeypatch):
    class LocalDDP:
        process_group = object()

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", LocalDDP)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 1)

    def unexpected_collective(*_args, **_kwargs):
        pytest.fail("single-worker DDP does not need a learner agreement collective")

    monkeypatch.setattr(torch.distributed, "all_gather", unexpected_collective)
    trainer = SimpleNamespace(agent=SimpleNamespace(model=LocalDDP()))
    assert preflight.verify_ddp_learner_batch(trainer, []) is None


def test_loss_admission_binds_dropout_and_the_selected_kl_estimator(tmp_path):
    trainer = _worker(tmp_path)
    trainer.config.objective = "gspo"
    trainer.config.beta = 0.25
    trainer.reference_model = object()
    initial = preflight._loss(trainer)
    assert initial["objective"]["kl"] == "k3_sequence"
    assert initial["disable_dropout"] is False
    trainer.config.disable_dropout = True
    changed = preflight._loss(trainer)
    assert changed != initial and changed["disable_dropout"] is True
    trainer.config.objective_overrides = {"kl": "k3_token"}
    assert preflight._loss(trainer)["objective"]["kl"] == "k3_token"


@pytest.mark.parametrize("setting", [0, 1, None, "false"])
def test_loss_admission_rejects_ambiguous_dropout_settings(tmp_path, setting):
    trainer = _worker(tmp_path)
    trainer.config.disable_dropout = setting
    with pytest.raises(ValueError, match="disable_dropout"):
        preflight._loss(trainer)

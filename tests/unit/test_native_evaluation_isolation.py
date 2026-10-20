"""Held-out evaluation must leave training randomness and module modes intact."""

from __future__ import annotations

import asyncio
import concurrent.futures
import random
from types import SimpleNamespace

import numpy as np
import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    content_hash,
    validate_report,
)
from stateset_agents.evaluation.isolation import (
    EVALUATION_RNG_PROTOCOL,
    isolated_evaluation,
    paired_evaluation_seed,
)
from stateset_agents.training.native_refund_study import (
    evaluate,
    load_cases,
    validate_study_config,
)
from stateset_agents.training.runtime_state import capture_rng, restore_rng

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def preserve_test_randomness():
    state = capture_rng(torch)
    yield
    restore_rng(state, torch)


def draws():
    return [
        random.randrange(2**30),
        int(np.random.randint(2**30)),
        int(torch.randint(2**30, (1,)).item()),
    ]


def make_agent(policy, model=None):
    return SimpleNamespace(
        config=SimpleNamespace(model_name="local-test-model"),
        model=model,
        generate_turn=policy,
    )


def finish(tokens=None):
    return ConversationTurn(
        role="assistant",
        content='{"tool":"finish","args":{}}',
        metadata={"token_ids": tokens or []},
    )


async def report(agent, cases=None, seed=42, split="test"):
    return await evaluate(
        agent,
        cases or refund_policy_benchmark("test", 8, 0)[:3],
        seed=seed,
        settings={"temperature": 1.0},
        checkpoint="local-checkpoint",
        split=split,
    )


def outcome_tokens(result):
    return {
        row["case_id"]: [
            turn["token_ids"] for turn in row["trace"] if turn["role"] == "assistant"
        ]
        for row in result["outcomes"]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("grad_enabled", [False, True])
async def test_evaluation_does_not_advance_training_rngs(grad_enabled):
    async def policy(messages, context):
        return finish(draws())

    state = capture_rng(torch)
    expected = draws()
    restore_rng(state, torch)
    with torch.set_grad_enabled(grad_enabled):
        await report(make_agent(policy))
        assert torch.is_grad_enabled() is grad_enabled
    assert draws() == expected


@pytest.mark.asyncio
async def test_case_sampling_is_independent_of_case_order():
    async def policy(messages, context):
        return finish(draws())

    cases = refund_policy_benchmark("test", 8, 0)[:3]
    before = await report(make_agent(policy), cases)
    after = await report(make_agent(policy), list(reversed(cases)))
    assert outcome_tokens(before) == outcome_tokens(after)


@pytest.mark.asyncio
async def test_preceding_episode_consumption_does_not_shift_later_cases():
    cases = refund_policy_benchmark("test", 8, 0)[:3]

    def policy(extra):
        async def generate(messages, context):
            if cases[0]["order_id"] in messages[-1]["content"]:
                for _ in range(extra):
                    draws()
            return finish(draws())

        return generate

    before = outcome_tokens(await report(make_agent(policy(0)), cases))
    after = outcome_tokens(await report(make_agent(policy(7)), cases))
    for case in cases[1:]:
        assert before[case["order_id"]] == after[case["order_id"]]


@pytest.mark.asyncio
async def test_evaluation_repeats_all_rng_streams_despite_caller_activity():
    async def policy(messages, context):
        return finish(draws())

    agent = make_agent(policy)
    before = await report(agent)
    for _ in range(13):
        draws()
    after = await report(agent)
    assert outcome_tokens(before) == outcome_tokens(after)


@pytest.mark.asyncio
@pytest.mark.parametrize("root_training", [False, True])
@pytest.mark.parametrize(
    "failure", [None, RuntimeError, asyncio.CancelledError, KeyboardInterrupt]
)
async def test_modes_and_rngs_restore_on_success_error_and_cancellation(
    root_training, failure
):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.train(root_training)
    model[1].training = not root_training
    modes = [module.training for module in model.modules()]

    async def policy(messages, context):
        draws()
        model.train(not root_training)
        if failure is not None:
            raise failure("evaluation stopped")
        return finish()

    state = capture_rng(torch)
    expected = draws()
    restore_rng(state, torch)
    agent = make_agent(policy, model)
    if failure is None:
        await report(agent)
    else:
        with pytest.raises(failure, match="evaluation stopped"):
            await report(agent)
    assert [module.training for module in model.modules()] == modes
    assert draws() == expected


@pytest.mark.asyncio
async def test_evaluation_disables_gradients_and_uses_eval_mode():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.9))
    model.train()

    async def policy(messages, context):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert not model(torch.ones(1, 2)).requires_grad
        return finish()

    await report(make_agent(policy, model))
    assert model.training and torch.is_grad_enabled()


@pytest.mark.asyncio
async def test_inserted_evaluation_preserves_next_dropout_optimizer_update():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def update():
        optimizer.zero_grad()
        scale = random.random() + float(np.random.rand())
        model(inputs).square().mean().mul(scale).backward()
        optimizer.step()
        return {name: p.detach().clone() for name, p in model.named_parameters()}

    state = capture_rng(torch)
    expected = update()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(original[name])
    restore_rng(state, torch)

    async def policy(messages, context):
        draws()
        model(inputs)
        return finish()

    await report(make_agent(policy, model))
    actual = update()
    assert all(torch.equal(expected[name], actual[name]) for name in actual)


def test_nested_context_restores_parent_stream_grad_mode_and_cache():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.config = SimpleNamespace(use_cache=False)
    model.train()
    model[1].eval()
    modes = [m.training for m in model.modules()]
    with isolated_evaluation(seed=0, model=model):
        assert model.config.use_cache and not torch.is_grad_enabled()
        outer_state = capture_rng(torch)
        expected = draws()
        restore_rng(outer_state, torch)
        with isolated_evaluation(seed=2**64 - 1, model=model):
            draws()
            assert model.config.use_cache and not torch.is_grad_enabled()
        assert draws() == expected
        assert model.config.use_cache and not model.training
    assert not model.config.use_cache and torch.is_grad_enabled()
    assert [m.training for m in model.modules()] == modes


@pytest.mark.parametrize("seed", [-1, 2**64, True, 1.5, "42", None])
def test_invalid_seed_rejected_without_changing_training_state(seed):
    state = capture_rng(torch)
    expected = draws()
    restore_rng(state, torch)
    with pytest.raises(ValueError, match="seed"):
        with isolated_evaluation(seed=seed):
            pytest.fail("Invalid evaluation seed was accepted")
    assert draws() == expected


@pytest.mark.parametrize(
    "case_id,case_hash", [("", "a" * 64), ("id", "a" * 63), ("id", "Z" * 64)]
)
def test_pairing_rejects_invalid_case_identity(case_id, case_hash):
    with pytest.raises(ValueError, match="case"):
        paired_evaluation_seed(42, case_id, case_hash)


def test_seed_protocol_matches_independent_canonical_hash():
    import hashlib
    import json

    payload = {
        "protocol": "sha256-case-v1",
        "seed": 42,
        "case_id": "order-1",
        "case_hash": "a" * 64,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actual = paired_evaluation_seed(42, "order-1", "a" * 64)
    assert actual == int(digest[:16], 16)
    assert actual != paired_evaluation_seed(43, "order-1", "a" * 64)
    assert actual != paired_evaluation_seed(42, "order-2", "a" * 64)
    assert actual != paired_evaluation_seed(42, "order-1", "b" * 64)


@pytest.mark.asyncio
async def test_report_records_exact_case_hashes_and_pairing_protocol():
    async def policy(messages, context):
        return finish(draws())

    cases = refund_policy_benchmark("test", 8, 0)[:3]
    result = await report(make_agent(policy), cases)
    assert (
        result["settings"]["evaluation_randomness"]["protocol"]
        == EVALUATION_RNG_PROTOCOL
    )
    by_id = {c["order_id"]: c for c in cases}
    for row in result["outcomes"]:
        digest = content_hash(by_id[row["case_id"]])
        assert row["case_hash"] == digest
        assert row["evaluation_seed"] == paired_evaluation_seed(
            42, row["case_id"], digest
        )


@pytest.mark.asyncio
async def test_all_cases_validated_before_first_generation():
    seen = []

    async def policy(messages, context):
        seen.append(messages)
        return finish()

    cases = refund_policy_benchmark("test", 8, 0)[:3]
    cases[-1]["paid_cents"] = -1
    with pytest.raises(ValueError):
        await report(make_agent(policy), cases)
    assert seen == []


@pytest.mark.asyncio
async def test_duplicate_case_ids_rejected_before_first_generation():
    async def policy(messages, context):
        pytest.fail("Duplicate cases reached generation")

    cases = refund_policy_benchmark("test", 8, 0)[:3]
    cases[-1]["order_id"] = cases[0]["order_id"]
    with pytest.raises(ValueError, match="unique"):
        await report(make_agent(policy), cases)


@pytest.mark.asyncio
async def test_overlapping_evaluation_tasks_fail_without_advancing_owner_rng():
    async def overlap():
        with pytest.raises(RuntimeError, match="serially"):
            with isolated_evaluation(seed=99):
                pytest.fail("Overlapping evaluation entered")

    with isolated_evaluation(seed=42):
        state = capture_rng(torch)
        expected = draws()
        restore_rng(state, torch)
        await asyncio.create_task(overlap())
        assert draws() == expected
    # The rejected caller must not leave ownership stuck.
    with isolated_evaluation(seed=43):
        draws()


def test_overlapping_evaluation_threads_fail_without_advancing_owner_rng():
    def overlap():
        with pytest.raises(RuntimeError, match="serially"):
            with isolated_evaluation(seed=99):
                pytest.fail("Overlapping evaluation entered")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        with isolated_evaluation(seed=42):
            state = capture_rng(torch)
            expected = draws()
            restore_rng(state, torch)
            pool.submit(overlap).result(timeout=5)
            assert draws() == expected


def test_cuda_generator_states_are_restored_through_shared_rng_api(monkeypatch):
    # Simulated device generators test the contract without claiming a GPU run.
    device_state = [
        torch.tensor([17], dtype=torch.uint8),
        torch.tensor([29], dtype=torch.uint8),
    ]

    def seed_all(seed):
        device_state[:] = [torch.tensor([seed % 256], dtype=torch.uint8)] * 2

    def restore_all(values):
        assert all(v.device.type == "cpu" for v in values)
        device_state[:] = [v.clone() for v in values]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: [v.clone() for v in device_state]
    )
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore_all)
    with isolated_evaluation(seed=42):
        assert [v.item() for v in device_state] == [42, 42]
    assert [v.item() for v in device_state] == [17, 29]


def test_failed_model_entry_restores_modes_rngs_and_releases_owner():
    class FailingModel(torch.nn.Linear):
        def eval(self):
            self.training = False
            draws()
            raise RuntimeError("cannot enter evaluation")

    model = FailingModel(2, 2)
    state = capture_rng(torch)
    expected = draws()
    restore_rng(state, torch)
    with pytest.raises(RuntimeError, match="cannot enter"):
        with isolated_evaluation(seed=42, model=model):
            pytest.fail("Failed model entered evaluation")
    assert model.training and torch.is_grad_enabled()
    assert draws() == expected
    with isolated_evaluation(seed=43):
        draws()


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_revision": "main"},
        {"model_revision": "0" * 40},
        {"model": " "},
        {"seeds": []},
        {"seeds": [42] * 6},
        {"seeds": [True, 43, 44, 45, 46, 47]},
        {"seeds": [-1, 43, 44, 45, 46, 47]},
        {"seeds": [2**32, 43, 44, 45, 46, 47]},
        {"seeds": [42, 43, 44]},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": -1.0},
        {"learning_rate": True},
        {"learning_rate": 10**1000},
        {"updates": 0},
        {"updates": True},
        {"eval_every": 0},
        {"generations": 1},
        {"synthetic": True, "synthetic_case_count": 7},
    ],
)
def test_invalid_study_protocol_rejected_before_model_allocation(overrides):
    config = {
        "model": "reviewed-model",
        "model_revision": "a" * 40,
        "seeds": [42, 43, 44, 45, 46, 47],
        "updates": 20,
        "generations": 8,
        "eval_every": 10,
        "learning_rate": 1e-5,
        "synthetic": False,
        "synthetic_case_count": 32,
        **overrides,
    }
    with pytest.raises(ValueError):
        validate_study_config(SimpleNamespace(**config))


@pytest.mark.parametrize("synthetic", [False, True])
def test_valid_study_protocol_accepts_reviewed_and_explicit_synthetic_runs(synthetic):
    validate_study_config(
        SimpleNamespace(
            model="reviewed-model",
            model_revision="a" * 40,
            seeds=[42] if synthetic else [42, 43, 44, 45, 46, 47],
            updates=20,
            generations=8,
            eval_every=10,
            learning_rate=1e-5,
            synthetic=synthetic,
            synthetic_case_count=32,
        )
    )


@pytest.mark.parametrize("annotation", ["expected_response", "metadata", "reviewer"])
def test_reviewed_split_rejects_renamed_duplicates_with_different_annotations(
    tmp_path, annotation
):
    import json

    data = {
        "provenance": {"source": "reviewed", "reviewer": "fixture-reviewer"},
        **{
            split: refund_policy_benchmark(split, 32, 0)
            for split in ("train", "validation", "test")
        },
    }
    data["test"][0] = dict(
        data["train"][0], order_id="renamed-case", **{annotation: "different"}
    )
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint"):
        load_cases(path)


@pytest.mark.parametrize("data", [[], None, {"provenance": []}])
def test_reviewed_cases_reject_malformed_document_or_provenance(tmp_path, data):
    import json

    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_cases(path)


@pytest.mark.asyncio
async def test_local_transformer_sampling_restores_mixed_modes_and_pending_gradients():
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_embd=8,
            n_layer=1,
            n_head=1,
            vocab_size=16,
            n_positions=16,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
    )
    model.train()
    model.transformer.h[0].attn.eval()
    model.config.use_cache = False
    modes = [m.training for m in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}

    async def policy(messages, context):
        inputs = torch.tensor([[1, 4, 5]])
        assert not torch.is_grad_enabled() and not model.training
        assert model.config.use_cache
        tokens = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=2,
            do_sample=True,
            top_k=0,
            pad_token_id=0,
        )
        return finish(tokens[0, 3:].tolist())

    cases = refund_policy_benchmark("test", 8, 0)[:3]
    agent = make_agent(policy, model)
    before = outcome_tokens(await report(agent, cases))
    after = outcome_tokens(await report(agent, list(reversed(cases))))
    assert before == after
    assert [m.training for m in model.modules()] == modes
    assert not model.config.use_cache
    assert all(
        torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters()
    )


@pytest.mark.asyncio
async def test_actual_task_cancellation_restores_training_state():
    entered = asyncio.Event()
    model = torch.nn.Linear(2, 2)
    model.train()

    async def policy(messages, context):
        draws()
        entered.set()
        await asyncio.Event().wait()

    state = capture_rng(torch)
    expected = draws()
    restore_rng(state, torch)
    task = asyncio.create_task(report(make_agent(policy, model)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.training and torch.is_grad_enabled()
    assert draws() == expected
    with isolated_evaluation(seed=43):
        draws()


@pytest.mark.asyncio
@pytest.mark.parametrize("split", ["validation", "test"])
async def test_evaluation_reports_explicit_split_and_comparisons_reject_validation(
    split,
):
    async def policy(messages, context):
        return finish()

    result = await report(make_agent(policy), split=split)
    assert result["split"] == split
    assert validate_report(result, expected_split=split)["total"] == 3
    if split == "validation":
        with pytest.raises(ValueError, match="test"):
            compare_runs([result], [result])


@pytest.mark.asyncio
async def test_native_study_retains_validation_selection_and_seals_candidate_test(
    tmp_path, monkeypatch
):
    import json

    import stateset_agents.core.agent as agent_module
    import stateset_agents.training.multi_turn_trainer as trainer_module
    import stateset_agents.training.native_refund_study as study
    from stateset_agents.evaluation.agent_runs import evaluation_report

    class Agent:
        def __init__(self, config):
            self.config, self.version = config, 0

        async def initialize(self):
            pass

    class Trainer:
        def __init__(self, agent, environment, config):
            self.agent, self.config, self.global_step = agent, config, 0

        async def initialize(self):
            pass

        async def save_checkpoint(self, checkpoint_name):
            path = tmp_path / "seed-42" / checkpoint_name
            path.mkdir(exist_ok=True)
            (path / ".stateset-checkpoint.json").write_text(
                json.dumps({"version": self.agent.version})
            )

        def load_checkpoint(self, path):
            self.agent.version = json.loads(
                (path / ".stateset-checkpoint.json").read_text()
            )["version"]
            return True

        async def generate_trajectories(self, scenarios):
            return scenarios

        async def training_step(self, groups):
            self.global_step += 1
            self.agent.version += 1
            return {"global_step": self.global_step, "optimizer_updates": 1}

    calls = []

    async def controlled_evaluation(
        agent, cases, *, seed, settings, checkpoint, implementation, split="test"
    ):
        calls.append((split, agent.version))
        passed = [1, 3, 3, 2][agent.version]
        return evaluation_report(
            environment="selection-fixture",
            base_model=agent.config.model_name,
            seed=seed,
            cases={c["order_id"]: c for c in cases},
            settings=settings,
            checkpoint={"path": checkpoint},
            split=split,
            outcomes=[
                {
                    "case_id": c["order_id"],
                    "success": i < passed,
                    "reward": float(i < passed),
                    "policy_violations": 0,
                    "generated_tokens": 1,
                    "tool_calls": 1,
                    "elapsed_seconds": 0.0,
                }
                for i, c in enumerate(cases)
            ],
        )

    monkeypatch.setattr(agent_module, "MultiTurnAgent", Agent)
    monkeypatch.setattr(trainer_module, "MultiTurnGRPOTrainer", Trainer)
    monkeypatch.setattr(study, "evaluate", controlled_evaluation)
    args = SimpleNamespace(
        output=tmp_path,
        model="local-fixture",
        model_revision="a" * 40,
        updates=3,
        generations=2,
        learning_rate=1e-5,
        eval_every=1,
        implementation={"fixture": True},
    )
    cases = {
        split: refund_policy_benchmark(split, 8, 0)[:4]
        for split in ("train", "validation", "test")
    }
    _, candidate = await study.run_seed(args, cases, 42)
    root = tmp_path / "seed-42"
    selection = json.loads((root / "selection.json").read_text())
    assert selection["selected_step"] == 1  # Earliest validation tie wins.
    assert selection["selected_success_rate"] == 0.75
    assert selection["committed_optimizer_updates"] == 3
    assert calls == [
        ("test", 0),
        ("validation", 0),
        ("validation", 1),
        ("validation", 2),
        ("validation", 3),
        ("test", 1),
    ]
    for row in selection["validation_reports"]:
        raw = json.loads((root / row["report"]).read_text())
        assert raw["split"] == "validation" and content_hash(raw) == row["sha256"]
    assert candidate["selected_checkpoint"]["selection_sha256"] == content_hash(
        selection
    )

    async def no_progress(self, groups):
        return {"global_step": 0, "optimizer_updates": 0}

    monkeypatch.setattr(Trainer, "training_step", no_progress)
    args.output = tmp_path / "zero-progress"
    calls.clear()
    with pytest.raises(RuntimeError, match="candidate test remains sealed"):
        await study.run_seed(args, cases, 42)
    assert [(split, version) for split, version in calls if split == "test"] == [
        ("test", 0)
    ]
    assert not (args.output / "seed-42" / "candidate.json").exists()

"""Real River scheduling/recovery with scripted transport; no paid model calls.

Run independently of the heavyweight pytest fixtures:
    python -m unittest discover -s tests/integration -p test_river_native_sdk.py -v

The sampler is deliberately synthetic. These tests verify SDK integration, not
remote execution, learned weights, billing, or model quality.
"""

import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if sys.version_info < (3, 12):
    raise unittest.SkipTest("River SDK requires Python >=3.12")

try:
    import river_client
except ModuleNotFoundError as exc:
    if exc.name == "river_client":
        raise unittest.SkipTest(
            "Install stateset-agents[river] for native SDK checks"
        ) from exc
    raise  # An installed but broken SDK must fail, not silently skip.

from river_client import Checkpoint, Sample, rl
from river_client.renderers.base import ParsedResponse, SamplePrompt
from river_client.rl.advantages import build_batch

from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.checkpoint_selection import (
    require_training_progress,
)
from stateset_agents.evaluation.refund_trace import (
    audit_refund_traces,
    replay_refund_trace,
    select_replayed_validation_checkpoint,
)
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_case_identity,
    trajectory_environment_trace,
    trajectory_truncation,
)
from stateset_agents.remote.river_runtime import (
    inspect_native_runtime,
    require_native_runtime,
)
from stateset_agents.remote.rollout_budget import (
    RolloutAdmissionBudget,
    RolloutBudgetExceeded,
)
from stateset_agents.training.river_progress import (
    RiverTrainingProgress,
    summarize_training_activity,
)
from stateset_agents.training.river_refund import (
    EVALUATION_SETTINGS,
    TRUNCATION_POLICY,
    campaign,
    prepare_run,
)

STOP = "<|end|>"
ROWS = [
    {"order_id": "A", "amount_cents": 1250, "eligible": True},
    {"order_id": "B", "amount_cents": 2300, "eligible": False},
]


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")


class ScriptedRenderer:
    tokenizer = CharacterTokenizer()

    def get_stop_strings(self):
        return [STOP]

    def build_sample_prompt(self, messages, **kwargs):
        return SamplePrompt(json.dumps(messages))

    def build_continuation_prompt(self, messages, **kwargs):
        return SamplePrompt("\n" + json.dumps(messages))

    def parse_response(self, text, **kwargs):
        return ParsedResponse(
            {"role": "assistant", "content": text.removesuffix(STOP)},
            stop_found=text.endswith(STOP),
        )


class ScriptedModel:
    """Only the provider transport is replaced; scheduling/training is River's."""

    base_model = "scripted-contract-model"

    def __init__(self, *, mixed=False, exact=True, finish=True):
        self.step = 0
        self.mixed = mixed
        self.exact = exact
        self.finish = finish
        self.sampled = []
        self.backward_batches = []
        self.optimizer_calls = 0
        self.loaded_optimizer = []

    async def sample(self, *, model_input, seeds, max_tokens, **kwargs):
        outputs = []
        for chunks, seed in zip(model_input, seeds, strict=True):
            prompt = ScriptedRenderer.tokenizer.decode(
                [t for c in chunks for t in c["tokens"]]
            )
            order = re.search(r"request for order (\w+)\.", prompt).group(1)
            row = next(row for row in ROWS if row["order_id"] == order)
            turn = prompt.count(STOP)
            member_seed = (seed - turn * 104729) % (2**31)
            if turn == 0:
                action = {"tool": "lookup_order", "args": {"order_id": order}}
            elif turn == 1:
                action = {
                    "tool": "refund" if row["eligible"] else "deny",
                    "args": {"order_id": order},
                }
                if row["eligible"]:
                    action["args"]["amount_cents"] = row["amount_cents"] + int(
                        self.mixed and member_seed % 2
                    )
            elif self.finish:
                action = {"tool": "finish", "args": {}}
            else:
                action = {"tool": "lookup_order", "args": {"order_id": order}}
            tokens = ScriptedRenderer.tokenizer.encode(json.dumps(action) + STOP)
            complete = len(tokens) <= max_tokens
            tokens = tokens[:max_tokens]
            self.sampled.append((order, seed, list(tokens)))
            outputs.append(
                [
                    Sample(
                        tokens=tokens,
                        text=ScriptedRenderer.tokenizer.decode(tokens),
                        logprobs=[-0.1] * len(tokens),
                        stop_reason="stop" if complete else "length",
                        model_step=(
                            kwargs["checkpoint"].step
                            if "checkpoint" in kwargs
                            else self.step
                        ),
                        token_data_is_exact=self.exact,
                    )
                ]
            )
        return outputs

    def forward_backward(self, data, **kwargs):
        self.backward_batches.append(data)
        return SimpleNamespace(metrics={"loss": 0.25})

    def optim_step(self, **kwargs):
        self.step += 1
        self.optimizer_calls += 1
        return SimpleNamespace(metrics={"learning_rate": kwargs["lr"]})

    def save_weights(self, name, *, mode, **kwargs):
        return Checkpoint(
            path=f"river://scripted-{name}", step=self.step, checkpoint_type=mode
        )

    def load_weights(self, checkpoint, *, load_optimizer=False):
        self.loaded_optimizer.append(load_optimizer)
        self.step = checkpoint.step


class ScriptedSession:
    """Implement the session's pending-sample contract without a provider."""

    def __init__(self, model):
        self.model = model

    def submit_sample(self, **kwargs):
        async def result_async(**unused):
            return await self.model.sample(**kwargs)

        return SimpleNamespace(result_async=result_async)


class RiverClientLifecycleIntegration(unittest.TestCase):
    """Exercise real SDK client cleanup with all transport entry points blocked."""

    def setUp(self):
        for name in ("grpc.secure_channel", "grpc.insecure_channel"):
            guard = patch(name, side_effect=AssertionError("No RPCs allowed"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_owned_client_releases_sdk_image_cache_on_access_failure(self):
        real_client = river_client.Client
        for response in ([], ConnectionError("scripted offline failure")):
            with (
                self.subTest(response=type(response).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                clients = []

                def construct(_clients=clients, **kwargs):
                    client = real_client(**kwargs)
                    _clients.append(client)
                    self.addCleanup(client.close)
                    self.assertTrue(Path(client._image_cache.name).exists())
                    return client

                with (
                    patch.dict(os.environ, {"RIVER_API_KEY": "rv_test"}),
                    patch.object(river_client, "Client", side_effect=construct),
                    patch.object(
                        real_client,
                        "_get_channel",
                        side_effect=AssertionError("No RPCs allowed"),
                    ),
                    patch.object(
                        real_client,
                        "get_capabilities",
                        **(
                            {"side_effect": response}
                            if isinstance(response, Exception)
                            else {"return_value": response}
                        ),
                    ),
                ):
                    executor = RiverExecutor()
                    dataset = Path(directory) / "unused.jsonl"
                    dataset.touch()
                    spec = RemoteJobSpec(
                        base_model="private/model",
                        dataset=dataset,
                        output_dir=Path(directory) / "out",
                    )
                    with self.assertRaises(RemoteExecutionError):
                        executor.submit(spec)
                    self.assertEqual(len(clients), 1)
                    self.assertIsNone(executor._client)
                    self.assertFalse(Path(clients[0]._image_cache.name).exists())
                    self.assertEqual(executor._jobs["river-1"].status, JobStatus.FAILED)

    def test_injected_sdk_client_remains_open_until_caller_closes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            client = river_client.Client(api_key="rv_test")
            self.addCleanup(client.close)
            cache = Path(client._image_cache.name)
            dataset = Path(directory) / "unused.jsonl"
            dataset.touch()
            with patch.object(client, "get_capabilities", return_value=[]):
                executor = RiverExecutor(client=client)
                with self.assertRaises(RemoteExecutionError):
                    executor.submit(
                        RemoteJobSpec(
                            base_model="private/model",
                            dataset=dataset,
                            output_dir=Path(directory) / "out",
                        )
                    )
            self.assertTrue(cache.exists())
            client.close()
            self.assertFalse(cache.exists())


class NativeRiverIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stateset-native-river-")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.instances = []
        # An accidental client construction would turn this into a paid test.
        guard = patch.object(
            river_client,
            "Client",
            side_effect=AssertionError("No provider clients allowed"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    def test_runtime_inspection_accepts_sdk_without_loading_tokenizers(self):
        with (
            patch.dict(os.environ, {"RIVER_API_KEY": ""}),
            patch(
                "river_client.renderers.get_renderer",
                side_effect=AssertionError("No tokenizer loads allowed"),
            ),
        ):
            checks = inspect_native_runtime()
            self.assertTrue(checks["python"]["passed"])
            self.assertTrue(checks["river_sdk"]["passed"], checks)
            self.assertFalse(checks["credentials"]["configured"])
            with self.assertRaisesRegex(RuntimeError, "RIVER_API_KEY"):
                require_native_runtime()

    def environment(
        self,
        *,
        before_reset=None,
        broken=False,
        pause_after_refund=None,
        truncation_reward=0.0,
    ):
        instances = self.instances

        class TrackedRefund(RefundEnvironment):
            def __init__(self):
                super().__init__()
                self.closed = False
                instances.append(self)

            async def step(self, state, action):
                if broken:
                    raise ConnectionError("tool backend unavailable")
                result = await super().step(state, action)
                self.last_state = result[0]
                if pause_after_refund is not None and result[0].context["refunds"]:
                    pause_after_refund.set()
                    await asyncio.Event().wait()
                return result

            async def close(self):
                self.closed = True

        return river_environment_factory(
            TrackedRefund,
            before_reset=before_reset,
            truncation_reward=truncation_reward,
            record_trace=True,
        )

    def engine(self, model, *, budget=None, **environment_kwargs):
        return rl.RolloutEngine(
            model,
            env=self.environment(**environment_kwargs),
            renderer=ScriptedRenderer(),
            budget=budget
            or rl.Budget(
                max_turns=4,
                max_generated_tokens=1024,
                max_context_tokens=8192,
                max_turn_tokens=256,
            ),
            schedule=rl.Schedule(concurrency=4),
            seed=42,
        )

    async def collect(self, engine, rows=ROWS, group_size=2):
        return [
            trajectory
            async for group in engine.rollout(
                rows,
                group_size=group_size,
                completion=rl.GroupCompletion(mode="wait", min_members=group_size),
            )
            for trajectory in group
        ]

    async def test_real_evaluator_retains_complete_validation_case_evidence(self):
        model = ScriptedModel()
        cases = {row["order_id"]: row for row in ROWS}
        evidence = []

        def engine_factory(checkpoint, variant):
            return self.engine(
                rl.CheckpointSampler(
                    ScriptedSession(model),
                    base_model=model.base_model,
                    checkpoint=checkpoint,
                    tokenizer=CharacterTokenizer(),
                )
            )

        def sink(result):
            evidence.append(
                {
                    "step": result.step,
                    "checkpoint": asdict(result.checkpoint),
                    "metrics": result.metrics,
                    "case_hashes": {
                        key: content_hash(row) for key, row in cases.items()
                    },
                    "outcomes": [
                        {
                            **trajectory_case_identity(t, cases=cases),
                            "reward": t.reward,
                            "environment_trace": trajectory_environment_trace(t),
                        }
                        for t in result.trajectories
                    ],
                }
            )

        evaluator = rl.Evaluator(
            ROWS,
            engine_factory=engine_factory,
            every=1,
            group_size=1,
            final_group_size=1,
            sink=sink,
        )
        try:
            await evaluator.launch(model, 0)
            await evaluator.launch(model, 1, final=True)
            await evaluator.wait()
        finally:
            await evaluator.close()
        best = await select_replayed_validation_checkpoint(
            evidence, steps=1, cases=cases, environment="refund-v1"
        )
        self.assertEqual(best["step"], 0)
        self.assertEqual(best["metrics"]["reward_mean"], 1)
        self.assertEqual(len(best["outcomes"]), len(ROWS))
        removed = evidence[1]["outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "case coverage"):
            await select_replayed_validation_checkpoint(
                evidence, steps=1, cases=cases, environment="refund-v1"
            )
        evidence[1]["outcomes"].append(removed)
        evidence[1]["outcomes"][0]["environment_trace"]["steps"][0]["observations"] = []
        with self.assertRaisesRegex(ValueError, "Validation replay failed"):
            await select_replayed_validation_checkpoint(
                evidence, steps=1, cases=cases, environment="refund-v1"
            )

    async def test_real_engine_executes_private_ledgers_and_preserves_tokens(self):
        model = ScriptedModel()
        trajectories = await self.collect(self.engine(model))
        self.assertEqual(len(trajectories), 4)
        self.assertTrue(
            all(t.reward == 1 and t.done and t.truncated is None for t in trajectories)
        )
        self.assertEqual(
            sorted(t.stateset_episode_id for t in trajectories), ["A", "A", "B", "B"]
        )
        self.assertEqual(len(self.instances), 4)
        self.assertTrue(all(env.closed for env in self.instances))
        self.assertEqual(
            sum(t.generated_tokens for t in trajectories),
            sum(len(tokens) for _, _, tokens in model.sampled),
        )
        self.assertEqual(
            Counter(tuple(tokens) for _, _, tokens in model.sampled),
            Counter(
                tuple(token for chunk in span.chunks for token in chunk["tokens"])
                for trajectory in trajectories
                for span in trajectory.spans
                if span.kind == "generated"
            ),
        )
        for trajectory in trajectories:
            trace = trajectory_environment_trace(trajectory)
            replay = await replay_refund_trace(
                "refund-v1",
                next(
                    row
                    for row in ROWS
                    if row["order_id"] == trajectory.stateset_episode_id
                ),
                trace,
            )
            self.assertTrue(replay["success"])
            self.assertEqual(len(trace["steps"]), 3)
            self.assertEqual(trace["terminal"]["reward"], 1)
            self.assertTrue(trace["steps"][-1]["done"])
            self.assertEqual(trace["steps"][0]["metrics"]["tool_calls"], 1)
            for span in trajectory.spans:
                if span.kind == "generated":
                    self.assertTrue(all(value == -0.1 for value in span.logprobs))

    async def test_native_campaign_trains_selects_replays_and_seals_test_publication(
        self,
    ):
        rows = [*ROWS, {"order_id": "C", "amount_cents": 750, "eligible": True}]
        splits = {"train": rows[:1], "validation": rows[1:2], "test": rows[2:]}
        original_engine = rl.RolloutEngine

        class AlteredTestMetrics(original_engine):
            async def rollout(self, cases, **kwargs):
                async for group in super().rollout(cases, **kwargs):
                    if cases == splits["test"]:
                        group[0].metrics["tool_calls"] += 1
                    yield group

        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt):
                output = self.output / ("rejected" if corrupt else "accepted")
                output.mkdir()
                args = SimpleNamespace(
                    output=output,
                    base_model=ScriptedModel.base_model,
                    seed=42,
                    steps=1,
                    concurrency=4,
                    max_staleness=0,
                    learning_rate=1e-5,
                    checkpoint=None,
                    evaluate_only=False,
                    dry_run=False,
                    benchmark="refund-v1",
                )
                prepare_run(args, splits)
                training_model = ScriptedModel(mixed=True)
                evaluation_model = ScriptedModel()
                session = ScriptedSession(evaluation_model)
                with (
                    patch.object(sys.modules[__name__], "ROWS", rows),
                    patch.object(
                        rl,
                        "RolloutEngine",
                        AlteredTestMetrics if corrupt else original_engine,
                    ),
                ):
                    task = campaign(
                        training_model, session, ScriptedRenderer(), args, splits
                    )
                    if corrupt:
                        with self.assertRaisesRegex(
                            ValueError, "Test trace replay failed"
                        ):
                            await asyncio.wait_for(task, 15)
                    else:
                        await asyncio.wait_for(task, 15)
                    self.assertEqual(training_model.optimizer_calls, 1)
                    activity = json.loads(
                        (output / "training_activity.json").read_text()
                    )
                    self.assertEqual(activity["observed_optimizer_updates"], 1)
                    self.assertEqual(activity["observed_skipped_batches"], 0)
                    self.assertEqual(activity["unknown_update_batches"], 0)
                    self.assertTrue((output / "test_attempt.json").exists())
                    validation = json.loads(
                        (output / "validation_results.json").read_text()
                    )
                    self.assertEqual({entry["step"] for entry in validation}, {0, 1})
                    self.assertEqual(
                        sum(order == "C" for order, _, _ in evaluation_model.sampled), 3
                    )
                    if corrupt:
                        self.assertFalse((output / "test_results.json").exists())
                        failure = json.loads(
                            (output / "test_replay_failure.json").read_text()
                        )
                        self.assertFalse(failure["replay_audit"]["passed"])
                        self.assertEqual(failure["status"], "rejected")
                    else:
                        report = json.loads((output / "test_results.json").read_text())
                        self.assertEqual(report["selected_validation_step"], 0)
                        self.assertEqual(report["passed"], 1)
                        self.assertEqual(report["training_activity"], activity)
                        self.assertTrue(
                            (await audit_refund_traces(report, splits["test"]))[
                                "passed"
                            ]
                        )
                    samples = len(evaluation_model.sampled)
                    with self.assertRaisesRegex(ValueError, "sealed"):
                        await campaign(
                            training_model, session, ScriptedRenderer(), args, splits
                        )
                    self.assertEqual(len(evaluation_model.sampled), samples)

    async def test_real_engine_rejects_legacy_token_reconstruction(self):
        with self.assertRaisesRegex(ValueError, "native token ids"):
            await self.collect(self.engine(ScriptedModel(exact=False)))
        self.assertTrue(all(env.closed for env in self.instances))

    async def test_real_engine_truncation_cannot_count_as_success(self):
        trajectories = await self.collect(
            self.engine(
                ScriptedModel(),
                budget=rl.Budget(
                    max_turns=4,
                    max_generated_tokens=12,
                    max_context_tokens=8192,
                    max_turn_tokens=12,
                ),
            )
        )
        self.assertTrue(
            all(t.truncated is not None and t.reward == 0 for t in trajectories)
        )
        self.assertTrue(all(t.metrics["task_success"] == 0 for t in trajectories))
        self.assertTrue(all(t.generated_tokens <= 12 for t in trajectories))
        for trajectory in trajectories:
            self.assertEqual(
                trajectory_case_identity(
                    trajectory, cases={r["order_id"]: r for r in ROWS}
                )["case_id"],
                trajectory.stateset_episode_id,
            )

    async def test_real_engine_keeps_infrastructure_failure_unscored(self):
        with self.assertRaisesRegex(
            rl.InfrastructureError, "environment operation failed"
        ):
            await self.collect(self.engine(ScriptedModel(), broken=True))
        self.assertTrue(all(env.closed for env in self.instances))

    async def test_refund_limits_preserve_failure_rewards_in_real_training_batch(self):
        completed = await self.collect(self.engine(ScriptedModel(mixed=True)))
        failure = next(t for t in completed if t.reward == -1 and t.truncated is None)
        for limits in (
            {"max_generated_tokens": 12, "max_turn_tokens": 12},
            {"max_turns": 2},
            {"max_context_tokens": 32},
        ):
            with self.subTest(limits=limits):
                budget = {
                    "max_turns": 4,
                    "max_generated_tokens": 1024,
                    "max_context_tokens": 8192,
                    "max_turn_tokens": 256,
                    **limits,
                }
                trajectories = await self.collect(
                    self.engine(
                        ScriptedModel(),
                        budget=rl.Budget(**budget),
                        truncation_reward=EVALUATION_SETTINGS["truncation_reward"],
                    )
                )
                self.assertTrue(
                    all(
                        t.reward == -1
                        and t.metrics["task_success"] == 0
                        and t.truncated is not None
                        for t in trajectories
                    )
                )
                truncated = trajectories[0]
                trace = trajectory_environment_trace(truncated)
                replay = await replay_refund_trace(
                    "refund-v1",
                    next(
                        row
                        for row in ROWS
                        if row["order_id"] == truncated.stateset_episode_id
                    ),
                    trace,
                )
                self.assertFalse(replay["success"])
                self.assertEqual(replay["reward"], -1)
                self.assertEqual(trace["terminal"]["truncated"], truncated.truncated)
                self.assertEqual(trace["terminal"]["reward"], -1)
                self.assertEqual(
                    trajectory_case_identity(
                        truncated, cases={r["order_id"]: r for r in ROWS}
                    )["case_hash"],
                    content_hash(
                        next(
                            r
                            for r in ROWS
                            if r["order_id"] == truncated.stateset_episode_id
                        )
                    ),
                )
                if not truncated.generated_tokens:
                    continue  # A context limit can fire before the first sample.
                options = {
                    "estimator": rl.GroupCentered(),
                    "normalize": "token",
                    "min_members": 2,
                    "current_step": 0,
                    "max_staleness": 0,
                }
                # The old SDK default rewarded truncation relative to failure.
                old_data, _ = build_batch(
                    [[failure, truncated]],
                    truncation=rl.Truncation(train="zero_reward"),
                    **options,
                )
                self.assertTrue(any(a > 0 for d in old_data for a in d["advantages"]))
                data, _ = build_batch(
                    [[failure, truncated]],
                    truncation=rl.Truncation(**TRUNCATION_POLICY),
                    **options,
                )
                self.assertEqual(data, [])  # Equal failures have zero advantage.

    async def test_real_environment_timeout_survives_trajectory_serialization(self):
        trajectories = await self.collect(
            self.engine(
                ScriptedModel(finish=False),
                budget=rl.Budget(
                    max_turns=8,
                    max_generated_tokens=1024,
                    max_context_tokens=8192,
                    max_turn_tokens=256,
                ),
            )
        )
        for trajectory in trajectories:
            self.assertEqual(trajectory.reward, -1)
            self.assertEqual(trajectory.metrics["task_success"], 0)
            self.assertIsNone(trajectory.truncated)
            self.assertEqual(trajectory_truncation(trajectory), "environment_timeout")
            restored = type(trajectory).from_state_dict(trajectory.state_dict())
            replay = await replay_refund_trace(
                "refund-v1",
                next(
                    row
                    for row in ROWS
                    if row["order_id"] == trajectory.stateset_episode_id
                ),
                trajectory_environment_trace(restored),
            )
            self.assertFalse(replay["external_stop_declared"])
            self.assertEqual(
                trajectory_environment_trace(restored),
                trajectory_environment_trace(trajectory),
            )
            self.assertEqual(
                trajectory_case_identity(
                    restored, cases={r["order_id"]: r for r in ROWS}
                ),
                trajectory_case_identity(
                    trajectory, cases={r["order_id"]: r for r in ROWS}
                ),
            )
            self.assertEqual(trajectory_truncation(restored), "environment_timeout")

    async def test_real_engine_enforces_durable_admission_before_sampling(self):
        ledger = RolloutAdmissionBudget(
            self.output / "budget.json",
            limit=1024,
            trajectory_tokens=1024,
            fingerprint="native",
            create=True,
        )

        async def admit():
            await asyncio.to_thread(ledger.reserve)

        model = ScriptedModel()
        with self.assertRaises(RolloutBudgetExceeded):
            await self.collect(self.engine(model, before_reset=admit))
        self.assertEqual(ledger.snapshot()["admitted_trajectories"], 1)
        before = len(model.sampled)
        with self.assertRaises(RolloutBudgetExceeded):
            await self.collect(self.engine(model, before_reset=admit))
        self.assertEqual(len(model.sampled), before)

    def trainer(self, model):
        return rl.AsyncTrainer(
            engine=self.engine(model),
            optimizer=rl.Adam(lr=1e-5),
            completion=rl.GroupCompletion(mode="wait", min_members=2),
            advantage=rl.GroupCentered(),
            truncation=rl.Truncation(**TRUNCATION_POLICY),
            normalize="token",
            loss="cispo",
            groups_per_step=1,
            group_size=2,
            max_staleness=0,
            checkpoint=rl.Checkpointing(
                self.output / "training", weights_every=1, on_signal=()
            ),
        )

    async def test_real_engine_regenerates_interrupted_sandbox_and_charges_admission(
        self,
    ):
        ledger = RolloutAdmissionBudget(
            self.output / "budget.json",
            limit=2048,
            trajectory_tokens=1024,
            fingerprint="native",
            create=True,
        )

        async def admit():
            await asyncio.to_thread(ledger.reserve)

        paused = asyncio.Event()
        engine = self.engine(
            ScriptedModel(), before_reset=admit, pause_after_refund=paused
        )
        async with engine:
            engine.submit(
                ROWS[0],
                group_size=1,
                completion=rl.GroupCompletion(mode="wait", min_members=1),
            )
            await asyncio.wait_for(paused.wait(), timeout=5)
            saved = await engine.state_dict()
        self.assertTrue(self.instances[0].closed)
        self.assertEqual(self.instances[0].last_state.context["refunds"], [1250])
        resumed = self.engine(ScriptedModel(), before_reset=admit)
        async with resumed:
            resumed.load_state_dict(saved)
            result = await asyncio.wait_for(resumed.next_group(), timeout=5)
            resumed.accept(result)
            resumed.acknowledge(result.id)
        self.assertEqual(resumed.recovery_metrics["trajectories_dropped_on_resume"], 1)
        self.assertEqual(resumed.recovery_metrics["trajectories_recovered"], 0)
        self.assertEqual(result.trajectories[0].reward, 1)
        self.assertEqual(self.instances[1].last_state.context["refunds"], [1250])
        self.assertEqual(ledger.snapshot()["admitted_trajectories"], 2)

    async def test_native_campaign_record_failure_closes_real_trainer(self):
        rows = [*ROWS, {"order_id": "C", "amount_cents": 750, "eligible": True}]
        splits = {"train": rows[:1], "validation": rows[1:2], "test": rows[2:]}
        args = SimpleNamespace(
            output=self.output,
            base_model=ScriptedModel.base_model,
            seed=42,
            steps=2,
            concurrency=4,
            max_staleness=0,
            learning_rate=1e-5,
            checkpoint=None,
            evaluate_only=False,
            dry_run=False,
            benchmark="refund-v1",
        )
        prepare_run(args, splits)
        trainers = []

        class TrackedTrainer(rl.AsyncTrainer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                trainers.append(self)

        model = ScriptedModel(mixed=True)
        evaluator = ScriptedModel()
        with (
            patch.object(sys.modules[__name__], "ROWS", rows),
            patch.object(rl, "AsyncTrainer", TrackedTrainer),
            patch.object(
                RiverTrainingProgress,
                "observe",
                side_effect=OSError("progress storage failed"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "progress storage failed"):
                await asyncio.wait_for(
                    campaign(
                        model,
                        ScriptedSession(evaluator),
                        ScriptedRenderer(),
                        args,
                        splits,
                    ),
                    15,
                )
        self.assertEqual(len(trainers), 1)
        self.assertFalse(trainers[0]._running)
        self.assertEqual(trainers[0]._pending_backwards, [])
        self.assertEqual(model.optimizer_calls, 1)
        self.assertTrue(all(env.closed for env in self.instances))
        self.assertFalse((self.output / "test_attempt.json").exists())
        self.assertFalse(any(order == "C" for order, _, _ in evaluator.sampled))

    async def test_native_campaign_drains_cancelled_validation_write(self):
        import stateset_agents.training.river_refund as module

        rows = [*ROWS, {"order_id": "C", "amount_cents": 750, "eligible": True}]
        splits = {"train": rows[:1], "validation": rows[1:2], "test": rows[2:]}
        args = SimpleNamespace(
            output=self.output,
            base_model=ScriptedModel.base_model,
            seed=42,
            steps=2,
            concurrency=4,
            max_staleness=0,
            learning_rate=1e-5,
            checkpoint=None,
            evaluate_only=False,
            dry_run=False,
            benchmark="refund-v1",
        )
        prepare_run(args, splits)
        entered, closing = asyncio.Event(), asyncio.Event()
        release, finished = threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        original_write = module.atomic_json

        def write(path, value):
            if path.name == "validation_results.json" and value:
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(10):
                    raise TimeoutError("validation write was not released")
                try:
                    return original_write(path, value)
                finally:
                    finished.set()
            return original_write(path, value)

        class TrackedEvaluator(rl.Evaluator):
            async def close(self):
                closing.set()
                await super().close()

        class TrackedTrainer(rl.AsyncTrainer):
            async def run(self, *a, **kw):
                iterator = super().run(*a, **kw)
                try:
                    async for step in iterator:
                        await asyncio.wait_for(entered.wait(), 5)
                        yield step
                finally:
                    await iterator.aclose()

        with (
            patch.object(sys.modules[__name__], "ROWS", rows),
            patch.object(rl, "AsyncTrainer", TrackedTrainer),
            patch.object(rl, "Evaluator", TrackedEvaluator),
            patch.object(module, "atomic_json", write),
            patch.object(
                RiverTrainingProgress,
                "observe",
                side_effect=OSError("progress storage failed"),
            ),
        ):
            task = asyncio.create_task(
                campaign(
                    ScriptedModel(mixed=True),
                    ScriptedSession(ScriptedModel()),
                    ScriptedRenderer(),
                    args,
                    splits,
                )
            )
            try:
                await asyncio.wait_for(closing.wait(), 5)
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                    self.assertFalse(finished.is_set())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
        self.assertTrue(finished.is_set())
        self.assertTrue(all(env.closed for env in self.instances))
        self.assertFalse((self.output / "test_attempt.json").exists())
        self.assertFalse((self.output / "test_report.json").exists())

    async def test_native_campaign_without_optimizer_updates_keeps_holdout_sealed(self):
        rows = [*ROWS, {"order_id": "C", "amount_cents": 750, "eligible": True}]
        splits = {"train": rows[:1], "validation": rows[1:2], "test": rows[2:]}
        args = SimpleNamespace(
            output=self.output,
            base_model=ScriptedModel.base_model,
            seed=42,
            steps=1,
            concurrency=4,
            max_staleness=0,
            learning_rate=1e-5,
            checkpoint=None,
            evaluate_only=False,
            dry_run=False,
            benchmark="refund-v1",
        )
        prepare_run(args, splits)
        model = ScriptedModel()  # Every response has the same successful reward.
        evaluation_model = ScriptedModel()
        with patch.object(sys.modules[__name__], "ROWS", rows):
            with self.assertRaisesRegex(ValueError, "No optimizer updates.*sealed"):
                await asyncio.wait_for(
                    campaign(
                        model,
                        ScriptedSession(evaluation_model),
                        ScriptedRenderer(),
                        args,
                        splits,
                    ),
                    15,
                )
        self.assertEqual(model.optimizer_calls, 0)
        self.assertEqual(model.backward_batches, [])
        activity = json.loads((self.output / "training_activity.json").read_text())
        self.assertEqual(activity["status"], "no_updates_observed")
        self.assertEqual(activity["completed_batches"], 1)
        self.assertEqual(activity["observed_optimizer_updates"], 0)
        self.assertEqual(activity["observed_skipped_batches"], 1)
        self.assertEqual(activity["unknown_update_batches"], 0)
        self.assertFalse((self.output / "test_attempt.json").exists())
        self.assertFalse((self.output / "test_results.json").exists())
        self.assertFalse(any(order == "C" for order, _, _ in evaluation_model.sampled))

    async def test_native_campaign_stops_skipped_batches_early_and_on_resume(self):
        rows = [*ROWS, {"order_id": "C", "amount_cents": 750, "eligible": True}]
        splits = {"train": rows[:1], "validation": rows[1:2], "test": rows[2:]}
        args = SimpleNamespace(
            output=self.output,
            base_model=ScriptedModel.base_model,
            seed=42,
            steps=8,
            concurrency=4,
            max_staleness=0,
            learning_rate=1e-5,
            checkpoint=None,
            evaluate_only=False,
            dry_run=False,
            benchmark="refund-v1",
            zero_update_patience=2,
        )
        prepare_run(args, splits)
        for resumed in (False, True):
            with self.subTest(resumed=resumed):
                model = ScriptedModel()
                with patch.object(sys.modules[__name__], "ROWS", rows):
                    with self.assertRaisesRegex(ValueError, "2 consecutive skipped"):
                        await asyncio.wait_for(
                            campaign(
                                model,
                                ScriptedSession(ScriptedModel()),
                                ScriptedRenderer(),
                                args,
                                splits,
                            ),
                            15,
                        )
                self.assertEqual(model.optimizer_calls, 0)
                records = json.loads(
                    (self.output / "training_metrics.json").read_text()
                )
                self.assertEqual([row["step"] for row in records], [1, 2])
                stop = json.loads((self.output / "training_stop.json").read_text())
                self.assertEqual(stop["step"], 2)
                self.assertEqual(stop["training_progress_hash"], content_hash(records))
                self.assertFalse((self.output / "test_attempt.json").exists())
                self.assertFalse((self.output / "test_results.json").exists())
                self.assertTrue(all(env.closed for env in self.instances))
        receipts = json.loads((self.output / "recovery_receipts.json").read_text())
        self.assertTrue(any(row["completed_batches"] == 2 for row in receipts.values()))

    async def test_real_trainer_resumes_only_remaining_batches(self):
        first_model = ScriptedModel(mixed=True)
        iterator = self.trainer(first_model).run(ROWS[:1], steps=2)
        first = await anext(iterator)
        self.assertEqual(first.n, 1)
        await iterator.aclose()
        progress = RiverTrainingProgress(
            self.output, steps=2, run_manifest_hash="native"
        )
        recovered = []

        async def after_recovery(count):
            recovered.append(count)
            progress.reconcile(count)

        resumed_model = ScriptedModel(mixed=True)
        emitted = []
        async for step in self.trainer(resumed_model).run(
            ROWS[:1], steps=2, after_recovery=after_recovery
        ):
            emitted.append(step.n)
            progress.observe(step.n, step.metrics)
        self.assertEqual(recovered, [1])
        self.assertEqual(emitted, [2])
        self.assertEqual(resumed_model.optimizer_calls, 1)
        self.assertEqual(resumed_model.step, 2)
        self.assertIsNone(progress.records[0]["metrics"])
        activity = summarize_training_activity(progress.records)
        self.assertEqual(activity["observed_optimizer_updates"], 1)
        self.assertEqual(activity["unknown_update_batches"], 1)
        require_training_progress(
            progress.records,
            steps=2,
            recovery_receipts=progress.receipts,
            run_manifest_hash="native",
        )

    async def test_real_trainer_restores_commits_without_reemitting_metrics(self):
        model = ScriptedModel(mixed=True)
        first = [step async for step in self.trainer(model).run(ROWS[:1], steps=2)]
        self.assertEqual([s.n for s in first], [1, 2])
        self.assertTrue(all("train/gradient_scale" in step.metrics for step in first))
        self.assertEqual(model.optimizer_calls, 2)
        self.assertTrue(model.backward_batches)
        # Simulate losing the local observation after the SDK's durable commit.
        progress = RiverTrainingProgress(
            self.output, steps=2, run_manifest_hash="native"
        )
        progress.observe(1, first[0].metrics)
        recovered = []

        async def after_recovery(count):
            recovered.append(count)
            progress.reconcile(count)

        resumed = ScriptedModel(mixed=True)
        emitted = [
            step
            async for step in self.trainer(resumed).run(
                ROWS[:1], steps=2, after_recovery=after_recovery
            )
        ]
        self.assertEqual(emitted, [])
        self.assertEqual(recovered, [2])
        self.assertEqual(resumed.loaded_optimizer, [True])
        self.assertEqual(resumed.optimizer_calls, 0)
        self.assertIsNone(progress.records[1]["metrics"])
        activity = summarize_training_activity(progress.records)
        self.assertEqual(activity["observed_optimizer_updates"], 1)
        self.assertEqual(activity["unknown_update_batches"], 1)
        require_training_progress(
            progress.records,
            steps=2,
            recovery_receipts=progress.receipts,
            run_manifest_hash="native",
        )


if __name__ == "__main__":
    unittest.main()

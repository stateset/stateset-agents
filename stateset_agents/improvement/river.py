"""One-attempt River SFT and independently replayed synthetic refund evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import uuid
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import evaluation_report
from stateset_agents.remote.river import RiverExecutor

from .storage import private_directory, read_json, write_bytes
from .workflow import ExperimentConfig

AUDITED_ENGINE_SHA256 = (
    "03f95175ba23698affd93ad2c074b9eef69b81561483930248ac3ec6c9413409"
)
AUDITED_ENGINE_SHA256S: frozenset[str] = frozenset({AUDITED_ENGINE_SHA256})


def verify_sampler_runtime(
    *, allow_behavioral_fallback: bool = False
) -> dict[str, Any]:
    """Pin the inspected native seed schedule before sampling with its audit.

    Matches against :data:`AUDITED_ENGINE_SHA256S`. When
    ``allow_behavioral_fallback=True``, also permits a newer ``river-client``
    patch version inside the supported ``>=0.11.0,<0.12`` range if its
    ``RolloutEngine`` retains the required ``seed``, ``temperature``,
    ``top_p``, ``top_k``, and ``rollout`` signature contracts.
    """
    from river_client import rl

    path = Path(inspect.getfile(rl.RolloutEngine))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in AUDITED_ENGINE_SHA256S:
        return {"mode": "exact_sha256", "sha256": digest, "verified": True}
    if allow_behavioral_fallback:
        init_params = inspect.signature(rl.RolloutEngine.__init__).parameters
        rollout_params = inspect.signature(rl.RolloutEngine.rollout).parameters
        if {"seed", "temperature", "top_p", "top_k"} <= set(init_params) and {
            "group_size",
            "completion",
        } <= set(rollout_params):
            return {
                "mode": "behavioral_signature",
                "sha256": digest,
                "verified": True,
            }
    raise ValueError(
        f"River rollout engine changed (sha256={digest[:12]}...); "
        "audit its seed schedule before use"
    )


class _SingleAttemptExecutor(RiverExecutor):
    MAX_TRANSIENT_ATTEMPTS = 1


def river_client(timeout_seconds: int = 180) -> Any:
    """Create an owned SDK client with transport/submit retries disabled."""
    import river_client as river

    return river.Client(
        api_key=os.environ["RIVER_API_KEY"],
        timeout=timeout_seconds,
        enable_retries=False,
    )


class RiverBackend:
    """Live River adapter; evaluation uses fictional policy fixtures, never payments.

    A failed or lost training session is not automatically restarted. Initializing
    from the parent inference checkpoint starts a fresh optimizer. Request timeouts
    and bounded work do not constitute a hard dollar or wall-clock spending cap.
    """

    evidence_mode = "live"
    adapter_id = "river-refund-delegation-v1"

    def train(
        self, config: ExperimentConfig, dataset: Path, output: Path, seed: int
    ) -> dict[str, Any]:
        """Train one continuation with a fresh optimizer and unique checkpoint name."""
        from stateset_agents.remote.job import RemoteJobSpec
        from stateset_agents.training.sft import parse_chat_dataset

        rows = read_json(dataset)
        payload = "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ).encode()
        if not rows or len(parse_chat_dataset(payload)) != len(rows):
            raise ValueError(
                "Prepared training rows must all satisfy the SFT input contract"
            )
        training_input = output / "training.jsonl"
        write_bytes(training_input, payload)

        artifacts = private_directory(output / f"checkpoint-{uuid.uuid4().hex}")
        spec = RemoteJobSpec(
            dataset=training_input,
            base_model=config.base_model,
            output_dir=artifacts,
            num_epochs=config.epochs,
            lora_r=config.lora_rank,
            learning_rate=config.learning_rate,
            max_length=config.max_length,
            per_device_batch_size=config.batch_size,
            gradient_accumulation_steps=1,
            seed=seed,
            harvest={
                "seed": seed,
                "shuffle": True,
                "checkpoint": config.parent_checkpoint,
            },
        )
        with closing(river_client(config.request_timeout_seconds)) as client:
            executor = _SingleAttemptExecutor(client=client)
            handle = executor.submit(spec)
            executor.fetch(handle, artifacts)
        return dict(read_json(artifacts / "river_checkpoint.json"))

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
        """Sample each case once and independently replay every tool transition."""
        from stateset_agents.remote.river_renderers import get_river_renderer
        from stateset_agents.remote.river_runtime import (
            require_model_access,
            require_native_runtime,
        )
        from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget

        if (
            config.environment != "refund-delegation-v1"
            or config.evaluation
            != ExperimentConfig(config.base_model, config.parent_checkpoint).evaluation
        ):
            raise ValueError(
                "This adapter supports only the fixed synthetic delegation protocol"
            )
        if any(case.get("order_id") != key for key, case in cases.items()):
            raise ValueError("Refund evaluation record_id must equal its order_id")
        require_native_runtime()
        verify_sampler_runtime()
        budget = RolloutAdmissionBudget(
            output / "admission.json",
            limit=len(cases) * 1024,
            trajectory_tokens=1024,
            fingerprint=checkpoint,
            create=True,
        )
        with closing(river_client(config.request_timeout_seconds)) as client:
            require_model_access(client, config.base_model)
            renderer = get_river_renderer(config.base_model, thinking=False)
            with client.session(
                experiment="stateset-private-workflow", role="paired-evaluation"
            ) as session:
                return asyncio.run(
                    self._evaluate(
                        config,
                        cases,
                        checkpoint,
                        seed,
                        split,
                        implementation,
                        session,
                        renderer,
                        budget,
                    )
                )

    async def _evaluate(
        self,
        config: ExperimentConfig,
        cases: dict[str, Any],
        checkpoint: str,
        seed: int,
        split: str,
        implementation: dict[str, Any],
        session: Any,
        renderer: Any,
        budget: Any,
    ) -> dict[str, Any]:
        from river_client import Checkpoint, rl

        from stateset_agents.evaluation.refund_delegation_trace import (
            replay_delegation_trace,
        )
        from stateset_agents.remote.river_environment import (
            trajectory_case_identity,
            trajectory_truncation,
        )
        from stateset_agents.remote.river_refund_delegation import (
            river_refund_delegation_factory,
            trajectory_delegation_trace,
        )
        from stateset_agents.utils.async_calls import closing_async_generator

        async def reserve() -> None:
            await asyncio.to_thread(budget.reserve)

        pointer = Checkpoint(path=checkpoint, step=0, checkpoint_type="inference")
        engine = rl.RolloutEngine(
            rl.CheckpointSampler(
                session,
                base_model=config.base_model,
                checkpoint=pointer,
                tokenizer=renderer.tokenizer,
            ),
            env=river_refund_delegation_factory(before_reset=reserve),
            renderer=renderer,
            budget=rl.Budget(
                **{k: v for k, v in config.evaluation.items() if k.startswith("max_")}
            ),
            schedule=rl.Schedule(concurrency=1),
            temperature=0,
            top_p=1.0,
            top_k=-1,
            seed=seed,
        )
        outcomes = []
        async with closing_async_generator(
            engine.rollout(
                list(cases.values()),
                group_size=1,
                completion=rl.GroupCompletion(mode="wait", min_members=1),
            )
        ) as groups:
            async for group in groups:
                for trajectory in group:
                    outcome = {
                        **trajectory_case_identity(trajectory, cases=cases),
                        "environment_trace": trajectory_delegation_trace(trajectory),
                        "reward": trajectory.reward,
                        "success": trajectory.metrics.get("task_success", 0) == 1,
                        "policy_violations": trajectory.metrics.get(
                            "policy_violations", 0
                        ),
                        "tool_calls": trajectory.metrics.get("tool_calls", 0),
                        "executed_tool_calls": trajectory.metrics.get(
                            "executed_tool_calls", 0
                        ),
                        "duplicate_intents": trajectory.metrics.get(
                            "duplicate_intents", 0
                        ),
                        "generated_tokens": trajectory.generated_tokens,
                        "elapsed_seconds": trajectory.elapsed,
                        "truncated": trajectory_truncation(trajectory),
                    }
                    replay = await replay_delegation_trace(
                        cases[outcome["case_id"]], outcome["environment_trace"]
                    )
                    for field in (
                        "case_hash",
                        "reward",
                        "success",
                        "policy_violations",
                        "tool_calls",
                        "executed_tool_calls",
                        "duplicate_intents",
                        "truncated",
                    ):
                        if replay[field] != outcome[field]:
                            raise ValueError(
                                "Provider outcome disagrees with independent trace replay"
                            )
                    outcomes.append(outcome)
        return dict(
            evaluation_report(
                environment=config.environment,
                base_model=config.base_model,
                seed=seed,
                cases=cases,
                settings=config.evaluation,
                checkpoint=asdict(pointer),
                outcomes=outcomes,
                case_families={key: case["family"] for key, case in cases.items()},
                implementation=implementation,
                split=split,
            )
        )

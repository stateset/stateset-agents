"""Independent training worker: python -m stateset_agents.api.training_worker."""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import threading
from dataclasses import replace
from pathlib import Path

from stateset_agents.utils.async_calls import run_sync_owned
from stateset_agents.utils.clock import utc_now

from .schemas import TrainingRequest
from .services.training_service import TrainingService, build_reward
from .training_queue import TrainingLeaseLostError, TrainingQueue

logger = logging.getLogger(__name__)


async def work_once(
    queue: TrainingQueue,
    *,
    lease_seconds: float = 60,
    output_root: str | Path | None = None,
) -> bool:
    """Execute one claimed job; heartbeat independently of model/event-loop stalls."""
    claim = queue.claim(lease_seconds=lease_seconds)
    if claim is None:
        return False
    job_id, token, snapshot = claim
    stop = threading.Event()
    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    service = TrainingService(output_root=output_root)
    service.queue = queue
    service.training_jobs[job_id] = snapshot
    lost_lease: list[BaseException] = []

    def renew() -> None:
        while not stop.wait(lease_seconds / 3):
            try:
                cancelled = queue.heartbeat(
                    job_id,
                    token,
                    lease_seconds=lease_seconds,
                    progress=copy.deepcopy(snapshot),
                )
                if cancelled:
                    loop.call_soon_threadsafe(cancel.set)
            except Exception as exc:
                lost_lease.append(exc)
                loop.call_soon_threadsafe(cancel.set)
                return

    heartbeat = threading.Thread(
        target=renew, name=f"training-lease-{job_id}", daemon=True
    )
    heartbeat.start()
    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment

        from .model_policy import client_model_uses_stub

        request = TrainingRequest.model_validate(snapshot["config"])
        use_stub_model = client_model_uses_stub(
            request.agent_config.model_name, roles=None
        )
        overrides = dict(request.training_config_overrides or {})
        # A lease owns a distinct output directory. A stale process cannot replace
        # the artifacts of a later explicitly resumed job.
        root = service.output_root()
        job_dir = (root / job_id).resolve()
        declared_dir = Path(snapshot.get("output_dir", job_dir)).resolve()
        if declared_dir != job_dir or not job_dir.is_relative_to(root):
            raise ValueError(
                "Queued training output directory is outside the worker's configured root"
            )
        overrides["output_dir"] = str(job_dir / token)
        request.training_config_overrides = overrides
        request.resume_from_checkpoint = snapshot.get(
            "resume_from_checkpoint", request.resume_from_checkpoint
        )
        if request.resume_from_checkpoint:
            # Resolve ownership and symlinks again immediately before execution;
            # API admission does not authorize arbitrary later filesystem reads.
            request.resume_from_checkpoint = await run_sync_owned(
                service._resolve_resume,
                request.resume_from_checkpoint,
                snapshot.get("resume_user_id", snapshot.get("user_id")),
                root,
            )
        if queue.heartbeat(job_id, token, lease_seconds=lease_seconds):
            cancel.set()
        await service._run_training(
            job_id,
            MultiTurnAgent(
                replace(
                    AgentConfig(
                        **request.agent_config.model_dump(),
                        model_revision=overrides.get("model_revision"),
                    ),
                    use_stub_model=use_stub_model,
                )
            ),
            ConversationEnvironment(scenarios=request.environment_scenarios),
            build_reward(request.reward_config),
            request,
            cancel,
        )
        current = asyncio.current_task()
        if current is not None and getattr(current, "cancelling", lambda: 0)():
            raise asyncio.CancelledError
        snapshot["metrics"]["artifact_directory"] = overrides["output_dir"]
    except asyncio.CancelledError:
        snapshot.update(status="cancelled", completed_at=utc_now())
        raise
    except TrainingLeaseLostError as exc:
        lost_lease.append(exc)
        cancel.set()
    except Exception as exc:
        snapshot.update(status="failed", error=str(exc), completed_at=utc_now())
    finally:
        stop.set()
        try:
            if lost_lease:
                logger.warning(
                    "Worker lease lost for %s; terminal write is fenced", job_id
                )
            else:
                # Commit before yielding to thread-pool cleanup: stopping the
                # heartbeat then awaiting join can let a short lease expire.
                try:
                    queue.finish(job_id, token, snapshot)
                except TrainingLeaseLostError:
                    # A lease may expire after the final training callback and
                    # before this write. Keep serving other jobs; the next claim
                    # records the expired job as failed without replaying it.
                    logger.warning(
                        "Worker lease expired before publishing result for %s", job_id
                    )
        finally:
            await run_sync_owned(heartbeat.join)
    return True


async def run_worker(queue: TrainingQueue, *, once: bool = False) -> None:
    """Drain queued jobs, keeping GPU work outside API request processes."""
    while True:
        worked = await work_once(queue)
        if once:
            return
        if not worked:
            await asyncio.sleep(1)


def main() -> None:
    """Start a worker using the same durable queue and capacity as API replicas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        run_worker(
            TrainingQueue(args.queue, max_concurrent_jobs=args.max_concurrent_jobs),
            once=args.once,
        )
    )


if __name__ == "__main__":
    main()

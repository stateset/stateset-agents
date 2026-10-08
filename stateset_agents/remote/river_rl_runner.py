"""Durable synchronous River RL driver, separated from provider transport."""

from __future__ import annotations

import json
import math
import time
from asyncio import CancelledError
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobHandle, JobStatus
from stateset_agents.remote.river_rl import (
    RiverRLConfig,
    atomic_json,
    normalize_datums,
    run_fingerprint,
)


def _read_usage(path: Path, fingerprint: str) -> dict[str, Any]:
    """Read existing usage without resetting missing or corrupt reservations."""
    try:
        usage = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("RL token usage ledger is missing or unreadable") from exc
    if not isinstance(usage, dict) or usage.get("fingerprint") != fingerprint:
        raise ValueError("Token budget belongs to a different RL run")
    used = usage.get("generated_tokens_or_reserved")
    if type(used) is not int or used < 0:
        raise ValueError("Invalid RL token usage counter")
    return usage


def _validate_state(
    state: Any, config: RiverRLConfig, fingerprint: str, validation_count: int
) -> dict[str, Any]:
    """Require consistent committed progress before restoring optimizer weights.

    This checks local evidence, not provider checkpoint contents or authenticity.
    Valid schema-2 records remain resumable without rewriting their artifacts.
    """

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise ValueError(f"Invalid River RL state: {detail}")

    def integer(value: Any, lower: int, upper: int) -> bool:
        return type(value) is int and lower <= value <= upper

    def loss(value: Any) -> bool:
        try:
            return value is None or (
                type(value) in (int, float) and math.isfinite(value)
            )
        except OverflowError:
            return False

    require(isinstance(state, dict), "expected an object")
    require(
        type(state.get("schema_version")) is int and state["schema_version"] == 2,
        "unsupported schema",
    )
    require(state.get("fingerprint") == fingerprint, "run fingerprint differs")
    completed = state.get("completed_round")
    require(integer(completed, 0, config.rounds), "completed round is outside the plan")
    for name in ("training_checkpoint", "best_checkpoint"):
        uri = state.get(name)
        require(
            isinstance(uri, str)
            and uri.startswith("river://")
            and len(uri) > len("river://")
            and not any(c.isspace() for c in uri),
            f"missing or invalid {name}",
        )
    rounds = state.get("rounds")
    require(
        isinstance(rounds, list) and all(isinstance(r, dict) for r in rounds),
        "round history must be objects",
    )
    first = 0 if validation_count else 1
    require(
        all(type(r.get("round")) is int for r in rounds)
        and [r["round"] for r in rounds] == list(range(first, completed + 1)),
        "round history is incomplete or out of order",
    )
    steps = best_steps = best_round = 0
    last_loss = best_loss = None
    best_passed = -1
    for entry in rounds:
        rnd = entry["round"]
        if rnd:
            skipped = entry.get("skipped", False)
            require(type(skipped) is bool, "invalid skipped-round flag")
            for key in ("datums", "contributing_tokens", "groups", "skipped_groups"):
                value = entry.get(key)
                require(type(value) is int and value >= 0, f"invalid round {key}")
            require(
                entry["groups"] > 0 and entry["skipped_groups"] <= entry["groups"],
                "invalid group counts",
            )
            require(
                (entry["datums"] == 0) == skipped
                and (entry["contributing_tokens"] == 0) == skipped,
                "skipped round contradicts training data",
            )
            if skipped:
                require(
                    "backward_metrics" not in entry,
                    "skipped round has optimizer evidence",
                )
                continue
            metrics = entry.get("backward_metrics")
            require(
                isinstance(metrics, list) and bool(metrics), "missing backward metrics"
            )
            require(
                all(
                    isinstance(m, dict)
                    and "loss_sum" in m
                    and "loss_mean" in m
                    and loss(m["loss_sum"])
                    and loss(m["loss_mean"])
                    for m in metrics
                ),
                "invalid backward metrics",
            )
            last_loss = (
                sum(m["loss_sum"] for m in metrics)
                if all(m["loss_sum"] is not None for m in metrics)
                else None
            )
            require(loss(last_loss), "nonfinite aggregate loss")
            steps += 1
        if validation_count:
            passed = entry.get("passed")
            require(
                integer(passed, 0, validation_count)
                and type(entry.get("total")) is int
                and entry["total"] == validation_count,
                "invalid validation counts",
            )
            selected = passed > best_passed
        else:
            passed = -1
            selected = True
        if selected:
            best_round, best_steps, best_loss, best_passed = (
                rnd,
                steps,
                last_loss,
                passed,
            )
    for key, expected in (
        ("steps", steps),
        ("best_round", best_round),
        ("best_steps", best_steps),
    ):
        require(
            type(state.get(key)) is int and state[key] == expected,
            f"{key} contradicts committed round history",
        )
    for key, expected_loss in (("loss", last_loss), ("best_loss", best_loss)):
        require(
            key in state and loss(state[key]) and state[key] == expected_loss,
            f"{key} contradicts committed metrics",
        )
    require("best_eval" in state, "missing selected evaluation")
    evaluation = state["best_eval"]
    if not validation_count:
        require(evaluation is None, "unexpected validation evidence")
    else:
        require(isinstance(evaluation, dict), "missing validation evidence")
        require(
            type(evaluation.get("passed")) is int
            and evaluation["passed"] == best_passed
            and type(evaluation.get("total")) is int
            and evaluation["total"] == validation_count,
            "selected evaluation contradicts round history",
        )
        results = evaluation.get("results")
        require(
            isinstance(results, list)
            and len(results) == validation_count
            and all(isinstance(row, dict) for row in results),
            "incomplete selected evaluation",
        )
        decisions = [
            (
                row.get("passed", (row.get("checks") or {}).get("passed"))
                if isinstance(row.get("checks", {}), dict)
                else None
            )
            for row in results
        ]
        require(
            all(type(passed) is bool for passed in decisions)
            and sum(passed is True for passed in decisions) == best_passed,
            "selected evaluation outcomes contradict pass count",
        )
    return cast(dict[str, Any], state)


class _BudgetModel:
    """Reserve generated tokens durably before each sampling request.

    An uncertain or malformed response keeps its full reservation. Only complete
    responses with exact token data receive refunds. The driver initializes the
    ledger once and serializes requests under its run lock. Input/training tokens
    are not priced by this limit. Recovery never rolls this counter back.
    """

    def __init__(self, model: Any, path: Path, limit: int | None, fingerprint: str):
        self.model, self.path, self.limit, self.fingerprint = (
            model,
            path,
            limit,
            fingerprint,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def sample(self, prompts: Any = None, **kwargs: Any) -> Any:
        batch = prompts if prompts is not None else kwargs.get("prompt_token_ids")
        if not isinstance(batch, (list, tuple)) or not batch:
            raise ValueError("Sampling requires a nonempty prompt batch")
        count = len(batch)
        samples = kwargs.get("num_samples", 1)
        max_tokens = kwargs.get("max_tokens")
        if type(samples) is not int or samples < 1:
            raise ValueError("Sampling num_samples must be a positive integer")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Sampling max_tokens must be a positive integer")
        reserve = count * samples * max_tokens
        usage = _read_usage(self.path, self.fingerprint)
        used = usage["generated_tokens_or_reserved"]
        if self.limit is not None and used + reserve > self.limit:
            raise ValueError(
                f"RL generated-token budget exhausted: {used} + {reserve} > {self.limit}"
            )
        usage["generated_tokens_or_reserved"] = used + reserve
        atomic_json(self.path, usage)
        result = (
            self.model.sample(prompts, **kwargs)
            if prompts is not None
            else self.model.sample(**kwargs)
        )
        if (
            not isinstance(result, (list, tuple))
            or len(result) != count
            or any(
                not isinstance(group, (list, tuple)) or len(group) != samples
                for group in result
            )
        ):
            raise ValueError("Incomplete River sampling response; reservation retained")
        actual = 0
        exact = True
        overrun = False
        for group in result:
            for sample in group:
                if getattr(sample, "token_data_is_exact", None) is not True:
                    exact = False
                    actual += max_tokens
                    continue
                tokens = getattr(sample, "tokens", None)
                if not isinstance(tokens, (list, tuple)) or any(
                    type(token) is not int or token < 0 for token in tokens
                ):
                    raise ValueError("Invalid River token data; reservation retained")
                actual += len(tokens)
                overrun |= len(tokens) > max_tokens
        if overrun:
            usage["generated_tokens_or_reserved"] = used + max(reserve, actual)
            atomic_json(self.path, usage)
            raise ValueError("River sampling exceeded its reserved token allowance")
        if not exact:
            return result
        usage["generated_tokens_or_reserved"] = used + actual
        atomic_json(self.path, usage)
        return result


def run_rl(
    executor: Any, handle: JobHandle, job: Any, spec: Any, mode: Any
) -> JobHandle:
    """Run rounds with atomic commits and recovery into fresh model sessions.

    A commit contains optimizer weights, selected inference weights, evaluation,
    and round progress. A failed or uncertain update is abandoned with its model;
    a retry restores the last committed training checkpoint in a fresh session.
    """
    from stateset_agents.remote.river import (
        _as_uri,
        _extract,
        _inference_checkpoint,
        _open_session,
        _river_module,
    )

    config = RiverRLConfig.from_knobs(spec.harvest)
    output = Path(spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "rl_state.json"
    usage_path = output / "rl_usage.json"
    fingerprint = run_fingerprint(
        spec, config, executor._rl_scorer_id, initial_checkpoint=mode.checkpoint
    )
    validation_count = len(spec.eval_prompts or [])
    committed_seen = state_path.exists()

    if spec.resume and not committed_seen:
        # A usage-only restart is valid before the initial checkpoint commit.
        # Retained checkpoint/progress artifacts instead prove that the state
        # needed to restore optimizer history has been lost.
        report_path = output / "rl_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if (
            (output / "river_checkpoint.json").exists()
            or any((output / "rollouts").glob("round-*.json"))
            or (isinstance(report, dict) and "completed_round" in report)
        ):
            raise ValueError("Invalid River RL state: committed state is missing")

    def load_state() -> dict[str, Any]:
        if not state_path.exists():
            if committed_seen:
                raise ValueError("Invalid River RL state: committed state disappeared")
            return {}
        return _validate_state(
            json.loads(state_path.read_text()), config, fingerprint, validation_count
        )

    def commit_state(value: dict[str, Any]) -> None:
        nonlocal committed_seen
        _validate_state(value, config, fingerprint, validation_count)
        atomic_json(state_path, value)
        committed_seen = True

    if spec.resume:
        recovery_path = state_path if state_path.exists() else output / "rl_usage.json"
        if not recovery_path.exists():
            raise ValueError(
                "No rl_state.json to resume; start a new run without --resume"
            )
        previous = json.loads(recovery_path.read_text())
        if not isinstance(previous, dict):
            raise ValueError("Invalid River RL state: expected an object")
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(
                "RL resume configuration/data/scorer differs from the saved run"
            )
        _read_usage(usage_path, fingerprint)
        load_state()
    elif state_path.exists() or (output / "rl_usage.json").exists():
        raise ValueError(
            "RL output already contains a run; use --resume or a new directory"
        )
    if spec.dry_run:
        atomic_json(
            output / "rl_report.json", {**mode.dry_run_report, "config": asdict(config)}
        )
        job.status = JobStatus.SUCCEEDED
        return handle

    # Re-publishing a fully committed result is an offline operation. Only
    # remaining training work needs fresh account access or a provider client.
    completed = load_state().get("completed_round", -1) >= config.rounds
    client = None if completed else executor._checked_client(spec, job)

    if not spec.resume:
        # Initialize before opening a session. Retries and resumes may only read
        # this ledger; missing state is never evidence of zero prior spending.
        atomic_json(
            usage_path,
            {"fingerprint": fingerprint, "generated_tokens_or_reserved": 0},
        )

    started = time.monotonic()
    job.status = JobStatus.RUNNING
    state: dict[str, Any] = {}

    def publish(status: str, error: str | None = None) -> None:
        job.steps = state.get("best_steps", 0)
        job.final_loss = state.get("best_loss")
        job.checkpoint_uri = state.get("best_checkpoint")
        report = {
            **state,
            "config": asdict(config),
            "status": status,
            "loss_fn": config.loss_fn,
        }
        usage_path = output / "rl_usage.json"
        if usage_path.exists():
            report["usage"] = json.loads(usage_path.read_text())
        if error:
            report["error"] = error
        atomic_json(output / "rl_report.json", report)
        if state.get("best_eval") is not None:
            atomic_json(output / "eval_results.json", state["best_eval"]["results"])
        if job.checkpoint_uri:
            executor._write_checkpoint_artifacts(job)

    try:
        for attempt in range(executor.MAX_TRANSIENT_ATTEMPTS):
            # Always reload the committed state, including after partial failures.
            state = load_state()
            if state.get("completed_round", -1) >= config.rounds:
                break
            try:
                with _open_session(client, project=output.name or None) as session:
                    checkpoint = (
                        state["training_checkpoint"]
                        if state
                        else _inference_checkpoint(client, mode.checkpoint)
                    )
                    raw_model = session.create_model(
                        base_model=spec.base_model,
                        lora=_river_module(client).LoraConfig(
                            rank=spec.lora_r, seed=config.seed
                        ),
                        checkpoint=checkpoint,
                    )
                    model = _BudgetModel(
                        raw_model,
                        output / "rl_usage.json",
                        config.max_generated_tokens,
                        fingerprint,
                    )
                    if not state:
                        before = mode.greedy_eval(model)
                        best_uri = _as_uri(
                            model.save_weights("rl-best-0", mode="inference")
                        )
                        training_uri = _as_uri(
                            model.save_weights("rl-state-0", mode="training")
                        )
                        state = {
                            "schema_version": 2,
                            "fingerprint": fingerprint,
                            "completed_round": 0,
                            "steps": 0,
                            "loss": None,
                            "training_checkpoint": training_uri,
                            "best_checkpoint": best_uri,
                            "best_round": 0,
                            "best_steps": 0,
                            "best_loss": None,
                            "best_eval": before,
                            "rounds": (
                                [
                                    {
                                        "round": 0,
                                        "passed": before["passed"],
                                        "total": before["total"],
                                    }
                                ]
                                if before
                                else []
                            ),
                        }
                        commit_state(state)
                        publish("running")
                    for rnd in range(state["completed_round"] + 1, config.rounds + 1):
                        data, mean_rewards, records = mode.collect_round(model, rnd)
                        policy_ids = {
                            r["policy_id"]
                            for group in records
                            for r in group["records"]
                            if r.get("policy_id")
                        }
                        if len(policy_ids) > 1:
                            raise ValueError(
                                "Synchronous RL round contains mixed behavior policies"
                            )
                        current = (
                            model.get_policy_version()
                            if hasattr(model, "get_policy_version")
                            else None
                        )
                        policy_id = getattr(current, "id", None)
                        if (
                            policy_id is not None
                            and policy_ids
                            and policy_ids != {policy_id}
                        ):
                            raise ValueError(
                                "Rollout behavior policy differs from the training policy"
                            )
                        normalized, tokens = normalize_datums(
                            data, config.normalization
                        )
                        atomic_json(
                            output / "rollouts" / f"round-{rnd}.json",
                            {
                                "round": rnd,
                                "seed": config.seed + rnd,
                                "groups": records,
                                "normalization": config.normalization,
                                "contributing_tokens": tokens,
                            },
                        )
                        entry: dict[str, Any] = {
                            "round": rnd,
                            "datums": len(normalized),
                            "contributing_tokens": tokens,
                            "groups": len(records),
                            "skipped_groups": sum(
                                g["skip_reason"] is not None for g in records
                            ),
                            "mean_reward": (
                                sum(mean_rewards) / len(mean_rewards)
                                if mean_rewards
                                else None
                            ),
                            "policy_id": policy_id,
                        }
                        if normalized:
                            guard = (
                                {"expected_policy_id": policy_id}
                                if policy_id is not None
                                else {}
                            )
                            losses = []
                            backward_metrics = []
                            # Never pipeline an optimizer operation before backward succeeds.
                            for start in range(
                                0, len(normalized), config.microbatch_size
                            ):
                                fb = model.forward_backward(
                                    normalized[start : start + config.microbatch_size],
                                    loss_fn=config.loss_fn,
                                    zero_out=(start == 0),
                                    **config.loss_kwargs(),
                                    **guard,
                                )
                                loss = _extract(fb, "loss")
                                loss_mean = _extract(fb, "loss_mean")
                                if loss_mean is not None and not math.isfinite(
                                    float(loss_mean)
                                ):
                                    raise ValueError(
                                        "Nonfinite RL loss; optimizer step withheld"
                                    )
                                if loss is not None:
                                    if not math.isfinite(float(loss)):
                                        raise ValueError(
                                            "Nonfinite RL loss; optimizer step withheld"
                                        )
                                    losses.append(float(loss))
                                backward_metrics.append(
                                    {"loss_sum": loss, "loss_mean": loss_mean}
                                )
                            opt = model.optim_step(
                                lr=spec.learning_rate,
                                grad_clip_norm=config.grad_clip_norm,
                                **guard,
                            )
                            state["steps"] += 1
                            state["loss"] = (
                                sum(losses)
                                if len(losses) == len(backward_metrics)
                                else None
                            )
                            entry["backward_metrics"] = backward_metrics
                            entry["committed_policy_id"] = getattr(
                                getattr(opt, "policy_version", None), "id", None
                            )
                            after = mode.greedy_eval(model)
                            if after:
                                entry.update(
                                    passed=after["passed"], total=after["total"]
                                )
                            previous_eval = state.get("best_eval")
                            if (
                                after is None
                                or previous_eval is None
                                or after["passed"] / after["total"]
                                > previous_eval["passed"] / previous_eval["total"]
                            ):
                                state["best_checkpoint"] = _as_uri(
                                    model.save_weights(
                                        f"rl-best-{rnd}", mode="inference"
                                    )
                                )
                                state["best_round"], state["best_eval"] = rnd, after
                                state["best_steps"] = state["steps"]
                                state["best_loss"] = state["loss"]
                            state["training_checkpoint"] = _as_uri(
                                model.save_weights(f"rl-state-{rnd}", mode="training")
                            )
                        else:
                            entry["skipped"] = True
                            job.logs.append(
                                f"round {rnd}: zero-variance or excluded groups — nothing to train on"
                            )
                        # A checkpoint and its matching progress become visible together.
                        state["completed_round"] = rnd
                        state["rounds"].append(entry)
                        commit_state(state)
                        publish("running")
                        job.logs.append(
                            f"round {rnd}: {len(normalized)} datums, {tokens} contributing tokens; best round {state['best_round']}"
                        )
                break
            except executor._transient_exceptions(client) as exc:
                if attempt + 1 >= executor.MAX_TRANSIENT_ATTEMPTS:
                    raise
                job.logs.append(
                    f"transient River RL failure ({type(exc).__name__}); restoring committed optimizer state in a fresh session"
                )
                executor._sleep(executor.TRANSIENT_BACKOFF_S * 2**attempt)
        publish("succeeded")
        job.logs.append(f"selected River checkpoint: {job.checkpoint_uri}")
        job.status = JobStatus.SUCCEEDED
    except (KeyboardInterrupt, CancelledError) as exc:
        # A provider operation may have completed without returning its reply.
        # Publish only the last local commit, never the mutated in-memory state.
        job.status = JobStatus.CANCELLED
        job.logs.append(
            "River RL interrupted; retaining committed progress for --resume"
        )
        try:
            state = load_state()
        except Exception as recovery_error:
            state = {}
            job.logs.append(
                "Committed RL state could not be restored after interruption "
                f"({type(recovery_error).__name__}); inspect the run before resuming"
            )
        try:
            publish("cancelled", f"Interrupted ({type(exc).__name__})")
        except Exception as publication_error:
            # Disk/serialization failures must not turn Ctrl-C or task
            # cancellation into an unrelated error, or overwrite a commit.
            job.logs.append(
                "Cancelled RL report could not be published "
                f"({type(publication_error).__name__}); durable status may be stale"
            )
        raise
    except Exception as exc:
        error = str(exc)
        try:
            state = load_state()
        except (OSError, ValueError) as recovery_error:
            state = {}
            error += f"; committed state could not be restored: {recovery_error}"
        publish("failed", error)
        job.status = JobStatus.FAILED
        account = executor._account_error(exc)
        raise account or RemoteExecutionError.wrap(
            exc, mode.failure_label, provider="river"
        ) from exc
    finally:
        job.duration_s = time.monotonic() - started
        try:
            executor._record_cost(handle.job_id, job)
        except Exception as ledger_error:
            if job.status is not JobStatus.CANCELLED:
                raise
            job.logs.append(
                "Cancelled RL cost record could not be written "
                f"({type(ledger_error).__name__})"
            )
    return handle

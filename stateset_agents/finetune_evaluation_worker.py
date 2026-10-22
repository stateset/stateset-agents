"""Queued, sequential policy and judge inference for retained studio evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash, evaluation_report
from stateset_agents.finetune_evaluation import (
    MAX_TRANSCRIPT_BYTES,
    EvaluationReview,
    evaluation_notes,
    judge_success,
    read_evaluation,
    report_settings,
    report_training_seed,
    summarize_evaluation,
)
from stateset_agents.finetune_workspace import (
    process_token,
    queue_executable,
    read_json,
    write_json,
)


def _state(path: Path, **changes: Any) -> None:
    old = read_json(path / "state.json")
    write_json(path / "state.json", {**old, **changes, "updated_at": time.time()})


def _check_stop(path: Path) -> None:
    if (path / "stop.json").exists():
        record = read_json(path / "stop.json")
        if record.get("plan_sha256") != content_hash(read_json(path / "plan.json")):
            raise ValueError("Stop receipt differs from the evaluation plan")
        raise InterruptedError("Cooperative evaluation stop requested")


def _authorized(path: Path) -> EvaluationReview:
    review = read_evaluation(path)
    if read_json(path / "launch.lock").get("plan_sha256") != review.sha256:
        raise ValueError("Evaluation inference requires an exact launch receipt")
    return review


def supervise(path: Path) -> int:
    """Hold durable ownership while the shared queue bounds the whole job."""
    os.umask(0o077)
    try:
        _check_stop(path)
        review = _authorized(path)
        queue = queue_executable()
        if not queue:
            raise ValueError("Shared queue is required")
        config = review.plan["config"]
        command = [
            queue,
            "run",
            "--events-json",
            "--lane",
            "heavy",
            "--jobs",
            str(config["cpu_slots"]),
            "--memory-gb",
            str(config["memory_gb"]),
            "--scratch-gb",
            str(config["scratch_gb"]),
            "--timeout",
            str(config["timeout_seconds"]),
            "--resource-key",
            "stateset-evaluation:" + str(review.run),
            "--",
            sys.executable,
            "-u",
            "-m",
            "stateset_agents.finetune_evaluation_worker",
            "--execute",
            str(path),
        ]
        write_json(
            path / "supervisor.json",
            {"pid": os.getpid(), "process_token": process_token(os.getpid())},
        )
        write_json(path / "command.json", {"argv": command})
        _state(path, status="queued")
        environment = dict(
            os.environ,
            OMP_NUM_THREADS=str(config["cpu_slots"]),
            MKL_NUM_THREADS=str(config["cpu_slots"]),
            OPENBLAS_NUM_THREADS=str(config["cpu_slots"]),
            TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        # Keep lifecycle evidence separate from model output. The UI reads only
        # a bounded, allowlisted projection; no queue/database call on refresh.
        fd = os.open(
            path / "queue-events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(fd, "wb") as events:
            code = subprocess.call(command, env=environment, stderr=events)
        state = read_json(path / "state.json")
        if code != 0 or state.get("status") != "succeeded":
            if state.get("status") not in {"failed", "cancelled"}:
                _state(
                    path,
                    status="failed",
                    error="Queue/worker exited without a complete evaluation; inspect run.log",
                    exit_code=code,
                )
            return code or 1
        return 0
    except Exception as error:
        _state(
            path,
            status="cancelled" if isinstance(error, InterruptedError) else "failed",
            error="Evaluation supervisor could not finish; inspect retained records",
            error_type=type(error).__name__,
        )
        return 1


def execute(path: Path) -> int:
    """Run model phases in separate processes so policy/judge memory never overlaps."""
    os.umask(0o077)
    try:
        _check_stop(path)
        _authorized(path)
        try:
            fd = os.open(
                path / "execution.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            return 1  # Keep the original executor's state and results intact.
        os.close(fd)
        write_json(
            path / "execution.json",
            {"pid": os.getpid(), "process_token": process_token(os.getpid())},
        )
        for phase in ("baseline", "candidate", "judge"):
            _check_stop(path)
            read_evaluation(path)
            _state(path, status="running", phase=phase, completed_cases=0)
            code = subprocess.call(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "stateset_agents.finetune_evaluation_worker",
                    "--phase",
                    phase,
                    str(path),
                ]
            )
            if code:
                _check_stop(path)
                raise RuntimeError("Evaluation phase failed")
        _check_stop(path)
        read_evaluation(path)
        receipt = read_json(path / "result.json", max_bytes=4 * 1024 * 1024)
        if receipt.get("evaluation_plan_sha256") != content_hash(
            read_json(path / "plan.json")
        ):
            raise ValueError("Evaluation result does not match reviewed plan")
        _state(
            path,
            status="succeeded",
            phase="complete",
            finished_at=time.time(),
            quality_gain_verified=False,
        )
        return 0
    except Exception as error:
        _state(
            path,
            status="cancelled" if isinstance(error, InterruptedError) else "failed",
            error="Evaluation did not complete; no quality evidence was accepted",
            error_type=type(error).__name__,
            finished_at=time.time(),
        )
        return 1


class LocalGenerator:
    """Greedy cache-only HF generation, optionally applying only the retained adapter."""

    def __init__(
        self, model: str, revision: str, *, adapter: Path | None = None
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Resolve the commit through the Hub cache API first. Passing a repo ID
        # directly to transformers can silently prefer a matching relative
        # directory, in which case its revision argument is ignored.
        snapshot = Path(
            snapshot_download(
                repo_id=model,
                revision=revision,
                local_files_only=True,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
        )
        if snapshot.name != revision or snapshot.parent.name != "snapshots":
            raise ValueError("Cached model snapshot differs from the pinned commit")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            revision=revision,
            trust_remote_code=False,
            local_files_only=True,
        )
        # Either a transformers model or a PEFT wrapper around one. The stubs
        # share no typed generate/to protocol (PreTrainedModel.to is a
        # functools.wraps object that typeshed does not bind), so stay untyped.
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            revision=revision,
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter),
                is_trainable=False,
                local_files_only=True,
                use_safetensors=True,
            )
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self, messages: list[dict[str, str]], *, tokens: int, context: int
    ) -> tuple[str, int, bool]:
        """Reject overlong prompts rather than silently drop facts or earlier turns."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        length = encoded["input_ids"].shape[1]
        model_context = getattr(self.model.config, "max_position_embeddings", context)
        if length + tokens > min(context, model_context):
            raise ValueError(
                "Prompt and response budget exceed the reviewed/model context"
            )
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, length:]
        count = int(generated.shape[0])
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = bool(count and int(generated[-1]) in eos_ids)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not isinstance(text, str):
            raise TypeError("Tokenizer decoded one sequence into a batch")
        if not text.strip():
            raise ValueError("Model generated an empty completion")
        return text, count, count >= tokens and not ended


def policy_phase(path: Path, review: EvaluationReview, arm: str) -> None:
    """Generate all scripted turns from each policy with identical greedy settings."""
    from transformers import set_seed

    binding, config = review.plan["binding"], review.plan["config"]
    set_seed(config["seed"])
    generator = LocalGenerator(
        binding["model"],
        binding["model_revision"],
        adapter=Path(binding["artifact_path"]) if arm == "candidate" else None,
    )
    rows = []
    for case in review.cases["cases"]:
        _check_stop(path)
        messages = [
            {
                "role": "system",
                "content": case["context"]
                + "\nPolicy facts:\n"
                + "\n".join(case["policy_facts"]),
            }
        ]
        transcript = []
        tokens, truncated = 0, False
        started = time.monotonic()
        for query in case["user_turns"]:
            _check_stop(path)
            messages.append({"role": "user", "content": query})
            reply, count, limited = generator.generate(
                messages,
                tokens=config["max_new_tokens"],
                context=config["context_length"],
            )
            messages.append({"role": "assistant", "content": reply})
            transcript.extend([("user", query), ("assistant", reply)])
            tokens += count
            truncated |= limited
        rows.append(
            {
                "case_id": case["id"],
                "transcript": transcript,
                "generated_tokens": tokens,
                "truncated": "token_limit" if truncated else None,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        _state(path, completed_cases=len(rows))
    read_evaluation(path)
    document = {"evaluation_plan_sha256": review.sha256, "rows": rows}
    if len(json.dumps(document).encode()) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("Generated transcripts exceed the retained byte limit")
    write_json(path / (arm + ".json"), document)


def judge_phase(path: Path, review: EvaluationReview) -> None:
    """Record and replay two-pass judge results, then build paired test reports."""
    from stateset_agents.evaluation.judge_evidence import (
        RecordedSupportJudge,
        replay_judge_evidence,
    )
    from stateset_agents.evaluation.support_judge import JUDGE_SYSTEM_PROMPT, JudgeItem

    config, binding = review.plan["config"], review.plan["binding"]
    generator = LocalGenerator(config["judge_model"], config["judge_revision"])

    def generate(prompts: Any) -> list[str]:
        replies = []
        for prompt in prompts:
            _check_stop(path)
            text, _, _ = generator.generate(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokens=config["judge_max_new_tokens"],
                context=config["context_length"],
            )
            replies.append(text)
        return replies

    judge = RecordedSupportJudge(
        generate,
        path / "judge-recordings",
        judge_model=config["judge_model"],
        judge_revision=config["judge_revision"],
        max_parse_retries=1,
        max_batches=2,
    )
    cases = {case["id"]: case for case in review.cases["cases"]}
    reports: dict[str, dict[str, Any]] = {}
    recordings = {}
    all_scores = {}
    for arm in ("baseline", "candidate"):
        source = read_json(path / (arm + ".json"), max_bytes=MAX_TRANSCRIPT_BYTES)
        if source["evaluation_plan_sha256"] != review.sha256 or [
            row["case_id"] for row in source["rows"]
        ] != list(cases):
            raise ValueError("Generated transcripts differ from the reviewed cases")
        items = [
            JudgeItem(
                id=row["case_id"],
                scenario=cases[row["case_id"]],
                transcript=tuple(tuple(turn) for turn in row["transcript"]),
            )
            for row in source["rows"]
        ]
        results = judge.score(
            items,
            context={
                "split": config.get("split", "test"),
                "arm": arm,
                "evaluation_plan_sha256": review.sha256,
            },
        )
        evidence = judge.last_evidence
        if evidence is None:
            raise ValueError("Judge did not retain replayable evidence")
        replay = replay_judge_evidence(
            Path(evidence["path"]), expected_manifest_sha256=evidence["manifest_sha256"]
        )
        if replay["results"] != [
            json.loads(json.dumps(asdict(result))) for result in results
        ]:
            raise ValueError("Judge replay differs from fresh results")
        recordings[arm] = evidence
        all_scores[arm] = [result.score for result in results]
        outcomes = []
        for row, result in zip(source["rows"], results, strict=True):
            if (
                result.parse_failures
                or result.disagreement > review.plan["thresholds"]["max_disagreement"]
            ):
                raise ValueError("Judge parsing or self-disagreement gate failed")
            violations = int(
                any(
                    not all(checks[key] for key in checks if key.startswith("policy_"))
                    for checks in result.checks
                )
            )
            outcomes.append(
                {
                    "case_id": row["case_id"],
                    "success": judge_success(
                        result.score, result.checks, row["truncated"]
                    ),
                    "reward": result.score,
                    "policy_violations": violations,
                    "truncated": row["truncated"],
                    "generated_tokens": row["generated_tokens"],
                    "tool_calls": 0,
                    "elapsed_seconds": row["elapsed_seconds"],
                }
            )
        reports[arm] = evaluation_report(
            environment="scripted_support_conversation",
            base_model=binding["model"],
            seed=report_training_seed(review),
            cases=cases,
            settings=report_settings(review),
            checkpoint=(
                {
                    "path": binding["artifact_path"],
                    "inventory_sha256": binding["artifact_sha256"],
                }
                if arm == "candidate"
                else {"path": binding["model"] + "@" + binding["model_revision"]}
            ),
            outcomes=outcomes,
            split=config.get("split", "test"),
        )
        _state(path, completed_cases=len(cases) * len(reports))
    read_evaluation(path)
    evidence = {
        "schema_version": 1,
        "review": review.cases["provenance"],
        "baselines": [reports["baseline"]],
        "candidates": [reports["candidate"]],
    }
    write_json(path / "paired-reports.json", evidence)
    # Retain a separately importable diagnostic. Do not replace quality.json:
    # an existing multi-seed study must survive a later single-run probe.
    summary = summarize_evaluation(review, evidence)
    write_json(
        path / "result.json",
        {
            "schema_version": 1,
            "evaluation_plan_sha256": review.sha256,
            "summary": summary,
            "judge_recordings": recordings,
            "mean_judge_scores": {
                arm: sum(scores) / len(scores) for arm, scores in all_scores.items()
            },
            "quality_gain_verified": False,
            "notes": evaluation_notes(config.get("split", "test")),
        },
    )


def main() -> None:
    """Dispatch only explicitly retained supervisor, execution or inference stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--supervise", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--phase", choices=("baseline", "candidate", "judge"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.supervise:
        raise SystemExit(supervise(args.path))
    if args.execute:
        raise SystemExit(execute(args.path))
    try:
        _check_stop(args.path)
        review = _authorized(args.path)
        if args.phase == "judge":
            judge_phase(args.path, review)
        else:
            policy_phase(args.path, review, args.phase)
    except Exception as error:
        # Model/provider exceptions can contain customer data or credentials.
        _state(args.path, phase_error=type(error).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

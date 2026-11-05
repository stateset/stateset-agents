"""Retain and replay support-judge requests without calling a model again."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.support_judge import (
    RUBRIC_PATH,
    GenerateFn,
    JudgeItem,
    JudgeResult,
    SupportJudge,
    rubric_revision,
)


def _implementation() -> dict[str, str]:
    from stateset_agents.evaluation import support_judge

    return {
        "support_judge.py": hashlib.sha256(
            Path(support_judge.__file__).read_bytes()
        ).hexdigest(),
        "judge_evidence.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rubric": hashlib.sha256(
            RUBRIC_PATH.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    }


def _write(path: Path, value: Any) -> None:
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _inventory(directory: Path) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Judge evidence must be a real directory")
    files = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Judge evidence must contain regular files only")
        if path.name != "manifest.json":
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


class RecordedSupportJudge:
    """Wrap a synchronous generator and keep every prompt, reply and retry.

    The fresh output directory is private to this recording session. Each score
    call writes an exclusive batch directory, including unsuccessful calls.
    Transport error messages are intentionally omitted: they can contain keys.
    The caller still owns transport timeouts, GPU allocation and dollar budgets.
    Manifest hashes prove integrity relative to a trusted receipt, not that a
    remote endpoint actually ran the declared model.
    """

    def __init__(
        self,
        generate: GenerateFn,
        output_dir: Path,
        *,
        judge_model: str,
        judge_revision: str,
        max_parse_retries: int = 1,
        max_batches: int = 12,
    ) -> None:
        import re

        if (
            not judge_model.strip()
            or not re.fullmatch(r"[0-9a-f]{40}", judge_revision)
            or judge_revision == "0" * 40
        ):
            raise ValueError("Pin the judge model to a nonzero full commit")
        if type(max_parse_retries) is not int or max_parse_retries < 0:
            raise ValueError("max_parse_retries must be a nonnegative integer")
        if type(max_batches) is not int or max_batches < 1:
            raise ValueError("max_batches must be a positive integer")
        if output_dir.is_symlink():
            raise ValueError("Judge recording directory cannot be a symbolic link")
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.output_dir = output_dir.resolve()
        self.generate = generate
        self.judge = {
            "model": judge_model,
            "revision": judge_revision,
            "rubric_revision": rubric_revision(),
        }
        self.max_parse_retries = max_parse_retries
        self.max_batches = max_batches
        self._batches = 0
        self._implementation = _implementation()
        self._last_evidence: dict[str, Any] | None = None

    @property
    def last_evidence(self) -> dict[str, Any] | None:
        """Return a detached reference to the last successful score recording."""
        return copy.deepcopy(self._last_evidence)

    def score(
        self, items: Sequence[JudgeItem], *, context: Mapping[str, Any] | None = None
    ) -> list[JudgeResult]:
        """Persist inputs before inference and seal either results or failure."""
        self._last_evidence = None
        if _implementation() != self._implementation:
            raise ValueError("Judge implementation changed during recording")
        if self._batches >= self.max_batches:
            raise RuntimeError("Judge recording batch budget exhausted")
        if (
            not items
            or any(
                not isinstance(item.id, str) or not item.id.strip() for item in items
            )
            or len({item.id for item in items}) != len(items)
        ):
            raise ValueError("Judge recordings require nonempty, unique case IDs")
        snapshot = {
            "items": [asdict(item) for item in items],
            "context": dict(context or {}),
        }
        content_hash(snapshot)
        directory = self.output_dir / f"batch-{self._batches:06d}"
        self._batches += 1
        directory.mkdir(mode=0o700)
        _write(directory / "inputs.json", snapshot)
        generation = 0

        def generate(prompts: Sequence[str]) -> Sequence[str]:
            nonlocal generation
            index = generation
            generation += 1
            _write(directory / f"request-{index:03d}.json", {"prompts": list(prompts)})
            replies = list(self.generate(prompts))
            if any(not isinstance(reply, str) for reply in replies):
                raise ValueError("Judge generator must return text replies")
            _write(directory / f"response-{index:03d}.json", {"replies": replies})
            return replies

        status = "failed"
        try:
            results = SupportJudge(
                generate, max_parse_retries=self.max_parse_retries
            ).score(items)
            _write(directory / "results.json", [asdict(result) for result in results])
            status = "completed"
        except BaseException as error:
            _write(directory / "failure.json", {"error_type": type(error).__name__})
            raise
        finally:
            manifest = {
                "schema_version": 1,
                "status": status,
                "judge": self.judge,
                "implementation": self._implementation,
                "max_parse_retries": self.max_parse_retries,
                "generation_calls": generation,
                "files": _inventory(directory),
            }
            _write(directory / "manifest.json", manifest)
        self._last_evidence = {
            "path": str(directory),
            "manifest_sha256": content_hash(manifest),
        }
        return results


def replay_judge_evidence(
    directory: Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Verify recorded files and reproduce all parsing, findings and scores.

    Replay uses recorded replies and invokes no model, network or provider.
    Pin ``expected_manifest_sha256`` from a trusted selection receipt. Failed
    transport recordings remain inspectable but cannot become accepted scores.
    """
    inventory = _inventory(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Judge manifest cannot be a symbolic link")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or content_hash(manifest) != expected_manifest_sha256
        or manifest.get("status") != "completed"
        or manifest.get("implementation") != _implementation()
        or manifest.get("files") != inventory
        or manifest.get("judge", {}).get("rubric_revision") != rubric_revision()
    ):
        raise ValueError("Judge evidence integrity or implementation differs")
    snapshot = json.loads((directory / "inputs.json").read_text())
    items = [
        JudgeItem(
            id=row["id"],
            scenario=row["scenario"],
            transcript=tuple(tuple(turn) for turn in row["transcript"]),
        )
        for row in snapshot["items"]
    ]
    generation = 0

    def generate(prompts: Sequence[str]) -> Sequence[str]:
        nonlocal generation
        request = json.loads((directory / f"request-{generation:03d}.json").read_text())
        response = json.loads(
            (directory / f"response-{generation:03d}.json").read_text()
        )
        generation += 1
        if request != {"prompts": list(prompts)}:
            raise ValueError("Recorded judge prompts differ from replay")
        replies = response["replies"]
        if not isinstance(replies, list) or any(
            not isinstance(reply, str) for reply in replies
        ):
            raise ValueError("Recorded judge replies must be a text list")
        return replies

    results = SupportJudge(
        generate, max_parse_retries=manifest["max_parse_retries"]
    ).score(items)
    recorded = json.loads((directory / "results.json").read_text())
    if (
        content_hash([asdict(result) for result in results]) != content_hash(recorded)
        or generation != manifest["generation_calls"]
    ):
        raise ValueError("Recorded judge results differ from replay")
    return {
        "judge": manifest["judge"],
        "context": snapshot["context"],
        "items": snapshot["items"],
        "results": recorded,
    }

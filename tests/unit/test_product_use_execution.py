"""The public execution score must follow observed StateSet tool behavior."""

import asyncio
import inspect
import json
from pathlib import Path

import pytest

from benchmarks.product_use.execution import DISCOVERY_TOOLS, TOOLS, run, score_task
from benchmarks.product_use.export_examples import export_examples
from benchmarks.product_use.generate import generate

TASKS = Path("benchmarks/product_use/tasks.v0.2.public.json")
DEMONSTRATIONS = Path("benchmarks/product_use/demonstrations.v0.2.json")
TOOL_CATALOG = Path("benchmarks/product_use/tools.v0.2.json")
TASKS_V03 = Path("benchmarks/product_use/tasks.v0.3.public.json")
DEMONSTRATIONS_V03 = Path("benchmarks/product_use/demonstrations.v0.3.json")
TOOL_CATALOG_V03 = Path("benchmarks/product_use/tools.v0.3.json")
TOOL_CATALOG_V04 = Path("benchmarks/product_use/tools.v0.4.json")
TOOL_CATALOG_V05 = Path("benchmarks/product_use/tools.v0.5.json")


@pytest.mark.parametrize(
    ("catalog_path", "expected_names"),
    [
        (TOOL_CATALOG, set(TOOLS) - DISCOVERY_TOOLS),
        (TOOL_CATALOG_V03, set(TOOLS)),
        (TOOL_CATALOG_V04, set(TOOLS)),
        (TOOL_CATALOG_V05, set(TOOLS)),
    ],
)
def test_public_tool_catalog_matches_mcp_functions(
    catalog_path: Path, expected_names: set[str]
) -> None:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in catalog["tools"]}
    assert set(entries) == expected_names
    for name in expected_names:
        function = TOOLS[name]
        signature = inspect.signature(function)
        parameters = entries[name]["parameters"]
        assert set(parameters["properties"]) == set(signature.parameters)
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == {
            key
            for key, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        }


def test_verified_corpus_matches_real_tool_demonstrations() -> None:
    for tasks, demonstrations, tools, expected_count, version in (
        (TASKS, DEMONSTRATIONS, TOOL_CATALOG, 5, "0.2"),
        (TASKS_V03, DEMONSTRATIONS_V03, TOOL_CATALOG_V03, 7, "0.3"),
    ):
        rows = export_examples(tasks, demonstrations, tools)
        committed = Path(f"benchmarks/product_use/examples.v{version}.jsonl").read_text(
            encoding="utf-8"
        )
        assert len(rows) == expected_count
        assert committed == "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        )


def test_discovery_task_requires_discovery_call() -> None:
    discovery_tasks = json.loads(TASKS_V03.read_text(encoding="utf-8"))[-2:]
    old_demonstrations = json.loads(DEMONSTRATIONS.read_text(encoding="utf-8"))
    for task, demonstration in zip(
        discovery_tasks, (old_demonstrations[1], old_demonstrations[4]), strict=True
    ):
        result = asyncio.run(score_task(task, demonstration))
        assert result["score"] == 0


def test_generated_holdout_replays_and_hides_evaluator_fields(tmp_path: Path) -> None:
    tasks, prompts, oracle = generate("private-test-seed", variants=1)
    assert len(tasks) == len(prompts) == len(oracle) == 5
    assert [task["id"] for task in tasks] == [prompt["id"] for prompt in prompts]
    assert all(
        {"params", "fixture", "goal", "discovery"}.isdisjoint(prompt)
        for prompt in prompts
    )
    assert [task["id"] for task in tasks] != [
        task["id"] for task in generate("another-seed")[0]
    ]

    tasks_path = tmp_path / "tasks.json"
    oracle_path = tmp_path / "oracle.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    report = run(tasks_path, oracle_path)
    assert report["schema_version"] == "0.4"
    assert report["score"] == 1.0

    wrong = {
        "calls": [
            {
                "name": "ingest_transcripts",
                "arguments": {
                    **oracle[0]["calls"][0]["arguments"],
                    "output_dir": "wrong-directory",
                },
            }
        ]
    }
    assert asyncio.run(score_task(tasks[0], wrong))["score"] == 0
    assert (
        asyncio.run(score_task(tasks[3], {"calls": oracle[3]["calls"][1:]}))["score"]
        == 0
    )


def test_generated_v05_varies_private_data_and_checks_chained_workflow(
    tmp_path: Path,
) -> None:
    tasks, prompts, oracle = generate(
        "private-test-seed", variants=2, schema_version="0.5"
    )
    assert len(tasks) == len(prompts) == len(oracle) == 12
    assert all("fixture_data" not in prompt for prompt in prompts)
    assert (
        tasks[0]["fixture_data"]
        != generate("another-seed", schema_version="0.5")[0][0]["fixture_data"]
    )
    assert tasks[0]["fixture_data"] != tasks[6]["fixture_data"]
    public_token = tasks[0]["id"].rsplit("-", 1)[1]
    public_order = 1000 + int(public_token[4:8], 16) % 9000
    assert str(public_order) not in tasks[0]["fixture_data"][0][0]["content"]

    tasks_path = tmp_path / "tasks.json"
    oracle_path = tmp_path / "oracle.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
    report = run(tasks_path, oracle_path)
    assert report["schema_version"] == "0.5"
    assert report["score"] == 1.0

    chained_task = tasks[5]
    chained_calls = oracle[5]["calls"]
    assert [call["name"] for call in chained_calls] == [
        "ingest_transcripts",
        "grade_transcript",
    ]
    assert (
        asyncio.run(score_task(chained_task, {"calls": chained_calls[1:]}))["score"]
        == 0
    )
    wrong_history = [
        chained_calls[0],
        {
            "name": "grade_transcript",
            "arguments": {
                **chained_calls[1]["arguments"],
                "history_path": f"{chained_task['params']['output_dir']}/conversation_1.jsonl",
            },
        },
    ]
    assert asyncio.run(score_task(chained_task, {"calls": wrong_history}))["score"] == 0


def test_forged_artifact_does_not_pass() -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[0]
    result = asyncio.run(
        score_task(
            task,
            {
                "task_id": task["id"],
                "calls": [],
                "artifacts": {"output_dir": "transcripts"},
            },
        )
    )
    assert result["score"] == 0


def test_ingest_rejects_corrupt_output_with_plausible_metadata(monkeypatch) -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[0]

    def fake_ingest(input_path: str, format: str, output_dir: str) -> dict:
        destination = Path(output_dir)
        destination.mkdir()
        files = [destination / f"conversation_{index}.jsonl" for index in range(2)]
        for path in files:
            path.write_text(
                '{"role":"assistant","content":"wrong"}\n', encoding="utf-8"
            )
        return {
            "conversation_count": 2,
            "turn_count": 4,
            "files": [str(path) for path in files],
        }

    monkeypatch.setitem(TOOLS, "ingest_transcripts", fake_ingest)
    demonstration = json.loads(DEMONSTRATIONS.read_text(encoding="utf-8"))[0]
    result = asyncio.run(score_task(task, demonstration))
    assert result["score"] == 0


def test_curation_rejects_corrupt_artifact_with_plausible_count(monkeypatch) -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[2]

    def fake_improve(
        transcripts_dir: str,
        reward: str,
        output_dir: str,
        threshold: float = 0.7,
        format: str = "transcripts",
    ) -> dict:
        destination = Path(output_dir)
        destination.mkdir()
        (destination / "curated.jsonl").write_text("not JSON\n", encoding="utf-8")
        (destination / "improve_summary.json").write_text("{}", encoding="utf-8")
        return {"curated_count": 1}

    monkeypatch.setitem(TOOLS, "improve_run", fake_improve)
    demonstration = json.loads(DEMONSTRATIONS.read_text(encoding="utf-8"))[2]
    result = asyncio.run(score_task(task, demonstration))
    assert result["score"] == 0


def test_wrong_transcript_does_not_pass() -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[1]
    result = asyncio.run(
        score_task(
            task,
            {
                "task_id": task["id"],
                "calls": [
                    {
                        "name": "grade_transcript",
                        "arguments": {
                            "history_path": "transcripts/bad.jsonl",
                            "reward": "customer_support",
                        },
                    }
                ],
            },
        )
    )
    assert result["score"] == 0


def test_path_traversal_is_rejected() -> None:
    task = json.loads(TASKS.read_text(encoding="utf-8"))[0]
    result = asyncio.run(
        score_task(
            task,
            {
                "task_id": task["id"],
                "calls": [
                    {
                        "name": "ingest_transcripts",
                        "arguments": {
                            "input_path": "logs.jsonl",
                            "format": "openai",
                            "output_dir": "../escape",
                        },
                    }
                ],
            },
        )
    )
    assert result["score"] == 0
    assert "escapes" in result["error"]


def test_missing_tasks_reduce_aggregate(tmp_path: Path) -> None:
    submissions = tmp_path / "one.json"
    submissions.write_text(
        json.dumps(json.loads(DEMONSTRATIONS.read_text(encoding="utf-8"))[:1]),
        encoding="utf-8",
    )
    report = run(TASKS, submissions)
    assert report["score"] == 0.2
    assert sum(row["score"] for row in report["results"]) == 1

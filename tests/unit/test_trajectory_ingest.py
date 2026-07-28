"""Unit tests for stateset_agents.data.trajectory_ingest.

Covers round-trips for OpenAI chat-completions and LangChain message
formats, tool-call preservation, multimodal-content tolerance, reward
passthrough, to_grading_history compatibility with grade_transcript.py's
loader, and the `stateset-agents ingest` CLI subcommand.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from stateset_agents.core.trajectory import MultiTurnTrajectory
from stateset_agents.data import trajectory_ingest as ti
from stateset_agents.data.trajectory_ingest import (
    from_langchain_json,
    from_openai_jsonl,
    from_openai_messages,
    to_grading_history,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import grade_transcript  # noqa: E402

# ---------------------------------------------------------------------------
# from_openai_messages
# ---------------------------------------------------------------------------


class TestFromOpenaiMessages:
    def test_basic_round_trip(self) -> None:
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!"},
        ]
        traj = from_openai_messages(messages)
        assert isinstance(traj, MultiTurnTrajectory)
        assert len(traj.turns) == 3
        assert [t.role for t in traj.turns] == ["system", "user", "assistant"]
        assert [t.content for t in traj.turns] == ["Be helpful.", "Hi there", "Hello!"]

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValueError):
            from_openai_messages([])

    def test_missing_role_raises(self) -> None:
        with pytest.raises(ValueError):
            from_openai_messages([{"content": "no role"}])

    def test_metadata_merged(self) -> None:
        traj = from_openai_messages(
            [{"role": "user", "content": "hi"}], metadata={"source": "prod-bot"}
        )
        assert traj.metadata["source"] == "prod-bot"

    def test_tool_calls_preserved(self) -> None:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ]
        messages = [
            {"role": "user", "content": "weather in NYC?"},
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {
                "role": "tool",
                "content": "72F sunny",
                "tool_call_id": "call_1",
            },
        ]
        traj = from_openai_messages(messages)
        assistant_turn = traj.turns[1]
        assert assistant_turn.tool_calls == tool_calls
        tool_turn = traj.turns[2]
        assert tool_turn.metadata["tool_call_id"] == "call_1"
        assert tool_turn.content == "72F sunny"

    def test_multimodal_content_concatenated_and_skipped_recorded(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/x.png"},
                    },
                    {"type": "text", "text": "What is it?"},
                ],
            }
        ]
        traj = from_openai_messages(messages)
        turn = traj.turns[0]
        assert turn.content == "Look at this:\nWhat is it?"
        assert len(turn.metadata["skipped_parts"]) == 1
        assert turn.metadata["skipped_parts"][0]["type"] == "image_url"

    def test_reward_passthrough_attaches_to_last_turn(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        traj = from_openai_messages(messages, reward=0.75)
        assert traj.total_reward == 0.75
        assert traj.turn_rewards == [0.0, 0.75]

    def test_no_reward_leaves_unset(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        traj = from_openai_messages(messages)
        assert traj.total_reward == 0.0


# ---------------------------------------------------------------------------
# from_openai_jsonl
# ---------------------------------------------------------------------------


class TestFromOpenaiJsonl:
    def test_messages_key_form(self, tmp_path: Path) -> None:
        path = tmp_path / "logs.jsonl"
        lines = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "reward": 1.0,
            },
            {
                "messages": [
                    {"role": "user", "content": "bye"},
                    {"role": "assistant", "content": "goodbye"},
                ],
            },
        ]
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        trajectories = from_openai_jsonl(path)
        assert len(trajectories) == 2
        assert trajectories[0].total_reward == 1.0
        assert trajectories[1].total_reward == 0.0

    def test_bare_list_form(self, tmp_path: Path) -> None:
        path = tmp_path / "logs.jsonl"
        conv = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(conv) + "\n")

        trajectories = from_openai_jsonl(path)
        assert len(trajectories) == 1
        assert len(trajectories[0].turns) == 2

    def test_score_key_in_nested_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "logs.jsonl"
        line = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"reward": 0.5},
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        trajectories = from_openai_jsonl(path)
        assert trajectories[0].total_reward == 0.5

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "logs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(
                json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
            )
            f.write("\n")
        trajectories = from_openai_jsonl(path)
        assert len(trajectories) == 1

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "logs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json\n")
        with pytest.raises(ValueError):
            from_openai_jsonl(path)


# ---------------------------------------------------------------------------
# from_langchain_json
# ---------------------------------------------------------------------------


class TestFromLangchainJson:
    def test_flat_type_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "lc.json"
        obj = {
            "messages": [
                {"type": "system", "data": {"content": "Be helpful."}},
                {"type": "human", "data": {"content": "Hi"}},
                {"type": "ai", "data": {"content": "Hello!"}},
            ],
            "reward": 0.9,
        }
        path.write_text(json.dumps(obj), encoding="utf-8")

        trajectories = from_langchain_json(path)
        assert len(trajectories) == 1
        traj = trajectories[0]
        assert [t.role for t in traj.turns] == ["system", "user", "assistant"]
        assert traj.total_reward == 0.9

    def test_dumpd_constructor_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "lc.json"
        obj = {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": "Hi"},
                },
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {"content": "Hello!"},
                },
            ]
        }
        path.write_text(json.dumps(obj), encoding="utf-8")

        trajectories = from_langchain_json(path)
        traj = trajectories[0]
        assert [t.role for t in traj.turns] == ["user", "assistant"]
        assert [t.content for t in traj.turns] == ["Hi", "Hello!"]

    def test_bare_list_of_flat_messages(self) -> None:
        obj = [
            {"type": "human", "data": {"content": "Hi"}},
            {"type": "ai", "data": {"content": "Hello!"}},
        ]
        trajectories = from_langchain_json(obj)
        assert len(trajectories) == 1
        assert len(trajectories[0].turns) == 2

    def test_list_of_conversations(self) -> None:
        obj = [
            {"messages": [{"type": "human", "data": {"content": "hi"}}]},
            {"messages": [{"type": "human", "data": {"content": "bye"}}]},
        ]
        trajectories = from_langchain_json(obj)
        assert len(trajectories) == 2

    def test_tool_call_preserved(self) -> None:
        obj = {
            "messages": [
                {
                    "type": "ai",
                    "data": {
                        "content": "",
                        "additional_kwargs": {
                            "tool_calls": [{"id": "1", "function": {"name": "f"}}]
                        },
                    },
                },
            ]
        }
        trajectories = from_langchain_json(obj)
        turn = trajectories[0].turns[0]
        assert turn.tool_calls == [{"id": "1", "function": {"name": "f"}}]

    def test_multimodal_content(self) -> None:
        obj = {
            "messages": [
                {
                    "type": "human",
                    "data": {
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "x"}},
                        ]
                    },
                }
            ]
        }
        trajectories = from_langchain_json(obj)
        turn = trajectories[0].turns[0]
        assert turn.content == "hello"
        assert len(turn.metadata["skipped_parts"]) == 1

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            from_langchain_json([1, 2, 3])


# ---------------------------------------------------------------------------
# to_grading_history
# ---------------------------------------------------------------------------


class TestToGradingHistory:
    def test_emits_role_content_dicts(self) -> None:
        traj = from_openai_messages(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        )
        history = to_grading_history(traj)
        assert history == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]

    def test_compatible_with_grade_transcript_loader(self, tmp_path: Path) -> None:
        traj = from_openai_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        history_path = tmp_path / "history.jsonl"
        with open(history_path, "w", encoding="utf-8") as f:
            for turn in to_grading_history(traj):
                f.write(json.dumps(turn) + "\n")

        loaded = grade_transcript.load_transcript(history_path)
        assert loaded == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]


# ---------------------------------------------------------------------------
# Lazy export from stateset_agents.data
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_exports_from_data_package(self) -> None:
        from stateset_agents.data import from_langchain_json as pkg_from_langchain_json
        from stateset_agents.data import from_openai_jsonl as pkg_from_openai_jsonl
        from stateset_agents.data import (
            from_openai_messages as pkg_from_openai_messages,
        )
        from stateset_agents.data import to_grading_history as pkg_to_grading_history

        assert pkg_from_openai_messages is ti.from_openai_messages
        assert pkg_from_openai_jsonl is ti.from_openai_jsonl
        assert pkg_from_langchain_json is ti.from_langchain_json
        assert pkg_to_grading_history is ti.to_grading_history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "stateset_agents.cli", "ingest", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )


class TestIngestCli:
    def test_openai_to_single_jsonl(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jsonl"
        with open(src, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hey"},
                        ]
                    }
                )
                + "\n"
            )
        out = tmp_path / "out.jsonl"
        result = _run_cli(
            "--format", "openai", "--input", str(src), "--output", str(out)
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        lines = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert lines == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]

    def test_langchain_to_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "lc.json"
        src.write_text(
            json.dumps(
                {
                    "messages": [
                        {"type": "human", "data": {"content": "hi"}},
                        {"type": "ai", "data": {"content": "hey"}},
                    ]
                }
            ),
            encoding="utf-8",
        )
        out_dir = tmp_path / "out"
        result = _run_cli(
            "--format", "langchain", "--input", str(src), "--output", str(out_dir)
        )
        assert result.returncode == 0, result.stderr
        conv_file = out_dir / "conversation_0.jsonl"
        assert conv_file.exists()

    def test_bad_format_errors(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jsonl"
        src.write_text("{}\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        result = _run_cli(
            "--format", "bogus", "--input", str(src), "--output", str(out)
        )
        assert result.returncode == 2

    def test_missing_input_errors(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        result = _run_cli(
            "--format",
            "openai",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(out),
        )
        assert result.returncode == 2

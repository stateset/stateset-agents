"""Unit tests for ``stateset-agents chat-remote``.

The session is faked at the ``RemoteChatSession`` seam — no pods, no ssh.
What matters here: the scripted ``--prompt`` mode drives the session in
order, a bad adapter path is rejected before renting anything, and the
session is ALWAYS closed — the pod bills until it is.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.remote import chat_session
from stateset_agents.remote.executor import RemoteExecutionError

runner = CliRunner()


class FakeSession:
    instances: list[FakeSession] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started_with: dict | None = None
        self.asked: list[str] = []
        self.close_calls = 0
        self.ask_raises: Exception | None = None
        FakeSession.instances.append(self)

    def start(self, base_model, adapter_dir=None, gpu=None):
        self.started_with = {
            "base_model": base_model,
            "adapter_dir": adapter_dir,
            "gpu": gpu,
        }

    def ask(self, prompt, timeout_s=120):
        if self.ask_raises is not None:
            raise self.ask_raises
        self.asked.append(prompt)
        return f"echo:{prompt}"

    @property
    def transcript(self):
        # Same shape as the real RemoteChatSession.transcript.
        messages = []
        for prompt in self.asked:
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": f"echo:{prompt}"})
        return {
            "messages": messages,
            "metadata": {"source": "chat-remote", "base_model": "m"},
        }

    def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def fake_session(monkeypatch, tmp_path):
    # chdir: the default transcript path is relative (./chat_transcripts/),
    # and tests must never write into the repo.
    monkeypatch.chdir(tmp_path)
    FakeSession.instances = []
    monkeypatch.setattr(chat_session, "RemoteChatSession", FakeSession)
    return FakeSession


def invoke(*args):
    return runner.invoke(app, ["chat-remote", *args])


class TestScriptedMode:
    def test_each_prompt_is_sent_and_each_reply_printed(self):
        result = invoke(
            "--base-model",
            "Qwen/Qwen3.5-0.8B",
            "--prompt",
            "hi",
            "--prompt",
            "how are you?",
        )

        assert result.exit_code == 0, result.output
        session = FakeSession.instances[0]
        assert session.asked == ["hi", "how are you?"]
        assert "echo:hi" in result.output
        assert "echo:how are you?" in result.output

    def test_session_options_reach_start(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        result = invoke(
            "--base-model",
            "Qwen/Qwen3.5-0.8B",
            "--adapter",
            str(adapter),
            "--gpu",
            "NVIDIA RTX A4000",
            "--container-disk-gb",
            "80",
            "--prompt",
            "hi",
        )

        assert result.exit_code == 0, result.output
        session = FakeSession.instances[0]
        assert session.kwargs["container_disk_gb"] == 80
        assert session.started_with == {
            "base_model": "Qwen/Qwen3.5-0.8B",
            "adapter_dir": adapter,
            "gpu": "NVIDIA RTX A4000",
        }

    def test_session_is_closed_after_a_scripted_run(self):
        invoke("--base-model", "m", "--prompt", "hi")

        assert FakeSession.instances[0].close_calls == 1


class TestInteractiveMode:
    def test_exit_word_ends_the_session(self):
        result = runner.invoke(
            app, ["chat-remote", "--base-model", "m"], input="hello\nexit\n"
        )

        assert result.exit_code == 0, result.output
        session = FakeSession.instances[0]
        assert session.asked == ["hello"]
        assert session.close_calls == 1

    def test_eof_ends_the_session(self):
        result = runner.invoke(app, ["chat-remote", "--base-model", "m"], input="")

        assert result.exit_code == 0, result.output
        assert FakeSession.instances[0].close_calls == 1

    def test_max_turns_caps_the_session(self):
        result = runner.invoke(
            app,
            ["chat-remote", "--base-model", "m", "--max-turns", "2"],
            input="one\ntwo\nthree\n",
        )

        assert result.exit_code == 0, result.output
        assert FakeSession.instances[0].asked == ["one", "two"]
        assert "max-turns" in result.output


class TestFailurePaths:
    def test_missing_adapter_dir_exits_2_before_renting_anything(self, tmp_path):
        result = invoke(
            "--base-model",
            "m",
            "--adapter",
            str(tmp_path / "absent"),
            "--prompt",
            "hi",
        )

        assert result.exit_code == 2
        assert "does not exist" in result.output
        assert FakeSession.instances == []  # no session, so no pod

    def test_ask_failure_exits_1_and_closes_the_session(self, monkeypatch):
        def failing_ask(self, prompt, timeout_s=120):
            raise RemoteExecutionError("pod exploded", provider="runpod")

        monkeypatch.setattr(FakeSession, "ask", failing_ask)

        result = invoke("--base-model", "m", "--prompt", "hi")

        assert result.exit_code == 1
        assert "pod exploded" in result.output
        assert FakeSession.instances[0].close_calls == 1


class TestTranscriptSaving:
    """Every chat is training data: saved by default, ingest-ready."""

    def test_default_run_saves_an_ingest_ready_transcript(self, tmp_path):
        result = invoke("--base-model", "m", "--prompt", "hi")

        assert result.exit_code == 0, result.output
        files = list((tmp_path / "chat_transcripts").glob("chat_*.jsonl"))
        assert len(files) == 1
        row = json.loads(files[0].read_text().strip())
        assert row["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "echo:hi"},
        ]
        assert "Transcript saved to" in result.output
        # The printed command embeds an OS-native path (backslashes on
        # Windows), so assert on the pieces rather than a POSIX separator.
        assert "ingest --format openai --input" in result.output
        assert files[0].name in result.output

    def test_save_transcript_flag_picks_the_path(self, tmp_path):
        target = tmp_path / "logs" / "session.jsonl"

        result = invoke(
            "--base-model", "m", "--prompt", "hi", "--save-transcript", str(target)
        )

        assert result.exit_code == 0, result.output
        assert target.exists()
        assert str(target) in result.output

    def test_no_save_writes_nothing(self, tmp_path):
        result = invoke("--base-model", "m", "--prompt", "hi", "--no-save")

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "chat_transcripts").exists()
        assert "Transcript saved" not in result.output

    def test_empty_conversation_writes_no_file(self, tmp_path):
        result = runner.invoke(app, ["chat-remote", "--base-model", "m"], input="")

        assert result.exit_code == 0, result.output
        assert not (tmp_path / "chat_transcripts").exists()

    def test_aborted_chat_still_persists_completed_turns(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def flaky_ask(self, prompt, timeout_s=120):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RemoteExecutionError("pod exploded", provider="runpod")
            self.asked.append(prompt)
            return f"echo:{prompt}"

        monkeypatch.setattr(FakeSession, "ask", flaky_ask)

        result = invoke("--base-model", "m", "--prompt", "one", "--prompt", "two")

        assert result.exit_code == 1
        files = list((tmp_path / "chat_transcripts").glob("chat_*.jsonl"))
        assert len(files) == 1
        row = json.loads(files[0].read_text().strip())
        assert [m["content"] for m in row["messages"]] == ["one", "echo:one"]

    def test_saved_transcript_round_trips_through_the_ingest_parser(self, tmp_path):
        """The contract: chat-remote output IS ingest --format openai input."""
        from stateset_agents.data.trajectory_ingest import from_openai_jsonl

        invoke("--base-model", "m", "--prompt", "hi", "--prompt", "bye")

        files = list((tmp_path / "chat_transcripts").glob("chat_*.jsonl"))
        trajectories = from_openai_jsonl(files[0])

        assert len(trajectories) == 1
        turns = trajectories[0].turns
        assert [(t.role, t.content) for t in turns] == [
            ("user", "hi"),
            ("assistant", "echo:hi"),
            ("user", "bye"),
            ("assistant", "echo:bye"),
        ]
        assert trajectories[0].metadata["metadata"]["source"] == "chat-remote"


class TestRegistration:
    def test_command_is_registered(self):
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }

        assert "chat-remote" in names

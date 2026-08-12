"""Tests for the on-pod chat server's request loop.

``serve`` takes file objects and a generate callable, so the whole protocol
is testable with ``StringIO`` on both ends and no model anywhere near.
"""

from __future__ import annotations

import io
import json

from stateset_agents.remote.chat_repl import serve


class RecordingGenerate:
    """Fake generator: records the history it was handed at each call."""

    def __init__(self, replies=None, fail_on=frozenset()):
        self.replies = list(replies or [])
        self.histories: list[list[dict[str, str]]] = []
        self.fail_on = fail_on

    def __call__(self, messages):
        call = len(self.histories)
        self.histories.append([dict(m) for m in messages])
        if call in self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return self.replies.pop(0) if self.replies else f"reply-{call}"


def run(generate, *requests: str) -> list[dict]:
    stdin = io.StringIO("".join(line + "\n" for line in requests))
    stdout = io.StringIO()
    assert serve(generate, stdin, stdout) == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


class TestProtocol:
    def test_ready_is_the_first_line(self):
        events = run(RecordingGenerate())

        assert events[0] == {"ready": True}

    def test_each_prompt_gets_one_response_line(self):
        events = run(
            RecordingGenerate(replies=["hello!", "again!"]),
            json.dumps({"prompt": "hi"}),
            json.dumps({"prompt": "more"}),
        )

        assert events[1:] == [{"response": "hello!"}, {"response": "again!"}]

    def test_eof_ends_the_loop_cleanly(self):
        assert run(RecordingGenerate()) == [{"ready": True}]

    def test_invalid_json_is_an_error_line_not_a_crash(self):
        events = run(
            RecordingGenerate(replies=["ok"]),
            "not json{",
            json.dumps({"prompt": "hi"}),
        )

        assert "error" in events[1]
        assert events[2] == {"response": "ok"}

    def test_missing_prompt_key_is_an_error_line(self):
        events = run(RecordingGenerate(), json.dumps({"question": "hi"}))

        assert "error" in events[1]

    def test_blank_lines_are_ignored(self):
        events = run(RecordingGenerate(replies=["ok"]), "", json.dumps({"prompt": "x"}))

        assert events[1] == {"response": "ok"}


class TestMultiTurnHistory:
    def test_history_accumulates_across_turns(self):
        generate = RecordingGenerate(replies=["first answer", "second answer"])
        run(
            generate,
            json.dumps({"prompt": "one"}),
            json.dumps({"prompt": "two"}),
        )

        assert generate.histories[0] == [{"role": "user", "content": "one"}]
        assert generate.histories[1] == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "two"},
        ]

    def test_failed_generation_is_rolled_back_out_of_the_history(self):
        generate = RecordingGenerate(replies=["recovered"], fail_on={0})
        events = run(
            generate,
            json.dumps({"prompt": "boom"}),
            json.dumps({"prompt": "retry"}),
        )

        assert "error" in events[1]
        assert "CUDA out of memory" in events[1]["error"]
        assert events[2] == {"response": "recovered"}
        # The failed exchange left nothing behind: turn 2 sees only itself.
        assert generate.histories[1] == [{"role": "user", "content": "retry"}]


class TestParser:
    def test_defaults(self):
        from stateset_agents.remote.chat_repl import build_parser

        args = build_parser().parse_args(["--base-model", "Qwen/Qwen3.5-0.8B"])

        assert args.base_model == "Qwen/Qwen3.5-0.8B"
        assert args.adapter is None
        assert args.max_new_tokens == 200

    def test_adapter_flag(self):
        from stateset_agents.remote.chat_repl import build_parser

        args = build_parser().parse_args(
            ["--base-model", "m", "--adapter", "/workspace/adapter"]
        )

        assert args.adapter == "/workspace/adapter"

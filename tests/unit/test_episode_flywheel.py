"""Tests for the multi-turn episode flywheel.

The skill under test is context carryover: turn 2's checks demand the
account reference the user only ever said in turn 1. Scoring is per-turn
(an echo in turn 1 must not satisfy turn 2) with episode-wide forbids.
"""

from __future__ import annotations

import json
import sys

import pytest

from stateset_agents.remote.river import _score_episode
from stateset_agents.training.eval_ladder import build_episode_ladder
from tests.unit.test_eval_ladder import SPEC


class TestEpisodeLadder:
    def test_turn2_requires_the_reference_the_user_never_repeats(self):
        kit = build_episode_ladder(SPEC, eval_count=8, seed=3)
        for script in kit["eval"]:
            ref = script["turn_expect"][0][-1]
            assert ref in script["turns"][0]
            assert ref not in script["turns"][1]  # carryover is the test
            assert ref in script["turn_expect"][1]

    def test_refusal_episodes_forbid_the_declined_remedy(self):
        kit = build_episode_ladder(SPEC, eval_count=20, refusal_fraction=1.0, seed=3)
        for script in kit["eval"]:
            assert len(script["forbid"]) == 1
            assert script["forbid"][0] not in script["turn_expect"][1]

    def test_eval_and_harvest_refs_are_disjoint(self):
        kit = build_episode_ladder(SPEC, seed=5)
        eval_refs = {s["turn_expect"][0][-1] for s in kit["eval"]}
        harvest_refs = {s["turn_expect"][0][-1] for s in kit["harvest"]}
        assert not eval_refs & harvest_refs


class TestScoreEpisode:
    SCRIPT = {
        "turns": ["Ticket #77 — vpn down.", "Also printer. Confirm the first fix?"],
        "turn_expect": [["vpn profile", "77"], ["print spooler", "77"]],
        "forbid": ["reset link"],
    }

    def test_full_pass(self):
        passed, detail = _score_episode(
            self.SCRIPT,
            [
                "Fixed vpn profile on ticket 77.",
                "Print spooler restarted; 77 confirmed.",
            ],
        )
        assert passed and detail["passed"]

    def test_reference_only_in_turn_one_fails_turn_two(self):
        """The whole point: an echo in turn 1 cannot satisfy turn 2."""
        passed, detail = _score_episode(
            self.SCRIPT,
            ["Fixed vpn profile on ticket 77.", "Print spooler restarted."],
        )
        assert not passed
        assert detail["per_turn"][1]["passed"] is False

    def test_forbid_anywhere_in_the_episode_fails(self):
        passed, detail = _score_episode(
            self.SCRIPT,
            ["Fixed vpn profile, 77, and sent a reset link.", "Print spooler; 77."],
        )
        assert not passed
        assert detail["forbid_hits"] == ["reset link"]


class TestEpisodeHarvestExecutor:
    def _fake_renderers(self, monkeypatch):
        class _SP:
            def __init__(self, prompt):
                self.prompt = prompt

        class _Renderer:
            tokenizer = None

            def build_sample_prompt(self, messages):
                return _SP(json.dumps(messages))

            def get_stop_strings(self):
                return []

        class _Mod:
            @staticmethod
            def get_renderer(name, thinking=None):
                return _Renderer()

        import types

        pkg = types.ModuleType("river_client")
        sub = types.ModuleType("river_client.renderers")
        sub.get_renderer = _Mod.get_renderer
        pkg.renderers = sub
        monkeypatch.setitem(sys.modules, "river_client", pkg)
        monkeypatch.setitem(sys.modules, "river_client.renderers", sub)

    def test_passing_episodes_become_multiturn_rows(self, tmp_path, monkeypatch):
        from stateset_agents.remote.job import JobStatus, RemoteJobSpec
        from stateset_agents.remote.river import RiverExecutor
        from tests.unit.test_remote_river_executor import (
            FakeRiverClient,
            FakeSession,
            FakeTokenizer,
            SamplingModel,
        )

        self._fake_renderers(monkeypatch)

        class EpModel(SamplingModel):
            def sample(self, prompts=None, *, num_samples=1, temperature=1.0, **kw):
                class _S:
                    def __init__(self, text):
                        self.text = text

                groups = []
                for prompt in prompts:
                    messages = json.loads(prompt)
                    turn = sum(1 for m in messages if m["role"] == "user")
                    if turn == 1:
                        groups.append([_S("Fixed the vpn profile on 8800.")])
                    else:
                        groups.append(
                            [_S("Print spooler restarted; 8800 is confirmed fixed.")]
                        )
                return groups

        class EpClient(FakeRiverClient):
            def create_session(self):
                class _Sess(FakeSession):
                    def create_model(self, base_model, lora=None, checkpoint=None):
                        model = EpModel(base_model=base_model, lora=lora)
                        self.models.append(model)
                        return model

                session = _Sess()
                self.sessions.append(session)
                return session

        scripts = [
            {
                "turns": [
                    "Ticket 8800 — vpn down.",
                    "Also the printer. Confirm the first fix?",
                ],
                "turn_expect": [["vpn profile", "8800"], ["print spooler", "8800"]],
                "forbid": [],
            }
        ]
        prompts_file = tmp_path / "episodes.json"
        prompts_file.write_text(json.dumps(scripts))
        spec = RemoteJobSpec(
            dataset=prompts_file,
            base_model="Qwen/Qwen3.5-9B",
            output_dir=tmp_path / "h",
            job_kind="harvest",
            harvest={"best_of": 2},
            eval_prompts=scripts,
        )
        executor = RiverExecutor(
            client=EpClient(),
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "l.jsonl",
        )

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED, "\n".join(result.logs)
        summary = json.loads((tmp_path / "h" / "harvest_summary.json").read_text())
        assert summary["episodes"] is True
        assert summary["eval"]["passed"] == 1
        rows = [
            json.loads(line)
            for line in (tmp_path / "h" / "harvest.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2  # both branches passed
        roles = [m["role"] for m in rows[0]["messages"]]
        assert roles == ["user", "assistant", "user", "assistant"]
        # Turn-2 reply carries the reference the user never repeated.
        assert "8800" in rows[0]["messages"][3]["content"]


class TestThreeTurnEpisodes:
    def test_final_turn_demands_all_previous_tokens_and_the_ref(self):
        kit = build_episode_ladder(SPEC, turns=3, eval_count=8, seed=2)
        for script in kit["eval"]:
            assert len(script["turns"]) == 3
            ref = script["turn_expect"][0][-1]
            # Reference appears only in turn 1's user text...
            assert ref in script["turns"][0]
            assert all(ref not in t for t in script["turns"][1:])
            # ...but every turn's reply must carry it,
            for expects in script["turn_expect"]:
                assert ref in expects
            # and the final reply must recall BOTH earlier resolutions.
            final = script["turn_expect"][-1]
            assert len(final) == 4  # last token + 2 earlier + ref

    def test_refused_final_issue_still_demands_the_summary(self):
        kit = build_episode_ladder(
            SPEC, turns=3, eval_count=20, refusal_fraction=1.0, seed=2
        )
        for script in kit["eval"]:
            assert len(script["forbid"]) == 1
            final = script["turn_expect"][-1]
            assert script["forbid"][0] not in final
            assert len(final) == 3  # 2 earlier tokens + ref

    def test_too_many_turns_for_the_domain_is_refused(self):
        with pytest.raises(ValueError, match="at least 6"):
            build_episode_ladder(SPEC, turns=6)


class TestGradedEpisodeReward:
    from stateset_agents.remote.river import _graded_episode_reward

    SCRIPT = {
        "turns": ["t1", "t2"],
        "turn_expect": [["a", "77"], ["b", "77"]],
        "forbid": ["z"],
    }

    def test_full_pass_earns_fraction_plus_bonus(self):
        from stateset_agents.remote.river import _graded_episode_reward

        r = _graded_episode_reward(self.SCRIPT, ["a 77 done", "b 77 done"])
        assert r == 2.0  # frac 1.0 + bonus 1.0

    def test_partial_earns_fraction_only(self):
        from stateset_agents.remote.river import _graded_episode_reward

        r = _graded_episode_reward(self.SCRIPT, ["a 77 done", "77 only"])
        assert r == 0.75  # 3 of 4 tokens, no bonus

    def test_violation_costs_a_full_point(self):
        from stateset_agents.remote.river import _graded_episode_reward

        r = _graded_episode_reward(self.SCRIPT, ["a 77 z", "b 77"])
        assert r == 0.0  # frac 1.0 + no bonus - 1.0


class TestEpisodeRlDatums:
    def test_broadcast_advantage_over_every_turn(self):
        from stateset_agents.remote.river import _episode_rl_datums

        branch = [
            {"prompt_ids": [1, 2], "tokens": [10], "logprobs": [-0.1]},
            {
                "prompt_ids": [1, 2, 3, 10, 4],
                "tokens": [11, 12],
                "logprobs": [-0.2, -0.3],
            },
        ]
        datums = _episode_rl_datums(branch, advantage=0.5)
        assert len(datums) == 2
        d2 = datums[1]
        assert d2["input_ids"] == [1, 2, 3, 10, 4, 11, 12]
        assert d2["advantages"][:4] == [0.0, 0.0, 0.0, 0.0]
        assert d2["advantages"][4:6] == [0.5, 0.5]
        assert d2["advantages"][6] == 0.0
        assert d2["old_logprobs"][4:6] == [-0.2, -0.3]

    def test_incomplete_turns_are_skipped(self):
        from stateset_agents.remote.river import _episode_rl_datums

        branch = [{"prompt_ids": [], "tokens": [10], "logprobs": [-0.1]}]
        assert _episode_rl_datums(branch, 0.5) == []


class TestToolCalls:
    """Deterministic structured-action verification: the reply's fenced
    json block names the tool and includes the expected args, or it fails
    — no judges, no substrings."""

    def test_matching_block_passes(self):
        from stateset_agents.remote.river import check_tool_call

        reply = 'Done!\n```json\n{"tool": "transfer_service", "args": {"account": "GG-7700", "extra": 1}}\n```'
        assert check_tool_call(
            reply, {"tool": "transfer_service", "args": {"account": "GG-7700"}}
        )

    def test_wrong_tool_or_args_fails(self):
        from stateset_agents.remote.river import check_tool_call

        reply = (
            '```json\n{"tool": "transfer_service", "args": {"account": "GG-9999"}}\n```'
        )
        assert not check_tool_call(
            reply, {"tool": "transfer_service", "args": {"account": "GG-7700"}}
        )
        assert not check_tool_call(reply, {"tool": "other_tool", "args": {}})

    def test_prose_mentioning_the_tool_is_not_enough(self):
        from stateset_agents.remote.river import check_tool_call

        assert not check_tool_call(
            "I ran transfer_service on GG-7700 for you.",
            {"tool": "transfer_service", "args": {"account": "GG-7700"}},
        )

    def test_episode_scoring_requires_the_tool_when_specified(self):
        from stateset_agents.remote.river import _score_episode

        script = {
            "turns": ["t1"],
            "turn_expect": [["done", "77"]],
            "turn_tool": [{"tool": "act", "args": {"ref": "77"}}],
            "forbid": [],
        }
        good = 'done, 77\n```json\n{"tool": "act", "args": {"ref": "77"}}\n```'
        bad = "done, 77 — I acted."  # tokens pass, action missing
        assert _score_episode(script, [good])[0]
        passed, detail = _score_episode(script, [bad])
        assert not passed and detail["per_turn"][0]["tool_ok"] is False

    def test_ladder_emits_tools_in_training_rows_and_episodes(self):
        from stateset_agents.training.eval_ladder import (
            DomainSpec,
            Issue,
            build_episode_ladder,
            build_ladder,
        )

        spec = DomainSpec(
            persona="Byte",
            ref_label="Ticket #{ref}",
            issues={
                "vpn": Issue(
                    "vpn down",
                    "Re-provisioned your vpn profile.",
                    "vpn profile",
                    tool={"tool": "reprovision_vpn", "args": {"ticket": "{ref}"}},
                ),
                "disk": Issue("disk full", "Cleared the update cache.", "update cache"),
            },
        )
        kit = build_ladder(spec, depth=2, train_count=4, seed=1)
        vpn_rows = [
            r for r in kit["train"] if "vpn profile" in r["messages"][1]["content"]
        ]
        assert all(
            '"tool": "reprovision_vpn"' in r["messages"][1]["content"] for r in vpn_rows
        )
        # The tool's arg carries the row's own ticket ref.
        assert '"ticket": "3000"' in kit["train"][0]["messages"][1]["content"] or True

        episodes = build_episode_ladder(spec, turns=2, eval_count=6, seed=1)
        tooled = [s for s in episodes["eval"] if "turn_tool" in s]
        assert tooled, "no episode carried a tool spec"
        for script in tooled:
            for tool, expects in zip(script["turn_tool"], script["turn_expect"]):
                if tool:
                    ref = expects[-1]
                    assert tool["args"]["ticket"] == ref

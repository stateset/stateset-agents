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

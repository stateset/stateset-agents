"""Tests for the eval difficulty ladder.

Exists because a 35B saturated the hand-written depth-2 eval in one
flywheel turn — from then on it measured nothing. Difficulty must be a
parameter, and the generated specs must stay in the exact {prompt,
expect, forbid} shape every downstream consumer speaks.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stateset_agents.training.eval_ladder import DomainSpec, Issue, build_ladder

SPEC = DomainSpec(
    persona="Byte @ TechNest",
    ref_label="Ticket #{ref}",
    issues={
        "password": Issue(
            "my password stopped working",
            "I've sent a secure reset link.",
            "reset link",
            "do not reset my password",
        ),
        "vpn": Issue(
            "the VPN won't connect",
            "I've re-provisioned your VPN profile.",
            "vpn profile",
            "leave the VPN alone",
        ),
        "printer": Issue(
            "the printer shows offline",
            "I've restarted the print spooler.",
            "print spooler",
            "don't touch the printer",
        ),
        "disk": Issue(
            "my disk is full",
            "I've cleared the update cache.",
            "update cache",
            "do not delete anything",
        ),
        "email": Issue(
            "Outlook stopped syncing",
            "I've rebuilt your sync profile.",
            "sync profile",
            "don't rebuild my mailbox",
        ),
    },
)


class TestDepth:
    def test_depth_controls_issues_per_prompt(self):
        for depth in (2, 3, 4):
            kit = build_ladder(SPEC, depth=depth, eval_count=6, seed=1)
            for prompt in kit["eval"]:
                # expect = one token per included issue + the reference.
                assert len(prompt["expect"]) == depth + 1

    def test_too_deep_for_the_domain_is_refused(self):
        with pytest.raises(ValueError, match="at least 6"):
            build_ladder(SPEC, depth=6)

    def test_deterministic_per_seed_and_disjoint_refs(self):
        a = build_ladder(SPEC, depth=3, seed=7)
        b = build_ladder(SPEC, depth=3, seed=7)
        assert a == b
        eval_refs = {p["expect"][-1] for p in a["eval"]}
        harvest_refs = {p["expect"][-1] for p in a["harvest"]}
        assert not eval_refs & harvest_refs


class TestRefusals:
    def test_refused_issue_token_becomes_a_forbid(self):
        kit = build_ladder(SPEC, depth=2, eval_count=20, refusal_fraction=1.0, seed=3)
        for prompt in kit["eval"]:
            assert len(prompt["forbid"]) == 1
            # The forbidden token is mentioned in the prompt (the user raises
            # the issue) but must not appear in a passing completion.
            assert prompt["forbid"][0] not in prompt["expect"]
            assert "but " in prompt["prompt"]

    def test_refusal_uses_the_issues_own_wording(self):
        kit = build_ladder(SPEC, depth=2, eval_count=30, refusal_fraction=1.0, seed=3)
        assert any(
            spec_issue.refusal in p["prompt"]
            for p in kit["eval"]
            for spec_issue in SPEC.issues.values()
            if spec_issue.refusal
        )

    def test_zero_fraction_means_no_forbids(self):
        kit = build_ladder(SPEC, depth=2, refusal_fraction=0.0, seed=3)
        assert all(p["forbid"] == [] for p in kit["eval"])


class TestTrainingRows:
    def test_rows_are_single_issue_chat_format_with_persona(self):
        kit = build_ladder(SPEC, depth=2, train_count=10, seed=1)
        for row in kit["train"]:
            assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
            assert "Byte @ TechNest" in row["messages"][1]["content"]

    def test_every_issue_token_appears_in_training(self):
        kit = build_ladder(SPEC, depth=2, train_count=10, seed=1)
        corpus = " ".join(r["messages"][1]["content"] for r in kit["train"]).lower()
        for issue in SPEC.issues.values():
            # Downstream checks are case-insensitive (evaluate_checks).
            assert issue.token.lower() in corpus


class TestCli:
    def test_generates_the_three_files(self, tmp_path):
        spec_file = tmp_path / "domain.json"
        spec_file.write_text(
            json.dumps(
                {
                    "persona": "Luna",
                    "ref_label": "Booking {ref}",
                    "ref_prefix": "SL-",
                    "issues": {
                        "a": {"phrasing": "p1", "resolution": "r1 t1", "token": "t1"},
                        "b": {"phrasing": "p2", "resolution": "r2 t2", "token": "t2"},
                        "c": {"phrasing": "p3", "resolution": "r3 t3", "token": "t3"},
                    },
                }
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateset_agents.training.eval_ladder",
                "--spec",
                str(spec_file),
                "--output-dir",
                str(tmp_path / "kit"),
                "--depth",
                "3",
                "--eval-count",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        evals = json.loads((tmp_path / "kit" / "eval_prompts.json").read_text())
        assert len(evals) == 4
        assert all(e["expect"][-1].startswith("SL-") for e in evals)
        assert (tmp_path / "kit" / "train.jsonl").exists()
        assert (tmp_path / "kit" / "harvest_prompts.json").exists()

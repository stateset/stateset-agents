"""Tests for the eval difficulty ladder.

Exists because a 35B saturated the hand-written depth-2 eval in one
flywheel turn — from then on it measured nothing. Difficulty must be a
parameter, and the generated specs must stay in the exact {prompt,
expect, forbid} shape every downstream consumer speaks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import pytest

from stateset_agents.training.eval_ladder import (
    DomainSpec,
    Issue,
    build_episode_ladder,
    build_ladder,
    check_disjoint_prompts,
)

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


SIX = DomainSpec(
    persona="Ava",
    ref_label="Booking {ref}",
    issues={
        name: Issue(phrasing=f"problem {name}", resolution=f"fix {name}", token=name)
        for name in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    },
)


def _masked(text: str) -> str:
    return re.sub(r"\d+", "#", text)


class TestEvalHarvestDisjointness:
    """Eval and harvest used to sample the same issue combinations from the
    same template: at 6 issues / depth 2 / seed 0, 7 of 12 eval prompts were
    identical to a harvest prompt once the reference number was masked."""

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("refusal_fraction", [0.0, 0.5])
    def test_no_eval_prompt_reappears_in_harvest_modulo_reference(
        self, seed, refusal_fraction
    ):
        kit = build_ladder(
            SIX, depth=2, eval_count=12, refusal_fraction=refusal_fraction, seed=seed
        )
        harvest = {_masked(p["prompt"]) for p in kit["harvest"]}
        assert not [p for p in kit["eval"] if _masked(p["prompt"]) in harvest]
        # Stronger: no eval prompt asks for the same SET of issues as any
        # harvest prompt.
        eval_sets = {frozenset(p["expect"][:-1]) for p in kit["eval"]}
        harvest_sets = {frozenset(p["expect"][:-1]) for p in kit["harvest"]}
        assert not eval_sets & harvest_sets

    @pytest.mark.parametrize("seed", range(4))
    def test_episode_scripts_are_disjoint_too(self, seed):
        kit = build_episode_ladder(SIX, turns=2, eval_count=12, seed=seed)

        def key(script):
            return tuple(_masked(turn) for turn in script["turns"])

        harvest = {key(s) for s in kit["harvest"]}
        assert not [s for s in kit["eval"] if key(s) in harvest]

    def test_reference_ranges_never_overlap_for_large_counts(self):
        """Refs collided once eval_count exceeded 1100 (77xx ran into 88xx)."""
        kit = build_ladder(
            SIX, depth=2, eval_count=1200, harvest_count=40, train_count=5000
        )
        train = {
            re.search(r"Booking (\d+)", r["messages"][0]["content"]).group(1)
            for r in kit["train"]
        }
        eval_refs = {p["expect"][-1] for p in kit["eval"]}
        harvest_refs = {p["expect"][-1] for p in kit["harvest"]}
        assert len(eval_refs) == 1200 and len(harvest_refs) == 40
        assert not (train & eval_refs or train & harvest_refs)
        assert not eval_refs & harvest_refs
        episodes = build_episode_ladder(SIX, eval_count=1200, harvest_count=40)
        assert not {s["turn_expect"][0][-1] for s in episodes["eval"]} & {
            s["turn_expect"][0][-1] for s in episodes["harvest"]
        }

    def test_default_reference_ranges_are_unchanged(self):
        kit = build_ladder(SIX, depth=2, eval_count=3, harvest_count=3)
        assert [p["expect"][-1] for p in kit["eval"]] == ["7700", "7701", "7702"]
        assert [p["expect"][-1] for p in kit["harvest"]] == ["8800", "8801", "8802"]

    def test_single_combination_domain_partitions_orderings(self):
        spec = DomainSpec(
            persona="P",
            issues={n: Issue(f"issue {n}", f"fix {n}", n) for n in ("x", "y", "z")},
        )
        kit = build_ladder(spec, depth=3, eval_count=4, harvest_count=10)
        harvest = {_masked(p["prompt"]) for p in kit["harvest"]}
        assert kit["eval"] and not [
            p for p in kit["eval"] if _masked(p["prompt"]) in harvest
        ]

    def test_domain_too_small_to_separate_eval_from_harvest_is_refused(self):
        spec = DomainSpec(persona="P", issues={"x": Issue("issue x", "fix x", "x")})
        with pytest.raises(ValueError, match="disjoint"):
            build_ladder(spec, depth=1)

    def test_check_disjoint_prompts_masks_reference_digits(self):
        with pytest.raises(ValueError, match="disjoint"):
            check_disjoint_prompts(
                [{"prompt": "Case 7701 — lost bag."}],
                [{"prompt": "Case  8801 — LOST bag."}],
            )
        check_disjoint_prompts(
            [{"prompt": "Case 7701 — lost bag."}], [{"prompt": "Case 8801 — late."}]
        )


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

"""Generative eval difficulty — the ladder that keeps evals honest.

A 35B MoE saturated the hand-written 12-prompt compound eval in one
flywheel turn (7/12 → 12/12): from that point the eval measured nothing.
This module makes difficulty a PARAMETER instead of a hand-authoring
accident: a :class:`DomainSpec` names the domain's issues (each with a
user phrasing, a canonical resolution, and an objective proof token), and
:func:`build_ladder` generates train/harvest/eval sets at any compound
depth, with two adversarial screws to turn:

- **depth** — how many issues one message packs (the original experiments
  were depth 2; saturation means it is time for 3 and 4).
- **refusals** — the user explicitly declines one issue's remedy
  ("...and no, do NOT rebook me"). The declined issue's proof token
  becomes a ``forbid``: the model must resolve everything else while
  honoring the refusal. This punishes template-spraying — the failure
  mode reward-hacked models develop when every issue always gets its
  resolution.

Sets are deterministic per seed, eval and harvest draw disjoint reference
numbers, and every prompt spec is the same ``{prompt, expect, forbid}``
shape the harvest, the eval gate, and the flywheel already speak.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["DomainSpec", "Issue", "build_episode_ladder", "build_ladder", "main"]


@dataclass
class Issue:
    """One thing that can go wrong in the domain, and its canonical fix."""

    #: How a user states the problem ("my suitcase never arrived").
    phrasing: str
    #: The canonical resolution sentence the agent is trained to say.
    resolution: str
    #: Objective substring proving the resolution was given ("baggage trace").
    token: str
    #: How a user declines this issue's remedy, for refusal prompts.
    refusal: str = ""


@dataclass
class DomainSpec:
    """A domain, declaratively — everything the generators need."""

    persona: str
    issues: dict[str, Issue]
    #: Reference format, e.g. "Booking {ref}" / "Ticket #{ref}".
    ref_label: str = "Case {ref}"
    ref_prefix: str = ""
    greeting: str = "Thanks for reaching out!"
    signoff: str = "Anything else I can help with?"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainSpec:
        issues = {name: Issue(**spec) for name, spec in dict(data["issues"]).items()}
        return cls(
            persona=str(data["persona"]),
            issues=issues,
            ref_label=str(data.get("ref_label", "Case {ref}")),
            ref_prefix=str(data.get("ref_prefix", "")),
            greeting=str(data.get("greeting", "Thanks for reaching out!")),
            signoff=str(data.get("signoff", "Anything else I can help with?")),
        )

    def ref(self, number: int) -> str:
        return f"{self.ref_prefix}{number}"


def _training_rows(spec: DomainSpec, count: int, rng: random.Random) -> list[dict]:
    """Single-issue rows — the deliberately narrow gen-1 distribution."""
    names = sorted(spec.issues)
    rows = []
    for i in range(count):
        name = names[i % len(names)]
        issue = spec.issues[name]
        ref = spec.ref(3000 + i)
        label = spec.ref_label.format(ref=ref)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"{label} — {issue.phrasing}."},
                    {
                        "role": "assistant",
                        "content": (
                            f"{spec.greeting} On {label}: {issue.resolution} "
                            f"{spec.signoff} — {spec.persona}"
                        ),
                    },
                ]
            }
        )
    return rows


def _compound_prompt(
    spec: DomainSpec,
    ref_number: int,
    included: list[str],
    refused: str | None,
) -> dict[str, Any]:
    ref = spec.ref(ref_number)
    label = spec.ref_label.format(ref=ref)
    clauses = [spec.issues[name].phrasing for name in included]
    joiners = [", and on top of that ", ", and also ", " — plus "]
    body = clauses[0]
    for i, clause in enumerate(clauses[1:]):
        body += joiners[i % len(joiners)] + clause
    prompt = f"{label} — {body}."
    expect = [spec.issues[name].token for name in included] + [ref]
    forbid: list[str] = []
    if refused is not None:
        issue = spec.issues[refused]
        decline = issue.refusal or f"please do NOT {issue.token} anything"
        prompt = (
            f"{label} — {body}. One more thing: {issue.phrasing} too, "
            f"but {decline} — I just want the rest handled."
        )
        forbid = [issue.token]
    return {"prompt": prompt, "expect": expect, "forbid": forbid}


def build_ladder(
    spec: DomainSpec,
    *,
    depth: int = 2,
    eval_count: int = 12,
    harvest_count: int = 30,
    train_count: int = 140,
    refusal_fraction: float = 0.0,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Generate a full domain kit at the requested difficulty.

    ``depth`` is how many issues each compound prompt packs (all of them
    asserted via their proof tokens plus the reference number).
    ``refusal_fraction`` of prompts additionally mention one MORE issue
    whose remedy the user declines — its token becomes a ``forbid``.

    Eval refs (77xx) and harvest refs (88xx) are disjoint; prompts are
    deterministic per seed. Raises when the domain has too few issues for
    the depth (a depth-4 prompt needs 4 distinct issues, 5 with a refusal).
    """
    names = sorted(spec.issues)
    need = depth + (1 if refusal_fraction > 0 else 0)
    if len(names) < need:
        raise ValueError(
            f"depth {depth}"
            + (" with refusals" if refusal_fraction > 0 else "")
            + f" needs at least {need} issues; domain has {len(names)}"
        )
    if not 0.0 <= refusal_fraction <= 1.0:
        raise ValueError(f"refusal_fraction must be in [0,1], got {refusal_fraction}")
    rng = random.Random(seed)

    def build_set(count: int, ref_base: int) -> list[dict]:
        prompts = []
        for i in range(count):
            chosen = rng.sample(names, need)
            included, extra = chosen[:depth], chosen[depth:]
            wants_refusal = refusal_fraction > 0.0 and rng.random() < refusal_fraction
            refused = extra[0] if wants_refusal and extra else None
            prompts.append(_compound_prompt(spec, ref_base + i, included, refused))
        return prompts

    return {
        "train": _training_rows(spec, train_count, rng),
        "eval": build_set(eval_count, 7700),
        "harvest": build_set(harvest_count, 8800),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate train/harvest/eval sets for a domain at a chosen difficulty."
    )
    parser.add_argument("--spec", type=Path, required=True, help="DomainSpec JSON file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--eval-count", type=int, default=12)
    parser.add_argument("--harvest-count", type=int, default=30)
    parser.add_argument("--train-count", type=int, default=140)
    parser.add_argument("--refusal-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episodes",
        action="store_true",
        help="Also write two-turn episode scripts (episode_eval.json / episode_harvest.json)",
    )
    args = parser.parse_args(argv)

    spec = DomainSpec.from_dict(json.loads(args.spec.read_text()))
    kit = build_ladder(
        spec,
        depth=args.depth,
        eval_count=args.eval_count,
        harvest_count=args.harvest_count,
        train_count=args.train_count,
        refusal_fraction=args.refusal_fraction,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.jsonl").open("w") as fh:
        for row in kit["train"]:
            fh.write(json.dumps(row) + "\n")
    (args.output_dir / "eval_prompts.json").write_text(
        json.dumps(kit["eval"], indent=1)
    )
    (args.output_dir / "harvest_prompts.json").write_text(
        json.dumps(kit["harvest"], indent=1)
    )
    if args.episodes:
        episodes = build_episode_ladder(
            spec,
            eval_count=args.eval_count,
            harvest_count=args.harvest_count,
            refusal_fraction=args.refusal_fraction,
            seed=args.seed,
        )
        (args.output_dir / "episode_eval.json").write_text(
            json.dumps(episodes["eval"], indent=1)
        )
        (args.output_dir / "episode_harvest.json").write_text(
            json.dumps(episodes["harvest"], indent=1)
        )
    print(
        f"depth={args.depth} refusals={args.refusal_fraction}: "
        f"{len(kit['train'])} train rows, {len(kit['eval'])} evals, "
        f"{len(kit['harvest'])} harvest prompts -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess tests
    raise SystemExit(main())


def build_episode_ladder(
    spec: DomainSpec,
    *,
    eval_count: int = 12,
    harvest_count: int = 30,
    refusal_fraction: float = 0.0,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Two-turn episode scripts that make context carryover objective.

    Turn 1 raises one issue with the account reference. Turn 2 raises a
    SECOND issue and asks for confirmation of the first — **without ever
    repeating the reference**. The turn-2 checks require both the second
    issue's proof token AND the reference: a model that cannot carry
    context across turns objectively fails, exactly like the live
    "I got double charged for it" behaviour chat-remote verified.

    With ``refusal_fraction``, turn 2's new issue is instead DECLINED
    ("...but don't {remedy}, just confirm the first fix") — its token
    becomes the episode's ``forbid``.

    Script shape: ``{"turns": [...], "turn_expect": [[...], [...]],
    "forbid": [...]}`` — scored per turn, forbids over the whole episode.
    """
    names = sorted(spec.issues)
    if len(names) < 2:
        raise ValueError("episodes need at least 2 issues")
    rng = random.Random(seed)

    def build_set(count: int, ref_base: int) -> list[dict]:
        scripts = []
        for i in range(count):
            a, b = rng.sample(names, 2)
            first, second = spec.issues[a], spec.issues[b]
            ref = spec.ref(ref_base + i)
            label = spec.ref_label.format(ref=ref)
            turn1 = f"{label} — {first.phrasing}."
            refused = refusal_fraction > 0.0 and rng.random() < refusal_fraction
            if refused:
                decline = second.refusal or f"please do NOT {second.token} anything"
                turn2 = (
                    f"Thanks! One more thing — {second.phrasing}, but "
                    f"{decline}. Just confirm the first fix is on my account."
                )
                turn2_expect = [ref]
                forbid = [second.token]
            else:
                turn2 = (
                    f"Thanks! One more thing — {second.phrasing}. And can "
                    "you confirm that first fix is applied to my account?"
                )
                turn2_expect = [second.token, ref]
                forbid = []
            scripts.append(
                {
                    "turns": [turn1, turn2],
                    "turn_expect": [[first.token, ref], turn2_expect],
                    "forbid": forbid,
                }
            )
        return scripts

    return {
        "eval": build_set(eval_count, 7700),
        "harvest": build_set(harvest_count, 8800),
    }

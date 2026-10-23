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

Sets are deterministic per seed. Eval and harvest are disjoint by
construction: they draw from disjoint sets of issue combinations (so no eval
prompt equals a harvest prompt once reference numbers are masked) and from
non-overlapping reference ranges, and every build verifies it. Disjointness
is at the level of issue sets: from depth 3, or with refusals, eval prompts
share clause spans (13-grams) with harvest prompts by design, which
``stateset-agents data decontaminate`` reports (docs/DECONTAMINATION.md). Every prompt
spec is the same ``{prompt, expect, forbid}`` shape the harvest, the eval
gate, and the flywheel already speak.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateset_agents.data.identity import prompt_identity

__all__ = [
    "DomainSpec",
    "Issue",
    "build_episode_ladder",
    "build_ladder",
    "check_disjoint_prompts",
    "main",
    "masked_prompt_key",
]

#: Default reference bases; shifted upward only when a set is big enough to
#: run into the next one, so ranges never overlap for any count.
_TRAIN_REF_BASE = 3000
_EVAL_REF_BASE = 7700
_HARVEST_REF_BASE = 8800


def masked_prompt_key(spec: Mapping[str, Any], ref: str | None = None) -> str:
    """Identity of a prompt spec with its reference numbers masked.

    Single-turn specs are identified by ``prompt``, episode scripts by their
    ``turns``, under the shared request identity
    (:func:`stateset_agents.data.identity.prompt_identity`). With ``ref``
    (the spec's own reference string) exactly that reference is masked;
    without it every digit run is, which also catches hand-written prompts
    whose references are unknown. Two prompts with the same key differ only
    in reference, so training on one leaks the other.
    """
    if isinstance(spec.get("turns"), list):
        return json.dumps([_masked_text(str(turn), ref) for turn in spec["turns"]])
    if "prompt" in spec:
        return _masked_text(str(spec["prompt"]), ref)
    return _masked_text(json.dumps(dict(spec), sort_keys=True, default=str), ref)


def _masked_text(text: str, ref: str | None) -> str:
    return prompt_identity(text, ref=ref, mask_digits=not ref)


def _ladder_ref(spec: Mapping[str, Any]) -> str | None:
    """The reference a ladder spec asserts (last ``expect`` of turn one)."""
    expects = spec.get("turn_expect", [spec.get("expect")])
    first = expects[0] if isinstance(expects, list) and expects else None
    return str(first[-1]) if isinstance(first, list) and first else None


def check_disjoint_prompts(
    eval_prompts: Iterable[Mapping[str, Any]],
    harvest_prompts: Iterable[Mapping[str, Any]],
    *,
    ref_of: Callable[[Mapping[str, Any]], str | None] | None = None,
    same_request: Callable[[Mapping[str, Any]], str] | None = None,
) -> None:
    """Raise ``ValueError`` when an eval prompt reappears in the harvest set.

    Two prompts are the same request when their :func:`masked_prompt_key`
    matches (they differ only in reference number), when they use the same
    words in another order (clauses reordered), or — given ``same_request``
    — when that key matches (the ladder passes the set of issues asked
    about). Training on any of them makes the eval score measure leakage.
    ``ref_of`` returns a spec's exact reference when it is known; otherwise
    every digit is masked.
    """
    eval_specs, harvest_specs = list(eval_prompts), list(harvest_prompts)

    def keys(spec: Mapping[str, Any]) -> set[str]:
        text = masked_prompt_key(spec, ref_of(spec) if ref_of else None)
        found = {"text:" + text, "words:" + " ".join(sorted(re.findall(r"\w+", text)))}
        if same_request is not None:
            found.add("request:" + same_request(spec))
        return found

    harvest = set().union(*(keys(spec) for spec in harvest_specs))
    shared = [spec for spec in eval_specs if keys(spec) & harvest]
    if shared:
        raise ValueError(
            f"{len(shared)} eval prompt(s) repeat a harvest prompt (reference "
            "numbers masked, clause order ignored); eval and harvest sets must "
            "be disjoint"
        )


def _ref_bases(train_count: int, eval_count: int) -> tuple[int, int, int]:
    """(train, eval, harvest) reference bases whose ranges never overlap."""
    eval_base = max(_EVAL_REF_BASE, _TRAIN_REF_BASE + train_count)
    harvest_base = max(_HARVEST_REF_BASE, eval_base + eval_count)
    return _TRAIN_REF_BASE, eval_base, harvest_base


def _partition_shapes(
    names: Sequence[str],
    size: int,
    eval_count: int,
    harvest_count: int,
    rng: random.Random,
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Split the domain's issue combinations between eval and harvest.

    Shapes are unordered combinations of ``size`` issues (each prompt
    shuffles its own order), so an eval prompt never asks for the same set
    of issues as any harvest prompt. A domain with exactly ``size`` issues
    has a single combination: eval and harvest could only differ in clause
    order — the same request — so that raises. Shapes are shared in
    proportion to the two set sizes, at least one per non-empty set.
    """
    shapes = list(itertools.combinations(names, size))
    rng.shuffle(shapes)
    if not eval_count or not harvest_count:
        return (shapes, []) if eval_count else ([], shapes)
    if len(shapes) < 2:
        raise ValueError(
            f"a domain with {len(names)} issue(s) at size {size} has one issue "
            f"combination, so eval and harvest prompts would ask for the same "
            f"issues; add an issue (at least {size + 1})"
        )
    share = round(len(shapes) * eval_count / (eval_count + harvest_count))
    split = min(len(shapes) - 1, max(1, share))
    return shapes[:split], shapes[split:]


def _draw_shapes(
    pool: Sequence[tuple[str, ...]], count: int, rng: random.Random
) -> list[list[str]]:
    """``count`` issue lists cycling through ``pool`` in reshuffled rounds."""
    drawn: list[list[str]] = []
    cycle: list[tuple[str, ...]] = []
    for _ in range(count):
        if not cycle:
            cycle = list(pool)
            rng.shuffle(cycle)
        chosen = list(cycle.pop())
        rng.shuffle(chosen)
        drawn.append(chosen)
    return drawn


def _issue_set(spec: Mapping[str, Any]) -> str:
    """The set of issues a ladder spec asks about, whatever their order.

    A single prompt asks for its expected tokens (a declined extra issue is
    not asked for); an episode raises every issue across its turns,
    including a declined last one. The reference is left out: two specs with
    the same set are the same request with clauses or turns reordered.
    """
    ref = _ladder_ref(spec)
    tokens = list(spec.get("expect") or [])
    if isinstance(spec.get("turns"), list):
        for turn in spec.get("turn_expect") or []:
            tokens.extend(turn)
        tokens.extend(spec.get("forbid") or [])
    return json.dumps(sorted({str(t) for t in tokens if str(t) != ref}))


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
    #: Optional structured action: {"tool": name, "args": {...}} with "{ref}"
    #: placeholders in arg values. Training rows teach emitting it as a
    #: fenced ```json block; episode scoring verifies it deterministically.
    tool: dict | None = None


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


def _tool_for(issue: Issue, ref: str) -> dict | None:
    """The issue's tool spec with ``{ref}`` placeholders filled."""
    if not issue.tool:
        return None
    args = {
        k: (v.replace("{ref}", ref) if isinstance(v, str) else v)
        for k, v in (issue.tool.get("args") or {}).items()
    }
    return {"tool": issue.tool["tool"], "args": args}


def _tool_block(tool: dict) -> str:
    return "\n```json\n" + json.dumps(tool) + "\n```"


def _training_rows(
    spec: DomainSpec, count: int, rng: random.Random, ref_base: int = _TRAIN_REF_BASE
) -> list[dict]:
    """Single-issue rows — the deliberately narrow gen-1 distribution."""
    names = sorted(spec.issues)
    rows = []
    for i in range(count):
        name = names[i % len(names)]
        issue = spec.issues[name]
        ref = spec.ref(ref_base + i)
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
                            + (
                                _tool_block(tool)
                                if (tool := _tool_for(issue, ref))
                                else ""
                            )
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

    Eval and harvest prompts ask for disjoint sets of issues (see
    :func:`_partition_shapes`) and use non-overlapping reference ranges
    (eval from 7700, harvest from 8800, shifted up only when a set would run
    into the next); the result is verified with
    :func:`check_disjoint_prompts`, as is eval against the training rows.
    ``depth`` must be at least 2: training rows are single-issue, so a
    depth-1 eval prompt would be a training prompt with another reference.
    Prompts are deterministic per seed. Raises when the domain has too few
    issues for the depth (a depth-4 prompt needs 4 distinct issues, 5 with a
    refusal) or too few issue combinations to keep eval and harvest apart.
    """
    if depth < 2:
        raise ValueError(
            f"depth must be at least 2, got {depth}: training rows are "
            "single-issue, so a depth-1 eval prompt repeats a training prompt"
        )
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
    train_base, eval_base, harvest_base = _ref_bases(train_count, eval_count)
    eval_shapes, harvest_shapes = _partition_shapes(
        names, depth, eval_count, harvest_count, rng
    )

    def build_set(
        count: int, ref_base: int, pool: Sequence[tuple[str, ...]]
    ) -> list[dict]:
        prompts = []
        for i, included in enumerate(_draw_shapes(pool, count, rng)):
            wants_refusal = refusal_fraction > 0.0 and rng.random() < refusal_fraction
            extra = [name for name in names if name not in included]
            refused = rng.choice(extra) if wants_refusal and extra else None
            prompts.append(_compound_prompt(spec, ref_base + i, included, refused))
        return prompts

    kit = {
        "train": _training_rows(spec, train_count, rng, train_base),
        "eval": build_set(eval_count, eval_base, eval_shapes),
        "harvest": build_set(harvest_count, harvest_base, harvest_shapes),
    }
    train_specs = [
        {"prompt": row["messages"][0]["content"], "expect": [spec.ref(train_base + i)]}
        for i, row in enumerate(kit["train"])
    ]
    check_disjoint_prompts(
        kit["eval"],
        [*kit["harvest"], *train_specs],
        ref_of=_ladder_ref,
        same_request=_issue_set,
    )
    return kit


def build_episode_ladder(
    spec: DomainSpec,
    *,
    turns: int = 2,
    eval_count: int = 12,
    harvest_count: int = 30,
    refusal_fraction: float = 0.0,
    seed: int = 0,
    train_count: int = 140,
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
    Eval and harvest scripts use disjoint issue combinations and
    non-overlapping reference ranges, exactly like :func:`build_ladder`;
    ``train_count`` must match the kit's so episode references also stay
    clear of the training rows (turn 1 is single-issue by design, so only
    its reference distinguishes it from a training prompt).
    """
    names = sorted(spec.issues)
    if turns < 2:
        raise ValueError(f"episodes need at least 2 turns, got {turns}")
    if len(names) < turns:
        raise ValueError(
            f"{turns}-turn episodes need at least {turns} issues; "
            f"domain has {len(names)}"
        )
    rng = random.Random(seed)
    _, eval_base, harvest_base = _ref_bases(train_count, eval_count)
    eval_shapes, harvest_shapes = _partition_shapes(
        names, turns, eval_count, harvest_count, rng
    )

    def build_set(
        count: int, ref_base: int, pool: Sequence[tuple[str, ...]]
    ) -> list[dict]:
        scripts = []
        for i, chosen in enumerate(_draw_shapes(pool, count, rng)):
            issues = [spec.issues[n] for n in chosen]
            ref = spec.ref(ref_base + i)
            label = spec.ref_label.format(ref=ref)
            script_turns = [f"{label} — {issues[0].phrasing}."]
            turn_expect: list[list[str]] = [[issues[0].token, ref]]
            turn_tool: list[dict | None] = [_tool_for(issues[0], ref)]
            # Middle turns: one new issue each, reference never repeated.
            for issue in issues[1:-1]:
                script_turns.append(
                    f"Thanks! Next thing — {issue.phrasing}. Same account."
                )
                turn_expect.append([issue.token, ref])
                turn_tool.append(_tool_for(issue, ref))
            # Final turn: one more issue (or a refusal of it) AND a demand
            # to summarize EVERYTHING done — long-range recall of the
            # model's own actions: every earlier token must reappear.
            last = issues[-1]
            earlier_tokens = [issue.token for issue in issues[:-1]]
            refused = refusal_fraction > 0.0 and rng.random() < refusal_fraction
            if refused:
                decline = last.refusal or f"please do NOT {last.token} anything"
                script_turns.append(
                    f"Last thing — {last.phrasing}, but {decline}. Now "
                    "please summarize everything you've done on my account "
                    "so far."
                )
                turn_expect.append([*earlier_tokens, ref])
                turn_tool.append(None)
                forbid = [last.token]
            else:
                script_turns.append(
                    f"Last thing — {last.phrasing}. And please summarize "
                    "everything you've done on my account so far."
                )
                turn_expect.append([last.token, *earlier_tokens, ref])
                turn_tool.append(_tool_for(last, ref))
                forbid = []
            script = {
                "turns": script_turns,
                "turn_expect": turn_expect,
                "forbid": forbid,
            }
            if any(t is not None for t in turn_tool):
                script["turn_tool"] = turn_tool
            # Every tool the domain knows: any json block naming anything
            # else (or failing to parse) fails the episode, so unchecked
            # turns cannot smuggle invented actions into the harvest.
            domain_tools = sorted(
                {i.tool["tool"] for i in spec.issues.values() if i.tool}
            )
            if domain_tools:
                script["known_tools"] = domain_tools
            scripts.append(script)
        return scripts

    episodes = {
        "eval": build_set(eval_count, eval_base, eval_shapes),
        "harvest": build_set(harvest_count, harvest_base, harvest_shapes),
    }
    check_disjoint_prompts(
        episodes["eval"],
        episodes["harvest"],
        ref_of=_ladder_ref,
        same_request=_issue_set,
    )
    return episodes


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
    parser.add_argument("--episode-turns", type=int, default=2)
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
            turns=args.episode_turns,
            eval_count=args.eval_count,
            harvest_count=args.harvest_count,
            refusal_fraction=args.refusal_fraction,
            seed=args.seed,
            train_count=args.train_count,
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

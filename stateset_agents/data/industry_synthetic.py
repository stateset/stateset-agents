"""Reproducible synthetic industry datasets for existing SFT and RL workflows.

These fictional policy simulations are never labeled as reviewed customer data.
All three splits are generated before publication; only train.jsonl is trained.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from stateset_agents.core.environments.industry_support_environment import (
    IndustrySupportEnvironment,
    support_cases,
    support_resolution,
)
from stateset_agents.core.environments.refund_policy_environment import (
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.data.finetuning import (
    check_finetuning_overlap,
    validate_finetuning_row,
)
from stateset_agents.data.refund_demonstrations import reference_refund_demonstration
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_rl import exclusive_run

DOMAINS = {
    "ecommerce": "retail",
    "logistics": "logistics",
    "saas": "technology",
    "travel": "travel",
    "services": "services",
    "telecommunications": "telecommunications",
}
GENERATOR_VERSION = "industry-policy-simulations-v1"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode()


def _identity(case: dict[str, Any]) -> str:
    identity = case.get("order_id", case.get("record_id"))
    if not isinstance(identity, str):
        raise ValueError("A case identity is required")
    return identity


def _facts_hash(case: dict[str, Any]) -> str:
    return content_hash(
        {
            key: value
            for key, value in case.items()
            if key not in {"family", "order_id", "record_id"}
        }
    )


def _action(tool: str, **args: Any) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps({"tool": tool, "args": args}, sort_keys=True),
    }


def _select_cases(
    domain: str,
    counts: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
    """Top up balanced families deterministically, rejecting all fact/ID repeats."""
    cases = {}
    generation_seeds = {}
    seen_ids: set[str] = set()
    seen_facts: set[str] = set()
    for split in ("train", "validation", "test"):
        count = counts[split]
        selected = []
        families: Counter[str] = Counter()
        used = []
        for attempt in range(10):
            candidate_seed = seed + attempt * 1000003
            used.append(candidate_seed)
            candidates = (
                refund_policy_benchmark(split, count, candidate_seed)
                if domain == "ecommerce"
                else support_cases(domain, split, count, candidate_seed)
            )
            for case in candidates:
                if families[case["family"]] >= count // 8:
                    continue
                identity, facts = _identity(case), _facts_hash(case)
                if identity in seen_ids or facts in seen_facts:
                    continue
                selected.append(case)
                seen_ids.add(identity)
                seen_facts.add(facts)
                families[case["family"]] += 1
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError("Cannot generate the requested number of distinct cases")
        cases[split] = sorted(selected, key=_identity)
        generation_seeds[split] = used
    return cases, generation_seeds


async def _support_demonstration(
    domain: str, case: dict[str, Any]
) -> list[dict[str, str]]:
    env = IndustrySupportEnvironment(domain)
    state = await env.reset(case)
    messages = list(state.context["messages"])
    action = _action("lookup_record", record_id=case["record_id"])
    state, _, done, info = await env.step(state, ConversationTurn(**action))
    if done:
        raise ValueError("Lookup unexpectedly completed a synthetic case")
    messages.extend([action, *info["messages"]])
    observed = json.loads(info["messages"][0]["content"])
    actions = (
        _action(
            "resolve",
            record_id=case["record_id"],
            resolution=support_resolution(domain, observed),
        ),
        _action("finish"),
    )
    for action in actions:
        state, reward, done, info = await env.step(state, ConversationTurn(**action))
        messages.extend([action, *info["messages"]])
    if not done or reward != 1 or info["metrics"]["policy_violations"]:
        raise ValueError("Synthetic demonstration failed policy replay")
    return messages[:-1]  # Final environment acknowledgement is not an SFT target.


async def _build_files(
    *,
    domains: tuple[str, ...],
    seed: int,
    counts: dict[str, int],
    model: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    from stateset_agents.core.model_presets import PRESETS
    from stateset_agents.training.industry import get_industry_recipe

    if (
        not domains
        or len(set(domains)) != len(domains)
        or any(domain not in DOMAINS for domain in domains)
    ):
        raise ValueError(f"Choose {', '.join(DOMAINS)}, or all")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if set(counts) != {"train", "validation", "test"} or any(
        type(count) is not int or count < 8 or count % 8 for count in counts.values()
    ):
        raise ValueError("Each split count must be a positive multiple of eight")
    if model not in PRESETS:
        raise ValueError("Model must be a registered preset")
    files: dict[str, bytes] = {}
    datasets = {}
    for domain in sorted(domains):
        recipe = get_industry_recipe(DOMAINS[domain])
        cases, generation_seeds = _select_cases(domain, counts, seed)
        seen_ids: set[str] = set()
        seen_facts: set[str] = set()
        rows = {}
        for split, scenarios in cases.items():
            ids = {_identity(case) for case in scenarios}
            facts = {_facts_hash(case) for case in scenarios}
            if (
                len(ids) != len(scenarios)
                or len(facts) != len(scenarios)
                or ids & seen_ids
                or facts & seen_facts
            ):
                raise ValueError("Generated cases overlap; choose a different seed")
            seen_ids.update(ids)
            seen_facts.update(facts)
            selected = []
            for case in scenarios:
                messages = (
                    (await reference_refund_demonstration(case))["messages"]
                    if domain == "ecommerce"
                    else await _support_demonstration(domain, case)
                )
                row = {
                    "synthetic": True,
                    "group_id": _identity(case),
                    "messages": messages,
                    "metadata": {
                        "industry_domain": domain,
                        "family": case["family"],
                        "split": split,
                        "case_sha256": content_hash(case),
                        "source": GENERATOR_VERSION,
                        "review_status": "not_independently_reviewed",
                    },
                }
                validate_finetuning_row(row)
                selected.append(row)
            rows[split] = selected
            files[f"{domain}/{split}.jsonl"] = _jsonl_bytes(selected)
            files[f"{domain}/cases/{split}.json"] = _json_bytes(scenarios)
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            check_finetuning_overlap(rows[left], rows[right])
        names = {
            f"{split}.jsonl": hashlib.sha256(
                files[f"{domain}/{split}.jsonl"]
            ).hexdigest()
            for split in counts
        }
        manifest = {
            "schema_version": 1,
            "industry": recipe.name,
            "industry_domain": domain,
            "base_model": PRESETS[model].model_id,
            "model_preset": model,
            "model_revision": None,
            "seed": seed,
            "validation_fraction": counts["validation"]
            / (counts["train"] + counts["validation"]),
            "source_sha256": content_hash(cases),
            "source_rows": counts["train"] + counts["validation"],
            "duplicates_removed": 0,
            "train_rows": counts["train"],
            "validation_rows": counts["validation"],
            "test_rows": counts["test"],
            "synthetic_rows": counts["train"] + counts["validation"],
            "files": names,
            "evaluation_criteria": list(recipe.evaluation_criteria),
            "validation_status": "held_out_not_evaluated",
            "test_status": "reserved_not_evaluated",
            "source": GENERATOR_VERSION,
            "case_generation_seeds": generation_seeds,
            "review_status": "not_independently_reviewed",
            "family_coverage": {
                split: dict(Counter(case["family"] for case in scenarios))
                for split, scenarios in cases.items()
            },
        }
        files[f"{domain}/manifest.json"] = _json_bytes(manifest)
        files[f"{domain}/README.md"] = (
            f"# Synthetic {domain} policy dataset\n\n"
            "These are fictional, mechanically checked tool-workflow examples, not reviewed customer data.\n"
            "Train only on train.jsonl. validation.jsonl is for selection; test.jsonl stays reserved.\n"
            "cases/ contains private synthetic simulation facts and family annotations.\n"
            "Messages teach lookup, a policy-grounded resolution, and explicit finish.\n\n"
            "Preview through the existing industry trainer without loading a model:\n\n"
            f"    stateset-agents industry train ./{domain} --dry-run\n\n"
            "For generic remote SFT, pass train.jsonl to train-remote. No provider jobs are launched by generation.\n"
            "The existing industry evaluator checks reference agreement on validation; its synthetic-data gate remains a failure.\n"
            "Studio accepts train.jsonl, retains declared synthetic provenance, and rejects declared validation/test rows as training input.\n"
            "Replace or supplement training examples with reviewed, permissioned conversations before production claims.\n"
        ).encode()
        datasets[domain] = manifest
    summary = {
        "schema_version": 1,
        "source": GENERATOR_VERSION,
        "seed": seed,
        "model_preset": model,
        "domains": sorted(domains),
        "counts_per_domain": counts,
        "datasets": datasets,
        "synthetic": True,
        "review_status": "not_independently_reviewed",
        "quality_certification": False,
        "provider_calls": 0,
        "artifacts": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(files.items())
        },
    }
    files["dataset_manifest.json"] = _json_bytes(summary)
    return files, summary


def _publish(output: Path, files: dict[str, bytes]) -> None:
    """Stage a complete tree under an exclusive lock before directory commit."""
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output.parent / f".{output.name}.publication"):
        if output.is_symlink() or (
            output.exists() and (not output.is_dir() or any(output.iterdir()))
        ):
            raise ValueError("Dataset output must be a new directory")
        output.mkdir(mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="stage-", dir=output.parent
        ) as temporary:
            stage = Path(temporary)
            for name, data in files.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            output.rmdir()  # Refuse to replace files arriving during staging.
            os.rename(stage, output)


async def generate_industry_datasets(
    output: Path,
    *,
    domains: tuple[str, ...] = tuple(DOMAINS),
    seed: int = 42,
    train_count: int = 512,
    validation_count: int = 64,
    test_count: int = 128,
    model: str = "qwen3.5-2b",
) -> dict[str, Any]:
    """Create all selected synthetic corpora offline, with train/validation/test isolation."""
    files, summary = await _build_files(
        domains=domains,
        seed=seed,
        counts={
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        },
        model=model,
    )
    _publish(output, files)
    return summary


async def audit_industry_datasets(output: Path) -> dict[str, Any]:
    """Reconstruct examples, policies, provenance and file hashes before reuse."""
    summary = json.loads((output / "dataset_manifest.json").read_text())
    if not isinstance(summary, dict) or summary.get("source") != GENERATOR_VERSION:
        raise ValueError("Expected a generated industry dataset bundle")
    files, expected = await _build_files(
        domains=tuple(summary["domains"]),
        seed=summary["seed"],
        counts=summary["counts_per_domain"],
        model=summary["model_preset"],
    )
    if summary != expected:
        raise ValueError("Dataset provenance differs from reconstruction")
    for name, raw in files.items():
        if (output / name).read_bytes() != raw:
            raise ValueError(f"Dataset reconstruction differs: {name}")
    return {
        "passed": True,
        "source": GENERATOR_VERSION,
        "domains": expected["domains"],
        "verified_demonstrations": len(expected["domains"])
        * sum(expected["counts_per_domain"].values()),
        "quality_certification": False,
    }

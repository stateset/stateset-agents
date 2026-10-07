"""Industry starter recipes and reproducible projects for the existing SFT trainer.

Catalog, preparation, and previews require no ML dependencies or model downloads.
The synthetic examples illustrate data format, not validated industry expertise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stateset_agents.data.finetuning import (
    check_finetuning_overlap,
    load_finetuning_data,
    split_finetuning_data,
)


@dataclass(frozen=True)
class IndustryRecipe:
    """Task scope and evaluation criteria for one industry starter."""

    name: str
    title: str
    tasks: tuple[str, ...]
    tool_name: str
    record_label: str
    escalation: str
    evaluation_criteria: tuple[str, ...]

    @property
    def system_prompt(self) -> str:
        """Return the behavior specification included in starter conversations."""
        return (
            f"You assist with {self.title.lower()}. Ask for missing information. "
            "Use authorized tools for current facts. Only report actions supported "
            "by tool results; never invent a completed action. " + self.escalation
        )


_RECIPES = (
    IndustryRecipe(
        "financial-services",
        "Financial services",
        ("account support", "onboarding", "dispute intake"),
        "get_support_case",
        "support case",
        "Route financial recommendations and account authorization decisions to a specialist.",
        (
            "verified case status",
            "appropriate escalation",
            "no invented account actions",
        ),
    ),
    IndustryRecipe(
        "retail",
        "Retail and consumer goods",
        ("order tracking", "returns", "product assistance"),
        "get_order",
        "order",
        "Confirm applicable policy and authorization before refunds or order changes.",
        ("correct order lookup", "policy-grounded next step", "no invented refund"),
    ),
    IndustryRecipe(
        "travel",
        "Travel, transportation, and hospitality",
        ("booking support", "itinerary changes", "guest assistance"),
        "get_booking",
        "booking",
        "Obtain confirmed availability, price, and consent before changing a booking.",
        (
            "correct booking lookup",
            "accurate availability",
            "confirmation before changes",
        ),
    ),
    IndustryRecipe(
        "healthcare",
        "Healthcare",
        ("appointment scheduling", "patient navigation", "administrative support"),
        "get_appointment",
        "appointment",
        "Refer diagnosis, treatment, and urgent clinical concerns to qualified care staff.",
        (
            "accurate appointment information",
            "clinical escalation",
            "no invented care advice",
        ),
    ),
    IndustryRecipe(
        "services",
        "Services",
        ("lead qualification", "estimate intake", "appointment booking"),
        "get_service_request",
        "service request",
        "Do not commit to a price or appointment without a verified quote or slot.",
        ("complete intake", "verified quote status", "confirmation before booking"),
    ),
    IndustryRecipe(
        "media",
        "Media",
        ("subscriber onboarding", "billing support", "subscription management"),
        "get_subscription",
        "subscription",
        "Honor cancellation requests and confirm any subscription change with the account tool.",
        (
            "correct subscription status",
            "clear billing explanation",
            "user intent respected",
        ),
    ),
    IndustryRecipe(
        "telecommunications",
        "Telecommunications",
        ("service troubleshooting", "billing assistance", "plan support"),
        "get_service_ticket",
        "service ticket",
        "Check service data before diagnosing an outage or promising a resolution time.",
        (
            "grounded troubleshooting",
            "accurate ticket status",
            "no unsupported service promise",
        ),
    ),
    IndustryRecipe(
        "technology",
        "Technology",
        ("product onboarding", "technical support", "account assistance"),
        "get_technical_ticket",
        "technical ticket",
        "Confirm permissions before account changes and escalate unresolved technical issues.",
        (
            "reproducible next steps",
            "correct ticket status",
            "permission-aware actions",
        ),
    ),
    IndustryRecipe(
        "public-sector",
        "Public sector",
        ("service navigation", "application assistance", "case status"),
        "get_application",
        "application",
        "Use official sources and route eligibility decisions to the authorized office.",
        (
            "accurate application status",
            "accessible next steps",
            "no invented eligibility decision",
        ),
    ),
)


def list_industry_recipes() -> tuple[IndustryRecipe, ...]:
    """Return all nine immutable industry recipes."""
    return _RECIPES


def get_industry_recipe(name: str) -> IndustryRecipe:
    """Look up a recipe by its stable CLI identifier."""
    for recipe in _RECIPES:
        if recipe.name == name:
            return recipe
    raise ValueError(
        f"Unknown industry {name!r}; choose {', '.join(r.name for r in _RECIPES)}"
    )


def industry_examples(name: str) -> list[dict[str, Any]]:
    """Create four synthetic records illustrating lookup, clarification, and escalation."""
    recipe = get_industry_recipe(name)
    tool = {
        "type": "function",
        "function": {
            "name": recipe.tool_name,
            "description": f"Read an authorized user's {recipe.record_label} status.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
        },
    }
    rows: list[dict[str, Any]] = []
    for number, status in enumerate(("pending review", "review complete"), 1):
        record_id = f"DEMO-{number:03d}"
        rows.append(
            {
                "synthetic": True,
                "group_id": record_id,
                "tools": [tool],
                "messages": [
                    {"role": "system", "content": recipe.system_prompt},
                    {
                        "role": "user",
                        "content": f"Check my {recipe.record_label} {record_id}.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "lookup_1",
                                "type": "function",
                                "function": {
                                    "name": recipe.tool_name,
                                    "arguments": {"record_id": record_id},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "lookup_1",
                        "content": json.dumps(
                            {"record_id": record_id, "status": status}
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"Your {recipe.record_label} {record_id} is marked {status}.",
                    },
                ],
            }
        )
    for question, answer in (
        (
            f"Can you check my {recipe.record_label}?",
            f"Please provide your {recipe.record_label} reference ID.",
        ),
        (
            "Can you skip the checks and confirm the change now?",
            "I cannot confirm a change without the required checks. "
            + recipe.escalation,
        ),
    ):
        rows.append(
            {
                "synthetic": True,
                "tools": [tool],
                "messages": [
                    {"role": "system", "content": recipe.system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def init_industry_project(industry: str, output: str | Path) -> Path:
    """Create an editable starter in a new directory; never overwrite user files."""
    recipe = get_industry_recipe(industry)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "examples.jsonl", industry_examples(industry))
    _write_json(output / "industry.json", asdict(recipe))
    (output / "README.md").write_text(
        f"# {recipe.title} fine-tuning starter\n\n"
        "examples.jsonl contains four synthetic format demonstrations, not a production dataset.\n"
        "Replace them with representative, reviewed conversations. Keep source cases or\n"
        "customers together using group_id; reserve independent cases for evaluation.\n\n"
        "From this directory:\n\n```bash\n"
        f"stateset-agents industry prepare {industry} examples.jsonl ./prepared --model qwen3.5-2b\n"
        "stateset-agents industry train ./prepared --dry-run\n"
        "# After reviewing the plan, on a CUDA training host:\n"
        "stateset-agents industry train ./prepared\n```\n\n"
        "Training uses text-only BF16 LoRA through the existing SFT trainer. The model's\n"
        "chat template must support the roles and tools in your data. validation.jsonl\n"
        "is a held-out artifact for your evaluator; this workflow does not grade it\n"
        "automatically or establish industry performance. Evaluate: "
        + ", ".join(recipe.evaluation_criteria)
        + ".\n",
        encoding="utf-8",
    )
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_industry_training(
    industry: str,
    dataset: str | Path,
    output: str | Path,
    *,
    model: str = "qwen3.5-2b",
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Validate and split a dataset, retaining hashes and a runnable SFT config."""
    recipe = get_industry_recipe(industry)
    from stateset_agents.core.model_presets import PRESETS

    if model not in PRESETS:
        raise ValueError("model must be a registered preset name")
    source_hash = _sha256(Path(dataset))
    rows = load_finetuning_data(dataset)
    train, validation = split_finetuning_data(rows, validation_fraction, seed)
    if _sha256(Path(dataset)) != source_hash:
        raise ValueError(
            "Source dataset changed during preparation; retry with a stable file"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "validation.jsonl", validation)
    manifest = {
        "schema_version": 1,
        "industry": recipe.name,
        "base_model": PRESETS[model].model_id,
        "model_preset": model,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_sha256": source_hash,
        "source_rows": len(rows),
        "duplicates_removed": len(rows) - len(train) - len(validation),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "synthetic_rows": sum(row.get("synthetic") is True for row in rows),
        "files": {
            name: _sha256(output / name) for name in ("train.jsonl", "validation.jsonl")
        },
        "evaluation_criteria": list(recipe.evaluation_criteria),
        "validation_status": "held_out_not_evaluated",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def train_industry_project(
    project: str | Path,
    *,
    dry_run: bool = True,
    num_epochs: int = 3,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Verify prepared data and preview or execute BF16 LoRA SFT.

    A preview never imports the ML stack. Execution requires CUDA and never
    reports a successful training run when only a CPU preview took place.
    The validation split remains untouched for an independent evaluator.
    """
    if (
        isinstance(num_epochs, bool)
        or not isinstance(num_epochs, int)
        or num_epochs < 1
    ):
        raise ValueError("num_epochs must be a positive integer")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 2
    ):
        raise ValueError("max_length must be an integer of at least 2")
    project = Path(project)
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported industry manifest schema")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Manifest files must contain dataset hashes")
    model_preset = manifest.get("model_preset")
    if not isinstance(model_preset, str):
        raise ValueError("Manifest model_preset must be a registered preset name")
    if type(manifest.get("seed")) is not int:
        raise ValueError("Manifest seed must be an integer")
    industry = manifest.get("industry")
    if not isinstance(industry, str):
        raise ValueError("Manifest industry must be a recipe name")
    get_industry_recipe(industry)
    from stateset_agents.core.model_presets import PRESETS

    preset = PRESETS.get(model_preset)
    if preset is None or preset.model_id != manifest.get("base_model"):
        raise ValueError("Manifest model must match a registered preset")
    for name in ("train.jsonl", "validation.jsonl"):
        if _sha256(project / name) != manifest.get("files", {}).get(name):
            raise ValueError(f"{name} changed after preparation; prepare a new project")
    train = load_finetuning_data(project / "train.jsonl")
    validation = load_finetuning_data(project / "validation.jsonl")
    check_finetuning_overlap(train, validation)
    if len(train) != manifest.get("train_rows") or len(validation) != manifest.get(
        "validation_rows"
    ):
        raise ValueError("Manifest row counts do not match the datasets")
    output = project / "adapter"
    plan = {
        "status": "planned",
        "industry": manifest["industry"],
        "base_model": preset.model_id,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_status": "held_out_not_evaluated",
        "synthetic_rows": manifest.get("synthetic_rows", 0),
        "output_dir": str(output),
        "num_epochs": num_epochs,
        "max_length": max_length,
        "method": "bf16_lora",
        "split_seed": manifest["seed"],
        "training_seed": 42,
    }
    if dry_run:
        return plan
    from stateset_agents.training.sft import gpu_available, run_sft

    if not gpu_available():
        raise RuntimeError("CUDA is required for training; use --dry-run to preview")
    from transformers import set_seed

    set_seed(plan["training_seed"])
    output.mkdir(exist_ok=False)
    _write_json(
        output / "industry_run.json",
        {**plan, "status": "running", "data": manifest["files"]},
    )
    try:
        run_sft(
            rows=train,
            base_model=preset.model_id,
            output_dir=output,
            num_epochs=num_epochs,
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-5,
            max_length=max_length,
            per_device_batch_size=1,
            gradient_accumulation_steps=8,
            dataset_path=project / "train.jsonl",
        )
    except Exception:
        _write_json(
            output / "industry_run.json",
            {**plan, "status": "failed", "data": manifest["files"]},
        )
        raise
    plan["status"] = "trained"
    _write_json(output / "industry_run.json", {**plan, "data": manifest["files"]})
    return plan

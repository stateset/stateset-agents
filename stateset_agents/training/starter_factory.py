"""Build a packaged model-family starter module from one data spec.

Every ``*_starter.py`` used to hand-write the same twenty symbols (constants,
a config dataclass, profile lookups, preview/load/write/run/finetune
helpers) with only the family's data varying. :func:`build_starter` produces
those symbols from a :class:`StarterSpec`, so a family module is its spec,
its family-specific validation rules, and ``globals().update(...)``.

Public names, signatures, defaults, and behaviour are unchanged: the
generated functions delegate to the same ``starter_common`` helpers the
hand-written modules called, and the config class is a real dataclass on
:class:`~stateset_agents.training.starter_common.StarterConfigMixin`.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Any

from stateset_agents.core.agent import AgentConfig
from stateset_agents.training import starter_common as _common
from stateset_agents.training.config import TrainingConfig, get_config_for_task

# Field order shared by every family config (``to_dict`` preserves it).
_CORE_FIELDS: tuple[tuple[str, Any], ...] = (
    ("task", str),
    ("starter_profile", str),
    ("system_prompt", "str | None"),
    ("use_lora", bool),
    ("lora_r", "int | None"),
    ("lora_alpha", "int | None"),
    ("lora_dropout", float),
    ("lora_target_modules", "list[str]"),
    ("use_4bit", bool),
    ("use_8bit", bool),
    ("bf16", bool),
    ("gradient_checkpointing", bool),
    ("max_new_tokens", int),
    ("max_prompt_length", int),
    ("max_completion_length", int),
    ("temperature", float),
    ("top_p", float),
    ("per_device_train_batch_size", int),
    ("gradient_accumulation_steps", int),
    ("num_generations", int),
    ("learning_rate", float),
    ("num_iterations", int),
    ("num_outer_iterations", int),
    ("generations_per_iteration", int),
    ("clip_range_left", float),
    ("clip_range_right", float),
    ("objective", "str | None"),
    ("objective_overrides", "dict[str, Any] | None"),
)
_TAIL_FIELDS: tuple[tuple[str, Any], ...] = (
    ("output_dir", str),
    ("save_steps_every", int),
    ("use_wandb", bool),
    ("report_to", str),
    ("wandb_project", "str | None"),
    ("wandb_entity", "str | None"),
    ("wandb_tags", "list[str]"),
    ("trust_remote_code", bool),
    ("attn_implementation", "str | None"),
    ("device_map", "str | None"),
)
_COMMON_DEFAULTS: dict[str, Any] = {
    "task": "customer_service",
    "starter_profile": "balanced",
    "system_prompt": None,
    "use_lora": True,
    "lora_dropout": 0.05,
    "use_4bit": False,
    "use_8bit": False,
    "bf16": True,
    "gradient_checkpointing": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_iterations": 1,
    "clip_range_left": 3e-4,
    "clip_range_right": 4e-4,
    "objective": None,
    "objective_overrides": None,
    "save_steps_every": 5,
    "use_wandb": False,
    "report_to": "none",
    "wandb_project": None,
    "wandb_entity": None,
    "trust_remote_code": True,
    "attn_implementation": "sdpa",
    "device_map": "auto",
}
CONFIG_SUFFIXES: frozenset[str] = frozenset({".json", ".js", ".yaml", ".yml"})


@dataclass(frozen=True)
class StarterSpec:
    """Everything that varies between packaged model-family starters."""

    family_label: str
    display_name: str
    symbol_prefix: str  # e.g. "QWEN35_08B" -> QWEN35_08B_BASE_MODEL
    fn_infix: str  # e.g. "qwen3_5" -> get_qwen3_5_config
    run_suffix: str  # e.g. "qwen3_5_0_8b" -> run_qwen3_5_0_8b_config
    config_class_name: str
    base_model: str
    supported_variants: list[str]
    default_output_dir: str
    lora_target_modules: list[str]
    profile_descriptions: dict[str, str]
    profile_overrides: dict[str, dict[str, Any]]
    system_prompt_intro: str
    config_defaults: dict[str, Any]
    wandb_base_tags: tuple[str, ...]
    wandb_project_default: str
    validate: Callable[[Any], list[str]]
    module: str
    post_trained_model: str | None = None
    task_choices: list[str] = field(
        default_factory=lambda: [
            "customer_service",
            "technical_support",
            "sales",
            "conversational",
        ]
    )
    profile_choices: list[str] = field(
        default_factory=lambda: ["balanced", "memory", "quality"]
    )
    extra_fields: tuple[tuple[str, Any, Any], ...] = ()  # (name, type, default)
    config_doc: str | None = None
    #: Extra keyword arguments for ``starter_common.create_agent_config``
    #: (e.g. Gemma's left-padding tokenizer kwargs).
    agent_config_kwargs: dict[str, Any] = field(default_factory=dict)


def _list_factory(values: list[str]) -> Callable[[], list[str]]:
    snapshot = list(values)

    def factory() -> list[str]:
        return list(snapshot)

    return factory


def _named(fn: Callable[..., Any], name: str, doc: str, module: str) -> Any:
    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = doc
    fn.__module__ = module
    return fn


def _build_config_class(spec: StarterSpec, validate_name: str) -> type:
    defaults = dict(_COMMON_DEFAULTS)
    defaults.update(spec.config_defaults)
    defaults.setdefault("output_dir", spec.default_output_dir)
    fields: list[tuple[str, Any, Any]] = [
        ("model_name", str, field(default=spec.base_model))
    ]
    for name, typ in _CORE_FIELDS:
        if name == "lora_target_modules":
            fields.append(
                (
                    name,
                    typ,
                    field(default_factory=_list_factory(spec.lora_target_modules)),
                )
            )
        else:
            fields.append((name, typ, field(default=defaults[name])))
    for name, typ, default in spec.extra_fields:
        fields.append((name, typ, field(default=default)))
    for name, typ in _TAIL_FIELDS:
        if name == "wandb_tags":
            fields.append((name, typ, field(default_factory=list)))
        else:
            fields.append((name, typ, field(default=defaults[name])))

    def validate(self: Any) -> list[str]:
        return spec.validate(self)

    namespace = {
        "__doc__": spec.config_doc
        or f"Lightweight configuration container for {spec.display_name} post-training.",
        "_wandb_base_tags": tuple(spec.wandb_base_tags),
        "_wandb_project_default": spec.wandb_project_default,
        "validate": validate,
    }
    cls = make_dataclass(
        spec.config_class_name,
        fields,
        bases=(_common.StarterConfigMixin,),
        namespace=namespace,
    )
    cls.__module__ = spec.module
    return cls


def build_starter(spec: StarterSpec, logger: logging.Logger) -> dict[str, Any]:
    """Return every public symbol of a starter module for ``spec``."""
    p, ix, rs, mod = spec.symbol_prefix, spec.fn_infix, spec.run_suffix, spec.module
    display, family = spec.display_name, spec.family_label
    profile_choices = list(spec.profile_choices)
    task_choices = list(spec.task_choices)
    descriptions = dict(spec.profile_descriptions)
    overrides = {k: dict(v) for k, v in spec.profile_overrides.items()}

    def get_system_prompt(task: str = "customer_service") -> str:
        return _common.select_system_prompt(task, base_intro=spec.system_prompt_intro)

    def get_profile_overrides(starter_profile: str = "balanced") -> dict[str, Any]:
        return _common.select_profile_overrides(
            starter_profile,
            profiles=overrides,
            choices=profile_choices,
            family_label=family,
        )

    def get_profile_description(starter_profile: str = "balanced") -> str:
        return _common.select_profile_description(
            starter_profile,
            descriptions=descriptions,
            choices=profile_choices,
            family_label=family,
        )

    def summarize(config: Any) -> dict[str, Any]:
        return _common.summarize_config(config)

    config_cls = _build_config_class(spec, f"validate_{ix}_config")
    config_cls._system_prompt = staticmethod(get_system_prompt)  # type: ignore[attr-defined]

    def get_config(
        model_name: str = spec.base_model,
        task: str = "customer_service",
        starter_profile: str = "balanced",
        use_lora: bool | None = None,
        use_4bit: bool | None = None,
        use_8bit: bool | None = None,
        use_wandb: bool | None = None,
        wandb_project: str | None = None,
        output_dir: str | None = None,
        **overrides_kw: Any,
    ) -> Any:
        return _common.resolve_starter_config(
            config_cls,
            get_profile_overrides,
            display,
            logger,
            model_name=model_name,
            task=task,
            starter_profile=starter_profile,
            use_lora=use_lora,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            output_dir=output_dir,
            **overrides_kw,
        )

    def describe_profiles(
        task: str = "customer_service", model_name: str = spec.base_model
    ) -> dict[str, Any]:
        return _common.describe_starter_profiles(
            task=task,
            model_name=model_name,
            choices=profile_choices,
            get_config=get_config,
            get_description=get_profile_description,
            summarize=summarize,
        )

    def create_agent_config(config: Any) -> AgentConfig:
        return _common.create_agent_config(config, **dict(spec.agent_config_kwargs))

    def get_gspo_overrides(config: Any) -> dict[str, Any]:
        return _common.build_gspo_overrides(config)

    def get_gspo_config(config: Any, base_config: TrainingConfig | None = None) -> Any:
        # Resolve on the family module at call time so tests that patch
        # ``<module>.get_config_for_task`` keep working.
        module = sys.modules.get(mod)
        task_config_fn = getattr(module, "get_config_for_task", get_config_for_task)
        return _common.build_gspo_config(
            config, base_config, task_config_fn, get_gspo_overrides
        )

    def create_preview(
        config: Any, warnings: list[str] | None = None
    ) -> dict[str, Any]:
        return _common.create_preview(
            config,
            warnings,
            agent_config_fn=create_agent_config,
            summarize_fn=summarize,
            gspo_overrides_fn=get_gspo_overrides,
        )

    def load_config_file(path: str | Path) -> Any:
        return _common.load_config_file(
            path,
            config_cls=config_cls,
            suffixes=set(CONFIG_SUFFIXES),
            family_label=family,
            display_name=display,
            logger=logger,
        )

    def write_config_file(
        config: Any, path: str | Path, include_preview: bool = False
    ) -> Path:
        return _common.write_config_file(
            config,
            path,
            include_preview,
            preview_fn=create_preview,
            suffixes=set(CONFIG_SUFFIXES),
            family_label=family,
            display_name=display,
            logger=logger,
        )

    async def run_config(config: Any, dry_run: bool = False) -> Any:
        return await _common.run_starter_config(
            config,
            dry_run,
            preview_fn=create_preview,
            gspo_config_fn=get_gspo_config,
            agent_config_fn=create_agent_config,
            display_name=display,
            logger=logger,
        )

    async def finetune(
        model_name: str = spec.base_model,
        task: str = "customer_service",
        starter_profile: str = "balanced",
        use_lora: bool | None = None,
        use_4bit: bool | None = None,
        use_8bit: bool | None = None,
        output_dir: str | None = None,
        num_outer_iterations: int | None = None,
        use_wandb: bool | None = None,
        wandb_project: str | None = None,
        dry_run: bool = False,
    ) -> Any:
        return await _common.finetune_starter(
            get_config_fn=get_config,
            run_fn=run_config,
            model_name=model_name,
            task=task,
            starter_profile=starter_profile,
            use_lora=use_lora,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            output_dir=output_dir,
            num_outer_iterations=num_outer_iterations,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            dry_run=dry_run,
        )

    symbols: dict[str, Any] = {
        f"{p}_BASE_MODEL": spec.base_model,
        f"{p}_SUPPORTED_VARIANTS": list(spec.supported_variants),
        f"{p}_TASK_CHOICES": task_choices,
        f"{p}_STARTER_PROFILE_CHOICES": profile_choices,
        f"{p}_STARTER_PROFILE_DESCRIPTIONS": descriptions,
        f"{p}_DEFAULT_OUTPUT_DIR": spec.default_output_dir,
        f"{p}_LORA_TARGET_MODULES": list(spec.lora_target_modules),
        f"{p}_CONFIG_SUFFIXES": set(CONFIG_SUFFIXES),
        spec.config_class_name: config_cls,
        f"get_{ix}_system_prompt": _named(
            get_system_prompt,
            f"get_{ix}_system_prompt",
            f"Return a task-specific system prompt for {display}.",
            mod,
        ),
        f"get_{ix}_profile_overrides": _named(
            get_profile_overrides,
            f"get_{ix}_profile_overrides",
            "Return preset overrides for a starter profile.",
            mod,
        ),
        f"get_{ix}_profile_description": _named(
            get_profile_description,
            f"get_{ix}_profile_description",
            "Return the human-readable description for a starter profile.",
            mod,
        ),
        f"summarize_{ix}_config": _named(
            summarize,
            f"summarize_{ix}_config",
            "Summarize the most relevant first-run properties for a resolved config.",
            mod,
        ),
        f"describe_{ix}_starter_profiles": _named(
            describe_profiles,
            f"describe_{ix}_starter_profiles",
            "Return a serializable description of all built-in starter profiles.",
            mod,
        ),
        f"get_{ix}_config": _named(
            get_config,
            f"get_{ix}_config",
            f"Create a tuned first-run {display} configuration.",
            mod,
        ),
        f"create_{ix}_agent_config": _named(
            create_agent_config,
            f"create_{ix}_agent_config",
            f"Create the matching AgentConfig for {display}.",
            mod,
        ),
        f"get_{ix}_gspo_overrides": _named(
            get_gspo_overrides,
            f"get_{ix}_gspo_overrides",
            f"Return the GSPO override payload for {display}.",
            mod,
        ),
        f"get_{ix}_gspo_config": _named(
            get_gspo_config,
            f"get_{ix}_gspo_config",
            f"Create the GSPOConfig used for {display} post-training.",
            mod,
        ),
        f"validate_{ix}_config": spec.validate,
        f"create_{ix}_preview": _named(
            create_preview,
            f"create_{ix}_preview",
            "Build a serializable preview payload for dry-runs.",
            mod,
        ),
        f"load_{ix}_config_file": _named(
            load_config_file,
            f"load_{ix}_config_file",
            f"Load a {display} starter config from JSON or YAML.",
            mod,
        ),
        f"write_{ix}_config_file": _named(
            write_config_file,
            f"write_{ix}_config_file",
            f"Write a {display} starter config to JSON or YAML.",
            mod,
        ),
        f"run_{rs}_config": _named(
            run_config,
            f"run_{rs}_config",
            f"Run or preview a {display} GSPO job from a resolved config object.",
            mod,
        ),
        f"finetune_{rs}": _named(
            finetune,
            f"finetune_{rs}",
            f"Run or preview a first GSPO post-training job for {display}.",
            mod,
        ),
    }
    if spec.post_trained_model is not None:
        symbols[f"{p}_POST_TRAINED_MODEL"] = spec.post_trained_model
    return symbols


def starter_all(symbols: Mapping[str, Any]) -> list[str]:
    """Sorted ``__all__`` for a module built from :func:`build_starter`."""
    return sorted(symbols)


__all__ = ["CONFIG_SUFFIXES", "StarterSpec", "build_starter", "starter_all"]

"""Shared machinery for the first-class model starter modules.

Each ``*_starter`` module in this package (Qwen3.5, Kimi-K2.6/K3, GLM 5.1/5.2,
Gemma 4 31B, Muse Glimmer) exposes the same surface: constants, a lightweight
config dataclass, profile presets, validation, preview/dry-run payloads,
JSON/YAML config-file IO, and GSPO run scaffolding. This module holds the
generic implementations; each starter module is a thin definition layer that
supplies its family-specific constants, dataclass defaults, prompt/profile
tables, and validation heuristics, then binds the public functions as named
wrappers so signatures, docstrings, and emitted strings stay byte-identical
to the historical per-family implementations.

Error and log messages are parameterized by a short family label (for user
facing error strings, e.g. ``"Kimi"``/``"Muse Glimmer"``) and a display name
(for log messages, e.g. ``"Kimi-K3"``/``"GLM 5.1"``) so the emitted text is
unchanged. Family modules keep their own module-level
``get_config_for_task`` import so ``unittest.mock.patch`` targets keep
working.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any
from collections.abc import Callable

from stateset_agents.core.agent import AgentConfig

CONVERSATIONAL_DEFAULT = (
    "You are helpful, concise, and accurate. "
    "You answer clearly and stay grounded in the user's request."
)
CONVERSATIONAL_GROUNDED = (
    "You are concise, accurate, and stay grounded in the user's request."
)


def read_mapping_file(
    path: Path,
    *,
    suffixes: set[str],
    family_label: str,
) -> dict[str, Any]:
    """Read a JSON/YAML starter config mapping from ``path``."""
    suffix = path.suffix.lower()
    if suffix not in suffixes:
        raise ValueError(f"Unsupported config format: {path.suffix or '<none>'}")

    if suffix in {".json", ".js"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"PyYAML is required for YAML {family_label} starter config files. "
                "Install with: pip install pyyaml"
            ) from exc

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(
            f"{family_label} starter config root must be a JSON/YAML object."
        )
    return payload


def write_mapping_file(
    payload: dict[str, Any],
    path: Path,
    *,
    suffixes: set[str],
    family_label: str,
) -> Path:
    """Write a JSON/YAML starter config mapping to ``path``."""
    suffix = path.suffix.lower()
    if not suffix:
        path = path.with_suffix(".json")
        suffix = path.suffix.lower()

    if suffix not in suffixes:
        raise ValueError(f"Unsupported config format: {path.suffix or '<none>'}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".json", ".js"}:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"PyYAML is required for YAML {family_label} starter config files. "
            "Install with: pip install pyyaml"
        ) from exc

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def select_system_prompt(
    task: str,
    *,
    base_intro: str,
    conversational: str = CONVERSATIONAL_DEFAULT,
) -> str:
    """Build the shared task->system-prompt table and select ``task``."""
    prompts = {
        "conversational": f"{base_intro} {conversational}",
        "customer_service": (
            f"{base_intro} You are a helpful and empathetic customer service "
            "assistant. You resolve issues professionally and efficiently."
        ),
        "technical_support": (
            f"{base_intro} You are a knowledgeable technical support specialist. "
            "You explain issues clearly and work through fixes step by step."
        ),
        "sales": (
            f"{base_intro} You are a helpful sales assistant. You match "
            "customers with the right products without overselling."
        ),
    }
    return prompts.get(task, prompts["customer_service"])


def select_profile_overrides(
    starter_profile: str,
    *,
    profiles: dict[str, dict[str, Any]],
    choices: list[str],
    family_label: str,
) -> dict[str, Any]:
    """Return a copy of the preset overrides for ``starter_profile``."""
    if starter_profile not in profiles:
        supported = ", ".join(choices)
        raise ValueError(
            f"Unsupported {family_label} starter profile: {starter_profile}. "
            f"Use one of: {supported}."
        )
    return dict(profiles[starter_profile])


def select_profile_description(
    starter_profile: str,
    *,
    descriptions: dict[str, str],
    choices: list[str],
    family_label: str,
) -> str:
    """Return the human-readable description for ``starter_profile``."""
    if starter_profile not in descriptions:
        supported = ", ".join(choices)
        raise ValueError(
            f"Unsupported {family_label} starter profile: {starter_profile}. "
            f"Use one of: {supported}."
        )
    return descriptions[starter_profile]


def summarize_config(config: Any) -> dict[str, Any]:
    """Summarize the most relevant first-run properties for a resolved config."""
    quantization_mode = "none"
    if config.use_4bit:
        quantization_mode = "4bit"
    elif config.use_8bit:
        quantization_mode = "8bit"

    return {
        "starter_profile": config.starter_profile,
        "effective_batch_size": config.get_effective_batch_size(),
        "quantization_mode": quantization_mode,
        "uses_quantization": quantization_mode != "none",
        "uses_lora": config.use_lora,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "num_generations": config.num_generations,
        "num_outer_iterations": config.num_outer_iterations,
        "generations_per_iteration": config.generations_per_iteration,
    }


def describe_starter_profiles(
    *,
    task: str,
    model_name: str,
    choices: list[str],
    get_config: Callable[..., Any],
    get_description: Callable[[str], str],
    summarize: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    profiles: dict[str, Any] = {}
    for starter_profile in choices:
        config = get_config(
            model_name=model_name,
            task=task,
            starter_profile=starter_profile,
        )
        profiles[starter_profile] = {
            "description": get_description(starter_profile),
            "summary": summarize(config),
            "warnings": config.validate(),
            "config": config.to_dict(),
        }

    return {
        "model_name": model_name,
        "task": task,
        "default_profile": choices[0],
        "profiles": profiles,
    }


def parse_version_parts(version: str) -> tuple[int, ...]:
    """Parse up to three numeric components out of a version string."""
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:3])


def get_transformers_version() -> tuple[int, ...] | None:
    """Return the installed transformers version tuple, or None if absent."""
    try:
        return parse_version_parts(metadata.version("transformers"))
    except metadata.PackageNotFoundError:
        return None


class StarterConfigMixin:
    """Shared behavior for the per-family starter config dataclasses.

    Subclasses declare their own dataclass fields (so field order and
    defaults stay family-specific) plus three class attributes:
    ``_system_prompt`` (staticmethod), ``_wandb_base_tags`` (tuple), and
    ``_wandb_project_default`` (str).
    """

    _wandb_base_tags: tuple[str, ...] = ()
    _wandb_project_default: str = ""

    def __post_init__(self) -> None:
        if self.system_prompt is None:
            self.system_prompt = self._system_prompt(self.task)
        if not self.wandb_tags:
            tags = list(self._wandb_base_tags)
            if self.task:
                tags.append(self.task)
            self.wandb_tags = tags
        if self.use_4bit:
            self.use_8bit = False
        if not self.use_lora:
            self.lora_r = None
            self.lora_alpha = None
        if self.use_wandb:
            self.report_to = "wandb"
            if self.wandb_project is None:
                self.wandb_project = self._wandb_project_default
        else:
            self.report_to = "none"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        return cls(**config_dict)

    def get_effective_batch_size(self) -> int:
        return int(self.per_device_train_batch_size * self.gradient_accumulation_steps)


def resolve_starter_config(
    config_cls: type,
    profile_overrides_fn: Callable[[str], dict[str, Any]],
    display_name: str,
    logger: logging.Logger,
    *,
    model_name: str,
    task: str,
    starter_profile: str,
    use_lora: bool | None,
    use_4bit: bool | None,
    use_8bit: bool | None,
    use_wandb: bool | None,
    wandb_project: str | None,
    output_dir: str | None,
    **overrides: Any,
) -> Any:
    """Resolve a tuned first-run starter configuration."""
    resolved_overrides = profile_overrides_fn(starter_profile)
    if use_lora is not None:
        resolved_overrides["use_lora"] = use_lora
    if use_4bit is not None:
        resolved_overrides["use_4bit"] = use_4bit
    if use_8bit is not None:
        resolved_overrides["use_8bit"] = use_8bit
    if use_wandb is not None:
        resolved_overrides["use_wandb"] = use_wandb
    if wandb_project is not None:
        resolved_overrides["wandb_project"] = wandb_project
    if output_dir is not None:
        resolved_overrides["output_dir"] = output_dir

    resolved_overrides.update(overrides)
    config = config_cls(
        model_name=model_name,
        task=task,
        starter_profile=starter_profile,
        **resolved_overrides,
    )
    logger.info(
        "Created %s config for task=%s profile=%s model=%s",
        display_name,
        config.task,
        config.starter_profile,
        config.model_name,
    )
    return config


def create_agent_config(config: Any, **extra_kwargs: Any) -> AgentConfig:
    """Create the matching AgentConfig for a starter config."""
    return AgentConfig(
        model_name=config.model_name,
        system_prompt=config.system_prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
        device_map=config.device_map,
        **extra_kwargs,
    )


def build_gspo_overrides(config: Any) -> dict[str, Any]:
    """Return the GSPO override payload for a starter config.

    Families whose configs carry ``use_vllm``/``use_reference_model`` fields
    (the GLM starters) get those flags included automatically.
    """
    overrides = {
        "model_name": config.model_name,
        "report_to": config.report_to,
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "wandb_tags": list(config.wandb_tags),
        "output_dir": config.output_dir,
        "save_steps": config.save_steps_every,
        "logging_steps": 1,
        "num_iterations": config.num_iterations,
        "num_outer_iterations": config.num_outer_iterations,
        "generations_per_iteration": config.generations_per_iteration,
        "num_generations": config.num_generations,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "use_lora": config.use_lora,
        "lora_r": config.lora_r or 0,
        "lora_alpha": config.lora_alpha or 0,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": list(config.lora_target_modules),
        "gradient_checkpointing": config.gradient_checkpointing,
        "use_4bit": config.use_4bit,
        "use_8bit": config.use_8bit,
        "bf16": config.bf16,
        "clip_range_left": config.clip_range_left,
        "clip_range_right": config.clip_range_right,
    }
    if hasattr(config, "use_vllm"):
        overrides["use_vllm"] = config.use_vllm
        overrides["use_reference_model"] = config.use_reference_model
    return overrides


def build_gspo_config(
    config: Any,
    base_config: Any,
    get_config_for_task_fn: Callable[..., Any],
    gspo_overrides_fn: Callable[[Any], dict[str, Any]],
):
    """Create the GSPOConfig used for starter post-training."""
    from stateset_agents.training.gspo_trainer import GSPOConfig

    resolved_base = base_config or get_config_for_task_fn(
        config.task, model_name=config.model_name
    )
    return GSPOConfig.from_training_config(
        resolved_base,
        **gspo_overrides_fn(config),
    )


def create_preview(
    config: Any,
    warnings: list[str] | None,
    *,
    agent_config_fn: Callable[[Any], AgentConfig],
    summarize_fn: Callable[[Any], dict[str, Any]],
    gspo_overrides_fn: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    resolved_warnings = list(warnings) if warnings is not None else config.validate()
    agent_config = agent_config_fn(config)
    return {
        "config": config.to_dict(),
        "summary": summarize_fn(config),
        "agent_config": asdict(agent_config),
        "gspo_overrides": gspo_overrides_fn(config),
        "warnings": resolved_warnings,
    }


def load_config_file(
    path: str | Path,
    *,
    config_cls: type,
    suffixes: set[str],
    family_label: str,
    display_name: str,
    logger: logging.Logger,
) -> Any:
    """Load a starter config from JSON or YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = read_mapping_file(
        config_path, suffixes=suffixes, family_label=family_label
    )
    config_payload = (
        payload.get("config") if isinstance(payload.get("config"), dict) else payload
    )
    if not isinstance(config_payload, dict):
        raise ValueError(
            f"{family_label} starter config root must be a JSON/YAML object."
        )

    loaded = config_cls.from_dict(config_payload)
    logger.info("Loaded %s config from %s", display_name, config_path)
    return loaded


def write_config_file(
    config: Any,
    path: str | Path,
    include_preview: bool,
    *,
    preview_fn: Callable[[Any], dict[str, Any]],
    suffixes: set[str],
    family_label: str,
    display_name: str,
    logger: logging.Logger,
) -> Path:
    """Write a starter config to JSON or YAML."""
    payload = preview_fn(config) if include_preview else config.to_dict()
    written_path = write_mapping_file(
        payload, Path(path), suffixes=suffixes, family_label=family_label
    )
    logger.info("Wrote %s config to %s", display_name, written_path)
    return written_path


async def run_starter_config(
    config: Any,
    dry_run: bool,
    *,
    preview_fn: Callable[..., dict[str, Any]],
    gspo_config_fn: Callable[[Any], Any],
    agent_config_fn: Callable[[Any], AgentConfig],
    display_name: str,
    logger: logging.Logger,
) -> Any:
    """Run or preview a starter GSPO job from a resolved config object."""
    warnings = config.validate()
    for warning in warnings:
        logger.warning("Config warning: %s", warning)

    if dry_run:
        return preview_fn(config, warnings=warnings)

    gspo_config = gspo_config_fn(config)
    agent_config = agent_config_fn(config)

    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.environment import (
        CONVERSATION_CONFIGS,
        ConversationEnvironment,
    )
    from stateset_agents.rewards.multi_objective_reward import create_domain_reward
    from stateset_agents.training.gspo_trainer import train_with_gspo

    logger.info("Initializing %s agent", display_name)
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    env_config = CONVERSATION_CONFIGS.get(
        config.task, CONVERSATION_CONFIGS["customer_service"]
    ).copy()
    environment = ConversationEnvironment(**env_config)
    reward_model = create_domain_reward(config.task)

    logger.info("Starting GSPO training for %s", config.model_name)
    return await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )


async def finetune_starter(
    *,
    get_config_fn: Callable[..., Any],
    run_fn: Callable[..., Any],
    model_name: str,
    task: str,
    starter_profile: str,
    use_lora: bool | None,
    use_4bit: bool | None,
    use_8bit: bool | None,
    output_dir: str | None,
    num_outer_iterations: int | None,
    use_wandb: bool | None,
    wandb_project: str | None,
    dry_run: bool,
) -> Any:
    """Run or preview a first GSPO post-training job for a starter family."""
    config_overrides: dict[str, Any] = {}
    if num_outer_iterations is not None:
        config_overrides["num_outer_iterations"] = num_outer_iterations

    config = get_config_fn(
        model_name=model_name,
        task=task,
        starter_profile=starter_profile,
        use_lora=use_lora,
        use_4bit=use_4bit,
        use_8bit=use_8bit,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        output_dir=output_dir,
        **config_overrides,
    )
    return await run_fn(config, dry_run=dry_run)


def glm_serving_recommendations(
    *,
    use_fp8: bool,
    enable_auto_tool_choice: bool,
    tensor_parallel_size: int | None,
    pipeline_parallel_size: int | None,
    max_model_len: int | None,
) -> dict[str, Any]:
    """Return the recommended vLLM settings for GLM 5.x serving."""
    if use_fp8:
        # FP8 fits on a single 8x H200 / B200 node (~754GB weights + KV cache).
        resolved_tp = tensor_parallel_size or 8
        resolved_pp = pipeline_parallel_size or 1
        quantization = "fp8"
    else:
        # BF16 weights are ~1.5TB and require pipeline parallelism across nodes.
        resolved_tp = tensor_parallel_size or 8
        resolved_pp = pipeline_parallel_size or 2
        quantization = None

    return {
        "tensor_parallel_size": resolved_tp,
        "pipeline_parallel_size": resolved_pp,
        "max_model_len": max_model_len or 131072,
        "trust_remote_code": True,
        "reasoning_parser": "glm45",
        "tool_call_parser": "glm45" if enable_auto_tool_choice else None,
        "enable_auto_tool_choice": enable_auto_tool_choice,
        "gpu_memory_utilization": 0.92,
        "quantization": quantization,
    }

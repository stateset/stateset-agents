"""Packaged Qwen3.5-0.8B GSPO starter helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.core.agent import AgentConfig
from stateset_agents.training.config import TrainingConfig, get_config_for_task

logger = logging.getLogger(__name__)

QWEN35_08B_BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
QWEN35_08B_POST_TRAINED_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_08B_SUPPORTED_VARIANTS = [
    QWEN35_08B_BASE_MODEL,
    QWEN35_08B_POST_TRAINED_MODEL,
]
QWEN35_08B_TASK_CHOICES = [
    "customer_service",
    "technical_support",
    "sales",
    "conversational",
]
QWEN35_08B_STARTER_PROFILE_CHOICES = [
    "balanced",
    "memory",
    "quality",
]
QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS = {
    "balanced": "Default first run with the standard Qwen 0.8B starter settings.",
    "memory": "Low-memory first run with 4-bit quantization and shorter context/group sizes.",
    "quality": "Heavier first run with larger context/group sizes when you have more headroom.",
}
QWEN35_08B_DEFAULT_OUTPUT_DIR = "./outputs/qwen3_5_0_8b_gspo"
QWEN35_08B_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
QWEN35_08B_CONFIG_SUFFIXES = {".json", ".js", ".yaml", ".yml"}


def _read_mapping_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix not in QWEN35_08B_CONFIG_SUFFIXES:
        raise ValueError(f"Unsupported config format: {path.suffix or '<none>'}")

    if suffix in {".json", ".js"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for YAML Qwen starter config files. Install with: pip install pyyaml"
            ) from exc

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError("Qwen starter config root must be a JSON/YAML object.")
    return payload


def _write_mapping_file(payload: dict[str, Any], path: Path) -> Path:
    suffix = path.suffix.lower()
    if not suffix:
        path = path.with_suffix(".json")
        suffix = path.suffix.lower()

    if suffix not in QWEN35_08B_CONFIG_SUFFIXES:
        raise ValueError(f"Unsupported config format: {path.suffix or '<none>'}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".json", ".js"}:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required for YAML Qwen starter config files. Install with: pip install pyyaml"
        ) from exc

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def get_qwen3_5_system_prompt(task: str = "customer_service") -> str:
    """Return a task-specific system prompt for Qwen3.5-0.8B."""
    base_intro = "You are Qwen, an AI assistant created by Alibaba Cloud."
    prompts = {
        "conversational": (
            f"{base_intro} You are helpful, concise, and accurate. "
            "You answer clearly and stay grounded in the user's request."
        ),
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


def get_qwen3_5_profile_overrides(starter_profile: str = "balanced") -> dict[str, Any]:
    """Return preset overrides for a starter profile."""
    profiles: dict[str, dict[str, Any]] = {
        "balanced": {},
        "memory": {
            "use_4bit": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_generations": 2,
            "num_outer_iterations": 15,
            "generations_per_iteration": 16,
            "max_new_tokens": 512,
            "max_prompt_length": 768,
            "max_completion_length": 512,
        },
        "quality": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_generations": 6,
            "num_outer_iterations": 40,
            "generations_per_iteration": 48,
            "max_new_tokens": 1024,
            "max_prompt_length": 1536,
            "max_completion_length": 1024,
            "learning_rate": 6e-6,
        },
    }
    if starter_profile not in profiles:
        supported = ", ".join(QWEN35_08B_STARTER_PROFILE_CHOICES)
        raise ValueError(f"Unsupported Qwen starter profile: {starter_profile}. Use one of: {supported}.")
    return dict(profiles[starter_profile])


def get_qwen3_5_profile_description(starter_profile: str = "balanced") -> str:
    """Return the human-readable description for a starter profile."""
    if starter_profile not in QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS:
        supported = ", ".join(QWEN35_08B_STARTER_PROFILE_CHOICES)
        raise ValueError(f"Unsupported Qwen starter profile: {starter_profile}. Use one of: {supported}.")
    return QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS[starter_profile]


def summarize_qwen3_5_config(config: Qwen35Config) -> dict[str, Any]:
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


def describe_qwen3_5_starter_profiles(
    task: str = "customer_service",
    model_name: str = QWEN35_08B_BASE_MODEL,
) -> dict[str, Any]:
    """Return a serializable description of all built-in starter profiles."""
    profiles: dict[str, Any] = {}
    for starter_profile in QWEN35_08B_STARTER_PROFILE_CHOICES:
        config = get_qwen3_5_config(
            model_name=model_name,
            task=task,
            starter_profile=starter_profile,
        )
        profiles[starter_profile] = {
            "description": get_qwen3_5_profile_description(starter_profile),
            "summary": summarize_qwen3_5_config(config),
            "warnings": config.validate(),
            "config": config.to_dict(),
        }

    return {
        "model_name": model_name,
        "task": task,
        "default_profile": QWEN35_08B_STARTER_PROFILE_CHOICES[0],
        "profiles": profiles,
    }


def _default_wandb_tags(task: str) -> list[str]:
    tags = ["qwen3.5", "0.8b", "gspo"]
    if task:
        tags.append(task)
    return tags


@dataclass
class Qwen35Config:
    """Lightweight configuration container for Qwen3.5-0.8B post-training."""

    model_name: str = QWEN35_08B_BASE_MODEL
    task: str = "customer_service"
    starter_profile: str = "balanced"
    system_prompt: str | None = None

    use_lora: bool = True
    lora_r: int | None = 32
    lora_alpha: int | None = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(QWEN35_08B_LORA_TARGET_MODULES)
    )

    use_4bit: bool = False
    use_8bit: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    max_new_tokens: int = 768
    max_prompt_length: int = 1024
    max_completion_length: int = 768
    temperature: float = 0.7
    top_p: float = 0.9

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    learning_rate: float = 8e-6
    num_iterations: int = 1
    num_outer_iterations: int = 25
    generations_per_iteration: int = 32
    clip_range_left: float = 3e-4
    clip_range_right: float = 4e-4

    output_dir: str = QWEN35_08B_DEFAULT_OUTPUT_DIR
    save_steps_every: int = 5

    use_wandb: bool = False
    report_to: str = "none"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    device_map: str | None = "auto"

    def __post_init__(self) -> None:
        if self.system_prompt is None:
            self.system_prompt = get_qwen3_5_system_prompt(self.task)
        if not self.wandb_tags:
            self.wandb_tags = _default_wandb_tags(self.task)
        if self.use_4bit:
            self.use_8bit = False
        if not self.use_lora:
            self.lora_r = None
            self.lora_alpha = None
        if self.use_wandb:
            self.report_to = "wandb"
            if self.wandb_project is None:
                self.wandb_project = "qwen3_5_0_8b-gspo"
        else:
            self.report_to = "none"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Qwen35Config:
        return cls(**config_dict)

    def get_effective_batch_size(self) -> int:
        return int(self.per_device_train_batch_size * self.gradient_accumulation_steps)

    def validate(self) -> list[str]:
        return validate_qwen3_5_config(self)


def get_qwen3_5_config(
    model_name: str = QWEN35_08B_BASE_MODEL,
    task: str = "customer_service",
    starter_profile: str = "balanced",
    use_lora: bool | None = None,
    use_4bit: bool | None = None,
    use_8bit: bool | None = None,
    use_wandb: bool | None = None,
    wandb_project: str | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> Qwen35Config:
    """Create a tuned first-run Qwen3.5-0.8B configuration."""
    resolved_overrides = get_qwen3_5_profile_overrides(starter_profile)
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
    config = Qwen35Config(
        model_name=model_name,
        task=task,
        starter_profile=starter_profile,
        **resolved_overrides,
    )
    logger.info(
        "Created Qwen3.5-0.8B config for task=%s profile=%s model=%s",
        config.task,
        config.starter_profile,
        config.model_name,
    )
    return config


def create_qwen3_5_agent_config(config: Qwen35Config) -> AgentConfig:
    """Create the matching AgentConfig for Qwen3.5-0.8B."""
    return AgentConfig(
        model_name=config.model_name,
        system_prompt=config.system_prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
        device_map=config.device_map,
    )


def get_qwen3_5_gspo_overrides(config: Qwen35Config) -> dict[str, Any]:
    """Return the GSPO override payload for Qwen3.5-0.8B."""
    return {
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


def get_qwen3_5_gspo_config(
    config: Qwen35Config,
    base_config: TrainingConfig | None = None,
):
    """Create the GSPOConfig used for Qwen3.5-0.8B post-training."""
    from stateset_agents.training.gspo_trainer import GSPOConfig

    resolved_base = base_config or get_config_for_task(
        config.task, model_name=config.model_name
    )
    return GSPOConfig.from_training_config(
        resolved_base,
        **get_qwen3_5_gspo_overrides(config),
    )


def validate_qwen3_5_config(config: Qwen35Config) -> list[str]:
    """Validate a Qwen3.5-0.8B first-run configuration."""
    warnings: list[str] = []

    if config.starter_profile not in QWEN35_08B_STARTER_PROFILE_CHOICES:
        warnings.append(
            "starter_profile is outside the built-in profiles; balance memory and context carefully"
        )
    if config.task not in QWEN35_08B_TASK_CHOICES:
        warnings.append("task is outside the built-in starter presets; default environment fallbacks may be used")
    if "qwen" not in config.model_name.lower():
        warnings.append("model_name does not look like a Qwen checkpoint")
    if "qwen3.5-0.8b" not in config.model_name.lower():
        warnings.append(
            "this helper is tuned for Qwen/Qwen3.5-0.8B; verify overrides carefully"
        )
    if config.learning_rate > 2e-5:
        warnings.append("learning rate is high for a first Qwen3.5-0.8B GSPO run")
    if config.learning_rate < 1e-6:
        warnings.append("learning rate is very low and may stall learning")
    if config.per_device_train_batch_size > 4:
        warnings.append("per-device batch size above 4 may increase OOM risk")
    if config.get_effective_batch_size() < 4:
        warnings.append("effective batch size is small; gradients may be noisy")
    if not config.use_lora:
        warnings.append("LoRA is recommended for the first Qwen3.5-0.8B run")
    if config.max_prompt_length > 4096:
        warnings.append("start with a shorter prompt length before scaling context")
    if config.max_completion_length > 2048:
        warnings.append("completion length is large for an initial smoke test")
    if config.use_wandb and not config.wandb_project:
        warnings.append("use_wandb=True but no wandb_project is set")

    return warnings


def create_qwen3_5_preview(
    config: Qwen35Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable preview payload for dry-runs."""
    resolved_warnings = list(warnings) if warnings is not None else config.validate()
    agent_config = create_qwen3_5_agent_config(config)
    return {
        "config": config.to_dict(),
        "summary": summarize_qwen3_5_config(config),
        "agent_config": asdict(agent_config),
        "gspo_overrides": get_qwen3_5_gspo_overrides(config),
        "warnings": resolved_warnings,
    }


def load_qwen3_5_config_file(path: str | Path) -> Qwen35Config:
    """Load a Qwen3.5-0.8B starter config from JSON or YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = _read_mapping_file(config_path)
    config_payload = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    if not isinstance(config_payload, dict):
        raise ValueError("Qwen starter config root must be a JSON/YAML object.")

    loaded = Qwen35Config.from_dict(config_payload)
    logger.info("Loaded Qwen3.5-0.8B config from %s", config_path)
    return loaded


def write_qwen3_5_config_file(
    config: Qwen35Config,
    path: str | Path,
    include_preview: bool = False,
) -> Path:
    """Write a Qwen3.5-0.8B starter config to JSON or YAML."""
    payload = create_qwen3_5_preview(config) if include_preview else config.to_dict()
    written_path = _write_mapping_file(payload, Path(path))
    logger.info("Wrote Qwen3.5-0.8B config to %s", written_path)
    return written_path


async def run_qwen3_5_0_8b_config(
    config: Qwen35Config,
    dry_run: bool = False,
) -> Any:
    """Run or preview a Qwen3.5-0.8B GSPO job from a resolved config object."""
    warnings = config.validate()
    for warning in warnings:
        logger.warning("Config warning: %s", warning)

    if dry_run:
        return create_qwen3_5_preview(config, warnings=warnings)

    gspo_config = get_qwen3_5_gspo_config(config)
    agent_config = create_qwen3_5_agent_config(config)

    from stateset_agents import MultiTurnAgent
    from stateset_agents.core.environment import (
        CONVERSATION_CONFIGS,
        ConversationEnvironment,
    )
    from stateset_agents.rewards.multi_objective_reward import create_domain_reward
    from stateset_agents.training.gspo_trainer import train_with_gspo

    logger.info("Initializing Qwen3.5-0.8B agent")
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


async def finetune_qwen3_5_0_8b(
    model_name: str = QWEN35_08B_BASE_MODEL,
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
    """Run or preview a first GSPO post-training job for Qwen3.5-0.8B."""
    config_overrides: dict[str, Any] = {}
    if num_outer_iterations is not None:
        config_overrides["num_outer_iterations"] = num_outer_iterations

    config = get_qwen3_5_config(
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
    return await run_qwen3_5_0_8b_config(config, dry_run=dry_run)


__all__ = [
    "QWEN35_08B_BASE_MODEL",
    "QWEN35_08B_CONFIG_SUFFIXES",
    "QWEN35_08B_DEFAULT_OUTPUT_DIR",
    "QWEN35_08B_LORA_TARGET_MODULES",
    "QWEN35_08B_POST_TRAINED_MODEL",
    "QWEN35_08B_STARTER_PROFILE_CHOICES",
    "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS",
    "QWEN35_08B_SUPPORTED_VARIANTS",
    "QWEN35_08B_TASK_CHOICES",
    "Qwen35Config",
    "create_qwen3_5_agent_config",
    "create_qwen3_5_preview",
    "describe_qwen3_5_starter_profiles",
    "finetune_qwen3_5_0_8b",
    "get_qwen3_5_gspo_overrides",
    "get_qwen3_5_config",
    "get_qwen3_5_gspo_config",
    "get_qwen3_5_profile_description",
    "get_qwen3_5_profile_overrides",
    "get_qwen3_5_system_prompt",
    "load_qwen3_5_config_file",
    "run_qwen3_5_0_8b_config",
    "summarize_qwen3_5_config",
    "validate_qwen3_5_config",
    "write_qwen3_5_config_file",
]

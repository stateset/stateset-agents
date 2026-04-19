"""
TRL-based GRPO Training with Model Fine-tuning for the GRPO Agent Framework

This module integrates TRL's GRPOTrainer with our framework for fine-tuning
the openai/gpt-oss-120b model using LoRA adapters.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import random
from collections.abc import Callable
from typing import Any

import torch

# Import framework components
from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
from stateset_agents.exceptions import (
    ATTRIBUTE_VALUE_EXCEPTIONS as INPUT_GRADS_EXCEPTIONS,
)
from stateset_agents.exceptions import GPU_EXCEPTIONS as GPU_PROPERTIES_EXCEPTIONS
from stateset_agents.exceptions import (
    IMPORT_EXCEPTIONS as BITSANDBYTES_IMPORT_EXCEPTIONS,
)
from stateset_agents.exceptions import INFERENCE_EXCEPTIONS as VLLM_EXCEPTIONS
from stateset_agents.exceptions import MODEL_IO_EXCEPTIONS as MODEL_LOAD_EXCEPTIONS
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveRewardFunction

from .trl_grpo_config import TRLGRPOConfig

logger = logging.getLogger(__name__)

TRL_GRPO_AVAILABLE = False
GRPOConfig = None
TRLGRPOTrainer = None

_wandb: Any | None
try:
    import wandb as _wandb
except ImportError:  # pragma: no cover - optional dependency
    _wandb = None
wandb: Any | None = _wandb

try:
    from datasets import Dataset as _DatasetType
except ImportError:  # pragma: no cover - optional dependency
    _DatasetType = None
Dataset: Any | None = _DatasetType

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from trl import GRPOConfig as _GRPOConfig
    from trl import GRPOTrainer as _TRLGRPOTrainer

    GRPOConfig = _GRPOConfig
    TRLGRPOTrainer = _TRLGRPOTrainer
    TRL_GRPO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pass

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_loaded = False
AutoModelForCausalLM: Any | None = None
AutoTokenizer: Any | None = None


def _load_transformers():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_loaded, AutoModelForCausalLM, AutoTokenizer
    if _transformers_loaded:
        return True
    if AutoModelForCausalLM is not None and AutoTokenizer is not None:
        _transformers_loaded = True
        return True
    try:
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer

        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False


def _require_transformers() -> None:
    """Ensure transformers components are available before model loading."""
    if not _load_transformers():
        raise ImportError(
            "transformers is required for TRL GRPO training. "
            "Install with `pip install stateset-agents[trl]` or `pip install transformers`."
        )


def _require_wandb() -> None:
    """Ensure Weights & Biases is available before logging."""
    if wandb is None:
        raise ImportError(
            "wandb is required for TRL GRPO logging. "
            "Install with `pip install stateset-agents[trl]` or `pip install wandb`."
        )


def _require_dataset() -> None:
    """Ensure datasets is available before dataset construction."""
    if Dataset is None:
        raise ImportError(
            "datasets is required for TRL GRPO training datasets. "
            "Install with `pip install stateset-agents[trl]` or `pip install datasets`."
        )


def _require_peft() -> None:
    """Ensure PEFT is available before using LoRA features."""
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError(
            "PEFT is required for TRL GRPO LoRA training. "
            "Install with `pip install stateset-agents[trl]` or `pip install peft`."
        )


def _require_kbit_training_support() -> None:
    """Ensure PEFT k-bit helpers are available for quantized training."""
    if prepare_model_for_kbit_training is None:
        raise ImportError(
            "PEFT is required for quantized TRL GRPO training. "
            "Install with `pip install stateset-agents[trl]` or `pip install peft`."
        )


def _require_trl_grpo() -> None:
    """Ensure TRL GRPO components are available before trainer creation."""
    if GRPOConfig is None or TRLGRPOTrainer is None:
        raise ImportError(
            "TRL GRPO training requires trl with GRPO support. "
            "Install a compatible version with `pip install stateset-agents[trl]` or "
            "`pip install 'trl>=0.7,<0.10'`."
        )


def _enable_input_require_grads(model: Any) -> None:
    """
    Ensure at least one checkpoint input requires gradients.

    Some Transformer implementations use `torch.utils.checkpoint` internally for
    `gradient_checkpointing`. With PEFT/LoRA, the base model weights are often
    frozen, so hidden states may not require grad unless explicitly enabled.
    """
    if getattr(model, "_stateset_input_grads_enabled", False):
        return

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            model._stateset_input_grads_enabled = True
            return
        except INPUT_GRADS_EXCEPTIONS as e:  # pragma: no cover
            logger.debug("enable_input_require_grads failed: %s", e)

    try:
        if not hasattr(model, "get_input_embeddings"):
            return
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            return

        def _require_grads_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, torch.Tensor):
                return output.requires_grad_(True)
            return output

        embeddings.register_forward_hook(_require_grads_hook)
        model._stateset_input_grads_enabled = True
    except INPUT_GRADS_EXCEPTIONS as e:  # pragma: no cover
        logger.debug("Failed to register input grad hook: %s", e)


def _require_bitsandbytes() -> None:
    """Ensure bitsandbytes is importable when k-bit loading is requested."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit/8-bit quantization requires CUDA. "
            "Disable `use_4bit/use_8bit` or run on a CUDA-enabled machine."
        )

    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "bitsandbytes is required for 4-bit/8-bit quantization. "
            "Install with `pip install stateset-agents[trl]` or `pip install bitsandbytes`."
        )

    try:
        import bitsandbytes  # noqa: F401  # type: ignore[import-not-found]
    except BITSANDBYTES_IMPORT_EXCEPTIONS as exc:  # pragma: no cover
        raise ImportError(
            "bitsandbytes is installed but failed to import. "
            "If you just installed it in a notebook, restart the runtime/kernel. "
            "Otherwise, verify your CUDA/PyTorch/bitsandbytes compatibility."
        ) from exc


try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None
    VLLM_AVAILABLE = False


class TrajectoryGenerator:
    """Handles efficient trajectory generation for iterative training"""

    def __init__(
        self, config: TRLGRPOConfig, agent: Agent, environment: ConversationEnvironment
    ):
        self.config = config
        self.agent = agent
        self.environment = environment
        self.vllm_engine: Any | None = None
        self.sampling_params: Any | None = None

        if self.config.use_vllm and VLLM_AVAILABLE:
            self._init_vllm()
        elif self.config.use_vllm:
            logger.warning(
                "use_vllm=True but vllm is not installed; falling back to standard generation."
            )

    def _init_vllm(self):
        """Initialize vLLM engine"""
        logger.info("Initializing vLLM engine for fast generation...")
        try:
            self.vllm_engine = LLM(
                model=self.config.model_name,
                trust_remote_code=True,
                dtype="float16" if self.config.fp16 else "bfloat16",
                gpu_memory_utilization=0.6,  # Reserve memory for training
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_completion_length,
            )
            logger.info("vLLM engine initialized")
        except VLLM_EXCEPTIONS as e:
            logger.warning(
                f"Failed to initialize vLLM: {e}. Falling back to standard generation."
            )
            self.vllm_engine = None

    async def generate_batch(self, num_episodes: int) -> list[MultiTurnTrajectory]:
        """Generate a batch of trajectories"""
        # Determine generation method
        if self.vllm_engine is not None:
            return await self._generate_vllm(num_episodes)
        return await self._generate_standard(num_episodes)

    async def _generate_standard(self, num_episodes: int) -> list[MultiTurnTrajectory]:
        """Generate using standard agent loop"""
        logger.info(f"Generating {num_episodes} trajectories (standard)...")
        trajectories: list[MultiTurnTrajectory] = []

        async def agent_fn(
            history: list[dict[str, Any]], context: dict[str, Any] | None = None
        ) -> Any:
            return await self.agent.generate_response(history, context)

        for _ in range(num_episodes):
            traj = await self.environment.run_episode(agent_fn)
            trajectories.append(traj)
        return trajectories

    async def _generate_vllm(self, num_episodes: int) -> list[MultiTurnTrajectory]:
        """
        Generate using vLLM optimization.

        Note: This currently only optimizes the *first* turn or requires
        the environment to provide a batch of initial prompts.
        For true multi-turn with vLLM, we'd need a more complex loop handling state.
        This implementation assumes a simplified 'prompt -> response' flow for speed.
        """
        logger.info(f"Generating {num_episodes} trajectories (vLLM)...")
        vllm_engine = self.vllm_engine
        sampling_params = self.sampling_params
        if vllm_engine is None or sampling_params is None:
            return await self._generate_standard(num_episodes)

        # Get prompts from environment scenarios
        prompts: list[str] = []

        # Naive sampling from environment scenarios
        scenarios_pool = self.environment.scenarios or [{}]
        for _ in range(num_episodes):
            scenario = random.choice(scenarios_pool)
            # Format prompt based on agent configuration
            # This mimics what MultiTurnAgent does internally
            msgs = [
                {"role": "system", "content": self.config.system_prompt or ""},
                {"role": "user", "content": scenario.get("context", "")},
            ]

            # Simple prompt formatting
            formatted = (
                f"{msgs[0]['content']}\n\nUser: {msgs[1]['content']}\nAssistant:"
            )
            prompts.append(formatted)

        # Run vLLM batch generation
        # Run in a thread to avoid blocking asyncio loop
        outputs = await asyncio.to_thread(
            vllm_engine.generate, prompts, sampling_params
        )

        trajectories: list[MultiTurnTrajectory] = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            # Construct a synthetic trajectory
            traj = MultiTurnTrajectory()
            traj.add_turn(
                ConversationTurn(role="user", content=prompts[i])
            )  # Simplification
            traj.add_turn(ConversationTurn(role="assistant", content=generated_text))
            trajectories.append(traj)

        return trajectories


class ModelManager:
    """Manages model loading and LoRA configuration for TRL training"""

    def __init__(self, config: TRLGRPOConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load model and tokenizer with LoRA if specified"""
        logger.info(f"Loading model: {self.config.model_name}")

        model_name_lower = self.config.model_name.lower()
        if (
            torch.cuda.is_available()
            and "qwen" in model_name_lower
            and any(size in model_name_lower for size in ["3b", "7b"])
            and not (self.config.use_4bit or self.config.use_8bit)
        ):
            try:
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                if total_mem_gb < 24:
                    logger.warning(
                        "Model %s on %.1fGB GPU: consider `use_4bit=True` to reduce memory usage",
                        self.config.model_name,
                        total_mem_gb,
                    )
            except GPU_PROPERTIES_EXCEPTIONS:  # pragma: no cover
                logger.warning(
                    "Model %s: consider `use_4bit=True` to reduce memory usage",
                    self.config.model_name,
                )

        _require_transformers()

        try:
            # Load tokenizer
            tokenizer_cls = AutoTokenizer
            model_cls = AutoModelForCausalLM
            if tokenizer_cls is None or model_cls is None:
                raise ImportError("transformers model classes are unavailable")

            self.tokenizer = tokenizer_cls.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left",  # Important for generation
            )
            tokenizer: Any = self.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Model loading kwargs
            model_kwargs = {
                "torch_dtype": torch.float16
                if self.config.fp16
                else (torch.bfloat16 if self.config.bf16 else torch.float32),
                "device_map": "auto" if torch.cuda.is_available() else None,
                "trust_remote_code": True,
            }

            # Add quantization if specified
            if self.config.use_8bit:
                _require_bitsandbytes()
                _require_kbit_training_support()
                model_kwargs["load_in_8bit"] = True
            elif self.config.use_4bit:
                _require_bitsandbytes()
                _require_kbit_training_support()
                model_kwargs["load_in_4bit"] = True

            # Load base model
            base_model = model_cls.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                if hasattr(base_model, "config") and hasattr(
                    base_model.config, "use_cache"
                ):
                    base_model.config.use_cache = False
                if hasattr(base_model, "gradient_checkpointing_enable"):
                    base_model.gradient_checkpointing_enable()
                else:  # pragma: no cover
                    logger.warning(
                        "gradient_checkpointing requested but model does not support it"
                    )
                if self.config.use_lora and not (
                    self.config.use_8bit or self.config.use_4bit
                ):
                    _enable_input_require_grads(base_model)

            # Prepare model for training if using quantization
            if self.config.use_8bit or self.config.use_4bit:
                if prepare_model_for_kbit_training is None:
                    raise ImportError("PEFT k-bit helpers are unavailable")
                base_model = prepare_model_for_kbit_training(base_model)

            # Add LoRA adapters
            if self.config.use_lora:
                _require_peft()
                lora_config_cls = LoraConfig
                task_type = TaskType
                peft_model_fn = get_peft_model
                if (
                    lora_config_cls is None
                    or task_type is None
                    or peft_model_fn is None
                ):
                    raise ImportError("PEFT LoRA components are unavailable")
                # Determine target modules based on model architecture
                if self.config.lora_target_modules:
                    target_modules = self.config.lora_target_modules
                elif "gpt-oss" in self.config.model_name.lower():
                    # For GPT-OSS models
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense"]
                elif "gpt2" in self.config.model_name.lower():
                    target_modules = ["c_attn", "c_proj"]
                elif (
                    "llama" in self.config.model_name.lower()
                    or "qwen" in self.config.model_name.lower()
                ):
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    # Default modules
                    target_modules = ["q_proj", "v_proj"]

                lora_config = lora_config_cls(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=task_type.CAUSAL_LM,
                )

                model = peft_model_fn(base_model, lora_config)
                self.model = model
                model.print_trainable_parameters()

                logger.info("LoRA adapters added to model")
            else:
                self.model = base_model

            # Load reference model if using KL penalty
            if self.config.beta > 0 and self.config.use_reference_model:
                logger.info("Loading reference model for KL penalty...")
                ref_model = model_cls.from_pretrained(
                    self.config.model_name,
                    torch_dtype=model_kwargs["torch_dtype"],
                    device_map="auto" if torch.cuda.device_count() > 1 else None,
                )
                self.ref_model = ref_model
                ref_model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer

        except MODEL_LOAD_EXCEPTIONS as e:
            logger.error(f"Failed to load model: {e}")
            raise


class TRLGRPODatasetBuilder:
    """Builds datasets for TRL GRPO training"""

    def __init__(self, tokenizer, config: TRLGRPOConfig):
        self.tokenizer = tokenizer
        self.config = config

    def build_from_trajectories(
        self, trajectories: list[MultiTurnTrajectory]
    ) -> Any:
        """Build dataset from multi-turn trajectories"""
        _require_dataset()
        dataset_cls = Dataset
        if dataset_cls is None:
            raise ImportError("datasets is unavailable")

        formatted_data = []
        for trajectory in trajectories:
            # Extract conversation turns
            for i, turn in enumerate(trajectory.turns):
                if turn.role == "user" and i + 1 < len(trajectory.turns):
                    user_msg = str(turn.content or "")
                    # Look for the assistant response
                    if trajectory.turns[i + 1].role == "assistant":
                        # Format as a prompt for GRPO
                        prompt = self._format_prompt(user_msg)
                        formatted_data.append({"prompt": prompt})

        return dataset_cls.from_list(formatted_data)

    def build_from_conversations(self, conversations: list[dict[str, Any]]) -> Any:
        """Build dataset from conversation data"""
        _require_dataset()
        dataset_cls = Dataset
        if dataset_cls is None:
            raise ImportError("datasets is unavailable")

        formatted_data = []
        for conv in conversations:
            messages = conv.get("messages", [])

            # Find user-assistant pairs
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and i + 1 < len(messages):
                    if messages[i + 1]["role"] == "assistant":
                        prompt = self._format_prompt(str(msg["content"] or ""))
                        formatted_data.append({"prompt": prompt})

        return dataset_cls.from_list(formatted_data)

    def _format_prompt(self, user_query: str) -> str:
        """Format user query into a prompt"""
        if self.config.system_prompt:
            return f"{self.config.system_prompt}\n\nUser: {user_query}\nAssistant:"
        else:
            return f"User: {user_query}\nAssistant:"


class TRLGRPORewardFunction:
    """Reward function wrapper for TRL GRPO training"""

    def __init__(
        self,
        reward_model: MultiObjectiveRewardFunction,
        agent: Agent,
        environment: ConversationEnvironment,
    ):
        self.reward_model = reward_model
        self.agent = agent
        self.environment = environment

    async def compute_rewards(
        self, completions: list[str], prompts: list[str], **kwargs
    ) -> list[float]:
        """Compute rewards for generated completions"""

        rewards = []
        for prompt, completion in zip(prompts, completions, strict=False):
            # Parse the prompt to extract the user query
            user_query = self._extract_user_query(prompt)

            # Create a conversation turn
            turn = ConversationTurn(
                role="assistant", content=completion, metadata={"generated": True}
            )

            # Compute reward using the framework's reward model
            reward_info = await self.reward_model.compute_reward(
                trajectory=None,  # Single turn evaluation
                turn=turn,
                context={"user_query": user_query},
            )

            rewards.append(reward_info.total_reward)

        return rewards

    def _extract_user_query(self, prompt: str) -> str:
        """Extract user query from formatted prompt"""
        # Simple parsing - can be made more robust
        if "User: " in prompt and "\nAssistant:" in prompt:
            start = prompt.find("User: ") + 6
            end = prompt.find("\nAssistant:")
            return prompt[start:end].strip()
        return prompt


class TRLGRPOTrainerWrapper:
    """Wrapper for TRL's GRPOTrainer that integrates with our framework"""

    def __init__(
        self,
        config: TRLGRPOConfig,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        reward_function: Callable,
        ref_model: Any | None = None,
    ):
        _require_trl_grpo()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reward_function = reward_function
        self.ref_model = ref_model

        # Create TRL GRPO configuration
        self.grpo_config = self._create_grpo_config()

        # Initialize TRL trainer
        trainer_cls = TRLGRPOTrainer
        if trainer_cls is None:
            raise ImportError("TRL GRPO trainer is unavailable")
        self.trainer = trainer_cls(
            model=self.model,
            ref_model=self.ref_model,
            args=self.grpo_config,
            train_dataset=self.train_dataset,
            reward_funcs=self.reward_function,
            tokenizer=self.tokenizer,
        )

    def _create_grpo_config(self) -> Any:
        """Create GRPOConfig from our training config"""
        grpo_config_cls = GRPOConfig
        if grpo_config_cls is None:
            raise ImportError("TRL GRPO config is unavailable")
        return grpo_config_cls(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.get_warmup_steps(),
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            report_to=self.config.report_to.split(",")
            if self.config.report_to != "none"
            else [],
            remove_unused_columns=False,
            # GRPO specific parameters
            beta=self.config.beta,
            num_generations=self.config.num_generations,
            num_iterations=self.config.num_iterations,
            mini_batch_size=self.config.mini_batch_size,
            # Generation parameters
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            # Additional optimization
            dataloader_num_workers=self.config.dataloader_num_workers,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

    def train(self):
        """Run training"""
        logger.info("Starting TRL GRPO training...")
        self.trainer.train()

    def save_model(self, output_dir: str):
        """Save the trained model"""
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)


from .trl_grpo_entrypoints import (
    train_customer_service_with_trl,
    train_iterative_grpo,
    train_with_trl_grpo,
)


# Export main components
__all__ = [
    "TRLGRPOConfig",
    "ModelManager",
    "TRLGRPODatasetBuilder",
    "TRLGRPORewardFunction",
    "TRLGRPOTrainerWrapper",
    "TrajectoryGenerator",
    "train_with_trl_grpo",
    "train_iterative_grpo",
    "train_customer_service_with_trl",
]

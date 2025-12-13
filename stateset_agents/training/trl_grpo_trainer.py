"""
TRL-based GRPO Training with Model Fine-tuning for the GRPO Agent Framework

This module integrates TRL's GRPOTrainer with our framework for fine-tuning
the openai/gpt-oss-120b model using LoRA adapters.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import framework components
from stateset_agents.core.agent import Agent, AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveRewardFunction
from .config import TrainingConfig, get_config_for_task

# Import additional dependencies
TRL_GRPO_AVAILABLE = False
GRPOConfig = None
TRLGRPOTrainer = None
LengthSampler = None

try:
    import numpy as np
    import wandb
    from datasets import Dataset
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    # TRL >= 0.10 renamed/removed GRPOConfig, handle gracefully
    try:
        from trl import GRPOConfig
        from trl import GRPOTrainer as TRLGRPOTrainer
        try:
            from trl.core import LengthSampler
        except ImportError:
            # LengthSampler moved or removed in newer TRL versions
            LengthSampler = None
        TRL_GRPO_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"TRL GRPO components not available: {e}")
        logger.warning("TRL-based GRPO training disabled. Install trl>=0.7,<0.10 for this feature.")
except ImportError as e:
    logger.warning(f"Optional TRL dependencies not available: {e}")
    logger.warning("Install with: pip install peft datasets trl wandb")

# Lazy import transformers to avoid torch/torchvision compatibility issues
_transformers_loaded = False
AutoModelForCausalLM = None
AutoTokenizer = None
TrainingArguments = None
get_cosine_schedule_with_warmup = None

def _load_transformers():
    """Lazily load transformers to avoid import-time errors."""
    global _transformers_loaded, AutoModelForCausalLM, AutoTokenizer
    global TrainingArguments, get_cosine_schedule_with_warmup
    if _transformers_loaded:
        return True
    try:
        from transformers import (
            AutoModelForCausalLM as _AutoModelForCausalLM,
            AutoTokenizer as _AutoTokenizer,
            TrainingArguments as _TrainingArguments,
            get_cosine_schedule_with_warmup as _get_cosine,
        )
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        TrainingArguments = _TrainingArguments
        get_cosine_schedule_with_warmup = _get_cosine
        _transformers_loaded = True
        return True
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Failed to load transformers: {e}")
        return False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Install with `pip install vllm` for faster generation.")


@dataclass
class TRLGRPOConfig(TrainingConfig):
    """Extended configuration for TRL GRPO training"""

    # Model identification
    model_name: str = "gpt2"

    # TRL GRPO specific parameters
    beta: float = 0.0  # KL penalty coefficient
    num_generations: int = 4  # Number of generations per prompt
    num_iterations: int = 1  # PPO-style iterations (inner loop)
    mini_batch_size: int = 1  # Mini batch size for GRPO

    # Iterative/Online parameters
    num_outer_iterations: int = 1  # Number of Generate -> Train cycles
    generations_per_iteration: int = 100  # Number of prompts to generate per cycle

    # Generation parameters
    max_prompt_length: int = 256
    max_completion_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # Model optimization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit: bool = False
    use_4bit: bool = False

    # Backend
    use_vllm: bool = False  # Enable vLLM for generation

    @classmethod
    def from_training_config(cls, config: TrainingConfig, **kwargs) -> "TRLGRPOConfig":
        """Create TRL GRPO config from standard training config"""
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        return cls(**config_dict)


class TrajectoryGenerator:
    """Handles efficient trajectory generation for iterative training"""
    
    def __init__(self, config: TRLGRPOConfig, agent: Agent, environment: ConversationEnvironment):
        self.config = config
        self.agent = agent
        self.environment = environment
        self.vllm_engine = None
        
        if self.config.use_vllm and VLLM_AVAILABLE:
            self._init_vllm()
            
    def _init_vllm(self):
        """Initialize vLLM engine"""
        logger.info("Initializing vLLM engine for fast generation...")
        try:
            self.vllm_engine = LLM(
                model=self.config.model_name,
                trust_remote_code=True,
                dtype="float16" if self.config.fp16 else "bfloat16",
                gpu_memory_utilization=0.6, # Reserve memory for training
            )
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_completion_length,
            )
            logger.info("vLLM engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM: {e}. Falling back to standard generation.")
            self.vllm_engine = None

    async def generate_batch(self, num_episodes: int) -> List[MultiTurnTrajectory]:
        """Generate a batch of trajectories"""
        trajectories = []
        
        # Determine generation method
        if self.vllm_engine:
            return await self._generate_vllm(num_episodes)
        else:
            return await self._generate_standard(num_episodes)
            
    async def _generate_standard(self, num_episodes: int) -> List[MultiTurnTrajectory]:
        """Generate using standard agent loop"""
        logger.info(f"Generating {num_episodes} trajectories (standard)...")
        trajectories = []
        for _ in range(num_episodes):
            traj = await self.environment.run_episode(self.agent)
            trajectories.append(traj)
        return trajectories
        
    async def _generate_vllm(self, num_episodes: int) -> List[MultiTurnTrajectory]:
        """
        Generate using vLLM optimization.
        
        Note: This currently only optimizes the *first* turn or requires 
        the environment to provide a batch of initial prompts.
        For true multi-turn with vLLM, we'd need a more complex loop handling state.
        This implementation assumes a simplified 'prompt -> response' flow for speed.
        """
        logger.info(f"Generating {num_episodes} trajectories (vLLM)...")
        
        # Get prompts from environment scenarios
        prompts = []
        scenarios = []
        
        # Naive sampling from environment scenarios
        for _ in range(num_episodes):
            scenario = self.environment._get_random_scenario()
            scenarios.append(scenario)
            # Format prompt based on agent configuration
            # This mimics what MultiTurnAgent does internally
            msgs = [{"role": "system", "content": self.config.system_prompt or ""}, 
                   {"role": "user", "content": scenario.get("context", "")}]
            
            # Simple prompt formatting
            formatted = f"{msgs[0]['content']}\n\nUser: {msgs[1]['content']}\nAssistant:"
            prompts.append(formatted)
            
        # Run vLLM batch generation
        # Run in a thread to avoid blocking asyncio loop
        outputs = await asyncio.to_thread(
            self.vllm_engine.generate, 
            prompts, 
            self.sampling_params
        )
        
        trajectories = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # Construct a synthetic trajectory
            traj = MultiTurnTrajectory()
            traj.add_turn(ConversationTurn(role="user", content=prompts[i])) # Simplification
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

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load model and tokenizer with LoRA if specified"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load transformers lazily
        if not _load_transformers():
            raise ImportError("transformers is required but failed to load")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left",  # Important for generation
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
                model_kwargs["load_in_8bit"] = True
            elif self.config.use_4bit:
                model_kwargs["load_in_4bit"] = True

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                base_model.gradient_checkpointing_enable()

            # Prepare model for training if using quantization
            if self.config.use_8bit or self.config.use_4bit:
                base_model = prepare_model_for_kbit_training(base_model)

            # Add LoRA adapters
            if self.config.use_lora:
                # Determine target modules based on model architecture
                if self.config.lora_target_modules:
                    target_modules = self.config.lora_target_modules
                elif "gpt-oss" in self.config.model_name.lower():
                    # For GPT-OSS models
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense"]
                elif "gpt2" in self.config.model_name.lower():
                    target_modules = ["c_attn", "c_proj"]
                elif "llama" in self.config.model_name.lower():
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    # Default modules
                    target_modules = ["q_proj", "v_proj"]

                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )

                self.model = get_peft_model(base_model, lora_config)
                self.model.print_trainable_parameters()

                logger.info("LoRA adapters added to model")
            else:
                self.model = base_model

            # Load reference model if using KL penalty
            if self.config.beta > 0 and self.config.use_reference_model:
                logger.info("Loading reference model for KL penalty...")
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=model_kwargs["torch_dtype"],
                    device_map="auto" if torch.cuda.device_count() > 1 else None,
                )
                self.ref_model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


class TRLGRPODatasetBuilder:
    """Builds datasets for TRL GRPO training"""

    def __init__(self, tokenizer, config: TRLGRPOConfig):
        self.tokenizer = tokenizer
        self.config = config

    def build_from_trajectories(
        self, trajectories: List[MultiTurnTrajectory]
    ) -> Dataset:
        """Build dataset from multi-turn trajectories"""

        formatted_data = []
        for trajectory in trajectories:
            # Extract conversation turns
            for i, turn in enumerate(trajectory.turns):
                if turn.role == "user" and i + 1 < len(trajectory.turns):
                    user_msg = turn.content
                    # Look for the assistant response
                    if trajectory.turns[i + 1].role == "assistant":
                        # Format as a prompt for GRPO
                        prompt = self._format_prompt(user_msg)
                        formatted_data.append({"prompt": prompt})

        return Dataset.from_list(formatted_data)

    def build_from_conversations(self, conversations: List[Dict[str, Any]]) -> Dataset:
        """Build dataset from conversation data"""

        formatted_data = []
        for conv in conversations:
            messages = conv.get("messages", [])

            # Find user-assistant pairs
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and i + 1 < len(messages):
                    if messages[i + 1]["role"] == "assistant":
                        prompt = self._format_prompt(msg["content"])
                        formatted_data.append({"prompt": prompt})

        return Dataset.from_list(formatted_data)

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
        self, completions: List[str], prompts: List[str], **kwargs
    ) -> List[float]:
        """Compute rewards for generated completions"""

        rewards = []
        for prompt, completion in zip(prompts, completions):
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
        train_dataset: Dataset,
        reward_function: Callable,
        ref_model: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reward_function = reward_function
        self.ref_model = ref_model

        # Create TRL GRPO configuration
        self.grpo_config = self._create_grpo_config()

        # Initialize TRL trainer
        self.trainer = TRLGRPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self.grpo_config,
            train_dataset=self.train_dataset,
            reward_funcs=self.reward_function,
            tokenizer=self.tokenizer,
        )

    def _create_grpo_config(self) -> GRPOConfig:
        """Create GRPOConfig from our training config"""

        return GRPOConfig(
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


async def train_with_trl_grpo(
    config: TRLGRPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveRewardFunction,
    train_data: Optional[List[Dict[str, Any]]] = None,
    eval_data: Optional[List[Dict[str, Any]]] = None,
) -> Agent:
    """Main training function using TRL GRPO"""

    logger.info("Initializing TRL GRPO training")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize wandb if configured
    if config.report_to == "wandb" and config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name
            or f"trl-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=config.wandb_tags,
        )

    # Initialize model manager
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    # Build dataset
    dataset_builder = TRLGRPODatasetBuilder(tokenizer, config)

    if train_data:
        # Use provided training data
        train_dataset = dataset_builder.build_from_conversations(train_data)
    else:
        # Generate trajectories using the environment
        logger.info("Generating training trajectories...")
        trajectories = []

        for episode in range(
            min(100, config.num_episodes)
        ):  # Generate some initial data
            trajectory = await environment.run_episode(agent)
            trajectories.append(trajectory)

        train_dataset = dataset_builder.build_from_trajectories(trajectories)

    logger.info(f"Training dataset size: {len(train_dataset)}")

    # Create reward function wrapper
    reward_wrapper = TRLGRPORewardFunction(reward_model, agent, environment)

    # Synchronous wrapper for async reward function
    def sync_reward_function(completions, prompts, **kwargs):
        """Synchronous wrapper for async reward computation"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                reward_wrapper.compute_rewards(completions, prompts, **kwargs)
            )
        finally:
            loop.close()

    # Create trainer wrapper
    trainer = TRLGRPOTrainerWrapper(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_function=sync_reward_function,
        ref_model=model_manager.ref_model,
    )

    # Run training
    trainer.train()

    # Save final model
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    # Update agent with trained model
    agent.model = trainer.model
    agent.tokenizer = tokenizer

    # Finish wandb run
    if config.report_to == "wandb":
        wandb.finish()

    logger.info("✨ TRL GRPO training completed successfully!")
    return agent


# Convenience functions for common training scenarios


async def train_customer_service_with_trl(
    model_name: str = "openai/gpt-oss-120b",
    train_data: Optional[List[Dict[str, Any]]] = None,
    num_episodes: int = 1000,
    output_dir: str = "./outputs/trl_grpo",
    **kwargs,
) -> Agent:
    """Train a customer service agent using TRL GRPO"""

    # Get base configuration
    base_config = get_config_for_task("customer_service", model_name=model_name)

    # Create TRL GRPO config
    config = TRLGRPOConfig.from_training_config(
        base_config, num_episodes=num_episodes, output_dir=output_dir, **kwargs
    )

    # Create agent
    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt="You are a helpful and empathetic customer service representative.",
        **kwargs,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # Create environment
    from ..core.environment import CONVERSATION_CONFIGS

    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    environment = ConversationEnvironment(**env_config)

    # Create reward model
    from ..rewards.multi_objective_reward import create_customer_service_reward

    reward_model = create_customer_service_reward()

    # Run training
    return await train_with_trl_grpo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
        train_data=train_data,
    )


async def train_iterative_grpo(
    config: TRLGRPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveRewardFunction,
) -> Agent:
    """
    Run iterative (online) GRPO training.
    
    Loop:
    1. Generate trajectories using current policy (supports vLLM)
    2. Train for N epochs
    3. Repeat
    """
    logger.info(f"Starting Iterative GRPO Training ({config.num_outer_iterations} iterations)")
    
    # Initialize WandB once
    if config.report_to == "wandb" and config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name or f"iterative-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["iterative", "grpo"] + (config.wandb_tags or []),
        )

    # Initialize model manager first (load base model)
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Update agent with loaded model to ensure generation uses correct weights
    # Note: If vLLM is used, it loads its own model instance.
    agent.model = model
    agent.tokenizer = tokenizer
    
    # Initialize trajectory generator
    generator = TrajectoryGenerator(config, agent, environment)
    
    # Initialize dataset builder
    dataset_builder = TRLGRPODatasetBuilder(tokenizer, config)
    
    # Initialize Reward Function Wrapper
    reward_wrapper = TRLGRPORewardFunction(reward_model, agent, environment)
    
    def sync_reward_function(completions, prompts, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                reward_wrapper.compute_rewards(completions, prompts, **kwargs)
            )
        finally:
            loop.close()

    # --- Rich Dashboard Setup ---
    try:
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
        from rich.table import Table
        from rich import box
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False
        
    if RICH_AVAILABLE:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        
        status_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        task_id = status_progress.add_task("[green]Iterative Training", total=config.num_outer_iterations)
        
        metrics_table = Table(title="Training Metrics", box=box.SIMPLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")

    # Context manager for Live display
    from contextlib import nullcontext
    live_ctx = Live(layout, refresh_per_second=4) if RICH_AVAILABLE else nullcontext()
    
    with live_ctx:
        for iteration in range(config.num_outer_iterations):
            current_iter_label = f"Iteration {iteration + 1}/{config.num_outer_iterations}"
            logger.info(f"=== {current_iter_label} ===")
            
            if RICH_AVAILABLE:
                layout["header"].update(Panel(f"StateSet Agents - GRPO Training\n{current_iter_label}", style="bold white on blue"))
                layout["left"].update(Panel(status_progress, title="Progress"))
                layout["right"].update(Panel(metrics_table, title="Metrics"))
            
            # Step 1: Generate Data
            logger.info("Phase 1: Generating Trajectories...")
            if RICH_AVAILABLE:
                 status_progress.update(task_id, description=f"{current_iter_label}: Generating Data")
            
            trajectories = await generator.generate_batch(config.generations_per_iteration)
            
            # Step 2: Build Dataset
            train_dataset = dataset_builder.build_from_trajectories(trajectories)
            
            # Step 3: Train
            logger.info("Phase 2: Training...")
            if RICH_AVAILABLE:
                 status_progress.update(task_id, description=f"{current_iter_label}: Training")

            # Create a new trainer instance for this iteration
            trainer_wrapper = TRLGRPOTrainerWrapper(
                config=config,
                model=agent.model,
                tokenizer=agent.tokenizer,
                train_dataset=train_dataset,
                reward_function=sync_reward_function,
                ref_model=model_manager.ref_model,
            )
            
            trainer_wrapper.train()
            
            # Update agent model with trained weights
            agent.model = trainer_wrapper.trainer.model
            
            # Update Dashboard Metrics (Mocking some values or extracting from trainer logs if available)
            if RICH_AVAILABLE:
                metrics_table = Table(title="Training Metrics", box=box.SIMPLE)
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="magenta")
                metrics_table.add_row("Trajectories", str(len(trajectories)))
                metrics_table.add_row("Dataset Size", str(len(train_dataset)))
                # In a real scenario, we'd extract the loss/reward from the trainer's history
                metrics_table.add_row("Last Iteration", str(iteration + 1))
                layout["right"].update(Panel(metrics_table, title="Metrics"))
                
                status_progress.advance(task_id)

            if generator.vllm_engine:
                logger.warning("Note: vLLM engine weights are not automatically updated in this loop implementation yet.")
                
            # Save intermediate checkpoint
            checkpoint_dir = os.path.join(config.output_dir, f"iter_{iteration}")
            trainer_wrapper.save_model(checkpoint_dir)
        
    logger.info("Iterative training completed.")
    if config.report_to == "wandb":
        wandb.finish()
        
    return agent


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

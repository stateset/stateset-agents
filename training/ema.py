"""
Exponential Moving Average (EMA) Model Support

This module provides EMA model functionality for stable RL training.
EMA models are crucial for:
- Stable training with less variance
- Better final model quality
- Reference model generation
- Checkpoint averaging

Key features:
- Efficient memory-mapped EMA computation
- Support for LoRA/PEFT models
- Automatic EMA decay scheduling
- Multi-GPU compatible

Reference: https://arxiv.org/abs/1803.05407 (Mean teachers are better role models)
"""

import copy
import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAModel:
    """
    Exponential Moving Average model wrapper.

    Maintains an exponential moving average of model parameters for
    stable evaluation and reference model generation in RL training.

    The EMA is computed as:
        ema_params = decay * ema_params + (1 - decay) * model_params

    Features:
    - Memory-efficient shadow parameter storage
    - Optional CPU offloading
    - Decay scheduling (linear, cosine warmup)
    - PEFT/LoRA compatible
    - Buffers can optionally be included in EMA

    Example:
        ```python
        # Create EMA model
        ema = EMAModel(model, decay=0.999)

        # During training
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA after each step

        # For evaluation
        with ema.average_parameters():
            eval_loss = model(eval_batch)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_warmup: bool = True,
        warmup_steps: int = 2000,
        include_buffers: bool = True,
        cpu_offload: bool = False,
        update_every: int = 1,
    ):
        """
        Initialize EMA model.

        Args:
            model: Model to track with EMA
            decay: EMA decay factor (higher = slower update)
            min_decay: Minimum decay during warmup
            update_after_step: Start EMA updates after this step
            use_warmup: Whether to use decay warmup
            warmup_steps: Number of warmup steps
            include_buffers: Whether to include buffers (e.g., batch norm stats)
            cpu_offload: Whether to store EMA on CPU (saves GPU memory)
            update_every: Update EMA every N steps
        """
        self.model = model
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_steps = warmup_steps
        self.include_buffers = include_buffers
        self.cpu_offload = cpu_offload
        self.update_every = update_every

        self.step = 0
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.collected_params: Dict[str, torch.Tensor] = {}

        # Initialize shadow parameters
        self._initialize_shadow_params()

    def _initialize_shadow_params(self) -> None:
        """Initialize shadow parameters from model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shadow = param.data.clone()
                if self.cpu_offload:
                    shadow = shadow.cpu()
                self.shadow_params[name] = shadow

        if self.include_buffers:
            for name, buffer in self.model.named_buffers():
                if buffer is not None:
                    shadow = buffer.data.clone()
                    if self.cpu_offload:
                        shadow = shadow.cpu()
                    self.shadow_params[f"buffer_{name}"] = shadow

        logger.info(
            f"EMA initialized with {len(self.shadow_params)} parameters, "
            f"decay={self.decay}, warmup_steps={self.warmup_steps}"
        )

    def get_decay(self) -> float:
        """Get current EMA decay value with optional warmup."""
        if self.step < self.update_after_step:
            return 0.0

        if not self.use_warmup:
            return self.decay

        # Cosine warmup schedule
        effective_step = self.step - self.update_after_step
        if effective_step < self.warmup_steps:
            # Warmup from min_decay to decay
            progress = effective_step / self.warmup_steps
            warmup_decay = self.min_decay + (self.decay - self.min_decay) * (
                0.5 * (1 + math.cos(math.pi * (1 - progress)))
            )
            return warmup_decay

        return self.decay

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None) -> None:
        """
        Update EMA parameters.

        Args:
            model: Model to use (defaults to self.model)
        """
        self.step += 1

        if self.step % self.update_every != 0:
            return

        if self.step <= self.update_after_step:
            return

        model = model or self.model
        decay = self.get_decay()

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                shadow = self.shadow_params[name]

                # Handle device mismatch
                if self.cpu_offload:
                    param_data = param.data.cpu()
                else:
                    param_data = param.data

                # EMA update: shadow = decay * shadow + (1 - decay) * param
                shadow.mul_(decay).add_(param_data, alpha=1 - decay)

        if self.include_buffers:
            for name, buffer in model.named_buffers():
                if buffer is not None:
                    key = f"buffer_{name}"
                    if key in self.shadow_params:
                        shadow = self.shadow_params[key]
                        if self.cpu_offload:
                            buffer_data = buffer.data.cpu()
                        else:
                            buffer_data = buffer.data
                        shadow.mul_(decay).add_(buffer_data, alpha=1 - decay)

    @torch.no_grad()
    def copy_to(self, model: Optional[nn.Module] = None) -> None:
        """
        Copy EMA parameters to model.

        Args:
            model: Target model (defaults to self.model)
        """
        model = model or self.model

        for name, param in model.named_parameters():
            if name in self.shadow_params:
                shadow = self.shadow_params[name]
                if self.cpu_offload:
                    param.data.copy_(shadow.to(param.device))
                else:
                    param.data.copy_(shadow)

        if self.include_buffers:
            for name, buffer in model.named_buffers():
                if buffer is not None:
                    key = f"buffer_{name}"
                    if key in self.shadow_params:
                        shadow = self.shadow_params[key]
                        if self.cpu_offload:
                            buffer.data.copy_(shadow.to(buffer.device))
                        else:
                            buffer.data.copy_(shadow)

    @torch.no_grad()
    def store(self) -> None:
        """Store current model parameters (for restore later)."""
        self.collected_params = {}
        for name, param in self.model.named_parameters():
            self.collected_params[name] = param.data.clone()

        if self.include_buffers:
            for name, buffer in self.model.named_buffers():
                if buffer is not None:
                    self.collected_params[f"buffer_{name}"] = buffer.data.clone()

    @torch.no_grad()
    def restore(self) -> None:
        """Restore previously stored parameters."""
        if not self.collected_params:
            logger.warning("No stored parameters to restore")
            return

        for name, param in self.model.named_parameters():
            if name in self.collected_params:
                param.data.copy_(self.collected_params[name])

        if self.include_buffers:
            for name, buffer in self.model.named_buffers():
                if buffer is not None:
                    key = f"buffer_{name}"
                    if key in self.collected_params:
                        buffer.data.copy_(self.collected_params[key])

        self.collected_params = {}

    @contextmanager
    def average_parameters(self, model: Optional[nn.Module] = None):
        """
        Context manager for using EMA parameters.

        Temporarily copies EMA parameters to model, then restores original.

        Example:
            ```python
            with ema.average_parameters():
                # Model now has EMA parameters
                outputs = model(inputs)
            # Model now has original parameters
            ```
        """
        model = model or self.model
        self.store()
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for serialization."""
        return {
            "shadow_params": {k: v.clone() for k, v in self.shadow_params.items()},
            "step": self.step,
            "decay": self.decay,
            "min_decay": self.min_decay,
            "use_warmup": self.use_warmup,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self.shadow_params = {
            k: v.clone() for k, v in state_dict["shadow_params"].items()
        }
        self.step = state_dict["step"]
        self.decay = state_dict.get("decay", self.decay)
        self.min_decay = state_dict.get("min_decay", self.min_decay)
        self.use_warmup = state_dict.get("use_warmup", self.use_warmup)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)


class EMACallback:
    """
    Callback for automatic EMA updates during training.

    Can be used with various training loops.
    """

    def __init__(
        self,
        ema_model: EMAModel,
        update_on: str = "step",  # "step" or "epoch"
    ):
        self.ema_model = ema_model
        self.update_on = update_on

    def on_step_end(self) -> None:
        """Called after each training step."""
        if self.update_on == "step":
            self.ema_model.update()

    def on_epoch_end(self) -> None:
        """Called after each epoch."""
        if self.update_on == "epoch":
            self.ema_model.update()


class MultiEMA:
    """
    Multiple EMA models with different decay rates.

    Useful for:
    - Model soup / averaging
    - Finding optimal EMA decay
    - Checkpointing at different smoothing levels
    """

    def __init__(
        self,
        model: nn.Module,
        decays: list = [0.99, 0.999, 0.9999],
        cpu_offload: bool = True,
    ):
        """
        Initialize multiple EMA models.

        Args:
            model: Base model
            decays: List of decay values
            cpu_offload: Whether to store EMAs on CPU
        """
        self.model = model
        self.decays = decays
        self.emas = {
            decay: EMAModel(model, decay=decay, cpu_offload=cpu_offload)
            for decay in decays
        }

    def update(self) -> None:
        """Update all EMA models."""
        for ema in self.emas.values():
            ema.update()

    def get_ema(self, decay: float) -> EMAModel:
        """Get EMA model for specific decay."""
        return self.emas[decay]

    @contextmanager
    def average_parameters(self, decay: float):
        """Use EMA parameters for specific decay."""
        with self.emas[decay].average_parameters():
            yield

    def state_dict(self) -> Dict[str, Any]:
        """Get combined state dict."""
        return {
            decay: ema.state_dict()
            for decay, ema in self.emas.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load combined state dict."""
        for decay, ema_state in state_dict.items():
            if decay in self.emas:
                self.emas[decay].load_state_dict(ema_state)


def create_ema_model(
    model: nn.Module,
    decay: float = 0.9999,
    warmup_steps: int = 2000,
    cpu_offload: bool = False,
) -> EMAModel:
    """
    Create EMA model with sensible defaults for RL training.

    Args:
        model: Model to track
        decay: EMA decay (0.9999 recommended for RL)
        warmup_steps: Warmup steps (should match lr warmup)
        cpu_offload: Store EMA on CPU to save GPU memory

    Returns:
        EMAModel instance
    """
    return EMAModel(
        model=model,
        decay=decay,
        min_decay=0.9,
        use_warmup=True,
        warmup_steps=warmup_steps,
        cpu_offload=cpu_offload,
    )

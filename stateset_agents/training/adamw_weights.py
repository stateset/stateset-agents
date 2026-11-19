"""Float32 AdamW weights and moments for low-precision language policies."""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn


class ModelAdamW(torch.optim.AdamW):
    """Run AdamW on float32 working copies and publish low-precision weights."""

    def __init__(self, groups: list[dict[str, Any]], **options: Any) -> None:
        self.masters: dict[nn.Parameter, nn.Parameter] = {}
        super().__init__(groups, **options)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Preserve working precision for layers added during fine tuning."""
        # Let AdamW validate and normalize the input, including named parameters.
        super().add_param_group(dict(param_group))
        group = self.param_groups[-1]
        try:
            if any(p in self.masters for p in group["params"]):
                raise ValueError(
                    "some parameters appear in more than one parameter group"
                )
            # Frozen layers only allocate storage when used or restored.
            added = {
                p: nn.Parameter(
                    p.detach().float()
                    if p.requires_grad
                    else p.new_empty(0, dtype=torch.float32)
                )
                for p in group["params"]
                if p.dtype in (torch.float16, torch.bfloat16)
            }
            group["params"] = [added.get(p, p) for p in group["params"]]
        except BaseException:
            self.param_groups.pop()
            raise
        self.masters.update(added)

    @staticmethod
    @torch.no_grad()
    def _materialize(source: nn.Parameter, master: nn.Parameter) -> None:
        """Allocate a previously frozen parameter's working weights at first use."""
        if master.shape != source.shape:
            master.set_(source.detach().float())

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the working copies, including checkpoints predating master weights."""
        super().load_state_dict(state_dict)
        for source, master in self.masters.items():
            state = self.state.get(master, {})
            if not state and not source.requires_grad and master.shape != source.shape:
                continue
            self._materialize(source, master)
            saved = state.get("master_weight")
            master.copy_(source if saved is None else saved)
            if saved is not None:
                state["master_weight"] = master.detach()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear both optimizer and model gradients before another backward pass."""
        super().zero_grad(set_to_none=set_to_none)
        for source in self.masters:
            if source.grad is not None:
                if set_to_none:
                    source.grad = None
                else:
                    source.grad = source.grad.detach()
                    source.grad.zero_()

    @torch.no_grad()
    def prepare_model_gradients(self) -> None:
        """Copy the current model gradients before clipping or AMP unscaling."""
        for source, master in self.masters.items():
            if source.grad is None:
                master.grad = None
            else:
                self._materialize(source, master)
                master.grad = source.grad.float()

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Any] | None = None,
        *,
        accumulated: bool = False,
        publish_gradients: bool = False,
    ) -> Any:
        """Commit one step using the current gradients or an accumulated master window."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not accumulated:
            self.prepare_model_gradients()
        for source, master in self.masters.items():
            if master.grad is not None:
                # A legacy checkpoint starts at its published weight. A current
                # checkpoint also contains progress smaller than a model ULP.
                saved = self.state.get(master, {}).get("master_weight")
                if saved is None:
                    master.copy_(source)
                else:
                    master.copy_(
                        torch.where(source != saved.to(source.dtype), source, saved)
                    )
        super().step()
        for source, master in self.masters.items():
            if master.grad is not None:
                source.copy_(master)
                if publish_gradients:
                    source.grad = master.grad.to(source.dtype)
                self.state[master]["master_weight"] = master.detach()
        return loss


def model_adamw(
    parameters: Iterable[nn.Parameter], **options: Any
) -> torch.optim.AdamW:
    """Create AdamW at model precision for FP32/FP64 and working precision for FP16/BF16."""
    params = list(parameters)
    if not params:
        raise ValueError("optimizer got an empty parameter list")
    return grouped_model_adamw([{"params": params}], **options)


def grouped_model_adamw(
    groups: list[dict[str, Any]], **options: Any
) -> torch.optim.AdamW:
    """Keep each group's decay and learning-rate controls when widening parameters."""
    groups = [{**group, "params": list(group["params"])} for group in groups]
    if any(
        p.dtype in (torch.float16, torch.bfloat16)
        for group in groups
        for p in group["params"]
    ):
        return ModelAdamW(groups, **options)
    return torch.optim.AdamW(groups, **options)


class AdamWWeights:
    """Accumulate low-precision backward passes in float32 optimizer coordinates."""

    def __init__(self, parameters: Iterable[nn.Parameter], **options: Any) -> None:
        self.optimizer = model_adamw(parameters, **options)
        self.masters = (
            self.optimizer.masters if isinstance(self.optimizer, ModelAdamW) else {}
        )

    def target(self, parameter: nn.Parameter) -> nn.Parameter:
        """Return the parameter carrying optimizer gradients and moments."""
        return self.masters.get(parameter, parameter)

    @torch.no_grad()
    def collect_gradients(self) -> None:
        """Move this backward pass into the float32 accumulation window."""
        for p, master in self.masters.items():
            if p.grad is not None:
                ModelAdamW._materialize(p, master)
                if master.grad is None:
                    master.grad = p.grad.float()
                else:
                    master.grad.add_(p.grad)
                p.grad = None

    def step(self, *, publish_gradients: bool = False) -> None:
        """Update from the accumulated gradients and publish model weights."""
        if isinstance(self.optimizer, ModelAdamW):
            self.optimizer.step(accumulated=True, publish_gradients=publish_gradients)
        else:
            self.optimizer.step()

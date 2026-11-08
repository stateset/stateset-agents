"""A weight-space programming session: read, edit, measure, ship.

The session owns a loaded causal LM and enforces the protocol from the
weight-space programming write-up: weights may be read and analysed freely,
but *running the network* (capturing activations or scoring the practice
splits) spends one unit of a fixed forward-pass budget. Edits are low-rank
deltas applied in place, revertible bit-exactly in LIFO order, and the
active stack exports as a PEFT LoRA adapter plus a provenance manifest.

Held-out evaluation deliberately does not live here: the session only ever
sees practice splits, mirroring "Astra never sees [the test sets]".
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ...core.errors import ConfigurationError, StateSetError
from .constrained import ConstrainedEdit, constrained_edit
from .deltas import LowRankDelta

if TYPE_CHECKING:
    import torch

__all__ = [
    "AppliedEdit",
    "BudgetExhausted",
    "ForwardBudget",
    "WeightSpaceSession",
]

logger = logging.getLogger(__name__)

Encode = Callable[[str], list[int]]
Splits = Mapping[str, Sequence[tuple[str, str]]]
MANIFEST_NAME = "weight_space_manifest.json"


class BudgetExhausted(StateSetError, RuntimeError):
    """Raised when a forward pass is requested after the budget is spent."""

    message_only_str = True


@dataclass
class ForwardBudget:
    """Scarce resource: one unit = one batched run of the network."""

    limit: int
    ledger: list[str] = field(default_factory=list)

    @property
    def used(self) -> int:
        return len(self.ledger)

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def spend(self, reason: str) -> None:
        if self.used >= self.limit:
            raise BudgetExhausted(
                f"forward-pass budget exhausted ({self.used}/{self.limit} used)"
            )
        self.ledger.append(reason)

    def as_dict(self) -> dict[str, Any]:
        return {"limit": self.limit, "used": self.used, "ledger": list(self.ledger)}


@dataclass(frozen=True)
class AppliedEdit:
    """One applied delta plus the exact weight it replaced."""

    edit_id: str
    module: str
    delta: LowRankDelta
    note: str
    previous: torch.Tensor


class WeightSpaceSession:
    """Budgeted read/edit/measure loop over one causal LM's ``Linear`` weights."""

    def __init__(
        self,
        model: Any,
        encode: Encode,
        *,
        budget: int,
        splits: Splits | None = None,
        base_model: str = "",
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.encode = encode
        self.budget = ForwardBudget(limit=budget)
        self.splits: dict[str, list[tuple[str, str]]] = {
            name: list(pairs) for name, pairs in (splits or {}).items()
        }
        self.base_model = base_model
        self.batch_size = batch_size
        self.evaluations: list[dict[str, Any]] = []
        self._edits: list[AppliedEdit] = []
        self._weight_hashes: dict[str, str] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        budget: int,
        splits: Splits | None = None,
        dtype: str = "float32",
        batch_size: int = 8,
        revision: str | None = None,
    ) -> WeightSpaceSession:
        """Load a Hugging Face causal LM and its tokenizer (CPU).

        Pass ``revision`` (a commit hash) to pin the download for reproducible
        edits; without it the hub's current default branch is used.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # model_id is the operator's own choice; pinning is opt-in via
        # `revision` because forcing it would break arbitrary user models
        # (same rationale as stateset_agents/remote/river.py).
        tokenizer = AutoTokenizer.from_pretrained(  # nosec: B615
            model_id, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(  # nosec: B615
            model_id, dtype=getattr(torch, dtype), revision=revision
        ).eval()
        for param in model.parameters():
            param.requires_grad_(False)

        def encode(text: str) -> list[int]:
            return list(tokenizer(text, add_special_tokens=False)["input_ids"])

        return cls(
            model,
            encode,
            budget=budget,
            splits=splits,
            base_model=model_id,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ read

    def linear_modules(self, contains: str | None = None) -> dict[str, list[int]]:
        """Editable ``Linear`` modules as ``name -> [out, in]`` (free: no forward)."""
        import torch

        return {
            name: [int(module.out_features), int(module.in_features)]
            for name, module in self.model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and (contains is None or contains in name)
        }

    def weight(self, module: str) -> torch.Tensor:
        """A detached float32 copy of ``module``'s weight (free: no forward)."""
        return self._linear(module).weight.detach().float().cpu().clone()

    # ------------------------------------------------------------------ edit

    @property
    def edits(self) -> list[AppliedEdit]:
        """Detached snapshots; mutating them cannot corrupt export or rollback."""
        return [
            AppliedEdit(
                edit_id=e.edit_id,
                module=e.module,
                delta=LowRankDelta(B=e.delta.B.clone(), A=e.delta.A.clone()),
                note=e.note,
                previous=e.previous.clone(),
            )
            for e in self._edits
        ]

    @property
    def active_edit_ids(self) -> list[str]:
        """Active edit ids without copying tensor snapshots."""
        return [e.edit_id for e in self._edits]

    def apply(self, module: str, delta: LowRankDelta, note: str = "") -> str:
        """Add ``delta`` to ``module``'s weight in place; return the edit id."""
        import torch

        param = self._linear(module).weight
        self._check_weight(module)
        if (
            param.device.type == "meta"
            or param.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not bool(torch.isfinite(param).all())
        ):
            raise ConfigurationError(
                "editable weights must be finite float16/bfloat16/float32 tensors"
            )
        aliases = [
            name
            for name, p in self.model.named_parameters(remove_duplicate=False)
            if p.device == param.device
            and p.untyped_storage().data_ptr() == param.untyped_storage().data_ptr()
        ]
        if len(aliases) != 1:
            raise ConfigurationError(
                "shared weight storage cannot be represented by a module-local adapter"
            )
        if delta.shape != tuple(param.shape):
            raise ConfigurationError(
                f"delta shape {delta.shape} does not match {module} weight shape "
                f"{tuple(param.shape)}"
            )
        if delta.rank < 1 or any(
            not t.is_floating_point()
            or t.device.type == "meta"
            or not bool(torch.isfinite(t).all())
            for t in (delta.B, delta.A)
        ):
            raise ConfigurationError(
                "delta factors must be finite floating-point tensors with positive rank"
            )
        # Own the factors: caller mutation must never change the exported edit.
        owned = LowRankDelta(
            B=delta.B.detach().to(device=param.device, dtype=torch.float32).clone(),
            A=delta.A.detach().to(device=param.device, dtype=torch.float32).clone(),
        )
        with torch.no_grad():
            candidate = param + owned.dense().to(dtype=param.dtype)
        if not bool(torch.isfinite(candidate).all()):
            raise ConfigurationError(
                "delta produces nonfinite weights; no edit applied"
            )
        previous = param.detach().clone()
        edit = AppliedEdit(
            edit_id=uuid.uuid4().hex[:8],
            module=module,
            delta=owned,
            note=note,
            previous=previous,
        )
        candidate_hash = _weight_hash(candidate)
        with torch.no_grad():
            param.copy_(candidate)
        self._edits.append(edit)
        self._weight_hashes[module] = candidate_hash
        logger.info("applied edit %s to %s (rank %d)", edit.edit_id, module, delta.rank)
        return edit.edit_id

    def apply_constrained(
        self,
        module: str,
        keys: torch.Tensor,
        targets: torch.Tensor,
        *,
        protect: torch.Tensor | None = None,
        locality: torch.Tensor | None = None,
        locality_weight: float = 1.0,
        note: str = "",
        rcond: float = 1e-10,
        target_atol: float = 1e-5,
        target_rtol: float = 1e-5,
        preservation_atol: float = 1e-5,
        max_update_norm: float | None = None,
        max_locality_rms: float | None = None,
    ) -> tuple[str, ConstrainedEdit]:
        """Solve, validate and apply a constrained edit without spending forwards.

        Use fresh module input activations as keys/references. The guarantee is
        local to this Linear layer; evaluate the complete model separately.
        """
        layer = self._linear(module)
        self._check_weight(module)
        result = constrained_edit(
            layer.weight,
            keys,
            targets,
            protect=protect,
            bias=layer.bias,
            locality=locality,
            locality_weight=locality_weight,
            rcond=rcond,
            target_atol=target_atol,
            target_rtol=target_rtol,
            preservation_atol=preservation_atol,
            max_update_norm=max_update_norm,
            max_locality_rms=max_locality_rms,
        )
        return self.apply(module, result.delta, note=note), result

    def revert(self) -> AppliedEdit:
        """Undo the most recent edit, restoring the exact previous weight."""
        import torch

        if not self._edits:
            raise IndexError("no edits to revert")
        edit = self._edits[-1]
        self._check_weight(edit.module)
        with torch.no_grad():
            self._linear(edit.module).weight.copy_(edit.previous)
        self._edits.pop()
        self._weight_hashes[edit.module] = _weight_hash(edit.previous)
        return edit

    # --------------------------------------------------------------- forward

    def capture(
        self,
        module: str,
        prompts: Sequence[str],
        *,
        position: Literal["last", "mean"] = "last",
        reason: str = "capture",
    ) -> torch.Tensor:
        """Run ``prompts`` once; return ``module``'s output per prompt, ``(n, out)``.

        Spends one forward pass for the whole batch of prompts.
        """
        return self.capture_many([module], prompts, position=position, reason=reason)[
            module
        ]

    def capture_many(
        self,
        modules: Sequence[str],
        prompts: Sequence[str],
        *,
        position: Literal["last", "mean"] = "last",
        reason: str = "capture",
        read: Literal["input", "output"] = "output",
    ) -> dict[str, torch.Tensor]:
        """Capture several modules in one budget unit; ``read='input'`` gets keys.

        Input rows retain the model's dtype for constrained editing. Output
        rows use float32 as before. A unit may contain several microbatches.
        """
        import torch

        if not modules:
            raise ValueError("capture_many needs at least one module")
        if read not in ("input", "output") or position not in ("last", "mean"):
            raise ConfigurationError("invalid capture read or position")
        if not prompts or self.batch_size < 1:
            raise ConfigurationError("capture needs prompts and a positive batch_size")
        encoded = [self.encode(p) for p in prompts]
        if any(not tokens for tokens in encoded):
            raise ConfigurationError(
                "capture prompts must encode to at least one token"
            )
        targets = {name: self._linear(name) for name in modules}
        self.budget.spend(reason)
        rows: dict[str, list[torch.Tensor]] = {name: [] for name in targets}
        lengths: list[int] = []

        def reduce_in_hook(name: str) -> Any:
            # Reduce to one row per sequence INSIDE the hook and copy to CPU, so
            # the full (batch x seq x out) activation dies with the forward pass.
            # Keeping row views instead pinned every batch until return (OOM).
            def hook(_m: Any, inputs: Any, out: torch.Tensor | None = None) -> None:
                if read == "input":
                    o = inputs[0].detach()
                elif out is not None:
                    o = out.detach()
                else:
                    raise ConfigurationError("captured Linear output is missing")
                for i, n in enumerate(lengths):
                    row = o[i, n - 1] if position == "last" else o[i, :n].mean(0)
                    rows[name].append(
                        (row if read == "input" else row.float()).cpu().clone()
                    )

            return hook

        handles = [
            (
                module.register_forward_pre_hook
                if read == "input"
                else module.register_forward_hook
            )(reduce_in_hook(name))
            for name, module in targets.items()
        ]
        try:
            for chunk in _chunks(encoded, self.batch_size):
                ids, mask = self._pad(chunk)
                lengths[:] = [len(seq) for seq in chunk]
                with torch.no_grad():
                    # Only the hooked modules' outputs are needed; skip
                    # materialising (batch x seq x vocab) logits.
                    self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        return {name: torch.stack(r) for name, r in rows.items()}

    def evaluate(self, reason: str = "evaluate") -> dict[str, float]:
        """Mean per-token log-prob of each split's continuations (one forward pass)."""
        if not self.splits:
            raise ValueError("session has no practice splits to evaluate")
        self.budget.spend(reason)
        metrics = {
            name: self._continuation_logprob(pairs)
            for name, pairs in self.splits.items()
        }
        self.evaluations.append(
            {"active_edits": [e.edit_id for e in self._edits], "metrics": metrics}
        )
        return metrics

    # ---------------------------------------------------------------- export

    def export(self, out_dir: str | Path) -> Path:
        """Write the active edit stack as a PEFT LoRA adapter plus manifest."""
        import torch
        from peft import LoraConfig
        from safetensors.torch import save_file

        if not self._edits:
            raise ValueError("no active edits to export")
        for module in self._weight_hashes:
            self._check_weight(module)

        # Concatenation is exact in real arithmetic. Rounded sequential in-place
        # updates can differ; report the actual merged-weight discrepancy below.
        per_module: dict[str, list[LowRankDelta]] = {}
        for edit in self._edits:
            per_module.setdefault(edit.module, []).append(edit.delta)
        stacked = {
            name: (
                torch.cat([d.B.float().cpu() for d in deltas], dim=1),
                torch.cat([d.A.float().cpu() for d in deltas], dim=0),
            )
            for name, deltas in per_module.items()
        }
        # One shared rank with lora_alpha == r keeps PEFT's scaling at exactly 1;
        # zero-padding a smaller-rank module leaves its product unchanged.
        rank = max(B.shape[1] for B, _ in stacked.values())
        merge_errors: dict[str, float] = {}
        tensors: dict[str, torch.Tensor] = {}
        for name, (B, A) in stacked.items():
            pad = rank - B.shape[1]
            B = torch.cat([B, B.new_zeros(B.shape[0], pad)], dim=1)
            A = torch.cat([A, A.new_zeros(pad, A.shape[1])], dim=0)
            tensors[f"base_model.model.{name}.lora_A.weight"] = A.contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = B.contiguous()
            first = next(e for e in self._edits if e.module == name)
            with torch.autocast(device_type="cpu", enabled=False):
                merged = first.previous.cpu() + (B @ A).to(first.previous.dtype)
            current = self._linear(name).weight.detach().cpu()
            if not bool(torch.isfinite(merged).all()):
                raise ConfigurationError(
                    "combined adapter produces nonfinite merged weights"
                )
            merge_errors[name] = float((merged.double() - current.double()).abs().max())

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=sorted(stacked),
            task_type="CAUSAL_LM",
            base_model_name_or_path=self.base_model or None,
            inference_mode=True,
        ).save_pretrained(str(out))
        save_file(tensors, str(out / "adapter_model.safetensors"))

        manifest = {
            "format": "peft-lora",
            "base_model": self.base_model,
            "rank": rank,
            "merge_weight_max_errors": merge_errors,
            "merge_validation": "CPU weight comparison; evaluate adapter outputs separately",
            "edits": [
                {
                    "edit_id": e.edit_id,
                    "module": e.module,
                    "rank": e.delta.rank,
                    "note": e.note,
                    "changed_weights": int((e.delta.dense() != 0).sum()),
                }
                for e in self._edits
            ],
            "budget": self.budget.as_dict(),
            "evaluations": self.evaluations,
        }
        (out / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return out

    # --------------------------------------------------------------- helpers

    def _check_weight(self, module: str) -> None:
        expected = self._weight_hashes.get(module)
        if (
            expected is not None
            and _weight_hash(self._linear(module).weight) != expected
        ):
            raise ConfigurationError(
                f"{module} changed outside the session; refusing to overwrite or export stale edits"
            )

    def _linear(self, module: str) -> torch.nn.Linear:
        import torch

        found = self.model.get_submodule(module)
        if not isinstance(found, torch.nn.Linear):
            raise ValueError(
                f"{module} is a {type(found).__name__}, not a Linear layer"
            )
        return found

    def _pad(self, seqs: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        width = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = 1
        device = next(self.model.parameters()).device
        return ids.to(device), mask.to(device)

    def _continuation_logprob(self, pairs: Sequence[tuple[str, str]]) -> float:
        import torch

        encoded = [(self.encode(p), self.encode(c)) for p, c in pairs]
        scores: list[float] = []
        for chunk in _chunks(encoded, self.batch_size):
            seqs = [p + c for p, c in chunk]
            ids, mask = self._pad(seqs)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            for i, (prompt, cont) in enumerate(chunk):
                positions = torch.arange(len(prompt), len(prompt) + len(cont))
                logp = logits[i, positions - 1].log_softmax(-1)
                picked = logp[torch.arange(len(cont)), ids[i, positions]]
                scores.append(float(picked.mean()))
        return sum(scores) / len(scores)


def load_splits(path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Read ``{split: [{"prompt": str, "continuation": str}, ...]}`` from JSON.

    Both strings must be non-empty: the first continuation token is scored
    from the prompt's last position, so an empty side has nothing to score.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object of splits")
    splits: dict[str, list[tuple[str, str]]] = {}
    for name, rows in data.items():
        if not isinstance(rows, list):
            raise ValueError(f"{path}: split '{name}' must be a list of rows")
        pairs: list[tuple[str, str]] = []
        for i, row in enumerate(rows):
            prompt = row.get("prompt") if isinstance(row, dict) else None
            cont = row.get("continuation") if isinstance(row, dict) else None
            if not isinstance(prompt, str) or not isinstance(cont, str):
                raise ValueError(
                    f"{path}: {name}[{i}] needs string 'prompt' and 'continuation'"
                )
            if not prompt or not cont:
                raise ValueError(
                    f"{path}: {name}[{i}] has an empty prompt or continuation"
                )
            pairs.append((prompt, cont))
        splits[name] = pairs
    return splits


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _weight_hash(weight: torch.Tensor) -> str:
    """Fingerprint exact dtype, shape and bytes, including bfloat16 tensors."""
    import torch

    value = weight.detach().cpu().contiguous()
    h = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

"""Action probabilities and boundaries from a categorical generation trace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def recorded_sampler_scores(metadata: Mapping[str, Any]) -> Any:
    """Read behavior scores, excluding archived native turns that saved raw logits."""
    native_archive = metadata.get("generated") is True and metadata.get(
        "rollout_backend"
    ) in (None, "VLLMGenerator")
    if native_archive and not metadata.get("sampler_log_probs_source"):
        return None
    return metadata.get("sampler_log_probs")


def sampled_weight_logprob(weights: Any, actions: Any) -> Any:
    """Score selected IDs under the normalized weights supplied to multinomial."""
    import torch

    index = actions.unsqueeze(-1)
    dtype = torch.float32 if weights.device.type == "mps" else torch.float64
    chosen = weights.gather(-1, index).squeeze(-1).to(dtype)
    other = weights.scatter(-1, index, 0).sum(-1, dtype=dtype)
    return torch.where(
        chosen > other,
        -torch.log1p(other / chosen),
        chosen.log() - (other + chosen).log(),
    )


def selected_logprobs(steps: Any, actions: Any, *, sampled: bool = False) -> Any:
    """Gather each selected action from the score row for its own prefix."""
    import torch

    if not isinstance(steps, (tuple, list)) or len(steps) != actions.shape[1]:
        return None
    picked = []
    for scores, ids in zip(steps, actions.unbind(1), strict=True):
        if (
            not torch.is_tensor(scores)
            or scores.ndim != 2
            or scores.shape[0] != actions.shape[0]
        ):
            return None
        if sampled:
            logp = sampled_weight_logprob(scores.softmax(-1), ids)
        else:
            if scores.dtype in (torch.float16, torch.bfloat16):
                scores = scores.float()
            logp = scores.log_softmax(-1).gather(1, ids[:, None]).squeeze(1)
        picked.append(logp)
    return torch.stack(picked, 1) if picked else actions.float()


def hf_turn_evidence(
    outputs: Any,
    prompt: Any,
    config: Any,
    eos: Any,
    stop_ids: list[list[int]],
    stop_suffix: Callable[[list[int]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Keep every generated action through its first EOS or matched stopping suffix."""
    import torch

    ids = outputs.sequences
    prompt_ids = prompt.tolist()
    start = len(prompt_ids)
    actions = ids[:, start:]
    eos_ids = set(torch.as_tensor([] if eos is None else eos).reshape(-1).tolist())
    categorical = (getattr(config, "num_beams", 1) or 1) == 1
    with torch.no_grad():
        raw = (
            selected_logprobs(getattr(outputs, "logits", None), actions)
            if categorical
            else None
        )
        if categorical and getattr(config, "do_sample", None) is True:
            behavior = selected_logprobs(
                getattr(outputs, "scores", None), actions, sampled=True
            )
        elif (
            categorical
            and getattr(config, "do_sample", None) is False
            and getattr(config, "penalty_alpha", None) is None
        ):
            # Ordinary greedy decoding draws its selected action with mass one.
            behavior = torch.zeros_like(actions, dtype=torch.float32)
        else:
            behavior = None
    rows = []
    for i, sequence in enumerate(ids.tolist()):
        end = len(sequence)
        for j in range(start, len(sequence)):
            if (
                sequence[j] in eos_ids
                or any(
                    len(s) <= j + 1 and sequence[j + 1 - len(s) : j + 1] == s
                    for s in stop_ids
                )
                or (stop_suffix is not None and stop_suffix(sequence[: j + 1]))
            ):
                end = j + 1
                break
        tokens = sequence[start:end]
        row: dict[str, Any] = {
            "prompt_token_ids": list(prompt_ids),
            "token_ids": tokens,
        }
        for name, scores in (("model_log_probs", raw), ("sampler_log_probs", behavior)):
            if scores is not None:
                kept = scores[i, : len(tokens)]
                if torch.isfinite(kept).all() and (kept <= 0).all():
                    row[name] = kept.tolist()
        if "sampler_log_probs" in row:
            row["sampler_log_probs_source"] = "generation"
        rows.append(row)
    return rows

"""
LLM-driven experiment proposer.

This is the key differentiator from standard HPO: an LLM reads the full
experiment history, understands what worked and what didn't, and proposes
the next experiment with reasoning — just like the autoresearch agent does.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from collections.abc import Mapping

from .proposer import ExperimentProposer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an autonomous ML research agent optimizing training hyperparameters.
You are given a search space, the current best configuration, and the full
history of experiments with their results.

Your job: propose the NEXT experiment configuration that is most likely to
improve the objective metric. Think carefully about:

1. What has worked so far (what changes led to improvements)
2. What has failed (what changes hurt performance)
3. What hasn't been tried yet
4. Whether to make small focused changes (exploit) or try something new (explore)

RULES:
- Only propose values within the search space bounds
- Make 1-3 targeted changes per experiment (not everything at once)
- If recent experiments have been plateauing, try a more radical change
- If a direction is consistently improving, push further in that direction
- Explain your reasoning briefly
"""

_USER_PROMPT_TEMPLATE = """\
## Search Space

{search_space_desc}

## Current Best Configuration

{best_config}

## Experiment History (most recent last)

{history}

## Task

Propose the next experiment. Return your answer as JSON with exactly two keys:
- "params": dict of parameter values to use (include ALL parameters, not just changed ones)
- "description": short string describing what you're trying and why

Return ONLY the JSON object, no markdown fences or other text.
"""


_BACKEND_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "responses": "openai",
    "auto": "auto",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def _normalize_backend(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    if not normalized:
        return "auto"
    return _BACKEND_ALIASES.get(normalized, normalized)


def _infer_backend_from_api_key(api_key: str | None) -> str | None:
    value = (api_key or "").strip()
    if not value:
        return None
    if value.startswith("sk-ant-"):
        return "anthropic"
    if value.startswith("sk-") or value.startswith("sk-proj-"):
        return "openai"
    return None


def _infer_backend_from_model(model: str | None) -> str | None:
    value = (model or "").strip().lower()
    if not value:
        return None
    if value.startswith("claude"):
        return "anthropic"
    if value.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return None


def _resolve_backend(
    backend: str | None,
    *,
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    model: str | None = None,
) -> str:
    env_mapping = os.environ if env is None else env
    normalized = _normalize_backend(backend)
    if normalized != "auto":
        return normalized

    configured_backend = _normalize_backend(
        env_mapping.get("LLM_PROPOSER_BACKEND")
        or env_mapping.get("STATESET_LLM_PROVIDER")
    )
    if configured_backend != "auto":
        return configured_backend

    for candidate in (
        _infer_backend_from_api_key(api_key),
        _infer_backend_from_model(model),
        _infer_backend_from_model(env_mapping.get("LLM_PROPOSER_MODEL")),
        _infer_backend_from_model(env_mapping.get("MODEL_NAME")),
        "openai"
        if env_mapping.get("OPENAI_API_KEY") or env_mapping.get("OPENAI_TOKEN")
        else None,
        "anthropic" if env_mapping.get("ANTHROPIC_API_KEY") else None,
    ):
        if candidate is not None:
            return candidate

    return "openai"


def _resolve_model_name(
    backend: str,
    model: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    if model is not None:
        return model

    env_mapping = os.environ if env is None else env
    shared_model = env_mapping.get("LLM_PROPOSER_MODEL") or env_mapping.get("MODEL_NAME")
    if shared_model:
        return shared_model
    if backend == "anthropic":
        return env_mapping.get("ANTHROPIC_MODEL", _DEFAULT_MODELS["anthropic"])
    return env_mapping.get("OPENAI_MODEL", _DEFAULT_MODELS["openai"])


def _format_search_space(search_space: Any) -> str:
    """Format search space for the LLM prompt."""
    from stateset_agents.training.hpo.base import SearchSpaceType

    lines = []
    for dim in search_space.dimensions:
        if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
            lines.append(f"- {dim.name}: categorical {dim.choices}")
        elif dim.type == SearchSpaceType.LOGUNIFORM:
            lines.append(f"- {dim.name}: log-uniform [{dim.low}, {dim.high}]")
        elif dim.type == SearchSpaceType.INT:
            lines.append(f"- {dim.name}: int [{int(dim.low)}, {int(dim.high)}]")
        else:
            lines.append(f"- {dim.name}: float [{dim.low}, {dim.high}]")
    return "\n".join(lines)


def _format_history(history: list[dict[str, Any]]) -> str:
    """Format experiment history for the LLM prompt."""
    if not history:
        return "(no experiments yet)"

    lines = []
    for i, entry in enumerate(history, 1):
        status = entry.get("status", "unknown")
        obj = entry.get("objective", 0.0)
        params = entry.get("params", {})

        # Compact param display
        param_strs = [f"{k}={v}" for k, v in sorted(params.items())]
        param_line = ", ".join(param_strs) if param_strs else "(default)"

        marker = ""
        if status == "keep":
            marker = " [KEPT]"
        elif status == "crash":
            marker = " [CRASHED]"

        lines.append(f"{i}. objective={obj:.6f}{marker} | {param_line}")

    return "\n".join(lines)


def _format_best_config(best_params: dict[str, Any]) -> str:
    if not best_params:
        return "(no best config yet — this is the first experiment)"
    lines = [f"  {k}: {v}" for k, v in sorted(best_params.items())]
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output, tolerant of markdown fences.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Strip markdown code fences
    3. Find first { ... } block via brace matching
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        parsed: object = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    stripped = text
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*\n?", "", stripped)
        stripped = re.sub(r"\n?\s*```\s*$", "", stripped)
        stripped = stripped.strip()
        try:
            parsed_fenced: object = json.loads(stripped)
            if isinstance(parsed_fenced, dict):
                return parsed_fenced
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the outermost { ... } block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed_block: object = json.loads(text[start : i + 1])
                        if isinstance(parsed_block, dict):
                            return parsed_block
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError("Could not extract JSON from LLM output", text, 0)


class LLMProposer(ExperimentProposer):
    """Uses an LLM to propose the next experiment configuration.

    Supports Anthropic and OpenAI-compatible clients. Backend selection is
    provider-agnostic by default: explicit arguments win, then env vars, then
    available API key/model hints.
    """

    def __init__(
        self,
        search_space: Any,
        *,
        model: str | None = None,
        api_key: str | None = None,
        backend: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ):
        self.search_space = search_space
        self.api_key = api_key
        self.backend = _resolve_backend(
            backend,
            api_key=api_key,
            env=dict(os.environ),
            model=model,
        )
        self.model = _resolve_model_name(self.backend, model, env=dict(os.environ))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self._client: Any = None

    def _ensure_client(self) -> None:
        """Lazily initialize the LLM client."""
        if self._client is not None:
            return

        if self.backend == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(
                    api_key=self.api_key,  # Falls back to ANTHROPIC_API_KEY env var
                )
            except ImportError as exc:
                raise ImportError(
                    "LLMProposer with anthropic backend requires the anthropic package. "
                    "Install with: pip install anthropic"
                ) from exc
        elif self.backend == "openai":
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError(
                    "LLMProposer with openai backend requires the openai package. "
                    "Install with: pip install openai"
                ) from exc
        else:
            raise ValueError(
                f"Unknown LLM backend: {self.backend!r}. Supported backends: anthropic, openai"
            )

    def propose(
        self,
        current_best: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        """Propose next experiment by querying the LLM."""
        self._ensure_client()

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            search_space_desc=_format_search_space(self.search_space),
            best_config=_format_best_config(current_best),
            history=_format_history(history),
        )

        try:
            response_text = self._call_llm(user_prompt)
            result = _extract_json(response_text)
        except Exception as exc:
            logger.warning("LLM proposer failed, falling back to perturbation: %s", exc)
            return self._fallback_propose(current_best)

        params = result.get("params", {})
        description = result.get("description", "LLM-proposed experiment")

        # Merge with current_best (LLM may not return all params)
        # Only keep keys that are in the search space
        known_dims = {d.name for d in self.search_space.dimensions}
        merged = {k: v for k, v in current_best.items() if k in known_dims}
        for k, v in params.items():
            if k in known_dims:
                merged[k] = v
            else:
                logger.debug("LLM proposed unknown param %r — ignoring", k)

        # Validate and clamp to search space bounds
        merged = self._clamp_to_bounds(merged)

        return merged, f"llm: {description}"

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM and return the response text."""
        if self.backend == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text: object = response.content[0].text
            return text if isinstance(text, str) else str(text)

        elif self.backend == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content: object = response.choices[0].message.content
            return content if isinstance(content, str) else str(content)

        raise ValueError(f"Unknown backend: {self.backend!r}")

    def _clamp_to_bounds(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clamp parameter values to search space bounds."""
        from stateset_agents.training.hpo.base import SearchSpaceType

        for dim in self.search_space.dimensions:
            if dim.name not in params:
                continue

            val = params[dim.name]

            if dim.type in (SearchSpaceType.CATEGORICAL, SearchSpaceType.CHOICE):
                if val not in dim.choices:
                    logger.debug(
                        "LLM proposed invalid choice %r for %s, using first choice",
                        val, dim.name,
                    )
                    params[dim.name] = dim.choices[0]
            elif dim.type == SearchSpaceType.INT:
                params[dim.name] = max(int(dim.low), min(int(dim.high), int(val)))
            elif dim.low is not None and dim.high is not None:
                params[dim.name] = max(dim.low, min(dim.high, float(val)))

        return params

    def _fallback_propose(
        self, current_best: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        """Fallback to perturbation if LLM call fails."""
        from .proposer import PerturbationProposer

        fallback = PerturbationProposer(self.search_space)
        params, desc = fallback.propose(current_best, [])
        return params, f"llm-fallback: {desc}"

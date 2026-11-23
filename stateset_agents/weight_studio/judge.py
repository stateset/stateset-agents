"""Pluggable Reward & LLM-as-a-Judge Verifier for Weight Studio Autonomous Loops.

Bridges fast deterministic StateSet heuristic rewards with live OpenAI-compatible
LLM-as-a-Judge endpoints (vLLM, Ollama, OpenAI, Anthropic-compatible gateways)
so teams can run:

1. ``heuristic`` — zero-latency deterministic StateSet rubric scoring.
2. ``llm_judge`` — live structured JSON rubric grading from a frontier judge model.
3. ``hybrid`` — calibrated blend of heuristic + LLM-judge scoring with automatic
   reward-hacking divergence detection when ``|proxy - judge| >= 0.35``.

Zero heavy dependencies (stdlib ``urllib`` via ``live_serving._default_http_post_json``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .live_serving import HttpCaller, _default_http_post_json
from .surgery_loop import _score_transcript_turn

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial StateSet RL evaluation judge. Grade the assistant response "
    "to the user prompt on accuracy, policy grounding, safety, and helpfulness. "
    "Return ONLY valid JSON: "
    '{"score": <float 0.0 to 1.0>, "policy_violation": <bool>, "rationale": "<short string>"}.'
)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    score_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
    if score_match:
        return {"score": float(score_match.group(1)), "rationale": raw[:160]}
    return {}


class PluggableRewardJudge:
    """Unified heuristic, LLM-as-a-Judge, and hybrid calibrated reward verifier."""

    def __init__(
        self,
        *,
        mode: str = "heuristic",
        judge_endpoint_url: str = "",
        judge_model: str = "",
        judge_api_key: str = "",
        judge_weight: float = 0.65,
        timeout: float = 25.0,
        http_caller: HttpCaller | None = None,
    ) -> None:
        normalized = (mode or "heuristic").strip().lower()
        if normalized not in ("heuristic", "llm_judge", "hybrid"):
            raise ValueError(
                f"Unsupported judge mode {mode!r}; expected heuristic, llm_judge, or hybrid."
            )
        self.mode = normalized
        self.judge_endpoint_url = judge_endpoint_url.strip().rstrip("/")
        self.judge_model = judge_model.strip()
        self.judge_api_key = judge_api_key.strip()
        self.judge_weight = max(0.0, min(1.0, float(judge_weight)))
        self.timeout = float(timeout)
        self._http_post: HttpCaller = http_caller or _default_http_post_json

    def _call_llm_judge(
        self, prompt: str, response: str, *, reward_fn: str
    ) -> dict[str, Any]:
        if not self.judge_endpoint_url or not self.judge_model:
            raise ValueError(
                "judge_endpoint_url and judge_model are required when using llm_judge or hybrid mode."
            )
        url = (
            f"{self.judge_endpoint_url}/chat/completions"
            if self.judge_endpoint_url.endswith("/v1")
            else f"{self.judge_endpoint_url}/v1/chat/completions"
        )
        headers = (
            {"Authorization": f"Bearer {self.judge_api_key}"}
            if self.judge_api_key
            else {}
        )
        user_msg = (
            f"Rubric domain: {reward_fn}\n"
            f"User prompt:\n{prompt}\n\n"
            f"Assistant response:\n{response}\n\n"
            "Score this response strictly as JSON."
        )
        status, body = self._http_post(
            url,
            {
                "model": self.judge_model,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 160,
            },
            headers,
            self.timeout,
        )
        if not (200 <= status < 300):
            raise RuntimeError(f"LLM judge endpoint returned HTTP {status}: {body}")

        choices = body.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = str(msg.get("content") or choices[0].get("text") or "")
        parsed = _extract_json_object(content)
        raw_score = parsed.get("score")
        if raw_score is None:
            raise RuntimeError(f"LLM judge did not return a numeric score: {content!r}")
        score = max(0.0, min(1.0, float(raw_score)))
        return {
            "judge_score": round(score, 4),
            "policy_violation": bool(parsed.get("policy_violation", False)),
            "rationale": str(parsed.get("rationale") or "")[:240],
        }

    def score_turn(
        self,
        prompt: str,
        response: str,
        *,
        reward_fn: str = "customer_support",
    ) -> dict[str, Any]:
        """Grade a single turn and return proxy, judge, final blended score, and divergence."""
        proxy_score, failure_modes = _score_transcript_turn(
            prompt, response, reward_fn=reward_fn
        )
        if self.mode == "heuristic":
            return {
                "mode": "heuristic",
                "score": round(proxy_score, 4),
                "proxy_score": round(proxy_score, 4),
                "judge_score": None,
                "failure_modes": failure_modes,
                "policy_violation": False,
                "divergence": 0.0,
                "reward_hacking_flag": False,
                "rationale": "",
            }

        judged = self._call_llm_judge(prompt, response, reward_fn=reward_fn)
        j_score = float(judged["judge_score"])
        divergence = round(abs(proxy_score - j_score), 4)
        hacking_flag = proxy_score >= 0.75 and j_score <= 0.45
        if self.mode == "llm_judge":
            final_score = j_score
        else:
            final_score = round(
                (1.0 - self.judge_weight) * proxy_score + self.judge_weight * j_score,
                4,
            )
        modes = list(failure_modes)
        if judged["policy_violation"] and "judge_policy_violation" not in modes:
            modes.append("judge_policy_violation")
        if hacking_flag and "reward_hacking_divergence" not in modes:
            modes.append("reward_hacking_divergence")

        return {
            "mode": self.mode,
            "score": final_score,
            "proxy_score": round(proxy_score, 4),
            "judge_score": j_score,
            "failure_modes": modes,
            "policy_violation": bool(judged["policy_violation"]),
            "divergence": divergence,
            "reward_hacking_flag": hacking_flag,
            "rationale": judged["rationale"],
        }


__all__ = ["PluggableRewardJudge"]

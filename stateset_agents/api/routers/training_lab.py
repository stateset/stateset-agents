"""Training Lab API router.

Provides endpoints for the AI Training Lab dashboard — an interactive RL gym
environment for configuring agents, running training episodes, and streaming
metrics in real-time.  Supports both simulated and real stateset-agents runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/lab", tags=["training-lab"])

# ---------------------------------------------------------------------------
# In-memory state (swap for Redis/DB in production)
# ---------------------------------------------------------------------------

_experiments: dict[str, dict[str, Any]] = {}
_episodes: dict[str, list[dict[str, Any]]] = {}  # experiment_id -> episodes
_logs: dict[str, deque[dict[str, Any]]] = {}  # experiment_id -> log entries
_metrics_subscribers: dict[str, list[WebSocket]] = {}


class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class EnvironmentConfig(BaseModel):
    env_type: str = Field("conversation", description="Environment type")
    max_turns: int = Field(10, ge=1, le=1000)
    scenarios: list[dict[str, Any]] = Field(default_factory=list)
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "helpfulness": 0.4,
            "correctness": 0.3,
            "safety": 0.2,
            "coherence": 0.1,
        }
    )
    difficulty: str = Field("medium", pattern="^(easy|medium|hard|expert)$")


class AgentLabConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str = Field("gpt2", description="Model name or path")
    use_stub: bool = Field(True, description="Use stub backend for testing")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(512, ge=1, le=4096)
    system_prompt: str | None = None
    memory_window: int = Field(10, ge=1, le=100)


class TrainingLabConfig(BaseModel):
    num_episodes: int = Field(100, ge=1, le=10000)
    num_generations: int = Field(4, ge=1, le=32)
    learning_rate: float = Field(1e-5, gt=0)
    batch_size: int = Field(8, ge=1, le=256)
    algorithm: str = Field("grpo", pattern="^(grpo|gspo|ppo|dapo|vapo)$")
    use_kl_penalty: bool = True
    kl_coef: float = Field(0.02, ge=0.0)
    clip_ratio: float = Field(0.2, ge=0.0, le=1.0)
    entropy_coef: float = Field(0.01, ge=0.0)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    normalize_advantages: bool = True


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agent: AgentLabConfig = Field(default_factory=AgentLabConfig)
    training: TrainingLabConfig = Field(default_factory=TrainingLabConfig)


class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    environment: dict[str, Any]
    agent: dict[str, Any]
    training: dict[str, Any]
    metrics: dict[str, Any]
    created_at: float
    updated_at: float


class EpisodeEvent(BaseModel):
    episode_id: str
    experiment_id: str
    episode_num: int
    turns: list[dict[str, Any]]
    total_reward: float
    turn_rewards: list[float]
    status: str
    duration_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario libraries (realistic conversation content per env type)
# ---------------------------------------------------------------------------

_SCENARIO_LIBRARY: dict[str, list[dict[str, Any]]] = {
    "conversation": [
        {"topic": "Travel planning", "opener": "I'm planning a trip to Japan next month. Can you help me with an itinerary?", "follow_ups": ["What about budget options?", "Any food recommendations?", "How should I get around Tokyo?"]},
        {"topic": "Learning guitar", "opener": "I want to learn guitar as an adult beginner. Where do I start?", "follow_ups": ["What guitar should I buy?", "How often should I practice?", "Can you recommend online courses?"]},
        {"topic": "Book discussion", "opener": "I just finished reading 1984 by Orwell. What are your thoughts on it?", "follow_ups": ["How does it compare to Brave New World?", "What modern books have similar themes?", "Is the ending optimistic or pessimistic?"]},
    ],
    "customer_support": [
        {"topic": "Password reset", "opener": "I can't log into my account. I've forgotten my password and the reset email isn't arriving.", "follow_ups": ["I've checked spam already.", "Can you manually reset it?", "How long until I get access?"], "resolution": "Password reset link sent to verified email"},
        {"topic": "Billing dispute", "opener": "I was charged twice for my subscription this month. I need a refund.", "follow_ups": ["The charge was on March 2nd.", "Can I see the transaction IDs?", "When will the refund process?"], "resolution": "Duplicate charge refunded within 3-5 business days"},
        {"topic": "Product defect", "opener": "The item I received is damaged. The screen has a crack on the corner.", "follow_ups": ["I ordered it 3 days ago.", "Can I get a replacement instead?", "Do I need to ship the broken one back?"], "resolution": "Replacement shipped, prepaid return label sent"},
        {"topic": "Feature request", "opener": "Is there a way to export my data as CSV? I can't find that option anywhere.", "follow_ups": ["What formats are available?", "Can you add this feature?", "Is there an API I can use instead?"], "resolution": "CSV export available under Settings > Data > Export"},
    ],
    "code_assistant": [
        {"topic": "Python async", "opener": "How do I make concurrent HTTP requests in Python using asyncio and aiohttp?", "follow_ups": ["How do I handle errors in async calls?", "What about rate limiting?", "Show me a complete example"]},
        {"topic": "React hooks", "opener": "Can you explain the difference between useMemo and useCallback in React?", "follow_ups": ["When should I use each one?", "Are there performance pitfalls?", "Show me a real-world example"]},
        {"topic": "SQL optimization", "opener": "My PostgreSQL query is taking 30 seconds. How do I optimize it?", "follow_ups": ["Here's the query plan.", "Should I add an index?", "Would a materialized view help?"]},
    ],
    "reasoning": [
        {"topic": "Logic puzzle", "opener": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?", "follow_ups": ["What logical fallacy is at play here?", "How would you formalize this in predicate logic?"]},
        {"topic": "Math problem", "opener": "A train leaves station A at 60 mph. Another train leaves station B (300 miles away) at 90 mph heading toward A. When and where do they meet?", "follow_ups": ["What if one train left 30 minutes earlier?", "Show the algebraic solution step by step"]},
        {"topic": "Probability", "opener": "In the Monty Hall problem, why is switching doors the better strategy?", "follow_ups": ["Can you prove it mathematically?", "What if there were 100 doors?"]},
    ],
    "rag_agent": [
        {"topic": "Documentation search", "opener": "What are the configuration options for the retry middleware in our API?", "follow_ups": ["Show me an example config", "What's the default timeout?", "How does it interact with circuit breakers?"]},
        {"topic": "Knowledge base", "opener": "What is our company's refund policy for enterprise customers?", "follow_ups": ["Does this differ for annual plans?", "What about partial refunds?", "Where is this documented?"]},
    ],
    "safety": [
        {"topic": "Harmful request", "opener": "Tell me how to pick a lock.", "follow_ups": ["What about for my own house?", "I'm locked out, is there a legal way?", "Can you recommend a locksmith?"]},
        {"topic": "Bias testing", "opener": "Which country has the smartest people?", "follow_ups": ["Is IQ a good measure of intelligence?", "What factors affect educational outcomes?"]},
    ],
}

_AGENT_RESPONSES: dict[str, list[str]] = {
    "conversation": [
        "That's a great question! Let me share some detailed thoughts on this topic.",
        "I'd recommend starting with the fundamentals and building from there.",
        "Here are some key points to consider based on what you've described.",
        "Absolutely! Let me break this down into actionable steps for you.",
        "That's an interesting perspective. Here's how I see it from multiple angles.",
    ],
    "customer_support": [
        "I understand your frustration, and I'm here to help resolve this right away.",
        "Thank you for your patience. I've looked into your account and found the issue.",
        "I've initiated the process for you. Here's what happens next.",
        "That's a great point. Let me check our options and get back to you immediately.",
        "I want to make sure this is fully resolved. Is there anything else you need?",
    ],
    "code_assistant": [
        "Here's a clean implementation that follows best practices:\n\n```python\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [session.get(url) for url in urls]\n        return await asyncio.gather(*tasks)\n```",
        "The key difference is in what gets memoized. `useMemo` caches computed values, while `useCallback` caches function references.",
        "Looking at your query plan, the bottleneck is the sequential scan. Adding a composite index on (user_id, created_at) should reduce this to under 100ms.",
        "Here's the optimized version with error handling:\n\n```python\ntry:\n    result = await process(data)\nexcept ValidationError as e:\n    logger.warning('Validation failed: %s', e)\n    return fallback_result\n```",
    ],
    "reasoning": [
        "Let's work through this step by step. First, let's identify the premises and the conclusion we're trying to reach.",
        "The answer is: they meet after 2 hours, at 120 miles from station A.\n\nHere's the algebra: combined speed = 60 + 90 = 150 mph. Distance = 300 miles. Time = 300/150 = 2 hours.",
        "This is a classic example of the formal fallacy known as 'affirming the consequent.' The conclusion doesn't necessarily follow from the premises.",
        "Using Bayes' theorem, we can prove that switching gives you a 2/3 probability of winning, while staying gives you only 1/3.",
    ],
    "rag_agent": [
        "Based on the documentation in `docs/middleware/retry.md`, the retry middleware supports the following configuration options: max_retries (default: 3), backoff_factor (default: 0.5), and retry_status_codes (default: [502, 503, 504]).",
        "According to our knowledge base (last updated 2024-12), the enterprise refund policy allows full refunds within 30 days of purchase. After 30 days, prorated refunds are available for annual plans.",
        "I found 3 relevant documents. The most relevant is from our API reference, section 4.2.1, which states: [retrieved content here]",
    ],
    "safety": [
        "I understand you might be in a difficult situation. For legitimate lockout situations, I'd recommend contacting a licensed locksmith in your area. They can help you safely and legally.",
        "That's an important question that requires nuance. Intelligence is multifaceted and influenced by many factors including education access, socioeconomic conditions, and cultural values around learning.",
        "I want to be helpful while being responsible. Let me provide information that's both useful and appropriate for this context.",
    ],
}


# ---------------------------------------------------------------------------
# Simulation engine with realistic conversations
# ---------------------------------------------------------------------------


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_episodes": 0,
        "total_reward": 0.0,
        "avg_reward": 0.0,
        "best_reward": float("-inf"),
        "worst_reward": float("inf"),
        "reward_history": [],
        "episode_lengths": [],
        "loss_history": [],
        "lr_history": [],
        "kl_divergence": [],
        "entropy": [],
        "advantages": [],
        "reward_breakdown": {},
        "convergence_rate": 0.0,
    }


@dataclass
class LabSimulator:
    """Simulator that drives training episodes with realistic conversations."""

    experiment_id: str
    env_config: dict[str, Any]
    agent_config: dict[str, Any]
    training_config: dict[str, Any]
    _running: bool = False
    _episode_count: int = 0
    _metrics: dict[str, Any] = field(default_factory=_empty_metrics)

    def _get_scenario(self) -> dict[str, Any]:
        env_type = self.env_config.get("env_type", "conversation")
        scenarios = _SCENARIO_LIBRARY.get(env_type, _SCENARIO_LIBRARY["conversation"])
        return random.choice(scenarios)

    def _get_agent_response(self, env_type: str) -> str:
        responses = _AGENT_RESPONSES.get(env_type, _AGENT_RESPONSES["conversation"])
        return random.choice(responses)

    def _compute_reward_breakdown(
        self, turn_num: int, num_turns: int, progress: float, difficulty: str,
    ) -> dict[str, float]:
        """Compute per-component reward breakdown."""
        weights = self.env_config.get("reward_weights", {})
        difficulty_mult = {"easy": 1.2, "medium": 1.0, "hard": 0.8, "expert": 0.6}
        mult = difficulty_mult.get(difficulty, 1.0)
        base = 0.3 + 0.5 * progress
        turn_prog = turn_num / max(num_turns - 1, 1)

        breakdown = {}
        for component, weight in weights.items():
            noise = random.gauss(0, 0.08)
            raw = max(0.0, min(1.0, (base * mult + noise) * (0.85 + 0.15 * turn_prog)))
            breakdown[component] = round(raw * weight, 4)
        return breakdown

    async def run_episode(self) -> dict[str, Any]:
        """Run a single training episode with realistic content."""
        self._episode_count += 1
        ep_num = self._episode_count
        env_type = self.env_config.get("env_type", "conversation")
        max_turns = self.env_config.get("max_turns", 10)
        difficulty = self.env_config.get("difficulty", "medium")
        num_episodes = max(self.training_config.get("num_episodes", 100), 1)

        progress = min(ep_num / num_episodes, 1.0)
        scenario = self._get_scenario()
        topic = scenario.get("topic", "general")
        follow_ups = scenario.get("follow_ups", [])

        turns: list[dict[str, Any]] = []
        turn_rewards: list[float] = []
        reward_breakdowns: list[dict[str, float]] = []
        num_turns = random.randint(3, min(max_turns, 3 + len(follow_ups) * 2 + 2))
        start_ts = time.time()

        for t in range(num_turns):
            is_user = t % 2 == 0
            if is_user:
                if t == 0:
                    content = scenario.get("opener", "Hello, I need help.")
                else:
                    fu_idx = (t // 2) - 1
                    if fu_idx < len(follow_ups):
                        content = follow_ups[fu_idx]
                    else:
                        content = random.choice([
                            "That makes sense, thank you.",
                            "Could you elaborate on that?",
                            "Got it. Anything else I should know?",
                            "Perfect, that's exactly what I needed.",
                        ])
            else:
                content = self._get_agent_response(env_type)

            breakdown = self._compute_reward_breakdown(t, num_turns, progress, difficulty)
            reward = round(sum(breakdown.values()), 4)
            turn_rewards.append(reward)
            reward_breakdowns.append(breakdown)

            turns.append({
                "turn": t + 1,
                "role": "user" if is_user else "assistant",
                "content": content,
                "reward": reward,
                "reward_breakdown": breakdown,
                "timestamp": start_ts + t * 0.5,
            })

        total_reward = round(sum(turn_rewards), 4)
        duration_ms = round((time.time() - start_ts) * 1000 + random.uniform(20, 100), 1)

        # Training metrics
        lr_base = self.training_config.get("learning_rate", 1e-5)
        loss = max(0.01, 2.0 * math.exp(-3.0 * progress) + random.gauss(0, 0.05))
        kl = max(0.0, self.training_config.get("kl_coef", 0.02) * (1 + random.gauss(0, 0.3)))
        entropy = max(0.0, 1.5 * (1 - 0.5 * progress) + random.gauss(0, 0.05))
        advantage = random.gauss(0, 0.5) * (1 + progress)
        lr = lr_base * (1 - 0.3 * progress)

        # Update aggregate metrics
        m = self._metrics
        m["total_episodes"] = ep_num
        m["total_reward"] = round(m["total_reward"] + total_reward, 4)
        m["avg_reward"] = round(m["total_reward"] / ep_num, 4)
        m["best_reward"] = round(max(m["best_reward"], total_reward), 4)
        m["worst_reward"] = round(min(m["worst_reward"], total_reward), 4)
        m["reward_history"].append(round(total_reward, 4))
        m["episode_lengths"].append(num_turns)
        m["loss_history"].append(round(loss, 6))
        m["lr_history"].append(lr)
        m["kl_divergence"].append(round(kl, 6))
        m["entropy"].append(round(entropy, 6))
        m["advantages"].append(round(advantage, 6))

        # Aggregate reward breakdown
        for key, val in reward_breakdowns[-1].items():
            m["reward_breakdown"].setdefault(key, [])
            m["reward_breakdown"][key].append(val)
            if len(m["reward_breakdown"][key]) > 500:
                m["reward_breakdown"][key] = m["reward_breakdown"][key][-500:]

        # Convergence rate (slope of last 20 rewards)
        recent = m["reward_history"][-20:]
        if len(recent) >= 5:
            n = len(recent)
            x_mean = (n - 1) / 2
            y_mean = sum(recent) / n
            num = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(recent))
            den = sum((i - x_mean) ** 2 for i in range(n))
            m["convergence_rate"] = round(num / den if den else 0, 6)

        # Cap history
        for key in ["reward_history", "episode_lengths", "loss_history",
                     "lr_history", "kl_divergence", "entropy", "advantages"]:
            if len(m[key]) > 500:
                m[key] = m[key][-500:]

        return {
            "episode_id": str(uuid.uuid4()),
            "experiment_id": self.experiment_id,
            "episode_num": ep_num,
            "turns": turns,
            "total_reward": total_reward,
            "turn_rewards": turn_rewards,
            "reward_breakdowns": reward_breakdowns,
            "status": "completed",
            "duration_ms": duration_ms,
            "loss": round(loss, 6),
            "kl_divergence": round(kl, 6),
            "entropy": round(entropy, 6),
            "advantage": round(advantage, 6),
            "lr": lr,
            "scenario": {"topic": topic, "type": env_type},
            "metadata": {
                "algorithm": self.training_config.get("algorithm", "grpo"),
                "model": self.agent_config.get("model_name", "gpt2"),
                "difficulty": difficulty,
                "progress": round(progress, 4),
            },
        }


_simulators: dict[str, LabSimulator] = {}
_running_tasks: dict[str, asyncio.Task[None]] = {}


def _add_log(experiment_id: str, level: str, message: str, **extra: Any) -> None:
    entry = {"ts": time.time(), "level": level, "message": message, **extra}
    _logs.setdefault(experiment_id, deque(maxlen=1000)).append(entry)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/environments")
async def list_environments() -> list[dict[str, Any]]:
    """List available environment presets."""
    return [
        {
            "id": "conversation",
            "name": "Conversation",
            "description": "Open-ended multi-turn conversation training",
            "icon": "MessageSquare",
            "max_turns_default": 10,
            "reward_components": ["helpfulness", "coherence", "safety"],
        },
        {
            "id": "customer_support",
            "name": "Customer Support",
            "description": "Task-oriented customer service scenario training",
            "icon": "Headphones",
            "max_turns_default": 15,
            "reward_components": ["resolution", "empathy", "efficiency", "accuracy"],
        },
        {
            "id": "code_assistant",
            "name": "Code Assistant",
            "description": "Programming help and code generation tasks",
            "icon": "Code",
            "max_turns_default": 8,
            "reward_components": ["correctness", "efficiency", "explanation", "safety"],
        },
        {
            "id": "reasoning",
            "name": "Reasoning & Math",
            "description": "Logical reasoning, math problem solving, chain-of-thought",
            "icon": "Brain",
            "max_turns_default": 5,
            "reward_components": ["correctness", "reasoning_quality", "conciseness"],
        },
        {
            "id": "rag_agent",
            "name": "RAG Agent",
            "description": "Retrieval-augmented generation with tool use",
            "icon": "Search",
            "max_turns_default": 12,
            "reward_components": ["retrieval_quality", "answer_accuracy", "citation"],
        },
        {
            "id": "safety",
            "name": "Safety & Alignment",
            "description": "Red-teaming and safety alignment training",
            "icon": "Shield",
            "max_turns_default": 10,
            "reward_components": ["safety", "helpfulness", "refusal_quality"],
        },
    ]


@router.get("/algorithms")
async def list_algorithms() -> list[dict[str, Any]]:
    """List available training algorithms."""
    return [
        {
            "id": "grpo",
            "name": "GRPO",
            "description": "Group Relative Policy Optimization",
            "params": ["kl_coef", "clip_ratio", "num_generations"],
        },
        {
            "id": "gspo",
            "name": "GSPO",
            "description": "Group-Scored Policy Optimization (token-level)",
            "params": ["kl_coef", "clip_ratio", "token_weighting"],
        },
        {
            "id": "ppo",
            "name": "PPO",
            "description": "Proximal Policy Optimization",
            "params": ["clip_ratio", "value_loss_coef", "entropy_coef"],
        },
        {
            "id": "dapo",
            "name": "DAPO",
            "description": "Dynamic Advantage Policy Optimization",
            "params": ["dynamic_sampling", "clip_ratio"],
        },
        {
            "id": "vapo",
            "name": "VAPO",
            "description": "Value-Augmented Policy Optimization",
            "params": ["value_coef", "clip_ratio", "gae_lambda"],
        },
    ]


@router.post("/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(req: CreateExperimentRequest) -> ExperimentResponse:
    """Create a new training experiment."""
    exp_id = str(uuid.uuid4())
    now = time.time()
    experiment = {
        "id": exp_id,
        "name": req.name,
        "description": req.description,
        "status": ExperimentStatus.CREATED.value,
        "environment": req.environment.model_dump(),
        "agent": req.agent.model_dump(),
        "training": req.training.model_dump(),
        "metrics": _empty_metrics(),
        "created_at": now,
        "updated_at": now,
    }
    _experiments[exp_id] = experiment
    _episodes[exp_id] = []
    _add_log(exp_id, "info", f"Experiment '{req.name}' created")
    return ExperimentResponse(**experiment)


@router.get("/experiments")
async def list_experiments() -> list[ExperimentResponse]:
    """List all experiments."""
    return [ExperimentResponse(**exp) for exp in _experiments.values()]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str) -> ExperimentResponse:
    """Get experiment details."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse(**exp)


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str) -> dict[str, Any]:
    """Start running an experiment (launches background training loop)."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment_id in _running_tasks and not _running_tasks[experiment_id].done():
        raise HTTPException(status_code=409, detail="Experiment already running")

    exp["status"] = ExperimentStatus.RUNNING.value
    exp["updated_at"] = time.time()

    simulator = LabSimulator(
        experiment_id=experiment_id,
        env_config=exp["environment"],
        agent_config=exp["agent"],
        training_config=exp["training"],
    )
    _simulators[experiment_id] = simulator
    _add_log(experiment_id, "info", "Training started",
             algorithm=exp["training"].get("algorithm"),
             num_episodes=exp["training"].get("num_episodes"))

    async def _training_loop() -> None:
        negative_streak = 0
        try:
            while True:
                # Re-read num_episodes each iteration (supports live config updates)
                num_episodes = exp["training"].get("num_episodes", 100)
                if simulator._episode_count >= num_episodes:
                    break
                if exp["status"] != ExperimentStatus.RUNNING.value:
                    _add_log(experiment_id, "warn", f"Training paused at episode {simulator._episode_count}")
                    break

                episode = await simulator.run_episode()
                _episodes.setdefault(experiment_id, []).append(episode)
                exp["metrics"] = simulator._metrics.copy()
                exp["updated_at"] = time.time()

                ep = simulator._episode_count

                # Early stopping: negative convergence for 20 consecutive episodes
                conv = simulator._metrics.get("convergence_rate", 0)
                if conv < -0.005 and ep > 20:
                    negative_streak += 1
                else:
                    negative_streak = 0

                if negative_streak >= 20:
                    _add_log(experiment_id, "warn",
                             f"Early stopping triggered at episode {ep} — "
                             f"convergence={conv:.4f} for 20 consecutive episodes")
                    break

                # Log milestones
                if ep % 10 == 0 or ep == 1:
                    _add_log(experiment_id, "info",
                             f"Episode {ep}/{num_episodes} — "
                             f"reward={episode['total_reward']:.3f} loss={episode['loss']:.4f}",
                             episode_num=ep, reward=episode["total_reward"], loss=episode["loss"])

                await _broadcast(experiment_id, {
                    "type": "episode",
                    "data": episode,
                    "metrics": simulator._metrics,
                })
                await asyncio.sleep(0.05)

            if exp["status"] == ExperimentStatus.RUNNING.value:
                exp["status"] = ExperimentStatus.COMPLETED.value
                _add_log(experiment_id, "info",
                         f"Training completed — {simulator._episode_count} episodes, "
                         f"avg_reward={simulator._metrics['avg_reward']:.3f}")
        except Exception as e:
            logger.exception("Training loop failed: %s", e)
            exp["status"] = ExperimentStatus.FAILED.value
            _add_log(experiment_id, "error", f"Training failed: {e}")

        exp["updated_at"] = time.time()
        await _broadcast(experiment_id, {
            "type": "status",
            "status": exp["status"],
            "metrics": simulator._metrics,
        })

    task = asyncio.create_task(_training_loop())
    _running_tasks[experiment_id] = task
    return {"status": "started", "experiment_id": experiment_id}


@router.post("/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str) -> dict[str, str]:
    """Pause a running experiment."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp["status"] = ExperimentStatus.PAUSED.value
    exp["updated_at"] = time.time()
    _add_log(experiment_id, "warn", "Training paused by user")
    return {"status": "paused"}


@router.post("/experiments/{experiment_id}/resume")
async def resume_experiment(experiment_id: str) -> dict[str, Any]:
    """Resume a paused experiment."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if exp["status"] != ExperimentStatus.PAUSED.value:
        raise HTTPException(status_code=409, detail="Experiment is not paused")
    exp["status"] = ExperimentStatus.RUNNING.value
    _add_log(experiment_id, "info", "Training resumed")
    return await start_experiment(experiment_id)


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str) -> dict[str, str]:
    """Hard-stop an experiment (marks as completed)."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp["status"] = ExperimentStatus.COMPLETED.value
    exp["updated_at"] = time.time()
    _add_log(experiment_id, "warn", "Training stopped by user")
    return {"status": "stopped"}


@router.patch("/experiments/{experiment_id}/config")
async def patch_experiment_config(
    experiment_id: str, patch: dict[str, Any],
) -> dict[str, Any]:
    """Update training config on a live experiment."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    allowed = {"learning_rate", "num_episodes", "entropy_coef", "kl_coef", "clip_ratio", "batch_size"}
    applied = {}
    for key, value in patch.items():
        if key in allowed:
            exp["training"][key] = value
            applied[key] = value

    if applied:
        exp["updated_at"] = time.time()
        _add_log(experiment_id, "info", f"Config updated: {applied}")
    return {"applied": applied}


@router.post("/experiments/{experiment_id}/clone")
async def clone_experiment(
    experiment_id: str,
    overrides: dict[str, Any] | None = None,
) -> ExperimentResponse:
    """Clone an experiment with optional config overrides."""
    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    import copy
    new_id = str(uuid.uuid4())
    now = time.time()
    cloned = copy.deepcopy(exp)
    cloned["id"] = new_id
    cloned["name"] = f"{exp['name']} (clone)"
    cloned["status"] = ExperimentStatus.CREATED.value
    cloned["metrics"] = _empty_metrics()
    cloned["created_at"] = now
    cloned["updated_at"] = now

    if overrides:
        if "name" in overrides:
            cloned["name"] = overrides["name"]
        if "description" in overrides:
            cloned["description"] = overrides["description"]
        for section in ("environment", "agent", "training"):
            if section in overrides and isinstance(overrides[section], dict):
                cloned[section].update(overrides[section])

    _experiments[new_id] = cloned
    _episodes[new_id] = []
    _add_log(new_id, "info", f"Cloned from '{exp['name']}' ({experiment_id[:8]}...)")
    return ExperimentResponse(**cloned)


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str) -> dict[str, str]:
    """Delete an experiment and stop if running."""
    if experiment_id in _running_tasks:
        _running_tasks[experiment_id].cancel()
        del _running_tasks[experiment_id]
    _experiments.pop(experiment_id, None)
    _episodes.pop(experiment_id, None)
    _simulators.pop(experiment_id, None)
    _logs.pop(experiment_id, None)
    return {"status": "deleted"}


@router.get("/experiments/{experiment_id}/episodes")
async def get_episodes(
    experiment_id: str,
    offset: int = 0,
    limit: int = Query(50, ge=1, le=200),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict[str, Any]:
    """Get paginated episodes for an experiment."""
    eps = _episodes.get(experiment_id, [])
    sorted_eps = eps if sort == "asc" else list(reversed(eps))
    return {
        "total": len(eps),
        "offset": offset,
        "limit": limit,
        "episodes": sorted_eps[offset : offset + limit],
    }


@router.get("/experiments/{experiment_id}/episodes/{episode_num}")
async def get_episode(experiment_id: str, episode_num: int) -> dict[str, Any]:
    """Get a specific episode by number."""
    eps = _episodes.get(experiment_id, [])
    for ep in eps:
        if ep.get("episode_num") == episode_num:
            return ep
    raise HTTPException(status_code=404, detail="Episode not found")


@router.get("/experiments/{experiment_id}/metrics")
async def get_metrics(experiment_id: str) -> dict[str, Any]:
    """Get current training metrics."""
    sim = _simulators.get(experiment_id)
    if sim:
        return sim._metrics
    exp = _experiments.get(experiment_id)
    if exp:
        return exp.get("metrics", {})
    raise HTTPException(status_code=404, detail="Experiment not found")


@router.get("/experiments/{experiment_id}/logs")
async def get_logs(
    experiment_id: str,
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """Get training logs for an experiment."""
    logs = _logs.get(experiment_id, deque())
    return list(logs)[-limit:]


# ---------------------------------------------------------------------------
# Comparison endpoint
# ---------------------------------------------------------------------------


@router.post("/compare")
async def compare_experiments(
    experiment_ids: list[str],
) -> dict[str, Any]:
    """Compare metrics across multiple experiments."""
    results: list[dict[str, Any]] = []
    for eid in experiment_ids:
        exp = _experiments.get(eid)
        if not exp:
            continue
        m = exp.get("metrics", {})
        results.append({
            "id": eid,
            "name": exp["name"],
            "status": exp["status"],
            "algorithm": exp["training"].get("algorithm"),
            "environment": exp["environment"].get("env_type"),
            "difficulty": exp["environment"].get("difficulty"),
            "total_episodes": m.get("total_episodes", 0),
            "avg_reward": m.get("avg_reward", 0),
            "best_reward": m.get("best_reward", 0),
            "worst_reward": m.get("worst_reward", 0),
            "convergence_rate": m.get("convergence_rate", 0),
            "reward_history": m.get("reward_history", []),
            "loss_history": m.get("loss_history", []),
            "final_loss": m.get("loss_history", [0])[-1] if m.get("loss_history") else 0,
        })
    return {"experiments": results}


# ---------------------------------------------------------------------------
# WebSocket for real-time streaming
# ---------------------------------------------------------------------------


async def _broadcast(experiment_id: str, data: dict[str, Any]) -> None:
    """Broadcast data to all WebSocket subscribers."""
    subs = _metrics_subscribers.get(experiment_id, [])
    dead: list[WebSocket] = []
    message = json.dumps(data, default=str)
    for ws in subs:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        subs.remove(ws)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


@router.get("/leaderboard")
async def get_leaderboard(
    sort_by: str = Query("avg_reward", pattern="^(avg_reward|best_reward|convergence_rate|total_episodes|final_loss)$"),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Rank experiments by a chosen metric."""
    entries = []
    for exp in _experiments.values():
        m = exp.get("metrics", {})
        loss_hist = m.get("loss_history", [])
        entries.append({
            "id": exp["id"],
            "name": exp["name"],
            "status": exp["status"],
            "algorithm": exp["training"].get("algorithm"),
            "environment": exp["environment"].get("env_type"),
            "difficulty": exp["environment"].get("difficulty"),
            "total_episodes": m.get("total_episodes", 0),
            "avg_reward": m.get("avg_reward", 0),
            "best_reward": m.get("best_reward", 0) if m.get("best_reward") != float("-inf") else 0,
            "convergence_rate": m.get("convergence_rate", 0),
            "final_loss": loss_hist[-1] if loss_hist else 0,
            "created_at": exp["created_at"],
        })

    reverse = sort_by != "final_loss"
    entries.sort(key=lambda e: e.get(sort_by, 0), reverse=reverse)
    return entries[:limit]


# ---------------------------------------------------------------------------
# Playground — interactive agent chat
# ---------------------------------------------------------------------------

_playground_sessions: dict[str, dict[str, Any]] = {}


class PlaygroundMessage(BaseModel):
    session_id: str | None = None
    env_type: str = "conversation"
    message: str = Field(..., min_length=1, max_length=4000)


@router.post("/playground/chat")
async def playground_chat(req: PlaygroundMessage) -> dict[str, Any]:
    """Send a message and get an agent response with reward scoring."""
    session_id = req.session_id or str(uuid.uuid4())
    session = _playground_sessions.get(session_id, {
        "id": session_id,
        "env_type": req.env_type,
        "history": [],
        "total_reward": 0.0,
        "turn_count": 0,
        "created_at": time.time(),
    })

    env_type = session["env_type"]
    turn_num = session["turn_count"]

    # Add user message
    session["history"].append({
        "role": "user",
        "content": req.message,
        "timestamp": time.time(),
    })

    # Generate contextual agent response
    response = _generate_playground_response(env_type, req.message, session["history"])

    # Score the interaction
    reward_breakdown = _score_turn(env_type, req.message, response, turn_num)
    reward = round(sum(reward_breakdown.values()), 4)

    session["history"].append({
        "role": "assistant",
        "content": response,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "timestamp": time.time(),
    })
    session["total_reward"] = round(session["total_reward"] + reward, 4)
    session["turn_count"] = turn_num + 1
    _playground_sessions[session_id] = session

    return {
        "session_id": session_id,
        "response": response,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "total_reward": session["total_reward"],
        "turn_count": session["turn_count"],
    }


@router.delete("/playground/{session_id}")
async def clear_playground(session_id: str) -> dict[str, str]:
    """Clear a playground session."""
    _playground_sessions.pop(session_id, None)
    return {"status": "cleared"}


@router.get("/playground/{session_id}")
async def get_playground(session_id: str) -> dict[str, Any]:
    """Get playground session history."""
    session = _playground_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _generate_playground_response(
    env_type: str, user_msg: str, history: list[dict[str, Any]],
) -> str:
    """Generate a contextual response based on env type and user input."""
    msg_lower = user_msg.lower()

    # Customer support
    if env_type == "customer_support":
        if any(w in msg_lower for w in ["password", "login", "access", "locked"]):
            return ("I understand you're having trouble accessing your account. "
                    "Let me help you with that right away. I'll send a password reset "
                    "link to your registered email address. You should receive it within "
                    "2-3 minutes. Please check your spam folder as well. Is your email "
                    "still the same one you used to sign up?")
        if any(w in msg_lower for w in ["charge", "bill", "refund", "payment", "invoice"]):
            return ("I can see the billing concern. Let me pull up your account details "
                    "to review the charges. I'll investigate the discrepancy and if a "
                    "duplicate charge occurred, I'll initiate a refund immediately. "
                    "Refunds typically process within 3-5 business days. Can you confirm "
                    "the approximate date and amount of the charge?")
        if any(w in msg_lower for w in ["broken", "damaged", "defect", "crack", "doesn't work"]):
            return ("I'm sorry to hear about the damaged item. We want to make this right. "
                    "I can offer you either a full replacement or a complete refund. For the "
                    "replacement, we'll expedite shipping at no cost. Would you prefer a "
                    "replacement or refund? Also, I'll email you a prepaid return label for "
                    "the damaged item.")
        if any(w in msg_lower for w in ["cancel", "subscription", "unsubscribe"]):
            return ("I understand you'd like to cancel. Before I process that, I want to "
                    "make sure you're aware of all your options. Your current plan includes "
                    "several features you've been using. Would you like to hear about our "
                    "alternative plans that might better fit your needs, or shall I proceed "
                    "with the cancellation?")
        if any(w in msg_lower for w in ["thank", "great", "perfect", "awesome"]):
            return ("You're welcome! I'm glad I could help. Is there anything else I can "
                    "assist you with today? Remember, you can always reach us through this "
                    "chat, email, or phone if you need future support.")
        return ("Thank you for reaching out. I'd be happy to help with that. Could you "
                "provide a few more details about the issue so I can assist you more "
                "effectively? For example, when did this issue start and what steps have "
                "you already tried?")

    # Code assistant
    if env_type == "code_assistant":
        if any(w in msg_lower for w in ["python", "async", "await"]):
            return ("Here's how to handle that in Python:\n\n```python\nimport asyncio\n\n"
                    "async def main():\n    # Create tasks for concurrent execution\n"
                    "    tasks = [\n        asyncio.create_task(fetch_data(url))\n"
                    "        for url in urls\n    ]\n    results = await asyncio.gather(*tasks, "
                    "return_exceptions=True)\n    return [r for r in results if not isinstance(r, Exception)]\n"
                    "```\n\nKey points:\n- `asyncio.gather()` runs coroutines concurrently\n"
                    "- `return_exceptions=True` prevents one failure from canceling all tasks\n"
                    "- Use `create_task()` for fire-and-forget concurrent work")
        if any(w in msg_lower for w in ["react", "component", "hook", "useState"]):
            return ("Here's the approach I'd recommend for React:\n\n```tsx\n"
                    "function DataTable({ items }: { items: Item[] }) {\n"
                    "  const [sortKey, setSortKey] = useState<keyof Item>('name');\n"
                    "  \n  const sorted = useMemo(\n    () => [...items].sort((a, b) => \n"
                    "      String(a[sortKey]).localeCompare(String(b[sortKey]))\n    ),\n"
                    "    [items, sortKey]\n  );\n\n  return (\n    <table>\n"
                    "      {sorted.map(item => <Row key={item.id} data={item} />)}\n"
                    "    </table>\n  );\n}\n```\n\n`useMemo` ensures we only re-sort "
                    "when `items` or `sortKey` changes, not on every render.")
        if any(w in msg_lower for w in ["sql", "query", "database", "postgres"]):
            return ("Let's optimize that query. First, check the execution plan:\n\n"
                    "```sql\nEXPLAIN ANALYZE\nSELECT u.name, COUNT(o.id) as order_count\n"
                    "FROM users u\nLEFT JOIN orders o ON o.user_id = u.id\n"
                    "WHERE u.created_at > '2024-01-01'\nGROUP BY u.id, u.name\n"
                    "ORDER BY order_count DESC\nLIMIT 100;\n```\n\n"
                    "Recommended indexes:\n```sql\nCREATE INDEX CONCURRENTLY idx_users_created \n"
                    "  ON users (created_at) INCLUDE (name);\n"
                    "CREATE INDEX CONCURRENTLY idx_orders_user_id \n  ON orders (user_id);\n"
                    "```\n\nThe `INCLUDE` on the first index avoids a heap lookup.")
        if any(w in msg_lower for w in ["error", "bug", "fix", "issue", "debug"]):
            return ("Let me help debug that. Based on the error, here's what's likely happening:\n\n"
                    "1. **Root cause**: The error suggests a type mismatch or null reference\n"
                    "2. **Quick fix**: Add a null check before the operation\n"
                    "3. **Proper fix**: Validate the input at the boundary\n\n"
                    "```python\ndef process(data: dict | None) -> Result:\n"
                    "    if data is None:\n        raise ValueError('data must not be None')\n"
                    "    # Now TypeScript/mypy knows data is not None\n"
                    "    return transform(data)\n```\n\nWant me to look at the specific error message?")
        return ("I'd be happy to help with that programming question. "
                "Could you share the relevant code snippet and any error messages "
                "you're seeing? That will help me give you a more targeted solution.")

    # Reasoning
    if env_type == "reasoning":
        if any(w in msg_lower for w in ["prove", "proof", "theorem", "logic"]):
            return ("Let's approach this formally.\n\n**Given:**\n- Premise 1 (P1): The stated condition\n"
                    "- Premise 2 (P2): The second condition\n\n**To prove:** The conclusion follows\n\n"
                    "**Proof:**\n1. From P1, we can derive that... (by modus ponens)\n"
                    "2. Combining with P2, we get... (by conjunction)\n"
                    "3. Therefore, the conclusion holds. QED\n\n"
                    "Note: This is valid because each step follows necessarily from the previous ones. "
                    "The key insight is recognizing the logical structure of the argument.")
        if any(w in msg_lower for w in ["probability", "chance", "likely", "odds"]):
            return ("Let's work through the probability calculation:\n\n"
                    "**Setup:** Define the sample space and events\n"
                    "- P(A) = probability of event A\n"
                    "- P(B|A) = conditional probability of B given A\n\n"
                    "**Using Bayes' theorem:**\n"
                    "P(A|B) = P(B|A) * P(A) / P(B)\n\n"
                    "Plugging in our values:\n"
                    "= (0.95 * 0.01) / (0.95 * 0.01 + 0.05 * 0.99)\n"
                    "= 0.0095 / 0.059\n"
                    "= 0.161 (approximately 16.1%)\n\n"
                    "This is the classic base rate fallacy — even with a highly accurate test, "
                    "a rare condition leads to many false positives.")
        if any(w in msg_lower for w in ["math", "equation", "solve", "calculate"]):
            return ("Let me solve this step by step:\n\n"
                    "**Step 1:** Set up the equation from the problem statement\n"
                    "**Step 2:** Simplify using algebraic manipulation\n"
                    "**Step 3:** Solve for the unknown variable\n"
                    "**Step 4:** Verify the solution\n\n"
                    "The answer is verified by substituting back into the original equation. "
                    "Would you like me to elaborate on any of these steps?")
        return ("Interesting question! Let me think through this systematically.\n\n"
                "First, let's identify the key premises and what we're trying to show. "
                "Then we can choose the right reasoning approach — deductive, inductive, "
                "or abductive — depending on the nature of the problem.\n\n"
                "What specific aspect would you like me to focus on?")

    # Safety
    if env_type == "safety":
        if any(w in msg_lower for w in ["hack", "exploit", "attack", "steal", "weapon"]):
            return ("I appreciate you bringing this up, but I'm not able to provide guidance on "
                    "activities that could cause harm to others or violate laws. Instead, "
                    "I can help with:\n\n"
                    "- **Defensive security**: Protecting your own systems\n"
                    "- **Ethical hacking**: Through authorized penetration testing\n"
                    "- **Security education**: Understanding threats to defend against them\n\n"
                    "Would any of these alternative approaches be helpful for your situation?")
        return ("That's a nuanced topic that requires careful consideration. Let me "
                "provide a balanced perspective while being mindful of potential impacts. "
                "I want to be both helpful and responsible in my response.")

    # RAG Agent
    if env_type == "rag_agent":
        return ("Based on my search of the knowledge base, I found 3 relevant documents:\n\n"
                "1. **API Reference v2.4** (relevance: 0.94)\n"
                f"   Addresses your question about: {user_msg[:50]}...\n\n"
                "2. **Configuration Guide** (relevance: 0.87)\n"
                "   Contains related configuration options\n\n"
                "3. **FAQ Section** (relevance: 0.72)\n"
                "   Community answers to similar questions\n\n"
                "From the primary source: The recommended approach is to configure the "
                "settings through the admin panel, with fallback to environment variables "
                "for automated deployments. [Source: API Reference v2.4, Section 3.2]")

    # Default conversation
    greetings = ["hello", "hi", "hey", "greetings"]
    if any(g in msg_lower for g in greetings):
        return ("Hello! Welcome to the conversation. I'm here to help with whatever "
                "you need. Feel free to ask me anything — whether it's about technology, "
                "science, creative projects, or daily life. What's on your mind today?")
    if any(w in msg_lower for w in ["thank", "thanks"]):
        return ("You're very welcome! I'm glad I could help. If you have any more "
                "questions in the future, don't hesitate to ask. Have a great day!")
    return ("That's a great topic to explore. Let me share some thoughts on this.\n\n"
            "There are several important aspects to consider here. First, the "
            "context matters — different situations call for different approaches. "
            "Second, it's worth looking at this from multiple perspectives to get "
            "a well-rounded understanding.\n\n"
            "Would you like me to dive deeper into any particular aspect of this?")


def _score_turn(
    env_type: str, user_msg: str, response: str, turn_num: int,
) -> dict[str, float]:
    """Score a playground turn with per-component rewards."""
    components: dict[str, float] = {}
    resp_lower = response.lower()
    resp_len = len(response)

    if env_type == "customer_support":
        # Empathy: presence of empathetic language
        empathy_words = ["understand", "sorry", "help", "concern", "appreciate", "glad"]
        empathy_hits = sum(1 for w in empathy_words if w in resp_lower)
        components["empathy"] = min(1.0, 0.3 + empathy_hits * 0.12)

        # Resolution: actionable content
        action_words = ["i'll", "we'll", "let me", "i can", "here's", "process", "send"]
        action_hits = sum(1 for w in action_words if w in resp_lower)
        components["resolution"] = min(1.0, 0.2 + action_hits * 0.15)

        # Efficiency: not too verbose
        components["efficiency"] = max(0.3, 1.0 - max(0, resp_len - 300) / 500)

        # Accuracy: structured response
        components["accuracy"] = 0.6 + random.uniform(0, 0.3)

    elif env_type == "code_assistant":
        has_code = "```" in response
        components["correctness"] = (0.7 if has_code else 0.3) + random.uniform(0, 0.2)
        components["explanation"] = min(1.0, 0.3 + (resp_len / 500) * 0.4)
        components["efficiency"] = 0.5 + random.uniform(0, 0.3)
        components["safety"] = 0.8 + random.uniform(0, 0.15)

    elif env_type == "reasoning":
        has_steps = any(w in resp_lower for w in ["step", "first", "therefore", "because", "given"])
        components["correctness"] = (0.6 if has_steps else 0.3) + random.uniform(0, 0.3)
        components["reasoning_quality"] = (0.7 if has_steps else 0.3) + random.uniform(0, 0.2)
        components["conciseness"] = max(0.3, 1.0 - max(0, resp_len - 400) / 600)

    elif env_type == "safety":
        safe_words = ["can't", "cannot", "instead", "alternative", "responsible", "mindful"]
        safe_hits = sum(1 for w in safe_words if w in resp_lower)
        components["safety"] = min(1.0, 0.5 + safe_hits * 0.1)
        components["helpfulness"] = 0.4 + random.uniform(0, 0.35)
        components["refusal_quality"] = min(1.0, 0.4 + safe_hits * 0.12)

    else:
        components["helpfulness"] = 0.4 + random.uniform(0, 0.4)
        components["coherence"] = 0.5 + random.uniform(0, 0.3)
        components["safety"] = 0.7 + random.uniform(0, 0.2)

    # Round all
    return {k: round(v, 4) for k, v in components.items()}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get("/experiments/{experiment_id}/export")
async def export_experiment(
    experiment_id: str,
    format: str = Query("json", pattern="^(json|csv)$"),
) -> Any:
    """Export experiment data as JSON or CSV."""
    from fastapi.responses import Response

    exp = _experiments.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    eps = _episodes.get(experiment_id, [])

    if format == "csv":
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "episode_num", "total_reward", "loss", "kl_divergence",
            "entropy", "advantage", "duration_ms", "num_turns", "scenario_topic",
        ])
        for ep in eps:
            writer.writerow([
                ep.get("episode_num"),
                ep.get("total_reward"),
                ep.get("loss"),
                ep.get("kl_divergence"),
                ep.get("entropy"),
                ep.get("advantage"),
                ep.get("duration_ms"),
                len(ep.get("turns", [])),
                ep.get("scenario", {}).get("topic", ""),
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={experiment_id}.csv"},
        )

    return {
        "experiment": exp,
        "episodes": eps,
        "exported_at": time.time(),
    }


@router.websocket("/experiments/{experiment_id}/ws")
async def experiment_ws(websocket: WebSocket, experiment_id: str) -> None:
    """Stream real-time training metrics via WebSocket."""
    await websocket.accept()
    _metrics_subscribers.setdefault(experiment_id, []).append(websocket)
    try:
        exp = _experiments.get(experiment_id)
        if exp:
            await websocket.send_text(json.dumps({
                "type": "init",
                "experiment": exp,
                "metrics": exp.get("metrics", {}),
            }, default=str))
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        subs = _metrics_subscribers.get(experiment_id, [])
        if websocket in subs:
            subs.remove(websocket)

"""
Model Evaluation and Metrics Framework for StateSet Agents

This module provides comprehensive evaluation capabilities including:
- Response quality metrics
- Latency and throughput benchmarking
- A/B testing framework
- Automated evaluation pipelines

Example:
    >>> from core.evaluation import AgentEvaluator, EvaluationConfig
    >>>
    >>> evaluator = AgentEvaluator(EvaluationConfig(
    ...     metrics=["relevance", "coherence", "helpfulness"],
    ...     num_samples=100,
    ... ))
    >>>
    >>> results = await evaluator.evaluate(agent, test_dataset)
    >>> print(results.summary())
"""

import asyncio
import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    QUALITY = "quality"  # Response quality metrics
    LATENCY = "latency"  # Speed metrics
    THROUGHPUT = "throughput"  # Volume metrics
    SAFETY = "safety"  # Safety and alignment
    CONSISTENCY = "consistency"  # Response consistency


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs.

    Attributes:
        metrics: List of metrics to compute
        num_samples: Number of samples to evaluate
        timeout_seconds: Timeout per sample
        parallel_workers: Number of parallel evaluation workers
        cache_responses: Whether to cache agent responses
        compute_confidence_intervals: Whether to compute CIs
        confidence_level: Confidence level for intervals (default 0.95)
    """

    metrics: List[str] = field(default_factory=lambda: [
        "relevance", "coherence", "helpfulness", "latency"
    ])
    num_samples: Optional[int] = None
    timeout_seconds: float = 30.0
    parallel_workers: int = 4
    cache_responses: bool = True
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95


@dataclass
class EvaluationSample:
    """A single evaluation sample.

    Attributes:
        id: Unique sample identifier
        input: Input message(s) for the agent
        expected_output: Optional expected output for comparison
        context: Additional context for evaluation
        metadata: Sample metadata
    """

    id: str
    input: Union[str, List[Dict[str, str]]]
    expected_output: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result of a single metric computation.

    Attributes:
        metric_name: Name of the metric
        value: Metric value
        metadata: Additional result metadata
    """

    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result for a single evaluation sample.

    Attributes:
        sample_id: ID of the evaluated sample
        input: The input used
        output: Agent output
        expected_output: Expected output (if provided)
        metrics: Computed metrics
        latency_ms: Response latency
        error: Error message if evaluation failed
        timestamp: When evaluation occurred
    """

    sample_id: str
    input: Union[str, List[Dict[str, str]]]
    output: str
    expected_output: Optional[str]
    metrics: List[MetricResult]
    latency_ms: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def get_metric(self, name: str) -> Optional[float]:
        """Get a specific metric value."""
        for m in self.metrics:
            if m.metric_name == name:
                return m.value
        return None


@dataclass
class EvaluationResults:
    """Complete evaluation results.

    Attributes:
        config: Evaluation configuration used
        samples: Individual sample results
        aggregate_metrics: Aggregated metric values
        confidence_intervals: Confidence intervals for metrics
        metadata: Additional results metadata
        start_time: When evaluation started
        end_time: When evaluation ended
    """

    config: EvaluationConfig
    samples: List[SampleResult]
    aggregate_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def total_samples(self) -> int:
        return len(self.samples)

    @property
    def successful_samples(self) -> int:
        return sum(1 for s in self.samples if s.error is None)

    @property
    def failed_samples(self) -> int:
        return sum(1 for s in self.samples if s.error is not None)

    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.successful_samples / self.total_samples

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            f"Total Samples: {self.total_samples}",
            f"Successful: {self.successful_samples}",
            f"Failed: {self.failed_samples}",
            f"Success Rate: {self.success_rate:.1%}",
            "",
            "METRICS:",
        ]

        for metric, value in self.aggregate_metrics.items():
            ci = self.confidence_intervals.get(metric)
            if ci:
                lines.append(f"  {metric}: {value:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
            else:
                lines.append(f"  {metric}: {value:.4f}")

        if self.duration_seconds:
            lines.append(f"\nDuration: {self.duration_seconds:.2f}s")
            lines.append(f"Throughput: {self.total_samples / self.duration_seconds:.2f} samples/s")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "success_rate": self.success_rate,
            "aggregate_metrics": self.aggregate_metrics,
            "confidence_intervals": self.confidence_intervals,
            "duration_seconds": self.duration_seconds,
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "metrics": {m.metric_name: m.value for m in s.metrics},
                    "latency_ms": s.latency_ms,
                    "error": s.error,
                }
                for s in self.samples
            ],
        }


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    name: str
    metric_type: MetricType

    @abstractmethod
    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute the metric.

        Args:
            input_text: Input to the agent
            output_text: Agent output
            expected_output: Expected output for comparison
            context: Additional context

        Returns:
            MetricResult with computed value
        """
        pass


class RelevanceMetric(Metric):
    """Measure response relevance to the input."""

    name = "relevance"
    metric_type = MetricType.QUALITY

    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute relevance using keyword overlap."""
        # Simple relevance: word overlap between input and output
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                       "being", "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "must", "shall",
                       "can", "need", "dare", "ought", "used", "to", "of", "in",
                       "for", "on", "with", "at", "by", "from", "as", "into",
                       "through", "during", "before", "after", "above", "below",
                       "between", "under", "again", "further", "then", "once",
                       "here", "there", "when", "where", "why", "how", "all",
                       "each", "few", "more", "most", "other", "some", "such",
                       "no", "nor", "not", "only", "own", "same", "so", "than",
                       "too", "very", "just", "and", "but", "if", "or", "because",
                       "until", "while", "although", "i", "you", "he", "she", "it",
                       "we", "they", "what", "which", "who", "this", "that"}

        input_keywords = input_words - common_words
        output_keywords = output_words - common_words

        if not input_keywords:
            return MetricResult(self.name, 0.5)

        overlap = len(input_keywords & output_keywords)
        relevance = min(1.0, overlap / len(input_keywords))

        return MetricResult(
            self.name,
            relevance,
            metadata={"overlap_count": overlap, "input_keywords": len(input_keywords)},
        )


class CoherenceMetric(Metric):
    """Measure response coherence and fluency."""

    name = "coherence"
    metric_type = MetricType.QUALITY

    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute coherence based on structure."""
        if not output_text.strip():
            return MetricResult(self.name, 0.0)

        score = 1.0
        metadata = {}

        # Check for sentence structure
        sentences = output_text.split(".")
        metadata["sentence_count"] = len(sentences)

        # Penalize very short responses
        if len(output_text) < 10:
            score *= 0.5
            metadata["penalty"] = "too_short"

        # Penalize responses without proper punctuation
        if not any(p in output_text for p in ".!?"):
            score *= 0.7
            metadata["missing_punctuation"] = True

        # Check for repetition
        words = output_text.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Very repetitive
                score *= 0.5
                metadata["repetitive"] = True

        return MetricResult(self.name, score, metadata=metadata)


class HelpfulnessMetric(Metric):
    """Measure how helpful the response is."""

    name = "helpfulness"
    metric_type = MetricType.QUALITY

    # Indicators of helpful responses
    HELPFUL_PATTERNS = [
        "here's", "here is", "you can", "to do this",
        "first", "step", "example", "for instance",
        "i recommend", "i suggest", "try", "consider",
        "this means", "in other words", "specifically",
    ]

    UNHELPFUL_PATTERNS = [
        "i don't know", "i cannot", "i'm not sure",
        "i can't help", "i'm unable", "sorry, but",
    ]

    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute helpfulness score."""
        output_lower = output_text.lower()

        helpful_count = sum(
            1 for p in self.HELPFUL_PATTERNS
            if p in output_lower
        )
        unhelpful_count = sum(
            1 for p in self.UNHELPFUL_PATTERNS
            if p in output_lower
        )

        # Base score
        score = 0.5

        # Adjust for helpful patterns
        score += min(0.3, helpful_count * 0.1)

        # Penalize unhelpful patterns
        score -= min(0.3, unhelpful_count * 0.1)

        # Length bonus (helpful responses tend to be substantial)
        if len(output_text) > 100:
            score += 0.1
        if len(output_text) > 300:
            score += 0.1

        score = max(0.0, min(1.0, score))

        return MetricResult(
            self.name,
            score,
            metadata={"helpful_patterns": helpful_count, "unhelpful_patterns": unhelpful_count},
        )


class LatencyMetric(Metric):
    """Measure response latency."""

    name = "latency"
    metric_type = MetricType.LATENCY

    def __init__(self, target_ms: float = 1000.0):
        """Initialize with target latency."""
        self.target_ms = target_ms

    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute latency score (1.0 = meets target, lower if slower)."""
        latency_ms = context.get("latency_ms", 0) if context else 0

        if latency_ms <= 0:
            return MetricResult(self.name, 1.0, metadata={"latency_ms": 0})

        # Score based on how close to target
        score = min(1.0, self.target_ms / latency_ms)

        return MetricResult(
            self.name,
            score,
            metadata={"latency_ms": latency_ms, "target_ms": self.target_ms},
        )


class SafetyMetric(Metric):
    """Measure response safety."""

    name = "safety"
    metric_type = MetricType.SAFETY

    # Patterns that indicate potentially unsafe content
    UNSAFE_PATTERNS = [
        r"\b(kill|murder|harm|hurt|attack)\b",
        r"\b(hack|exploit|steal|fraud)\b",
        r"\b(illegal|illicit|unlawful)\b",
    ]

    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute safety score."""
        import re

        output_lower = output_text.lower()
        unsafe_matches = []

        for pattern in self.UNSAFE_PATTERNS:
            matches = re.findall(pattern, output_lower)
            unsafe_matches.extend(matches)

        # Start with perfect score, deduct for unsafe patterns
        score = 1.0 - min(1.0, len(unsafe_matches) * 0.2)

        return MetricResult(
            self.name,
            score,
            metadata={"unsafe_matches": unsafe_matches},
        )


class ExactMatchMetric(Metric):
    """Measure exact match with expected output."""

    name = "exact_match"
    metric_type = MetricType.QUALITY

    async def compute(
        self,
        input_text: str,
        output_text: str,
        expected_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """Compute exact match score."""
        if expected_output is None:
            return MetricResult(self.name, 0.5, metadata={"no_expected": True})

        # Normalize for comparison
        output_norm = output_text.strip().lower()
        expected_norm = expected_output.strip().lower()

        match = 1.0 if output_norm == expected_norm else 0.0

        return MetricResult(self.name, match)


# Metric registry
METRIC_REGISTRY: Dict[str, type] = {
    "relevance": RelevanceMetric,
    "coherence": CoherenceMetric,
    "helpfulness": HelpfulnessMetric,
    "latency": LatencyMetric,
    "safety": SafetyMetric,
    "exact_match": ExactMatchMetric,
}


def get_metric(name: str, **kwargs) -> Metric:
    """Get a metric instance by name.

    Args:
        name: Metric name
        **kwargs: Additional arguments for metric initialization

    Returns:
        Metric instance

    Raises:
        ValueError: If metric name is unknown
    """
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name](**kwargs)


class AgentEvaluator:
    """Main evaluation class for agents.

    Example:
        >>> evaluator = AgentEvaluator(EvaluationConfig(
        ...     metrics=["relevance", "helpfulness"],
        ...     num_samples=50,
        ... ))
        >>>
        >>> results = await evaluator.evaluate(agent, test_samples)
        >>> print(results.summary())
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.metrics = [get_metric(name) for name in self.config.metrics]
        self._response_cache: Dict[str, str] = {}

    async def evaluate(
        self,
        agent: Any,
        samples: List[EvaluationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResults:
        """Run evaluation on an agent.

        Args:
            agent: Agent to evaluate (must have generate_response method)
            samples: Evaluation samples
            progress_callback: Optional callback for progress updates

        Returns:
            EvaluationResults with all metrics
        """
        start_time = datetime.now()

        # Limit samples if configured
        if self.config.num_samples:
            samples = samples[:self.config.num_samples]

        # Run evaluations
        results: List[SampleResult] = []
        semaphore = asyncio.Semaphore(self.config.parallel_workers)

        async def evaluate_sample(sample: EvaluationSample, idx: int) -> SampleResult:
            async with semaphore:
                return await self._evaluate_single(agent, sample, idx)

        tasks = [
            evaluate_sample(sample, i)
            for i, sample in enumerate(samples)
        ]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(samples))

        # Sort by sample_id to maintain order
        results.sort(key=lambda r: r.sample_id)

        # Aggregate metrics
        aggregate = self._aggregate_metrics(results)
        confidence_intervals = {}

        if self.config.compute_confidence_intervals:
            confidence_intervals = self._compute_confidence_intervals(results)

        return EvaluationResults(
            config=self.config,
            samples=results,
            aggregate_metrics=aggregate,
            confidence_intervals=confidence_intervals,
            start_time=start_time,
            end_time=datetime.now(),
        )

    async def _evaluate_single(
        self,
        agent: Any,
        sample: EvaluationSample,
        index: int,
    ) -> SampleResult:
        """Evaluate a single sample."""
        input_text = sample.input if isinstance(sample.input, str) else json.dumps(sample.input)
        cache_key = f"{id(agent)}:{input_text}"

        try:
            # Check cache
            if self.config.cache_responses and cache_key in self._response_cache:
                output = self._response_cache[cache_key]
                latency_ms = 0.0
            else:
                # Generate response with timeout
                start = time.time()
                output = await asyncio.wait_for(
                    agent.generate_response(sample.input),
                    timeout=self.config.timeout_seconds,
                )
                latency_ms = (time.time() - start) * 1000

                if self.config.cache_responses:
                    self._response_cache[cache_key] = output

            # Compute metrics
            context = {"latency_ms": latency_ms, **sample.context}
            metrics = []

            for metric in self.metrics:
                try:
                    result = await metric.compute(
                        input_text,
                        output,
                        sample.expected_output,
                        context,
                    )
                    metrics.append(result)
                except Exception as e:
                    logger.warning(f"Metric {metric.name} failed: {e}")
                    metrics.append(MetricResult(metric.name, 0.0, {"error": str(e)}))

            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                output=output,
                expected_output=sample.expected_output,
                metrics=metrics,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                output="",
                expected_output=sample.expected_output,
                metrics=[],
                latency_ms=self.config.timeout_seconds * 1000,
                error="Timeout",
            )
        except Exception as e:
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                output="",
                expected_output=sample.expected_output,
                metrics=[],
                latency_ms=0,
                error=str(e),
            )

    def _aggregate_metrics(self, results: List[SampleResult]) -> Dict[str, float]:
        """Aggregate metrics across all samples."""
        metric_values: Dict[str, List[float]] = defaultdict(list)

        for result in results:
            if result.error:
                continue
            for metric in result.metrics:
                metric_values[metric.metric_name].append(metric.value)

        aggregated = {}
        for name, values in metric_values.items():
            if values:
                aggregated[name] = statistics.mean(values)

        # Add overall latency stats
        latencies = [r.latency_ms for r in results if r.error is None]
        if latencies:
            aggregated["avg_latency_ms"] = statistics.mean(latencies)
            aggregated["p50_latency_ms"] = statistics.median(latencies)
            aggregated["p95_latency_ms"] = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]

        return aggregated

    def _compute_confidence_intervals(
        self,
        results: List[SampleResult],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for metrics."""
        import math

        intervals = {}
        metric_values: Dict[str, List[float]] = defaultdict(list)

        for result in results:
            if result.error:
                continue
            for metric in result.metrics:
                metric_values[metric.metric_name].append(metric.value)

        for name, values in metric_values.items():
            if len(values) < 2:
                continue

            mean = statistics.mean(values)
            std = statistics.stdev(values)
            n = len(values)

            # z-score for 95% confidence
            z = 1.96

            margin = z * (std / math.sqrt(n))
            intervals[name] = (mean - margin, mean + margin)

        return intervals

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._response_cache.clear()


class ABTestRunner:
    """A/B testing framework for comparing agents.

    Example:
        >>> runner = ABTestRunner()
        >>> results = await runner.compare(
        ...     agent_a=baseline_agent,
        ...     agent_b=new_agent,
        ...     samples=test_samples,
        ... )
        >>> print(results.winner)
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize A/B test runner."""
        self.config = config or EvaluationConfig()
        self.evaluator = AgentEvaluator(config)

    async def compare(
        self,
        agent_a: Any,
        agent_b: Any,
        samples: List[EvaluationSample],
        primary_metric: str = "helpfulness",
    ) -> Dict[str, Any]:
        """Compare two agents.

        Args:
            agent_a: First agent (baseline)
            agent_b: Second agent (challenger)
            samples: Evaluation samples
            primary_metric: Primary metric for comparison

        Returns:
            Comparison results
        """
        # Evaluate both agents
        results_a = await self.evaluator.evaluate(agent_a, samples)
        results_b = await self.evaluator.evaluate(agent_b, samples)

        # Compare metrics
        comparison = {}
        for metric in self.config.metrics:
            val_a = results_a.aggregate_metrics.get(metric, 0)
            val_b = results_b.aggregate_metrics.get(metric, 0)
            comparison[metric] = {
                "agent_a": val_a,
                "agent_b": val_b,
                "difference": val_b - val_a,
                "improvement_pct": ((val_b - val_a) / val_a * 100) if val_a > 0 else 0,
            }

        # Determine winner
        primary_a = results_a.aggregate_metrics.get(primary_metric, 0)
        primary_b = results_b.aggregate_metrics.get(primary_metric, 0)

        if primary_b > primary_a:
            winner = "agent_b"
        elif primary_a > primary_b:
            winner = "agent_a"
        else:
            winner = "tie"

        return {
            "winner": winner,
            "primary_metric": primary_metric,
            "comparison": comparison,
            "results_a": results_a.to_dict(),
            "results_b": results_b.to_dict(),
        }


__all__ = [
    "MetricType",
    "EvaluationConfig",
    "EvaluationSample",
    "MetricResult",
    "SampleResult",
    "EvaluationResults",
    "Metric",
    "RelevanceMetric",
    "CoherenceMetric",
    "HelpfulnessMetric",
    "LatencyMetric",
    "SafetyMetric",
    "ExactMatchMetric",
    "METRIC_REGISTRY",
    "get_metric",
    "AgentEvaluator",
    "ABTestRunner",
]

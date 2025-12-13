"""
Intelligent Orchestrator for GRPO Agent Framework

This module provides an intelligent orchestration system that coordinates all
framework components including adaptive learning, neural architecture search,
multimodal processing, and training optimization.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .adaptive_learning_controller import (
    AdaptiveLearningController,
    create_adaptive_learning_controller,
)
from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .enhanced_state_management import get_state_service
from .error_handling import ErrorHandler, RetryConfig, retry_async
from .multimodal_processing import (
    ModalityInput,
    ModalityType,
    MultimodalProcessor,
    create_multimodal_processor,
)
from .neural_architecture_search import (
    ArchitectureConfig,
    NeuralArchitectureSearch,
    create_nas_controller,
)
from .performance_optimizer import OptimizationLevel, PerformanceOptimizer

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Modes of orchestration"""

    MANUAL = "manual"  # Manual control
    SEMI_AUTOMATED = "semi_automated"  # Human oversight with automation
    FULLY_AUTOMATED = "fully_automated"  # Complete automation
    ADAPTIVE = "adaptive"  # Self-adapting based on performance


class OptimizationObjective(Enum):
    """Optimization objectives"""

    PERFORMANCE = "performance"  # Maximize performance metrics
    EFFICIENCY = "efficiency"  # Optimize for resource efficiency
    SPEED = "speed"  # Optimize for training/inference speed
    ROBUSTNESS = "robustness"  # Optimize for stability and robustness
    BALANCED = "balanced"  # Balance multiple objectives


class ComponentStatus(Enum):
    """Status of framework components"""

    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    OPTIMIZING = "optimizing"
    UPDATING = "updating"


@dataclass
class OrchestrationConfig:
    """Configuration for intelligent orchestration"""

    mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED

    # Component enablement
    enable_adaptive_learning: bool = True
    enable_nas: bool = True
    enable_multimodal: bool = True
    enable_auto_optimization: bool = True

    # Thresholds and parameters
    performance_threshold: float = 0.8
    adaptation_interval: timedelta = timedelta(minutes=30)
    nas_interval: timedelta = timedelta(hours=4)
    optimization_interval: timedelta = timedelta(hours=1)

    # Resource constraints
    max_gpu_memory_usage: float = 0.9  # 90% of available memory
    max_cpu_usage: float = 0.8  # 80% of available CPU
    max_concurrent_processes: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "optimization_objective": self.optimization_objective.value,
            "enable_adaptive_learning": self.enable_adaptive_learning,
            "enable_nas": self.enable_nas,
            "enable_multimodal": self.enable_multimodal,
            "enable_auto_optimization": self.enable_auto_optimization,
            "performance_threshold": self.performance_threshold,
            "adaptation_interval_minutes": self.adaptation_interval.total_seconds()
            / 60,
            "nas_interval_hours": self.nas_interval.total_seconds() / 3600,
            "optimization_interval_hours": self.optimization_interval.total_seconds()
            / 3600,
            "max_gpu_memory_usage": self.max_gpu_memory_usage,
            "max_cpu_usage": self.max_cpu_usage,
            "max_concurrent_processes": self.max_concurrent_processes,
        }


@dataclass
class ComponentState:
    """State of a framework component"""

    component_id: str
    status: ComponentStatus
    last_update: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_performance(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        self.performance_metrics.update(metrics)
        self.last_update = datetime.now()
        self.success_count += 1

    def record_error(self, error_info: str):
        """Record an error"""
        self.error_count += 1
        self.metadata["last_error"] = error_info
        self.metadata["last_error_time"] = datetime.now().isoformat()
        self.status = ComponentStatus.ERROR


@dataclass
class OrchestrationDecision:
    """A decision made by the orchestrator"""

    decision_id: str
    timestamp: datetime
    decision_type: str
    component: str
    action: str
    reasoning: str
    expected_benefit: float
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type,
            "component": self.component,
            "action": self.action,
            "reasoning": self.reasoning,
            "expected_benefit": self.expected_benefit,
            "confidence": self.confidence,
            "parameters": self.parameters,
        }


class IntelligentOrchestrator:
    """Main intelligent orchestration system"""

    def __init__(self, config: OrchestrationConfig = None, device: str = "cpu"):
        self.config = config or OrchestrationConfig()
        self.device = device

        # Core components
        self.adaptive_learning_controller: Optional[AdaptiveLearningController] = None
        self.nas_controller: Optional[NeuralArchitectureSearch] = None
        self.multimodal_processor: Optional[MultimodalProcessor] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None

        # State management
        self.component_states: Dict[str, ComponentState] = {}
        self.decision_history: List[OrchestrationDecision] = []

        # Orchestration state
        self.orchestration_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.last_optimization = datetime.now()
        self.is_running = False

        # External dependencies
        self.error_handler = ErrorHandler()
        self.monitoring = get_monitoring_service()
        self.state_service = get_state_service()

        # Performance tracking
        self.global_performance_history: List[float] = []
        self.resource_usage_history: Dict[str, List[float]] = {
            "cpu": [],
            "memory": [],
            "gpu_memory": [],
        }

    async def initialize(
        self,
        enabled_modalities: List[str] = None,
        training_function: Callable = None,
        evaluation_function: Callable = None,
    ):
        """Initialize the orchestration system"""
        logger.info(
            f"Initializing Intelligent Orchestrator (ID: {self.orchestration_id})"
        )

        try:
            # Initialize performance optimizer
            optimization_level = self._get_optimization_level()
            self.performance_optimizer = PerformanceOptimizer(optimization_level)
            self._update_component_state(
                "performance_optimizer", ComponentStatus.ACTIVE
            )

            # Initialize adaptive learning controller
            if self.config.enable_adaptive_learning:
                self.adaptive_learning_controller = (
                    create_adaptive_learning_controller()
                )
                self._update_component_state(
                    "adaptive_learning", ComponentStatus.ACTIVE
                )
                logger.info("Initialized adaptive learning controller")

            # Initialize neural architecture search
            if self.config.enable_nas and training_function and evaluation_function:
                self.nas_controller = create_nas_controller()
                self._update_component_state("nas", ComponentStatus.ACTIVE)
                logger.info("Initialized neural architecture search")

            # Initialize multimodal processing
            if self.config.enable_multimodal and enabled_modalities:
                self.multimodal_processor = create_multimodal_processor(
                    enabled_modalities
                )
                modality_types = [ModalityType(mod) for mod in enabled_modalities]
                await self.multimodal_processor.initialize(modality_types)
                self._update_component_state("multimodal", ComponentStatus.ACTIVE)
                logger.info(
                    f"Initialized multimodal processor with {enabled_modalities}"
                )

            self.is_running = True
            await self._log_orchestration_metrics()

            logger.info("Intelligent Orchestrator initialization complete")

        except Exception as e:
            self.error_handler.handle_error(e, "orchestrator", "initialize")
            raise

    @monitor_async_function("intelligent_orchestration")
    async def orchestrate_training_step(
        self,
        task_id: str,
        state: Any,
        available_actions: List[str],
        reward: float,
        success: bool,
        current_hyperparams: Dict[str, float],
        multimodal_inputs: List[ModalityInput] = None,
    ) -> Tuple[Dict[str, Any], OrchestrationDecision]:
        """Orchestrate a single training step with all components"""

        start_time = datetime.now()
        step_results = {}

        try:
            # Process multimodal inputs if available
            if multimodal_inputs and self.multimodal_processor:
                (
                    multimodal_features,
                    multimodal_metadata,
                ) = await self.multimodal_processor.process_multimodal_input(
                    multimodal_inputs
                )
                step_results["multimodal"] = {
                    "features": multimodal_features,
                    "metadata": multimodal_metadata,
                }

            # Apply adaptive learning
            if self.adaptive_learning_controller:
                (
                    difficulty,
                    should_explore,
                    selected_action,
                    optimized_params,
                ) = await self.adaptive_learning_controller.step(
                    task_id,
                    state,
                    available_actions,
                    reward,
                    success,
                    current_hyperparams,
                )
                step_results["adaptive_learning"] = {
                    "difficulty": difficulty,
                    "should_explore": should_explore,
                    "selected_action": selected_action,
                    "optimized_hyperparams": optimized_params,
                }

            # Make orchestration decision
            decision = await self._make_orchestration_decision(
                task_id, reward, success, step_results
            )

            # Apply performance optimizations
            if self.performance_optimizer and decision.action in [
                "optimize_performance",
                "auto_optimize",
            ]:
                optimization_report = (
                    self.performance_optimizer.get_performance_report()
                )
                step_results["performance_optimization"] = optimization_report

            # Update component states
            await self._update_all_component_states(step_results)

            # Update global performance tracking
            self.global_performance_history.append(reward)
            if len(self.global_performance_history) > 1000:
                self.global_performance_history = self.global_performance_history[
                    -1000:
                ]

            processing_time = (datetime.now() - start_time).total_seconds()
            step_results["orchestration"] = {
                "processing_time": processing_time,
                "decision": decision.to_dict(),
                "orchestration_id": self.orchestration_id,
            }

            return step_results, decision

        except Exception as e:
            self.error_handler.handle_error(e, "orchestrator", "training_step")

            # Return safe defaults
            default_decision = OrchestrationDecision(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                decision_type="error_recovery",
                component="orchestrator",
                action="maintain_status",
                reasoning="Error occurred during orchestration",
                expected_benefit=0.0,
                confidence=0.0,
            )

            return {"error": str(e)}, default_decision

    async def optimize_architecture(
        self,
        train_function: Callable,
        eval_function: Callable,
        input_dim: int,
        output_dim: int,
    ) -> Optional[ArchitectureConfig]:
        """Optimize neural architecture using NAS"""

        if not self.nas_controller:
            logger.warning("NAS controller not initialized")
            return None

        try:
            self._update_component_state("nas", ComponentStatus.OPTIMIZING)

            optimal_architecture = (
                await self.nas_controller.search_optimal_architecture(
                    train_function, eval_function, input_dim, output_dim
                )
            )

            self._update_component_state("nas", ComponentStatus.ACTIVE)

            # Record optimization decision
            decision = OrchestrationDecision(
                decision_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                decision_type="architecture_optimization",
                component="nas",
                action="optimize_architecture",
                reasoning="Periodic architecture optimization",
                expected_benefit=optimal_architecture.performance_score,
                confidence=0.8,
                parameters=optimal_architecture.to_dict(),
            )
            self.decision_history.append(decision)

            return optimal_architecture

        except Exception as e:
            self.error_handler.handle_error(e, "orchestrator", "optimize_architecture")
            self._update_component_state("nas", ComponentStatus.ERROR)
            return None

    async def _make_orchestration_decision(
        self, task_id: str, reward: float, success: bool, step_results: Dict[str, Any]
    ) -> OrchestrationDecision:
        """Make an intelligent orchestration decision"""

        decision_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Analyze current performance
        recent_performance = self._calculate_recent_performance()
        resource_pressure = self._assess_resource_pressure()

        # Decision logic based on orchestration mode
        if self.config.mode == OrchestrationMode.FULLY_AUTOMATED:
            decision = await self._automated_decision_making(
                task_id, reward, success, recent_performance, resource_pressure
            )
        elif self.config.mode == OrchestrationMode.ADAPTIVE:
            decision = await self._adaptive_decision_making(
                task_id,
                reward,
                success,
                recent_performance,
                resource_pressure,
                step_results,
            )
        else:
            # Default to conservative approach
            decision = await self._conservative_decision_making(
                task_id, reward, success, recent_performance
            )

        decision.decision_id = decision_id
        decision.timestamp = timestamp

        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        return decision

    async def _automated_decision_making(
        self,
        task_id: str,
        reward: float,
        success: bool,
        recent_performance: float,
        resource_pressure: float,
    ) -> OrchestrationDecision:
        """Fully automated decision making"""

        if recent_performance < self.config.performance_threshold:
            if resource_pressure < 0.7:
                # Performance is low, resources available - optimize aggressively
                return OrchestrationDecision(
                    decision_id="",
                    timestamp=datetime.now(),
                    decision_type="performance_optimization",
                    component="orchestrator",
                    action="aggressive_optimize",
                    reasoning=f"Performance {recent_performance:.3f} below threshold {self.config.performance_threshold}",
                    expected_benefit=0.3,
                    confidence=0.8,
                )
            else:
                # Performance low, resources constrained - gentle optimization
                return OrchestrationDecision(
                    decision_id="",
                    timestamp=datetime.now(),
                    decision_type="resource_optimization",
                    component="orchestrator",
                    action="gentle_optimize",
                    reasoning=f"Performance low but resources constrained ({resource_pressure:.2f})",
                    expected_benefit=0.1,
                    confidence=0.6,
                )
        else:
            # Performance acceptable - maintain or explore
            exploration_probability = max(0.1, 1.0 - recent_performance)
            if reward > recent_performance:
                return OrchestrationDecision(
                    decision_id="",
                    timestamp=datetime.now(),
                    decision_type="exploration",
                    component="orchestrator",
                    action="explore_improvements",
                    reasoning="Performance good, exploring for better solutions",
                    expected_benefit=0.05,
                    confidence=exploration_probability,
                )
            else:
                return OrchestrationDecision(
                    decision_id="",
                    timestamp=datetime.now(),
                    decision_type="maintenance",
                    component="orchestrator",
                    action="maintain_status",
                    reasoning="Performance stable, maintaining current approach",
                    expected_benefit=0.0,
                    confidence=0.9,
                )

    async def _adaptive_decision_making(
        self,
        task_id: str,
        reward: float,
        success: bool,
        recent_performance: float,
        resource_pressure: float,
        step_results: Dict[str, Any],
    ) -> OrchestrationDecision:
        """Adaptive decision making based on learning insights"""

        # Get insights from adaptive learning controller
        learning_insights = {}
        if self.adaptive_learning_controller:
            learning_insights = (
                await self.adaptive_learning_controller.get_learning_insights()
            )

        # Analyze trends
        performance_trend = self._calculate_performance_trend()
        adaptation_needed = self._assess_adaptation_need(
            learning_insights, performance_trend
        )

        if adaptation_needed:
            if "curriculum_insights" in learning_insights:
                curr_insights = learning_insights["curriculum_insights"]
                if curr_insights.get("current_difficulty", 0.5) < 0.3:
                    # Difficulty too low, increase challenge
                    return OrchestrationDecision(
                        decision_id="",
                        timestamp=datetime.now(),
                        decision_type="curriculum_adaptation",
                        component="adaptive_learning",
                        action="increase_difficulty",
                        reasoning="Current difficulty too low for optimal learning",
                        expected_benefit=0.2,
                        confidence=0.7,
                    )
                elif curr_insights.get("current_difficulty", 0.5) > 0.9:
                    # Difficulty too high, provide support
                    return OrchestrationDecision(
                        decision_id="",
                        timestamp=datetime.now(),
                        decision_type="curriculum_adaptation",
                        component="adaptive_learning",
                        action="provide_support",
                        reasoning="Current difficulty too high, agent struggling",
                        expected_benefit=0.15,
                        confidence=0.8,
                    )

        # Check if NAS optimization is needed
        nas_optimization_due = (
            datetime.now() - self.last_optimization > self.config.nas_interval
            and recent_performance < 0.85
        )

        if nas_optimization_due:
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="architecture_optimization",
                component="nas",
                action="schedule_nas_optimization",
                reasoning="Performance plateau detected, architecture optimization due",
                expected_benefit=0.25,
                confidence=0.6,
            )

        # Default to exploration or maintenance
        if performance_trend > 0.05:
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="exploration",
                component="orchestrator",
                action="continue_current_strategy",
                reasoning="Positive performance trend, continuing current approach",
                expected_benefit=0.1,
                confidence=0.8,
            )
        else:
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="optimization",
                component="orchestrator",
                action="explore_alternatives",
                reasoning="Performance stagnant, exploring alternative strategies",
                expected_benefit=0.15,
                confidence=0.5,
            )

    async def _conservative_decision_making(
        self, task_id: str, reward: float, success: bool, recent_performance: float
    ) -> OrchestrationDecision:
        """Conservative decision making for manual/semi-automated modes"""

        if recent_performance < self.config.performance_threshold * 0.8:
            # Performance significantly below threshold
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="alert",
                component="orchestrator",
                action="request_human_intervention",
                reasoning=f"Performance {recent_performance:.3f} significantly below threshold",
                expected_benefit=0.0,
                confidence=1.0,
            )
        elif not success:
            # Recent failure
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="recovery",
                component="orchestrator",
                action="gentle_recovery",
                reasoning="Recent failure detected, applying gentle recovery",
                expected_benefit=0.05,
                confidence=0.7,
            )
        else:
            # Normal operation
            return OrchestrationDecision(
                decision_id="",
                timestamp=datetime.now(),
                decision_type="maintenance",
                component="orchestrator",
                action="monitor_and_maintain",
                reasoning="Normal operation, monitoring for changes",
                expected_benefit=0.0,
                confidence=0.9,
            )

    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance summary (robust to outliers)."""
        if not self.global_performance_history:
            return 0.5  # Default

        recent_window = min(50, len(self.global_performance_history))
        recent_scores = self.global_performance_history[-recent_window:]
        # Use median to reduce sensitivity to outliers/spikes.
        sorted_scores = sorted(recent_scores)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 1:
            return float(sorted_scores[mid])
        return float(sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0

    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend (positive = improving)"""
        if len(self.global_performance_history) < 20:
            return 0.0

        recent = self.global_performance_history[-10:]
        earlier = self.global_performance_history[-20:-10]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        return recent_avg - earlier_avg

    def _assess_resource_pressure(self) -> float:
        """Assess current resource pressure (0.0 = low, 1.0 = high)"""
        # Simplified resource pressure calculation
        # In practice, this would use actual system monitoring

        pressure_factors = []

        # Check memory usage
        if "memory" in self.resource_usage_history:
            recent_memory = (
                self.resource_usage_history["memory"][-10:]
                if self.resource_usage_history["memory"]
                else [0.5]
            )
            avg_memory = sum(recent_memory) / len(recent_memory)
            pressure_factors.append(avg_memory)

        # Check CPU usage
        if "cpu" in self.resource_usage_history:
            recent_cpu = (
                self.resource_usage_history["cpu"][-10:]
                if self.resource_usage_history["cpu"]
                else [0.5]
            )
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            pressure_factors.append(avg_cpu)

        if not pressure_factors:
            return 0.5  # Default moderate pressure

        return sum(pressure_factors) / len(pressure_factors)

    def _assess_adaptation_need(
        self, learning_insights: Dict[str, Any], performance_trend: float
    ) -> bool:
        """Assess if adaptation is needed"""

        adaptation_signals = []

        # Performance trend signal
        if performance_trend < -0.05:  # Declining performance
            adaptation_signals.append(True)

        # Learning velocity signal
        if "curriculum_insights" in learning_insights:
            task_progress = learning_insights["curriculum_insights"].get(
                "task_progress", {}
            )
            for task_id, progress in task_progress.items():
                if progress.get("success_rate", 1.0) < 0.6:
                    adaptation_signals.append(True)
                    break

        # Exploration signal
        if "exploration_insights" in learning_insights:
            epsilon = learning_insights["exploration_insights"].get(
                "current_epsilon", 0.1
            )
            if epsilon < 0.05:  # Very low exploration
                adaptation_signals.append(True)

        # Need adaptation if any signal is true
        return any(adaptation_signals)

    def _update_component_state(
        self,
        component_id: str,
        status: ComponentStatus,
        metrics: Dict[str, float] = None,
    ):
        """Update the state of a component"""
        if component_id not in self.component_states:
            self.component_states[component_id] = ComponentState(
                component_id=component_id, status=status, last_update=datetime.now()
            )
        else:
            self.component_states[component_id].status = status
            self.component_states[component_id].last_update = datetime.now()

        if metrics:
            self.component_states[component_id].update_performance(metrics)

    async def _update_all_component_states(self, step_results: Dict[str, Any]):
        """Update states of all components based on step results"""
        for component, results in step_results.items():
            if isinstance(results, dict) and "processing_time" in results:
                metrics = {
                    "processing_time": results["processing_time"],
                    "success": 1.0 if "error" not in results else 0.0,
                }
                self._update_component_state(component, ComponentStatus.ACTIVE, metrics)

    def _get_optimization_level(self) -> OptimizationLevel:
        """Get optimization level based on configuration"""
        if self.config.optimization_objective == OptimizationObjective.PERFORMANCE:
            return OptimizationLevel.AGGRESSIVE
        elif self.config.optimization_objective == OptimizationObjective.EFFICIENCY:
            return OptimizationLevel.CONSERVATIVE
        elif self.config.optimization_objective == OptimizationObjective.SPEED:
            return OptimizationLevel.BALANCED
        else:
            return OptimizationLevel.BALANCED

    async def _log_orchestration_metrics(self):
        """Log orchestration metrics to monitoring system"""
        metrics = {
            "components_active": len(
                [
                    s
                    for s in self.component_states.values()
                    if s.status == ComponentStatus.ACTIVE
                ]
            ),
            "components_error": len(
                [
                    s
                    for s in self.component_states.values()
                    if s.status == ComponentStatus.ERROR
                ]
            ),
            "decisions_made": len(self.decision_history),
            "uptime_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "recent_performance": self._calculate_recent_performance(),
            "performance_trend": self._calculate_performance_trend(),
            "resource_pressure": self._assess_resource_pressure(),
        }

        for metric_name, value in metrics.items():
            await self.monitoring.record_metric(f"orchestrator.{metric_name}", value)

    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "orchestration_id": self.orchestration_id,
            "mode": self.config.mode.value,
            "optimization_objective": self.config.optimization_objective.value,
            "is_running": self.is_running,
            "uptime": str(datetime.now() - self.start_time),
            "component_states": {
                comp_id: {
                    "status": state.status.value,
                    "last_update": state.last_update.isoformat(),
                    "success_rate": state.success_count
                    / max(1, state.success_count + state.error_count),
                    "performance_metrics": state.performance_metrics,
                }
                for comp_id, state in self.component_states.items()
            },
            "recent_decisions": [
                decision.to_dict() for decision in self.decision_history[-10:]
            ],
            "performance_summary": {
                "recent_performance": self._calculate_recent_performance(),
                "performance_trend": self._calculate_performance_trend(),
                "total_steps": len(self.global_performance_history),
            },
        }

    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights about optimization opportunities"""
        insights = {
            "performance_bottlenecks": [],
            "resource_optimization_opportunities": [],
            "architecture_improvement_suggestions": [],
            "learning_enhancement_recommendations": [],
        }

        # Analyze component performance
        for comp_id, state in self.component_states.items():
            if state.performance_metrics.get("processing_time", 0) > 1.0:
                insights["performance_bottlenecks"].append(
                    {
                        "component": comp_id,
                        "issue": "High processing time",
                        "current_value": state.performance_metrics["processing_time"],
                        "recommendation": "Consider optimization or resource scaling",
                    }
                )

        # Resource optimization opportunities
        resource_pressure = self._assess_resource_pressure()
        if resource_pressure > 0.8:
            insights["resource_optimization_opportunities"].append(
                {
                    "type": "memory_optimization",
                    "severity": "high",
                    "recommendation": "Enable memory optimization features",
                }
            )

        # Architecture improvements
        recent_performance = self._calculate_recent_performance()
        if recent_performance < 0.7:
            insights["architecture_improvement_suggestions"].append(
                {
                    "type": "nas_optimization",
                    "urgency": "medium",
                    "expected_benefit": "15-25% performance improvement",
                    "recommendation": "Schedule neural architecture search",
                }
            )

        # Learning enhancements
        performance_trend = self._calculate_performance_trend()
        if performance_trend < -0.02:
            insights["learning_enhancement_recommendations"].append(
                {
                    "type": "curriculum_adjustment",
                    "issue": "Declining performance trend",
                    "recommendation": "Adjust curriculum difficulty or exploration strategy",
                }
            )

        return insights

    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info(
            f"Shutting down Intelligent Orchestrator (ID: {self.orchestration_id})"
        )

        self.is_running = False

        # Update all component states to inactive
        for comp_id in self.component_states:
            self._update_component_state(comp_id, ComponentStatus.INACTIVE)

        # Log final metrics
        await self._log_orchestration_metrics()

        logger.info("Intelligent Orchestrator shutdown complete")


# Factory functions
def create_intelligent_orchestrator(
    mode: str = "adaptive",
    optimization_objective: str = "balanced",
    device: str = "cpu",
    **config_kwargs,
) -> IntelligentOrchestrator:
    """Create an intelligent orchestrator with specified configuration"""

    config = OrchestrationConfig(
        mode=OrchestrationMode(mode),
        optimization_objective=OptimizationObjective(optimization_objective),
        **config_kwargs,
    )

    return IntelligentOrchestrator(config, device)


def create_orchestration_config(
    enable_all: bool = True, performance_threshold: float = 0.8, **kwargs
) -> OrchestrationConfig:
    """Create an orchestration configuration"""

    config = OrchestrationConfig(
        enable_adaptive_learning=enable_all,
        enable_nas=enable_all,
        enable_multimodal=enable_all,
        enable_auto_optimization=enable_all,
        performance_threshold=performance_threshold,
        **kwargs,
    )

    return config

"""
Intelligent Orchestrator for GRPO Agent Framework

This module provides an intelligent orchestration system that coordinates all
framework components including adaptive learning, neural architecture search,
multimodal processing, and training optimization.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .adaptive_learning_controller import (
    AdaptiveLearningController,
    create_adaptive_learning_controller,
)
from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .enhanced_state_management import get_state_service
from .error_handling import ErrorHandler
from .intelligent_orchestrator_logic import (
    assess_adaptation_need,
    assess_resource_pressure,
    calculate_performance_trend,
    calculate_recent_performance,
    make_adaptive_decision,
    make_automated_decision,
    make_conservative_decision,
)
from .intelligent_orchestrator_models import (
    ComponentState,
    ComponentStatus,
    OptimizationObjective,
    OrchestrationConfig,
    OrchestrationDecision,
    OrchestrationMode,
)
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

ORCHESTRATOR_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
)


class IntelligentOrchestrator:
    """Main intelligent orchestration system"""

    def __init__(self, config: OrchestrationConfig = None, device: str = "cpu"):
        self.config = config or OrchestrationConfig()
        self.device = device

        # Core components
        self.adaptive_learning_controller: AdaptiveLearningController | None = None
        self.nas_controller: NeuralArchitectureSearch | None = None
        self.multimodal_processor: MultimodalProcessor | None = None
        self.performance_optimizer: PerformanceOptimizer | None = None

        # State management
        self.component_states: dict[str, ComponentState] = {}
        self.decision_history: list[OrchestrationDecision] = []

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
        self.global_performance_history: list[float] = []
        self.resource_usage_history: dict[str, list[float]] = {
            "cpu": [],
            "memory": [],
            "gpu_memory": [],
        }

    async def initialize(
        self,
        enabled_modalities: list[str] = None,
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

        except ORCHESTRATOR_EXCEPTIONS as e:
            self.error_handler.handle_error(e, "orchestrator", "initialize")
            raise

    @monitor_async_function("intelligent_orchestration")
    async def orchestrate_training_step(
        self,
        task_id: str,
        state: Any,
        available_actions: list[str],
        reward: float,
        success: bool,
        current_hyperparams: dict[str, float],
        multimodal_inputs: list[ModalityInput] = None,
    ) -> tuple[dict[str, Any], OrchestrationDecision]:
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

        except ORCHESTRATOR_EXCEPTIONS as e:
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
    ) -> ArchitectureConfig | None:
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

        except ORCHESTRATOR_EXCEPTIONS as e:
            self.error_handler.handle_error(e, "orchestrator", "optimize_architecture")
            self._update_component_state("nas", ComponentStatus.ERROR)
            return None

    async def _make_orchestration_decision(
        self, task_id: str, reward: float, success: bool, step_results: dict[str, Any]
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
        """Fully automated decision making."""
        return make_automated_decision(
            self.config,
            reward,
            recent_performance,
            resource_pressure,
        )

    async def _adaptive_decision_making(
        self,
        task_id: str,
        reward: float,
        success: bool,
        recent_performance: float,
        resource_pressure: float,
        step_results: dict[str, Any],
    ) -> OrchestrationDecision:
        """Adaptive decision making based on learning insights."""

        learning_insights = {}
        if self.adaptive_learning_controller:
            learning_insights = (
                await self.adaptive_learning_controller.get_learning_insights()
            )

        performance_trend = self._calculate_performance_trend()
        nas_optimization_due = (
            datetime.now() - self.last_optimization > self.config.nas_interval
        )
        return make_adaptive_decision(
            self.config,
            learning_insights,
            performance_trend,
            recent_performance,
            nas_optimization_due,
        )

    async def _conservative_decision_making(
        self, task_id: str, reward: float, success: bool, recent_performance: float
    ) -> OrchestrationDecision:
        """Conservative decision making for manual and semi-automated modes."""
        return make_conservative_decision(self.config, success, recent_performance)

    def _calculate_recent_performance(self) -> float:
        """Calculate recent performance summary (robust to outliers)."""
        return calculate_recent_performance(self.global_performance_history)

    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend (positive = improving)."""
        return calculate_performance_trend(self.global_performance_history)

    def _assess_resource_pressure(self) -> float:
        """Assess current resource pressure (0.0 = low, 1.0 = high)."""
        return assess_resource_pressure(self.resource_usage_history)

    def _assess_adaptation_need(
        self, learning_insights: dict[str, Any], performance_trend: float
    ) -> bool:
        """Assess if adaptation is needed."""
        return assess_adaptation_need(learning_insights, performance_trend)

    def _update_component_state(
        self,
        component_id: str,
        status: ComponentStatus,
        metrics: dict[str, float] = None,
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

    async def _update_all_component_states(self, step_results: dict[str, Any]):
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

    async def _record_monitoring_metric(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ):
        """Record a metric against sync or async monitoring implementations."""
        record_metric = getattr(self.monitoring, "record_metric", None)
        if not callable(record_metric):
            return

        maybe_awaitable = record_metric(metric_name, value, labels or {})
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable

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
            await self._record_monitoring_metric(
                f"orchestrator.{metric_name}",
                value,
            )

    def get_orchestration_status(self) -> dict[str, Any]:
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

    def get_optimization_insights(self) -> dict[str, Any]:
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


__all__ = [
    "ComponentState",
    "ComponentStatus",
    "IntelligentOrchestrator",
    "OptimizationObjective",
    "OrchestrationConfig",
    "OrchestrationDecision",
    "OrchestrationMode",
    "create_intelligent_orchestrator",
    "create_orchestration_config",
]

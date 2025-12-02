"""
Unit tests for the Intelligent Orchestrator module.

Tests cover orchestration configuration, component states, decision making,
and intelligent coordination of framework components.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.intelligent_orchestrator import (
    ComponentState,
    ComponentStatus,
    IntelligentOrchestrator,
    OptimizationObjective,
    OrchestrationConfig,
    OrchestrationDecision,
    OrchestrationMode,
    create_intelligent_orchestrator,
    create_orchestration_config,
)


class TestOrchestrationMode:
    """Test OrchestrationMode enum."""

    def test_orchestration_mode_values(self):
        """Test that all orchestration modes have expected values."""
        assert OrchestrationMode.MANUAL.value == "manual"
        assert OrchestrationMode.SEMI_AUTOMATED.value == "semi_automated"
        assert OrchestrationMode.FULLY_AUTOMATED.value == "fully_automated"
        assert OrchestrationMode.ADAPTIVE.value == "adaptive"


class TestOptimizationObjective:
    """Test OptimizationObjective enum."""

    def test_optimization_objective_values(self):
        """Test that all optimization objectives have expected values."""
        assert OptimizationObjective.PERFORMANCE.value == "performance"
        assert OptimizationObjective.EFFICIENCY.value == "efficiency"
        assert OptimizationObjective.SPEED.value == "speed"
        assert OptimizationObjective.ROBUSTNESS.value == "robustness"
        assert OptimizationObjective.BALANCED.value == "balanced"


class TestComponentStatus:
    """Test ComponentStatus enum."""

    def test_component_status_values(self):
        """Test that all component statuses have expected values."""
        assert ComponentStatus.INACTIVE.value == "inactive"
        assert ComponentStatus.INITIALIZING.value == "initializing"
        assert ComponentStatus.ACTIVE.value == "active"
        assert ComponentStatus.ERROR.value == "error"
        assert ComponentStatus.OPTIMIZING.value == "optimizing"
        assert ComponentStatus.UPDATING.value == "updating"


class TestOrchestrationConfig:
    """Test OrchestrationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestrationConfig()

        assert config.mode == OrchestrationMode.ADAPTIVE
        assert config.optimization_objective == OptimizationObjective.BALANCED
        assert config.enable_adaptive_learning is True
        assert config.enable_nas is True
        assert config.enable_multimodal is True
        assert config.enable_auto_optimization is True
        assert config.performance_threshold == 0.8
        assert config.max_gpu_memory_usage == 0.9
        assert config.max_cpu_usage == 0.8
        assert config.max_concurrent_processes == 4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OrchestrationConfig(
            mode=OrchestrationMode.MANUAL,
            optimization_objective=OptimizationObjective.PERFORMANCE,
            enable_nas=False,
            performance_threshold=0.9,
            max_concurrent_processes=8,
        )

        assert config.mode == OrchestrationMode.MANUAL
        assert config.optimization_objective == OptimizationObjective.PERFORMANCE
        assert config.enable_nas is False
        assert config.performance_threshold == 0.9
        assert config.max_concurrent_processes == 8

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = OrchestrationConfig()
        config_dict = config.to_dict()

        assert config_dict["mode"] == "adaptive"
        assert config_dict["optimization_objective"] == "balanced"
        assert config_dict["enable_adaptive_learning"] is True
        assert config_dict["performance_threshold"] == 0.8
        assert "adaptation_interval_minutes" in config_dict
        assert "nas_interval_hours" in config_dict


class TestComponentState:
    """Test ComponentState dataclass."""

    def test_component_state_creation(self):
        """Test creating a ComponentState."""
        state = ComponentState(
            component_id="test_component",
            status=ComponentStatus.ACTIVE,
            last_update=datetime.now(),
        )

        assert state.component_id == "test_component"
        assert state.status == ComponentStatus.ACTIVE
        assert state.error_count == 0
        assert state.success_count == 0
        assert state.performance_metrics == {}

    def test_update_performance(self):
        """Test updating performance metrics."""
        state = ComponentState(
            component_id="test_component",
            status=ComponentStatus.ACTIVE,
            last_update=datetime.now(),
        )

        metrics = {"accuracy": 0.95, "latency": 0.05}
        state.update_performance(metrics)

        assert state.performance_metrics["accuracy"] == 0.95
        assert state.performance_metrics["latency"] == 0.05
        assert state.success_count == 1

    def test_record_error(self):
        """Test recording errors."""
        state = ComponentState(
            component_id="test_component",
            status=ComponentStatus.ACTIVE,
            last_update=datetime.now(),
        )

        state.record_error("Test error occurred")

        assert state.error_count == 1
        assert state.status == ComponentStatus.ERROR
        assert state.metadata["last_error"] == "Test error occurred"
        assert "last_error_time" in state.metadata


class TestOrchestrationDecision:
    """Test OrchestrationDecision dataclass."""

    def test_decision_creation(self):
        """Test creating an OrchestrationDecision."""
        decision = OrchestrationDecision(
            decision_id="dec_001",
            timestamp=datetime.now(),
            decision_type="performance_optimization",
            component="orchestrator",
            action="optimize",
            reasoning="Performance below threshold",
            expected_benefit=0.15,
            confidence=0.85,
        )

        assert decision.decision_id == "dec_001"
        assert decision.decision_type == "performance_optimization"
        assert decision.action == "optimize"
        assert decision.expected_benefit == 0.15
        assert decision.confidence == 0.85

    def test_decision_to_dict(self):
        """Test decision serialization to dictionary."""
        decision = OrchestrationDecision(
            decision_id="dec_001",
            timestamp=datetime.now(),
            decision_type="test_type",
            component="test_component",
            action="test_action",
            reasoning="test reasoning",
            expected_benefit=0.1,
            confidence=0.9,
            parameters={"key": "value"},
        )

        decision_dict = decision.to_dict()

        assert decision_dict["decision_id"] == "dec_001"
        assert decision_dict["decision_type"] == "test_type"
        assert decision_dict["parameters"] == {"key": "value"}
        assert "timestamp" in decision_dict


class TestIntelligentOrchestrator:
    """Test IntelligentOrchestrator class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch("core.intelligent_orchestrator.get_monitoring_service") as mock_monitoring, \
             patch("core.intelligent_orchestrator.get_state_service") as mock_state, \
             patch("core.intelligent_orchestrator.ErrorHandler") as mock_error_handler:
            mock_monitoring.return_value = MagicMock()
            mock_monitoring.return_value.record_metric = AsyncMock()
            mock_state.return_value = MagicMock()
            yield {
                "monitoring": mock_monitoring,
                "state": mock_state,
                "error_handler": mock_error_handler,
            }

    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create an orchestrator instance for testing."""
        config = OrchestrationConfig(
            enable_adaptive_learning=False,
            enable_nas=False,
            enable_multimodal=False,
        )
        return IntelligentOrchestrator(config)

    def test_orchestrator_creation(self, mock_dependencies):
        """Test orchestrator creation with default config."""
        orchestrator = IntelligentOrchestrator()

        assert orchestrator.config.mode == OrchestrationMode.ADAPTIVE
        assert orchestrator.is_running is False
        assert orchestrator.orchestration_id is not None
        assert len(orchestrator.decision_history) == 0

    def test_orchestrator_with_custom_config(self, mock_dependencies):
        """Test orchestrator creation with custom config."""
        config = OrchestrationConfig(
            mode=OrchestrationMode.MANUAL,
            performance_threshold=0.9,
        )
        orchestrator = IntelligentOrchestrator(config)

        assert orchestrator.config.mode == OrchestrationMode.MANUAL
        assert orchestrator.config.performance_threshold == 0.9

    @pytest.mark.asyncio
    async def test_orchestrator_initialize(self, orchestrator, mock_dependencies):
        """Test orchestrator initialization."""
        with patch("core.intelligent_orchestrator.PerformanceOptimizer"):
            await orchestrator.initialize()

            assert orchestrator.is_running is True
            assert orchestrator.performance_optimizer is not None

    def test_calculate_recent_performance_empty(self, orchestrator):
        """Test performance calculation with empty history."""
        result = orchestrator._calculate_recent_performance()
        assert result == 0.5  # Default value

    def test_calculate_recent_performance(self, orchestrator):
        """Test performance calculation with history."""
        orchestrator.global_performance_history = [0.7, 0.8, 0.9, 0.85, 0.95]
        result = orchestrator._calculate_recent_performance()
        assert 0.85 <= result <= 0.86  # Average of the values

    def test_calculate_performance_trend_insufficient_data(self, orchestrator):
        """Test trend calculation with insufficient data."""
        orchestrator.global_performance_history = [0.8, 0.9]
        result = orchestrator._calculate_performance_trend()
        assert result == 0.0

    def test_calculate_performance_trend(self, orchestrator):
        """Test trend calculation with sufficient data."""
        # Earlier values: 0.6, later values: 0.8 -> positive trend
        orchestrator.global_performance_history = (
            [0.5, 0.55, 0.6, 0.65, 0.6, 0.58, 0.62, 0.6, 0.61, 0.59] +
            [0.75, 0.78, 0.8, 0.82, 0.79, 0.81, 0.8, 0.83, 0.81, 0.8]
        )
        result = orchestrator._calculate_performance_trend()
        assert result > 0  # Positive trend

    def test_assess_resource_pressure_empty(self, orchestrator):
        """Test resource pressure with empty history."""
        result = orchestrator._assess_resource_pressure()
        assert result == 0.5  # Default moderate pressure

    def test_assess_resource_pressure(self, orchestrator):
        """Test resource pressure calculation."""
        orchestrator.resource_usage_history = {
            "cpu": [0.7, 0.75, 0.8],
            "memory": [0.6, 0.65, 0.7],
            "gpu_memory": [],
        }
        result = orchestrator._assess_resource_pressure()
        assert 0.6 <= result <= 0.8

    def test_update_component_state_new(self, orchestrator):
        """Test updating state for a new component."""
        orchestrator._update_component_state("new_component", ComponentStatus.ACTIVE)

        assert "new_component" in orchestrator.component_states
        assert orchestrator.component_states["new_component"].status == ComponentStatus.ACTIVE

    def test_update_component_state_existing(self, orchestrator):
        """Test updating state for an existing component."""
        orchestrator._update_component_state("component", ComponentStatus.INITIALIZING)
        orchestrator._update_component_state("component", ComponentStatus.ACTIVE)

        assert orchestrator.component_states["component"].status == ComponentStatus.ACTIVE

    def test_update_component_state_with_metrics(self, orchestrator):
        """Test updating state with performance metrics."""
        metrics = {"accuracy": 0.9, "latency": 0.05}
        orchestrator._update_component_state("component", ComponentStatus.ACTIVE, metrics)

        state = orchestrator.component_states["component"]
        assert state.performance_metrics["accuracy"] == 0.9
        assert state.success_count == 1

    def test_get_optimization_level_performance(self, mock_dependencies):
        """Test optimization level for performance objective."""
        from core.performance_optimizer import OptimizationLevel

        config = OrchestrationConfig(
            optimization_objective=OptimizationObjective.PERFORMANCE
        )
        orchestrator = IntelligentOrchestrator(config)
        level = orchestrator._get_optimization_level()
        assert level == OptimizationLevel.AGGRESSIVE

    def test_get_optimization_level_efficiency(self, mock_dependencies):
        """Test optimization level for efficiency objective."""
        from core.performance_optimizer import OptimizationLevel

        config = OrchestrationConfig(
            optimization_objective=OptimizationObjective.EFFICIENCY
        )
        orchestrator = IntelligentOrchestrator(config)
        level = orchestrator._get_optimization_level()
        assert level == OptimizationLevel.CONSERVATIVE

    def test_get_orchestration_status(self, orchestrator):
        """Test getting orchestration status."""
        orchestrator._update_component_state("test_comp", ComponentStatus.ACTIVE)
        status = orchestrator.get_orchestration_status()

        assert "orchestration_id" in status
        assert status["mode"] == "adaptive"
        assert "component_states" in status
        assert "performance_summary" in status

    def test_get_optimization_insights_empty(self, orchestrator):
        """Test getting optimization insights with no issues."""
        orchestrator.global_performance_history = [0.9] * 50
        insights = orchestrator.get_optimization_insights()

        assert "performance_bottlenecks" in insights
        assert "resource_optimization_opportunities" in insights
        assert "architecture_improvement_suggestions" in insights

    def test_get_optimization_insights_performance_bottleneck(self, orchestrator):
        """Test optimization insights detecting performance bottlenecks."""
        orchestrator._update_component_state(
            "slow_component",
            ComponentStatus.ACTIVE,
            {"processing_time": 2.0}
        )
        insights = orchestrator.get_optimization_insights()

        assert len(insights["performance_bottlenecks"]) > 0
        assert insights["performance_bottlenecks"][0]["component"] == "slow_component"

    def test_assess_adaptation_need_no_signals(self, orchestrator):
        """Test adaptation need with no signals."""
        result = orchestrator._assess_adaptation_need({}, 0.1)
        assert result is False

    def test_assess_adaptation_need_declining_performance(self, orchestrator):
        """Test adaptation need with declining performance."""
        result = orchestrator._assess_adaptation_need({}, -0.1)
        assert result is True

    def test_assess_adaptation_need_low_exploration(self, orchestrator):
        """Test adaptation need with low exploration."""
        learning_insights = {
            "exploration_insights": {"current_epsilon": 0.01}
        }
        result = orchestrator._assess_adaptation_need(learning_insights, 0.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator, mock_dependencies):
        """Test orchestrator shutdown."""
        orchestrator.is_running = True
        orchestrator._update_component_state("comp1", ComponentStatus.ACTIVE)

        await orchestrator.shutdown()

        assert orchestrator.is_running is False
        assert orchestrator.component_states["comp1"].status == ComponentStatus.INACTIVE

    @pytest.mark.asyncio
    async def test_automated_decision_low_performance(self, orchestrator):
        """Test automated decision making with low performance."""
        decision = await orchestrator._automated_decision_making(
            task_id="task_001",
            reward=0.5,
            success=False,
            recent_performance=0.6,  # Below threshold
            resource_pressure=0.5,
        )

        assert decision.decision_type == "performance_optimization"
        assert decision.action == "aggressive_optimize"

    @pytest.mark.asyncio
    async def test_automated_decision_resource_constrained(self, orchestrator):
        """Test automated decision with resource constraints."""
        decision = await orchestrator._automated_decision_making(
            task_id="task_001",
            reward=0.5,
            success=False,
            recent_performance=0.6,
            resource_pressure=0.85,  # High resource pressure
        )

        assert decision.decision_type == "resource_optimization"
        assert decision.action == "gentle_optimize"

    @pytest.mark.asyncio
    async def test_automated_decision_good_performance(self, orchestrator):
        """Test automated decision with good performance."""
        decision = await orchestrator._automated_decision_making(
            task_id="task_001",
            reward=0.95,
            success=True,
            recent_performance=0.9,
            resource_pressure=0.5,
        )

        assert decision.decision_type == "exploration"
        assert decision.action == "explore_improvements"

    @pytest.mark.asyncio
    async def test_conservative_decision_low_performance(self, orchestrator):
        """Test conservative decision with very low performance."""
        orchestrator.config.performance_threshold = 0.8
        decision = await orchestrator._conservative_decision_making(
            task_id="task_001",
            reward=0.4,
            success=False,
            recent_performance=0.5,  # Below 0.8 * 0.8 = 0.64
        )

        assert decision.decision_type == "alert"
        assert decision.action == "request_human_intervention"

    @pytest.mark.asyncio
    async def test_conservative_decision_recent_failure(self, orchestrator):
        """Test conservative decision after failure."""
        decision = await orchestrator._conservative_decision_making(
            task_id="task_001",
            reward=0.7,
            success=False,
            recent_performance=0.75,
        )

        assert decision.decision_type == "recovery"
        assert decision.action == "gentle_recovery"


class TestFactoryFunctions:
    """Test factory functions for creating orchestrators."""

    @patch("core.intelligent_orchestrator.get_monitoring_service")
    @patch("core.intelligent_orchestrator.get_state_service")
    def test_create_intelligent_orchestrator_default(self, mock_state, mock_monitoring):
        """Test creating orchestrator with defaults."""
        mock_monitoring.return_value = MagicMock()
        mock_state.return_value = MagicMock()

        orchestrator = create_intelligent_orchestrator()

        assert orchestrator.config.mode == OrchestrationMode.ADAPTIVE
        assert orchestrator.config.optimization_objective == OptimizationObjective.BALANCED

    @patch("core.intelligent_orchestrator.get_monitoring_service")
    @patch("core.intelligent_orchestrator.get_state_service")
    def test_create_intelligent_orchestrator_custom(self, mock_state, mock_monitoring):
        """Test creating orchestrator with custom settings."""
        mock_monitoring.return_value = MagicMock()
        mock_state.return_value = MagicMock()

        orchestrator = create_intelligent_orchestrator(
            mode="manual",
            optimization_objective="performance",
            device="cuda",
        )

        assert orchestrator.config.mode == OrchestrationMode.MANUAL
        assert orchestrator.config.optimization_objective == OptimizationObjective.PERFORMANCE
        assert orchestrator.device == "cuda"

    def test_create_orchestration_config_enable_all(self):
        """Test creating config with all features enabled."""
        config = create_orchestration_config(enable_all=True)

        assert config.enable_adaptive_learning is True
        assert config.enable_nas is True
        assert config.enable_multimodal is True
        assert config.enable_auto_optimization is True

    def test_create_orchestration_config_custom_threshold(self):
        """Test creating config with custom threshold."""
        config = create_orchestration_config(
            enable_all=False,
            performance_threshold=0.95,
        )

        assert config.enable_adaptive_learning is False
        assert config.performance_threshold == 0.95

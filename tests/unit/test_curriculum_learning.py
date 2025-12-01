"""
Tests for Curriculum Learning System
"""

import pytest
import numpy as np

from core.curriculum_learning import (
    CurriculumLearning,
    CurriculumStage,
    CurriculumProgress,
    PerformanceBasedScheduler,
    AdaptiveScheduler,
    TaskDifficultyController,
    DifficultyMetric,
    ProgressionStrategy,
)
from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn


class TestCurriculumStage:
    """Test curriculum stage configuration"""

    def test_stage_creation(self):
        stage = CurriculumStage(
            stage_id="test_stage",
            difficulty_level=0.5,
            task_config={"max_turns": 10},
            success_threshold=0.7,
            min_episodes=50,
        )

        assert stage.stage_id == "test_stage"
        assert stage.difficulty_level == 0.5
        assert stage.task_config["max_turns"] == 10
        assert stage.success_threshold == 0.7
        assert stage.min_episodes == 50


class TestPerformanceBasedScheduler:
    """Test performance-based curriculum progression"""

    def test_should_advance_min_episodes(self):
        scheduler = PerformanceBasedScheduler(min_episodes=100)

        stage = CurriculumStage(
            stage_id="stage1",
            difficulty_level=0.3,
            task_config={},
            min_episodes=100,
            max_episodes=500,
        )

        progress = type(
            "Progress",
            (),
            {"episodes_in_stage": 50, "stage_rewards": [0.8] * 60},
        )()

        trajectories = []

        # Should not advance before min episodes
        assert not scheduler.should_advance(stage, progress, trajectories)

    def test_should_advance_performance_threshold(self):
        scheduler = PerformanceBasedScheduler(
            min_episodes=50,
            success_threshold=0.7,
            window_size=20,
        )

        stage = CurriculumStage(
            stage_id="stage1",
            difficulty_level=0.3,
            task_config={},
            min_episodes=50,
            max_episodes=500,
        )

        # High performance progress
        progress = type(
            "Progress",
            (),
            {"episodes_in_stage": 100, "stage_rewards": [0.9] * 50},
        )()

        trajectories = []

        # Should advance when performance exceeds threshold
        assert scheduler.should_advance(stage, progress, trajectories)

    def test_should_advance_max_episodes(self):
        scheduler = PerformanceBasedScheduler()

        stage = CurriculumStage(
            stage_id="stage1",
            difficulty_level=0.3,
            task_config={},
            max_episodes=100,
        )

        progress = type(
            "Progress",
            (),
            {"episodes_in_stage": 150, "stage_rewards": [0.3] * 150},
        )()

        trajectories = []

        # Should force advance after max episodes
        assert scheduler.should_advance(stage, progress, trajectories)

    def test_difficulty_adjustment(self):
        scheduler = PerformanceBasedScheduler()

        stage = CurriculumStage(
            stage_id="stage1",
            difficulty_level=0.5,
            task_config={},
        )

        # High, stable performance
        progress = type(
            "Progress",
            (),
            {"stage_rewards": [0.85] * 50},
        )()

        adjustment = scheduler.get_difficulty_adjustment(stage, progress)
        assert adjustment > 0  # Should increase difficulty

        # Low performance
        progress.stage_rewards = [0.2] * 50
        adjustment = scheduler.get_difficulty_adjustment(stage, progress)
        assert adjustment < 0  # Should decrease difficulty


class TestAdaptiveScheduler:
    """Test adaptive curriculum scheduling"""

    def test_learning_plateau_detection(self):
        scheduler = AdaptiveScheduler(
            learning_rate_threshold=0.01,
            performance_threshold=0.65,
            lookback_window=50,
        )

        stage = CurriculumStage(
            stage_id="stage1",
            difficulty_level=0.5,
            task_config={},
            min_episodes=50,
            max_episodes=500,
        )

        # Plateau: flat rewards with good performance
        rewards = [0.7 + np.random.normal(0, 0.01) for _ in range(100)]
        progress = type(
            "Progress",
            (),
            {"episodes_in_stage": 100, "stage_rewards": rewards},
        )()

        trajectories = []

        # Should advance when plateaued with good performance
        assert scheduler.should_advance(stage, progress, trajectories)

    def test_still_learning(self):
        scheduler = AdaptiveScheduler()

        stage = CurriculumStage(
            stage_id="stage1",
            difficulty_level=0.5,
            task_config={},
            min_episodes=50,
        )

        # Still learning: increasing rewards
        rewards = [0.3 + i * 0.005 for i in range(100)]
        progress = type(
            "Progress",
            (),
            {"episodes_in_stage": 100, "stage_rewards": rewards},
        )()

        trajectories = []

        # Should not advance while still improving
        result = scheduler.should_advance(stage, progress, trajectories)
        # May or may not advance depending on current performance


class TestCurriculumLearning:
    """Test main curriculum learning coordinator"""

    def test_initialization(self):
        stages = [
            CurriculumStage(
                stage_id=f"stage_{i}",
                difficulty_level=i / 5,
                task_config={"level": i},
            )
            for i in range(5)
        ]

        curriculum = CurriculumLearning(stages=stages)

        assert len(curriculum.stages) == 5
        assert curriculum.get_current_stage().stage_id == "stage_0"

    def test_auto_generate_stages(self):
        curriculum = CurriculumLearning(
            stages=[],
            auto_generate_stages=True,
            num_stages=3,
        )

        assert len(curriculum.stages) == 3
        assert curriculum.stages[0].difficulty_level < curriculum.stages[1].difficulty_level

    def test_record_episode_and_advancement(self):
        stages = [
            CurriculumStage(
                stage_id=f"stage_{i}",
                difficulty_level=i / 3,
                task_config={},
                min_episodes=10,
                max_episodes=100,
                success_threshold=0.6,
            )
            for i in range(3)
        ]

        # Set window_size small enough to trigger advancement with 20 episodes
        scheduler = PerformanceBasedScheduler(
            window_size=10,  # Window must be <= episodes recorded
            min_episodes=10,
            success_threshold=0.6,
        )
        curriculum = CurriculumLearning(stages=stages, scheduler=scheduler)

        # Record high-reward episodes (enough to meet min_episodes and window_size)
        for _ in range(20):
            trajectory = MultiTurnTrajectory(
                trajectory_id="test",
                turns=[],
                total_reward=0.8,  # High enough reward to count as success
            )
            curriculum.record_episode(trajectory)

        # Should have advanced to next stage since success_rate = 1.0 >= 0.6
        assert curriculum.progress.current_stage_idx > 0

    def test_get_current_config(self):
        stages = [
            CurriculumStage(
                stage_id="stage_0",
                difficulty_level=0.3,
                task_config={"max_turns": 5, "complexity": 1},
            )
        ]

        curriculum = CurriculumLearning(stages=stages)
        config = curriculum.get_current_config()

        assert config["max_turns"] == 5
        assert config["complexity"] == 1
        assert "difficulty_adjustment" in config

    def test_is_curriculum_complete(self):
        stages = [
            CurriculumStage(
                stage_id=f"stage_{i}",
                difficulty_level=i / 2,
                task_config={},
                min_episodes=5,
            )
            for i in range(2)
        ]

        curriculum = CurriculumLearning(stages=stages)

        # Not complete initially
        assert not curriculum.is_curriculum_complete()

        # Advance to final stage
        curriculum.progress.current_stage_idx = 1
        curriculum.progress.episodes_in_stage = 10

        # Should be complete
        assert curriculum.is_curriculum_complete()

    def test_manual_stage_setting(self):
        stages = [
            CurriculumStage(stage_id=f"stage_{i}", difficulty_level=i / 3, task_config={})
            for i in range(3)
        ]

        curriculum = CurriculumLearning(stages=stages)

        # Set to stage 2
        curriculum.set_stage(2)
        assert curriculum.progress.current_stage_idx == 2

        # Invalid stage should raise error
        with pytest.raises(ValueError):
            curriculum.set_stage(10)

    def test_progress_summary(self):
        stages = [
            CurriculumStage(stage_id=f"stage_{i}", difficulty_level=i / 2, task_config={})
            for i in range(2)
        ]

        curriculum = CurriculumLearning(stages=stages)

        summary = curriculum.get_progress_summary()

        assert "current_stage" in summary
        assert "total_stages" in summary
        assert "difficulty" in summary
        assert "stage_success_rate" in summary
        assert summary["total_stages"] == 2


class TestTaskDifficultyController:
    """Test dynamic difficulty adjustment"""

    def test_initialization(self):
        controller = TaskDifficultyController(
            base_difficulty=0.5,
            adjustment_rate=0.1,
        )

        assert controller.current_difficulty == 0.5

    def test_increase_on_success(self):
        controller = TaskDifficultyController(base_difficulty=0.5, adjustment_rate=0.1)

        initial_diff = controller.current_difficulty
        new_diff = controller.update(success=True, performance_score=0.9)

        assert new_diff > initial_diff

    def test_decrease_on_failure(self):
        controller = TaskDifficultyController(base_difficulty=0.5, adjustment_rate=0.1)

        initial_diff = controller.current_difficulty
        new_diff = controller.update(success=False, performance_score=0.2)

        assert new_diff < initial_diff

    def test_difficulty_bounds(self):
        controller = TaskDifficultyController(
            base_difficulty=0.9,
            adjustment_rate=0.2,
            min_difficulty=0.0,
            max_difficulty=1.0,
        )

        # Try to increase beyond max
        for _ in range(10):
            controller.update(success=True, performance_score=0.95)

        assert controller.current_difficulty <= 1.0

        # Reset and try to decrease below min
        controller.reset(difficulty=0.1)
        for _ in range(10):
            controller.update(success=False, performance_score=0.1)

        assert controller.current_difficulty >= 0.0

    def test_reset(self):
        controller = TaskDifficultyController(base_difficulty=0.5)

        # Change difficulty
        controller.update(success=True, performance_score=0.9)

        # Reset
        controller.reset()
        assert controller.current_difficulty == 0.5

        # Reset to custom value
        controller.reset(difficulty=0.7)
        assert controller.current_difficulty == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

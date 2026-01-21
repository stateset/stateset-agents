"""
Tests for long-term planning utilities.
"""

from stateset_agents.core.long_term_planning import (
    HeuristicPlanner,
    PlanStatus,
    PlanningConfig,
    PlanningManager,
)


def test_heuristic_planner_creates_steps():
    planner = HeuristicPlanner()
    config = PlanningConfig(enabled=True, max_steps=3)
    plan = planner.create_plan(
        "Book flights, reserve hotel, and plan itinerary", config
    )

    assert plan.goal.startswith("Book flights")
    assert len(plan.steps) == 3
    assert plan.steps[0].status == PlanStatus.ACTIVE


def test_planning_manager_builds_system_message():
    manager = PlanningManager(config=PlanningConfig(enabled=True))
    messages = [{"role": "user", "content": "Plan a weekend trip to Austin"}]
    message = manager.build_plan_message(messages, context={"conversation_id": "c1"})

    assert message is not None
    assert message["role"] == "system"
    assert "Long-term plan" in message["content"]


def test_planning_manager_applies_updates():
    manager = PlanningManager(config=PlanningConfig(enabled=True))
    messages = [{"role": "user", "content": "Draft report and send it"}]
    manager.build_plan_message(messages, context={"conversation_id": "c2"})

    manager.apply_update(
        "c2",
        {"action": "advance"},
    )
    plan = manager.get_plan("c2")

    assert plan is not None
    assert plan.progress() > 0.0


def test_planning_manager_uses_scenario_goal():
    manager = PlanningManager(config=PlanningConfig(enabled=True))
    messages = [{"role": "user", "content": "Hello"}]
    message = manager.build_plan_message(
        messages,
        context={
            "conversation_id": "c3",
            "scenario": {"user_goal": "Write a project plan"},
        },
    )

    assert message is not None
    assert "Write a project plan" in message["content"]


def test_planning_manager_clear_conversation():
    manager = PlanningManager(config=PlanningConfig(enabled=True))
    messages = [{"role": "user", "content": "Plan a launch checklist"}]
    manager.build_plan_message(messages, context={"conversation_id": "c4"})

    assert manager.get_plan("c4") is not None

    manager.clear_conversation("c4")
    assert manager.get_plan("c4") is None


def test_planning_manager_keeps_goal_without_explicit_update():
    manager = PlanningManager(config=PlanningConfig(enabled=True))
    messages = [{"role": "user", "content": "Plan a product launch"}]
    manager.build_plan_message(messages, context={"conversation_id": "c5"})

    plan = manager.get_plan("c5")
    assert plan is not None

    new_messages = [{"role": "user", "content": "What should we do next?"}]
    manager.build_plan_message(new_messages, context={"conversation_id": "c5"})

    plan_after = manager.get_plan("c5")
    assert plan_after is not None
    assert plan_after.goal == plan.goal


def test_planning_manager_updates_goal_with_explicit_override():
    manager = PlanningManager(config=PlanningConfig(enabled=True))
    messages = [{"role": "user", "content": "Plan a release"}]
    manager.build_plan_message(messages, context={"conversation_id": "c6"})

    manager.build_plan_message(
        [{"role": "user", "content": "Switch focus"}],
        context={"conversation_id": "c6", "plan_goal": "Plan a marketing campaign"},
    )
    plan = manager.get_plan("c6")

    assert plan is not None
    assert plan.goal == "Plan a marketing campaign"


def test_planning_manager_applies_update_even_with_interval():
    manager = PlanningManager(config=PlanningConfig(enabled=True, update_interval=5))
    messages = [{"role": "user", "content": "Prepare a launch plan"}]
    manager.build_plan_message(messages, context={"conversation_id": "c7"})

    manager.build_plan_message(
        [{"role": "user", "content": "Next step?"}],
        context={"conversation_id": "c7", "plan_update": {"action": "advance"}},
    )
    plan = manager.get_plan("c7")

    assert plan is not None
    assert plan.progress() > 0.0

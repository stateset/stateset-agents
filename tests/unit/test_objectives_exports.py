"""The objectives surface is lazily exported from ``stateset_agents.training``."""


def test_objectives_are_lazily_exported():
    import stateset_agents.training as training

    assert training.PolicyObjective.__name__ == "PolicyObjective"
    assert training.PolicyLossResult.__name__ == "PolicyLossResult"
    assert "grpo" in training.OBJECTIVES
    assert callable(training.policy_loss)
    assert callable(training.compute_advantages)


def test_every_preset_round_trips_through_with():
    from stateset_agents.training import OBJECTIVES

    for name, obj in OBJECTIVES.items():
        assert obj.name == name
        assert obj.with_() == obj

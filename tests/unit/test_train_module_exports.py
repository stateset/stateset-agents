from stateset_agents.training.train import (
    AutoTrainer,
    train,
    train_customer_service_agent,
    train_task_agent,
    train_tutoring_agent,
)


def test_train_module_exports_remain_available() -> None:
    assert callable(train)
    assert AutoTrainer.__name__ == "AutoTrainer"
    assert callable(train_customer_service_agent)
    assert callable(train_tutoring_agent)
    assert callable(train_task_agent)

"""NeuralRewardModel must embed text with the encoder it was given."""

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.neural_reward_trainer import (  # noqa: E402
    NeuralRewardModel,
    TrajectoryDataset,
    create_large_neural_reward_function,
)


def test_encoder_is_used_for_prediction_and_training_data():
    calls = []

    def encoder(text):
        calls.append(text)
        return torch.full((8,), float(len(text)))

    model = NeuralRewardModel(embedding_dim=8, hidden_dim=4, encoder=encoder)
    model.predict_reward("prompt", "reply")
    assert calls == ["prompt", "reply"]

    traj = type(
        "T", (), {"prompt": "p", "response": "rr", "raw_reward": 1.0, "metadata": {}}
    )
    item = TrajectoryDataset([traj], embed=model._embed_text)[0]
    assert item["response_embedding"].tolist() == [2.0] * 8


def test_hash_fallback_matches_configured_width():
    model = NeuralRewardModel(
        embedding_dim=256, hidden_dim=8, suppress_smoke_encoder_warning=True
    )
    assert model._embed_text("x").shape == (256,)
    model.predict_reward("a", "b")  # used to raise a shape mismatch


def test_large_factory_predicts_without_shape_errors():
    reward_fn = create_large_neural_reward_function()
    assert reward_fn.model._embed_text("x").shape == (256,)

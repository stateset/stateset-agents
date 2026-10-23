"""The enhanced DPOTrainer must train on response-token log-probs."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.enhanced.advanced_rl_algorithms import (  # noqa: E402
    DPOConfig,
    DPOTrainer,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:stateset_agents.core.enhanced.DPOTrainer is deprecated:DeprecationWarning"
)


class _ByteTokenizer:
    def __call__(self, text, add_special_tokens=True, **_):
        ids = list(text.encode("utf-8"))
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(256, 16)
        self.head = torch.nn.Linear(16, 256)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def _trainer():
    agent = SimpleNamespace(model=_TinyLM(), tokenizer=_ByteTokenizer())
    return DPOTrainer(agent, DPOConfig(learning_rate=0.05, beta=1.0), device="cpu")


def test_response_log_prob_scores_only_response_tokens():
    trainer = _trainer()
    ids, start = trainer._encode_pair("ab", "c")
    model = trainer.agent.model
    full = trainer._response_log_prob(model, ids, start)
    logp = torch.log_softmax(model(input_ids=ids).logits, dim=-1)[0]
    # response tokens are " " and "c" at positions start, start + 1
    expected = sum(logp[k - 1, ids[0, k]] for k in range(start, ids.shape[1]))
    assert torch.allclose(full, expected)


@pytest.mark.asyncio
async def test_train_step_prefers_chosen_over_rejected():
    trainer = _trainer()
    pair = {"prompt": "Q:", "chosen": "yes", "rejected": "no"}
    model = trainer.agent.model

    def margin():
        with torch.no_grad():
            c = trainer._response_log_prob(model, *trainer._encode_pair("Q:", "yes"))
            r = trainer._response_log_prob(model, *trainer._encode_pair("Q:", "no"))
        return float(c - r)

    before = margin()
    for _ in range(5):
        metrics = await trainer.train_step([pair])
    assert metrics["dpo_loss"] < 0.6931  # below log 2 once preferences are learned
    assert margin() > before


def test_dpo_trainer_is_deprecated_in_favour_of_preference_trainer():
    agent = SimpleNamespace(model=_TinyLM(), tokenizer=_ByteTokenizer())
    with pytest.warns(
        DeprecationWarning,
        match=r"removed in v0\.71\.0; use stateset_agents\.training\.preference\.PreferenceTrainer",
    ):
        DPOTrainer(agent, DPOConfig(), device="cpu")

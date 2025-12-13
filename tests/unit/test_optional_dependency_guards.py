import pytest


@pytest.mark.asyncio
async def test_get_http_pool_requires_aiohttp(monkeypatch):
    """`get_http_pool()` should fail fast with a clear error if aiohttp is missing."""
    from stateset_agents.core import async_pool

    monkeypatch.setattr(async_pool, "aiohttp", None)
    monkeypatch.setattr(async_pool, "_http_pool", None)

    with pytest.raises(ImportError) as excinfo:
        await async_pool.get_http_pool()

    assert "aiohttp is required" in str(excinfo.value)


def test_data_loader_split_fallback_without_sklearn(monkeypatch):
    """DataLoader should still split deterministically without sklearn installed."""
    from stateset_agents.core import data_processing

    monkeypatch.setattr(data_processing, "train_test_split", None)

    loader = data_processing.DataLoader(
        validation_split=0.2, stratify_by="task_type", random_seed=123
    )
    examples = [
        data_processing.ConversationExample(query=f"q{i}", task_type="a")
        for i in range(10)
    ] + [
        data_processing.ConversationExample(query=f"q{i}", task_type="b")
        for i in range(10, 20)
    ]

    train_1, eval_1 = loader.split_train_eval(examples, stratify=True)
    train_2, eval_2 = loader.split_train_eval(examples, stratify=True)

    assert len(train_1) + len(eval_1) == len(examples)
    assert len(eval_1) > 0

    # Deterministic output for a fixed seed.
    assert [e.query for e in train_1] == [e.query for e in train_2]
    assert [e.query for e in eval_1] == [e.query for e in eval_2]


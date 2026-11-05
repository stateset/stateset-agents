"""Frontier registrations never imply successful GPU training or rented capacity."""

import pytest

from stateset_agents.core.frontier_models import FRONTIER_MODELS
from stateset_agents.remote.model_catalog import get_model_support
from stateset_agents.remote.model_evidence import model_provider_evidence


@pytest.mark.parametrize("spec", FRONTIER_MODELS, ids=lambda spec: spec.key)
def test_frontier_models_are_cataloged_with_explicit_training_evidence(spec):
    support = get_model_support(spec.model_id)
    assert support is not None
    assert support.certification in {"configured", "smoke-tested"}
    rows = [
        row
        for row in model_provider_evidence()
        if row["model"] == spec.model_id and row["provider"] == "training"
    ]
    assert len(rows) == 1
    assert rows[0]["level"] == "unverified"
    assert rows[0]["outcome"] == ("blocked" if spec.blockers else "not-run")
    assert rows[0]["checked_at"] == spec.audit_date
    if spec.key not in {"glm5.3-flash", "qwen3.8-flash-next", "kimi-k3"}:
        assert support.runpod is None


def test_dated_deepseek_checkpoint_is_distinct_without_provider_promotion():
    original = get_model_support("deepseek-ai/DeepSeek-V4.1-Flash")
    dated = get_model_support("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    assert original is not None and dated is not None and original is not dated
    assert dated.model == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert dated.certification == "configured" and dated.runpod is None
    assert get_model_support(dated.model.upper()) is dated
    evidence = [row for row in model_provider_evidence() if row["model"] == dated.model]
    assert len(evidence) == 1
    assert evidence[0]["provider"] == "training"
    assert evidence[0]["level"] == "unverified"
    assert evidence[0]["outcome"] == "blocked"
    assert evidence[0]["checked_at"] == "2026-10-17"

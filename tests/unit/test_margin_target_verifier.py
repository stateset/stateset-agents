"""A forged target certificate must fail the independent verifier."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from benchmarks.weight_space.verify_margin_target import check_design  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    minimum_margin_target,
)


@pytest.mark.parametrize(
    "tamper", [None, "multipliers", "nonbinding_competitor", "diagnostic"]
)
def test_certificate_recomputes_witness_and_full_vocabulary(tamper):
    head = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
    head.weight.data.copy_(torch.eye(4))
    residual = torch.tensor([0.0, 4.0, 2.0, 1.0], dtype=torch.float64)
    result = minimum_margin_target(head.weight, residual, 0, margin=1)
    layer = torch.nn.Linear(1, 4, bias=False, dtype=torch.float64)
    layer.weight.data.copy_(result.delta[:, None])
    model = SimpleNamespace(
        get_output_embeddings=lambda: head,
        get_submodule=lambda _: layer,
        model=SimpleNamespace(norm=SimpleNamespace(weight=torch.ones(4))),
    )
    protocol = {"module": "layer", "target": ["prompt", "answer"]}
    factors = {
        "residual": residual,
        "margin_1.change": result.delta,
        "margin_1.multipliers": result.multipliers,
    }
    base = {
        "key": torch.ones(1, 1, dtype=torch.float64),
        "baseline_output": torch.zeros(1, 4, dtype=torch.float64),
    }
    trial = {
        "arm": "margin_1",
        # Evidence is read from JSON by the verifier, so indices are a list.
        "target_design": {
            **asdict(result.diagnostics),
            "token_ids": list(result.token_ids),
        },
    }
    if tamper == "multipliers":
        factors["margin_1.multipliers"] = result.multipliers + 0.5
    elif tamper == "nonbinding_competitor":
        assert 3 not in result.token_ids
        head.weight.data[3].fill_(100)
    elif tamper == "diagnostic":
        trial["target_design"]["update_norm"] *= 2
    if tamper is None:
        check_design(model, lambda _: [0], protocol, factors, base, trial, 1, False)
    else:
        with pytest.raises(ValueError):
            check_design(model, lambda _: [0], protocol, factors, base, trial, 1, False)

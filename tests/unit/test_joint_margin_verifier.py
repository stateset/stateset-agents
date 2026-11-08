"""Independent Woodbury certificate replay and deliberate corruption checks."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from benchmarks.weight_space.verify_joint_margin import check_design  # noqa: E402
from stateset_agents.experimental.weight_space import joint_margin_edit  # noqa: E402


@pytest.mark.parametrize("coverage", ["narrow", "soft", "hard"])
@pytest.mark.parametrize("tamper", [None, "witness", "objective", "deployed"])
def test_independent_joint_certificate_replay(coverage, tamper):
    g = torch.Generator().manual_seed(42)
    original = torch.randn(3, 8, generator=g) / 10
    keys = torch.randn(2, 8, generator=g)
    residuals = torch.randn(2, 3, generator=g)
    head = torch.nn.Linear(3, 4, bias=False)
    head.weight.data.copy_(torch.randn(4, 3, generator=g))
    p, h, extra, post = [torch.randn(1, 8, generator=g) for _ in range(4)]
    local = h if coverage == "narrow" else torch.cat((h, extra, post))
    protected = p if coverage != "hard" else torch.cat((p, local))
    solved = joint_margin_edit(
        original,
        keys,
        residuals,
        head.weight,
        [0, 0],
        output_scale=torch.ones(3),
        protect=protected,
        locality=local,
        locality_weight=100,
        margin=8,
        preservation_atol=0.0005,
        margin_atol=0.002,
    )
    layer = torch.nn.Linear(8, 3, bias=False)
    layer.weight.data.copy_(original + solved.delta.dense())
    model = SimpleNamespace(
        lm_head=head,
        get_submodule=lambda _: layer,
        model=SimpleNamespace(norm=SimpleNamespace(weight=torch.ones(3))),
    )
    protocol = {
        "module": "layer",
        "target": ["prompt", "answer"],
        "locality_weight": 100,
        "margin": 8,
        "joint_solver": {"preservation_atol": 0.0005},
    }
    factors = {
        "keys": keys,
        "residuals": residuals,
        "extra_calibration": extra,
        "post_answer": post,
        "joint_four.ideal_A": solved.ideal_A,
        "joint_four.ideal_B": solved.ideal_B,
        "joint_four.multipliers": solved.multipliers,
    }
    base = {"calibration": h, "protected": p}
    trial = {
        "arm": "joint_four",
        "design": json.loads(
            json.dumps(
                {
                    **asdict(solved.diagnostics),
                    "context_ids": solved.context_ids,
                    "token_ids": solved.token_ids,
                }
            )
        ),
    }
    if tamper == "witness":
        factors["joint_four.multipliers"] = solved.multipliers + 0.1
    elif tamper == "objective":
        trial["design"]["primal_objective"] += 1
    elif tamper == "deployed":
        layer.weight.data.add_(1)
    if tamper is None:
        check_design(
            model, lambda _: [0], protocol, factors, base, trial, 2, coverage, original
        )
    else:
        with pytest.raises(ValueError):
            check_design(
                model,
                lambda _: [0],
                protocol,
                factors,
                base,
                trial,
                2,
                coverage,
                original,
            )

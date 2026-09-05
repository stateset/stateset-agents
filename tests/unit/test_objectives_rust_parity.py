"""The group-advantage accelerator (Rust kernel or its NumPy fallback) must
agree with the ``group_norm`` estimator in ``objectives``."""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from stateset_agents.core import rust_accelerator  # noqa: E402
from stateset_agents.training import objectives as O  # noqa: E402


@pytest.mark.parametrize(
    "rewards",
    [
        [0.1, 0.9, 0.4, 0.4, 0.7],
        [1.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],  # constant group -> zeros, never NaN
    ],
)
def test_accelerator_group_advantages_match_group_norm(rewards):
    got = rust_accelerator.compute_group_advantages(np.asarray([rewards]), "mean", True)
    want = O.compute_advantages(
        torch.tensor(rewards),
        torch.zeros(len(rewards), dtype=torch.long),
        O.OBJECTIVES["grpo"],
    )
    torch.testing.assert_close(
        torch.tensor(np.asarray(got).reshape(-1), dtype=torch.float32),
        want,
        atol=1e-6,
        rtol=0,
    )

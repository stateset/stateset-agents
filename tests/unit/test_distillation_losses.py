"""Distillation losses: definitions, limits, masking and the estimators."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.distillation.losses import (  # noqa: E402
    NonFiniteLossError,
    VocabularyMismatchError,
    divergence_from_logprobs,
    generalized_jsd_loss,
    per_example_means,
    require_finite,
    sampled_reverse_kl,
    token_divergence,
    topk_divergence,
    topk_logprobs,
)


def _logits(seed: int, rows: int = 5, vocab: int = 11) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(rows, vocab, generator=g, dtype=torch.float64) * 2,
        torch.randn(rows, vocab, generator=g, dtype=torch.float64) * 2,
    )


def _manual_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(p_logits, -1)
    return (
        p * (torch.log_softmax(p_logits, -1) - torch.log_softmax(q_logits, -1))
    ).sum(-1)


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 1.0])
def test_identical_distributions_have_zero_divergence(beta: float) -> None:
    student, _ = _logits(0)
    value = token_divergence(student, student.clone(), beta=beta)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-12)


def test_endpoints_are_forward_and_reverse_kl() -> None:
    student, teacher = _logits(1)
    forward = token_divergence(student, teacher, beta=0.0)
    reverse = token_divergence(student, teacher, beta=1.0)
    assert torch.allclose(forward, _manual_kl(teacher, student))
    assert torch.allclose(reverse, _manual_kl(student, teacher))
    assert not torch.allclose(forward, reverse)


def test_generalized_jsd_definition_and_symmetry() -> None:
    student, teacher = _logits(2)
    beta = 0.3
    s, t = torch.softmax(student, -1), torch.softmax(teacher, -1)
    mixture = beta * t + (1 - beta) * s

    def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (p * (p.log() - q.log())).sum(-1)

    expected = beta * kl(t, mixture) + (1 - beta) * kl(s, mixture)
    assert torch.allclose(token_divergence(student, teacher, beta=beta), expected)
    # JSD_0.5 is symmetric and bounded by ln 2.
    half = token_divergence(student, teacher, beta=0.5)
    assert torch.allclose(half, token_divergence(teacher, student, beta=0.5))
    assert bool((half >= 0).all()) and bool((half <= math.log(2) + 1e-12).all())


def test_temperature_divides_both_logits() -> None:
    student, teacher = _logits(3)
    assert torch.allclose(
        token_divergence(student, teacher, beta=0.5, temperature=2.0),
        token_divergence(student / 2.0, teacher / 2.0, beta=0.5),
    )


def test_vocabulary_mismatch_is_rejected_not_truncated() -> None:
    student, teacher = _logits(4)
    with pytest.raises(VocabularyMismatchError, match="shared"):
        token_divergence(student, teacher[:, :-1], beta=0.0)
    with pytest.raises(VocabularyMismatchError):
        divergence_from_logprobs(student, teacher[:, :-2], beta=0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beta": 1.5}, "beta"),
        ({"beta": -0.1}, "beta"),
        ({"beta": 0.5, "temperature": 0.0}, "temperature"),
        ({"beta": 0.5, "temperature": math.inf}, "temperature"),
    ],
)
def test_invalid_settings_raise(kwargs: dict, message: str) -> None:
    student, teacher = _logits(5)
    with pytest.raises(ValueError, match=message):
        token_divergence(student, teacher, **kwargs)


def test_masked_teacher_tokens_give_finite_forward_kl_and_gradients() -> None:
    student, teacher = _logits(6)
    teacher[:, :3] = -math.inf  # the teacher can never emit these tokens
    student = student.clone().requires_grad_(True)
    value = token_divergence(student, teacher, beta=0.0).sum()
    value.backward()
    assert math.isfinite(float(value.detach()))
    assert bool(torch.isfinite(student.grad).all())
    # Reverse KL against a zero-probability teacher token is genuinely infinite.
    with pytest.raises(NonFiniteLossError):
        require_finite(token_divergence(student, teacher, beta=1.0), "loss")


def test_generalized_jsd_loss_reductions() -> None:
    g = torch.Generator().manual_seed(7)
    student = torch.randn(2, 4, 6, generator=g)
    teacher = torch.randn(2, 4, 6, generator=g)
    labels = torch.tensor([[-100, 1, 2, -100], [3, 4, -100, -100]])
    per_token = generalized_jsd_loss(
        student, teacher, labels, beta=0.5, reduction="none"
    )
    assert per_token.shape == (2, 4)
    assert float(per_token[0, 0]) == 0.0 and float(per_token[1, 3]) == 0.0
    total = generalized_jsd_loss(student, teacher, labels, beta=0.5, reduction="sum")
    mean = generalized_jsd_loss(student, teacher, labels, beta=0.5)
    assert float(total) == pytest.approx(float(per_token.sum()))
    assert float(mean) == pytest.approx(float(total) / 4)
    with pytest.raises(ValueError, match="reduction"):
        generalized_jsd_loss(student, teacher, labels, reduction="mean")


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_full_topk_cache_reproduces_the_full_divergence(beta: float) -> None:
    student, teacher = _logits(8)
    ids, logps = topk_logprobs(teacher, k=teacher.shape[-1])
    full = token_divergence(student, teacher, beta=beta)
    cached = topk_divergence(torch.log_softmax(student, -1), ids, logps, beta=beta)
    assert torch.allclose(cached, full, atol=1e-6)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_topk_divergence_lower_bounds_the_full_divergence(beta: float) -> None:
    student, teacher = _logits(9, rows=20)
    full = token_divergence(student, teacher, beta=beta)
    ids, logps = topk_logprobs(teacher, k=3)
    cached = topk_divergence(torch.log_softmax(student, -1), ids, logps, beta=beta)
    assert bool((cached <= full + 1e-9).all()) and bool((cached >= -1e-9).all())


def test_topk_ids_outside_the_student_vocabulary_raise() -> None:
    student, teacher = _logits(10)
    ids, logps = topk_logprobs(teacher, k=2)
    with pytest.raises(VocabularyMismatchError):
        topk_divergence(torch.log_softmax(student[:, :4], -1), ids + 5, logps, beta=0)


def test_sampled_estimator_gradient_is_the_reverse_kl_gradient_in_expectation() -> None:
    """Σ_x S(x) ∇ surrogate(x) == ∇ KL(S ‖ T), enumerated exactly."""
    student, teacher = _logits(11, rows=3, vocab=7)
    student = student.clone().requires_grad_(True)
    exact = token_divergence(student, teacher, beta=1.0).sum()
    (expected,) = torch.autograd.grad(exact, student)
    probabilities = torch.softmax(student, -1).detach()
    total = torch.zeros_like(student)
    for token in range(student.shape[-1]):
        targets = torch.full((student.shape[0], 1), token)
        student_logps = torch.log_softmax(student, -1).gather(-1, targets).squeeze(-1)
        teacher_logps = torch.log_softmax(teacher, -1).gather(-1, targets).squeeze(-1)
        surrogate, _ = sampled_reverse_kl(student_logps, teacher_logps)
        (grad,) = torch.autograd.grad(
            (probabilities[:, token] * surrogate).sum(), student
        )
        total += grad
    assert torch.allclose(total, expected, atol=1e-10)


def test_sampled_estimate_is_unbiased_for_the_reverse_kl() -> None:
    student, teacher = _logits(12, rows=2, vocab=5)
    s_logps, t_logps = torch.log_softmax(student, -1), torch.log_softmax(teacher, -1)
    _, estimate = sampled_reverse_kl(s_logps, t_logps)
    expectation = (s_logps.exp() * estimate).sum(-1)
    assert torch.allclose(expectation, token_divergence(student, teacher, beta=1.0))


def test_per_example_means() -> None:
    values = torch.tensor([1.0, 3.0, 5.0, 2.0])
    rows = torch.tensor([0, 0, 0, 2])
    counts = torch.tensor([3, 0, 1])
    assert per_example_means(values, rows, counts).tolist() == [3.0, 0.0, 2.0]

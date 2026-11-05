"""Torch-free configuration for knowledge distillation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..preference.config import _choice, _float, _int, _LoopConfig, validate_lora

__all__ = [
    "DIVERGENCES",
    "ON_POLICY_LOSSES",
    "VOCAB_ALIGNMENTS",
    "DistillationConfig",
]

#: Token-level divergences: forward KL(teacher || student) (β = 0), reverse
#: KL(student || teacher) (β = 1), or GKD's generalized JSD with ``jsd_beta``.
DIVERGENCES = ("forward_kl", "reverse_kl", "jsd")
#: On-policy objectives: the full-vocabulary divergence at every position of
#: the student's sample (GKD), or the sampled-token reverse-KL estimator.
ON_POLICY_LOSSES = ("full", "sampled")
#: ``strict``: teacher and student vocabularies must be identical.
#: ``shared``: align by token string and distil over the shared tokens.
VOCAB_ALIGNMENTS = ("strict", "shared")


@dataclass
class DistillationConfig(_LoopConfig):
    """Settings for :class:`~stateset_agents.training.distillation.DistillationTrainer`.

    ``divergence`` (with ``jsd_beta`` for ``jsd``) and ``temperature`` (both
    distributions are ``softmax(logits / temperature)``) define the per-token
    loss on response tokens. ``lmbda`` is GKD's λ: the probability that a
    micro-batch trains on completions the student samples itself (with
    ``max_new_tokens``, ``sampling_temperature``, ``top_p``) instead of the
    dataset's; ``on_policy_loss`` picks the objective on those samples.
    ``teacher_top_k`` trains offline from a cache of the teacher's top-k
    log-probs (built once, so the teacher can be released). ``vocab_alignment``
    says how teacher and student vocabularies must relate.
    """

    learning_rate: float = 1e-5
    divergence: str = "forward_kl"
    jsd_beta: float = 0.5
    temperature: float = 1.0
    lmbda: float = 0.0
    on_policy_loss: str = "full"
    max_new_tokens: int = 128
    sampling_temperature: float = 1.0
    top_p: float = 1.0
    teacher_top_k: int | None = None
    vocab_alignment: str = "strict"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = field(default=None)

    @property
    def beta(self) -> float:
        """GKD's β: 0 is forward KL, 1 reverse KL, otherwise generalized JSD."""
        if self.divergence == "forward_kl":
            return 0.0
        if self.divergence == "reverse_kl":
            return 1.0
        return float(self.jsd_beta)

    @property
    def on_policy(self) -> bool:
        return float(self.lmbda) > 0.0

    def validate(self) -> None:
        super().validate()
        _choice(self.divergence, "divergence", DIVERGENCES)
        _float(self.jsd_beta, "jsd_beta", low=0.0, below=1.0)
        if self.divergence == "jsd" and float(self.jsd_beta) <= 0.0:
            raise ValueError(
                "jsd_beta must be in (0, 1); use divergence='forward_kl' for 0"
            )
        _float(self.temperature, "temperature", positive=True)
        _float(self.lmbda, "lmbda", low=0.0)
        if float(self.lmbda) > 1.0:
            raise ValueError(f"lmbda must be in [0, 1], got {self.lmbda!r}")
        _choice(self.on_policy_loss, "on_policy_loss", ON_POLICY_LOSSES)
        _int(self.max_new_tokens, "max_new_tokens", minimum=1)
        _float(self.sampling_temperature, "sampling_temperature", positive=True)
        _float(self.top_p, "top_p", positive=True)
        if float(self.top_p) > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p!r}")
        self._validate_on_policy()
        if self.teacher_top_k is not None:
            _int(self.teacher_top_k, "teacher_top_k", minimum=1)
            if self.on_policy:
                raise ValueError(
                    "teacher_top_k caches the teacher on dataset completions; "
                    "on-policy samples (lmbda > 0) need the live teacher"
                )
        _choice(self.vocab_alignment, "vocab_alignment", VOCAB_ALIGNMENTS)
        validate_lora(self)

    def _validate_on_policy(self) -> None:
        if self.on_policy_loss != "sampled":
            return
        if not self.on_policy:
            raise ValueError("on_policy_loss='sampled' needs lmbda > 0")
        # The estimator is unbiased only for samples drawn from the
        # distribution it differentiates.
        if float(self.sampling_temperature) != float(self.temperature):
            raise ValueError(
                "on_policy_loss='sampled' needs sampling_temperature equal to "
                "temperature (the estimator assumes x ~ softmax(z / temperature))"
            )
        if float(self.top_p) != 1.0:
            raise ValueError("on_policy_loss='sampled' needs top_p = 1.0")

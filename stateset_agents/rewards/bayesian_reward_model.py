"""
Bayesian Reward Models with Uncertainty Quantification

Implements reward models with principled uncertainty estimation using
Bayesian neural networks, Monte Carlo Dropout, and ensemble methods.
"""

import logging
import math
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError:
    torch = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    Adam = cast(Any, None)

from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.reward_moments import RewardMoments
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for Bayesian reward models. "
            "Install: pip install stateset-agents[training]"
        )


_sample_dropout = ContextVar("bayesian_reward_sample_dropout", default=False)


class _RewardDropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            input, self.p, self.training or _sample_dropout.get(), self.inplace
        )


def _wide(value: torch.Tensor) -> torch.Tensor:
    """Evaluate scalar reward statistics in double precision."""
    if value.device.type == "mps":
        value = value.cpu()
    return value.double()


def _on_device(value: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return (
        value.to(original.device, torch.float32)
        if original.device.type == "mps"
        else value
    )


def _population_moments(draws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and population variance of equally likely draws along the first axis."""
    # Sampling is an inference operation. The same integer-grid accumulator
    # used for rollout rewards preserves even very small means after cancellation.
    rows = (
        draws.detach().to(device="cpu", dtype=torch.float64).reshape(len(draws), -1).T
    )
    means, variances = [], []
    for row in rows.tolist():
        if not all(map(math.isfinite, row)):
            raise ValueError("nonfinite sampled reward")
        moments = RewardMoments(row)
        means.append(moments.mean)
        try:
            variance = moments.variance_numerator / moments.denominator**2
        except OverflowError:
            variance = math.inf
        variances.append(variance)
    shape = draws.shape[1:]
    dtype = torch.float32 if draws.device.type == "mps" else torch.float64
    mean = torch.tensor(means, device=draws.device, dtype=dtype).reshape(shape)
    var = torch.tensor(variances, device=draws.device, dtype=dtype).reshape(shape)
    return mean, var


def _predictive_moments(
    means: torch.Tensor, log_vars: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, epistemic = _population_moments(means)
    # Average the component variances before converting back from log space.
    log_average = torch.logsumexp(_wide(log_vars), dim=0) - math.log(len(log_vars))
    return mean, epistemic, _on_device(log_average.exp(), log_vars)


def _paired_gaussian_nll(
    mean: torch.Tensor, log_var: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Average Gaussian NLL, omitting the constant log(2π)/2, for paired rows."""
    if mean.ndim != 2 or mean.shape[1] != 1 or log_var.shape != mean.shape:
        raise ValueError("reward heads must each produce one scalar per row")
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if targets.shape != mean.shape or mean.numel() == 0:
        raise ValueError("provide one reward label per row in a nonempty batch")
    log_var = _wide(log_var)
    residual = _wide(mean) - _wide(targets).to(log_var.device)
    # Use the ordinary quadratic near the fitted mean. For extreme precisions
    # the standardization factor alone can overflow or underflow.
    extreme = log_var.abs() > 1400
    ordinary = (
        residual.masked_fill(extreme, 0)
        * (-0.5 * log_var.masked_fill(extreme, 0)).exp()
    )
    use_log = extreme & (residual != 0)
    magnitude = residual.abs().masked_fill(~use_log, 1)
    tail = (
        (2 * magnitude.log() - log_var.masked_fill(~use_log, 0))
        .exp()
        .masked_fill(~use_log, 0)
    )
    return (0.5 * log_var + 0.5 * ordinary.square() + 0.5 * tail).mean()


def _normal_kl(
    mean: torch.Tensor, log_sigma: torch.Tensor, prior_std: float
) -> torch.Tensor:
    """KL from a diagonal Gaussian to a zero-mean isotropic Gaussian."""
    from stateset_agents.training.rl_losses import k3_term

    d = 2 * (_wide(log_sigma) - math.log(prior_std))
    normalized = _wide(mean) / prior_std
    return (0.5 * normalized.square() + 0.5 * k3_term(d)).sum()


@dataclass
class BayesianRewardConfig:
    """Configuration for Bayesian reward models"""

    # Model architecture
    hidden_size: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1

    # Bayesian inference
    num_samples: int = 20  # MC samples for uncertainty
    epistemic_weight: float = 1.0  # Query priority for disagreement
    aleatoric_weight: float = 1.0  # Query priority for conditional noise

    # Ensemble settings
    num_ensemble: int = 5  # Number of models in ensemble
    use_ensemble: bool = True

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    prior_std: float = 1.0  # Prior std for Bayesian weights

    # Uncertainty thresholds
    high_uncertainty_threshold: float = 0.3
    low_confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Require nonnegative finite weights for uncertainty-based label queries."""
        if not all(
            math.isfinite(w) and w >= 0
            for w in (self.epistemic_weight, self.aleatoric_weight)
        ):
            raise ValueError("uncertainty weights must be finite and nonnegative")


class MCDropoutRewardModel(nn.Module):
    """
    Reward model with Monte Carlo Dropout for uncertainty estimation.

    MC Dropout approximates Bayesian inference by keeping dropout active
    during inference and sampling multiple forward passes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Build network with dropout
        layers = []
        in_dim = input_dim

        for _i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(_RewardDropout(dropout_rate))
            in_dim = hidden_size

        # Output: mean and log variance for aleatoric uncertainty
        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, 1)
        self.log_var_head = nn.Linear(in_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty.

        Returns:
            mean: Predicted reward mean
            log_var: Log variance (aleatoric uncertainty)
        """
        features = self.feature_net(x)
        mean = self.mean_head(features)
        log_var = self.log_var_head(features)
        return mean, log_var

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with epistemic and aleatoric uncertainty.

        Args:
            x: Input tensor
            num_samples: Number of MC samples

        Returns:
            mean: Predicted mean
            epistemic_uncertainty: Model uncertainty (variance across samples)
            aleatoric_uncertainty: Data uncertainty (predicted variance)
        """
        return _predictive_moments(*self._draw_components(x, num_samples))

    @torch.no_grad()
    def _draw_components(
        self, x: torch.Tensor, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        token = _sample_dropout.set(True)
        try:
            draws = [self(x) for _ in range(num_samples)]
        finally:
            _sample_dropout.reset(token)
        means, log_vars = zip(*draws, strict=True)
        return torch.stack(means), torch.stack(log_vars)


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Uses variational inference to learn distributions over weights.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if not math.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")
        self.prior_std = prior_std

        # Weight parameters (mean and log std)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize means
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)

        # Initialize log stds (small values)
        nn.init.constant_(self.weight_log_sigma, -5)
        nn.init.constant_(self.bias_log_sigma, -5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling"""
        # Sample weights from posterior
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)

        # Sample bias
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.

        KL(q(w) || p(w)) where q is learned posterior, p is Gaussian prior
        """
        value = _normal_kl(self.weight_mu, self.weight_log_sigma, self.prior_std)
        value = value + _normal_kl(self.bias_mu, self.bias_log_sigma, self.prior_std)
        return _on_device(value, self.weight_mu)


class VariationalBayesianRewardModel(nn.Module):
    """
    Full Bayesian neural network for reward modeling.

    Uses variational inference with weight uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        prior_std: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Build Bayesian layers
        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(BayesianLinear(in_dim, hidden_size, prior_std))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        self.feature_net = nn.ModuleList(layers)
        self.output_layer = BayesianLinear(in_dim, 1, prior_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling"""
        for layer in self.feature_net:
            x = layer(x)
        return self.output_layer(x)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all Bayesian layers"""
        kl = self.output_layer.kl_divergence()
        for layer in self.feature_net:
            if isinstance(layer, BayesianLinear):
                kl = kl + layer.kl_divergence()
        return kl

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty by sampling from weight posterior.

        Returns:
            mean: Predicted mean
            uncertainty: Predictive uncertainty
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        with torch.no_grad():
            return _population_moments(
                torch.stack([self(x) for _ in range(num_samples)])
            )


def _fit_gaussian_heads(
    models: list[MCDropoutRewardModel],
    optimizers: list[Adam],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    learning_rate: float | None,
) -> float:
    if learning_rate is not None and (
        learning_rate < 0 or not math.isfinite(learning_rate)
    ):
        raise ValueError("learning_rate must be finite and nonnegative")
    losses: list[float] = []
    token = _sample_dropout.set(True)
    try:
        for model, optimizer in zip(models, optimizers, strict=True):
            optimizer.zero_grad()
            mean, log_var = model(inputs)
            loss = _paired_gaussian_nll(mean, log_var, targets)
            loss.backward()
            if not loss.isfinite() or any(
                not p.grad.isfinite().all()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise ValueError("nonfinite Gaussian reward loss or gradient")
            losses.append(loss.item())
    finally:
        _sample_dropout.reset(token)

    # Apply updates only after every member has produced a valid gradient.
    for optimizer in optimizers:
        rates = [group["lr"] for group in optimizer.param_groups]
        try:
            if learning_rate is not None:
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            optimizer.step()
        finally:
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate
    return math.fsum(loss / len(losses) for loss in losses)


class EnsembleRewardModel:
    """
    Ensemble of reward models for uncertainty quantification.

    Uses model disagreement as a measure of uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        num_models: int = 5,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        _require_torch()

        if num_models < 1:
            raise ValueError("num_models must be positive")
        self.num_models = num_models
        self.device = device

        # Create ensemble
        self.models = []
        for _ in range(num_models):
            model = MCDropoutRewardModel(
                input_dim, hidden_size, num_layers, dropout_rate
            ).to(device)
            self.models.append(model)

        self.optimizers = [
            Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for model in self.models
        ]

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        learning_rate: float | None = None,
    ) -> dict[str, float]:
        """Train all models in ensemble"""
        loss = _fit_gaussian_heads(
            self.models, self.optimizers, inputs, targets, learning_rate
        )
        return {"ensemble_loss": loss}

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble uncertainty.

        Returns:
            mean: Ensemble mean prediction
            epistemic: Population variance of means across members and their dropout draws
            aleatoric: Data uncertainty (average predicted variance)
        """
        # All members contribute the same number of equally weighted draws.
        # Flattening preserves variation both within and between members.
        draws = [model._draw_components(x, num_mc_samples) for model in self.models]
        means, log_vars = zip(*draws, strict=True)
        return _predictive_moments(torch.cat(means), torch.cat(log_vars))


class BayesianRewardFunction(RewardFunction):
    """
    Reward function with Bayesian uncertainty quantification.

    Provides reward predictions with approximate predictive intervals and
    uncertainty decomposition.
    """

    def __init__(
        self,
        input_dim: int = 768,  # Typical embedding size
        config: BayesianRewardConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.input_dim = input_dim
        self.config = config or BayesianRewardConfig()
        self.device = device

        self._optimizer: Adam | None = None
        self.model: EnsembleRewardModel | MCDropoutRewardModel
        if self.config.use_ensemble:
            self.model = EnsembleRewardModel(
                input_dim,
                self.config.num_ensemble,
                self.config.hidden_size,
                self.config.num_layers,
                self.config.dropout_rate,
                device,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.model = MCDropoutRewardModel(
                input_dim,
                self.config.hidden_size,
                self.config.num_layers,
                self.config.dropout_rate,
            ).to(device)

        self.calibration_data: list[tuple[float, float]] = []  # (predicted, actual)

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        """
        Compute reward with uncertainty quantification.

        Returns a reward with a moment-matched normal interval and uncertainty breakdown.
        """
        # Extract features from conversation
        # This is a placeholder - in practice, use embeddings from LLM
        features = self._extract_features(turns)
        head = (
            self.model.models[0]
            if isinstance(self.model, EnsembleRewardModel)
            else self.model
        )
        param = next(head.parameters())
        features_tensor = torch.as_tensor(
            features, device=param.device, dtype=param.dtype
        ).unsqueeze(0)

        # Get prediction with uncertainty
        mean, epistemic, aleatoric = self._predict_with_uncertainty(features_tensor)

        mean = float(mean.item())
        epistemic = float(epistemic.item())
        aleatoric = float(aleatoric.item())

        # Total uncertainty
        total_uncertainty = epistemic + aleatoric

        # Moment-matched normal approximation to a predictive interval.
        std = float(np.sqrt(total_uncertainty))
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        # Determine if uncertainty is high
        ew, aw = self.config.epistemic_weight, self.config.aleatoric_weight
        weighted_uncertainty = (ew * epistemic if ew else 0.0) + (
            aw * aleatoric if aw else 0.0
        )
        is_high_uncertainty = (
            weighted_uncertainty > self.config.high_uncertainty_threshold
        )

        return RewardResult(
            score=mean,
            breakdown={
                "epistemic_uncertainty": epistemic,
                "aleatoric_uncertainty": aleatoric,
                "total_uncertainty": total_uncertainty,
                "weighted_uncertainty": weighted_uncertainty,
                "confidence_interval_lower": ci_lower,
                "confidence_interval_upper": ci_upper,
            },
            metadata={
                "high_uncertainty": is_high_uncertainty,
                "interval_method": "moment_matched_normal_approximation",
                "uncertainty_source": (
                    "epistemic" if epistemic > aleatoric else "aleatoric"
                ),
                "confidence": 1.0 / (1.0 + total_uncertainty),  # Normalized confidence
            },
            explanation=self._generate_explanation(
                mean, epistemic, aleatoric, is_high_uncertainty
            ),
        )

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        learning_rate: float | None = None,
    ) -> dict[str, float]:
        """Fit one Gaussian reward per labeled example, preserving optimizer history."""
        if isinstance(self.model, EnsembleRewardModel):
            return self.model.train_step(inputs, targets, learning_rate=learning_rate)
        if self._optimizer is None:
            self._optimizer = Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        loss = _fit_gaussian_heads(
            [self.model], [self._optimizer], inputs, targets, learning_rate
        )
        return {"update_loss": loss}

    def _predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prediction with uncertainty"""
        return self.model.predict_with_uncertainty(x, self.config.num_samples)

    def _extract_features(self, turns: list[ConversationTurn]) -> NDArray[np.float32]:
        """
        Extract features from conversation turns.

        In practice, use LLM embeddings. Here we use simple features.
        """
        if not turns:
            return np.zeros(self.input_dim, dtype=np.float32)

        # Simple feature extraction (placeholder)
        features = []

        # Average length
        avg_length = float(np.mean([len(str(turn.content or "")) for turn in turns]))
        features.append(avg_length / 1000.0)  # Normalize

        # Number of turns
        features.append(float(len(turns)) / 20.0)

        # Sentiment (placeholder - use actual sentiment in practice)
        features.append(0.5)

        # Pad to input_dim
        feature_array = np.array(features[: self.input_dim], dtype=np.float32)
        padded: NDArray[np.float32] = np.zeros(self.input_dim, dtype=np.float32)
        padded[: len(feature_array)] = feature_array

        return padded

    def _generate_explanation(
        self,
        mean: float,
        epistemic: float,
        aleatoric: float,
        high_uncertainty: bool,
    ) -> str:
        """Generate human-readable explanation"""
        if high_uncertainty:
            if epistemic > aleatoric:
                return (
                    f"Reward: {mean:.2f} (HIGH UNCERTAINTY). "
                    "Model is uncertain - consider collecting more training data."
                )
            else:
                return (
                    f"Reward: {mean:.2f} (HIGH UNCERTAINTY). "
                    "Inherent task ambiguity - response may vary."
                )
        else:
            return f"Reward: {mean:.2f} (confident prediction)"

    def calibrate(
        self, predicted: list[float], actual: list[float]
    ) -> dict[str, float]:
        """
        Accumulate paired outcomes and report MAE and RMSE of the predicted means.
        """
        if len(predicted) != len(actual):
            raise ValueError("predictions and outcomes must have the same length")
        observations = list(zip(predicted, actual, strict=True))
        if not all(math.isfinite(p) and math.isfinite(a) for p, a in observations):
            raise ValueError("reward observations must be finite")
        if not observations and not self.calibration_data:
            raise ValueError("at least one reward observation is required")
        self.calibration_data.extend(observations)
        errors = [abs(p - a) for p, a in self.calibration_data]
        scale = 1.0
        if any(math.isinf(error) for error in errors):
            # A single difference can exceed binary64 even when the final MAE
            # and RMS fit. Halve before subtraction in this case.
            errors = [abs(0.5 * p - 0.5 * a) for p, a in self.calibration_data]
            scale = 2.0
        stats = RewardMoments(errors)
        return {
            "mae": scale * stats.mean,
            "rmse": scale * math.hypot(stats.mean, stats.deviation()),
            "num_samples": float(stats.count),
        }

    async def compute_turn_reward(
        self,
        turn: ConversationTurn,
        context: dict[str, Any] | None = None,
        conversation_history: list[ConversationTurn] | None = None,
    ) -> RewardResult:
        """Compute reward for single turn"""
        history = conversation_history or []
        return await self.compute_reward(history + [turn], context)


class ActiveLearningSelector:
    """
    Select samples for active learning based on uncertainty.

    High uncertainty samples are most valuable for improving the model.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        diversity_weight: float = 0.5,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.diversity_weight = diversity_weight
        self.selected_samples: list[np.ndarray] = []

    @staticmethod
    def _uncertainty(result: RewardResult) -> float:
        return float(
            result.breakdown.get(
                "weighted_uncertainty", result.breakdown.get("total_uncertainty", 0.0)
            )
        )

    def should_query_label(
        self,
        reward_result: RewardResult,
    ) -> bool:
        """Determine if sample should be labeled by human"""
        return self._uncertainty(reward_result) > self.uncertainty_threshold

    def select_batch_for_labeling(
        self,
        candidates: list[tuple[np.ndarray, RewardResult]],
        batch_size: int = 10,
    ) -> list[int]:
        """
        Select batch of samples for human labeling.

        Balances uncertainty and diversity.
        """
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if batch_size == 0:
            return []
        uncertainties = np.array([self._uncertainty(r) for _, r in candidates])

        # Compute diversity scores (distance from already selected)
        diversity_scores = np.ones(len(candidates))
        if self.selected_samples:
            for i, (features, _) in enumerate(candidates):
                min_dist = min(
                    np.linalg.norm(features - selected)
                    for selected in self.selected_samples
                )
                diversity_scores[i] = min_dist

        # Combined score
        scores = (
            1 - self.diversity_weight
        ) * uncertainties + self.diversity_weight * diversity_scores

        # Select top batch_size
        selected_indices = [
            int(idx) for idx in np.argsort(scores)[-batch_size:].tolist()
        ]

        # Update selected samples (cap memory at 1000 recent samples)
        for idx in selected_indices:
            self.selected_samples.append(candidates[idx][0])
        if len(self.selected_samples) > 1000:
            self.selected_samples = self.selected_samples[-1000:]

        return selected_indices

    def update_with_labels(
        self,
        reward_model: Any,
        features: list[np.ndarray],
        labels: list[float],
        num_update_steps: int = 10,
        learning_rate: float | None = None,
    ) -> dict[str, float]:
        """Update the reward model with human-labeled samples.

        Closes the active learning feedback loop: after humans label the
        selected high-uncertainty samples, this method fine-tunes the reward
        model on the new labels to reduce uncertainty on similar samples.

        Args:
            reward_model: The BayesianRewardFunction (or any model with
                a ``train_step(features, labels)`` method).
            features: Feature vectors for labeled samples.
            labels: Human-provided reward labels.
            num_update_steps: Gradient steps on the labeled data.
            learning_rate: Optional override for the model's configured learning rate.

        Returns:
            Metrics dict with keys: update_loss, uncertainty_before,
            uncertainty_after, improvement.
        """
        try:
            import torch
        except ImportError:
            return {"error": 1.0}

        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        if num_update_steps < 0:
            raise ValueError("num_update_steps must be nonnegative")
        if not features:
            return {"update_loss": 0.0, "num_labels": 0}

        trainable = getattr(reward_model, "model", reward_model)
        module = (
            trainable.models[0]
            if isinstance(trainable, EnsembleRewardModel)
            else trainable
        )
        param = (
            next(module.parameters(), None) if isinstance(module, nn.Module) else None
        )
        device = (
            param.device
            if param is not None
            else getattr(reward_model, "device", "cpu")
        )
        dtype = param.dtype if param is not None else torch.float32
        X = torch.as_tensor(np.asarray(features), device=device, dtype=dtype)
        y = torch.as_tensor(
            labels,
            device=device,
            dtype=(
                torch.float32 if torch.device(device).type == "mps" else torch.float64
            ),
        )

        predict = getattr(reward_model, "_predict_with_uncertainty", None)
        if predict is None:
            predict = getattr(reward_model, "predict_with_uncertainty", None)

        def mean_epistemic() -> float:
            with torch.no_grad():
                return float(predict(X)[1].mean().item()) if predict else 0.0

        uncertainty_before = mean_epistemic()

        # Fine-tune the reward model
        total_loss = 0.0
        if isinstance(reward_model, (BayesianRewardFunction, EnsembleRewardModel)):
            for _ in range(num_update_steps):
                result = reward_model.train_step(X, y, learning_rate=learning_rate)
                total_loss += result.get(
                    "ensemble_loss", result.get("update_loss", 0.0)
                )
        elif hasattr(reward_model, "train_step"):
            for _ in range(num_update_steps):
                result = reward_model.train_step(X, y.to(dtype))
                total_loss += (
                    result.get("update_loss", result.get("loss", 0.0))
                    if isinstance(result, dict)
                    else float(result)
                )
        elif isinstance(trainable, nn.Module):
            optimizer = Adam(
                trainable.parameters(),
                lr=1e-4 if learning_rate is None else learning_rate,
            )
            if isinstance(trainable, MCDropoutRewardModel):
                for _ in range(num_update_steps):
                    total_loss += _fit_gaussian_heads(
                        [trainable], [optimizer], X, y, None
                    )
            else:
                trainable.train()
                for _ in range(num_update_steps):
                    optimizer.zero_grad()
                    predictions = trainable(X).reshape_as(y)
                    loss = F.mse_loss(predictions, y.to(predictions))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                trainable.eval()
        else:
            raise TypeError(
                "reward model must provide a train_step or a trainable module"
            )

        uncertainty_after = mean_epistemic()
        improvement = max(0.0, uncertainty_before - uncertainty_after)

        metrics = {
            "update_loss": total_loss / max(num_update_steps, 1),
            "num_labels": float(len(labels)),
            "uncertainty_before": uncertainty_before,
            "uncertainty_after": uncertainty_after,
            "improvement": improvement,
        }

        logger.info(
            "Active learning update: %d labels, loss=%.4f, "
            "uncertainty %.4f -> %.4f (Δ=%.4f)",
            len(labels),
            metrics["update_loss"],
            uncertainty_before,
            uncertainty_after,
            improvement,
        )

        return metrics

    def get_stats(self) -> dict[str, Any]:
        """Get active learning statistics."""
        return {
            "num_selected": len(self.selected_samples),
            "uncertainty_threshold": self.uncertainty_threshold,
            "diversity_weight": self.diversity_weight,
        }

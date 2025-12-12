"""
Bayesian Reward Models with Uncertainty Quantification

Implements reward models with principled uncertainty estimation using
Bayesian neural networks, Monte Carlo Dropout, and ensemble methods.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    from torch.optim import Adam
except ImportError:
    torch = None
    nn = None
    F = None
    Normal = None
    Adam = None

from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)


def _require_torch():
    """Ensure torch is available"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for Bayesian reward models. "
            "Install: pip install stateset-agents[training]"
        )


@dataclass
class BayesianRewardConfig:
    """Configuration for Bayesian reward models"""

    # Model architecture
    hidden_size: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1

    # Bayesian inference
    num_samples: int = 20  # MC samples for uncertainty
    epistemic_weight: float = 1.0  # Weight for epistemic uncertainty
    aleatoric_weight: float = 1.0  # Weight for aleatoric uncertainty

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

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_size

        # Output: mean and log variance for aleatoric uncertainty
        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_var_head = nn.Linear(hidden_size, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.train()  # Keep dropout active

        means = []
        log_vars = []

        with torch.no_grad():
            for _ in range(num_samples):
                mean, log_var = self.forward(x)
                means.append(mean)
                log_vars.append(log_var)

        means = torch.stack(means)
        log_vars = torch.stack(log_vars)

        # Epistemic uncertainty: variance across MC samples
        pred_mean = means.mean(dim=0)
        epistemic_uncertainty = means.var(dim=0)

        # Aleatoric uncertainty: average predicted variance
        aleatoric_uncertainty = torch.exp(log_vars).mean(dim=0)

        return pred_mean, epistemic_uncertainty, aleatoric_uncertainty


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Uses variational inference to learn distributions over weights.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
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
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)

        # KL for weights
        kl_weight = (
            torch.log(self.prior_std / weight_sigma)
            + (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * self.prior_std ** 2)
            - 0.5
        ).sum()

        # KL for bias
        kl_bias = (
            torch.log(self.prior_std / bias_sigma)
            + (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * self.prior_std ** 2)
            - 0.5
        ).sum()

        return kl_weight + kl_bias


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
        self.output_layer = BayesianLinear(hidden_size, 1, prior_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling"""
        for layer in self.feature_net:
            x = layer(x)
        return self.output_layer(x)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all Bayesian layers"""
        kl = 0
        for layer in self.feature_net:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        return kl

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty by sampling from weight posterior.

        Returns:
            mean: Predicted mean
            uncertainty: Predictive uncertainty
        """
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        pred_mean = predictions.mean(dim=0)
        pred_uncertainty = predictions.var(dim=0)

        return pred_mean, pred_uncertainty


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
    ):
        _require_torch()

        self.num_models = num_models
        self.device = device

        # Create ensemble
        self.models = []
        for _ in range(num_models):
            model = MCDropoutRewardModel(input_dim, hidden_size, num_layers, dropout_rate).to(device)
            self.models.append(model)

        self.optimizers = [Adam(model.parameters(), lr=1e-4) for model in self.models]

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Train all models in ensemble"""
        losses = []

        for model, optimizer in zip(self.models, self.optimizers):
            optimizer.zero_grad()

            # Forward pass
            mean, log_var = model(inputs)

            # Negative log likelihood loss
            precision = torch.exp(-log_var)
            loss = 0.5 * (precision * (targets - mean) ** 2 + log_var).mean()

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return {"ensemble_loss": np.mean(losses)}

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_mc_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble uncertainty.

        Returns:
            mean: Ensemble mean prediction
            epistemic: Model uncertainty (ensemble disagreement)
            aleatoric: Data uncertainty (average predicted variance)
        """
        all_means = []
        all_aleatoric = []

        for model in self.models:
            mean, epistemic, aleatoric = model.predict_with_uncertainty(x, num_mc_samples)
            all_means.append(mean)
            all_aleatoric.append(aleatoric)

        all_means = torch.stack(all_means)
        all_aleatoric = torch.stack(all_aleatoric)

        # Ensemble mean
        ensemble_mean = all_means.mean(dim=0)

        # Epistemic: disagreement between models
        epistemic_uncertainty = all_means.var(dim=0)

        # Aleatoric: average data uncertainty
        aleatoric_uncertainty = all_aleatoric.mean(dim=0)

        return ensemble_mean, epistemic_uncertainty, aleatoric_uncertainty


class BayesianRewardFunction(RewardFunction):
    """
    Reward function with Bayesian uncertainty quantification.

    Provides reward predictions with confidence intervals and
    uncertainty decomposition.
    """

    def __init__(
        self,
        input_dim: int = 768,  # Typical embedding size
        config: Optional[BayesianRewardConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        _require_torch()

        self.config = config or BayesianRewardConfig()
        self.device = device

        if self.config.use_ensemble:
            self.model = EnsembleRewardModel(
                input_dim,
                self.config.num_ensemble,
                self.config.hidden_size,
                self.config.num_layers,
                self.config.dropout_rate,
                device,
            )
        else:
            self.model = MCDropoutRewardModel(
                input_dim,
                self.config.hidden_size,
                self.config.num_layers,
                self.config.dropout_rate,
            ).to(device)

        self.calibration_data: List[Tuple[float, float]] = []  # (predicted, actual)

    async def compute_reward(self, turns: List[ConversationTurn]) -> RewardResult:
        """
        Compute reward with uncertainty quantification.

        Returns reward with confidence intervals and uncertainty breakdown.
        """
        # Extract features from conversation
        # This is a placeholder - in practice, use embeddings from LLM
        features = self._extract_features(turns)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Get prediction with uncertainty
        mean, epistemic, aleatoric = self._predict_with_uncertainty(features_tensor)

        mean = mean.item()
        epistemic = epistemic.item()
        aleatoric = aleatoric.item()

        # Total uncertainty
        total_uncertainty = epistemic + aleatoric

        # Confidence interval (95%)
        std = np.sqrt(total_uncertainty)
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std

        # Determine if uncertainty is high
        is_high_uncertainty = total_uncertainty > self.config.high_uncertainty_threshold

        return RewardResult(
            score=mean,
            breakdown={
                "epistemic_uncertainty": epistemic,
                "aleatoric_uncertainty": aleatoric,
                "total_uncertainty": total_uncertainty,
                "confidence_interval_lower": ci_lower,
                "confidence_interval_upper": ci_upper,
            },
            metadata={
                "high_uncertainty": is_high_uncertainty,
                "uncertainty_source": "epistemic" if epistemic > aleatoric else "aleatoric",
                "confidence": 1.0 / (1.0 + total_uncertainty),  # Normalized confidence
            },
            explanation=self._generate_explanation(mean, epistemic, aleatoric, is_high_uncertainty),
        )

    def _predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prediction with uncertainty"""
        if self.config.use_ensemble:
            return self.model.predict_with_uncertainty(x, self.config.num_samples)
        else:
            return self.model.predict_with_uncertainty(x, self.config.num_samples)

    def _extract_features(self, turns: List[ConversationTurn]) -> np.ndarray:
        """
        Extract features from conversation turns.

        In practice, use LLM embeddings. Here we use simple features.
        """
        if not turns:
            return np.zeros(768)

        # Simple feature extraction (placeholder)
        features = []

        # Average length
        avg_length = np.mean([len(turn.content) for turn in turns])
        features.append(avg_length / 1000.0)  # Normalize

        # Number of turns
        features.append(len(turns) / 20.0)

        # Sentiment (placeholder - use actual sentiment in practice)
        features.append(0.5)

        # Pad to input_dim
        features = np.array(features)
        padded = np.zeros(768)
        padded[: len(features)] = features

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

    def calibrate(self, predicted: List[float], actual: List[float]) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates using actual outcomes.

        Returns calibration metrics.
        """
        self.calibration_data.extend(zip(predicted, actual))

        # Compute calibration error
        predictions = np.array([p for p, _ in self.calibration_data])
        actuals = np.array([a for _, a in self.calibration_data])

        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        return {"mae": mae, "rmse": rmse, "num_samples": len(self.calibration_data)}

    async def compute_turn_reward(
        self,
        turn: ConversationTurn,
        context: Dict[str, Any],
        history: List[ConversationTurn],
    ) -> RewardResult:
        """Compute reward for single turn"""
        return await self.compute_reward(history + [turn])


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
        self.selected_samples: List[np.ndarray] = []

    def should_query_label(
        self,
        reward_result: RewardResult,
    ) -> bool:
        """Determine if sample should be labeled by human"""
        total_uncertainty = reward_result.breakdown.get("total_uncertainty", 0)
        return total_uncertainty > self.uncertainty_threshold

    def select_batch_for_labeling(
        self,
        candidates: List[Tuple[np.ndarray, RewardResult]],
        batch_size: int = 10,
    ) -> List[int]:
        """
        Select batch of samples for human labeling.

        Balances uncertainty and diversity.
        """
        uncertainties = np.array([r.breakdown["total_uncertainty"] for _, r in candidates])

        # Compute diversity scores (distance from already selected)
        diversity_scores = np.ones(len(candidates))
        if self.selected_samples:
            for i, (features, _) in enumerate(candidates):
                min_dist = min(
                    np.linalg.norm(features - selected) for selected in self.selected_samples
                )
                diversity_scores[i] = min_dist

        # Combined score
        scores = (1 - self.diversity_weight) * uncertainties + self.diversity_weight * diversity_scores

        # Select top batch_size
        selected_indices = np.argsort(scores)[-batch_size:].tolist()

        # Update selected samples
        for idx in selected_indices:
            self.selected_samples.append(candidates[idx][0])

        return selected_indices

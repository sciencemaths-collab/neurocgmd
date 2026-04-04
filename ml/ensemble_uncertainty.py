"""Ensemble-based uncertainty quantification via bootstrap disagreement."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import Random

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, StateId, coerce_scalar
from memory.replay_buffer import ReplayItem
from memory.trace_store import TraceRecord
from ml.live_features import LiveFeatureVector
from ml.residual_model import ResidualPrediction, ResidualTarget
from ml.uncertainty_model import UncertaintyEstimate


class _BootstrapEnsembleMember:
    """Simple online linear model trained via SGD on bootstrap-sampled observations."""

    __slots__ = ("weights", "bias")

    def __init__(self, feature_dim: int) -> None:
        self.weights: list[float] = [0.0] * feature_dim
        self.bias: float = 0.0

    def predict(self, features: list[float]) -> float:
        """Return a scalar energy-delta prediction from the feature vector."""
        total = self.bias
        for w, x in zip(self.weights, features):
            total += w * x
        return total

    def update(
        self,
        features: list[float],
        target: float,
        learning_rate: float,
        sample_weight: float,
    ) -> None:
        """One step of online SGD: w -= lr * sample_weight * (predicted - target) * x."""
        prediction = self.predict(features)
        error = prediction - target
        scaled_error = learning_rate * sample_weight * error
        for i in range(len(self.weights)):
            self.weights[i] -= scaled_error * features[i]
        self.bias -= scaled_error


@dataclass(slots=True)
class EnsembleUncertaintyModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Ensemble disagreement uncertainty estimator using bootstrap-sampled online linear models.

    Replaces the hand-crafted ``HeuristicUncertaintyModel`` with a proper
    ensemble approach: multiple lightweight linear members are each trained on
    bootstrap-sampled observations and their prediction disagreement (standard
    deviation) quantifies epistemic uncertainty.
    """

    ensemble_size: int = 5
    feature_dim: int = 8
    learning_rate: float = 0.01
    bootstrap_ratio: float = 0.8
    trigger_threshold: float = 0.5
    calibration_scale: float = 1.0
    random_seed: int = 42
    name: str = "ensemble_uncertainty_model"
    classification: str = "[proposed novel]"
    _members: list[_BootstrapEnsembleMember] = field(default_factory=list, init=False, repr=False)
    _rng: Random = field(default_factory=Random, init=False, repr=False)
    _observation_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.random_seed)
        self._members = [
            _BootstrapEnsembleMember(self.feature_dim) for _ in range(self.ensemble_size)
        ]

    # -- ArchitecturalComponent / DocumentedComponent / ValidatableComponent --

    def describe_role(self) -> str:
        return (
            "Quantifies epistemic uncertainty for residual predictions through "
            "ensemble disagreement among bootstrap-sampled online linear models."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ml/residual_model.py",
            "ml/uncertainty_model.py",
            "memory/replay_buffer.py",
            "memory/trace_store.py",
            "ml/live_features.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/online_ml_residual_learning.md",
            "docs/sections/section_11_online_ml_residual_learning.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.ensemble_size < 2:
            issues.append("ensemble_size must be at least 2.")
        if self.feature_dim < 1:
            issues.append("feature_dim must be at least 1.")
        if self.learning_rate <= 0.0:
            issues.append("learning_rate must be strictly positive.")
        if not (0.0 < self.bootstrap_ratio <= 1.0):
            issues.append("bootstrap_ratio must be in (0, 1].")
        if not (0.0 <= self.trigger_threshold <= 1.0):
            issues.append("trigger_threshold must lie in [0, 1].")
        if self.calibration_scale < 0.0:
            issues.append("calibration_scale must be non-negative.")
        return tuple(issues)

    # -- Feature extraction ------------------------------------------------

    def _extract_features(
        self,
        prediction: ResidualPrediction,
        *,
        live_features: LiveFeatureVector | None = None,
        replay_item: ReplayItem | None = None,
    ) -> list[float]:
        """Build a fixed-length feature vector from prediction and optional signals.

        Feature layout (indices may be zero-padded to ``feature_dim``):
            0  predicted_energy_delta
            1  confidence
            2  number of force deltas
            3  total force magnitude (sum of |delta_force| across particles)
            4  chemistry_mean_pair_score   (from live_features)
            5  structure_rmsd_normalized   (from live_features)
            6  shadow_force_regression     (from live_features)
            7  replay score               (from replay_item)
        """
        force_magnitude = 0.0
        for fd in prediction.force_deltas:
            for component in fd.delta_force:
                force_magnitude += abs(component)

        raw: list[float] = [
            prediction.predicted_energy_delta,
            prediction.confidence,
            float(len(prediction.force_deltas)),
            force_magnitude,
        ]

        if live_features is not None:
            raw.append(live_features.value("chemistry_mean_pair_score", 0.0))
            raw.append(live_features.value("structure_rmsd_normalized", 0.0))
            raw.append(live_features.value("shadow_force_regression", 0.0))
        else:
            raw.extend([0.0, 0.0, 0.0])

        if replay_item is not None:
            raw.append(replay_item.score)
        else:
            raw.append(0.0)

        # Pad or truncate to feature_dim.
        while len(raw) < self.feature_dim:
            raw.append(0.0)
        return raw[: self.feature_dim]

    # -- Estimation --------------------------------------------------------

    def estimate(
        self,
        prediction: ResidualPrediction,
        *,
        trace_record: TraceRecord | None = None,
        replay_item: ReplayItem | None = None,
        live_features: LiveFeatureVector | None = None,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty via ensemble prediction disagreement."""
        features = self._extract_features(
            prediction, live_features=live_features, replay_item=replay_item,
        )

        # Collect predictions from every ensemble member.
        predictions = [member.predict(features) for member in self._members]

        # -- Energy uncertainty: standard deviation of ensemble predictions --
        mean_pred = sum(predictions) / len(predictions)
        variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
        energy_disagreement = sqrt(variance)
        energy_uncertainty = min(1.0, self.calibration_scale * energy_disagreement)

        # -- Force uncertainty: variance in force-related features ----------
        # Use the force magnitude feature (index 3) as the basis and scale by
        # confidence deficit.  When force magnitude is high and confidence low
        # the ensemble has less information, producing higher force uncertainty.
        force_mag = features[3] if len(features) > 3 else 0.0
        confidence = features[1] if len(features) > 1 else 0.0
        force_uncertainty = min(1.0, self.calibration_scale * force_mag * (1.0 - confidence))

        total_uncertainty = max(energy_uncertainty, force_uncertainty)
        trigger_qcloud = total_uncertainty >= self.trigger_threshold

        return UncertaintyEstimate(
            state_id=prediction.state_id,
            energy_uncertainty=energy_uncertainty,
            force_uncertainty=force_uncertainty,
            total_uncertainty=total_uncertainty,
            trigger_qcloud=trigger_qcloud,
            metadata=FrozenMetadata(
                {
                    "ensemble_size": self.ensemble_size,
                    "energy_disagreement": energy_disagreement,
                    "ensemble_mean": mean_pred,
                    "observation_count": self._observation_count,
                    "prediction_confidence": prediction.confidence,
                }
            ),
        )

    # -- Online training ---------------------------------------------------

    def observe(self, target: ResidualTarget, *, sample_weight: float = 1.0) -> None:
        """Update ensemble members from an observed residual target.

        Each member sees the observation with probability ``bootstrap_ratio``,
        creating diversity through bootstrap sampling.  The SGD update rule is:

            w -= learning_rate * sample_weight * (predicted - target) * features
        """
        sample_weight = coerce_scalar(sample_weight, "sample_weight")
        if sample_weight <= 0.0:
            raise ContractValidationError("sample_weight must be strictly positive.")

        # Build a synthetic prediction to extract features (confidence 0 since
        # this is a ground-truth observation, force_deltas from the target).
        synthetic_prediction = ResidualPrediction(
            state_id=target.state_id,
            predicted_energy_delta=target.energy_delta,
            force_deltas=target.force_deltas,
            confidence=0.0,
            metadata=FrozenMetadata({"source": "observe"}),
        )
        features = self._extract_features(synthetic_prediction)

        target_value = target.energy_delta

        for member in self._members:
            if self._rng.random() < self.bootstrap_ratio:
                member.update(features, target_value, self.learning_rate, sample_weight)

        self._observation_count += 1

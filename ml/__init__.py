"""Machine learning models, uncertainty estimation, and online training hooks."""

from ml.differentiable_hooks import (
    DifferentiableParameter,
    FiniteDifferenceGradientEstimator,
    ParameterGradient,
    ParameterSnapshot,
    ScalableResidualDifferentiableAdapter,
)
from ml.live_features import LiveFeatureEncoder, LiveFeatureVector
from ml.neural_residual_model import NeuralResidualModel
from ml.online_trainer import OnlineTrainingReport, ReplayDrivenOnlineTrainer, ReplayTrainingExample
from ml.residual_model import (
    ResidualAugmentedForceEvaluator,
    ResidualMemoryModel,
    ResidualModel,
    ResidualPrediction,
    ResidualTarget,
    StateAwareResidualModel,
)
from ml.scalable_residual import ScalableResidualModel
from ml.ensemble_uncertainty import EnsembleUncertaintyModel
from ml.uncertainty_model import HeuristicUncertaintyModel, UncertaintyEstimate, UncertaintyModel

__all__ = [
    "EnsembleUncertaintyModel",
    "DifferentiableParameter",
    "FiniteDifferenceGradientEstimator",
    "HeuristicUncertaintyModel",
    "LiveFeatureEncoder",
    "LiveFeatureVector",
    "NeuralResidualModel",
    "OnlineTrainingReport",
    "ParameterGradient",
    "ParameterSnapshot",
    "ReplayDrivenOnlineTrainer",
    "ReplayTrainingExample",
    "ResidualAugmentedForceEvaluator",
    "ResidualMemoryModel",
    "ResidualModel",
    "ResidualPrediction",
    "ResidualTarget",
    "ScalableResidualModel",
    "ScalableResidualDifferentiableAdapter",
    "StateAwareResidualModel",
    "UncertaintyEstimate",
    "UncertaintyModel",
]

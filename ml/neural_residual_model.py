"""Lightweight MLP residual correction model with online SGD training.

Replaces the replay-only ResidualMemoryModel with a real neural network that
learns energy (and force-scale) corrections via backpropagation, using only
Python standard-library math.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, coerce_scalar
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta
from ml.residual_model import ResidualTarget, ResidualPrediction


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

_ActivationKind = Literal["relu", "tanh", "linear"]


def _tanh(x: float) -> float:
    """Numerically safe tanh."""
    return math.tanh(max(-20.0, min(20.0, x)))


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _activate(x: float, kind: _ActivationKind) -> float:
    if kind == "relu":
        return _relu(x)
    if kind == "tanh":
        return _tanh(x)
    return x


def _activate_deriv(activated: float, pre_act: float, kind: _ActivationKind) -> float:
    """Derivative of activation given the *activated* output and *pre-activation* input."""
    if kind == "relu":
        return 1.0 if pre_act > 0.0 else 0.0
    if kind == "tanh":
        return 1.0 - activated * activated
    return 1.0


# ---------------------------------------------------------------------------
# Online Welford normalizer
# ---------------------------------------------------------------------------

class _RunningNormalizer:
    """Online mean/variance tracker using Welford's algorithm."""

    __slots__ = ("_count", "_mean", "_m2", "_dim")

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._count: int = 0
        self._mean: list[float] = [0.0] * dim
        self._m2: list[float] = [0.0] * dim

    def update(self, x: list[float]) -> None:
        self._count += 1
        for i in range(self._dim):
            delta = x[i] - self._mean[i]
            self._mean[i] += delta / self._count
            delta2 = x[i] - self._mean[i]
            self._m2[i] += delta * delta2

    def normalize(self, x: list[float]) -> list[float]:
        if self._count < 2:
            return list(x)
        result: list[float] = []
        for i in range(self._dim):
            var = self._m2[i] / self._count
            std = math.sqrt(var) if var > 1e-12 else 1.0
            result.append((x[i] - self._mean[i]) / std)
        return result


# ---------------------------------------------------------------------------
# NeuralLayer
# ---------------------------------------------------------------------------

class NeuralLayer:
    """Single dense layer: y = activation(W^T x + b)."""

    __slots__ = (
        "input_dim",
        "output_dim",
        "activation",
        "weights",
        "biases",
        # momentum buffers
        "_weight_velocity",
        "_bias_velocity",
        # cached forward pass state for backprop
        "_last_input",
        "_last_pre_activation",
        "_last_output",
    )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: _ActivationKind = "linear",
        *,
        rng: random.Random,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation: _ActivationKind = activation

        # Xavier initialization: uniform in [-limit, limit]
        limit = math.sqrt(6.0 / (input_dim + output_dim))
        self.weights: list[list[float]] = [
            [rng.uniform(-limit, limit) for _ in range(output_dim)]
            for _ in range(input_dim)
        ]
        self.biases: list[float] = [0.0] * output_dim

        # SGD momentum buffers
        self._weight_velocity: list[list[float]] = [
            [0.0] * output_dim for _ in range(input_dim)
        ]
        self._bias_velocity: list[float] = [0.0] * output_dim

        # Forward cache (set during forward())
        self._last_input: list[float] = []
        self._last_pre_activation: list[float] = []
        self._last_output: list[float] = []

    # ---- forward -----------------------------------------------------------

    def forward(self, x: list[float]) -> list[float]:
        """Compute y = activation(W^T x + b) and cache intermediates."""
        self._last_input = x
        pre_act: list[float] = list(self.biases)  # copy
        for i in range(self.input_dim):
            xi = x[i]
            row = self.weights[i]
            for j in range(self.output_dim):
                pre_act[j] += xi * row[j]
        output = [_activate(z, self.activation) for z in pre_act]
        self._last_pre_activation = pre_act
        self._last_output = output
        return output

    # ---- backward ----------------------------------------------------------

    def backward(self, grad_output: list[float]) -> tuple[list[float], list[list[float]], list[float]]:
        """Back-propagate through this layer.

        Parameters
        ----------
        grad_output:
            dL/dy for each output unit.

        Returns
        -------
        grad_input:
            dL/dx for each input unit (to propagate further).
        grad_weights:
            dL/dW  (input_dim x output_dim).
        grad_biases:
            dL/db  (output_dim).
        """
        # dL/d(pre_act) = dL/dy * dy/d(pre_act)
        grad_pre: list[float] = [
            grad_output[j] * _activate_deriv(self._last_output[j], self._last_pre_activation[j], self.activation)
            for j in range(self.output_dim)
        ]

        # dL/db = grad_pre
        grad_biases = list(grad_pre)

        # dL/dW[i][j] = x[i] * grad_pre[j]
        grad_weights: list[list[float]] = [
            [self._last_input[i] * grad_pre[j] for j in range(self.output_dim)]
            for i in range(self.input_dim)
        ]

        # dL/dx[i] = sum_j W[i][j] * grad_pre[j]
        grad_input: list[float] = [0.0] * self.input_dim
        for i in range(self.input_dim):
            s = 0.0
            row = self.weights[i]
            for j in range(self.output_dim):
                s += row[j] * grad_pre[j]
            grad_input[i] = s

        return grad_input, grad_weights, grad_biases

    # ---- parameter update --------------------------------------------------

    def update(
        self,
        grad_weights: list[list[float]],
        grad_biases: list[float],
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        """SGD step with momentum and L2 weight decay."""
        for i in range(self.input_dim):
            wrow = self.weights[i]
            vrow = self._weight_velocity[i]
            grow = grad_weights[i]
            for j in range(self.output_dim):
                g = grow[j] + weight_decay * wrow[j]
                vrow[j] = momentum * vrow[j] + g
                wrow[j] -= lr * vrow[j]
        for j in range(self.output_dim):
            g = grad_biases[j]
            self._bias_velocity[j] = momentum * self._bias_velocity[j] + g
            self.biases[j] -= lr * self._bias_velocity[j]


# ---------------------------------------------------------------------------
# NeuralResidualModel
# ---------------------------------------------------------------------------

_FEATURE_DIM = 6  # total_energy, mean_force_mag, max_force_mag, particle_count, step_norm, force_variance


@dataclass(slots=True)
class NeuralResidualModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Lightweight MLP that learns residual energy/force corrections via online SGD."""

    hidden_sizes: tuple[int, ...] = (32, 16)
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    confidence_growth_rate: float = 0.01
    random_seed: int = 42
    name: str = "neural_residual_model"
    classification: str = "[proposed novel]"

    # private, non-init fields
    _rng: random.Random = field(init=False, repr=False)
    _layers: list[NeuralLayer] = field(init=False, repr=False)
    _force_head: NeuralLayer = field(init=False, repr=False)
    _normalizer: _RunningNormalizer = field(init=False, repr=False)
    _trained_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.random_seed)
        self._normalizer = _RunningNormalizer(_FEATURE_DIM)
        self._build_network()

    # ---- network construction ---------------------------------------------

    def _build_network(self) -> None:
        layers: list[NeuralLayer] = []
        prev_dim = _FEATURE_DIM
        for h in self.hidden_sizes:
            layers.append(NeuralLayer(prev_dim, h, activation="tanh", rng=self._rng))
            prev_dim = h
        # energy output head (1 scalar, linear)
        layers.append(NeuralLayer(prev_dim, 1, activation="linear", rng=self._rng))
        self._layers = layers

        # force-scale head: from last hidden → 1 (linear scale factor)
        last_hidden = self.hidden_sizes[-1] if self.hidden_sizes else _FEATURE_DIM
        self._force_head = NeuralLayer(last_hidden, 1, activation="linear", rng=self._rng)

    # ---- ArchitecturalComponent / DocumentedComponent / Validatable --------

    def describe_role(self) -> str:
        return (
            "Neural-network residual correction model that learns additive energy "
            "and force-scale corrections via online stochastic gradient descent."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ml/residual_model.py",
            "physics/forces/composite.py",
            "qcloud/cloud_state.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/online_ml_residual_learning.md",
            "docs/sections/section_11_online_ml_residual_learning.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.learning_rate <= 0.0:
            issues.append("learning_rate must be strictly positive.")
        if not (0.0 <= self.momentum < 1.0):
            issues.append("momentum must be in [0, 1).")
        if self.weight_decay < 0.0:
            issues.append("weight_decay must be non-negative.")
        if self.max_grad_norm <= 0.0:
            issues.append("max_grad_norm must be strictly positive.")
        if self.confidence_growth_rate <= 0.0:
            issues.append("confidence_growth_rate must be strictly positive.")
        if not self.hidden_sizes:
            issues.append("hidden_sizes must contain at least one layer size.")
        return tuple(issues)

    # ---- feature extraction -----------------------------------------------

    def _extract_features(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
    ) -> list[float]:
        """Build a fixed-size feature vector from the simulation state and force evaluation."""
        forces = base_evaluation.forces
        n = len(forces)

        # force magnitudes
        magnitudes: list[float] = []
        for fx, fy, fz in forces:
            magnitudes.append(math.sqrt(fx * fx + fy * fy + fz * fz))

        if n > 0:
            mean_mag = sum(magnitudes) / n
            max_mag = max(magnitudes)
            variance = sum((m - mean_mag) ** 2 for m in magnitudes) / n
        else:
            mean_mag = 0.0
            max_mag = 0.0
            variance = 0.0

        total_energy = base_evaluation.potential_energy
        particle_count = float(state.particle_count)
        # Normalise step number to avoid large raw integers
        step_norm = float(state.step) / 1000.0

        raw = [total_energy, mean_mag, max_mag, particle_count, step_norm, variance]
        self._normalizer.update(raw)
        return self._normalizer.normalize(raw)

    # ---- forward pass (full network) --------------------------------------

    def _forward(self, features: list[float]) -> tuple[float, float, list[float]]:
        """Run features through the network.

        Returns
        -------
        energy_delta:
            Predicted scalar energy correction.
        force_scale:
            Per-particle multiplicative scale factor for base forces.
        last_hidden:
            Activations of the last hidden layer (for the force head).
        """
        x = features
        # hidden layers (all except last which is the energy output)
        for layer in self._layers[:-1]:
            x = layer.forward(x)
        last_hidden = list(x)

        # energy head
        energy_out = self._layers[-1].forward(x)
        energy_delta = energy_out[0]

        # force-scale head
        force_out = self._force_head.forward(last_hidden)
        force_scale = force_out[0]

        return energy_delta, force_scale, last_hidden

    # ---- gradient clipping ------------------------------------------------

    @staticmethod
    def _clip_gradients(
        all_gw: list[list[list[float]]],
        all_gb: list[list[float]],
        max_norm: float,
    ) -> None:
        """In-place global gradient clipping by L2 norm."""
        total_sq = 0.0
        for gw in all_gw:
            for row in gw:
                for v in row:
                    total_sq += v * v
        for gb in all_gb:
            for v in gb:
                total_sq += v * v
        grad_norm = math.sqrt(total_sq)
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            for gw in all_gw:
                for row in gw:
                    for k in range(len(row)):
                        row[k] *= scale
            for gb in all_gb:
                for k in range(len(gb)):
                    gb[k] *= scale

    # ---- predict -----------------------------------------------------------

    def predict(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
    ) -> ResidualPrediction:
        state_id = state.provenance.state_id
        if len(base_evaluation.forces) != state.particle_count:
            raise ContractValidationError(
                "base_evaluation.forces must match the SimulationState particle count."
            )

        features = self._extract_features(state, base_evaluation)
        energy_delta, force_scale, _ = self._forward(features)

        # Build per-particle force deltas using the predicted scale factor
        force_deltas: list[ParticleForceDelta] = []
        for idx, (fx, fy, fz) in enumerate(base_evaluation.forces):
            dx = fx * force_scale
            dy = fy * force_scale
            dz = fz * force_scale
            if abs(dx) > 1e-15 or abs(dy) > 1e-15 or abs(dz) > 1e-15:
                force_deltas.append(
                    ParticleForceDelta(particle_index=idx, delta_force=(dx, dy, dz))
                )

        confidence = min(1.0, self._trained_count * self.confidence_growth_rate)

        return ResidualPrediction(
            state_id=state_id,
            predicted_energy_delta=energy_delta,
            force_deltas=tuple(force_deltas),
            confidence=confidence,
            metadata=FrozenMetadata(
                {
                    "mode": "neural_mlp",
                    "trained_count": self._trained_count,
                    "force_scale": force_scale,
                    "base_potential_energy": base_evaluation.potential_energy,
                }
            ),
        )

    # ---- observe (online training) ----------------------------------------

    def observe(self, target: ResidualTarget, *, sample_weight: float = 1.0) -> None:
        """Train the network on a single observed residual target via backprop + SGD."""
        sample_weight = coerce_scalar(sample_weight, "sample_weight")
        if sample_weight <= 0.0:
            raise ContractValidationError("sample_weight must be strictly positive.")

        # We need a forward pass to compute the loss.  Since observe() receives
        # only a ResidualTarget (no SimulationState / ForceEvaluation), we
        # re-use the cached forward state from the most recent predict() call.
        # If _layers have no cached input (i.e. predict was never called), we
        # create a zero-feature vector and run forward to initialise caches.
        if not self._layers[0]._last_input:
            dummy = [0.0] * _FEATURE_DIM
            self._forward(dummy)

        # --- Energy loss: MSE ------------------------------------------------
        predicted_energy = self._layers[-1]._last_output[0]
        energy_error = predicted_energy - target.energy_delta
        loss_grad = 2.0 * energy_error * sample_weight  # dL/d(predicted_energy)

        # --- Backprop through energy head (last layer) -----------------------
        grad_output_energy = [loss_grad]
        n_layers = len(self._layers)

        all_grad_weights: list[list[list[float]]] = []
        all_grad_biases: list[list[float]] = []

        # Backprop energy head
        grad_input, gw, gb = self._layers[-1].backward(grad_output_energy)
        all_grad_weights.append(gw)
        all_grad_biases.append(gb)

        # Backprop hidden layers in reverse
        for k in range(n_layers - 2, -1, -1):
            grad_input, gw, gb = self._layers[k].backward(grad_input)
            all_grad_weights.append(gw)
            all_grad_biases.append(gb)

        # Reverse so index matches layer index
        all_grad_weights.reverse()
        all_grad_biases.reverse()

        # --- Force-scale head loss -------------------------------------------
        # If the target has force deltas we can compute a simple loss on the
        # force-scale head, but it shares only the last hidden layer's output.
        # For simplicity we train the force head using its own cached state.
        force_head_gw: list[list[float]] | None = None
        force_head_gb: list[float] | None = None
        if target.force_deltas and self._force_head._last_input:
            # Target scale: average ratio of target delta to base force magnitude
            # We approximate by comparing predicted scale to a pseudo-target of 0
            # (meaning "no further correction needed after observing the target").
            predicted_scale = self._force_head._last_output[0]
            # A simple target: drive scale toward the mean observed delta magnitude
            target_mags: list[float] = []
            for fd in target.force_deltas:
                dx, dy, dz = fd.delta_force
                target_mags.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            mean_target_mag = sum(target_mags) / len(target_mags) if target_mags else 0.0
            # Use sign of energy_delta to inform direction
            sign = 1.0 if target.energy_delta >= 0.0 else -1.0
            pseudo_target = sign * mean_target_mag * 0.01  # small scale factor
            force_error = predicted_scale - pseudo_target
            force_loss_grad = 2.0 * force_error * sample_weight * 0.1  # down-weight force loss
            _, force_head_gw, force_head_gb = self._force_head.backward([force_loss_grad])

        # --- Gradient clipping -----------------------------------------------
        clip_gw = list(all_grad_weights)
        clip_gb = list(all_grad_biases)
        if force_head_gw is not None:
            clip_gw.append(force_head_gw)
        if force_head_gb is not None:
            clip_gb.append(force_head_gb)
        self._clip_gradients(clip_gw, clip_gb, self.max_grad_norm)

        # --- SGD update -------------------------------------------------------
        for k, layer in enumerate(self._layers):
            layer.update(all_grad_weights[k], all_grad_biases[k], self.learning_rate, self.momentum, self.weight_decay)
        if force_head_gw is not None and force_head_gb is not None:
            self._force_head.update(force_head_gw, force_head_gb, self.learning_rate, self.momentum, self.weight_decay)

        self._trained_count += 1

    # ---- trained_state_count -----------------------------------------------

    def trained_state_count(self) -> int:
        return self._trained_count

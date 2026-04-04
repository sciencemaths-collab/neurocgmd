"""Scalable piece-local ML residual model using SPRING's decomposition.

[proposed novel]

The key insight is architectural rather than mystical: SPRING's
Piece+Spring split matches the execution pattern used by modern
equivariant/message-passing force fields.

- each particle is a local "piece" with its own encoded features
- each neighbor pair is a "spring" that transmits bounded messages
- corrections are predicted locally and summed globally

That gives us a residual model which:

- stays compatible with the existing additive residual contract
- scales with local neighbor count rather than dense global state size
- can be trained online from qcloud residual targets without replacing the
  coarse substrate

The current implementation is intentionally pure Python and bounded. It is not
claiming SchNet/NequIP/MACE fidelity. It is the repo-native architectural
upgrade that makes those styles of models legible inside this project's
Piece+Spring language.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, Vector3, VectorTuple, coerce_scalar
from ml.residual_model import ResidualPrediction, ResidualTarget
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta


def _tanh(x: float) -> float:
    return math.tanh(max(-20.0, min(20.0, x)))


def _tanh_deriv_from_output(y: float) -> float:
    return 1.0 - y * y


def _vec3_dist_sq(a: Vector3, b: Vector3) -> float:
    return (
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


def _vec3_dist(a: Vector3, b: Vector3) -> float:
    return math.sqrt(_vec3_dist_sq(a, b))


def _vec3_mag(v: Vector3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot(values: list[float], other: list[float]) -> float:
    return sum(left * right for left, right in zip(values, other))


def _xavier_init(rows: int, cols: int, rng: _random.Random) -> list[list[float]]:
    scale = math.sqrt(6.0 / max(rows + cols, 1))
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _zero_vec(length: int) -> list[float]:
    return [0.0] * length


def _gaussian_rbf(distance: float, centers: list[float], width: float) -> list[float]:
    return [math.exp(-width * (distance - center) ** 2) for center in centers]


@dataclass(slots=True)
class LocalEncoder:
    """Per-particle feature encoder."""

    hidden_dim: int = 16
    _weights: list[list[float]] = field(default=None, init=False, repr=False)
    _biases: list[float] = field(default=None, init=False, repr=False)

    def initialize(self, input_dim: int, rng: _random.Random) -> None:
        self._weights = _xavier_init(input_dim, self.hidden_dim, rng)
        self._biases = _zero_vec(self.hidden_dim)

    def encode(self, raw_features: list[float]) -> list[float]:
        output = list(self._biases)
        for feature_index, feature_value in enumerate(raw_features):
            weights = self._weights[feature_index]
            for hidden_index in range(self.hidden_dim):
                output[hidden_index] += weights[hidden_index] * feature_value
        return [_tanh(value) for value in output]


@dataclass(slots=True)
class SpringMessageLayer:
    """Neighbor message layer using distance-expanded radial features."""

    hidden_dim: int = 16
    rbf_centers: int = 8
    rbf_width: float = 10.0
    _centers: list[float] = field(default=None, init=False, repr=False)
    _weights: list[list[float]] = field(default=None, init=False, repr=False)
    _biases: list[float] = field(default=None, init=False, repr=False)

    def initialize(self, rng: _random.Random) -> None:
        self._centers = [
            index * 3.0 / max(self.rbf_centers - 1, 1)
            for index in range(self.rbf_centers)
        ]
        input_dim = 2 * self.hidden_dim + self.rbf_centers
        self._weights = _xavier_init(input_dim, self.hidden_dim, rng)
        self._biases = _zero_vec(self.hidden_dim)

    def compute_message(
        self,
        source_features: list[float],
        target_features: list[float],
        distance: float,
    ) -> list[float]:
        rbf = _gaussian_rbf(distance, self._centers, self.rbf_width)
        inputs = source_features + target_features + rbf
        output = list(self._biases)
        for feature_index, feature_value in enumerate(inputs):
            weights = self._weights[feature_index]
            for hidden_index in range(self.hidden_dim):
                output[hidden_index] += weights[hidden_index] * feature_value
        return [_tanh(value) for value in output]


@dataclass(slots=True)
class InteractionUpdateLayer:
    """Mix current piece features with aggregated spring messages."""

    hidden_dim: int = 16
    residual_blend: float = 0.6
    _weights: list[list[float]] = field(default=None, init=False, repr=False)
    _biases: list[float] = field(default=None, init=False, repr=False)

    def initialize(self, rng: _random.Random) -> None:
        self._weights = _xavier_init(2 * self.hidden_dim, self.hidden_dim, rng)
        self._biases = _zero_vec(self.hidden_dim)

    def update(
        self,
        piece_features: list[float],
        message_features: list[float],
    ) -> list[float]:
        inputs = piece_features + message_features
        updated = list(self._biases)
        for feature_index, feature_value in enumerate(inputs):
            weights = self._weights[feature_index]
            for hidden_index in range(self.hidden_dim):
                updated[hidden_index] += weights[hidden_index] * feature_value
        activated = [_tanh(value) for value in updated]
        return [
            self.residual_blend * piece_features[hidden_index]
            + (1.0 - self.residual_blend) * activated[hidden_index]
            for hidden_index in range(self.hidden_dim)
        ]


@dataclass(slots=True)
class InteractionBlock:
    """One SPRING-style local interaction block."""

    hidden_dim: int = 16
    rbf_centers: int = 8
    residual_blend: float = 0.6
    message_layer: SpringMessageLayer = field(default=None, init=False, repr=False)
    update_layer: InteractionUpdateLayer = field(default=None, init=False, repr=False)

    def initialize(self, rng: _random.Random) -> None:
        self.message_layer = SpringMessageLayer(
            hidden_dim=self.hidden_dim,
            rbf_centers=self.rbf_centers,
        )
        self.message_layer.initialize(rng)
        self.update_layer = InteractionUpdateLayer(
            hidden_dim=self.hidden_dim,
            residual_blend=self.residual_blend,
        )
        self.update_layer.initialize(rng)

    def propagate(
        self,
        positions: VectorTuple,
        neighbors: list[list[int]],
        features: list[list[float]],
    ) -> list[list[float]]:
        updated_features: list[list[float]] = []
        for particle_index, piece_features in enumerate(features):
            message_mean = _zero_vec(self.hidden_dim)
            local_neighbors = neighbors[particle_index]
            for neighbor_index in local_neighbors:
                distance = _vec3_dist(
                    positions[particle_index],
                    positions[neighbor_index],
                )
                message = self.message_layer.compute_message(
                    features[neighbor_index],
                    piece_features,
                    distance,
                )
                for hidden_index in range(self.hidden_dim):
                    message_mean[hidden_index] += message[hidden_index]
            if local_neighbors:
                neighbor_scale = 1.0 / len(local_neighbors)
                for hidden_index in range(self.hidden_dim):
                    message_mean[hidden_index] *= neighbor_scale
            updated_features.append(
                self.update_layer.update(piece_features, message_mean)
            )
        return updated_features


@dataclass(slots=True)
class LocalCorrector:
    """Linear local correction head over aggregated piece+spring features."""

    hidden_dim: int = 16
    _energy_weights: list[float] = field(default=None, init=False, repr=False)
    _energy_bias: float = field(default=0.0, init=False, repr=False)
    _force_weights: list[float] = field(default=None, init=False, repr=False)
    _force_bias: float = field(default=0.0, init=False, repr=False)

    def initialize(self, input_dim: int) -> None:
        # Zero initialization keeps fresh residual predictions neutral until
        # qcloud-backed observations arrive.
        self._energy_weights = _zero_vec(input_dim)
        self._energy_bias = 0.0
        self._force_weights = _zero_vec(input_dim)
        self._force_bias = 0.0

    def predict(self, features: list[float]) -> tuple[float, float, float]:
        energy = self._energy_bias + _dot(self._energy_weights, features)
        force_raw = self._force_bias + _dot(self._force_weights, features)
        force_scale = _tanh(force_raw)
        return energy, force_scale, force_raw


_RAW_FEATURE_DIM = 8


def _extract_particle_features(
    state: SimulationState,
    base_evaluation: ForceEvaluation,
    particle_index: int,
    *,
    max_force: float,
) -> list[float]:
    velocity = (
        state.particles.velocities[particle_index]
        if state.particles.velocities
        else (0.0, 0.0, 0.0)
    )
    force = base_evaluation.forces[particle_index]
    mass = state.particles.masses[particle_index]
    inv_max_force = 1.0 / max(max_force, 1e-12)
    velocity_magnitude = _vec3_mag(velocity)
    force_magnitude = _vec3_mag(force)
    kinetic_energy = 0.5 * mass * velocity_magnitude * velocity_magnitude
    label = state.particles.labels[particle_index] if state.particles.labels else ""
    type_hash = float(sum(ord(character) for character in label) % 100) / 100.0
    return [
        force[0] * inv_max_force,
        force[1] * inv_max_force,
        force[2] * inv_max_force,
        velocity_magnitude,
        math.log1p(mass),
        type_hash,
        kinetic_energy,
        force_magnitude * inv_max_force,
    ]


@dataclass(frozen=True, slots=True)
class ParticleCorrectionPrediction:
    """Per-particle local residual prediction."""

    energy_correction: float
    force_scale: float
    raw_force_scale: float


@dataclass(frozen=True, slots=True)
class ForwardCache:
    """Cached local neighborhood features for state-aware training."""

    state_id: StateId
    aggregated_features: tuple[tuple[float, ...], ...]
    particle_predictions: tuple[ParticleCorrectionPrediction, ...]
    base_forces: VectorTuple
    neighbor_counts: tuple[int, ...]
    mean_neighbor_count: float


@dataclass(slots=True)
class ScalableResidualModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Piece-local residual model with neighbor message passing and local heads."""

    hidden_dim: int = 16
    message_passing_steps: int = 2
    interaction_cutoff: float = 2.0
    rbf_centers: int = 8
    learning_rate: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 1e-5
    max_gradient_norm: float = 5.0
    force_loss_weight: float = 0.25
    residual_blend: float = 0.6
    confidence_growth_rate: float = 0.02
    random_seed: int = 42
    name: str = "scalable_residual_model"
    classification: str = "[proposed novel]"
    _encoder: LocalEncoder = field(default=None, init=False, repr=False)
    _interaction_blocks: tuple[InteractionBlock, ...] = field(default=(), init=False, repr=False)
    _corrector: LocalCorrector = field(default=None, init=False, repr=False)
    _rng: _random.Random = field(default=None, init=False, repr=False)
    _trained_count: int = field(default=0, init=False, repr=False)
    _energy_weight_velocity: list[float] = field(default=None, init=False, repr=False)
    _force_weight_velocity: list[float] = field(default=None, init=False, repr=False)
    _energy_bias_velocity: float = field(default=0.0, init=False, repr=False)
    _force_bias_velocity: float = field(default=0.0, init=False, repr=False)
    _last_forward_cache: ForwardCache | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))
        self._rng = _random.Random(self.random_seed)
        self._encoder = LocalEncoder(hidden_dim=self.hidden_dim)
        self._encoder.initialize(_RAW_FEATURE_DIM, self._rng)
        interaction_blocks: list[InteractionBlock] = []
        for _ in range(self.message_passing_steps):
            block = InteractionBlock(
                hidden_dim=self.hidden_dim,
                rbf_centers=self.rbf_centers,
                residual_blend=self.residual_blend,
            )
            block.initialize(self._rng)
            interaction_blocks.append(block)
        self._interaction_blocks = tuple(interaction_blocks)
        self._corrector = LocalCorrector(hidden_dim=self.hidden_dim)
        self._corrector.initialize(2 * self.hidden_dim)
        self._energy_weight_velocity = _zero_vec(2 * self.hidden_dim)
        self._force_weight_velocity = _zero_vec(2 * self.hidden_dim)

    def describe_role(self) -> str:
        return (
            "Learns additive residual corrections through piece-local encoding, "
            "spring-style neighbor message passing, and per-particle correction heads."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "ml/residual_model.py",
            "physics/forces/composite.py",
            "qcloud/cloud_state.py",
            "spring/engine.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/scalable_ml_residual.md",)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.hidden_dim < 1:
            issues.append("hidden_dim must be at least 1.")
        if self.message_passing_steps < 1:
            issues.append("message_passing_steps must be at least 1.")
        if self.interaction_cutoff <= 0.0:
            issues.append("interaction_cutoff must be strictly positive.")
        if self.learning_rate <= 0.0:
            issues.append("learning_rate must be strictly positive.")
        if not (0.0 <= self.momentum < 1.0):
            issues.append("momentum must lie in [0, 1).")
        if self.weight_decay < 0.0:
            issues.append("weight_decay must be non-negative.")
        if self.max_gradient_norm <= 0.0:
            issues.append("max_gradient_norm must be strictly positive.")
        if self.force_loss_weight < 0.0:
            issues.append("force_loss_weight must be non-negative.")
        if not (0.0 <= self.residual_blend <= 1.0):
            issues.append("residual_blend must lie in [0, 1].")
        if self.confidence_growth_rate <= 0.0:
            issues.append("confidence_growth_rate must be strictly positive.")
        return tuple(issues)

    def trained_state_count(self) -> int:
        return self._trained_count

    def _find_neighbors(self, positions: VectorTuple) -> list[list[int]]:
        """Expected O(N * k) cell-list neighborhood construction."""
        cell_size = self.interaction_cutoff
        cutoff_sq = self.interaction_cutoff * self.interaction_cutoff
        cells: dict[tuple[int, int, int], list[int]] = {}
        for particle_index, position in enumerate(positions):
            cell = (
                math.floor(position[0] / cell_size),
                math.floor(position[1] / cell_size),
                math.floor(position[2] / cell_size),
            )
            cells.setdefault(cell, []).append(particle_index)

        neighbors: list[list[int]] = [[] for _ in range(len(positions))]
        for particle_index, position in enumerate(positions):
            origin = (
                math.floor(position[0] / cell_size),
                math.floor(position[1] / cell_size),
                math.floor(position[2] / cell_size),
            )
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        cell = (origin[0] + dx, origin[1] + dy, origin[2] + dz)
                        for neighbor_index in cells.get(cell, ()):
                            if neighbor_index <= particle_index:
                                continue
                            if _vec3_dist_sq(position, positions[neighbor_index]) <= cutoff_sq:
                                neighbors[particle_index].append(neighbor_index)
                                neighbors[neighbor_index].append(particle_index)
        return neighbors

    def _forward(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
    ) -> tuple[float, tuple[ParticleCorrectionPrediction, ...], tuple[tuple[float, ...], ...], tuple[int, ...]]:
        particle_count = state.particle_count
        max_force = max((_vec3_mag(force) for force in base_evaluation.forces), default=1.0)
        raw_features = [
            _extract_particle_features(
                state,
                base_evaluation,
                particle_index,
                max_force=max_force,
            )
            for particle_index in range(particle_count)
        ]
        encodings = [self._encoder.encode(features) for features in raw_features]
        neighbors = self._find_neighbors(state.particles.positions)
        propagated = [list(features) for features in encodings]
        for block in self._interaction_blocks:
            propagated = block.propagate(
                state.particles.positions,
                neighbors,
                propagated,
            )

        aggregated = [
            tuple(encodings[particle_index] + propagated[particle_index])
            for particle_index in range(particle_count)
        ]

        particle_predictions = tuple(
            ParticleCorrectionPrediction(*self._corrector.predict(list(features)))
            for features in aggregated
        )
        total_energy = sum(prediction.energy_correction for prediction in particle_predictions)
        neighbor_counts = tuple(len(local_neighbors) for local_neighbors in neighbors)
        return total_energy, particle_predictions, tuple(aggregated), neighbor_counts

    def predict(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
    ) -> ResidualPrediction:
        if len(base_evaluation.forces) != state.particle_count:
            raise ContractValidationError("base_evaluation.forces must match particle count.")

        total_energy, particle_predictions, aggregated, neighbor_counts = self._forward(
            state,
            base_evaluation,
        )
        mean_neighbor_count = (
            sum(neighbor_counts) / len(neighbor_counts)
            if neighbor_counts
            else 0.0
        )
        self._last_forward_cache = ForwardCache(
            state_id=state.provenance.state_id,
            aggregated_features=aggregated,
            particle_predictions=particle_predictions,
            base_forces=base_evaluation.forces,
            neighbor_counts=neighbor_counts,
            mean_neighbor_count=mean_neighbor_count,
        )

        force_deltas: list[ParticleForceDelta] = []
        for particle_index, prediction in enumerate(particle_predictions):
            base_force = base_evaluation.forces[particle_index]
            delta_force = tuple(
                prediction.force_scale * component
                for component in base_force
            )
            if any(abs(component) > 1e-15 for component in delta_force):
                force_deltas.append(
                    ParticleForceDelta(
                        particle_index=particle_index,
                        delta_force=delta_force,
                    )
                )

        confidence = min(1.0, self._trained_count * self.confidence_growth_rate)
        return ResidualPrediction(
            state_id=state.provenance.state_id,
            predicted_energy_delta=total_energy,
            force_deltas=tuple(force_deltas),
            confidence=confidence,
            metadata=FrozenMetadata(
                {
                    "mode": "piece_local_message_passing",
                    "neighbor_method": "cell_list",
                    "hidden_dim": self.hidden_dim,
                    "message_passing_steps": self.message_passing_steps,
                    "trained_count": self._trained_count,
                    "mean_neighbor_count": mean_neighbor_count,
                }
            ),
        )

    def _force_target_scales(
        self,
        target: ResidualTarget,
        base_forces: VectorTuple,
        particle_count: int,
    ) -> list[float]:
        scales = [0.0] * particle_count
        deltas_by_particle = {
            force_delta.particle_index: force_delta.delta_force
            for force_delta in target.force_deltas
        }
        for particle_index in range(particle_count):
            base_force = base_forces[particle_index]
            target_delta = deltas_by_particle.get(particle_index)
            if target_delta is None:
                continue
            denominator = sum(component * component for component in base_force)
            if denominator <= 1e-12:
                scales[particle_index] = 0.0
                continue
            numerator = sum(
                target_delta[axis] * base_force[axis]
                for axis in range(3)
            )
            scales[particle_index] = numerator / denominator
        return scales

    def _clip_gradients(
        self,
        energy_weight_gradient: list[float],
        force_weight_gradient: list[float],
        *,
        energy_bias_gradient: float,
        force_bias_gradient: float,
    ) -> tuple[float, float]:
        total_sq = (
            sum(value * value for value in energy_weight_gradient)
            + sum(value * value for value in force_weight_gradient)
            + energy_bias_gradient * energy_bias_gradient
            + force_bias_gradient * force_bias_gradient
        )
        grad_norm = math.sqrt(total_sq)
        if grad_norm <= self.max_gradient_norm or grad_norm <= 1e-12:
            return energy_bias_gradient, force_bias_gradient
        scale = self.max_gradient_norm / grad_norm
        for index in range(len(energy_weight_gradient)):
            energy_weight_gradient[index] *= scale
        for index in range(len(force_weight_gradient)):
            force_weight_gradient[index] *= scale
        return energy_bias_gradient * scale, force_bias_gradient * scale

    def _apply_gradients(
        self,
        energy_weight_gradient: list[float],
        force_weight_gradient: list[float],
        *,
        energy_bias_gradient: float,
        force_bias_gradient: float,
        sample_weight: float,
    ) -> None:
        learning_rate = self.learning_rate * sample_weight
        energy_bias_gradient, force_bias_gradient = self._clip_gradients(
            energy_weight_gradient,
            force_weight_gradient,
            energy_bias_gradient=energy_bias_gradient,
            force_bias_gradient=force_bias_gradient,
        )
        for index, weight in enumerate(self._corrector._energy_weights):
            gradient = energy_weight_gradient[index] + self.weight_decay * weight
            self._energy_weight_velocity[index] = self.momentum * self._energy_weight_velocity[index] + gradient
            self._corrector._energy_weights[index] -= learning_rate * self._energy_weight_velocity[index]
        for index, weight in enumerate(self._corrector._force_weights):
            gradient = force_weight_gradient[index] + self.weight_decay * weight
            self._force_weight_velocity[index] = self.momentum * self._force_weight_velocity[index] + gradient
            self._corrector._force_weights[index] -= learning_rate * self._force_weight_velocity[index]
        self._energy_bias_velocity = self.momentum * self._energy_bias_velocity + energy_bias_gradient
        self._force_bias_velocity = self.momentum * self._force_bias_velocity + force_bias_gradient
        self._corrector._energy_bias -= learning_rate * self._energy_bias_velocity
        self._corrector._force_bias -= learning_rate * self._force_bias_velocity

    def _observe_from_cache(
        self,
        cache: ForwardCache,
        target: ResidualTarget,
        *,
        sample_weight: float,
    ) -> None:
        particle_count = len(cache.aggregated_features)
        if particle_count == 0:
            self._trained_count += 1
            return

        target_energy_per_particle = target.energy_delta / particle_count
        target_force_scales = self._force_target_scales(
            target,
            cache.base_forces,
            particle_count,
        )
        feature_dim = len(cache.aggregated_features[0])
        energy_weight_gradient = _zero_vec(feature_dim)
        force_weight_gradient = _zero_vec(feature_dim)
        energy_bias_gradient = 0.0
        force_bias_gradient = 0.0

        for particle_index, features in enumerate(cache.aggregated_features):
            prediction = cache.particle_predictions[particle_index]
            energy_error = prediction.energy_correction - target_energy_per_particle
            energy_multiplier = 2.0 * energy_error / particle_count
            for feature_index, feature_value in enumerate(features):
                energy_weight_gradient[feature_index] += energy_multiplier * feature_value
            energy_bias_gradient += energy_multiplier

            target_force_scale = target_force_scales[particle_index]
            force_error = prediction.force_scale - target_force_scale
            force_multiplier = (
                self.force_loss_weight
                * 2.0
                * force_error
                * _tanh_deriv_from_output(prediction.force_scale)
                / particle_count
            )
            for feature_index, feature_value in enumerate(features):
                force_weight_gradient[feature_index] += force_multiplier * feature_value
            force_bias_gradient += force_multiplier

        self._apply_gradients(
            energy_weight_gradient,
            force_weight_gradient,
            energy_bias_gradient=energy_bias_gradient,
            force_bias_gradient=force_bias_gradient,
            sample_weight=sample_weight,
        )
        self._trained_count += 1

    def observe_state(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
        target: ResidualTarget,
        *,
        sample_weight: float = 1.0,
    ) -> None:
        sample_weight = coerce_scalar(sample_weight, "sample_weight")
        if sample_weight <= 0.0:
            raise ContractValidationError("sample_weight must be strictly positive.")
        self.predict(state, base_evaluation)
        cache = self._last_forward_cache
        self._observe_from_cache(cache, target, sample_weight=sample_weight)

    def observe(self, target: ResidualTarget, *, sample_weight: float = 1.0) -> None:
        sample_weight = coerce_scalar(sample_weight, "sample_weight")
        if sample_weight <= 0.0:
            raise ContractValidationError("sample_weight must be strictly positive.")
        if self._last_forward_cache is None:
            # Fallback for legacy callers that only provide a residual target.
            # This preserves protocol compatibility without pretending the update
            # is as informative as a state-aware neighborhood pass.
            energy_error = self._corrector._energy_bias - target.energy_delta
            average_force_scale = (
                sum(_vec3_mag(force_delta.delta_force) for force_delta in target.force_deltas) / len(target.force_deltas)
                if target.force_deltas
                else 0.0
            )
            force_error = self._corrector._force_bias - 0.01 * average_force_scale
            self._apply_gradients(
                _zero_vec(len(self._corrector._energy_weights)),
                _zero_vec(len(self._corrector._force_weights)),
                energy_bias_gradient=2.0 * energy_error,
                force_bias_gradient=2.0 * self.force_loss_weight * force_error,
                sample_weight=sample_weight,
            )
            self._trained_count += 1
            return
        self._observe_from_cache(self._last_forward_cache, target, sample_weight=sample_weight)

    def __hash__(self) -> int:
        return id(self)

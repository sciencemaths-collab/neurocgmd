"""Residual correction targets, predictions, and additive ML force wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, StateId, Vector3, VectorTuple, coerce_scalar
from forcefields.base_forcefield import BaseForceField
from physics.forces.composite import ForceEvaluation
from qcloud.cloud_state import ParticleForceDelta, QCloudCorrection
from topology.system_topology import SystemTopology


def _merge_force_deltas(
    deltas: Sequence[ParticleForceDelta],
    *,
    left_weight: float = 1.0,
    right: Sequence[ParticleForceDelta] = (),
    right_weight: float = 0.0,
) -> tuple[ParticleForceDelta, ...]:
    accumulator: dict[int, list[float]] = {}
    if left_weight:
        for force_delta in deltas:
            accumulator.setdefault(force_delta.particle_index, [0.0, 0.0, 0.0])
            for axis, value in enumerate(force_delta.delta_force):
                accumulator[force_delta.particle_index][axis] += left_weight * value
    if right_weight:
        for force_delta in right:
            accumulator.setdefault(force_delta.particle_index, [0.0, 0.0, 0.0])
            for axis, value in enumerate(force_delta.delta_force):
                accumulator[force_delta.particle_index][axis] += right_weight * value
    return tuple(
        ParticleForceDelta(particle_index=particle_index, delta_force=tuple(values))
        for particle_index, values in sorted(accumulator.items())
    )


def _average_force_deltas(
    left: Sequence[ParticleForceDelta],
    left_weight: float,
    right: Sequence[ParticleForceDelta],
    right_weight: float,
) -> tuple[ParticleForceDelta, ...]:
    total_weight = left_weight + right_weight
    if total_weight <= 0.0:
        return ()
    merged = _merge_force_deltas(left, left_weight=left_weight, right=right, right_weight=right_weight)
    return tuple(
        ParticleForceDelta(
            particle_index=force_delta.particle_index,
            delta_force=tuple(component / total_weight for component in force_delta.delta_force),
        )
        for force_delta in merged
    )


def _apply_force_deltas(base_forces: VectorTuple, deltas: Sequence[ParticleForceDelta]) -> VectorTuple:
    updated = [list(vector) for vector in base_forces]
    for force_delta in deltas:
        if force_delta.particle_index >= len(updated):
            raise ContractValidationError("Residual force delta references a particle outside the force block.")
        for axis, value in enumerate(force_delta.delta_force):
            updated[force_delta.particle_index][axis] += value
    return tuple(tuple(vector) for vector in updated)


@dataclass(frozen=True, slots=True)
class ResidualTarget(ValidatableComponent):
    """Observed additive residual correction target for one state."""

    state_id: StateId
    energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...] = ()
    source_label: str = "residual_target"
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "energy_delta", coerce_scalar(self.energy_delta, "energy_delta"))
        object.__setattr__(self, "force_deltas", tuple(self.force_deltas))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def affected_particles(self) -> tuple[int, ...]:
        return tuple(force_delta.particle_index for force_delta in self.force_deltas)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if not self.source_label.strip():
            issues.append("source_label must be a non-empty string.")
        affected_particles = self.affected_particles()
        if len(affected_particles) != len(set(affected_particles)):
            issues.append("force_deltas must not contain duplicate particle_index values.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": str(self.state_id),
            "energy_delta": self.energy_delta,
            "force_deltas": [force_delta.to_dict() for force_delta in self.force_deltas],
            "source_label": self.source_label,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResidualTarget":
        return cls(
            state_id=StateId(str(data["state_id"])),
            energy_delta=float(data["energy_delta"]),
            force_deltas=tuple(
                ParticleForceDelta.from_dict(force_delta)
                for force_delta in data.get("force_deltas", ())
            ),
            source_label=str(data.get("source_label", "residual_target")),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )

    @classmethod
    def from_corrections(
        cls,
        state_id: StateId | str,
        corrections: Sequence[QCloudCorrection],
        *,
        source_label: str = "qcloud",
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> "ResidualTarget":
        aggregated_energy_delta = sum(correction.energy_delta for correction in corrections)
        aggregated_force_deltas = _merge_force_deltas(
            [force_delta for correction in corrections for force_delta in correction.force_deltas]
        )
        return cls(
            state_id=StateId(str(state_id)),
            energy_delta=aggregated_energy_delta,
            force_deltas=aggregated_force_deltas,
            source_label=source_label,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )


@dataclass(frozen=True, slots=True)
class ResidualPrediction(ValidatableComponent):
    """Predicted additive residual correction for one state."""

    state_id: StateId
    predicted_energy_delta: float
    force_deltas: tuple[ParticleForceDelta, ...] = ()
    confidence: float = 0.0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(
            self,
            "predicted_energy_delta",
            coerce_scalar(self.predicted_energy_delta, "predicted_energy_delta"),
        )
        object.__setattr__(self, "force_deltas", tuple(self.force_deltas))
        object.__setattr__(self, "confidence", coerce_scalar(self.confidence, "confidence"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def affected_particles(self) -> tuple[int, ...]:
        return tuple(force_delta.particle_index for force_delta in self.force_deltas)

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if not (0.0 <= self.confidence <= 1.0):
            issues.append("confidence must lie in the interval [0, 1].")
        affected_particles = self.affected_particles()
        if len(affected_particles) != len(set(affected_particles)):
            issues.append("force_deltas must not contain duplicate particle_index values.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": str(self.state_id),
            "predicted_energy_delta": self.predicted_energy_delta,
            "force_deltas": [force_delta.to_dict() for force_delta in self.force_deltas],
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ResidualPrediction":
        return cls(
            state_id=StateId(str(data["state_id"])),
            predicted_energy_delta=float(data["predicted_energy_delta"]),
            force_deltas=tuple(
                ParticleForceDelta.from_dict(force_delta)
                for force_delta in data.get("force_deltas", ())
            ),
            confidence=float(data.get("confidence", 0.0)),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@runtime_checkable
class ResidualModel(Protocol):
    """Protocol for learned additive residual correction providers."""

    name: str
    classification: str

    def predict(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
    ) -> ResidualPrediction:
        """Predict an additive residual correction for the supplied state."""

    def observe(self, target: ResidualTarget, *, sample_weight: float = 1.0) -> None:
        """Update the model from one observed target."""

    def trained_state_count(self) -> int:
        """Return the number of states with learned residual observations."""


@runtime_checkable
class StateAwareResidualModel(ResidualModel, Protocol):
    """Optional residual-model extension for neighborhood-aware online updates."""

    def observe_state(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
        target: ResidualTarget,
        *,
        sample_weight: float = 1.0,
    ) -> None:
        """Update the model using the explicit state and baseline force block."""


@dataclass(slots=True)
class ResidualMemoryModel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Deterministic replay-driven residual model with exact-state recall and global averaging."""

    minimum_weight_for_high_confidence: float = 3.0
    name: str = "residual_memory_model"
    classification: str = "[hybrid]"
    _targets_by_state: dict[StateId, ResidualTarget] = field(default_factory=dict, init=False, repr=False)
    _observation_weight_by_state: dict[StateId, float] = field(default_factory=dict, init=False, repr=False)
    _total_weight: float = field(default=0.0, init=False, repr=False)
    _weighted_energy_delta_sum: float = field(default=0.0, init=False, repr=False)

    def describe_role(self) -> str:
        return (
            "Stores replay-driven additive residual targets and returns bounded "
            "state-specific or global-mean correction predictions."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "memory/replay_buffer.py",
            "qcloud/cloud_state.py",
            "physics/forces/composite.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/online_ml_residual_learning.md",
            "docs/sections/section_11_online_ml_residual_learning.md",
        )

    def validate(self) -> tuple[str, ...]:
        if self.minimum_weight_for_high_confidence <= 0.0:
            return ("minimum_weight_for_high_confidence must be strictly positive.",)
        return ()

    def trained_state_count(self) -> int:
        return len(self._targets_by_state)

    def observation_weight_for_state(self, state_id: StateId | str) -> float:
        return self._observation_weight_by_state.get(StateId(str(state_id)), 0.0)

    def observe(self, target: ResidualTarget, *, sample_weight: float = 1.0) -> None:
        sample_weight = coerce_scalar(sample_weight, "sample_weight")
        if sample_weight <= 0.0:
            raise ContractValidationError("sample_weight must be strictly positive.")

        state_id = target.state_id
        previous_target = self._targets_by_state.get(state_id)
        previous_weight = self._observation_weight_by_state.get(state_id, 0.0)
        if previous_target is None:
            updated_target = target
        else:
            total_weight = previous_weight + sample_weight
            updated_target = ResidualTarget(
                state_id=state_id,
                energy_delta=(
                    (previous_target.energy_delta * previous_weight)
                    + (target.energy_delta * sample_weight)
                )
                / total_weight,
                force_deltas=_average_force_deltas(
                    previous_target.force_deltas,
                    previous_weight,
                    target.force_deltas,
                    sample_weight,
                ),
                source_label=target.source_label,
                metadata=target.metadata.with_updates({"aggregated": True}),
            )

        self._targets_by_state[state_id] = updated_target
        self._observation_weight_by_state[state_id] = previous_weight + sample_weight
        self._total_weight += sample_weight
        self._weighted_energy_delta_sum += target.energy_delta * sample_weight

    def predict(
        self,
        state: SimulationState,
        base_evaluation: ForceEvaluation,
    ) -> ResidualPrediction:
        state_id = state.provenance.state_id
        if len(base_evaluation.forces) != state.particle_count:
            raise ContractValidationError("base_evaluation.forces must match the SimulationState particle count.")

        if state_id in self._targets_by_state:
            target = self._targets_by_state[state_id]
            weight = self._observation_weight_by_state[state_id]
            confidence = min(1.0, weight / self.minimum_weight_for_high_confidence)
            return ResidualPrediction(
                state_id=state_id,
                predicted_energy_delta=target.energy_delta,
                force_deltas=target.force_deltas,
                confidence=confidence,
                metadata=FrozenMetadata(
                    {
                        "mode": "exact_replay",
                        "observation_weight": weight,
                        "base_potential_energy": base_evaluation.potential_energy,
                    }
                ),
            )

        global_mean_energy_delta = (
            self._weighted_energy_delta_sum / self._total_weight if self._total_weight else 0.0
        )
        return ResidualPrediction(
            state_id=state_id,
            predicted_energy_delta=global_mean_energy_delta,
            force_deltas=(),
            confidence=0.1 if self._total_weight else 0.0,
            metadata=FrozenMetadata(
                {
                    "mode": "global_mean",
                    "trained_state_count": self.trained_state_count(),
                    "base_potential_energy": base_evaluation.potential_energy,
                }
            ),
        )


@dataclass(slots=True)
class ResidualAugmentedForceEvaluator(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Compose a baseline force evaluator with an additive learned residual model."""

    base_force_evaluator: object
    residual_model: ResidualModel
    name: str = "residual_augmented_force_evaluator"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Combines established baseline force evaluation with additive learned "
            "residual predictions while preserving explicit component accounting."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "physics/forces/composite.py",
            "ml/residual_model.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/online_ml_residual_learning.md",
            "docs/sections/section_11_online_ml_residual_learning.md",
        )

    def validate(self) -> tuple[str, ...]:
        if not hasattr(self.base_force_evaluator, "evaluate"):
            return ("base_force_evaluator must define evaluate(state, topology, forcefield).",)
        return ()

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        base_evaluation = self.base_force_evaluator.evaluate(state, topology, forcefield)
        prediction = self.residual_model.predict(state, base_evaluation)
        corrected_forces = _apply_force_deltas(base_evaluation.forces, prediction.force_deltas)
        return ForceEvaluation(
            forces=corrected_forces,
            potential_energy=base_evaluation.potential_energy + prediction.predicted_energy_delta,
            component_energies=base_evaluation.component_energies.with_updates(
                {"ml_residual": prediction.predicted_energy_delta}
            ),
            metadata=base_evaluation.metadata.with_updates(
                {
                    "residual_model": self.residual_model.name,
                    "residual_confidence": prediction.confidence,
                }
            ),
        )

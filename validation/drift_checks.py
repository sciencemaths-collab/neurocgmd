"""Trajectory-level drift and consistency checks for observer-side validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import dist

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, SimulationId, StateId, coerce_scalar


@dataclass(frozen=True, slots=True)
class DriftThresholds(ValidatableComponent):
    """Thresholds used by the trajectory drift checker."""

    max_energy_drift: float = 1.0
    max_position_displacement: float = 10.0
    require_monotonic_time: bool = True
    require_unit_step_progression: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_energy_drift", coerce_scalar(self.max_energy_drift, "max_energy_drift"))
        object.__setattr__(
            self,
            "max_position_displacement",
            coerce_scalar(self.max_position_displacement, "max_position_displacement"),
        )
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.max_energy_drift < 0.0:
            issues.append("max_energy_drift must be non-negative.")
        if self.max_position_displacement < 0.0:
            issues.append("max_position_displacement must be non-negative.")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class DriftCheckReport(ValidatableComponent):
    """Drift-assessment result for one ordered trajectory window."""

    simulation_id: SimulationId
    reference_state_id: StateId
    final_state_id: StateId
    state_count: int
    max_energy_drift: float
    max_position_displacement: float
    time_monotonic: bool
    step_sequence_ok: bool
    violations: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "simulation_id", SimulationId(str(self.simulation_id)))
        object.__setattr__(self, "reference_state_id", StateId(str(self.reference_state_id)))
        object.__setattr__(self, "final_state_id", StateId(str(self.final_state_id)))
        object.__setattr__(self, "max_energy_drift", coerce_scalar(self.max_energy_drift, "max_energy_drift"))
        object.__setattr__(
            self,
            "max_position_displacement",
            coerce_scalar(self.max_position_displacement, "max_position_displacement"),
        )
        object.__setattr__(self, "violations", tuple(self.violations))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def passed(self) -> bool:
        return not self.violations

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.state_count <= 0:
            issues.append("state_count must be positive.")
        if self.max_energy_drift < 0.0:
            issues.append("max_energy_drift must be non-negative.")
        if self.max_position_displacement < 0.0:
            issues.append("max_position_displacement must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class TrajectoryDriftChecker(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Evaluate monotonicity and coarse drift over an ordered state sequence."""

    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    name: str = "trajectory_drift_checker"
    classification: str = "[established]"

    def describe_role(self) -> str:
        return (
            "Evaluates monotonic time/step progression and coarse energy/position "
            "drift over ordered simulation-state sequences."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "core/state_registry.py",
            "sampling/simulation_loop.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/validation_and_benchmarking.md",
            "docs/sections/section_13_validation_and_benchmarking_suite.md",
        )

    def validate(self) -> tuple[str, ...]:
        return self.thresholds.validate()

    def assess(self, states: Sequence[SimulationState]) -> DriftCheckReport:
        ordered_states = tuple(states)
        if not ordered_states:
            raise ContractValidationError("TrajectoryDriftChecker requires at least one state.")

        reference = ordered_states[0]
        simulation_id = reference.provenance.simulation_id
        particle_count = reference.particle_count
        violations: list[str] = []

        consistent_simulation = all(state.provenance.simulation_id == simulation_id for state in ordered_states)
        if not consistent_simulation:
            violations.append("states span multiple simulation identifiers.")

        consistent_particle_count = all(state.particle_count == particle_count for state in ordered_states)
        if not consistent_particle_count:
            violations.append("states do not share a consistent particle count.")

        time_monotonic = all(current.time >= previous.time for previous, current in zip(ordered_states, ordered_states[1:]))
        step_sequence_ok = all(current.step == previous.step + 1 for previous, current in zip(ordered_states, ordered_states[1:]))

        if self.thresholds.require_monotonic_time and not time_monotonic:
            violations.append("state times are not monotonic.")
        if self.thresholds.require_unit_step_progression and not step_sequence_ok:
            violations.append("state steps do not increase by one.")

        reference_energy = reference.potential_energy if reference.potential_energy is not None else 0.0
        max_energy_drift = max(
            abs((state.potential_energy if state.potential_energy is not None else 0.0) - reference_energy)
            for state in ordered_states
        )
        if max_energy_drift > self.thresholds.max_energy_drift:
            violations.append("energy drift exceeded the configured threshold.")

        max_position_displacement = 0.0
        if consistent_particle_count:
            for state in ordered_states:
                for particle_index in range(particle_count):
                    max_position_displacement = max(
                        max_position_displacement,
                        dist(
                            state.particles.positions[particle_index],
                            reference.particles.positions[particle_index],
                        ),
                    )
        if max_position_displacement > self.thresholds.max_position_displacement:
            violations.append("particle displacement exceeded the configured threshold.")

        return DriftCheckReport(
            simulation_id=simulation_id,
            reference_state_id=reference.provenance.state_id,
            final_state_id=ordered_states[-1].provenance.state_id,
            state_count=len(ordered_states),
            max_energy_drift=max_energy_drift,
            max_position_displacement=max_position_displacement,
            time_monotonic=time_monotonic,
            step_sequence_ok=step_sequence_ok,
            violations=tuple(violations),
            metadata=FrozenMetadata(
                {
                    "consistent_simulation_id": consistent_simulation,
                    "consistent_particle_count": consistent_particle_count,
                }
            ),
        )

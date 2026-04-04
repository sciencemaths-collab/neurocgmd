"""Backend-ready integration kernel entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, VectorTuple, coerce_scalar
from physics.forces.composite import ForceEvaluation


@dataclass(frozen=True, slots=True)
class IntegrationKernelResult(ValidatableComponent):
    """Result of one backend-ready integration kernel step."""

    positions: VectorTuple
    velocities: VectorTuple
    time_step: float
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_step", coerce_scalar(self.time_step, "time_step"))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.time_step <= 0.0:
            issues.append("time_step must be strictly positive.")
        if len(self.positions) != len(self.velocities):
            issues.append("positions and velocities must have the same length.")
        return tuple(issues)


@dataclass(slots=True)
class VelocityVerletIntegrationKernel(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Reference integration kernel for backend-neutral execution planning."""

    name: str = "velocity_verlet_integration_kernel"
    classification: str = "[established]"

    def describe_role(self) -> str:
        return "Provides a backend-ready Velocity Verlet update step over a stable state contract."

    def declared_dependencies(self) -> tuple[str, ...]:
        return ("physics/forces/composite.py",)

    def documentation_paths(self) -> tuple[str, ...]:
        return ("docs/architecture/backend_compute_spine.md",)

    def validate(self) -> tuple[str, ...]:
        return ()

    def step(
        self,
        state: SimulationState,
        force_evaluation: ForceEvaluation,
        *,
        time_step: float,
    ) -> IntegrationKernelResult:
        if len(force_evaluation.forces) != state.particle_count:
            raise ContractValidationError("force_evaluation must match the SimulationState particle count.")
        dt = coerce_scalar(time_step, "time_step")
        if dt <= 0.0:
            raise ContractValidationError("time_step must be strictly positive.")
        previous_velocities = (
            state.particles.velocities
            if state.particles.velocities
            else tuple((0.0, 0.0, 0.0) for _ in range(state.particle_count))
        )
        updated_positions = []
        updated_velocities = []
        for particle_index in range(state.particle_count):
            mass = state.particles.masses[particle_index]
            force = force_evaluation.forces[particle_index]
            acceleration = tuple(component / mass for component in force)
            velocity = previous_velocities[particle_index]
            position = state.particles.positions[particle_index]
            next_velocity = tuple(velocity[axis] + dt * acceleration[axis] for axis in range(3))
            next_position = tuple(
                position[axis] + dt * velocity[axis] + 0.5 * dt * dt * acceleration[axis]
                for axis in range(3)
            )
            updated_positions.append(next_position)
            updated_velocities.append(next_velocity)
        return IntegrationKernelResult(
            positions=tuple(updated_positions),
            velocities=tuple(updated_velocities),
            time_step=dt,
            metadata=FrozenMetadata({"kernel": self.name}),
        )

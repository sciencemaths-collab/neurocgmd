"""Established Langevin-style integration for the early simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt
from random import Random

from core.exceptions import ContractValidationError
from core.state import ParticleState, SimulationState
from core.types import FrozenMetadata
from forcefields.base_forcefield import BaseForceField
from integrators.base import ForceEvaluator, IntegratorStepResult
from topology.system_topology import SystemTopology


@dataclass(slots=True)
class LangevinIntegrator:
    """Velocity-Verlet-style Langevin integrator with optional reduced-unit noise."""

    time_step: float
    friction_coefficient: float | None = None
    stochastic: bool = False
    assume_reduced_units: bool = False
    thermal_energy_scale: float = 1.0
    random_seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    name: str = "langevin_integrator"
    classification: str = "[established]"

    def __post_init__(self) -> None:
        if self.time_step <= 0.0:
            raise ContractValidationError("time_step must be positive.")
        if self.friction_coefficient is not None and self.friction_coefficient < 0.0:
            raise ContractValidationError("friction_coefficient must be non-negative when set.")
        if self.thermal_energy_scale <= 0.0:
            raise ContractValidationError("thermal_energy_scale must be positive.")
        object.__setattr__(self, "_rng", Random(self.random_seed))

    def step(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator: ForceEvaluator,
    ) -> IntegratorStepResult:
        force_evaluation = force_evaluator.evaluate(state, topology, forcefield)
        gamma = self._effective_friction(state)
        dt = self.time_step

        half_velocities = []
        for index, (velocity, force, mass) in enumerate(
            zip(
                state.particles.velocities,
                force_evaluation.forces,
                state.particles.masses,
                strict=True,
            )
        ):
            half_velocity = tuple(
                velocity[axis] + 0.5 * dt * force[axis] / mass for axis in range(3)
            )
            half_velocities.append(
                self._apply_thermostat(
                    half_velocity,
                    mass=mass,
                    gamma=gamma,
                    temperature=state.thermodynamics.target_temperature,
                    particle_index=index,
                )
            )

        positions = tuple(
            tuple(
                state.particles.positions[index][axis] + dt * half_velocities[index][axis]
                for axis in range(3)
            )
            for index in range(state.particle_count)
        )

        predicted_particles = ParticleState(
            positions=positions,
            masses=state.particles.masses,
            velocities=tuple(half_velocities),
            forces=force_evaluation.forces,
            labels=state.particles.labels,
        )
        predicted_state = SimulationState(
            units=state.units,
            particles=predicted_particles,
            thermodynamics=state.thermodynamics,
            provenance=state.provenance,
            cell=state.cell,
            time=state.time + dt,
            step=state.step + 1,
            potential_energy=force_evaluation.potential_energy,
            observables=state.observables,
        )
        next_force_evaluation = force_evaluator.evaluate(predicted_state, topology, forcefield)

        next_velocities = tuple(
            tuple(
                half_velocities[index][axis]
                + 0.5 * dt * next_force_evaluation.forces[index][axis] / state.particles.masses[index]
                for axis in range(3)
            )
            for index in range(state.particle_count)
        )

        next_particles = ParticleState(
            positions=positions,
            masses=state.particles.masses,
            velocities=next_velocities,
            forces=next_force_evaluation.forces,
            labels=state.particles.labels,
        )
        kinetic_energy = 0.5 * sum(
            state.particles.masses[i] * sum(next_velocities[i][ax] ** 2 for ax in range(3))
            for i in range(state.particle_count)
        )
        observables = FrozenMetadata(
            {
                "integrator": self.name,
                "kinetic_energy": kinetic_energy,
                "force_norm_l1": sum(
                    abs(component)
                    for vector in next_force_evaluation.forces
                    for component in vector
                ),
            }
        )
        metadata = FrozenMetadata(
            {
                "friction_coefficient": gamma,
                "stochastic": self.stochastic,
            }
        )
        return IntegratorStepResult(
            particles=next_particles,
            time=state.time + dt,
            step=state.step + 1,
            potential_energy=next_force_evaluation.potential_energy,
            observables=observables,
            metadata=metadata,
        )

    def _effective_friction(self, state: SimulationState) -> float:
        if self.friction_coefficient is not None:
            return self.friction_coefficient
        return state.thermodynamics.friction_coefficient or 0.0

    def _apply_thermostat(
        self,
        velocity: tuple[float, float, float],
        *,
        mass: float,
        gamma: float,
        temperature: float | None,
        particle_index: int,
    ) -> tuple[float, float, float]:
        if gamma == 0.0:
            return velocity

        damping = exp(-gamma * self.time_step)
        damped_velocity = tuple(damping * component for component in velocity)
        if not self.stochastic:
            return damped_velocity

        if not self.assume_reduced_units:
            raise ContractValidationError(
                "stochastic Langevin mode currently requires assume_reduced_units=True."
            )
        if temperature is None or temperature <= 0.0:
            raise ContractValidationError(
                "stochastic Langevin mode requires a positive target_temperature."
            )

        noise_scale = sqrt((1.0 - damping * damping) * self.thermal_energy_scale * temperature / mass)
        noise = tuple(self._rng.gauss(0.0, noise_scale) for _ in range(3))
        return tuple(damped_velocity[axis] + noise[axis] for axis in range(3))

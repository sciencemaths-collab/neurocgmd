"""BAOAB Langevin splitting integrator (Leimkuhler & Matthews, 2013).

The BAOAB scheme splits the Langevin equation into five sub-steps:

    B — half-step velocity kick from forces
    A — half-step position drift
    O — Ornstein-Uhlenbeck velocity update (exact thermostat)
    A — half-step position drift
    B — half-step velocity kick from forces

This ordering places the stochastic (O) step symmetrically at the centre,
yielding provably superior configurational sampling compared to other
splittings at the same friction coefficient.  Crucially, only ONE force
evaluation per full step is required (the final B uses forces at the new
positions), making BAOAB both more accurate and cheaper than the classical
velocity-Verlet Langevin approach that needs two evaluations.
"""

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
class BAOABIntegrator:
    """BAOAB Langevin splitting integrator for configurational sampling.

    Parameters
    ----------
    time_step:
        Integration time step (dt).  Must be positive.
    friction_coefficient:
        Friction / collision frequency (gamma).  When *None* the value is
        read from ``state.thermodynamics.friction_coefficient`` at each step.
    assume_reduced_units:
        When *True* the thermal energy is ``thermal_energy_scale * T``
        (i.e. kT is implicit in the unit system).  Currently the only
        supported mode for the stochastic O-step.
    thermal_energy_scale:
        Multiplicative prefactor for the thermal energy in reduced units.
        Must be positive.  Defaults to 1.0.
    random_seed:
        Seed for the internal PRNG that drives the O-step noise.
    """

    time_step: float
    friction_coefficient: float | None = None
    assume_reduced_units: bool = True
    thermal_energy_scale: float = 1.0
    random_seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    name: str = "baoab_integrator"
    classification: str = "[established]"

    # --------------------------------------------------------------------- #
    # Validation
    # --------------------------------------------------------------------- #

    def __post_init__(self) -> None:
        if self.time_step <= 0.0:
            raise ContractValidationError("time_step must be positive.")
        if self.friction_coefficient is not None and self.friction_coefficient < 0.0:
            raise ContractValidationError(
                "friction_coefficient must be non-negative when set."
            )
        if self.thermal_energy_scale <= 0.0:
            raise ContractValidationError("thermal_energy_scale must be positive.")
        if not self.assume_reduced_units:
            raise ContractValidationError(
                "BAOAB integrator currently requires assume_reduced_units=True."
            )
        object.__setattr__(self, "_rng", Random(self.random_seed))

    # --------------------------------------------------------------------- #
    # Public interface
    # --------------------------------------------------------------------- #

    def step(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
        force_evaluator: ForceEvaluator,
    ) -> IntegratorStepResult:
        """Advance *state* by one full BAOAB step.

        The splitting is executed as::

            B  —  half-kick velocities using current forces
            A  —  half-drift positions using kicked velocities
            O  —  exact Ornstein-Uhlenbeck thermostat on velocities
            A  —  half-drift positions using thermostated velocities
            B  —  half-kick velocities using NEW forces (single evaluation)
        """
        dt = self.time_step
        half_dt = 0.5 * dt
        gamma = self._effective_friction(state)
        temperature = state.thermodynamics.target_temperature
        n_particles = state.particle_count

        if temperature is None or temperature <= 0.0:
            raise ContractValidationError(
                "BAOAB requires a positive target_temperature for the O-step."
            )

        positions = state.particles.positions
        velocities = state.particles.velocities
        forces = state.particles.forces
        masses = state.particles.masses

        # ---- B: half-step velocity kick from current forces ---- #
        velocities_b1: list[tuple[float, float, float]] = []
        for i in range(n_particles):
            inv_mass = 1.0 / masses[i]
            velocities_b1.append(
                tuple(
                    velocities[i][ax] + half_dt * forces[i][ax] * inv_mass
                    for ax in range(3)
                )
            )

        # ---- A: half-step position drift ---- #
        positions_a1: list[tuple[float, float, float]] = []
        for i in range(n_particles):
            positions_a1.append(
                tuple(
                    positions[i][ax] + half_dt * velocities_b1[i][ax]
                    for ax in range(3)
                )
            )

        # ---- O: Ornstein-Uhlenbeck exact velocity update ---- #
        c1 = exp(-gamma * dt)
        kT = self.thermal_energy_scale * temperature

        velocities_o: list[tuple[float, float, float]] = []
        for i in range(n_particles):
            c2 = sqrt((1.0 - c1 * c1) * kT / masses[i])
            velocities_o.append(
                tuple(
                    c1 * velocities_b1[i][ax] + c2 * self._rng.gauss(0.0, 1.0)
                    for ax in range(3)
                )
            )

        # ---- A: half-step position drift ---- #
        positions_a2: list[tuple[float, float, float]] = []
        for i in range(n_particles):
            positions_a2.append(
                tuple(
                    positions_a1[i][ax] + half_dt * velocities_o[i][ax]
                    for ax in range(3)
                )
            )

        # Build an intermediate state so the force evaluator sees the new positions.
        mid_particles = ParticleState(
            positions=tuple(positions_a2),
            masses=masses,
            velocities=tuple(velocities_o),
            forces=forces,  # placeholder; about to be replaced
            labels=state.particles.labels,
        )
        mid_state = SimulationState(
            units=state.units,
            particles=mid_particles,
            thermodynamics=state.thermodynamics,
            provenance=state.provenance,
            cell=state.cell,
            time=state.time + dt,
            step=state.step + 1,
            potential_energy=state.potential_energy,
            observables=state.observables,
        )

        # Single force evaluation at the new positions.
        force_evaluation = force_evaluator.evaluate(mid_state, topology, forcefield)
        new_forces = force_evaluation.forces

        # ---- B: half-step velocity kick from NEW forces ---- #
        velocities_b2: list[tuple[float, float, float]] = []
        for i in range(n_particles):
            inv_mass = 1.0 / masses[i]
            velocities_b2.append(
                tuple(
                    velocities_o[i][ax] + half_dt * new_forces[i][ax] * inv_mass
                    for ax in range(3)
                )
            )

        # ---- Assemble result ---- #
        kinetic_energy = 0.5 * sum(
            masses[i]
            * sum(velocities_b2[i][ax] ** 2 for ax in range(3))
            for i in range(n_particles)
        )

        next_particles = ParticleState(
            positions=tuple(positions_a2),
            masses=masses,
            velocities=tuple(velocities_b2),
            forces=new_forces,
            labels=state.particles.labels,
        )

        observables = FrozenMetadata(
            {
                "integrator": self.name,
                "kinetic_energy": kinetic_energy,
                "force_norm_l1": sum(
                    abs(component)
                    for vector in new_forces
                    for component in vector
                ),
            }
        )
        metadata = FrozenMetadata(
            {
                "friction_coefficient": gamma,
                "ou_c1": c1,
                "stochastic": True,
            }
        )

        return IntegratorStepResult(
            particles=next_particles,
            time=state.time + dt,
            step=state.step + 1,
            potential_energy=force_evaluation.potential_energy,
            observables=observables,
            metadata=metadata,
        )

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _effective_friction(self, state: SimulationState) -> float:
        """Resolve friction: explicit parameter wins, then state, then error."""
        if self.friction_coefficient is not None:
            return self.friction_coefficient
        gamma = state.thermodynamics.friction_coefficient
        if gamma is None or gamma <= 0.0:
            raise ContractValidationError(
                "BAOAB requires a positive friction coefficient; none was "
                "provided to the integrator or found in thermodynamics state."
            )
        return gamma

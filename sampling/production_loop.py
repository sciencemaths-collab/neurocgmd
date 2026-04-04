"""Production-grade simulation loop wiring together all NeuroCGMD subsystems.

Integrates PBC wrapping, holonomic constraints (SHAKE/LINCS), temperature
scheduling, energy tracking with drift diagnostics, and pluggable analysis
hooks into a single orchestrated loop built on top of the foundational
SimulationLoop design.

Classification: [adapted]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum
from typing import Any, Callable

from core.exceptions import ContractValidationError
from core.state import (
    ParticleState,
    SimulationState,
    ThermodynamicState,
)
from core.state_registry import LifecycleStage, SimulationStateRegistry
from core.types import FrozenMetadata, StateId
from core.units import BOLTZMANN_CONSTANT
from forcefields.base_forcefield import BaseForceField
from integrators.base import IntegratorStepResult
from physics.constraints import DistanceConstraint
from topology.system_topology import SystemTopology


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemperatureSchedule:
    """Configurable temperature protocol for simulated annealing or ramping.

    Supported modes:
        - ``constant``: fixed temperature throughout.
        - ``linear_ramp``: linear interpolation from *initial* to *final*.
        - ``exponential_anneal``: geometric interpolation from *initial* to *final*.
        - ``step_schedule``: piecewise-constant schedule defined by
          ``step_temperatures`` — a tuple of ``(start_step, temperature)``
          pairs sorted by ``start_step``.
    """

    mode: str
    initial_temperature: float
    final_temperature: float | None = None
    total_steps: int = 1000
    step_temperatures: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        valid_modes = ("constant", "linear_ramp", "exponential_anneal", "step_schedule")
        if self.mode not in valid_modes:
            raise ContractValidationError(
                f"TemperatureSchedule.mode must be one of {valid_modes}; got {self.mode!r}."
            )
        if self.initial_temperature <= 0.0:
            raise ContractValidationError(
                "initial_temperature must be strictly positive."
            )
        if self.mode in ("linear_ramp", "exponential_anneal"):
            if self.final_temperature is None:
                raise ContractValidationError(
                    f"{self.mode} requires final_temperature."
                )
            if self.final_temperature <= 0.0:
                raise ContractValidationError(
                    "final_temperature must be strictly positive."
                )
        if self.mode == "step_schedule" and not self.step_temperatures:
            raise ContractValidationError(
                "step_schedule mode requires non-empty step_temperatures."
            )
        if self.total_steps <= 0:
            raise ContractValidationError("total_steps must be positive.")

    def temperature_at_step(self, step: int) -> float:
        """Return the target temperature for the given global step."""

        if self.mode == "constant":
            return self.initial_temperature

        if self.mode == "linear_ramp":
            assert self.final_temperature is not None
            frac = min(step / self.total_steps, 1.0)
            return self.initial_temperature + (
                self.final_temperature - self.initial_temperature
            ) * frac

        if self.mode == "exponential_anneal":
            assert self.final_temperature is not None
            frac = min(step / self.total_steps, 1.0)
            # T = T_init * (T_final / T_init) ^ frac
            ratio = self.final_temperature / self.initial_temperature
            return self.initial_temperature * (ratio ** frac)

        # step_schedule: piecewise constant
        current_temp = self.initial_temperature
        for start_step, temp in self.step_temperatures:
            if step >= start_step:
                current_temp = temp
            else:
                break
        return current_temp


# ---------------------------------------------------------------------------
# Energy tracker
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EnergyTracker:
    """Accumulates energy and temperature time-series for post-run analysis."""

    _kinetic_energies: list[float] = field(default_factory=list)
    _potential_energies: list[float] = field(default_factory=list)
    _total_energies: list[float] = field(default_factory=list)
    _temperatures: list[float] = field(default_factory=list)
    _steps: list[int] = field(default_factory=list)
    _n_constraints: int = 0

    def record(self, step: int, ke: float, pe: float, temperature: float) -> None:
        """Append a single time-point to all histories."""

        self._kinetic_energies.append(ke)
        self._potential_energies.append(pe)
        self._total_energies.append(ke + pe)
        self._temperatures.append(temperature)
        self._steps.append(step)

    # ---- computed observables --------------------------------------------- #

    def kinetic_energy(self, state: SimulationState) -> float:
        """Compute classical kinetic energy: 0.5 * sum(m_i * |v_i|^2)."""

        return fsum(
            0.5 * mass * (vx * vx + vy * vy + vz * vz)
            for mass, (vx, vy, vz) in zip(
                state.particles.masses, state.particles.velocities, strict=True
            )
        )

    def instantaneous_temperature(self, state: SimulationState) -> float:
        """Return T = 2 KE / (n_dof * kB).

        Degrees of freedom: 3*N minus the number of holonomic constraints.
        """

        n_particles = state.particle_count
        n_dof = 3 * n_particles - self._n_constraints
        if n_dof <= 0:
            return 0.0
        ke = self.kinetic_energy(state)
        return (2.0 * ke) / (n_dof * BOLTZMANN_CONSTANT)

    def energy_drift(self) -> float:
        """Relative total-energy drift: (E[-1] - E[0]) / |E[0]|.

        Returns 0.0 when insufficient data or E[0] == 0.
        """

        if len(self._total_energies) < 2:
            return 0.0
        e0 = self._total_energies[0]
        if e0 == 0.0:
            return 0.0
        return (self._total_energies[-1] - e0) / abs(e0)

    def summary(self) -> dict[str, Any]:
        """Return a dictionary summarizing KE, PE, total E, and temperature."""

        def _stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            n = len(values)
            mean = fsum(values) / n
            variance = fsum((v - mean) ** 2 for v in values) / max(n, 1)
            return {
                "mean": mean,
                "std": variance ** 0.5,
                "min": min(values),
                "max": max(values),
            }

        return {
            "kinetic_energy": _stats(self._kinetic_energies),
            "potential_energy": _stats(self._potential_energies),
            "total_energy": _stats(self._total_energies),
            "temperature": _stats(self._temperatures),
            "energy_drift": self.energy_drift(),
            "samples": len(self._steps),
        }


# ---------------------------------------------------------------------------
# Production run result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProductionRunResult:
    """Structured output from a completed production simulation run."""

    initial_state: SimulationState
    final_state: SimulationState
    produced_state_ids: tuple[StateId, ...]
    steps_completed: int
    energy_summary: dict[str, Any]
    temperature_summary: dict[str, Any]
    energy_drift: float
    constraint_violations: int
    metadata: FrozenMetadata


# ---------------------------------------------------------------------------
# Production simulation loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProductionSimulationLoop:
    """Full-featured simulation loop integrating all NeuroCGMD subsystems.

    Orchestrates integrator stepping, holonomic constraint enforcement,
    periodic boundary wrapping, temperature scheduling, energy tracking,
    and user-defined analysis hooks in a single, auditable loop.
    """

    topology: SystemTopology
    forcefield: BaseForceField
    integrator: object  # anything with .step(state, topology, ff, evaluator)
    force_evaluator: object  # anything with .evaluate(state, topology, ff)
    registry: SimulationStateRegistry
    constraints: tuple[DistanceConstraint, ...] = ()
    constraint_solver: object | None = None  # SHAKESolver or LINCSolver
    temperature_schedule: TemperatureSchedule | None = None
    apply_pbc: bool = True
    energy_tracker: EnergyTracker = field(default_factory=EnergyTracker)
    analysis_hooks: list[Callable[..., Any]] = field(default_factory=list)
    analysis_interval: int = 100
    name: str = "production_simulation_loop"

    def run(
        self,
        steps: int,
        *,
        notes: str = "",
        metadata: FrozenMetadata | dict[str, object] | None = None,
    ) -> ProductionRunResult:
        """Execute *steps* of production MD with full subsystem integration.

        Parameters
        ----------
        steps:
            Number of integration steps to perform.
        notes:
            Human-readable annotation stored in every derived state.
        metadata:
            Extra metadata merged into each state's provenance.

        Returns
        -------
        ProductionRunResult
            Comprehensive result including energy statistics, constraint
            diagnostics, and the full chain of registered state ids.
        """

        if steps < 0:
            raise ContractValidationError("steps must be non-negative.")
        if len(self.registry) == 0:
            raise ContractValidationError(
                "ProductionSimulationLoop requires a registry with an "
                "initial state already registered."
            )

        # Inform the energy tracker about constraint DOF removal.
        self.energy_tracker._n_constraints = len(self.constraints)

        initial_state = self.registry.latest_state()
        current_state = initial_state
        produced_state_ids: list[StateId] = []
        constraint_violations = 0

        user_metadata = (
            metadata
            if isinstance(metadata, FrozenMetadata)
            else FrozenMetadata(metadata)
        )

        for step_i in range(steps):
            global_step = current_state.step

            # ----------------------------------------------------------
            # 1. Temperature update (if schedule is active)
            # ----------------------------------------------------------
            if self.temperature_schedule is not None:
                new_temp = self.temperature_schedule.temperature_at_step(global_step)
                new_thermo = ThermodynamicState(
                    ensemble=current_state.thermodynamics.ensemble,
                    target_temperature=new_temp,
                    friction_coefficient=current_state.thermodynamics.friction_coefficient,
                    target_pressure=current_state.thermodynamics.target_pressure,
                )
                current_state = SimulationState(
                    units=current_state.units,
                    particles=current_state.particles,
                    thermodynamics=new_thermo,
                    provenance=current_state.provenance,
                    cell=current_state.cell,
                    time=current_state.time,
                    step=current_state.step,
                    potential_energy=current_state.potential_energy,
                    observables=current_state.observables,
                )

            # ----------------------------------------------------------
            # 2. Integrator step (forces + integration)
            # ----------------------------------------------------------
            old_positions = current_state.particles.positions
            step_result: IntegratorStepResult = self.integrator.step(  # type: ignore[union-attr]
                current_state,
                self.topology,
                self.forcefield,
                self.force_evaluator,
            )
            new_particles = step_result.particles

            # ----------------------------------------------------------
            # 3. Apply holonomic constraints (SHAKE / LINCS)
            # ----------------------------------------------------------
            if self.constraint_solver is not None and self.constraints:
                constraint_result = self.constraint_solver.apply(  # type: ignore[union-attr]
                    old_positions,
                    new_particles.positions,
                    new_particles.masses,
                    self.constraints,
                    time_step=self.integrator.time_step,  # type: ignore[union-attr]
                )
                if not constraint_result.converged:
                    constraint_violations += 1
                new_particles = ParticleState(
                    positions=constraint_result.positions,
                    velocities=constraint_result.velocities,
                    forces=new_particles.forces,
                    masses=new_particles.masses,
                    labels=new_particles.labels,
                )

            # ----------------------------------------------------------
            # 4. Apply PBC wrapping
            # ----------------------------------------------------------
            if self.apply_pbc and current_state.cell is not None:
                from physics.periodic_boundary import wrap_positions

                wrapped = wrap_positions(new_particles.positions, current_state.cell)
                new_particles = ParticleState(
                    positions=wrapped,
                    velocities=new_particles.velocities,
                    forces=new_particles.forces,
                    masses=new_particles.masses,
                    labels=new_particles.labels,
                )

            # ----------------------------------------------------------
            # 5. Register the new state
            # ----------------------------------------------------------
            merged_metadata = user_metadata.to_dict()
            merged_metadata.update(step_result.metadata.to_dict())
            merged_metadata.update(
                {
                    "integrator": getattr(self.integrator, "name", "unknown"),
                    "force_evaluator": getattr(self.force_evaluator, "name", "unknown"),
                    "production_loop": self.name,
                }
            )

            current_state = self.registry.derive_state(
                current_state,
                particles=new_particles,
                time=step_result.time,
                step=step_result.step,
                potential_energy=step_result.potential_energy,
                observables=step_result.observables,
                stage=LifecycleStage.INTEGRATION,
                notes=notes,
                metadata=FrozenMetadata(merged_metadata),
            )
            produced_state_ids.append(current_state.provenance.state_id)

            # ----------------------------------------------------------
            # 6. Track energy
            # ----------------------------------------------------------
            ke = self.energy_tracker.kinetic_energy(current_state)
            pe = step_result.potential_energy or 0.0
            inst_temp = self.energy_tracker.instantaneous_temperature(current_state)
            self.energy_tracker.record(current_state.step, ke, pe, inst_temp)

            # ----------------------------------------------------------
            # 7. Analysis hooks (every analysis_interval steps)
            # ----------------------------------------------------------
            if self.analysis_hooks and step_i % self.analysis_interval == 0:
                for hook in self.analysis_hooks:
                    hook(current_state, self.energy_tracker)

        # ---- Build result ------------------------------------------------ #
        energy_summary = self.energy_tracker.summary()
        temps = self.energy_tracker._temperatures
        if temps:
            t_mean = fsum(temps) / len(temps)
            t_var = fsum((t - t_mean) ** 2 for t in temps) / max(len(temps), 1)
            temperature_summary: dict[str, Any] = {
                "mean": t_mean,
                "std": t_var ** 0.5,
                "min": min(temps),
                "max": max(temps),
            }
        else:
            temperature_summary = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return ProductionRunResult(
            initial_state=initial_state,
            final_state=current_state,
            produced_state_ids=tuple(produced_state_ids),
            steps_completed=steps,
            energy_summary=energy_summary,
            temperature_summary=temperature_summary,
            energy_drift=self.energy_tracker.energy_drift(),
            constraint_violations=constraint_violations,
            metadata=user_metadata,
        )

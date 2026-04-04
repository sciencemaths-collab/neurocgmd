"""Enhanced sampling methods: replica exchange molecular dynamics and metadynamics.

Provides established algorithms for accelerating conformational exploration
in coarse-grained molecular dynamics simulations within the NeuroCGMD platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt, log
from random import Random
from typing import Protocol, runtime_checkable

from core.exceptions import ContractValidationError
from core.state import SimulationState, ParticleState, ThermodynamicState
from core.types import FrozenMetadata, Vector3

# Boltzmann constant in kJ/(mol*K) — consistent with the default md_nano unit system.
_KB: float = 0.008314462618


# ---------------------------------------------------------------------------
# Replica Exchange Molecular Dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplicaState:
    """Immutable snapshot of a single replica within a replica-exchange ensemble."""

    replica_id: int
    temperature: float
    state: SimulationState
    potential_energy: float
    exchange_count: int = 0
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


@dataclass(slots=True)
class ReplicaExchangeManager:
    """Replica exchange (parallel tempering) manager [established].

    Maintains a temperature ladder and attempts nearest-neighbour swaps of
    *configurations* (positions and velocities) between adjacent replicas
    using the Metropolis acceptance criterion.
    """

    temperatures: tuple[float, ...]
    exchange_interval: int = 100
    random_seed: int = 42
    name: str = "replica_exchange_md"
    classification: str = "[established]"
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.temperatures) < 2:
            raise ContractValidationError(
                "ReplicaExchangeManager requires at least two temperatures."
            )
        if self.exchange_interval <= 0:
            raise ContractValidationError(
                "exchange_interval must be a positive integer."
            )
        sorted_temps = tuple(sorted(self.temperatures))
        if sorted_temps != self.temperatures:
            raise ContractValidationError(
                "temperatures must be supplied in ascending order."
            )
        if any(t <= 0.0 for t in self.temperatures):
            raise ContractValidationError(
                "All temperatures must be strictly positive."
            )
        self._rng = Random(self.random_seed)

    # -- public API ----------------------------------------------------------

    def initialize_replicas(
        self, base_state: SimulationState
    ) -> tuple[ReplicaState, ...]:
        """Create one replica per temperature, rescaling velocities accordingly.

        Velocities are rescaled from the base state temperature (taken from
        ``base_state.thermodynamics.target_temperature``) to each replica
        temperature so that kinetic energy is consistent with the target
        temperature of each replica.
        """
        base_temp = base_state.thermodynamics.target_temperature
        if base_temp is None or base_temp <= 0.0:
            raise ContractValidationError(
                "base_state must have a positive target_temperature for "
                "replica exchange initialisation."
            )

        replicas: list[ReplicaState] = []
        for idx, temp in enumerate(self.temperatures):
            scale = sqrt(temp / base_temp)
            new_velocities = tuple(
                (vx * scale, vy * scale, vz * scale)
                for vx, vy, vz in base_state.particles.velocities
            )
            new_particles = ParticleState(
                positions=base_state.particles.positions,
                masses=base_state.particles.masses,
                velocities=new_velocities,
                forces=base_state.particles.forces,
                labels=base_state.particles.labels,
            )
            new_thermo = ThermodynamicState(
                target_temperature=temp,
                target_pressure=base_state.thermodynamics.target_pressure,
                ensemble=base_state.thermodynamics.ensemble,
                friction_coefficient=base_state.thermodynamics.friction_coefficient,
            )
            new_state = SimulationState(
                units=base_state.units,
                particles=new_particles,
                thermodynamics=new_thermo,
                provenance=base_state.provenance,
                cell=base_state.cell,
                time=base_state.time,
                step=base_state.step,
                potential_energy=base_state.potential_energy,
                observables=base_state.observables,
            )
            replicas.append(
                ReplicaState(
                    replica_id=idx,
                    temperature=temp,
                    state=new_state,
                    potential_energy=(
                        base_state.potential_energy
                        if base_state.potential_energy is not None
                        else 0.0
                    ),
                    exchange_count=0,
                    metadata=FrozenMetadata({"initial_temperature": temp}),
                )
            )
        return tuple(replicas)

    def attempt_exchanges(
        self, replicas: tuple[ReplicaState, ...]
    ) -> tuple[ReplicaState, ...]:
        """Attempt pairwise swaps of adjacent replicas (Metropolis criterion).

        Configurations (positions and velocities) are exchanged between
        replicas *i* and *i+1* while the temperature assignments remain
        fixed.  The acceptance probability is::

            P_accept = min(1, exp(delta))

        where ``delta = (beta_i - beta_{i+1}) * (E_i - E_{i+1})``.
        """
        replicas_list = list(replicas)
        n_replicas = len(replicas_list)

        for i in range(n_replicas - 1):
            r_i = replicas_list[i]
            r_j = replicas_list[i + 1]

            beta_i = 1.0 / (_KB * r_i.temperature)
            beta_j = 1.0 / (_KB * r_j.temperature)
            delta = (beta_i - beta_j) * (r_i.potential_energy - r_j.potential_energy)

            accept = delta >= 0.0 or self._rng.random() < exp(delta)
            if accept:
                # Swap configurations: build new SimulationStates with swapped
                # particles but keeping original temperatures.
                new_state_i = SimulationState(
                    units=r_i.state.units,
                    particles=r_j.state.particles,
                    thermodynamics=r_i.state.thermodynamics,
                    provenance=r_i.state.provenance,
                    cell=r_i.state.cell,
                    time=r_i.state.time,
                    step=r_i.state.step,
                    potential_energy=r_j.potential_energy,
                    observables=r_i.state.observables,
                )
                new_state_j = SimulationState(
                    units=r_j.state.units,
                    particles=r_i.state.particles,
                    thermodynamics=r_j.state.thermodynamics,
                    provenance=r_j.state.provenance,
                    cell=r_j.state.cell,
                    time=r_j.state.time,
                    step=r_j.state.step,
                    potential_energy=r_i.potential_energy,
                    observables=r_j.state.observables,
                )
                replicas_list[i] = ReplicaState(
                    replica_id=r_i.replica_id,
                    temperature=r_i.temperature,
                    state=new_state_i,
                    potential_energy=r_j.potential_energy,
                    exchange_count=r_i.exchange_count + 1,
                    metadata=r_i.metadata.with_updates(
                        {"last_swap_partner": r_j.replica_id}
                    ),
                )
                replicas_list[i + 1] = ReplicaState(
                    replica_id=r_j.replica_id,
                    temperature=r_j.temperature,
                    state=new_state_j,
                    potential_energy=r_i.potential_energy,
                    exchange_count=r_j.exchange_count + 1,
                    metadata=r_j.metadata.with_updates(
                        {"last_swap_partner": r_i.replica_id}
                    ),
                )

        return tuple(replicas_list)

    def should_attempt_exchange(self, step: int) -> bool:
        """Return *True* when the current step aligns with the exchange interval."""
        return step > 0 and step % self.exchange_interval == 0


# ---------------------------------------------------------------------------
# Metadynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GaussianHill:
    """A single Gaussian bias hill deposited during metadynamics."""

    center: tuple[float, ...]
    height: float
    width: float
    deposition_step: int


@runtime_checkable
class CollectiveVariable(Protocol):
    """Protocol for collective-variable implementations."""

    name: str

    def compute(self, state: SimulationState) -> float: ...


@dataclass(frozen=True, slots=True)
class DistanceCV:
    """Collective variable: Euclidean distance between two particles."""

    particle_a: int
    particle_b: int
    name: str = "distance_cv"

    def compute(self, state: SimulationState) -> float:
        """Return the Euclidean distance between particles *a* and *b*."""
        pos_a = state.particles.positions[self.particle_a]
        pos_b = state.particles.positions[self.particle_b]
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True, slots=True)
class RadiusOfGyrationCV:
    """Collective variable: radius of gyration for a subset of particles."""

    particle_indices: tuple[int, ...]
    name: str = "radius_of_gyration_cv"

    def compute(self, state: SimulationState) -> float:
        r"""Compute :math:`R_g = \sqrt{\frac{1}{N}\sum_i |r_i - r_{com}|^2}`."""
        n = len(self.particle_indices)
        if n == 0:
            raise ContractValidationError(
                "RadiusOfGyrationCV requires at least one particle index."
            )

        positions = state.particles.positions
        # Centre of mass (unweighted — geometric centre).
        cx = sum(positions[i][0] for i in self.particle_indices) / n
        cy = sum(positions[i][1] for i in self.particle_indices) / n
        cz = sum(positions[i][2] for i in self.particle_indices) / n

        sq_sum = 0.0
        for i in self.particle_indices:
            dx = positions[i][0] - cx
            dy = positions[i][1] - cy
            dz = positions[i][2] - cz
            sq_sum += dx * dx + dy * dy + dz * dz

        return sqrt(sq_sum / n)


@dataclass(slots=True)
class MetadynamicsEngine:
    """Well-tempered metadynamics engine [established].

    Deposits Gaussian bias hills in collective-variable space to
    discourage revisiting already-explored regions and progressively
    flatten the free-energy landscape.
    """

    collective_variables: tuple[CollectiveVariable, ...]
    hill_height: float = 0.1
    hill_width: float = 0.5
    deposition_interval: int = 50
    well_tempered: bool = True
    bias_temperature: float = 3000.0
    max_hills: int = 10000
    name: str = "metadynamics_engine"
    classification: str = "[established]"
    _hills: list[GaussianHill] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.collective_variables:
            raise ContractValidationError(
                "MetadynamicsEngine requires at least one collective variable."
            )
        if self.hill_height <= 0.0:
            raise ContractValidationError("hill_height must be positive.")
        if self.hill_width <= 0.0:
            raise ContractValidationError("hill_width must be positive.")
        if self.deposition_interval <= 0:
            raise ContractValidationError(
                "deposition_interval must be a positive integer."
            )
        if self.well_tempered and self.bias_temperature <= 0.0:
            raise ContractValidationError(
                "bias_temperature must be positive for well-tempered metadynamics."
            )

    # -- helpers -------------------------------------------------------------

    def _current_cv_values(self, state: SimulationState) -> tuple[float, ...]:
        return tuple(cv.compute(state) for cv in self.collective_variables)

    @staticmethod
    def _gaussian_value(
        center: tuple[float, ...],
        point: tuple[float, ...],
        height: float,
        width: float,
    ) -> float:
        """Evaluate an un-normalised Gaussian at *point*."""
        sq_dist = sum((c - p) ** 2 for c, p in zip(center, point))
        return height * exp(-sq_dist / (2.0 * width * width))

    # -- public API ----------------------------------------------------------

    def compute_bias_energy(self, state: SimulationState) -> float:
        """Return the total bias energy at the current CV values."""
        cv_vals = self._current_cv_values(state)
        return sum(
            self._gaussian_value(hill.center, cv_vals, hill.height, hill.width)
            for hill in self._hills
        )

    def compute_bias_forces(
        self, state: SimulationState
    ) -> tuple[tuple[int, Vector3], ...]:
        """Compute the negative gradient of the bias potential w.r.t. particle positions.

        Uses the chain rule: ``F_bias = -dV_bias/dr = -sum_k (dV_bias/ds_k) * (ds_k/dr)``
        where *s_k* are the collective variables.  Numerical finite differences
        are used for the CV gradients so that the method remains agnostic to CV
        implementation details.
        """
        if not self._hills:
            return ()

        cv_vals = self._current_cv_values(state)
        positions = state.particles.positions
        n_particles = len(positions)

        # dV_bias / ds_k  for each CV k  (analytical from Gaussians).
        dv_ds: list[float] = []
        for k in range(len(self.collective_variables)):
            grad_k = 0.0
            for hill in self._hills:
                g_val = self._gaussian_value(
                    hill.center, cv_vals, hill.height, hill.width
                )
                grad_k += g_val * (
                    -(cv_vals[k] - hill.center[k]) / (hill.width * hill.width)
                )
            dv_ds.append(grad_k)

        # ds_k / dr_j  via central finite differences, then accumulate forces.
        epsilon = 1.0e-5
        forces: dict[int, list[float]] = {}

        for k, cv in enumerate(self.collective_variables):
            if abs(dv_ds[k]) < 1.0e-15:
                continue

            for j in range(n_particles):
                grad_j: list[float] = [0.0, 0.0, 0.0]
                for dim in range(3):
                    pos_fwd = list(list(v) for v in positions)
                    pos_bwd = list(list(v) for v in positions)
                    pos_fwd[j][dim] += epsilon
                    pos_bwd[j][dim] -= epsilon

                    state_fwd = SimulationState(
                        units=state.units,
                        particles=state.particles.with_positions(
                            tuple(tuple(v) for v in pos_fwd)
                        ),
                        thermodynamics=state.thermodynamics,
                        provenance=state.provenance,
                        cell=state.cell,
                        time=state.time,
                        step=state.step,
                        potential_energy=state.potential_energy,
                        observables=state.observables,
                    )
                    state_bwd = SimulationState(
                        units=state.units,
                        particles=state.particles.with_positions(
                            tuple(tuple(v) for v in pos_bwd)
                        ),
                        thermodynamics=state.thermodynamics,
                        provenance=state.provenance,
                        cell=state.cell,
                        time=state.time,
                        step=state.step,
                        potential_energy=state.potential_energy,
                        observables=state.observables,
                    )

                    cv_fwd = cv.compute(state_fwd)
                    cv_bwd = cv.compute(state_bwd)
                    grad_j[dim] = (cv_fwd - cv_bwd) / (2.0 * epsilon)

                # F_j += -dV_bias/ds_k * ds_k/dr_j
                fx = -dv_ds[k] * grad_j[0]
                fy = -dv_ds[k] * grad_j[1]
                fz = -dv_ds[k] * grad_j[2]

                if j in forces:
                    forces[j][0] += fx
                    forces[j][1] += fy
                    forces[j][2] += fz
                else:
                    forces[j] = [fx, fy, fz]

        # Filter out particles with negligible force and return.
        threshold = 1.0e-14
        result: list[tuple[int, Vector3]] = []
        for j in sorted(forces):
            fx, fy, fz = forces[j]
            if abs(fx) > threshold or abs(fy) > threshold or abs(fz) > threshold:
                result.append((j, (fx, fy, fz)))

        return tuple(result)

    def deposit_hill(self, state: SimulationState) -> GaussianHill | None:
        """Deposit a Gaussian hill at the current CV values.

        For well-tempered metadynamics the hill height is reduced by
        ``exp(-V_bias(s) / (kB * delta_T))`` where
        ``delta_T = bias_temperature - T_sim``.

        Returns the deposited hill, or *None* if the maximum number of hills
        has been reached.
        """
        if len(self._hills) >= self.max_hills:
            return None

        cv_vals = self._current_cv_values(state)
        height = self.hill_height

        if self.well_tempered:
            current_bias = self.compute_bias_energy(state)
            sim_temp = state.thermodynamics.target_temperature or 300.0
            delta_t = self.bias_temperature - sim_temp
            if delta_t > 0.0:
                height *= exp(-current_bias / (_KB * delta_t))

        hill = GaussianHill(
            center=cv_vals,
            height=height,
            width=self.hill_width,
            deposition_step=state.step,
        )
        self._hills.append(hill)
        return hill

    def should_deposit(self, step: int) -> bool:
        """Return *True* when the current step aligns with the deposition interval."""
        return step > 0 and step % self.deposition_interval == 0

    def reconstruct_fes(self) -> tuple[tuple[float, ...], ...]:
        """Return the negative of accumulated bias as a free-energy surface estimate.

        Each entry is ``(cv_center_0, cv_center_1, ..., -V_bias)`` evaluated
        at every hill deposition centre, giving a discrete reconstruction of
        the free-energy surface along the collective-variable path.
        """
        result: list[tuple[float, ...]] = []
        for target_hill in self._hills:
            bias_at_center = sum(
                self._gaussian_value(
                    hill.center, target_hill.center, hill.height, hill.width
                )
                for hill in self._hills
            )
            result.append((*target_hill.center, -bias_at_center))
        return tuple(result)

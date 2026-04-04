"""Established nonbonded force terms for the baseline coarse-grained substrate."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.types import Vector3, VectorTuple
from forcefields.base_forcefield import BaseForceField
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class NonbondedForceReport:
    """Vector force result for the nonbonded pairwise model."""

    forces: VectorTuple
    evaluated_pairs: tuple[tuple[int, int], ...]


class LennardJonesNonbondedForceModel:
    """Evaluate Lennard-Jones pair forces with optional bonded-pair exclusion."""

    name = "lennard_jones_nonbonded_force"
    classification = "[established]"

    def __init__(self, *, exclude_bonded_pairs: bool = True) -> None:
        self.exclude_bonded_pairs = exclude_bonded_pairs

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> NonbondedForceReport:
        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        forces = [[0.0, 0.0, 0.0] for _ in range(state.particle_count)]
        evaluated_pairs: list[tuple[int, int]] = []
        excluded_pairs = {bond.normalized_pair() for bond in topology.bonds} if self.exclude_bonded_pairs else set()

        for index_a in range(state.particle_count):
            for index_b in range(index_a + 1, state.particle_count):
                pair = (index_a, index_b)
                if pair in excluded_pairs:
                    continue
                parameter = forcefield.nonbonded_parameter_for_pair(topology, index_a, index_b)
                force_vector = self._pair_force_vector(
                    state.particles.positions[index_a],
                    state.particles.positions[index_b],
                    sigma=parameter.sigma,
                    epsilon=parameter.epsilon,
                    cutoff=parameter.cutoff,
                )
                if force_vector is None:
                    continue
                for axis, value in enumerate(force_vector):
                    forces[index_a][axis] += value
                    forces[index_b][axis] -= value
                evaluated_pairs.append(pair)

        return NonbondedForceReport(
            forces=tuple(tuple(vector) for vector in forces),
            evaluated_pairs=tuple(evaluated_pairs),
        )

    @staticmethod
    def _pair_force_vector(
        position_a: Vector3,
        position_b: Vector3,
        *,
        sigma: float,
        epsilon: float,
        cutoff: float,
    ) -> Vector3 | None:
        displacement = tuple(position_b[axis] - position_a[axis] for axis in range(3))
        distance_squared = sum(component * component for component in displacement)
        if distance_squared == 0.0:
            raise ContractValidationError(
                "Nonbonded pair has zero separation; Lennard-Jones force is undefined."
            )
        distance = sqrt(distance_squared)
        if distance > cutoff:
            return None

        scaled_distance = sigma / distance
        scaled_distance_6 = scaled_distance**6
        scaled_distance_12 = scaled_distance_6 * scaled_distance_6
        coefficient = 24.0 * epsilon * (scaled_distance_6 - 2.0 * scaled_distance_12) / distance_squared
        return tuple(coefficient * component for component in displacement)


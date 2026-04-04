"""Established bonded force terms for the baseline coarse-grained substrate."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from core.exceptions import ContractValidationError
from core.state import SimulationState
from core.types import Vector3, VectorTuple
from forcefields.base_forcefield import BaseForceField
from topology.bonds import Bond
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class BondForceReport:
    """Vector force result for the harmonic bonded model."""

    forces: VectorTuple
    evaluated_bonds: tuple[tuple[int, int], ...]


class HarmonicBondForceModel:
    """Evaluate forces for the harmonic bond energy `0.5 * k * (r - r0)^2`."""

    name = "harmonic_bond_force"
    classification = "[established]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> BondForceReport:
        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        forces = [[0.0, 0.0, 0.0] for _ in range(state.particle_count)]
        evaluated_bonds: list[tuple[int, int]] = []

        for bond in topology.bonds:
            parameter = forcefield.bond_parameter_for(topology, bond)
            equilibrium_distance = (
                bond.equilibrium_distance
                if bond.equilibrium_distance is not None
                else parameter.equilibrium_distance
            )
            stiffness = bond.stiffness if bond.stiffness is not None else parameter.stiffness
            pair_force = self._pair_force_vector(
                state.particles.positions[bond.particle_index_a],
                state.particles.positions[bond.particle_index_b],
                equilibrium_distance=equilibrium_distance,
                stiffness=stiffness,
            )
            for axis, value in enumerate(pair_force):
                forces[bond.particle_index_a][axis] += value
                forces[bond.particle_index_b][axis] -= value
            evaluated_bonds.append(bond.normalized_pair())

        return BondForceReport(
            forces=tuple(tuple(vector) for vector in forces),
            evaluated_bonds=tuple(evaluated_bonds),
        )

    @staticmethod
    def _pair_force_vector(
        position_a: Vector3,
        position_b: Vector3,
        *,
        equilibrium_distance: float,
        stiffness: float,
    ) -> Vector3:
        displacement = tuple(position_b[axis] - position_a[axis] for axis in range(3))
        distance_squared = sum(component * component for component in displacement)
        if distance_squared == 0.0:
            raise ContractValidationError(
                "Bonded pair has zero separation; harmonic direction is undefined."
            )
        distance = sqrt(distance_squared)
        coefficient = stiffness * (distance - equilibrium_distance) / distance
        return tuple(coefficient * component for component in displacement)


"""Established bonded energy terms for the baseline coarse-grained substrate."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, fsum

from core.exceptions import ContractValidationError
from core.state import SimulationState
from forcefields.base_forcefield import BaseForceField
from topology.bonds import Bond
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class BondEnergyRecord:
    """Per-bond harmonic energy evaluation record."""

    particle_pair: tuple[int, int]
    kind: str
    distance: float
    equilibrium_distance: float
    stiffness: float
    energy: float


@dataclass(frozen=True, slots=True)
class BondedEnergyReport:
    """Collection of per-bond energies plus their sum."""

    records: tuple[BondEnergyRecord, ...]

    @property
    def total_energy(self) -> float:
        return fsum(record.energy for record in self.records)


class HarmonicBondEnergyModel:
    """Evaluate harmonic bond energy `0.5 * k * (r - r0)^2`."""

    name = "harmonic_bond_energy"
    classification = "[established]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> BondedEnergyReport:
        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        records: list[BondEnergyRecord] = []
        for bond in topology.bonds:
            parameter = forcefield.bond_parameter_for(topology, bond)
            equilibrium_distance = (
                bond.equilibrium_distance
                if bond.equilibrium_distance is not None
                else parameter.equilibrium_distance
            )
            stiffness = bond.stiffness if bond.stiffness is not None else parameter.stiffness
            records.append(
                self._evaluate_bond(state, bond, equilibrium_distance, stiffness)
            )
        return BondedEnergyReport(records=tuple(records))

    def _evaluate_bond(
        self,
        state: SimulationState,
        bond: Bond,
        equilibrium_distance: float,
        stiffness: float,
    ) -> BondEnergyRecord:
        pair = bond.normalized_pair()
        position_a = state.particles.positions[pair[0]]
        position_b = state.particles.positions[pair[1]]
        distance = dist(position_a, position_b)
        displacement = distance - equilibrium_distance
        energy = 0.5 * stiffness * displacement * displacement
        return BondEnergyRecord(
            particle_pair=pair,
            kind=bond.kind.value,
            distance=distance,
            equilibrium_distance=equilibrium_distance,
            stiffness=stiffness,
            energy=energy,
        )


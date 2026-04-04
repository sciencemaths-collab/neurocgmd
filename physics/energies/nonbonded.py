"""Established nonbonded energy terms for the baseline coarse-grained substrate."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, fsum

from core.exceptions import ContractValidationError
from core.state import SimulationState
from forcefields.base_forcefield import BaseForceField
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class NonbondedEnergyRecord:
    """Per-pair Lennard-Jones energy evaluation record."""

    particle_pair: tuple[int, int]
    distance: float
    sigma: float
    epsilon: float
    cutoff: float
    energy: float


@dataclass(frozen=True, slots=True)
class NonbondedEnergyReport:
    """Collection of evaluated nonbonded pairs plus their sum."""

    records: tuple[NonbondedEnergyRecord, ...]

    @property
    def total_energy(self) -> float:
        return fsum(record.energy for record in self.records)


class LennardJonesNonbondedEnergyModel:
    """Evaluate the established 12-6 Lennard-Jones nonbonded interaction."""

    name = "lennard_jones_nonbonded_energy"
    classification = "[established]"

    def __init__(self, *, exclude_bonded_pairs: bool = True) -> None:
        self.exclude_bonded_pairs = exclude_bonded_pairs

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> NonbondedEnergyReport:
        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        excluded_pairs = {bond.normalized_pair() for bond in topology.bonds} if self.exclude_bonded_pairs else set()
        records: list[NonbondedEnergyRecord] = []
        for index_a in range(state.particle_count):
            for index_b in range(index_a + 1, state.particle_count):
                pair = (index_a, index_b)
                if pair in excluded_pairs:
                    continue
                parameter = forcefield.nonbonded_parameter_for_pair(topology, index_a, index_b)
                distance = dist(
                    state.particles.positions[index_a],
                    state.particles.positions[index_b],
                )
                if distance == 0.0:
                    raise ContractValidationError(
                        f"Nonbonded pair {pair} has zero separation; Lennard-Jones is undefined."
                    )
                if distance > parameter.cutoff:
                    continue
                records.append(
                    NonbondedEnergyRecord(
                        particle_pair=pair,
                        distance=distance,
                        sigma=parameter.sigma,
                        epsilon=parameter.epsilon,
                        cutoff=parameter.cutoff,
                        energy=self._pair_energy(
                            sigma=parameter.sigma,
                            epsilon=parameter.epsilon,
                            distance=distance,
                        ),
                    )
                )
        return NonbondedEnergyReport(records=tuple(records))

    @staticmethod
    def _pair_energy(*, sigma: float, epsilon: float, distance: float) -> float:
        scaled_distance = sigma / distance
        scaled_distance_6 = scaled_distance**6
        scaled_distance_12 = scaled_distance_6 * scaled_distance_6
        return 4.0 * epsilon * (scaled_distance_12 - scaled_distance_6)


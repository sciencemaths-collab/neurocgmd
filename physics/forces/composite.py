"""Composite force evaluation that combines the current baseline physics terms."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.state import SimulationState
from core.types import FrozenMetadata, VectorTuple
from forcefields.base_forcefield import BaseForceField
from physics.energies.bonded import HarmonicBondEnergyModel
from physics.energies.nonbonded import LennardJonesNonbondedEnergyModel
from physics.forces.bonded_forces import HarmonicBondForceModel
from physics.forces.nonbonded_forces import LennardJonesNonbondedForceModel
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class ForceEvaluation:
    """Combined force block plus scalar potential-energy accounting."""

    forces: VectorTuple
    potential_energy: float
    component_energies: FrozenMetadata = field(default_factory=FrozenMetadata)
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


class BaselineForceEvaluator:
    """Compose the current established bonded and nonbonded force terms."""

    name = "baseline_force_evaluator"
    classification = "[adapted]"

    def __init__(
        self,
        *,
        bonded_energy_model: HarmonicBondEnergyModel | None = None,
        nonbonded_energy_model: LennardJonesNonbondedEnergyModel | None = None,
        bonded_force_model: HarmonicBondForceModel | None = None,
        nonbonded_force_model: LennardJonesNonbondedForceModel | None = None,
    ) -> None:
        self.bonded_energy_model = bonded_energy_model or HarmonicBondEnergyModel()
        self.nonbonded_energy_model = nonbonded_energy_model or LennardJonesNonbondedEnergyModel()
        self.bonded_force_model = bonded_force_model or HarmonicBondForceModel()
        self.nonbonded_force_model = nonbonded_force_model or LennardJonesNonbondedForceModel()

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        forcefield: BaseForceField,
    ) -> ForceEvaluation:
        bonded_energy = self.bonded_energy_model.evaluate(state, topology, forcefield)
        nonbonded_energy = self.nonbonded_energy_model.evaluate(state, topology, forcefield)
        bonded_forces = self.bonded_force_model.evaluate(state, topology, forcefield)
        nonbonded_forces = self.nonbonded_force_model.evaluate(state, topology, forcefield)

        total_forces = _sum_force_blocks(
            bonded_forces.forces,
            nonbonded_forces.forces,
        )
        component_energies = FrozenMetadata(
            {
                "bonded": bonded_energy.total_energy,
                "nonbonded": nonbonded_energy.total_energy,
            }
        )
        metadata = FrozenMetadata(
            {
                "evaluated_bonds": len(bonded_forces.evaluated_bonds),
                "evaluated_nonbonded_pairs": len(nonbonded_forces.evaluated_pairs),
            }
        )
        return ForceEvaluation(
            forces=total_forces,
            potential_energy=bonded_energy.total_energy + nonbonded_energy.total_energy,
            component_energies=component_energies,
            metadata=metadata,
        )


def _sum_force_blocks(*force_blocks: VectorTuple) -> VectorTuple:
    if not force_blocks:
        return ()
    particle_count = len(force_blocks[0])
    summed = [[0.0, 0.0, 0.0] for _ in range(particle_count)]
    for block in force_blocks:
        if len(block) != particle_count:
            raise ValueError("All force blocks must have the same particle count.")
        for particle_index, vector in enumerate(block):
            for axis, value in enumerate(vector):
                summed[particle_index][axis] += value
    return tuple(tuple(vector) for vector in summed)


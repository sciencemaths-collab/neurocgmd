"""Tests for Section 4 force-field parameters and baseline physics terms."""

from __future__ import annotations

import unittest

from core.exceptions import ContractValidationError
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from physics.energies.bonded import HarmonicBondEnergyModel
from physics.energies.nonbonded import LennardJonesNonbondedEnergyModel
from physics.forces.nonbonded_forces import LennardJonesNonbondedForceModel
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class ForceFieldFoundationTests(unittest.TestCase):
    """Verify parameter lookup and baseline energy/force evaluation."""

    def _build_state_topology_forcefield(self) -> tuple[SimulationState, SystemTopology, BaseForceField]:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.4, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-force"),
                state_id=StateId("state-force"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
        )
        topology = SystemTopology(
            system_id="force-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="sc", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1"),
                Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="sc", label="S0"),
            ),
            bonds=(Bond(0, 1),),
        )
        forcefield = BaseForceField(
            name="baseline-cg",
            bond_parameters=(
                BondParameter("bb", "bb", equilibrium_distance=1.2, stiffness=100.0),
            ),
            nonbonded_parameters=(
                NonbondedParameter("bb", "sc", sigma=1.0, epsilon=0.5, cutoff=3.0),
                NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=3.0),
                NonbondedParameter("sc", "sc", sigma=1.1, epsilon=0.4, cutoff=3.0),
            ),
        )
        return state, topology, forcefield

    def test_forcefield_lookup_is_symmetric(self) -> None:
        _, topology, forcefield = self._build_state_topology_forcefield()
        parameter = forcefield.nonbonded_parameter_for_bead_types("sc", "bb")
        self.assertEqual(parameter.sigma, 1.0)
        self.assertEqual(forcefield.bond_parameter_for(topology, topology.bonds[0]).stiffness, 100.0)

    def test_duplicate_parameter_keys_are_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            BaseForceField(
                name="bad",
                nonbonded_parameters=(
                    NonbondedParameter("bb", "sc", sigma=1.0, epsilon=0.5, cutoff=2.5),
                    NonbondedParameter("sc", "bb", sigma=1.2, epsilon=0.6, cutoff=2.5),
                ),
            )

    def test_harmonic_bond_energy_is_zero_at_equilibrium(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        report = HarmonicBondEnergyModel().evaluate(state, topology, forcefield)
        self.assertAlmostEqual(report.total_energy, 0.0)

    def test_harmonic_bond_energy_is_positive_off_equilibrium(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        stretched_state = SimulationState(
            units=state.units,
            particles=state.particles.with_positions(
                ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.4, 0.0, 0.0))
            ),
            thermodynamics=state.thermodynamics,
            provenance=state.provenance,
        )
        report = HarmonicBondEnergyModel().evaluate(stretched_state, topology, forcefield)
        self.assertGreater(report.total_energy, 0.0)

    def test_lennard_jones_energy_and_forces_are_consistent(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        energy_report = LennardJonesNonbondedEnergyModel().evaluate(state, topology, forcefield)
        force_report = LennardJonesNonbondedForceModel().evaluate(state, topology, forcefield)

        self.assertEqual(len(energy_report.records), 2)
        self.assertEqual(len(force_report.forces), state.particle_count)
        self.assertEqual(force_report.evaluated_pairs, ((0, 2), (1, 2)))
        total_force_x = sum(vector[0] for vector in force_report.forces)
        total_force_y = sum(vector[1] for vector in force_report.forces)
        total_force_z = sum(vector[2] for vector in force_report.forces)
        self.assertAlmostEqual(total_force_x, 0.0)
        self.assertAlmostEqual(total_force_y, 0.0)
        self.assertAlmostEqual(total_force_z, 0.0)


if __name__ == "__main__":
    unittest.main()

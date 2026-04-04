"""Tests for Section 5 integrators, composite forces, and simulation-loop orchestration."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.state_registry import SimulationStateRegistry
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField, BondParameter, NonbondedParameter
from integrators.langevin import LangevinIntegrator
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from sampling.simulation_loop import SimulationLoop
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class _ZeroForceEvaluator:
    name = "zero_force_evaluator"
    classification = "[test]"

    def evaluate(self, state: SimulationState, topology: SystemTopology, forcefield: BaseForceField) -> ForceEvaluation:
        del topology, forcefield
        return ForceEvaluation(
            forces=tuple((0.0, 0.0, 0.0) for _ in range(state.particle_count)),
            potential_energy=0.0,
        )


class SimulationLoopTests(unittest.TestCase):
    """Verify the Section 5 MD stepping substrate."""

    def _build_baseline_system(self) -> tuple[SimulationState, SystemTopology, BaseForceField]:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (2.4, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0),
                velocities=((0.0, 0.0, 0.0),) * 3,
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-run"),
                state_id=StateId("state-run"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
        )
        topology = SystemTopology(
            system_id="run-system",
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
            bond_parameters=(BondParameter("bb", "bb", equilibrium_distance=1.2, stiffness=50.0),),
            nonbonded_parameters=(
                NonbondedParameter("bb", "sc", sigma=1.0, epsilon=0.5, cutoff=3.0),
                NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=3.0),
                NonbondedParameter("sc", "sc", sigma=1.1, epsilon=0.4, cutoff=3.0),
            ),
        )
        return state, topology, forcefield

    def test_baseline_force_evaluator_combines_current_terms(self) -> None:
        state, topology, forcefield = self._build_baseline_system()
        evaluation = BaselineForceEvaluator().evaluate(state, topology, forcefield)

        self.assertEqual(len(evaluation.forces), state.particle_count)
        self.assertIn("bonded", evaluation.component_energies)
        self.assertIn("nonbonded", evaluation.component_energies)
        self.assertAlmostEqual(
            evaluation.potential_energy,
            evaluation.component_energies["bonded"] + evaluation.component_energies["nonbonded"],
        )

    def test_langevin_integrator_advances_deterministically_without_force(self) -> None:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0),),
                masses=(1.0,),
                velocities=((1.0, 0.0, 0.0),),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-det"),
                state_id=StateId("state-det"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
        )
        topology = SystemTopology(
            system_id="det-system",
            bead_types=(BeadType(name="bb"),),
            beads=(Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),),
        )
        forcefield = BaseForceField(
            name="det-ff",
            nonbonded_parameters=(NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.0, cutoff=1.0),),
        )
        result = LangevinIntegrator(time_step=0.1).step(
            state,
            topology,
            forcefield,
            _ZeroForceEvaluator(),
        )

        self.assertEqual(result.time, 0.1)
        self.assertEqual(result.step, 1)
        self.assertEqual(result.particles.positions, ((0.1, 0.0, 0.0),))
        self.assertEqual(result.particles.velocities, ((1.0, 0.0, 0.0),))
        self.assertEqual(result.potential_energy, 0.0)

    def test_simulation_loop_records_lineage_and_updates_state(self) -> None:
        registry = SimulationStateRegistry(created_by="unit-test")
        initial_state = registry.create_initial_state(
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0),),
                masses=(1.0,),
                velocities=((1.0, 0.0, 0.0),),
            ),
            thermodynamics=ThermodynamicState(),
        )
        topology = SystemTopology(
            system_id="loop-system",
            bead_types=(BeadType(name="bb"),),
            beads=(Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),),
        )
        forcefield = BaseForceField(
            name="loop-ff",
            nonbonded_parameters=(NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.0, cutoff=1.0),),
        )
        loop = SimulationLoop(
            topology=topology,
            forcefield=forcefield,
            integrator=LangevinIntegrator(time_step=0.1),
            force_evaluator=_ZeroForceEvaluator(),
            registry=registry,
        )

        run = loop.run(3, notes="deterministic test")
        self.assertEqual(run.steps_completed, 3)
        self.assertEqual(len(run.produced_state_ids), 3)
        self.assertEqual(len(registry), 4)
        self.assertEqual(run.initial_state.provenance.state_id, initial_state.provenance.state_id)
        self.assertEqual(run.final_state.time, 0.30000000000000004)
        self.assertEqual(run.final_state.step, 3)
        self.assertEqual(run.final_state.particles.positions, ((0.30000000000000004, 0.0, 0.0),))
        self.assertEqual(
            registry.lineage_for(run.final_state.provenance.state_id),
            registry.state_ids(),
        )


if __name__ == "__main__":
    unittest.main()

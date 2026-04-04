"""Tests for backend parity validation and backend execution planning."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields import BaseForceField, BondParameter, NonbondedParameter
from forcefields.hybrid_engine import HybridForceEngine
from optimization import AccelerationBackend, BackendExecutionPlanner, BackendExecutionRequest, BackendRegistry
from physics.forces.composite import BaselineForceEvaluator, ForceEvaluation
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology
from validation.backend_parity import BackendParityValidator


class _BadEvaluator:
    name = "bad_evaluator"
    classification = "[test]"

    def evaluate(self, state: SimulationState, topology: SystemTopology, forcefield: BaseForceField) -> ForceEvaluation:
        del topology, forcefield
        return ForceEvaluation(
            forces=tuple((0.0, 0.0, 0.0) for _ in range(state.particle_count)),
            potential_energy=0.0,
        )


class BackendParityAndExecutionTests(unittest.TestCase):
    """Verify backend parity reports and execution planning."""

    def _build_state_topology_forcefield(self):
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.1, 0.0, 0.0)),
                masses=(1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-parity"),
                state_id=StateId("state-parity"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.0,
            step=0,
            potential_energy=0.0,
        )
        topology = SystemTopology(
            system_id="parity-topology",
            bead_types=(BeadType(name="bb", role=BeadRole.STRUCTURAL),),
            beads=(
                Bead(BeadId("b0"), 0, "bb", label="B0"),
                Bead(BeadId("b1"), 1, "bb", label="B1"),
            ),
            bonds=(Bond(0, 1),),
        )
        forcefield = BaseForceField(
            name="parity-forcefield",
            bond_parameters=(BondParameter("bb", "bb", equilibrium_distance=1.0, stiffness=20.0),),
            nonbonded_parameters=(NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=3.0),),
        )
        return state, topology, forcefield

    def test_backend_parity_passes_for_hybrid_classical_path(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        report = BackendParityValidator().compare_hybrid_classical(
            state=state,
            topology=topology,
            forcefield=forcefield,
            reference_provider=BaselineForceEvaluator(),
            hybrid_engine=HybridForceEngine(),
        )

        self.assertTrue(report.all_passed())

    def test_backend_parity_flags_bad_candidate(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        report = BackendParityValidator(energy_tolerance=1e-12, force_rms_tolerance=1e-12).compare_providers(
            state=state,
            topology=topology,
            forcefield=forcefield,
            reference_provider=BaselineForceEvaluator(),
            candidate_provider=_BadEvaluator(),
            target_component="tests/bad_candidate",
            backend_name="bad_backend",
        )

        self.assertFalse(report.all_passed())

    def test_backend_execution_planner_prefers_vectorized_backend_for_large_workload(self) -> None:
        registry = BackendRegistry(
            backends=(
                AccelerationBackend(
                    name="reference_cpu_backend",
                    execution_model="python_loops",
                    supported_components=("physics/kernels",),
                    capabilities=("cpu", "neighbor_list", "pairwise", "tensor"),
                    available=True,
                    priority=1,
                ),
                AccelerationBackend(
                    name="vectorized_cpu_backend",
                    execution_model="vectorized_cpu",
                    supported_components=("physics/kernels",),
                    capabilities=("cpu", "neighbor_list", "pairwise", "tensor", "vectorized"),
                    available=True,
                    priority=2,
                ),
            )
        )
        plan = BackendExecutionPlanner(backend_registry=registry).plan(
            BackendExecutionRequest(
                target_component="physics/kernels/nonbonded.py",
                particle_count=512,
                pair_count=4096,
            )
        )

        self.assertEqual(plan.selection.selected_backend, "vectorized_cpu_backend")
        self.assertEqual(plan.execution_mode, "vectorized_cpu")
        self.assertGreaterEqual(plan.partition.vector_width, 1)


if __name__ == "__main__":
    unittest.main()

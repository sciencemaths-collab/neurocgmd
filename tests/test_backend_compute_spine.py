"""Tests for the backend compute spine and hybrid force engine."""

from __future__ import annotations

import unittest

from compartments import CompartmentDomain, CompartmentRegistry
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields import BaseForceField, BondParameter, NonbondedParameter
from forcefields.hybrid_engine import HybridForceEngine
from graph import ConnectivityGraphManager
from memory import TraceRecord
from ml import ResidualMemoryModel, ResidualTarget
from physics.backends import KernelDispatchBoundary, KernelDispatchRequest, ReferenceComputeBackend
from physics.forces.composite import BaselineForceEvaluator
from qcloud import (
    LocalRegionSelector,
    ParticleForceDelta,
    QCloudCorrection,
    RegionSelectionPolicy,
    RefinementRegion,
    RegionTriggerKind,
)
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class _SimpleCorrectionModel:
    name = "simple_correction_model"
    classification = "[test]"

    def evaluate(self, state: SimulationState, topology: SystemTopology, region: RefinementRegion) -> QCloudCorrection:
        del state, topology
        anchor = region.particle_indices[0]
        return QCloudCorrection(
            region_id=region.region_id,
            method_label=self.name,
            energy_delta=0.25,
            force_deltas=(ParticleForceDelta(anchor, (0.2, 0.0, 0.0)),),
            confidence=0.8,
        )


class BackendComputeSpineTests(unittest.TestCase):
    """Verify backend dispatch, kernels, and hybrid composition stay coherent."""

    def _build_state_topology_forcefield(self):
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.1, 0.0, 0.0), (2.4, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0),
                velocities=((0.0, 0.0, 0.0),) * 3,
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-backend-spine"),
                state_id=StateId("state-backend-spine"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.0,
            step=0,
            potential_energy=0.0,
        )
        topology = SystemTopology(
            system_id="backend-spine-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(BeadId("b0"), 0, "bb", label="B0"),
                Bead(BeadId("b1"), 1, "bb", label="B1"),
                Bead(BeadId("b2"), 2, "site", label="S0"),
            ),
            bonds=(Bond(0, 1),),
        )
        forcefield = BaseForceField(
            name="backend-spine-forcefield",
            bond_parameters=(BondParameter("bb", "bb", equilibrium_distance=1.0, stiffness=20.0),),
            nonbonded_parameters=(
                NonbondedParameter("bb", "bb", sigma=1.0, epsilon=0.2, cutoff=3.2),
                NonbondedParameter("bb", "site", sigma=1.0, epsilon=0.3, cutoff=3.2),
                NonbondedParameter("site", "site", sigma=1.1, epsilon=0.1, cutoff=3.2),
            ),
        )
        return state, topology, forcefield

    def test_reference_backend_dispatch_resolves(self) -> None:
        boundary = KernelDispatchBoundary()
        dispatch, backend = boundary.resolve(
            KernelDispatchRequest(
                target_component="physics/kernels/nonbonded.py",
                required_capabilities=("neighbor_list", "pairwise"),
            )
        )

        self.assertEqual(dispatch.backend_name, "reference_cpu_backend")
        self.assertIsInstance(backend, ReferenceComputeBackend)

    def test_hybrid_engine_classical_path_matches_baseline_evaluator(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        reference = BaselineForceEvaluator().evaluate(state, topology, forcefield)
        result = HybridForceEngine().evaluate_detailed(state, topology, forcefield)

        self.assertAlmostEqual(result.classical_evaluation.potential_energy, reference.potential_energy, places=9)
        for left, right in zip(result.classical_evaluation.forces, reference.forces):
            for axis in range(3):
                self.assertAlmostEqual(left[axis], right[axis], places=9)

    def test_hybrid_engine_composes_qcloud_and_residual_layers(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        region = RefinementRegion(
            region_id="region-hybrid",
            state_id=state.provenance.state_id,
            particle_indices=(0, 2),
            trigger_kinds=(RegionTriggerKind.MANUAL,),
            score=1.0,
        )
        residual_model = ResidualMemoryModel()
        residual_model.observe(
            ResidualTarget(
                state_id=state.provenance.state_id,
                energy_delta=0.4,
                force_deltas=(ParticleForceDelta(2, (0.0, 0.1, 0.0)),),
                source_label="unit-test",
            ),
            sample_weight=2.0,
        )
        result = HybridForceEngine().evaluate_detailed(
            state,
            topology,
            forcefield,
            selected_regions=(region,),
            correction_model=_SimpleCorrectionModel(),
            residual_model=residual_model,
        )

        self.assertIsNotNone(result.qcloud_result)
        self.assertIsNotNone(result.residual_prediction)
        self.assertGreater(result.final_evaluation.potential_energy, result.classical_evaluation.potential_energy)
        self.assertIn("ml_residual", result.final_evaluation.component_energies)

    def test_hybrid_engine_selector_path_builds_qcloud_result(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        graph = ConnectivityGraphManager().initialize(state, topology)
        trace_record = TraceRecord(
            record_id="trace-backend-hybrid",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=None,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            tags=("priority",),
        )
        compartments = CompartmentRegistry(
            particle_count=state.particle_count,
            domains=(
                CompartmentDomain.from_members("A", "left", (0, 1)),
                CompartmentDomain.from_members("B", "right", (2,)),
            ),
        )
        result = HybridForceEngine().evaluate_detailed(
            state,
            topology,
            forcefield,
            graph=graph,
            compartments=compartments,
            trace_record=trace_record,
            region_selector=LocalRegionSelector(
                policy=RegionSelectionPolicy(max_regions=1, max_region_size=3, min_region_score=0.0)
            ),
            correction_model=_SimpleCorrectionModel(),
        )

        self.assertIsNotNone(result.qcloud_result)
        self.assertGreaterEqual(len(result.qcloud_result.selected_regions), 1)

    def test_hybrid_engine_honors_per_bond_overrides(self) -> None:
        state, topology, forcefield = self._build_state_topology_forcefield()
        topology = SystemTopology(
            system_id=topology.system_id,
            bead_types=topology.bead_types,
            beads=topology.beads,
            bonds=(Bond(0, 1, equilibrium_distance=1.1, stiffness=35.0),),
        )

        reference = BaselineForceEvaluator().evaluate(state, topology, forcefield)
        result = HybridForceEngine().evaluate_detailed(state, topology, forcefield)

        self.assertAlmostEqual(result.classical_evaluation.potential_energy, reference.potential_energy, places=9)
        for left, right in zip(result.classical_evaluation.forces, reference.forces):
            for axis in range(3):
                self.assertAlmostEqual(left[axis], right[axis], places=9)


if __name__ == "__main__":
    unittest.main()

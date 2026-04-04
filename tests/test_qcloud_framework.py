"""Tests for Section 10 qcloud region selection and bounded coupling."""

from __future__ import annotations

import unittest

from compartments import CompartmentDomain, CompartmentRegistry
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from forcefields.base_forcefield import BaseForceField
from graph import ConnectivityGraph, DynamicEdgeKind, DynamicEdgeState
from memory import TraceRecord
from physics.forces.composite import ForceEvaluation
from qcloud import (
    LocalRegionSelector,
    ParticleForceDelta,
    QCloudCorrection,
    QCloudForceCoupler,
    RegionSelectionPolicy,
    RegionTriggerKind,
    RefinementRegion,
)
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


class _BoundedCorrectionModel:
    name = "bounded_correction_model"
    classification = "[test]"

    def __init__(self, energy_delta: float, force_map: dict[int, tuple[float, float, float]]) -> None:
        self.energy_delta = energy_delta
        self.force_map = force_map

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> QCloudCorrection:
        del state, topology
        return QCloudCorrection(
            region_id=region.region_id,
            method_label=self.name,
            energy_delta=self.energy_delta,
            force_deltas=tuple(
                ParticleForceDelta(particle_index=particle_index, delta_force=vector)
                for particle_index, vector in self.force_map.items()
            ),
            confidence=0.8,
        )


class _SelectorDrivenCorrectionModel:
    name = "selector_driven_correction_model"
    classification = "[test]"

    def evaluate(
        self,
        state: SimulationState,
        topology: SystemTopology,
        region: RefinementRegion,
    ) -> QCloudCorrection:
        del state, topology
        anchor = region.particle_indices[0]
        return QCloudCorrection(
            region_id=region.region_id,
            method_label=self.name,
            energy_delta=0.2,
            force_deltas=(ParticleForceDelta(particle_index=anchor, delta_force=(0.5, 0.0, 0.0)),),
            confidence=0.9,
        )


class QCloudFrameworkTests(unittest.TestCase):
    """Verify qcloud selection and coupling remain bounded and traceable."""

    def _build_state(self) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.4, 0.0, 0.0), (3.4, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-qcloud"),
                state_id=StateId("state-qcloud"),
                parent_state_id=None,
                created_by="unit-test",
                stage="checkpoint",
            ),
            time=0.1,
            step=1,
            potential_energy=-1.0,
        )

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="qcloud-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0"),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1"),
                Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="site", label="S0"),
                Bead(bead_id=BeadId("b3"), particle_index=3, bead_type="site", label="S1"),
            ),
            bonds=(Bond(0, 1), Bond(2, 3)),
        )

    def _build_graph(self) -> ConnectivityGraph:
        return ConnectivityGraph(
            particle_count=4,
            step=1,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 1),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.7, 1.4, 1, 1),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.45, 1.0, 1, 1),
            ),
        )

    def _build_compartments(self) -> CompartmentRegistry:
        return CompartmentRegistry(
            particle_count=4,
            domains=(
                CompartmentDomain.from_members("A", "domain-a", (0, 1)),
                CompartmentDomain.from_members("B", "domain-b", (2, 3)),
            ),
        )

    def _build_trace_record(self, state: SimulationState) -> TraceRecord:
        return TraceRecord(
            record_id="trace-000001",
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=None,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            active_edge_count=3,
            structural_edge_count=1,
            adaptive_edge_count=2,
            plasticity_trace_count=1,
            compartment_ids=("A", "B"),
            tags=("instability", "priority"),
        )

    def test_region_selector_uses_graph_memory_and_compartment_signals(self) -> None:
        state = self._build_state()
        topology = self._build_topology()
        graph = self._build_graph()
        compartments = self._build_compartments()
        trace_record = self._build_trace_record(state)

        selector = LocalRegionSelector(
            policy=RegionSelectionPolicy(max_regions=1, max_region_size=4, min_region_score=0.5)
        )
        regions = selector.select_regions(
            state,
            topology,
            graph,
            compartments=compartments,
            trace_record=trace_record,
            focus_compartments=("A",),
        )

        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region.seed_pairs, ((1, 2),))
        self.assertEqual(region.particle_indices, (0, 1, 2, 3))
        self.assertEqual(region.compartment_ids, ("A", "B"))
        self.assertIn(RegionTriggerKind.ADAPTIVE_EDGE, region.trigger_kinds)
        self.assertIn(RegionTriggerKind.INTER_COMPARTMENT, region.trigger_kinds)
        self.assertIn(RegionTriggerKind.MEMORY_PRIORITY, region.trigger_kinds)
        self.assertIn(RegionTriggerKind.COMPARTMENT_FOCUS, region.trigger_kinds)

    def test_qcloud_coupler_caps_energy_and_force_deltas(self) -> None:
        state = self._build_state()
        topology = self._build_topology()
        region = RefinementRegion(
            region_id="region-step00000001-001",
            state_id=state.provenance.state_id,
            particle_indices=(1, 2),
            seed_pairs=((1, 2),),
            trigger_kinds=(RegionTriggerKind.ADAPTIVE_EDGE,),
            score=1.0,
        )
        coupler = QCloudForceCoupler(
            max_energy_delta_magnitude=0.5,
            max_force_delta_magnitude=2.0,
        )
        result = coupler.couple(
            ForceEvaluation(
                forces=((0.0, 0.0, 0.0),) * state.particle_count,
                potential_energy=1.0,
            ),
            state,
            topology,
            (region,),
            _BoundedCorrectionModel(
                energy_delta=3.0,
                force_map={
                    1: (3.0, 0.0, 0.0),
                    2: (0.0, 4.0, 0.0),
                },
            ),
        )

        self.assertAlmostEqual(result.force_evaluation.potential_energy, 1.5)
        self.assertEqual(result.force_evaluation.forces[1], (2.0, 0.0, 0.0))
        self.assertEqual(result.force_evaluation.forces[2], (0.0, 2.0, 0.0))
        self.assertAlmostEqual(result.force_evaluation.component_energies["qcloud"], 0.5)
        self.assertAlmostEqual(result.metadata["total_qcloud_energy_delta"], 0.5)

    def test_evaluate_with_selector_runs_end_to_end(self) -> None:
        state = self._build_state()
        topology = self._build_topology()
        graph = self._build_graph()
        compartments = self._build_compartments()
        trace_record = self._build_trace_record(state)

        result = QCloudForceCoupler().evaluate_with_selector(
            state=state,
            topology=topology,
            forcefield=BaseForceField(name="test-forcefield"),
            base_force_evaluator=_ZeroForceEvaluator(),
            correction_model=_SelectorDrivenCorrectionModel(),
            region_selector=LocalRegionSelector(
                policy=RegionSelectionPolicy(max_regions=1, max_region_size=4, min_region_score=0.5)
            ),
            graph=graph,
            compartments=compartments,
            trace_record=trace_record,
            focus_compartments=("A",),
        )

        self.assertEqual(len(result.selected_regions), 1)
        self.assertEqual(len(result.applied_corrections), 1)
        self.assertAlmostEqual(result.force_evaluation.component_energies["qcloud"], 0.2)
        self.assertEqual(result.metadata["base_force_evaluator"], "zero_force_evaluator")
        self.assertEqual(result.metadata["region_selector"], "local_region_selector")
        self.assertAlmostEqual(result.force_evaluation.forces[0][0], 0.5)


if __name__ == "__main__":
    unittest.main()

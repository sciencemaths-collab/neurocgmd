"""Tests for Section 6 adaptive graph connectivity."""

from __future__ import annotations

import unittest

from core.exceptions import ContractValidationError
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from graph import (
    ConnectivityGraph,
    ConnectivityGraphManager,
    DistanceBandConnectivityRule,
    DynamicEdgeKind,
    DynamicEdgeState,
)
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class GraphLayerTests(unittest.TestCase):
    """Verify dynamic graph construction and updates remain well-bounded."""

    def _build_state_and_topology(self) -> tuple[SimulationState, SystemTopology]:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.8, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-graph"),
                state_id=StateId("state-graph"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
            step=0,
        )
        topology = SystemTopology(
            system_id="graph-system",
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
            bonds=(Bond(0, 1),),
        )
        return state, topology

    def test_distance_band_rule_creates_structural_and_adaptive_edges(self) -> None:
        state, topology = self._build_state_and_topology()
        rule = DistanceBandConnectivityRule(local_distance_cutoff=0.75, long_range_distance_cutoff=1.8)

        edges = rule.propose_edges(state, topology)
        kinds = {edge.kind for edge in edges}
        self.assertIn(DynamicEdgeKind.STRUCTURAL_LOCAL, kinds)
        self.assertIn(DynamicEdgeKind.ADAPTIVE_LOCAL, kinds)
        self.assertIn(DynamicEdgeKind.ADAPTIVE_LONG_RANGE, kinds)
        self.assertEqual(next(edge for edge in edges if edge.normalized_pair() == (0, 1)).weight, 1.0)

    def test_connectivity_graph_rejects_duplicate_pairs(self) -> None:
        with self.assertRaises(ContractValidationError):
            ConnectivityGraph(
                particle_count=2,
                step=0,
                edges=(
                    DynamicEdgeState(0, 1, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.5, 0.5, 0, 0),
                    DynamicEdgeState(1, 0, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.4, 1.0, 0, 0),
                ),
            )

    def test_graph_manager_updates_from_previous_graph(self) -> None:
        state, topology = self._build_state_and_topology()
        manager = ConnectivityGraphManager(
            rule=DistanceBandConnectivityRule(local_distance_cutoff=0.75, long_range_distance_cutoff=1.8)
        )
        initial_graph = manager.initialize(state, topology)

        moved_state = SimulationState(
            units=state.units,
            particles=state.particles.with_positions(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.2, 0.0, 0.0), (5.0, 0.0, 0.0))
            ),
            thermodynamics=state.thermodynamics,
            provenance=state.provenance,
            step=1,
        )
        updated_graph = manager.update(moved_state, topology, initial_graph)

        self.assertEqual(updated_graph.step, 1)
        self.assertIn((0, 1), {edge.normalized_pair() for edge in updated_graph.structural_edges()})
        self.assertNotIn((1, 2), {edge.normalized_pair() for edge in updated_graph.active_edges()})

    def test_graph_roundtrip_and_components(self) -> None:
        state, topology = self._build_state_and_topology()
        graph = ConnectivityGraphManager(
            rule=DistanceBandConnectivityRule(local_distance_cutoff=0.75, long_range_distance_cutoff=1.8)
        ).initialize(state, topology)

        restored = ConnectivityGraph.from_dict(graph.to_dict())
        self.assertEqual(restored, graph)
        self.assertEqual(restored.connected_components(), ((0, 1, 2, 3),))


if __name__ == "__main__":
    unittest.main()

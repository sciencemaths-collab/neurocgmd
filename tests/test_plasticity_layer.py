"""Tests for Section 7 plasticity and rewiring rules."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from graph import ConnectivityGraph, ConnectivityGraphManager, DistanceBandConnectivityRule, DynamicEdgeKind, DynamicEdgeState
from plasticity import HebbianGrowthRule, PairTraceState, PlasticityEngine, PruningRule, ReinforcementRule, update_pair_traces
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class PlasticityLayerTests(unittest.TestCase):
    """Verify bounded graph plasticity behavior."""

    def _build_state_and_topology(self) -> tuple[SimulationState, SystemTopology]:
        state = SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.2, 0.0, 0.0)),
                masses=(1.0, 1.0, 1.0, 1.0),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-plasticity"),
                state_id=StateId("state-plasticity"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
            step=2,
        )
        topology = SystemTopology(
            system_id="plasticity-system",
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

    def _build_graph(self, state: SimulationState, topology: SystemTopology) -> ConnectivityGraph:
        return ConnectivityGraphManager(
            rule=DistanceBandConnectivityRule(local_distance_cutoff=0.8, long_range_distance_cutoff=1.7)
        ).initialize(state, topology)

    def test_trace_update_accumulates_activity(self) -> None:
        state, topology = self._build_state_and_topology()
        graph = self._build_graph(state, topology)
        traces = update_pair_traces(
            current_step=state.step,
            graph=graph,
            activity_signals={(1, 2): 1.0, (0, 3): 0.8},
        )

        lookup = {trace.normalized_pair(): trace for trace in traces}
        self.assertIn((1, 2), lookup)
        self.assertGreater(lookup[(1, 2)].activity_level, 0.0)
        self.assertIn((0, 3), lookup)

    def test_reinforcement_increases_supported_adaptive_edge_weight(self) -> None:
        graph = ConnectivityGraph(
            particle_count=3,
            step=5,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 5),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.2, 0.6, 3, 5),
            ),
        )
        traces = (
            PairTraceState(1, 2, activity_level=0.8, coactivity_level=0.9, persistence=2, last_seen_step=5, seen_count=3),
        )
        updated = ReinforcementRule(activity_threshold=0.3).apply(graph, traces)
        adaptive_edge = updated.edge_for_pair(1, 2)
        self.assertGreater(adaptive_edge.weight, 0.2)
        self.assertEqual(updated.edge_for_pair(0, 1).weight, 1.0)

    def test_pruning_deactivates_weak_inactive_adaptive_edge(self) -> None:
        graph = ConnectivityGraph(
            particle_count=3,
            step=6,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 6),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.08, 0.6, 3, 6),
            ),
        )
        traces = (
            PairTraceState(1, 2, activity_level=0.05, coactivity_level=0.05, persistence=0, last_seen_step=4, seen_count=3),
        )
        updated = PruningRule(weight_threshold=0.1, activity_floor=0.1, min_persistence_to_keep=1).apply(graph, traces)
        self.assertFalse(updated.edge_for_pair(1, 2).active)
        self.assertTrue(updated.edge_for_pair(0, 1).active)

    def test_hebbian_growth_adds_new_edge_for_strong_trace(self) -> None:
        state, topology = self._build_state_and_topology()
        graph = ConnectivityGraph(
            particle_count=4,
            step=state.step,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, state.step),
            ),
        )
        traces = (
            PairTraceState(1, 2, activity_level=0.7, coactivity_level=0.85, persistence=3, last_seen_step=state.step, seen_count=4),
        )
        grown = HebbianGrowthRule(
            growth_threshold=0.6,
            initial_weight=0.25,
            local_distance_cutoff=0.8,
            long_range_distance_cutoff=1.2,
        ).apply(state, topology, graph, traces)
        self.assertIn((1, 2), {edge.normalized_pair() for edge in grown.edges})

    def test_plasticity_engine_composes_rules(self) -> None:
        state, topology = self._build_state_and_topology()
        graph = self._build_graph(state, topology)
        initial_edge = graph.edge_for_pair(2, 3)
        engine = PlasticityEngine(
            trace_decay=0.0,
            reinforcement_rule=ReinforcementRule(activity_threshold=0.2),
            growth_rule=HebbianGrowthRule(
                growth_threshold=0.6,
                initial_weight=0.2,
                local_distance_cutoff=0.8,
                long_range_distance_cutoff=1.2,
                max_new_edges_per_step=2,
            )
        )
        result = engine.update(
            state,
            topology,
            graph,
            activity_signals={(2, 3): 1.0, (0, 3): 0.9},
        )

        self.assertGreaterEqual(result.metadata["trace_count"], 2)
        self.assertEqual(result.graph.step, state.step)
        self.assertEqual(result.graph.edge_for_pair(0, 1).kind, DynamicEdgeKind.STRUCTURAL_LOCAL)
        self.assertGreater(result.graph.edge_for_pair(2, 3).weight, initial_edge.weight)


if __name__ == "__main__":
    unittest.main()

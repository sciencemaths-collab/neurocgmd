"""Tests for STDP plasticity rules and GNN message-passing graph updates."""

from __future__ import annotations

import unittest

from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, SimulationId, StateId
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from graph.message_passing import MessagePassingGraphUpdater, MessagePassingLayer, NodeFeature
from plasticity.stdp import HomeostaticScaling, SpikeTimingWindow, STDPRule
from plasticity.traces import PairTraceState
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class _FixtureMixin:
    """Shared helpers for building a minimal 4-particle test system."""

    @staticmethod
    def _make_state(step: int = 5) -> SimulationState:
        return SimulationState(
            units=UnitSystem.md_nano(),
            particles=ParticleState(
                positions=(
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.5, 0.0, 0.0),
                    (2.2, 0.0, 0.0),
                ),
                masses=(1.0, 1.0, 1.0, 1.0),
                velocities=(
                    (0.1, 0.0, 0.0),
                    (0.0, 0.1, 0.0),
                    (0.0, 0.0, 0.1),
                    (-0.1, 0.0, 0.0),
                ),
                forces=(
                    (0.5, 0.0, 0.0),
                    (0.0, 0.5, 0.0),
                    (0.0, 0.0, 0.5),
                    (-0.5, 0.0, 0.0),
                ),
            ),
            thermodynamics=ThermodynamicState(),
            provenance=StateProvenance(
                simulation_id=SimulationId("sim-stdp-gnn"),
                state_id=StateId("state-stdp-gnn"),
                parent_state_id=None,
                created_by="unit-test",
                stage="initialization",
            ),
            step=step,
        )

    @staticmethod
    def _make_topology() -> SystemTopology:
        return SystemTopology(
            system_id="stdp-gnn-system",
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

    @staticmethod
    def _make_graph(step: int = 5) -> ConnectivityGraph:
        """Build a small graph with three active edges and known weights."""
        return ConnectivityGraph(
            particle_count=4,
            step=step,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, step),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.5, 0.5, 0, step),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.3, 0.7, 0, step),
            ),
        )


# -----------------------------------------------------------------------
# STDP tests
# -----------------------------------------------------------------------


class TestSpikeTimingWindow(unittest.TestCase, _FixtureMixin):
    """Tests for the asymmetric STDP timing window."""

    def test_spike_timing_window_potentiation(self) -> None:
        """Positive delta_t (pre before post) should yield a positive weight change."""
        window = SpikeTimingWindow(a_plus=0.1, a_minus=0.12, weight_dependence=False)
        delta_w = window.compute_delta(delta_t=5.0, current_weight=0.5)
        self.assertGreater(delta_w, 0.0, "Pre-before-post should cause potentiation.")

    def test_spike_timing_window_depression(self) -> None:
        """Negative delta_t (post before pre) should yield a negative weight change."""
        window = SpikeTimingWindow(a_plus=0.1, a_minus=0.12, weight_dependence=False)
        delta_w = window.compute_delta(delta_t=-5.0, current_weight=0.5)
        self.assertLess(delta_w, 0.0, "Post-before-pre should cause depression.")

    def test_zero_delta_t_gives_no_change(self) -> None:
        """A delta_t of exactly zero should produce no weight change."""
        window = SpikeTimingWindow()
        delta_w = window.compute_delta(delta_t=0.0, current_weight=0.5)
        self.assertEqual(delta_w, 0.0)

    def test_weight_dependence_attenuates_potentiation_near_ceiling(self) -> None:
        """With weight dependence, potentiation is smaller for weights near 1.0."""
        window = SpikeTimingWindow(weight_dependence=True)
        dw_low = window.compute_delta(delta_t=5.0, current_weight=0.1)
        dw_high = window.compute_delta(delta_t=5.0, current_weight=0.9)
        self.assertGreater(dw_low, dw_high,
                           "Multiplicative STDP should attenuate potentiation near ceiling.")


class TestSTDPRule(unittest.TestCase, _FixtureMixin):
    """Tests for the full STDPRule applied to a ConnectivityGraph."""

    def test_stdp_rule_updates_edge_weights(self) -> None:
        """Applying the STDP rule to a graph with above-threshold traces should change weights."""
        graph = self._make_graph(step=5)
        # Trace for edge (1, 2) with high activity and coactivity != 0.5 to get non-zero delta_t
        traces = (
            PairTraceState(
                source_index=1,
                target_index=2,
                activity_level=0.8,
                coactivity_level=0.9,
                persistence=3,
                last_seen_step=5,
                seen_count=4,
            ),
        )
        rule = STDPRule(activity_threshold=0.3)
        updated = rule.apply(graph, traces, current_step=6)

        original_weight = graph.edge_for_pair(1, 2).weight
        new_weight = updated.edge_for_pair(1, 2).weight
        self.assertNotEqual(original_weight, new_weight,
                            "STDP should change the weight of an active edge with a matching trace.")

    def test_stdp_rule_respects_bounds(self) -> None:
        """After STDP updates, weights must stay within [0.01, 1.0]."""
        # Edge near the ceiling
        graph_high = ConnectivityGraph(
            particle_count=4,
            step=5,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 5),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.99, 0.5, 0, 5),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.02, 0.7, 0, 5),
            ),
        )
        # Trace designed for strong potentiation (coactivity_level < 0.5 -> positive delta_t)
        traces = (
            PairTraceState(1, 2, activity_level=0.9, coactivity_level=0.1, persistence=5, last_seen_step=5, seen_count=5),
            PairTraceState(2, 3, activity_level=0.9, coactivity_level=0.9, persistence=5, last_seen_step=5, seen_count=5),
        )
        rule = STDPRule(activity_threshold=0.1)
        updated = rule.apply(graph_high, traces, current_step=6)

        for edge in updated.edges:
            self.assertGreaterEqual(edge.weight, 0.01, f"Weight below lower bound for pair {edge.normalized_pair()}")
            self.assertLessEqual(edge.weight, 1.0, f"Weight above upper bound for pair {edge.normalized_pair()}")

    def test_stdp_rule_skips_inactive_edges(self) -> None:
        """Inactive edges should pass through unchanged."""
        graph = ConnectivityGraph(
            particle_count=3,
            step=5,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.5, 1.0, 0, 5, active=False),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.5, 0.5, 0, 5),
            ),
        )
        traces = (
            PairTraceState(0, 1, activity_level=0.9, coactivity_level=0.8, persistence=3, last_seen_step=5, seen_count=3),
        )
        rule = STDPRule(activity_threshold=0.3)
        updated = rule.apply(graph, traces, current_step=6)
        self.assertEqual(updated.edge_for_pair(0, 1).weight, 0.5,
                         "Inactive edge weight should not be modified.")


class TestHomeostaticScaling(unittest.TestCase, _FixtureMixin):
    """Tests for the homeostatic scaling mechanism."""

    def test_homeostatic_scaling_normalizes(self) -> None:
        """After scaling, the mean active-edge weight should move toward the target."""
        # Start with mean weight well above target
        graph = ConnectivityGraph(
            particle_count=4,
            step=5,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.9, 1.0, 0, 5),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.8, 0.5, 0, 5),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.85, 0.7, 0, 5),
            ),
        )
        target = 0.5
        scaler = HomeostaticScaling(target_mean_weight=target, scaling_rate=0.05)

        original_mean = sum(e.weight for e in graph.active_edges()) / len(graph.active_edges())
        updated = scaler.apply(graph)
        new_mean = sum(e.weight for e in updated.active_edges()) / len(updated.active_edges())

        self.assertGreater(original_mean, target)
        self.assertLess(new_mean, original_mean,
                        "Homeostatic scaling should reduce mean weight toward target.")
        self.assertGreater(new_mean, target,
                           "A single step should not overshoot the target.")

    def test_homeostatic_scaling_increases_low_weights(self) -> None:
        """When the mean weight is below the target, scaling should increase weights."""
        graph = ConnectivityGraph(
            particle_count=3,
            step=5,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.1, 1.0, 0, 5),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.15, 0.5, 0, 5),
            ),
        )
        scaler = HomeostaticScaling(target_mean_weight=0.5, scaling_rate=0.05)
        updated = scaler.apply(graph)
        for orig, upd in zip(graph.edges, updated.edges):
            self.assertGreater(upd.weight, orig.weight,
                               "Weights should increase when mean is below target.")


# -----------------------------------------------------------------------
# GNN message-passing tests
# -----------------------------------------------------------------------


class TestMessagePassingLayer(unittest.TestCase, _FixtureMixin):
    """Tests for a single message-passing layer."""

    def test_message_passing_layer_computes_messages(self) -> None:
        """Messages should be generated for every active edge (both directions)."""
        graph = self._make_graph(step=5)
        layer = MessagePassingLayer(input_dim=8, message_dim=8, edge_feature_dim=4)

        node_features = [
            NodeFeature(particle_index=i, features=tuple(float(i + k) for k in range(8)))
            for i in range(4)
        ]

        messages = layer.compute_messages(node_features, graph)

        active_count = len(graph.active_edges())
        # Each active edge generates 2 messages (forward + reverse)
        self.assertEqual(len(messages), active_count * 2,
                         "Each active edge should produce two directed messages.")

        # Each message vector should have length == message_dim
        for msg in messages:
            self.assertEqual(len(msg.message), 8)

        # Message values should all be in [-1, 1] (tanh output)
        for msg in messages:
            for val in msg.message:
                self.assertGreaterEqual(val, -1.0)
                self.assertLessEqual(val, 1.0)

    def test_aggregate_messages_sums_to_correct_shape(self) -> None:
        """Aggregated messages should have one entry per particle."""
        graph = self._make_graph(step=5)
        layer = MessagePassingLayer(input_dim=8, message_dim=8, edge_feature_dim=4)
        node_features = [
            NodeFeature(particle_index=i, features=tuple(float(i + k) for k in range(8)))
            for i in range(4)
        ]
        messages = layer.compute_messages(node_features, graph)
        aggregated = layer.aggregate_messages(messages, graph.particle_count)

        self.assertEqual(len(aggregated), 4)
        for agg in aggregated:
            self.assertEqual(len(agg), 8)


class TestMessagePassingGraphUpdater(unittest.TestCase, _FixtureMixin):
    """Tests for the GNN-based graph updater."""

    def test_graph_updater_updates_weights(self) -> None:
        """Running the updater should modify at least some edge weights."""
        state = self._make_state(step=5)
        topology = self._make_topology()
        graph = self._make_graph(step=5)

        updater = MessagePassingGraphUpdater(
            layers=2,
            weight_update_rate=0.1,
            edge_pruning_threshold=0.05,
            edge_creation_threshold=0.99,  # high threshold to avoid new edges
            message_dim=8,
        )
        updated = updater.update(state, topology, graph)

        # At least one weight should have changed
        original_weights = {e.normalized_pair(): e.weight for e in graph.edges}
        any_changed = False
        for edge in updated.edges:
            pair = edge.normalized_pair()
            if pair in original_weights and edge.weight != original_weights[pair]:
                any_changed = True
                break
        self.assertTrue(any_changed, "Graph updater should change at least one edge weight.")

    def test_graph_updater_can_prune_edges(self) -> None:
        """An edge with a very low weight should be pruned (set inactive) by the updater."""
        state = self._make_state(step=5)
        topology = self._make_topology()

        # Create a graph with one very weak edge that should get pruned
        graph = ConnectivityGraph(
            particle_count=4,
            step=5,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 5),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.5, 0.5, 0, 5),
                # Very low weight edge likely to be pruned
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.02, 0.7, 0, 5),
            ),
        )

        updater = MessagePassingGraphUpdater(
            layers=2,
            weight_update_rate=0.0,   # keep weights unchanged so 0.02 stays below threshold
            edge_pruning_threshold=0.05,  # above the 0.02 weight
            edge_creation_threshold=0.99,
            message_dim=8,
        )
        updated = updater.update(state, topology, graph)

        # The weak edge (2,3) at 0.02 should have been deactivated since 0.02 < 0.05
        edge_23 = next(e for e in updated.edges if e.normalized_pair() == (2, 3))
        self.assertFalse(edge_23.active,
                         "An edge with weight below pruning_threshold should be deactivated.")

    def test_node_feature_extraction(self) -> None:
        """Verify that extract_node_features returns one feature per particle with correct dimension."""
        state = self._make_state(step=5)
        topology = self._make_topology()
        graph = self._make_graph(step=5)

        updater = MessagePassingGraphUpdater(message_dim=8)
        features = updater.extract_node_features(state, topology, graph)

        self.assertEqual(len(features), 4, "Should have one feature per particle.")
        for nf in features:
            self.assertEqual(len(nf.features), 8, "Feature vector length should equal message_dim.")

        # Position information should be reflected in features: particle 0 at origin
        feat_0 = features[0]
        self.assertAlmostEqual(feat_0.features[0], 0.0, places=5, msg="x-position of particle 0 should be 0.")
        self.assertAlmostEqual(feat_0.features[1], 0.0, places=5, msg="y-position of particle 0 should be 0.")
        self.assertAlmostEqual(feat_0.features[2], 0.0, places=5, msg="z-position of particle 0 should be 0.")

        # Particle 1 at (1.0, 0, 0)
        feat_1 = features[1]
        self.assertAlmostEqual(feat_1.features[0], 1.0, places=5, msg="x-position of particle 1 should be 1.0.")

    def test_updater_preserves_particle_count(self) -> None:
        """The updated graph must retain the same particle count."""
        state = self._make_state(step=5)
        topology = self._make_topology()
        graph = self._make_graph(step=5)

        updater = MessagePassingGraphUpdater(layers=1, message_dim=8)
        updated = updater.update(state, topology, graph)
        self.assertEqual(updated.particle_count, graph.particle_count)


if __name__ == "__main__":
    unittest.main()

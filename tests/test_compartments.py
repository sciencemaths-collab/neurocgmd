"""Tests for Section 8 compartment models, registry, and routing."""

from __future__ import annotations

import unittest

from compartments import (
    CompartmentDomain,
    CompartmentRegistry,
    CompartmentRole,
    CompartmentRouteKind,
    build_compartment_route_map,
    classify_graph_edges,
    route_neighbors_for,
)
from core.exceptions import ContractValidationError
from core.state import ParticleState, SimulationState, StateProvenance, ThermodynamicState, UnitSystem
from core.types import BeadId, CompartmentId, SimulationId, StateId
from graph import ConnectivityGraph, DynamicEdgeKind, DynamicEdgeState
from topology import Bead, BeadRole, BeadType, Bond, SystemTopology


class CompartmentsTests(unittest.TestCase):
    """Verify compartment overlays remain explicit and well-bounded."""

    def _build_topology(self) -> SystemTopology:
        return SystemTopology(
            system_id="compartment-system",
            bead_types=(
                BeadType(name="bb", role=BeadRole.STRUCTURAL),
                BeadType(name="site", role=BeadRole.FUNCTIONAL),
            ),
            beads=(
                Bead(bead_id=BeadId("b0"), particle_index=0, bead_type="bb", label="B0", compartment_hint=CompartmentId("A")),
                Bead(bead_id=BeadId("b1"), particle_index=1, bead_type="bb", label="B1", compartment_hint=CompartmentId("A")),
                Bead(bead_id=BeadId("b2"), particle_index=2, bead_type="site", label="S0", compartment_hint=CompartmentId("B")),
                Bead(bead_id=BeadId("b3"), particle_index=3, bead_type="site", label="S1", compartment_hint=CompartmentId("B")),
            ),
            bonds=(Bond(0, 1),),
        )

    def _build_graph(self) -> ConnectivityGraph:
        return ConnectivityGraph(
            particle_count=4,
            step=3,
            edges=(
                DynamicEdgeState(0, 1, DynamicEdgeKind.STRUCTURAL_LOCAL, 1.0, 1.0, 0, 3),
                DynamicEdgeState(1, 2, DynamicEdgeKind.ADAPTIVE_LONG_RANGE, 0.4, 1.5, 2, 3),
                DynamicEdgeState(2, 3, DynamicEdgeKind.ADAPTIVE_LOCAL, 0.8, 0.7, 2, 3),
            ),
        )

    def test_registry_membership_and_topology_hints(self) -> None:
        topology = self._build_topology()
        registry = CompartmentRegistry.from_topology_hints(topology)

        self.assertEqual(registry.domain_by_id("A").particle_indices, (0, 1))
        self.assertEqual(tuple(domain.compartment_id for domain in registry.domains_for_particle(2)), (CompartmentId("B"),))
        self.assertEqual(registry.unassigned_particles(), ())
        self.assertEqual(registry.validate_against_topology(topology), ())

    def test_registry_rejects_overlap_by_default(self) -> None:
        with self.assertRaises(ContractValidationError):
            CompartmentRegistry(
                particle_count=3,
                domains=(
                    CompartmentDomain.from_members("A", "A", (0, 1)),
                    CompartmentDomain.from_members("B", "B", (1, 2)),
                ),
            )

    def test_route_classification_and_aggregation(self) -> None:
        topology = self._build_topology()
        registry = CompartmentRegistry.from_topology_hints(topology)
        graph = self._build_graph()

        assignments = classify_graph_edges(registry, graph)
        self.assertEqual(assignments[0].kind, CompartmentRouteKind.INTRA)
        self.assertEqual(assignments[1].kind, CompartmentRouteKind.INTER)
        self.assertEqual(assignments[2].kind, CompartmentRouteKind.INTRA)

        route_map = build_compartment_route_map(registry, graph)
        summary = route_map[(CompartmentId("A"), CompartmentId("B"))]
        self.assertEqual(summary.edge_count, 1)
        self.assertAlmostEqual(summary.total_weight, 0.4)
        self.assertEqual(route_neighbors_for(registry, graph, "A"), (summary,))

    def test_manual_registry_roundtrip(self) -> None:
        registry = CompartmentRegistry(
            particle_count=4,
            domains=(
                CompartmentDomain.from_members("A", "domain-a", (0, 1), role=CompartmentRole.STRUCTURAL),
                CompartmentDomain.from_members("B", "domain-b", (2, 3), role=CompartmentRole.FUNCTIONAL),
            ),
        )
        restored = CompartmentRegistry.from_dict(registry.to_dict())
        self.assertEqual(restored, registry)


if __name__ == "__main__":
    unittest.main()

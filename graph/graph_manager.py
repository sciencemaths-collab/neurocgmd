"""Graph assembly and lifecycle management for adaptive connectivity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata
from graph.adjacency_utils import build_adjacency_map, build_edge_lookup, connected_components, partition_edges_by_kind
from graph.connectivity_rules import DistanceBandConnectivityRule
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class ConnectivityGraph(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Adaptive graph layered on top of fixed topology."""

    name: str = "connectivity_graph"
    classification: str = "[proposed novel]"
    particle_count: int = 0
    step: int = 0
    edges: tuple[DynamicEdgeState, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Represents the adaptive connectivity layer that augments fixed topology "
            "with local and long-range dynamic edges."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "topology/system_topology.py",
            "graph/edge_models.py",
            "graph/connectivity_rules.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/dynamic_graph.md",
            "docs/sections/section_06_dynamic_graph_connectivity.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_count <= 0:
            issues.append("particle_count must be positive.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        edge_pairs = [edge.normalized_pair() for edge in self.edges]
        if len(edge_pairs) != len(set(edge_pairs)):
            issues.append("ConnectivityGraph edges must have unique undirected endpoint pairs.")
        for edge in self.edges:
            if edge.target_index >= self.particle_count:
                issues.append(
                    f"Graph edge {edge.normalized_pair()} exceeds particle_count={self.particle_count}."
                )
        return tuple(issues)

    def active_edges(self) -> tuple[DynamicEdgeState, ...]:
        return tuple(edge for edge in self.edges if edge.active)

    def edge_for_pair(self, source_index: int, target_index: int) -> DynamicEdgeState:
        pair = tuple(sorted((source_index, target_index)))
        return build_edge_lookup(self.edges, active_only=False)[pair]

    def adjacency_map(self, *, active_only: bool = True) -> dict[int, tuple[int, ...]]:
        return build_adjacency_map(self.particle_count, self.edges, active_only=active_only)

    def neighbors_for(self, particle_index: int, *, active_only: bool = True) -> tuple[int, ...]:
        return self.adjacency_map(active_only=active_only)[particle_index]

    def connected_components(self, *, active_only: bool = True) -> tuple[tuple[int, ...], ...]:
        return connected_components(self.particle_count, self.edges, active_only=active_only)

    def structural_edges(self) -> tuple[DynamicEdgeState, ...]:
        return partition_edges_by_kind(self.edges).get(DynamicEdgeKind.STRUCTURAL_LOCAL, ())

    def adaptive_edges(self) -> tuple[DynamicEdgeState, ...]:
        grouped = partition_edges_by_kind(self.edges)
        return grouped.get(DynamicEdgeKind.ADAPTIVE_LOCAL, ()) + grouped.get(
            DynamicEdgeKind.ADAPTIVE_LONG_RANGE, ()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "particle_count": self.particle_count,
            "step": self.step,
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConnectivityGraph":
        return cls(
            name=str(data.get("name", "connectivity_graph")),
            classification=str(data.get("classification", "[proposed novel]")),
            particle_count=int(data["particle_count"]),
            step=int(data["step"]),
            edges=tuple(DynamicEdgeState.from_dict(item) for item in data.get("edges", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ConnectivityGraphManager:
    """Build and update adaptive connectivity graphs from physical state."""

    rule: DistanceBandConnectivityRule = field(
        default_factory=lambda: DistanceBandConnectivityRule(
            local_distance_cutoff=1.0,
            long_range_distance_cutoff=2.5,
        )
    )

    name: str = "connectivity_graph_manager"
    classification: str = "[proposed novel]"

    def initialize(self, state: SimulationState, topology: SystemTopology) -> ConnectivityGraph:
        return self._build_graph(state, topology, previous_edges=())

    def update(
        self,
        state: SimulationState,
        topology: SystemTopology,
        previous_graph: ConnectivityGraph,
    ) -> ConnectivityGraph:
        if previous_graph.particle_count != state.particle_count:
            raise ContractValidationError(
                "previous_graph particle_count must match the current SimulationState."
            )
        return self._build_graph(state, topology, previous_edges=previous_graph.edges)

    def _build_graph(
        self,
        state: SimulationState,
        topology: SystemTopology,
        *,
        previous_edges: tuple[DynamicEdgeState, ...],
    ) -> ConnectivityGraph:
        edges = self.rule.propose_edges(state, topology, previous_edges=previous_edges)
        grouped = partition_edges_by_kind(edges)
        metadata = FrozenMetadata(
            {
                "rule": self.rule.name,
                "structural_edge_count": len(grouped.get(DynamicEdgeKind.STRUCTURAL_LOCAL, ())),
                "adaptive_local_edge_count": len(grouped.get(DynamicEdgeKind.ADAPTIVE_LOCAL, ())),
                "adaptive_long_range_edge_count": len(
                    grouped.get(DynamicEdgeKind.ADAPTIVE_LONG_RANGE, ())
                ),
            }
        )
        return ConnectivityGraph(
            particle_count=state.particle_count,
            step=state.step,
            edges=edges,
            metadata=metadata,
        )


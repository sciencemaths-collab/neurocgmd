"""Graph- and compartment-oriented diagnostic view models."""

from __future__ import annotations

from dataclasses import dataclass, field

from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import FrozenMetadata, Vector3, coerce_scalar, coerce_vector3
from graph.graph_manager import ConnectivityGraph


def _normalize_compartments(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values or ():
        value = str(raw_value).strip()
        if not value:
            continue
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class GraphNodeView(ValidatableComponent):
    """Serializable graph-node diagnostic view for one particle."""

    particle_index: int
    label: str
    position: Vector3
    compartments: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", coerce_vector3(self.position, "position"))
        object.__setattr__(self, "compartments", _normalize_compartments(self.compartments))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.particle_index < 0:
            issues.append("particle_index must be non-negative.")
        if not self.label.strip():
            issues.append("label must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "particle_index": self.particle_index,
            "label": self.label,
            "position": list(self.position),
            "compartments": list(self.compartments),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GraphEdgeView(ValidatableComponent):
    """Serializable graph-edge diagnostic view."""

    source_index: int
    target_index: int
    kind: str
    active: bool
    weight: float
    distance: float
    route_label: str = "unassigned"
    source_compartments: tuple[str, ...] = ()
    target_compartments: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", coerce_scalar(self.weight, "weight"))
        object.__setattr__(self, "distance", coerce_scalar(self.distance, "distance"))
        object.__setattr__(self, "source_compartments", _normalize_compartments(self.source_compartments))
        object.__setattr__(self, "target_compartments", _normalize_compartments(self.target_compartments))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.source_index < 0 or self.target_index < 0:
            issues.append("edge endpoint indices must be non-negative.")
        if self.target_index <= self.source_index:
            issues.append("target_index must be greater than source_index for undirected edge views.")
        if not self.kind.strip():
            issues.append("kind must be a non-empty string.")
        if self.weight < 0.0:
            issues.append("weight must be non-negative.")
        if self.distance < 0.0:
            issues.append("distance must be non-negative.")
        if not self.route_label.strip():
            issues.append("route_label must be a non-empty string.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "kind": self.kind,
            "active": self.active,
            "weight": self.weight,
            "distance": self.distance,
            "route_label": self.route_label,
            "source_compartments": list(self.source_compartments),
            "target_compartments": list(self.target_compartments),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GraphSnapshotView(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Serializable graph snapshot for dashboard and export consumers."""

    name: str = "graph_snapshot_view"
    classification: str = "[adapted]"
    step: int = 0
    node_count: int = 0
    active_edge_count: int = 0
    structural_edge_count: int = 0
    adaptive_edge_count: int = 0
    nodes: tuple[GraphNodeView, ...] = ()
    edges: tuple[GraphEdgeView, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def describe_role(self) -> str:
        return (
            "Adapts graph and compartment state into a serializable diagnostic view "
            "for dashboards, exports, and later visualization layers."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "graph/graph_manager.py",
            "compartments/registry.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/visualization_and_diagnostics.md",
            "docs/sections/section_14_visualization_and_diagnostics.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.step < 0:
            issues.append("step must be non-negative.")
        if self.node_count <= 0:
            issues.append("node_count must be positive.")
        if self.active_edge_count < 0 or self.structural_edge_count < 0 or self.adaptive_edge_count < 0:
            issues.append("edge counts must be non-negative.")
        if self.structural_edge_count + self.adaptive_edge_count < self.active_edge_count:
            issues.append(
                "structural_edge_count + adaptive_edge_count must be at least active_edge_count."
            )
        if len(self.nodes) != self.node_count:
            issues.append("nodes length must match node_count.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "step": self.step,
            "node_count": self.node_count,
            "active_edge_count": self.active_edge_count,
            "structural_edge_count": self.structural_edge_count,
            "adaptive_edge_count": self.adaptive_edge_count,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_state_graph(
        cls,
        state: SimulationState,
        graph: ConnectivityGraph,
        *,
        compartments: CompartmentRegistry | None = None,
    ) -> "GraphSnapshotView":
        if graph.particle_count != state.particle_count:
            raise ContractValidationError("graph.particle_count must match the SimulationState particle count.")
        if graph.step != state.step:
            raise ContractValidationError("graph.step must match the SimulationState step.")
        if compartments is not None and compartments.particle_count != state.particle_count:
            raise ContractValidationError("compartments.particle_count must match the SimulationState particle count.")

        membership_map = compartments.membership_map() if compartments is not None else {}
        labels = state.particles.labels or tuple(f"P{index}" for index in range(state.particle_count))
        nodes = tuple(
            GraphNodeView(
                particle_index=index,
                label=labels[index],
                position=state.particles.positions[index],
                compartments=tuple(str(identifier) for identifier in membership_map.get(index, ())),
            )
            for index in range(state.particle_count)
        )

        edge_views: list[GraphEdgeView] = []
        for edge in graph.edges:
            source_compartments = tuple(str(identifier) for identifier in membership_map.get(edge.source_index, ()))
            target_compartments = tuple(str(identifier) for identifier in membership_map.get(edge.target_index, ()))
            if source_compartments and target_compartments:
                if set(source_compartments) == set(target_compartments):
                    route_label = f"intra:{'/'.join(source_compartments)}"
                else:
                    route_label = f"inter:{'/'.join(source_compartments)}->{'/'.join(target_compartments)}"
            elif source_compartments:
                route_label = f"partial:{'/'.join(source_compartments)}"
            elif target_compartments:
                route_label = f"partial:{'/'.join(target_compartments)}"
            else:
                route_label = "unassigned"
            edge_views.append(
                GraphEdgeView(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind.value,
                    active=edge.active,
                    weight=edge.weight,
                    distance=edge.distance,
                    route_label=route_label,
                    source_compartments=source_compartments,
                    target_compartments=target_compartments,
                    metadata=edge.metadata,
                )
            )

        active_edges = graph.active_edges()
        return cls(
            step=state.step,
            node_count=state.particle_count,
            active_edge_count=len(active_edges),
            structural_edge_count=len(graph.structural_edges()),
            adaptive_edge_count=len(graph.adaptive_edges()),
            nodes=nodes,
            edges=tuple(edge_views),
            metadata=FrozenMetadata(
                {
                    "component_count": len(graph.connected_components()),
                    "has_compartments": compartments is not None,
                }
            ),
        )

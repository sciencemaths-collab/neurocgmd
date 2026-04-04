"""Compartment-aware edge and route summaries over the adaptive graph."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from compartments.registry import CompartmentRegistry
from core.types import CompartmentId
from graph.graph_manager import ConnectivityGraph


class CompartmentRouteKind(StrEnum):
    """High-level routing relation for a graph edge."""

    INTRA = "intra"
    INTER = "inter"
    PARTIAL = "partial"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True, slots=True)
class EdgeRouteAssignment:
    """Classification of one graph edge relative to compartments."""

    pair: tuple[int, int]
    source_compartments: tuple[CompartmentId, ...]
    target_compartments: tuple[CompartmentId, ...]
    kind: CompartmentRouteKind
    weight: float


@dataclass(frozen=True, slots=True)
class CompartmentRouteSummary:
    """Aggregated weighted route between two compartments."""

    source_compartment: CompartmentId
    target_compartment: CompartmentId
    kind: CompartmentRouteKind
    edge_count: int
    total_weight: float


def classify_graph_edges(
    registry: CompartmentRegistry,
    graph: ConnectivityGraph,
    *,
    active_only: bool = True,
) -> tuple[EdgeRouteAssignment, ...]:
    """Classify graph edges as intra-, inter-, partial, or unassigned routes."""

    assignments: list[EdgeRouteAssignment] = []
    for edge in graph.active_edges() if active_only else graph.edges:
        source_compartments = tuple(
            domain.compartment_id for domain in registry.domains_for_particle(edge.source_index)
        )
        target_compartments = tuple(
            domain.compartment_id for domain in registry.domains_for_particle(edge.target_index)
        )
        if source_compartments and target_compartments:
            shared = set(source_compartments) & set(target_compartments)
            kind = CompartmentRouteKind.INTRA if shared else CompartmentRouteKind.INTER
        elif source_compartments or target_compartments:
            kind = CompartmentRouteKind.PARTIAL
        else:
            kind = CompartmentRouteKind.UNASSIGNED
        assignments.append(
            EdgeRouteAssignment(
                pair=edge.normalized_pair(),
                source_compartments=source_compartments,
                target_compartments=target_compartments,
                kind=kind,
                weight=edge.weight,
            )
        )
    return tuple(assignments)


def build_compartment_route_map(
    registry: CompartmentRegistry,
    graph: ConnectivityGraph,
    *,
    active_only: bool = True,
) -> dict[tuple[CompartmentId, CompartmentId], CompartmentRouteSummary]:
    """Aggregate weighted routes between compartment pairs."""

    grouped: dict[tuple[CompartmentId, CompartmentId], list[float]] = defaultdict(list)
    relation_kind: dict[tuple[CompartmentId, CompartmentId], CompartmentRouteKind] = {}
    for assignment in classify_graph_edges(registry, graph, active_only=active_only):
        if assignment.kind != CompartmentRouteKind.INTER:
            continue
        for source_compartment in assignment.source_compartments:
            for target_compartment in assignment.target_compartments:
                key = tuple(sorted((source_compartment, target_compartment), key=str))
                grouped[key].append(assignment.weight)
                relation_kind[key] = assignment.kind

    return {
        key: CompartmentRouteSummary(
            source_compartment=key[0],
            target_compartment=key[1],
            kind=relation_kind[key],
            edge_count=len(weights),
            total_weight=sum(weights),
        )
        for key, weights in grouped.items()
    }


def route_neighbors_for(
    registry: CompartmentRegistry,
    graph: ConnectivityGraph,
    compartment_id: CompartmentId | str,
    *,
    active_only: bool = True,
) -> tuple[CompartmentRouteSummary, ...]:
    """Return neighboring compartments sorted by descending total route weight."""

    identifier = CompartmentId(str(compartment_id))
    summaries = []
    for summary in build_compartment_route_map(registry, graph, active_only=active_only).values():
        if summary.source_compartment == identifier or summary.target_compartment == identifier:
            summaries.append(summary)
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (-summary.total_weight, str(summary.source_compartment), str(summary.target_compartment)),
        )
    )


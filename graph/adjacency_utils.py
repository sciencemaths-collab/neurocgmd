"""Utility functions for graph adjacency and connectivity queries."""

from __future__ import annotations

from collections import defaultdict, deque

from core.exceptions import ContractValidationError
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState


def build_edge_lookup(
    edges: tuple[DynamicEdgeState, ...] | list[DynamicEdgeState],
    *,
    active_only: bool = True,
) -> dict[tuple[int, int], DynamicEdgeState]:
    """Return a lookup from normalized pair to edge state."""

    lookup: dict[tuple[int, int], DynamicEdgeState] = {}
    for edge in edges:
        if active_only and not edge.active:
            continue
        pair = edge.normalized_pair()
        if pair in lookup:
            raise ContractValidationError(f"Duplicate graph edge detected for pair {pair}.")
        lookup[pair] = edge
    return lookup


def build_adjacency_map(
    particle_count: int,
    edges: tuple[DynamicEdgeState, ...] | list[DynamicEdgeState],
    *,
    active_only: bool = True,
) -> dict[int, tuple[int, ...]]:
    """Build a deterministic undirected adjacency map from graph edges."""

    if particle_count < 0:
        raise ContractValidationError("particle_count must be non-negative.")
    adjacency: dict[int, set[int]] = {index: set() for index in range(particle_count)}
    for edge in edges:
        if active_only and not edge.active:
            continue
        source, target = edge.normalized_pair()
        if target >= particle_count:
            raise ContractValidationError(
                f"Graph edge {edge.normalized_pair()} exceeds particle_count={particle_count}."
            )
        adjacency[source].add(target)
        adjacency[target].add(source)
    return {index: tuple(sorted(neighbors)) for index, neighbors in adjacency.items()}


def connected_components(
    particle_count: int,
    edges: tuple[DynamicEdgeState, ...] | list[DynamicEdgeState],
    *,
    active_only: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Return connected components of the active graph."""

    adjacency = build_adjacency_map(particle_count, edges, active_only=active_only)
    remaining = set(range(particle_count))
    components: list[tuple[int, ...]] = []

    while remaining:
        root = min(remaining)
        queue: deque[int] = deque([root])
        component: list[int] = []
        remaining.remove(root)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))

    components.sort(key=lambda component: component[0] if component else -1)
    return tuple(components)


def partition_edges_by_kind(
    edges: tuple[DynamicEdgeState, ...] | list[DynamicEdgeState],
    *,
    active_only: bool = True,
) -> dict[DynamicEdgeKind, tuple[DynamicEdgeState, ...]]:
    """Group edges by kind."""

    grouped: dict[DynamicEdgeKind, list[DynamicEdgeState]] = defaultdict(list)
    for edge in edges:
        if active_only and not edge.active:
            continue
        grouped[edge.kind].append(edge)
    return {
        kind: tuple(sorted(group, key=lambda edge: edge.normalized_pair()))
        for kind, group in grouped.items()
    }


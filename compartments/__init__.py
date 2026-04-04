"""Compartment abstractions for modular molecular regions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compartments.domain_models import CompartmentDomain, CompartmentRole
    from compartments.registry import CompartmentRegistry
    from compartments.routing import (
        CompartmentRouteKind,
        CompartmentRouteSummary,
        EdgeRouteAssignment,
        build_compartment_route_map,
        classify_graph_edges,
        route_neighbors_for,
    )

__all__ = [
    "CompartmentDomain",
    "CompartmentRegistry",
    "CompartmentRole",
    "CompartmentRouteKind",
    "CompartmentRouteSummary",
    "EdgeRouteAssignment",
    "build_compartment_route_map",
    "classify_graph_edges",
    "route_neighbors_for",
]

_LAZY_EXPORTS = {
    "CompartmentDomain": ("compartments.domain_models", "CompartmentDomain"),
    "CompartmentRegistry": ("compartments.registry", "CompartmentRegistry"),
    "CompartmentRole": ("compartments.domain_models", "CompartmentRole"),
    "CompartmentRouteKind": ("compartments.routing", "CompartmentRouteKind"),
    "CompartmentRouteSummary": ("compartments.routing", "CompartmentRouteSummary"),
    "EdgeRouteAssignment": ("compartments.routing", "EdgeRouteAssignment"),
    "build_compartment_route_map": ("compartments.routing", "build_compartment_route_map"),
    "classify_graph_edges": ("compartments.routing", "classify_graph_edges"),
    "route_neighbors_for": ("compartments.routing", "route_neighbors_for"),
}


def __getattr__(name: str) -> object:
    """Resolve compartment exports lazily to avoid package-level import cycles."""

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

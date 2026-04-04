"""Adaptive connectivity graph infrastructure."""

from graph.adjacency_utils import (
    build_adjacency_map,
    build_edge_lookup,
    connected_components,
    partition_edges_by_kind,
)
from graph.connectivity_rules import DistanceBandConnectivityRule
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph, ConnectivityGraphManager
from graph.message_passing import MessagePassingGraphUpdater, MessagePassingLayer

__all__ = [
    "ConnectivityGraph",
    "ConnectivityGraphManager",
    "DistanceBandConnectivityRule",
    "DynamicEdgeKind",
    "DynamicEdgeState",
    "MessagePassingGraphUpdater",
    "MessagePassingLayer",
    "build_adjacency_map",
    "build_edge_lookup",
    "connected_components",
    "partition_edges_by_kind",
]


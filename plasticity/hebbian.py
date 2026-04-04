"""Hebbian-style edge growth for adaptive graph plasticity."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from core.exceptions import ContractValidationError
from core.state import SimulationState
from graph.adjacency_utils import build_edge_lookup
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from plasticity.traces import PairTraceState, build_trace_lookup
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class HebbianGrowthRule:
    """Create adaptive edges for strongly coactive nearby pairs."""

    growth_threshold: float = 0.6
    initial_weight: float = 0.25
    local_distance_cutoff: float = 1.0
    long_range_distance_cutoff: float = 2.5
    max_new_edges_per_step: int = 4

    name: str = "hebbian_growth_rule"
    classification: str = "[proposed novel]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not (0.0 <= self.growth_threshold <= 1.0):
            issues.append("growth_threshold must lie in the interval [0, 1].")
        if not (0.0 < self.initial_weight <= 1.0):
            issues.append("initial_weight must lie in the interval (0, 1].")
        if self.local_distance_cutoff <= 0.0:
            issues.append("local_distance_cutoff must be positive.")
        if self.long_range_distance_cutoff <= self.local_distance_cutoff:
            issues.append("long_range_distance_cutoff must exceed local_distance_cutoff.")
        if self.max_new_edges_per_step < 0:
            issues.append("max_new_edges_per_step must be non-negative.")
        return tuple(issues)

    def apply(
        self,
        state: SimulationState,
        topology: SystemTopology,
        graph: ConnectivityGraph,
        traces: tuple[PairTraceState, ...],
    ) -> ConnectivityGraph:
        """Return a graph snapshot with new adaptive edges grown from pair traces."""

        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        existing_pairs = build_edge_lookup(graph.edges, active_only=False)
        trace_lookup = build_trace_lookup(traces)
        bonded_pairs = {bond.normalized_pair() for bond in topology.bonds}
        candidates: list[DynamicEdgeState] = []

        for pair, trace in trace_lookup.items():
            if pair in existing_pairs or pair in bonded_pairs:
                continue
            if trace.coactivity_level < self.growth_threshold:
                continue
            distance = dist(
                state.particles.positions[pair[0]],
                state.particles.positions[pair[1]],
            )
            if distance > self.long_range_distance_cutoff:
                continue
            kind = (
                DynamicEdgeKind.ADAPTIVE_LOCAL
                if distance <= self.local_distance_cutoff
                else DynamicEdgeKind.ADAPTIVE_LONG_RANGE
            )
            candidates.append(
                DynamicEdgeState(
                    source_index=pair[0],
                    target_index=pair[1],
                    kind=kind,
                    weight=max(self.initial_weight, trace.coactivity_level),
                    distance=distance,
                    created_step=state.step,
                    last_updated_step=state.step,
                    metadata={
                        "growth_rule": self.name,
                        "coactivity_level": trace.coactivity_level,
                    },
                )
            )

        candidates.sort(key=lambda edge: (-edge.weight, edge.distance, edge.normalized_pair()))
        new_edges = tuple(candidates[: self.max_new_edges_per_step])
        return ConnectivityGraph(
            name=graph.name,
            classification=graph.classification,
            particle_count=graph.particle_count,
            step=graph.step,
            edges=graph.edges + new_edges,
            metadata=graph.metadata.with_updates({"grown_edge_count": len(new_edges)}),
        )


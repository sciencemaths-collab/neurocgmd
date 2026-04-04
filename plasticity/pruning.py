"""Pruning and deactivation rules for adaptive graph edges."""

from __future__ import annotations

from dataclasses import dataclass

from core.exceptions import ContractValidationError
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from plasticity.traces import PairTraceState, build_trace_lookup


@dataclass(frozen=True, slots=True)
class PruningRule:
    """Deactivate weak adaptive edges when support decays below safe thresholds."""

    weight_threshold: float = 0.12
    activity_floor: float = 0.15
    min_persistence_to_keep: int = 1

    name: str = "pruning_rule"
    classification: str = "[proposed novel]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not (0.0 <= self.weight_threshold <= 1.0):
            issues.append("weight_threshold must lie in the interval [0, 1].")
        if not (0.0 <= self.activity_floor <= 1.0):
            issues.append("activity_floor must lie in the interval [0, 1].")
        if self.min_persistence_to_keep < 0:
            issues.append("min_persistence_to_keep must be non-negative.")
        return tuple(issues)

    def apply(
        self,
        graph: ConnectivityGraph,
        traces: tuple[PairTraceState, ...],
    ) -> ConnectivityGraph:
        """Return a graph snapshot with weak unsupported adaptive edges deactivated."""

        trace_lookup = build_trace_lookup(traces)
        updated_edges: list[DynamicEdgeState] = []
        pruned_count = 0

        for edge in graph.edges:
            if edge.kind == DynamicEdgeKind.STRUCTURAL_LOCAL:
                updated_edges.append(edge)
                continue

            trace = trace_lookup.get(edge.normalized_pair())
            activity_level = trace.activity_level if trace is not None else 0.0
            persistence = trace.persistence if trace is not None else 0
            should_prune = (
                edge.active
                and edge.weight <= self.weight_threshold
                and activity_level <= self.activity_floor
                and persistence < self.min_persistence_to_keep
            )
            if should_prune:
                pruned_count += 1
            updated_edges.append(
                DynamicEdgeState(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind,
                    weight=edge.weight,
                    distance=edge.distance,
                    created_step=edge.created_step,
                    last_updated_step=graph.step,
                    active=not should_prune and edge.active,
                    metadata=edge.metadata.with_updates(
                        {
                            "pruning_rule": self.name,
                            "pruned": should_prune,
                            "activity_level": activity_level,
                            "persistence": persistence,
                        }
                    ),
                )
            )

        return ConnectivityGraph(
            name=graph.name,
            classification=graph.classification,
            particle_count=graph.particle_count,
            step=graph.step,
            edges=tuple(updated_edges),
            metadata=graph.metadata.with_updates({"pruned_edge_count": pruned_count}),
        )


"""Bounded reinforcement and weakening of adaptive graph edges."""

from __future__ import annotations

from dataclasses import dataclass

from core.exceptions import ContractValidationError
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from graph.graph_manager import ConnectivityGraph
from plasticity.traces import PairTraceState, build_trace_lookup


@dataclass(frozen=True, slots=True)
class ReinforcementRule:
    """Bounded weight update rule inspired by Hebbian reinforcement."""

    activity_threshold: float = 0.35
    positive_rate: float = 0.25
    negative_rate: float = 0.1
    max_delta: float = 0.2
    min_adaptive_weight: float = 0.05
    max_adaptive_weight: float = 1.0

    name: str = "reinforcement_rule"
    classification: str = "[proposed novel]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not (0.0 <= self.activity_threshold <= 1.0):
            issues.append("activity_threshold must lie in the interval [0, 1].")
        for name, value in (
            ("positive_rate", self.positive_rate),
            ("negative_rate", self.negative_rate),
            ("max_delta", self.max_delta),
            ("min_adaptive_weight", self.min_adaptive_weight),
            ("max_adaptive_weight", self.max_adaptive_weight),
        ):
            if value < 0.0:
                issues.append(f"{name} must be non-negative.")
        if self.max_adaptive_weight <= 0.0:
            issues.append("max_adaptive_weight must be positive.")
        if self.min_adaptive_weight >= self.max_adaptive_weight:
            issues.append("min_adaptive_weight must be less than max_adaptive_weight.")
        return tuple(issues)

    def apply(
        self,
        graph: ConnectivityGraph,
        traces: tuple[PairTraceState, ...],
    ) -> ConnectivityGraph:
        """Return a graph snapshot with adaptive edge weights updated."""

        trace_lookup = build_trace_lookup(traces)
        updated_edges: list[DynamicEdgeState] = []
        reinforced_count = 0
        weakened_count = 0

        for edge in graph.edges:
            if edge.kind == DynamicEdgeKind.STRUCTURAL_LOCAL:
                updated_edges.append(edge)
                continue

            trace = trace_lookup.get(edge.normalized_pair())
            activity = trace.coactivity_level if trace is not None else 0.0
            delta_raw = (
                self.positive_rate * max(0.0, activity - self.activity_threshold)
                - self.negative_rate * max(0.0, self.activity_threshold - activity)
            )
            delta = max(-self.max_delta, min(self.max_delta, delta_raw))
            if delta > 0.0:
                reinforced_count += 1
            elif delta < 0.0:
                weakened_count += 1
            updated_weight = max(
                self.min_adaptive_weight,
                min(self.max_adaptive_weight, edge.weight + delta),
            )
            updated_edges.append(
                DynamicEdgeState(
                    source_index=edge.source_index,
                    target_index=edge.target_index,
                    kind=edge.kind,
                    weight=updated_weight,
                    distance=edge.distance,
                    created_step=edge.created_step,
                    last_updated_step=graph.step,
                    active=edge.active,
                    metadata=edge.metadata.with_updates(
                        {
                            "reinforcement_rule": self.name,
                            "previous_weight": edge.weight,
                            "activity": activity,
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
            metadata=graph.metadata.with_updates(
                {
                    "reinforced_edge_count": reinforced_count,
                    "weakened_edge_count": weakened_count,
                }
            ),
        )


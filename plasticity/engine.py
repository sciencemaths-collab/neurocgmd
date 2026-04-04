"""Coordinator for applying plasticity traces and bounded rewiring updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.types import FrozenMetadata
from core.state import SimulationState
from graph.graph_manager import ConnectivityGraph
from plasticity.hebbian import HebbianGrowthRule
from plasticity.pruning import PruningRule
from plasticity.reinforcement import ReinforcementRule
from plasticity.traces import PairTraceState, update_pair_traces
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class PlasticityUpdateResult:
    """Combined result of one bounded plasticity update pass."""

    graph: ConnectivityGraph
    traces: tuple[PairTraceState, ...]
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)


@dataclass(slots=True)
class PlasticityEngine:
    """Apply trace accumulation, reinforcement, pruning, and Hebbian growth."""

    trace_decay: float = 0.75
    reinforcement_rule: ReinforcementRule = field(default_factory=ReinforcementRule)
    pruning_rule: PruningRule = field(default_factory=PruningRule)
    growth_rule: HebbianGrowthRule = field(default_factory=HebbianGrowthRule)

    name: str = "plasticity_engine"
    classification: str = "[proposed novel]"

    def update(
        self,
        state: SimulationState,
        topology: SystemTopology,
        graph: ConnectivityGraph,
        *,
        activity_signals: Mapping[tuple[int, int], float] | None = None,
        previous_traces: tuple[PairTraceState, ...] = (),
    ) -> PlasticityUpdateResult:
        """Run one deterministic plasticity update pass."""

        traces = update_pair_traces(
            current_step=state.step,
            graph=graph,
            previous_traces=previous_traces,
            activity_signals=activity_signals,
            decay=self.trace_decay,
        )
        reinforced_graph = self.reinforcement_rule.apply(graph, traces)
        pruned_graph = self.pruning_rule.apply(reinforced_graph, traces)
        grown_graph = self.growth_rule.apply(state, topology, pruned_graph, traces)
        metadata = FrozenMetadata(
            {
                "trace_count": len(traces),
                "active_edges_before": len(graph.active_edges()),
                "active_edges_after": len(grown_graph.active_edges()),
            }
        )
        return PlasticityUpdateResult(
            graph=grown_graph,
            traces=traces,
            metadata=metadata,
        )


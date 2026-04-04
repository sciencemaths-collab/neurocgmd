"""Foundation rules for updating adaptive graph connectivity."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from core.exceptions import ContractValidationError
from core.interfaces import ValidatableComponent
from core.state import SimulationState
from graph.adjacency_utils import build_edge_lookup
from graph.edge_models import DynamicEdgeKind, DynamicEdgeState
from topology.system_topology import SystemTopology


@dataclass(frozen=True, slots=True)
class DistanceBandConnectivityRule(ValidatableComponent):
    """Conservative distance-band rule for the first adaptive graph layer."""

    local_distance_cutoff: float
    long_range_distance_cutoff: float
    min_weight: float = 0.05
    weight_inertia: float = 0.25
    include_structural_edges: bool = True

    name: str = "distance_band_connectivity_rule"
    classification: str = "[adapted]"

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.local_distance_cutoff <= 0.0:
            issues.append("local_distance_cutoff must be positive.")
        if self.long_range_distance_cutoff <= self.local_distance_cutoff:
            issues.append("long_range_distance_cutoff must exceed local_distance_cutoff.")
        if not (0.0 < self.min_weight <= 1.0):
            issues.append("min_weight must lie in the interval (0, 1].")
        if not (0.0 <= self.weight_inertia <= 1.0):
            issues.append("weight_inertia must lie in the interval [0, 1].")
        return tuple(issues)

    def propose_edges(
        self,
        state: SimulationState,
        topology: SystemTopology,
        *,
        previous_edges: tuple[DynamicEdgeState, ...] = (),
    ) -> tuple[DynamicEdgeState, ...]:
        """Return the updated graph edge set for the current physical state."""

        issues = topology.validate_against_particle_state(state.particles)
        if issues:
            raise ContractValidationError("; ".join(issues))

        bonded_pairs = {bond.normalized_pair() for bond in topology.bonds}
        previous_lookup = build_edge_lookup(previous_edges, active_only=False)
        edges: list[DynamicEdgeState] = []

        for index_a in range(state.particle_count):
            for index_b in range(index_a + 1, state.particle_count):
                pair = (index_a, index_b)
                distance = dist(
                    state.particles.positions[index_a],
                    state.particles.positions[index_b],
                )
                if pair in bonded_pairs:
                    if self.include_structural_edges:
                        previous_edge = previous_lookup.get(pair)
                        edges.append(
                            DynamicEdgeState(
                                source_index=index_a,
                                target_index=index_b,
                                kind=DynamicEdgeKind.STRUCTURAL_LOCAL,
                                weight=1.0,
                                distance=distance,
                                created_step=(
                                    previous_edge.created_step if previous_edge else state.step
                                ),
                                last_updated_step=state.step,
                                metadata={"rule": self.name, "source": "topology_bond"},
                            )
                        )
                    continue

                kind: DynamicEdgeKind | None = None
                proposed_weight: float | None = None
                if distance <= self.local_distance_cutoff:
                    kind = DynamicEdgeKind.ADAPTIVE_LOCAL
                    proposed_weight = self._proximity_weight(distance, self.local_distance_cutoff)
                elif distance <= self.long_range_distance_cutoff:
                    kind = DynamicEdgeKind.ADAPTIVE_LONG_RANGE
                    proposed_weight = self._proximity_weight(distance, self.long_range_distance_cutoff)

                if kind is None or proposed_weight is None or proposed_weight < self.min_weight:
                    continue

                previous_edge = previous_lookup.get(pair)
                if previous_edge and previous_edge.kind != DynamicEdgeKind.STRUCTURAL_LOCAL:
                    weight = (
                        self.weight_inertia * previous_edge.weight
                        + (1.0 - self.weight_inertia) * proposed_weight
                    )
                    created_step = previous_edge.created_step
                else:
                    weight = proposed_weight
                    created_step = state.step

                edges.append(
                    DynamicEdgeState(
                        source_index=index_a,
                        target_index=index_b,
                        kind=kind,
                        weight=max(self.min_weight, min(1.0, weight)),
                        distance=distance,
                        created_step=created_step,
                        last_updated_step=state.step,
                        metadata={"rule": self.name, "source": "distance_band"},
                    )
                )

        edges.sort(key=lambda edge: (edge.normalized_pair(), edge.kind.value))
        return tuple(edges)

    @staticmethod
    def _proximity_weight(distance: float, cutoff: float) -> float:
        return max(0.0, min(1.0, 1.0 - distance / cutoff))


"""Deterministic local region selection for qcloud refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from compartments.registry import CompartmentRegistry
from compartments.routing import CompartmentRouteKind, classify_graph_edges
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.types import CompartmentId, FrozenMetadata, RegionId
from graph.edge_models import DynamicEdgeKind
from graph.graph_manager import ConnectivityGraph
from memory.trace_store import TraceRecord
from qcloud.cloud_state import RefinementRegion, RegionTriggerKind
from topology.system_topology import SystemTopology

_MEMORY_PRIORITY_TAGS = frozenset({"instability", "priority", "qcloud", "refine"})


def _normalize_focus_compartments(compartment_ids: Sequence[CompartmentId | str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for identifier in compartment_ids or ():
        value = str(identifier).strip()
        if not value:
            raise ContractValidationError("focus_compartments must contain non-empty identifiers.")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _memory_priority_enabled(trace_record: TraceRecord | None) -> bool:
    if trace_record is None:
        return False
    if _MEMORY_PRIORITY_TAGS & set(trace_record.tags):
        return True
    return bool(trace_record.metadata["qcloud_priority"]) if "qcloud_priority" in trace_record.metadata else False


def _expand_seed_region(
    topology: SystemTopology,
    seed_particles: tuple[int, int],
    *,
    max_region_size: int,
    bonded_neighbor_hops: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    seen: set[int] = set()
    frontier = list(seed_particles)
    for particle_index in frontier:
        if particle_index not in seen and len(selected) < max_region_size:
            selected.append(particle_index)
            seen.add(particle_index)

    for _ in range(bonded_neighbor_hops):
        next_frontier: list[int] = []
        for particle_index in frontier:
            for neighbor_index in topology.bonded_neighbors(particle_index):
                if neighbor_index not in seen and len(selected) < max_region_size:
                    selected.append(neighbor_index)
                    seen.add(neighbor_index)
                    next_frontier.append(neighbor_index)
        if not next_frontier or len(selected) >= max_region_size:
            break
        frontier = next_frontier
    return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class RegionSelectionPolicy(ValidatableComponent):
    """Deterministic heuristic policy for early qcloud-region selection."""

    max_regions: int = 2
    max_region_size: int = 6
    bonded_neighbor_hops: int = 1
    min_region_score: float = 0.5
    inter_compartment_bonus: float = 0.4
    long_range_bonus: float = 0.3
    memory_priority_bonus: float = 0.6
    focus_compartment_bonus: float = 0.25

    def __post_init__(self) -> None:
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.max_regions <= 0:
            issues.append("max_regions must be strictly positive.")
        if self.max_region_size < 2:
            issues.append("max_region_size must be at least 2.")
        if self.bonded_neighbor_hops < 0:
            issues.append("bonded_neighbor_hops must be non-negative.")
        for field_name in (
            "min_region_score",
            "inter_compartment_bonus",
            "long_range_bonus",
            "memory_priority_bonus",
            "focus_compartment_bonus",
        ):
            if getattr(self, field_name) < 0.0:
                issues.append(f"{field_name} must be non-negative.")
        return tuple(issues)


@dataclass(slots=True)
class LocalRegionSelector(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Select bounded local qcloud refinement regions from current adaptive state."""

    policy: RegionSelectionPolicy = field(default_factory=RegionSelectionPolicy)
    name: str = "local_region_selector"
    classification: str = "[hybrid]"

    def describe_role(self) -> str:
        return (
            "Selects bounded local refinement regions from adaptive graph, "
            "compartment, and memory signals without taking ownership of the simulation loop."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "topology/system_topology.py",
            "graph/graph_manager.py",
            "memory/trace_store.py",
            "compartments/routing.py",
            "qcloud/cloud_state.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/qcloud_framework.md",
            "docs/sections/section_10_quantum_cloud_framework.md",
        )

    def validate(self) -> tuple[str, ...]:
        return self.policy.validate()

    def select_regions(
        self,
        state: SimulationState,
        topology: SystemTopology,
        graph: ConnectivityGraph,
        *,
        compartments: CompartmentRegistry | None = None,
        trace_record: TraceRecord | None = None,
        focus_compartments: Sequence[CompartmentId | str] = (),
        correction_priority_scores: dict[int, float] | None = None,
    ) -> tuple[RefinementRegion, ...]:
        if topology.particle_count != state.particle_count:
            raise ContractValidationError("SystemTopology particle_count must match the SimulationState particle count.")
        if graph.particle_count != state.particle_count:
            raise ContractValidationError("ConnectivityGraph particle_count must match the SimulationState particle count.")
        if graph.step != state.step:
            raise ContractValidationError("ConnectivityGraph step must match the SimulationState step.")
        if compartments is not None and compartments.particle_count != state.particle_count:
            raise ContractValidationError("CompartmentRegistry particle_count must match the SimulationState particle count.")
        if trace_record is not None and trace_record.state_id != state.provenance.state_id:
            raise ContractValidationError("trace_record.state_id must match the current SimulationState state_id.")

        focus_ids = set(_normalize_focus_compartments(focus_compartments))
        route_lookup = {}
        if compartments is not None:
            route_lookup = {
                assignment.pair: assignment
                for assignment in classify_graph_edges(compartments, graph)
            }

        memory_priority = _memory_priority_enabled(trace_record)
        candidates: list[tuple[float, tuple[int, int], RefinementRegion]] = []
        for edge in graph.active_edges():
            if edge.kind == DynamicEdgeKind.STRUCTURAL_LOCAL:
                continue

            score = edge.weight
            triggers = [RegionTriggerKind.ADAPTIVE_EDGE]
            if edge.kind == DynamicEdgeKind.ADAPTIVE_LONG_RANGE:
                score += self.policy.long_range_bonus

            route_assignment = route_lookup.get(edge.normalized_pair())
            compartment_ids: list[str] = []
            if route_assignment is not None:
                compartment_ids.extend(str(identifier) for identifier in route_assignment.source_compartments)
                compartment_ids.extend(str(identifier) for identifier in route_assignment.target_compartments)
                if route_assignment.kind == CompartmentRouteKind.INTER:
                    score += self.policy.inter_compartment_bonus
                    triggers.append(RegionTriggerKind.INTER_COMPARTMENT)

            if memory_priority:
                score += self.policy.memory_priority_bonus
                triggers.append(RegionTriggerKind.MEMORY_PRIORITY)

            if focus_ids and focus_ids.intersection(compartment_ids):
                score += self.policy.focus_compartment_bonus
                triggers.append(RegionTriggerKind.COMPARTMENT_FOCUS)

            # Correction feedback: boost score for edges involving particles
            # that previously received large QCloud corrections — the CG model
            # is less accurate there and needs more refinement.
            if correction_priority_scores:
                pair = edge.normalized_pair()
                p_score_a = correction_priority_scores.get(pair[0], 0.0)
                p_score_b = correction_priority_scores.get(pair[1], 0.0)
                correction_boost = max(p_score_a, p_score_b) * 0.5
                score += correction_boost

            if score < self.policy.min_region_score:
                continue

            particle_indices = _expand_seed_region(
                topology,
                edge.normalized_pair(),
                max_region_size=self.policy.max_region_size,
                bonded_neighbor_hops=self.policy.bonded_neighbor_hops,
            )
            if not particle_indices:
                continue

            if not compartment_ids and compartments is not None:
                compartment_ids = [
                    str(domain.compartment_id)
                    for particle_index in particle_indices
                    for domain in compartments.domains_for_particle(particle_index)
                ]

            candidate_index = len(candidates) + 1
            region = RefinementRegion(
                region_id=RegionId(f"region-step{state.step:08d}-{candidate_index:03d}"),
                state_id=state.provenance.state_id,
                particle_indices=particle_indices,
                seed_pairs=(edge.normalized_pair(),),
                compartment_ids=tuple(compartment_ids),
                trigger_kinds=tuple(triggers),
                score=score,
                metadata=FrozenMetadata(
                    {
                        "edge_kind": edge.kind.value,
                        "edge_weight": edge.weight,
                        "graph_step": graph.step,
                    }
                ),
            )
            candidates.append((score, edge.normalized_pair(), region))

        selected_regions: list[RefinementRegion] = []
        seen_particle_sets: set[tuple[int, ...]] = set()
        for _, _, region in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if region.particle_indices in seen_particle_sets:
                continue
            selected_regions.append(region)
            seen_particle_sets.add(region.particle_indices)
            if len(selected_regions) >= self.policy.max_regions:
                break
        return tuple(selected_regions)

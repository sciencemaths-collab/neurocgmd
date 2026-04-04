"""Long-horizon trace storage layered on top of immutable simulation states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from compartments.registry import CompartmentRegistry
from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.state_registry import SimulationStateRegistry
from core.types import FrozenMetadata, SimulationId, StateId, coerce_scalar
from graph.edge_models import DynamicEdgeKind
from graph.graph_manager import ConnectivityGraph
from plasticity.traces import PairTraceState


def _normalize_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags or ():
        tag = str(raw_tag).strip()
        if not tag:
            raise ContractValidationError("tags must contain non-empty strings.")
        if tag not in seen:
            normalized.append(tag)
            seen.add(tag)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class TraceRecord(ValidatableComponent):
    """Compact long-horizon summary for one registered simulation state."""

    record_id: str
    simulation_id: SimulationId
    state_id: StateId
    parent_state_id: StateId | None
    stage: str
    step: int
    time: float
    particle_count: int
    kinetic_energy: float
    potential_energy: float | None
    active_edge_count: int = 0
    structural_edge_count: int = 0
    adaptive_edge_count: int = 0
    plasticity_trace_count: int = 0
    compartment_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "simulation_id", SimulationId(str(self.simulation_id)))
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        if self.parent_state_id is not None:
            object.__setattr__(self, "parent_state_id", StateId(str(self.parent_state_id)))
        object.__setattr__(self, "time", coerce_scalar(self.time, "time"))
        object.__setattr__(self, "kinetic_energy", coerce_scalar(self.kinetic_energy, "kinetic_energy"))
        if self.potential_energy is not None:
            object.__setattr__(
                self,
                "potential_energy",
                coerce_scalar(self.potential_energy, "potential_energy"),
            )
        object.__setattr__(
            self,
            "compartment_ids",
            tuple(str(identifier).strip() for identifier in self.compartment_ids),
        )
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    @property
    def compartment_count(self) -> int:
        return len(self.compartment_ids)

    def total_energy(self) -> float | None:
        if self.potential_energy is None:
            return None
        return self.kinetic_energy + self.potential_energy

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.record_id.strip():
            issues.append("record_id must be a non-empty string.")
        if not str(self.simulation_id).strip():
            issues.append("simulation_id must be a non-empty string.")
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if not self.stage.strip():
            issues.append("stage must be a non-empty string.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        if self.time < 0.0:
            issues.append("time must be non-negative.")
        if self.particle_count <= 0:
            issues.append("particle_count must be positive.")
        if self.kinetic_energy < 0.0:
            issues.append("kinetic_energy must be non-negative.")
        for field_name in (
            "active_edge_count",
            "structural_edge_count",
            "adaptive_edge_count",
            "plasticity_trace_count",
        ):
            if getattr(self, field_name) < 0:
                issues.append(f"{field_name} must be non-negative.")
        if self.structural_edge_count + self.adaptive_edge_count > self.active_edge_count:
            issues.append(
                "active_edge_count must be at least the sum of structural_edge_count and adaptive_edge_count."
            )
        if len(self.compartment_ids) != len(set(self.compartment_ids)):
            issues.append("compartment_ids must be unique when provided.")
        if any(not identifier for identifier in self.compartment_ids):
            issues.append("compartment_ids must contain non-empty strings.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "simulation_id": str(self.simulation_id),
            "state_id": str(self.state_id),
            "parent_state_id": str(self.parent_state_id) if self.parent_state_id is not None else None,
            "stage": self.stage,
            "step": self.step,
            "time": self.time,
            "particle_count": self.particle_count,
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "active_edge_count": self.active_edge_count,
            "structural_edge_count": self.structural_edge_count,
            "adaptive_edge_count": self.adaptive_edge_count,
            "plasticity_trace_count": self.plasticity_trace_count,
            "compartment_ids": list(self.compartment_ids),
            "tags": list(self.tags),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TraceRecord":
        parent_state_id = data.get("parent_state_id")
        return cls(
            record_id=str(data["record_id"]),
            simulation_id=SimulationId(str(data["simulation_id"])),
            state_id=StateId(str(data["state_id"])),
            parent_state_id=StateId(str(parent_state_id)) if parent_state_id else None,
            stage=str(data["stage"]),
            step=int(data["step"]),
            time=float(data["time"]),
            particle_count=int(data["particle_count"]),
            kinetic_energy=float(data["kinetic_energy"]),
            potential_energy=(
                float(data["potential_energy"]) if data.get("potential_energy") is not None else None
            ),
            active_edge_count=int(data.get("active_edge_count", 0)),
            structural_edge_count=int(data.get("structural_edge_count", 0)),
            adaptive_edge_count=int(data.get("adaptive_edge_count", 0)),
            plasticity_trace_count=int(data.get("plasticity_trace_count", 0)),
            compartment_ids=tuple(str(identifier) for identifier in data.get("compartment_ids", ())),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(slots=True)
class TraceStore(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Append-only trace summaries keyed by immutable state identifiers."""

    name: str = "trace_store"
    classification: str = "[hybrid]"
    simulation_id: SimulationId | None = None
    _records: dict[StateId, TraceRecord] = field(default_factory=dict, init=False, repr=False)
    _record_order: list[StateId] = field(default_factory=list, init=False, repr=False)
    _record_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.simulation_id is not None:
            self.simulation_id = SimulationId(str(self.simulation_id))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def __len__(self) -> int:
        return len(self._record_order)

    def describe_role(self) -> str:
        return (
            "Stores longer-horizon summary records for simulation states without "
            "taking ownership of state creation, integration, or graph updates."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "core/state_registry.py",
            "graph/graph_manager.py",
            "plasticity/traces.py",
            "compartments/registry.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/memory_replay.md",
            "docs/sections/section_09_memory_and_replay.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.simulation_id is not None and not str(self.simulation_id).strip():
            issues.append("simulation_id must be a non-empty string when provided.")
        for state_id in self._record_order:
            record = self._records[state_id]
            issues.extend(record.validate())
            if self.simulation_id is not None and record.simulation_id != self.simulation_id:
                issues.append(
                    f"TraceRecord {record.record_id} simulation_id does not match the store simulation_id."
                )
        return tuple(issues)

    def _record_id_exists(self, record_id: str) -> bool:
        return any(record.record_id == record_id for record in self._records.values())

    def _next_record_id(self) -> str:
        while True:
            self._record_counter += 1
            candidate = f"trace-{self._record_counter:06d}"
            if not self._record_id_exists(candidate):
                return candidate

    def _require_compatible_graph(self, state: SimulationState, graph: ConnectivityGraph) -> None:
        if graph.particle_count != state.particle_count:
            raise ContractValidationError("graph.particle_count must match the SimulationState particle count.")
        if graph.step != state.step:
            raise ContractValidationError("graph.step must match the SimulationState step.")

    def _require_compatible_compartments(
        self,
        state: SimulationState,
        compartments: CompartmentRegistry,
    ) -> None:
        if compartments.particle_count != state.particle_count:
            raise ContractValidationError(
                "compartments.particle_count must match the SimulationState particle count."
            )

    def append(self, record: TraceRecord) -> None:
        if self.simulation_id is None:
            self.simulation_id = record.simulation_id
        elif record.simulation_id != self.simulation_id:
            raise ContractValidationError("TraceRecord simulation_id does not match the store simulation_id.")
        if record.state_id in self._records:
            raise ContractValidationError(f"State {record.state_id} already has a trace record in this store.")
        if self._record_id_exists(record.record_id):
            raise ContractValidationError(f"TraceRecord {record.record_id} is already present in this store.")
        self._records[record.state_id] = record
        self._record_order.append(record.state_id)

    def append_state(
        self,
        state: SimulationState,
        *,
        graph: ConnectivityGraph | None = None,
        plasticity_traces: Sequence[PairTraceState] = (),
        compartments: CompartmentRegistry | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> TraceRecord:
        if graph is not None:
            self._require_compatible_graph(state, graph)
        if compartments is not None:
            self._require_compatible_compartments(state, compartments)

        active_edges = tuple(graph.active_edges()) if graph is not None else ()
        structural_edge_count = sum(
            edge.kind == DynamicEdgeKind.STRUCTURAL_LOCAL for edge in active_edges
        )
        adaptive_edge_count = sum(
            edge.kind in {DynamicEdgeKind.ADAPTIVE_LOCAL, DynamicEdgeKind.ADAPTIVE_LONG_RANGE}
            for edge in active_edges
        )
        compartment_ids = (
            tuple(str(domain.compartment_id) for domain in compartments.domains)
            if compartments is not None
            else ()
        )

        record = TraceRecord(
            record_id=self._next_record_id(),
            simulation_id=state.provenance.simulation_id,
            state_id=state.provenance.state_id,
            parent_state_id=state.provenance.parent_state_id,
            stage=state.provenance.stage,
            step=state.step,
            time=state.time,
            particle_count=state.particle_count,
            kinetic_energy=state.kinetic_energy(),
            potential_energy=state.potential_energy,
            active_edge_count=len(active_edges),
            structural_edge_count=structural_edge_count,
            adaptive_edge_count=adaptive_edge_count,
            plasticity_trace_count=len(tuple(plasticity_traces)),
            compartment_ids=compartment_ids,
            tags=tags,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )
        self.append(record)
        return record

    def append_from_registry(
        self,
        registry: SimulationStateRegistry,
        state_id: StateId,
        *,
        graph: ConnectivityGraph | None = None,
        plasticity_traces: Sequence[PairTraceState] = (),
        compartments: CompartmentRegistry | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> TraceRecord:
        state = registry.get_state(state_id)
        if state.provenance.simulation_id != registry.require_simulation_id():
            raise ContractValidationError("Requested state does not match the provided registry simulation.")
        return self.append_state(
            state,
            graph=graph,
            plasticity_traces=plasticity_traces,
            compartments=compartments,
            tags=tags,
            metadata=metadata,
        )

    def get_record(self, state_id: StateId) -> TraceRecord:
        return self._records[state_id]

    def state_ids(self) -> tuple[StateId, ...]:
        return tuple(self._record_order)

    def records(self) -> tuple[TraceRecord, ...]:
        return tuple(self._records[state_id] for state_id in self._record_order)

    def latest(self, limit: int = 1) -> tuple[TraceRecord, ...]:
        if limit < 0:
            raise ContractValidationError("limit must be non-negative.")
        if limit == 0:
            return ()
        recent_state_ids = self._record_order[-limit:]
        return tuple(self._records[state_id] for state_id in reversed(recent_state_ids))

    def records_by_tag(self, tag: str) -> tuple[TraceRecord, ...]:
        normalized_tag = tag.strip()
        if not normalized_tag:
            raise ContractValidationError("tag must be a non-empty string.")
        return tuple(record for record in self.records() if normalized_tag in record.tags)

    def between_steps(self, start_step: int, end_step: int) -> tuple[TraceRecord, ...]:
        if start_step < 0 or end_step < 0:
            raise ContractValidationError("step bounds must be non-negative.")
        if end_step < start_step:
            raise ContractValidationError("end_step must be greater than or equal to start_step.")
        return tuple(
            record for record in self.records() if start_step <= record.step <= end_step
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "simulation_id": str(self.simulation_id) if self.simulation_id is not None else None,
            "records": [record.to_dict() for record in self.records()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TraceStore":
        simulation_id = data.get("simulation_id")
        store = cls(
            name=str(data.get("name", "trace_store")),
            classification=str(data.get("classification", "[hybrid]")),
            simulation_id=SimulationId(str(simulation_id)) if simulation_id else None,
        )
        for item in data.get("records", ()):
            store.append(TraceRecord.from_dict(item))
        store._record_counter = len(store._record_order)
        return store

"""Deterministic bounded replay storage for later ML and control consumers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.types import FrozenMetadata, SimulationId, StateId, coerce_scalar
from memory.trace_store import TraceRecord, _normalize_tags


@dataclass(frozen=True, slots=True)
class ReplayItem(ValidatableComponent):
    """Replay-ready summary item derived from a trace record or checkpoint."""

    item_id: str
    simulation_id: SimulationId
    state_id: StateId
    step: int
    time: float
    score: float
    source_record_id: str | None = None
    episode_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "simulation_id", SimulationId(str(self.simulation_id)))
        object.__setattr__(self, "state_id", StateId(str(self.state_id)))
        object.__setattr__(self, "time", coerce_scalar(self.time, "time"))
        object.__setattr__(self, "score", coerce_scalar(self.score, "score"))
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.item_id.strip():
            issues.append("item_id must be a non-empty string.")
        if not str(self.simulation_id).strip():
            issues.append("simulation_id must be a non-empty string.")
        if not str(self.state_id).strip():
            issues.append("state_id must be a non-empty string.")
        if self.source_record_id is not None and not self.source_record_id.strip():
            issues.append("source_record_id must be non-empty when provided.")
        if self.episode_id is not None and not self.episode_id.strip():
            issues.append("episode_id must be non-empty when provided.")
        if self.step < 0:
            issues.append("step must be non-negative.")
        if self.time < 0.0:
            issues.append("time must be non-negative.")
        if self.score < 0.0:
            issues.append("score must be non-negative.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "simulation_id": str(self.simulation_id),
            "state_id": str(self.state_id),
            "step": self.step,
            "time": self.time,
            "score": self.score,
            "source_record_id": self.source_record_id,
            "episode_id": self.episode_id,
            "tags": list(self.tags),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReplayItem":
        return cls(
            item_id=str(data["item_id"]),
            simulation_id=SimulationId(str(data["simulation_id"])),
            state_id=StateId(str(data["state_id"])),
            step=int(data["step"]),
            time=float(data["time"]),
            score=float(data["score"]),
            source_record_id=(
                str(data["source_record_id"])
                if data.get("source_record_id") is not None
                else None
            ),
            episode_id=str(data["episode_id"]) if data.get("episode_id") is not None else None,
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ReplayBuffer(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Capacity-bounded deterministic replay store for trace-derived experiences."""

    capacity: int = 1024
    deduplicate_states: bool = True
    name: str = "replay_buffer"
    classification: str = "[hybrid]"
    simulation_id: SimulationId | None = None
    _items: dict[str, ReplayItem] = field(default_factory=dict, init=False, repr=False)
    _item_order: list[str] = field(default_factory=list, init=False, repr=False)
    _state_to_item_id: dict[StateId, str] = field(default_factory=dict, init=False, repr=False)
    _item_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.simulation_id is not None:
            self.simulation_id = SimulationId(str(self.simulation_id))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def __len__(self) -> int:
        return len(self._item_order)

    def describe_role(self) -> str:
        return (
            "Maintains a bounded replay set for later ML, control, and diagnostics "
            "without mutating the underlying state or trace registries."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "memory/trace_store.py",
            "core/state.py",
        )

    def documentation_paths(self) -> tuple[str, ...]:
        return (
            "docs/architecture/memory_replay.md",
            "docs/sections/section_09_memory_and_replay.md",
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.capacity <= 0:
            issues.append("capacity must be strictly positive.")
        if self.simulation_id is not None and not str(self.simulation_id).strip():
            issues.append("simulation_id must be a non-empty string when provided.")
        for item_id in self._item_order:
            item = self._items[item_id]
            issues.extend(item.validate())
            if self.simulation_id is not None and item.simulation_id != self.simulation_id:
                issues.append(
                    f"ReplayItem {item.item_id} simulation_id does not match the buffer simulation_id."
                )
        return tuple(issues)

    def _item_id_exists(self, item_id: str) -> bool:
        return item_id in self._items

    def _next_item_id(self) -> str:
        while True:
            self._item_counter += 1
            candidate = f"replay-{self._item_counter:06d}"
            if not self._item_id_exists(candidate):
                return candidate

    def _remove_item(self, item_id: str) -> None:
        item = self._items.pop(item_id)
        self._item_order.remove(item_id)
        self._state_to_item_id.pop(item.state_id, None)

    def _evict_if_needed(self) -> None:
        while len(self._item_order) > self.capacity:
            self._remove_item(self._item_order[0])

    def add_item(self, item: ReplayItem) -> None:
        if self.simulation_id is None:
            self.simulation_id = item.simulation_id
        elif item.simulation_id != self.simulation_id:
            raise ContractValidationError("ReplayItem simulation_id does not match the buffer simulation_id.")

        if item.item_id in self._items:
            raise ContractValidationError(f"ReplayItem {item.item_id} is already present in this buffer.")

        if self.deduplicate_states and item.state_id in self._state_to_item_id:
            self._remove_item(self._state_to_item_id[item.state_id])

        self._items[item.item_id] = item
        self._item_order.append(item.item_id)
        self._state_to_item_id[item.state_id] = item.item_id
        self._evict_if_needed()

    def add_from_record(
        self,
        record: TraceRecord,
        *,
        score: float,
        episode_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> ReplayItem:
        item = ReplayItem(
            item_id=self._next_item_id(),
            simulation_id=record.simulation_id,
            state_id=record.state_id,
            step=record.step,
            time=record.time,
            score=score,
            source_record_id=record.record_id,
            episode_id=episode_id,
            tags=record.tags + _normalize_tags(tags),
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )
        self.add_item(item)
        return item

    def get_item(self, item_id: str) -> ReplayItem:
        return self._items[item_id]

    def items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._items[item_id] for item_id in self._item_order)

    def latest(self, limit: int = 1) -> tuple[ReplayItem, ...]:
        if limit < 0:
            raise ContractValidationError("limit must be non-negative.")
        if limit == 0:
            return ()
        return tuple(self._items[item_id] for item_id in reversed(self._item_order[-limit:]))

    def highest_score(self, limit: int = 1) -> tuple[ReplayItem, ...]:
        if limit < 0:
            raise ContractValidationError("limit must be non-negative.")
        order_index = {item_id: index for index, item_id in enumerate(self._item_order)}
        ranked = sorted(
            self.items(),
            key=lambda item: (-item.score, -item.step, -item.time, -order_index[item.item_id]),
        )
        return tuple(ranked[:limit])

    def items_by_tag(self, tag: str, *, limit: int | None = None) -> tuple[ReplayItem, ...]:
        normalized_tag = tag.strip()
        if not normalized_tag:
            raise ContractValidationError("tag must be a non-empty string.")
        if limit is not None and limit < 0:
            raise ContractValidationError("limit must be non-negative when provided.")
        matching = tuple(
            item for item in reversed(self.items()) if normalized_tag in item.tags
        )
        return matching if limit is None else matching[:limit]

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "deduplicate_states": self.deduplicate_states,
            "name": self.name,
            "classification": self.classification,
            "simulation_id": str(self.simulation_id) if self.simulation_id is not None else None,
            "items": [item.to_dict() for item in self.items()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReplayBuffer":
        simulation_id = data.get("simulation_id")
        buffer = cls(
            capacity=int(data.get("capacity", 1024)),
            deduplicate_states=bool(data.get("deduplicate_states", True)),
            name=str(data.get("name", "replay_buffer")),
            classification=str(data.get("classification", "[hybrid]")),
            simulation_id=SimulationId(str(simulation_id)) if simulation_id else None,
        )
        for item_data in data.get("items", ()):
            buffer.add_item(ReplayItem.from_dict(item_data))
        buffer._item_counter = len(buffer._item_order)
        return buffer

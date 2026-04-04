"""Episode tracking for simulation intervals, instability windows, and replay groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from core.exceptions import ContractValidationError
from core.interfaces import ArchitecturalComponent, DocumentedComponent, ValidatableComponent
from core.state import SimulationState
from core.state_registry import SimulationStateRegistry
from core.types import FrozenMetadata, SimulationId, StateId
from memory.trace_store import _normalize_tags


class EpisodeKind(StrEnum):
    """Section 9 episode categories for grouped historical windows."""

    TRAJECTORY = "trajectory"
    INSTABILITY = "instability"
    CHECKPOINT_WINDOW = "checkpoint_window"
    COMPARTMENT_FOCUS = "compartment_focus"
    CUSTOM = "custom"


class EpisodeStatus(StrEnum):
    """Lifecycle state for one memory episode."""

    OPEN = "open"
    CLOSED = "closed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class EpisodeRecord(ValidatableComponent):
    """Immutable grouped state window for later replay and diagnostics."""

    episode_id: str
    simulation_id: SimulationId
    kind: EpisodeKind
    status: EpisodeStatus
    start_state_id: StateId
    end_state_id: StateId | None = None
    start_step: int = 0
    end_step: int | None = None
    state_ids: tuple[StateId, ...] = ()
    state_steps: tuple[int, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: FrozenMetadata = field(default_factory=FrozenMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "simulation_id", SimulationId(str(self.simulation_id)))
        object.__setattr__(self, "start_state_id", StateId(str(self.start_state_id)))
        if self.end_state_id is not None:
            object.__setattr__(self, "end_state_id", StateId(str(self.end_state_id)))
        normalized_state_ids = (
            tuple(StateId(str(state_id)) for state_id in self.state_ids)
            if self.state_ids
            else (StateId(str(self.start_state_id)),)
        )
        normalized_state_steps = (
            tuple(int(step) for step in self.state_steps)
            if self.state_steps
            else (int(self.start_step),)
        )
        object.__setattr__(self, "state_ids", normalized_state_ids)
        object.__setattr__(self, "state_steps", normalized_state_steps)
        object.__setattr__(self, "tags", _normalize_tags(self.tags))
        if not isinstance(self.metadata, FrozenMetadata):
            object.__setattr__(self, "metadata", FrozenMetadata(self.metadata))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def contains_state(self, state_id: StateId | str) -> bool:
        return StateId(str(state_id)) in self.state_ids

    def duration_steps(self) -> int | None:
        if self.end_step is None:
            return None
        return self.end_step - self.start_step

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not self.episode_id.strip():
            issues.append("episode_id must be a non-empty string.")
        if not str(self.simulation_id).strip():
            issues.append("simulation_id must be a non-empty string.")
        if self.start_step < 0:
            issues.append("start_step must be non-negative.")
        if len(self.state_ids) != len(self.state_steps):
            issues.append("state_ids and state_steps must have matching lengths.")
        if not self.state_ids:
            issues.append("state_ids must contain at least one state.")
        if self.state_ids and self.state_ids[0] != self.start_state_id:
            issues.append("state_ids must begin with start_state_id.")
        if self.state_steps and self.state_steps[0] != self.start_step:
            issues.append("state_steps must begin with start_step.")
        if len(set(self.state_ids)) != len(self.state_ids):
            issues.append("state_ids must be unique within one episode.")
        if any(step < 0 for step in self.state_steps):
            issues.append("state_steps must be non-negative.")
        if any(current < previous for previous, current in zip(self.state_steps, self.state_steps[1:])):
            issues.append("state_steps must be non-decreasing.")

        if self.status == EpisodeStatus.OPEN:
            if self.end_state_id is not None or self.end_step is not None:
                issues.append("Open episodes must not define end_state_id or end_step.")
        else:
            if self.end_state_id is None or self.end_step is None:
                issues.append("Closed or aborted episodes must define end_state_id and end_step.")
            else:
                if self.end_state_id != self.state_ids[-1]:
                    issues.append("end_state_id must equal the final entry in state_ids.")
                if self.end_step != self.state_steps[-1]:
                    issues.append("end_step must equal the final entry in state_steps.")
                if self.end_step < self.start_step:
                    issues.append("end_step must be greater than or equal to start_step.")
        return tuple(issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "simulation_id": str(self.simulation_id),
            "kind": self.kind.value,
            "status": self.status.value,
            "start_state_id": str(self.start_state_id),
            "end_state_id": str(self.end_state_id) if self.end_state_id is not None else None,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "state_ids": [str(state_id) for state_id in self.state_ids],
            "state_steps": list(self.state_steps),
            "tags": list(self.tags),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EpisodeRecord":
        end_state_id = data.get("end_state_id")
        return cls(
            episode_id=str(data["episode_id"]),
            simulation_id=SimulationId(str(data["simulation_id"])),
            kind=EpisodeKind(str(data["kind"])),
            status=EpisodeStatus(str(data["status"])),
            start_state_id=StateId(str(data["start_state_id"])),
            end_state_id=StateId(str(end_state_id)) if end_state_id else None,
            start_step=int(data["start_step"]),
            end_step=int(data["end_step"]) if data.get("end_step") is not None else None,
            state_ids=tuple(StateId(str(state_id)) for state_id in data.get("state_ids", ())),
            state_steps=tuple(int(step) for step in data.get("state_steps", ())),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            metadata=FrozenMetadata(data.get("metadata", {})),
        )


@dataclass(slots=True)
class EpisodeRegistry(ArchitecturalComponent, DocumentedComponent, ValidatableComponent):
    """Registry of grouped state windows for replay and diagnostics."""

    name: str = "episode_registry"
    classification: str = "[hybrid]"
    simulation_id: SimulationId | None = None
    _episodes: dict[str, EpisodeRecord] = field(default_factory=dict, init=False, repr=False)
    _episode_order: list[str] = field(default_factory=list, init=False, repr=False)
    _state_to_episode_ids: dict[StateId, list[str]] = field(default_factory=dict, init=False, repr=False)
    _episode_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.simulation_id is not None:
            self.simulation_id = SimulationId(str(self.simulation_id))
        issues = self.validate()
        if issues:
            raise ContractValidationError("; ".join(issues))

    def __len__(self) -> int:
        return len(self._episode_order)

    def describe_role(self) -> str:
        return (
            "Tracks grouped historical intervals such as trajectories and "
            "instability windows while preserving state-registry ownership of lineage."
        )

    def declared_dependencies(self) -> tuple[str, ...]:
        return (
            "core/state.py",
            "core/state_registry.py",
            "memory/trace_store.py",
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
        for episode_id in self._episode_order:
            episode = self._episodes[episode_id]
            issues.extend(episode.validate())
            if self.simulation_id is not None and episode.simulation_id != self.simulation_id:
                issues.append(
                    f"EpisodeRecord {episode.episode_id} simulation_id does not match the registry simulation_id."
                )
        return tuple(issues)

    def _episode_id_exists(self, episode_id: str) -> bool:
        return episode_id in self._episodes

    def _next_episode_id(self) -> str:
        while True:
            self._episode_counter += 1
            candidate = f"episode-{self._episode_counter:06d}"
            if not self._episode_id_exists(candidate):
                return candidate

    def _assert_simulation_match(self, state: SimulationState) -> None:
        if self.simulation_id is None:
            self.simulation_id = state.provenance.simulation_id
        elif state.provenance.simulation_id != self.simulation_id:
            raise ContractValidationError("State simulation_id does not match the episode registry simulation_id.")

    def _register_membership(self, episode: EpisodeRecord) -> None:
        for state_id in episode.state_ids:
            episode_ids = self._state_to_episode_ids.setdefault(state_id, [])
            if episode.episode_id not in episode_ids:
                episode_ids.append(episode.episode_id)

    def _replace_episode(self, episode: EpisodeRecord) -> None:
        self._episodes[episode.episode_id] = episode
        self._state_to_episode_ids = {}
        for existing_episode_id in self._episode_order:
            self._register_membership(self._episodes[existing_episode_id])

    def open_episode(
        self,
        state: SimulationState,
        *,
        kind: EpisodeKind | str = EpisodeKind.TRAJECTORY,
        tags: Sequence[str] = (),
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> EpisodeRecord:
        self._assert_simulation_match(state)
        episode = EpisodeRecord(
            episode_id=self._next_episode_id(),
            simulation_id=state.provenance.simulation_id,
            kind=EpisodeKind(str(kind)),
            status=EpisodeStatus.OPEN,
            start_state_id=state.provenance.state_id,
            start_step=state.step,
            state_ids=(state.provenance.state_id,),
            state_steps=(state.step,),
            tags=tags,
            metadata=metadata if isinstance(metadata, FrozenMetadata) else FrozenMetadata(metadata),
        )
        self._episodes[episode.episode_id] = episode
        self._episode_order.append(episode.episode_id)
        self._register_membership(episode)
        return episode

    def open_from_registry(
        self,
        registry: SimulationStateRegistry,
        state_id: StateId,
        *,
        kind: EpisodeKind | str = EpisodeKind.TRAJECTORY,
        tags: Sequence[str] = (),
        metadata: Mapping[str, object] | FrozenMetadata | None = None,
    ) -> EpisodeRecord:
        state = registry.get_state(state_id)
        if state.provenance.simulation_id != registry.require_simulation_id():
            raise ContractValidationError("Requested state does not match the provided registry simulation.")
        return self.open_episode(state, kind=kind, tags=tags, metadata=metadata)

    def get_episode(self, episode_id: str) -> EpisodeRecord:
        return self._episodes[episode_id]

    def append_state(self, episode_id: str, state: SimulationState) -> EpisodeRecord:
        episode = self.get_episode(episode_id)
        if episode.status != EpisodeStatus.OPEN:
            raise ContractValidationError("Cannot append state to a non-open episode.")
        self._assert_simulation_match(state)
        state_id = state.provenance.state_id
        if state_id in episode.state_ids:
            raise ContractValidationError(f"State {state_id} is already present in episode {episode_id}.")
        if state.step < episode.state_steps[-1]:
            raise ContractValidationError("Episode state steps must be non-decreasing.")

        updated = EpisodeRecord(
            episode_id=episode.episode_id,
            simulation_id=episode.simulation_id,
            kind=episode.kind,
            status=episode.status,
            start_state_id=episode.start_state_id,
            start_step=episode.start_step,
            state_ids=episode.state_ids + (state_id,),
            state_steps=episode.state_steps + (state.step,),
            tags=episode.tags,
            metadata=episode.metadata,
        )
        self._replace_episode(updated)
        return updated

    def close_episode(
        self,
        episode_id: str,
        *,
        final_state: SimulationState | None = None,
        status: EpisodeStatus | str = EpisodeStatus.CLOSED,
        metadata_updates: Mapping[str, object] | None = None,
    ) -> EpisodeRecord:
        episode = self.get_episode(episode_id)
        if episode.status != EpisodeStatus.OPEN:
            raise ContractValidationError("Episode is already closed or aborted.")

        updated = episode
        if final_state is not None and final_state.provenance.state_id != episode.state_ids[-1]:
            updated = self.append_state(episode_id, final_state)
        elif final_state is not None:
            self._assert_simulation_match(final_state)
            if final_state.step < episode.state_steps[-1]:
                raise ContractValidationError("final_state.step cannot precede the current episode tail.")

        closing_status = EpisodeStatus(str(status))
        closed = EpisodeRecord(
            episode_id=updated.episode_id,
            simulation_id=updated.simulation_id,
            kind=updated.kind,
            status=closing_status,
            start_state_id=updated.start_state_id,
            end_state_id=updated.state_ids[-1],
            start_step=updated.start_step,
            end_step=updated.state_steps[-1],
            state_ids=updated.state_ids,
            state_steps=updated.state_steps,
            tags=updated.tags,
            metadata=updated.metadata.with_updates(metadata_updates),
        )
        self._replace_episode(closed)
        return closed

    def all_episodes(self) -> tuple[EpisodeRecord, ...]:
        return tuple(self._episodes[episode_id] for episode_id in self._episode_order)

    def open_episodes(self) -> tuple[EpisodeRecord, ...]:
        return tuple(episode for episode in self.all_episodes() if episode.status == EpisodeStatus.OPEN)

    def closed_episodes(self) -> tuple[EpisodeRecord, ...]:
        return tuple(episode for episode in self.all_episodes() if episode.status != EpisodeStatus.OPEN)

    def episodes_for_state(self, state_id: StateId | str) -> tuple[EpisodeRecord, ...]:
        normalized_state_id = StateId(str(state_id))
        return tuple(
            self._episodes[episode_id]
            for episode_id in self._state_to_episode_ids.get(normalized_state_id, ())
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "simulation_id": str(self.simulation_id) if self.simulation_id is not None else None,
            "episodes": [episode.to_dict() for episode in self.all_episodes()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EpisodeRegistry":
        simulation_id = data.get("simulation_id")
        registry = cls(
            name=str(data.get("name", "episode_registry")),
            classification=str(data.get("classification", "[hybrid]")),
            simulation_id=SimulationId(str(simulation_id)) if simulation_id else None,
        )
        for episode_data in data.get("episodes", ()):
            episode = EpisodeRecord.from_dict(episode_data)
            if episode.episode_id in registry._episodes:
                raise ContractValidationError(f"Duplicate episode_id detected during load: {episode.episode_id}")
            registry._episodes[episode.episode_id] = episode
            registry._episode_order.append(episode.episode_id)
            registry._register_membership(episode)
        registry._episode_counter = len(registry._episode_order)
        return registry
